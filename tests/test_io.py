"""`Pipeline.run_to` writing chunks to disk, one file per table.

Parquet has `_reconcile` to guard the schema between chunks. These tests
cover the same hazard on the CSV path, and the schema a table with no rows
at all leaves in the file it still writes.
"""

from __future__ import annotations

import pandas as pd
import pytest

import synthweave as sw


class _ColumnReorderingNoiser:
    """A minimal third-party stage: hands back each chunk with its columns
    in a different order than it received them, exactly as the stage
    protocol permits."""

    def run(self, chunks, table, ctx):
        for i, chunk in enumerate(chunks):
            yield chunk[list(reversed(chunk.columns))] if i % 2 else chunk


class _ColumnDroppingNoiser:
    def run(self, chunks, table, ctx):
        for i, chunk in enumerate(chunks):
            yield chunk.drop(columns=chunk.columns[0]) if i == 1 else chunk


def _schema(*, chunk_size):
    person = sw.Entity(
        "person",
        count=6,
        attributes={"education": sw.Choice(["HS", "College"], [0.5, 0.5])},
        identifiers=[sw.Identifier("tax_id", prefix="TIN", digits=9)],
    )
    table = sw.Table(
        "t",
        grain=sw.PerEntity("person"),
        carry=["education"],
        identifiers=["tax_id"],
    )
    return sw.Schema(entities=[person], tables=[table], seed=1)


def test_csv_chunks_with_reordered_columns_still_land_under_the_right_header(tmp_path):
    schema = _schema(chunk_size=2)
    expected = sw.Pipeline(schema, chunk_size=2).run()["t"]

    sw.Pipeline(schema, chunk_size=2, noiser=_ColumnReorderingNoiser()).run_to(
        tmp_path, format="csv"
    )
    written = pd.read_csv(tmp_path / "t.csv", dtype=str)

    assert set(written.columns) == set(expected.columns)
    assert written["tax_id"].str.startswith("TIN").all()
    assert set(written["education"]) <= {"HS", "College"}


def test_a_csv_chunk_missing_a_column_is_rejected(tmp_path):
    schema = _schema(chunk_size=2)
    with pytest.raises(ValueError, match="tax_id"):
        sw.Pipeline(schema, chunk_size=2, noiser=_ColumnDroppingNoiser()).run_to(
            tmp_path, format="csv"
        )


# --- an empty table's file schema -------------------------------------------


def _one_table(coverage):
    """The same table twice, once with coverage that emits rows and once with
    coverage low enough to emit none. Every other input is identical, so any
    difference between the two files is caused by the row count alone."""
    person = sw.Entity(
        "person",
        count=6,
        attributes={"education": sw.Choice(["HS", "College"], [0.5, 0.5])},
        identifiers=[sw.Identifier("tax_id", prefix="TIN", digits=9)],
    )
    table = sw.Table(
        "t",
        grain=sw.PerEntity("person"),
        carry=["education"],
        identifiers=["tax_id"],
        columns={"score": sw.Integer(0, 100), "weight": sw.Uniform(0.0, 1.0)},
        coverage=coverage,
    )
    return sw.Schema(entities=[person], tables=[table], seed=4)


def _written_schema(tmp_path, coverage, name):
    pq = pytest.importorskip("pyarrow.parquet")
    out = tmp_path / name
    sw.Pipeline(_one_table(coverage)).run_to(out)
    return pq.read_schema(out / "t.parquet")


def test_a_parquet_file_for_a_table_with_no_rows_keeps_its_declared_schema(tmp_path):
    """A zero-row run must not change the file's schema, only its row count.

    This is the persisted half of the same guarantee `test_a_table_that_emits_
    no_rows_keeps_its_non_empty_dtypes` makes in memory, and it is the half
    that lasts: a file outlives the run, so a consumer reading a directory of
    parts gets a schema that depends on whether that partition happened to
    cover any entity.

    The assertion is on the file, read back, rather than on the frame. An
    in-memory check cannot see this: `write_empty` used to build its own
    stand-in frame with every column `object`, which pyarrow writes as the
    `null` type no matter what the in-memory result of the same run said.

    `score` and `weight` are the load-bearing columns. They are the only two
    whose rule declares a type, so they are the only two where `null` and the
    real type can be told apart.
    """
    pa = pytest.importorskip("pyarrow")
    empty = _written_schema(tmp_path, 0.0001, "empty")
    populated = _written_schema(tmp_path, 1.0, "populated")

    declared = ["score", "weight"]
    assert [empty.field(c).type for c in declared] == [
        populated.field(c).type for c in declared
    ], f"empty {empty} vs populated {populated}"
    assert empty.field("score").type == pa.int64()
    assert empty.names == populated.names


def test_an_untyped_column_in_an_empty_parquet_file_is_null_typed(tmp_path):
    """The limit, asserted so it cannot widen unnoticed.

    A column no rule types -- an identifier, a `Choice` over strings -- gets
    its populated type from pandas' inference over the values, and zero rows
    offer nothing to infer from. pyarrow writes such a column as `null`, where
    a populated run gives a string type. That is not a guess this fix can make
    better, and `null` is at least the honest answer: arrow promotes it to any
    other type when parts are concatenated, so a reader is not blocked.

    The populated type is deliberately not named: it is `string` on pandas 2
    and `large_string` on pandas 3, and pinning either would fail the other.
    """
    pa = pytest.importorskip("pyarrow")
    empty = _written_schema(tmp_path, 0.0001, "empty")
    populated = _written_schema(tmp_path, 1.0, "populated")

    for untyped in ("tax_id", "education"):
        assert empty.field(untyped).type == pa.null()
        assert pa.types.is_string(populated.field(untyped).type) or pa.types.is_large_string(
            populated.field(untyped).type
        )
