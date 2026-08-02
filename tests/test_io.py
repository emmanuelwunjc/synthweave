"""`Pipeline.run_to` writing chunks to disk, one file per table.

Parquet has `_reconcile` to guard the schema between chunks. These tests
cover the same hazard on the CSV path.
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
