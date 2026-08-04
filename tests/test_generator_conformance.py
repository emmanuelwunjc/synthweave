"""Conformance suite for stage 1: every registered `Generator`, checked.

`Generator` is a `runtime_checkable` Protocol matched on `emit` alone, so
anything with the right method name is accepted as a generator. Everything the
rest of the pipeline relies on is stated in prose and checked nowhere. This
file turns each of those statements into an assertion and runs it against
every implementation in `sw.available("generator")`.

The five clauses, each with the sentence it comes from:

1. determinism      CONTRIBUTING.md: "Determinism. Same schema and seed
                    produce identical output, always."
2. chunk invariance docs/GUIDE.md on `chunk_size`: "A memory setting only --
                    it never changes the output."
3. entity non-straddling
                    `RuleGenerator._entities_per_chunk`: "Chunking happens
                    over entities rather than rows so that an entity's rows
                    never straddle a boundary."
4. emitted columns  pipeline.py `_strip_reserved`: the bookkeeping columns
                    "never reach the user but every stage can rely on them
                    being present", and `Table.output_columns` is "the columns
                    this table produces, in the order a run emits them".
5. row count        `Entity.count` is the population, so a table at
                    `coverage=1.0` on `PerEntity` grain has one row per
                    entity and no more.

Clause 3 is the one nothing asserted before. It is also the only clause a
generator can break while every other clause still holds, which is why it
needs its own check rather than being assumed to fall out of the others.

Internal, not public, on purpose. `sw.check_rule` is public because a plugin
author writes `Rule` implementations constantly and needs a one-liner they can
call on their own object. A generator is rarer, and checking one needs a whole
`Schema` and a table name rather than a bare object, so the public one-liner
does not exist to be written. The frozen surface in `tests/test_public_api.py`
grows only when a user would reach for the name; nobody has asked for this
one. If a generator plugin ecosystem appears, promoting the `check_*`
functions below into `synthweave` is a small, deliberate follow-up.

Every check below is proved to have teeth by a deliberately non-conforming
generator further down the file. A conformance suite that nothing can fail is
the exact failure #53 exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import synthweave as sw
from synthweave.validation import ENTITY_KEY, RESERVED_PREFIX, ROW_KEY

# Small enough that the whole suite is fast, large enough that 13 splits every
# fixture into many chunks and 100_000 leaves each one whole.
SPLIT = 13
WHOLE = 100_000
SIZES = (SPLIT, 97, WHOLE)


class GeneratorConformanceError(AssertionError):
    """A generator broke one of the five clauses this file checks."""


# --- the harness ---------------------------------------------------------


def _chunks(generator, schema: sw.Schema, table_name: str, chunk_size: int) -> list[pd.DataFrame]:
    """One generator run's chunks, kept separate rather than concatenated.

    Chunk boundaries are the subject of two of the five clauses, so they must
    survive collection. A helper that concatenated would erase the evidence.
    """
    ctx = sw.RunContext(schema=schema, chunk_size=chunk_size)
    return list(generator.emit(schema.table(table_name), ctx))


def _rows(chunks: list[pd.DataFrame]) -> pd.DataFrame:
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _fail(clause: str, detail: str) -> None:
    raise GeneratorConformanceError(f"{clause}: {detail}")


def _same_rows(clause: str, expected: pd.DataFrame, actual: pd.DataFrame, what: str) -> None:
    if len(expected) != len(actual):
        _fail(clause, f"{what} produced {len(actual)} rows, expected {len(expected)}")
    if list(expected.columns) != list(actual.columns):
        _fail(clause, f"{what} produced columns {list(actual.columns)}, expected {list(expected.columns)}")
    try:
        pd.testing.assert_frame_equal(actual, expected)
    except AssertionError as exc:
        _fail(clause, f"{what} changed a value: {exc}")


def check_determinism(generator, schema: sw.Schema, table_name: str) -> None:
    """Clause 1. Two runs of the same table at the same chunk size agree."""
    first = _rows(_chunks(generator, schema, table_name, WHOLE))
    second = _rows(_chunks(generator, schema, table_name, WHOLE))
    _same_rows("determinism", first, second, "a second emit() of the same table")


def check_chunk_invariance(generator, schema: sw.Schema, table_name: str) -> None:
    """Clause 2. The rows are the same however the stream was cut up.

    Compares the concatenation, not the chunks: how many chunks a size
    produces is exactly what `chunk_size` is allowed to change.
    """
    reference = _rows(_chunks(generator, schema, table_name, SIZES[-1]))
    for size in SIZES[:-1]:
        other = _rows(_chunks(generator, schema, table_name, size))
        _same_rows("chunk invariance", reference, other, f"chunk_size={size}")


def check_entity_non_straddling(generator, schema: sw.Schema, table_name: str) -> None:
    """Clause 3. Every row of an entity arrives in the same chunk.

    The guarantee nothing asserted before #53. A generator that cut the stream
    by rows instead of by entities would satisfy every other clause here and
    still hand a consumer half an entity, which silently breaks anything that
    derives a per-entity fact from a chunk (a running total, a first-visit
    flag, a within-entity sort) without ever raising.

    Checked at the smallest chunk size, since a size that leaves the table in
    one chunk cannot show a boundary problem at all.
    """
    chunks = _chunks(generator, schema, table_name, SPLIT)
    seen: dict[object, int] = {}
    for index, chunk in enumerate(chunks):
        if ENTITY_KEY not in chunk.columns:
            _fail(
                "entity non-straddling",
                f"chunk {index} has no {ENTITY_KEY!r} column, so an entity's rows "
                f"cannot be located; it has {list(chunk.columns)}",
            )
        for key in pd.unique(chunk[ENTITY_KEY]):
            if key in seen and seen[key] != index:
                _fail(
                    "entity non-straddling",
                    f"entity {key!r} has rows in chunk {seen[key]} and again in chunk "
                    f"{index} at chunk_size={SPLIT}; an entity's rows must not be split "
                    f"across a chunk boundary",
                )
            seen[key] = index


def check_emitted_columns(generator, schema: sw.Schema, table_name: str) -> None:
    """Clause 4. Each chunk carries the bookkeeping keys and the declared columns.

    Both halves matter and neither is checked anywhere else. The reserved keys
    are what every later stage derives from and the pipeline strips them last,
    so a generator that omits one produces a table the linker cannot key. The
    declared columns are what the user asked for, in the order `output_columns`
    promises.
    """
    table = schema.table(table_name)
    expected = table.output_columns(with_identifiers=False)
    for index, chunk in enumerate(_chunks(generator, schema, table_name, SPLIT)):
        for key in (ENTITY_KEY, ROW_KEY):
            if key not in chunk.columns:
                _fail(
                    "emitted columns",
                    f"chunk {index} is missing the bookkeeping column {key!r} that every "
                    f"later stage derives from; it has {list(chunk.columns)}",
                )
        produced = [c for c in chunk.columns if not str(c).startswith(RESERVED_PREFIX)]
        if produced != expected:
            _fail(
                "emitted columns",
                f"chunk {index} produced {produced}, but table {table_name!r} declares "
                f"{expected}",
            )


def check_row_count(generator, schema: sw.Schema, table_name: str) -> None:
    """Clause 5. A fully covered PerEntity table has one row per entity.

    Only stated for that case, because it is the only one where config alone
    fixes the answer. `PerEvent` draws its own row count and `coverage` below
    1.0 keeps a hash-selected subset, so neither has a number to compare to.
    """
    table = schema.table(table_name)
    if not isinstance(table.grain, sw.PerEntity) or table.coverage.value != 1.0:
        raise ValueError(
            f"check_row_count only applies to a PerEntity table at coverage=1.0; "
            f"{table_name!r} is {type(table.grain).__name__} at coverage={table.coverage.value}"
        )
    expected = schema.entity(table.entity).count.value
    rows = _rows(_chunks(generator, schema, table_name, SPLIT))
    if len(rows) != expected:
        _fail(
            "row count",
            f"table {table_name!r} emitted {len(rows)} rows for an entity declaring "
            f"count={expected} at coverage=1.0",
        )
    distinct = rows[ENTITY_KEY].nunique()
    if distinct != expected:
        _fail(
            "row count",
            f"table {table_name!r} emitted {len(rows)} rows covering only {distinct} "
            f"distinct entities, but count={expected} at coverage=1.0 means one row each",
        )


# --- fixtures ------------------------------------------------------------


@pytest.fixture
def visits() -> sw.Table:
    """PerEvent grain: a variable number of rows per entity.

    The grain that makes straddling possible at all. PerEntity is one row per
    entity, so a chunk boundary between rows is also a boundary between
    entities no matter how the generator slices.
    """
    return sw.Table(
        "visits",
        grain=sw.PerEvent("person", low=1, high=6),
        carry=["education"],
        columns={"cost": sw.Uniform(0, 500)},
    )


@pytest.fixture
def grains(people, roster, wages, visits) -> sw.Schema:
    """One schema carrying all three grains, so every check sees each."""
    return sw.Schema(entities=[people], tables=[roster, wages, visits], seed=7)


ALL_TABLES = ("roster", "wages", "visits")


# --- deliberately non-conforming generators ------------------------------


def _reference(table: sw.Table, ctx: sw.RunContext) -> list[pd.DataFrame]:
    """What the built-in generator emits, as a starting point to corrupt.

    Each broken generator below delegates here and then breaks exactly one
    clause, so a failure cannot be blamed on anything else about it.
    """
    return list(sw.RuleGenerator().emit(table, ctx))


class NonDeterministicGenerator:
    """Reaches for RNG state instead of deriving from the key."""

    def emit(self, table, ctx):
        for chunk in _reference(table, ctx):
            chunk = chunk.copy()
            chunk[list(table.columns)[0]] = np.random.default_rng().random(len(chunk))
            yield chunk


class ChunkSizeDependentGenerator:
    """A value derived from how many rows arrived together.

    Deterministic at any fixed chunk size, so clause 1 cannot see it. Only
    comparing across chunk sizes can.
    """

    def emit(self, table, ctx):
        for chunk in _reference(table, ctx):
            chunk = chunk.copy()
            chunk[list(table.columns)[0]] = float(len(chunk))
            yield chunk


class RowChunkedGenerator:
    """Cuts the stream by rows instead of by entities.

    The realistic version of this bug: a generator that materializes rows and
    then slices them `chunk_size` at a time, which lands a boundary wherever
    it falls. Every other clause still holds -- the same rows, the same
    values, the same columns, the same count -- so this is precisely the
    generator that a suite without clause 3 would pass.
    """

    def emit(self, table, ctx):
        rows = _rows(_reference(table, ctx))
        keys = rows[ENTITY_KEY].to_numpy()
        # The first row that repeats the previous row's entity, i.e. a cut
        # point that provably lands inside one entity rather than between two.
        cut = next((i for i in range(1, len(keys)) if keys[i] == keys[i - 1]), None)
        if cut is None:
            # PerEntity grain: one row per entity, so no cut can straddle.
            yield rows
            return
        yield rows.iloc[:cut].reset_index(drop=True)
        yield rows.iloc[cut:].reset_index(drop=True)


class NoRowKeyGenerator:
    """Drops the row key every later stage derives row-level values from."""

    def emit(self, table, ctx):
        for chunk in _reference(table, ctx):
            yield chunk.drop(columns=[ROW_KEY])


class ExtraColumnGenerator:
    """Emits a column the table never declared."""

    def emit(self, table, ctx):
        for chunk in _reference(table, ctx):
            chunk = chunk.copy()
            chunk["undeclared"] = 1
            yield chunk


class TruncatingGenerator:
    """Stops one chunk early, so part of the population never appears."""

    def emit(self, table, ctx):
        chunks = _reference(table, ctx)
        yield from chunks[:-1]


class DuplicatingGenerator:
    """Emits the right number of rows for the wrong number of entities."""

    def emit(self, table, ctx):
        rows = _rows(_reference(table, ctx))
        half = len(rows) // 2
        yield pd.concat([rows.iloc[:half], rows.iloc[:half]], ignore_index=True)


# --- the checks, proved to have teeth ------------------------------------


def test_determinism_catches_a_generator_that_uses_rng_state(grains):
    with pytest.raises(GeneratorConformanceError, match="^determinism:"):
        check_determinism(NonDeterministicGenerator(), grains, "visits")


def test_chunk_invariance_catches_a_chunk_size_dependent_value(grains):
    with pytest.raises(GeneratorConformanceError, match="^chunk invariance:"):
        check_chunk_invariance(ChunkSizeDependentGenerator(), grains, "visits")


@pytest.mark.parametrize("table_name", ("wages", "visits"))
def test_non_straddling_catches_a_row_chunked_generator(table_name, grains):
    """The clause #53 names. Both multi-row grains, since the bug is invisible
    on PerEntity and a check that only ever saw PerEntity would prove nothing."""
    with pytest.raises(GeneratorConformanceError, match="^entity non-straddling:"):
        check_entity_non_straddling(RowChunkedGenerator(), grains, table_name)


def test_non_straddling_passes_the_same_generator_on_per_entity_grain(grains):
    """Guards the guard: `RowChunkedGenerator` is not simply always red.

    On PerEntity grain there is no cut that can split an entity, so the same
    deliberately broken generator conforms. Without this, the test above could
    be passing because the generator is broken in some other way.
    """
    check_entity_non_straddling(RowChunkedGenerator(), grains, "roster")


def test_emitted_columns_catches_a_missing_bookkeeping_key(grains):
    with pytest.raises(GeneratorConformanceError, match="^emitted columns:"):
        check_emitted_columns(NoRowKeyGenerator(), grains, "wages")


def test_emitted_columns_catches_an_undeclared_column(grains):
    with pytest.raises(GeneratorConformanceError, match="^emitted columns:"):
        check_emitted_columns(ExtraColumnGenerator(), grains, "wages")


def test_row_count_catches_a_truncated_population(grains):
    with pytest.raises(GeneratorConformanceError, match="^row count:"):
        check_row_count(TruncatingGenerator(), grains, "roster")


def test_row_count_catches_the_right_total_over_the_wrong_entities(grains):
    """A row total alone is not enough: half the population twice over is the
    declared count of rows and half the declared count of entities."""
    with pytest.raises(GeneratorConformanceError, match="^row count:"):
        check_row_count(DuplicatingGenerator(), grains, "roster")


def test_row_count_refuses_a_table_it_cannot_judge(grains):
    """PerEvent draws its own row count, so there is no declared number to
    compare against. Silently passing there would be a check that cannot fail."""
    with pytest.raises(ValueError, match="only applies to a PerEntity table"):
        check_row_count(sw.RuleGenerator(), grains, "visits")


# --- every registered generator ------------------------------------------

GENERATORS = sorted(sw.available("generator"))


@pytest.mark.parametrize("name", GENERATORS)
@pytest.mark.parametrize("table_name", ALL_TABLES)
def test_registered_generators_are_deterministic(name, table_name, grains):
    check_determinism(sw.resolve("generator", name), grains, table_name)


@pytest.mark.parametrize("name", GENERATORS)
@pytest.mark.parametrize("table_name", ALL_TABLES)
def test_registered_generators_are_chunk_invariant(name, table_name, grains):
    check_chunk_invariance(sw.resolve("generator", name), grains, table_name)


@pytest.mark.parametrize("name", GENERATORS)
@pytest.mark.parametrize("table_name", ALL_TABLES)
def test_registered_generators_never_straddle_an_entity(name, table_name, grains):
    check_entity_non_straddling(sw.resolve("generator", name), grains, table_name)


@pytest.mark.parametrize("name", GENERATORS)
@pytest.mark.parametrize("table_name", ALL_TABLES)
def test_registered_generators_emit_the_declared_columns(name, table_name, grains):
    check_emitted_columns(sw.resolve("generator", name), grains, table_name)


@pytest.mark.parametrize("name", GENERATORS)
def test_registered_generators_emit_one_row_per_entity(name, grains):
    check_row_count(sw.resolve("generator", name), grains, "roster")


def test_the_registry_is_not_empty():
    """Guards the guard: if `available("generator")` ever came back empty, every
    parametrized test above would collect zero cases and the file would report
    a clean pass while checking nothing."""
    assert GENERATORS, "no generators registered; the suite above checked nothing"
