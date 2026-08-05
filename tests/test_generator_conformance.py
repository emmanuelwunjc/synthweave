"""`sw.check_generator`: the conformance harness for a custom `Generator`.

`Generator` is a `runtime_checkable` Protocol matched on `emit` alone, so
nothing stops a plugin author from writing one that reaches for random state,
drops a bookkeeping column, invents a column, omits part of the population, or
cuts the stream by rows instead of by entities. These tests are the harness's
own proof that it catches each of those, one deliberately non-conforming
generator per clause.

Every non-conforming class below delegates to the built-in generator and then
breaks exactly one clause, so a failure it triggers cannot be blamed on
anything else about it. Each test asserts on the clause name the failure
carries, not merely that something went red: a conformance check that goes red
incidentally looks exactly like one that works.

Entity non-straddling is the clause #53 names as unasserted. It is also the
only one a generator can break with all four others intact, which is why
`RowChunkedGenerator` below is the most important class in this file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import synthweave as sw
from synthweave.validation import ENTITY_KEY, ROW_KEY

# Small enough to split every fixture into many chunks. Matches `SPLIT` in
# tests/invariants.py, which is the size the rest of the suite chunks at.
SPLIT = 13


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


# --- deliberately non-conforming generators ---------------------------------


def _reference(table, ctx) -> list[pd.DataFrame]:
    """What the built-in generator emits, as a starting point to corrupt."""
    return list(sw.RuleGenerator().emit(table, ctx))


def _one_frame(chunks: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


class NonDeterministicGenerator:
    """Reaches for RNG state instead of deriving from the key."""

    def emit(self, table, ctx):
        for chunk in _reference(table, ctx):
            chunk = chunk.copy()
            chunk[list(table.columns)[0]] = np.random.default_rng().random(len(chunk))
            yield chunk


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


class RowChunkedGenerator:
    """Cuts the stream by rows instead of by entities.

    The realistic version of this bug: a generator that materializes rows and
    then slices them `chunk_size` at a time, so a boundary lands wherever it
    falls. Every other clause still holds -- the same rows in the same order,
    the same values, the same columns, the same count -- which makes this
    precisely the generator a suite without the non-straddling clause passes.
    """

    def emit(self, table, ctx):
        rows = _one_frame(_reference(table, ctx))
        keys = rows[ENTITY_KEY].to_numpy()
        # The first row repeating the previous row's entity: a cut point that
        # provably lands inside one entity rather than between two.
        cut = next((i for i in range(1, len(keys)) if keys[i] == keys[i - 1]), None)
        if cut is None:
            # PerEntity grain: one row per entity, so no cut can straddle. Cut
            # in the middle anyway rather than yielding one chunk, so the
            # harness still has a real boundary to inspect. Emitting a single
            # chunk here would trip the refusal added for #150 and the test
            # below would stop proving what it is for.
            cut = max(1, len(rows) // 2)
        yield rows.iloc[:cut].reset_index(drop=True)
        yield rows.iloc[cut:].reset_index(drop=True)


class ChunkSizeDependentGenerator:
    """A value derived from how many rows arrived together.

    Deterministic at any fixed chunk size, so the determinism clause cannot
    see it. Only comparing across chunk sizes can.
    """

    def emit(self, table, ctx):
        for chunk in _reference(table, ctx):
            chunk = chunk.copy()
            chunk[list(table.columns)[0]] = float(len(chunk))
            yield chunk


class OmittingGenerator:
    """Silently leaves the last ten entities out of the population.

    Omits the same entities at every chunk size, so this is invisible to the
    other four clauses: the rows it does emit are deterministic, correctly
    columned, chunk invariant and never straddling.
    """

    OMITTED = 10

    def emit(self, table, ctx):
        total = ctx.schema.entity(table.entity).count.value
        gone = {f"{table.entity}:{i}" for i in range(total - self.OMITTED, total)}
        for chunk in _reference(table, ctx):
            kept = chunk[~chunk[ENTITY_KEY].isin(gone)]
            if len(kept):
                yield kept


class DuplicatingGenerator:
    """Emits the declared number of rows over half the declared entities."""

    def emit(self, table, ctx):
        rows = _one_frame(_reference(table, ctx))
        half = len(rows) // 2
        yield pd.concat([rows.iloc[:half], rows.iloc[:half]], ignore_index=True)


# --- one non-conforming generator per clause --------------------------------


def test_a_conforming_generator_passes(grains):
    for table_name in ALL_TABLES:
        sw.check_generator(sw.RuleGenerator(), grains, table=table_name)


def test_a_non_deterministic_generator_is_rejected(grains):
    with pytest.raises(sw.GeneratorConformanceError, match="^determinism:"):
        sw.check_generator(NonDeterministicGenerator(), grains, table="visits")


def test_a_generator_dropping_a_bookkeeping_key_is_rejected(grains):
    with pytest.raises(sw.GeneratorConformanceError, match="^emitted columns:"):
        sw.check_generator(NoRowKeyGenerator(), grains, table="wages")


def test_a_generator_emitting_an_undeclared_column_is_rejected(grains):
    with pytest.raises(sw.GeneratorConformanceError, match="^emitted columns:"):
        sw.check_generator(ExtraColumnGenerator(), grains, table="wages")


@pytest.mark.parametrize("table_name", ("wages", "visits"))
def test_a_row_chunked_generator_straddles_an_entity_and_is_rejected(table_name, grains):
    """The clause #53 names. Both multi-row grains, since the bug cannot occur
    on PerEntity and a check that only ever saw PerEntity would prove nothing."""
    with pytest.raises(sw.GeneratorConformanceError, match="^entity non-straddling:"):
        sw.check_generator(RowChunkedGenerator(), grains, table=table_name)


def test_a_chunk_size_dependent_generator_is_rejected(grains):
    with pytest.raises(sw.GeneratorConformanceError, match="^chunk invariance:"):
        sw.check_generator(ChunkSizeDependentGenerator(), grains, table="visits")


def test_a_generator_omitting_entities_is_rejected(grains):
    with pytest.raises(sw.GeneratorConformanceError, match="^row count:"):
        sw.check_generator(OmittingGenerator(), grains, table="roster")


def test_a_generator_covering_half_the_population_twice_is_rejected(grains):
    """A row total alone is not enough: half the population twice over is the
    declared count of rows and half the declared count of entities."""
    with pytest.raises(sw.GeneratorConformanceError, match="^row count:"):
        sw.check_generator(DuplicatingGenerator(), grains, table="roster")


# --- guards on the guards ---------------------------------------------------


def test_the_row_chunked_generator_passes_on_per_entity_grain(grains):
    """`RowChunkedGenerator` is not simply always red.

    On PerEntity grain no cut can split an entity, so the same deliberately
    broken generator conforms. Without this, the two failures above could be
    coming from something else about it entirely.
    """
    sw.check_generator(RowChunkedGenerator(), grains, table="roster")


def test_the_omitting_generator_passes_where_config_fixes_no_row_count(grains):
    """The row-count clause is documented as PerEntity-at-full-coverage only,
    and this is what that costs: the identical omission is invisible on
    PerEvent grain, because nothing in the config says how many rows a table
    whose grain draws its own counts should have. Asserting the gap keeps it a
    known limit rather than a check quietly doing nothing.
    """
    sw.check_generator(OmittingGenerator(), grains, table="visits")


def test_the_table_argument_may_be_omitted_for_a_single_table_schema(careers):
    sw.check_generator(sw.RuleGenerator(), careers)


def test_a_multi_table_schema_must_name_its_table(grains):
    with pytest.raises(ValueError, match="pass table="):
        sw.check_generator(sw.RuleGenerator(), grains)


@pytest.fixture
def tiny() -> sw.Schema:
    """Four entities: too few for the default `split_chunk_size` to cut.

    Chunking is over entities, and `RuleGenerator` sizes its stride as
    `chunk_size // rows_per_entity`, so at 13 this whole table lands in one
    chunk. A small schema is the natural thing to iterate against, which is
    what makes the gap it exposes expensive.
    """
    person = sw.Entity(
        "person",
        count=4,
        attributes={"e": sw.Choice(["a", "b"], [0.5, 0.5])},
        identifiers=[sw.Identifier("tax_id")],
    )
    table = sw.Table(
        "small",
        grain=sw.PerEvent("person", low=1, high=3),
        carry=["e"],
        columns={"c": sw.Uniform(0, 1)},
    )
    return sw.Schema(entities=[person], tables=[table], seed=1)


def test_a_split_size_that_does_not_split_is_refused(tiny):
    """The bug this closes: a conforming-looking pass over two clauses that
    examined nothing.

    Clauses 3 and 4 are both about what happens at a chunk boundary. Before
    this guard, a schema too small for `split_chunk_size` to cut produced one
    chunk, both clauses passed having seen no boundary, and `check_generator`
    returned `None` exactly as it does for a generator it genuinely verified.
    A caller error, so `ValueError` rather than `GeneratorConformanceError`:
    the generator did nothing wrong, the harness was configured so that it
    could not be tested.
    """
    with pytest.raises(ValueError, match="split_chunk_size"):
        sw.check_generator(sw.RuleGenerator(), tiny)


def test_a_split_size_small_enough_to_cut_the_tiny_schema_is_accepted(tiny):
    """The refusal above is fixable from the message, and this is the fix it
    names. Without this, the guard could be refusing every small schema
    outright rather than refusing only the sizes that cannot split one."""
    sw.check_generator(sw.RuleGenerator(), tiny, split_chunk_size=1)


# --- over the registry ------------------------------------------------------


@pytest.mark.parametrize("table_name", ALL_TABLES)
def test_every_registered_generator_conforms(table_name, grains):
    """The suite covers whatever is registered, including a plugin's own.

    Reading the registry rather than listing built-ins is the point: a third
    party's generator is held to the same contract without editing library
    code, and a built-in added later cannot drift out of coverage.
    """
    sw.register("generator", "conformance-test-rules")(sw.RuleGenerator())
    names = sw.available("generator")
    assert "conformance-test-rules" in names
    assert len(names) > 1, f"only {names} registered; the built-in generator is missing"

    for name in names:
        sw.check_generator(sw.resolve("generator", name), grains, table=table_name)
