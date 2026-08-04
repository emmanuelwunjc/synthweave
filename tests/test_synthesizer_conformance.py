"""`sw.check_synthesizer`: the conformance harness for a custom `Synthesizer`.

`Synthesizer` is a `runtime_checkable` Protocol matched on `run` alone, so
nothing stops a plugin author from writing one that drops rows, invents
columns, overwrites a column it was never asked for, reaches for random
state, or reads chunk-level state. These tests are the harness's own proof
that it catches each of those, one deliberately non-conforming synthesizer
per contract clause.

Every non-conforming class below breaks exactly one clause and is otherwise
identical to `KeyedSynth`, so a failure it triggers cannot be blamed on
anything else.
"""

from __future__ import annotations

import random

import pytest

import synthweave as sw


class KeyedSynth:
    """A conforming synthesizer.

    Writes each named column from the row key alone, which is what makes it
    deterministic and chunk invariant, and touches nothing else.
    """

    def __init__(self, columns=("wage",)):
        self.columns = list(columns)

    def _values(self, chunk, ctx, column):
        return sw.Integer(0, 100).draw(
            chunk["_sw_row"].to_numpy(), seed=ctx.seed, salt=f"keyed\x00{column}"
        )

    def run(self, chunks, table, ctx):
        for chunk in chunks:
            out = chunk.copy()
            for column in self.columns:
                out[column] = self._values(chunk, ctx, column)
            yield out


class RowDropper(KeyedSynth):
    """Drops every other row. A synthesizer changes values, never the rows."""

    def run(self, chunks, table, ctx):
        for chunk in super().run(chunks, table, ctx):
            yield chunk.iloc[::2]


class RowKeyDropper(KeyedSynth):
    """Strips the reserved row key. It reads as internal bookkeeping, so this
    is an easy mistake, and the harness has to name it rather than let a bare
    `KeyError` out of its own internals."""

    def run(self, chunks, table, ctx):
        for chunk in super().run(chunks, table, ctx):
            yield chunk.drop(columns=["_sw_row"])


class RowShuffler(KeyedSynth):
    """Keeps every row but hands them back in a different order. The row
    count alone cannot see this, which is why the check reads the row key."""

    def run(self, chunks, table, ctx):
        for chunk in super().run(chunks, table, ctx):
            yield chunk.iloc[::-1]


class ColumnInventor(KeyedSynth):
    """Invents a column nobody asked for. Stage 2 fills columns the schema
    declared; a new one would reach the user as a column the schema never
    promised."""

    def run(self, chunks, table, ctx):
        for chunk in super().run(chunks, table, ctx):
            chunk["hunch"] = 1
            yield chunk


def test_a_conforming_synthesizer_passes(careers):
    sw.check_synthesizer(KeyedSynth(), careers)


def test_a_row_dropping_synthesizer_is_rejected(careers):
    with pytest.raises(sw.SynthesizerConformanceError, match="rows"):
        sw.check_synthesizer(RowDropper(), careers)


def test_a_synthesizer_dropping_the_row_key_is_rejected(careers):
    """The failure mode this replaces: `_check_rows_survive` indexed
    `got[ROW_KEY]` unguarded, so this candidate escaped as a bare
    `KeyError: '_sw_row'` raised from inside the harness. A plugin author got
    the column name and nothing about which guarantee they had broken.
    """
    with pytest.raises(sw.SynthesizerConformanceError, match="reserved row key"):
        sw.check_synthesizer(RowKeyDropper(), careers)


def test_a_row_reordering_synthesizer_is_rejected(careers):
    with pytest.raises(sw.SynthesizerConformanceError, match="reordered the rows"):
        sw.check_synthesizer(RowShuffler(), careers)


def test_a_column_inventing_synthesizer_is_rejected(careers):
    with pytest.raises(sw.SynthesizerConformanceError, match="column"):
        sw.check_synthesizer(ColumnInventor(), careers)


class Meddler(KeyedSynth):
    """Writes a column outside the ones it declared. `columns=` is the whole
    statement of what a synthesizer touches, and the pipeline runs one
    synthesizer over every table."""

    def run(self, chunks, table, ctx):
        for chunk in super().run(chunks, table, ctx):
            chunk["tenure"] = self._values(chunk, ctx, "tenure")
            yield chunk


def test_a_synthesizer_writing_an_undeclared_column_is_rejected(careers):
    with pytest.raises(sw.SynthesizerConformanceError, match="did not declare"):
        sw.check_synthesizer(Meddler(), careers)


class RandomSynth(KeyedSynth):
    """Reaches for hidden RNG state instead of deriving from the row key."""

    def run(self, chunks, table, ctx):
        for chunk in chunks:
            out = chunk.copy()
            for column in self.columns:
                out[column] = [random.random() for _ in range(len(out))]
            yield out


def test_a_non_deterministic_synthesizer_is_rejected(careers):
    with pytest.raises(sw.SynthesizerConformanceError, match="not deterministic"):
        sw.check_synthesizer(RandomSynth(), careers)


class ChunkCounter(KeyedSynth):
    """Chunk invariance broken: the value depends on how many rows arrived
    together, not on the row itself."""

    def run(self, chunks, table, ctx):
        for chunk in chunks:
            out = chunk.copy()
            for column in self.columns:
                out[column] = len(out)
            yield out


def test_a_chunk_size_dependent_synthesizer_is_rejected(careers):
    with pytest.raises(sw.SynthesizerConformanceError, match="chunk invariant"):
        sw.check_synthesizer(ChunkCounter(), careers)


class ShrinkingSynth(KeyedSynth):
    """Conforming on its first run and one row short on every one after, so
    the two runs being compared do not even have the same shape."""

    def __init__(self, columns=("wage",)):
        super().__init__(columns)
        self.runs = 0

    def run(self, chunks, table, ctx):
        self.runs += 1
        for chunk in super().run(chunks, table, ctx):
            yield chunk if self.runs == 1 else chunk.iloc[1:]


def test_a_synthesizer_whose_runs_disagree_in_shape_is_rejected(careers):
    """The shape mismatch is reported as the determinism failure it is,
    rather than surfacing as a length error from the value comparison."""
    with pytest.raises(sw.SynthesizerConformanceError, match="not deterministic"):
        sw.check_synthesizer(ShrinkingSynth(), careers)


# --- over the registry ------------------------------------------------------

# A registered synthesizer that cannot be built from its name alone needs a
# factory here. The list of what to check comes from the registry, never from
# this dict: a built-in added without a factory fails
# `test_a_registered_synthesizer_needing_config_is_not_silently_skipped`'s
# guard rather than quietly going unchecked.
#
# `fit_cap` for `cart` is deliberately far below the fixture's 400 rows. At
# the default
# cap the whole table fits under it however it was chunked, so the fit-buffer
# truncation that makes fitting chunk invariant (issue I3) is never exercised
# and the chunk-invariance check passes vacuously for `cart`. Under a cap of
# 50 it is the truncation, and nothing else, that keeps the two chunk sizes
# fitting on the same 50 rows.
FACTORIES = {
    "cart": lambda: sw.CARTSynthesizer(
        columns=["sector", "wage"],
        predictors=["education"],
        fit_cap=sw.user(50, "test cap"),
    ),
}


def _instance(name: str):
    """A ready-to-check instance of the synthesizer registered as `name`."""
    if name in FACTORIES:
        return FACTORIES[name]()
    try:
        return sw.resolve("synthesizer", name)
    except TypeError as exc:
        pytest.fail(
            f"synthesizer {name!r} needs configuration ({exc}) and this suite has no "
            f"factory for it, so it would go unchecked. Add one to FACTORIES."
        )


def test_every_registered_synthesizer_conforms(careers):
    """The suite covers whatever is registered, including a plugin's own.

    Registering here rather than listing built-ins is the point: a third
    party's synthesizer is held to the same contract without editing library
    code, and a built-in added later cannot drift out of coverage.
    """
    sw.register("synthesizer", "conformance-test-keyed")(KeyedSynth(columns=["wage"]))
    names = sw.available("synthesizer")
    assert "conformance-test-keyed" in names
    assert len(names) > 1, f"only {names} registered; the built-in synthesizer is missing"

    for name in names:
        try:
            sw.check_synthesizer(_instance(name), careers)
        except sw.SynthesizerConformanceError as exc:
            pytest.fail(f"registered synthesizer {name!r} does not conform: {exc}")


def test_a_registered_synthesizer_needing_config_is_not_silently_skipped():
    """The drift guard itself fails when it should.

    A synthesizer that a bare name cannot build must stop the suite, not be
    skipped past. Without this, `_instance` could swallow one and the loop
    above would report a pass it never earned.
    """

    @sw.register("synthesizer", "conformance-test-needs-config")
    class NeedsConfig(KeyedSynth):
        def __init__(self, columns):
            super().__init__(columns)

    with pytest.raises(pytest.fail.Exception, match="no factory"):
        _instance("conformance-test-needs-config")


# --- what the harness needs to be told --------------------------------------


def test_a_multi_table_schema_needs_the_table_named(schema):
    """Silently picking one of several tables would check the synthesizer
    against a table its author never meant."""
    with pytest.raises(ValueError, match="pass table="):
        sw.check_synthesizer(KeyedSynth(), schema)
    sw.check_synthesizer(KeyedSynth(), schema, table="wages")


def test_a_synthesizer_that_does_not_say_what_it_writes_is_refused(careers):
    """Without `columns`, "leaves undeclared columns alone" has no meaning,
    so the harness asks rather than skipping the check."""

    class Anonymous:
        def run(self, chunks, table, ctx):
            yield from chunks

    with pytest.raises(ValueError, match="columns="):
        sw.check_synthesizer(Anonymous(), careers)
    sw.check_synthesizer(Anonymous(), careers, columns=["wage"])
