"""`check_synthesizer`: the conformance harness for a custom `Synthesizer`.

The synthesizer-shaped analogue of `check_rule`. `Synthesizer` is a
`runtime_checkable` Protocol matched on `run` alone, so `isinstance` says
nothing about the guarantees the stage docstrings promise. This runs a
candidate over real generated chunks and checks each of those guarantees,
raising `SynthesizerConformanceError` naming which one broke.

    sw.check_synthesizer(MySynthesizer(columns=["wage"]), schema)

Nothing here is specific to a built-in: the harness takes any object with a
`run(chunks, table, ctx)`, which is what lets a plugin author check their own
implementation against the same contract the built-ins meet.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .context import RunContext
from .registry import resolve
from .schema import Schema, Table
from .validation import ROW_KEY


class SynthesizerConformanceError(AssertionError):
    """A custom Synthesizer broke one of the contracts `check_synthesizer` verifies."""


def check_synthesizer(
    synthesizer: Any,
    schema: Schema,
    *,
    table: str | None = None,
    columns: Sequence[str] | None = None,
    generator: Any = "rules",
    chunk_size: int = 100_000,
    split_chunk_size: int = 7,
) -> None:
    """Verify a synthesizer honours the guarantees stage 2 promises.

    Runs `synthesizer.run` over chunks a real generator emitted for one of
    `schema`'s tables, and checks, in order:

    1. Rows survive. A synthesizer changes values; it never adds, drops or
       reorders rows. Checked on the reserved row key, so it sees a reorder
       that a row count alone would miss.
    2. Columns survive. Stage 2 fills columns the schema declared; it neither
       invents one nor drops one.
    3. Undeclared columns are untouched. Every column outside `columns` comes
       back exactly as the generator emitted it.
    4. Determinism. Running it twice over identical input gives identical
       output, so no hidden RNG state is involved.
    5. Chunk invariance. Running it at `split_chunk_size` gives the same
       output as at `chunk_size`. This is the guarantee that chunk size is a
       memory knob and nothing else, and the one a fit that buffers whole
       chunks breaks.

    Checks 4 and 5 are only as strong as the configuration they run under: a
    synthesizer whose fit cap sits above the whole fixture never exercises
    its buffering, so check it configured the way it will really be used.

    Raises `SynthesizerConformanceError` naming the guarantee that broke and
    showing the first differences, or returns `None` if the synthesizer
    passes.

    `table` names which of `schema`'s tables to run against, and may be
    omitted only when the schema has exactly one. `columns` names the columns
    the synthesizer writes, and defaults to its `columns` attribute.
    """
    target = _target_table(schema, table)
    declared = _declared_columns(synthesizer, columns)

    source = resolve("generator", generator)
    baseline_input = _emit(source, target, schema, chunk_size)
    baseline = _apply(synthesizer, source, target, schema, chunk_size)

    _check_rows_survive(baseline_input, baseline)
    _check_columns_survive(baseline_input, baseline)
    _check_undeclared_columns_untouched(baseline_input, baseline, declared)

    repeat = _apply(synthesizer, source, target, schema, chunk_size)
    _check_same_frame(
        baseline,
        repeat,
        "running the synthesizer twice over identical input gave different values back "
        "(it is not deterministic, e.g. it reaches for random state instead of deriving "
        "from the row key)",
    )

    split = _apply(synthesizer, source, target, schema, split_chunk_size)
    _check_same_frame(
        baseline,
        split,
        f"running the synthesizer at chunk_size={split_chunk_size} instead of "
        f"{chunk_size} changed the output (it is not chunk invariant, e.g. it reads "
        "chunk-level state or fits on whatever rows one chunk happened to hold). "
        "chunk_size is a memory knob and nothing else",
    )


# --- individual checks ------------------------------------------------------


def _check_rows_survive(given: pd.DataFrame, got: pd.DataFrame) -> None:
    if len(given) != len(got):
        raise SynthesizerConformanceError(
            f"the synthesizer changed how many rows there are: it was given {len(given)} "
            f"row(s) and returned {len(got)}. A synthesizer changes values in the rows it "
            "is handed; adding or dropping rows is the generator's job, not stage 2's."
        )
    if ROW_KEY not in got.columns:
        # Guarded rather than left to `got[ROW_KEY]`, which raises a bare
        # KeyError naming the column and nothing else. Dropping this column is
        # an easy mistake (it reads as internal bookkeeping), and a function
        # whose whole job is to name the broken guarantee should not be the one
        # that fails to.
        raise SynthesizerConformanceError(
            f"the synthesizer dropped {ROW_KEY!r}, the reserved row key. It is not "
            "internal bookkeeping to strip: every downstream stage keys on it, and "
            "the pipeline cannot line the synthesized values back up without it. "
            "Pass it through untouched."
        )
    before = given[ROW_KEY].tolist()
    after = got[ROW_KEY].tolist()
    if before != after:
        differing = [i for i, (b, a) in enumerate(zip(before, after)) if b != a]
        first = differing[0]
        raise SynthesizerConformanceError(
            f"the synthesizer reordered the rows it was given: {len(differing)} row(s) came "
            f"back under a different {ROW_KEY!r}, first at position {first} "
            f"(was {before[first]!r}, got {after[first]!r}). Row order is the generator's, "
            "and downstream stages key on it."
        )


def _check_columns_survive(given: pd.DataFrame, got: pd.DataFrame) -> None:
    before, after = list(given.columns), list(got.columns)
    added = [c for c in after if c not in before]
    dropped = [c for c in before if c not in after]
    if added or dropped:
        detail = ", ".join(
            part
            for part in (
                f"invented column(s) {added}" if added else "",
                f"dropped column(s) {dropped}" if dropped else "",
            )
            if part
        )
        raise SynthesizerConformanceError(
            f"the synthesizer changed which columns the table has: {detail}. Stage 2 fills "
            "columns the schema declared, so a column it adds reaches the user as one the "
            "schema never promised, and one it drops is a column the schema did promise."
        )


def _check_undeclared_columns_untouched(
    given: pd.DataFrame, got: pd.DataFrame, declared: Sequence[str]
) -> None:
    """Every column the synthesizer did not name comes back byte for byte.

    `columns=` is the whole statement of what a synthesizer writes. One
    synthesizer runs over every table in the pipeline, so a column it
    silently rewrites is one the user believes came from their schema.
    """
    for column in given.columns:
        if column in set(declared) or column not in got.columns:
            continue
        rows = _differing_rows(given[column], got[column])
        if len(rows):
            raise SynthesizerConformanceError(
                f"the synthesizer wrote to column {column!r}, which it did not declare "
                f"(it declared {list(declared)}). {len(rows)} row(s) differ, "
                f"e.g. {_show(given[column], got[column], rows)}."
            )


def _check_same_frame(expected: pd.DataFrame, actual: pd.DataFrame, reason: str) -> None:
    """Two runs of the same synthesizer must agree column for column."""
    if list(expected.columns) != list(actual.columns) or len(expected) != len(actual):
        raise SynthesizerConformanceError(
            f"{reason}. The two runs do not even have the same shape: "
            f"{expected.shape} with columns {list(expected.columns)} versus "
            f"{actual.shape} with columns {list(actual.columns)}"
        )
    for column in expected.columns:
        rows = _differing_rows(expected[column], actual[column])
        if len(rows):
            raise SynthesizerConformanceError(
                f"{reason}. Column {column!r} differs in {len(rows)} of {len(expected)} "
                f"row(s), e.g. {_show(expected[column], actual[column], rows)}"
            )


# --- plumbing ---------------------------------------------------------------


def _target_table(schema: Schema, table: str | None) -> Table:
    if table is not None:
        return schema.table(table)
    if len(schema.tables) != 1:
        raise ValueError(
            f"schema has {len(schema.tables)} tables "
            f"({sorted(t.name for t in schema.tables)}); pass table= to name the one "
            "to check the synthesizer against"
        )
    return schema.tables[0]


def _declared_columns(synthesizer: Any, columns: Sequence[str] | None) -> list[str]:
    declared = columns if columns is not None else getattr(synthesizer, "columns", None)
    if declared is None:
        raise ValueError(
            f"{type(synthesizer).__name__} has no `columns` attribute, so the harness "
            "cannot tell which columns it is allowed to write. Pass columns=[...] naming "
            "them."
        )
    return list(declared)


def _emit(source: Any, table: Table, schema: Schema, chunk_size: int) -> pd.DataFrame:
    """Everything the generator emits for `table`, as one frame."""
    ctx = RunContext(schema=schema, chunk_size=chunk_size)
    return _collect(source.emit(table, ctx))


def _apply(
    synthesizer: Any, source: Any, table: Table, schema: Schema, chunk_size: int
) -> pd.DataFrame:
    """The synthesizer's output over freshly generated chunks, as one frame.

    Chunks are re-emitted per call rather than replayed from a list, so the
    synthesizer sees the chunk boundaries the pipeline would really hand it
    at this `chunk_size`, and each call gets its own `RunContext` exactly as
    a separate `Pipeline.run()` would.
    """
    ctx = RunContext(schema=schema, chunk_size=chunk_size)
    return _collect(synthesizer.run(source.emit(table, ctx), table, ctx))


def _differing_rows(before: pd.Series, after: pd.Series) -> np.ndarray:
    """Positions where two columns disagree, counting null == null as agreement."""
    left, right = before.to_numpy(dtype=object), after.to_numpy(dtype=object)
    both_null = pd.isna(left) & pd.isna(right)
    return np.flatnonzero(~((left == right) | both_null))


def _show(before: pd.Series, after: pd.Series, rows: np.ndarray, limit: int = 3) -> str:
    return ", ".join(
        f"row {int(i)}: expected {before.iloc[int(i)]!r}, got {after.iloc[int(i)]!r}"
        for i in rows[:limit]
    )


def _collect(chunks: Iterable[pd.DataFrame]) -> pd.DataFrame:
    collected = [chunk for chunk in chunks if len(chunk)]
    if not collected:
        return pd.DataFrame()
    return pd.concat(collected, ignore_index=True)
