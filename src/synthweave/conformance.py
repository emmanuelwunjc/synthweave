"""`check_generator` and `check_synthesizer`: conformance harnesses for stages.

One public checker per registry kind, so a plugin author has a mechanical way
to verify the contracts their implementation must honour rather than reading
them out of docstrings. `check_rule` (in `rules.py`, where `Rule` lives) is the
same idea for the smallest of them.

`check_synthesizer` is the synthesizer-shaped analogue of `check_rule`.
`Synthesizer` is a `runtime_checkable` Protocol matched on `run` alone, so
`isinstance` says nothing about the guarantees the stage docstrings promise.
This runs a candidate over real generated chunks and checks each guarantee,
raising `SynthesizerConformanceError` naming which one broke.

    sw.check_synthesizer(MySynthesizer(columns=["wage"]), schema)

Nothing here is specific to a built-in: the harness takes any object with a
`run(chunks, table, ctx)`, which is what lets a plugin author check their own
implementation against the same contract the built-ins meet.

`check_generator` does the same for stage 1, over the five guarantees a
generator makes about the chunks it emits. It needs a whole `Schema` rather
than a bare object, because a generator's output is a function of the config
and nothing else; that is a real ergonomic difference from `check_rule`, and
the reason it takes `schema` positionally.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .context import RunContext
from .registry import resolve
from .schema import PerEntity, Schema, Table
from .validation import ENTITY_KEY, RESERVED_PREFIX, ROW_KEY


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


def _target_table(
    schema: Schema, table: str | None, *, kind: str = "synthesizer"
) -> Table:
    if table is not None:
        return schema.table(table)
    if len(schema.tables) != 1:
        raise ValueError(
            f"schema has {len(schema.tables)} tables "
            f"({sorted(t.name for t in schema.tables)}); pass table= to name the one "
            f"to check the {kind} against"
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


# --- stage 1 ----------------------------------------------------------------


class GeneratorConformanceError(AssertionError):
    """A custom Generator broke one of the guarantees `check_generator` verifies."""


def check_generator(
    generator: Any,
    schema: Schema,
    *,
    table: str | None = None,
    chunk_size: int = 100_000,
    split_chunk_size: int = 13,
) -> None:
    """Verify a generator honours the guarantees stage 1 promises.

    Runs `generator.emit` over one of `schema`'s tables at two chunk sizes and
    checks, in order:

    1. Determinism. Two runs of the same table at the same chunk size agree,
       so no hidden RNG state is involved.
    2. Emitted columns. Every chunk carries the reserved bookkeeping keys that
       later stages derive from, plus exactly the columns the table declares,
       in the order `Table.output_columns` promises.
    3. Entity non-straddling. Every row belonging to one entity arrives in the
       same chunk. Generation chunks over entities rather than rows precisely
       so that "an entity's rows never straddle a boundary", which is what
       lets a consumer treat one chunk as self describing. A generator that
       sliced by rows instead would satisfy every other clause here and still
       hand that consumer half an entity, so nothing but this clause can see
       it.
    4. Chunk invariance. Running at `split_chunk_size` yields the same rows as
       at `chunk_size`. chunk_size is a memory knob and nothing else.
    5. Row count. A `PerEntity` table at `coverage=1.0` has exactly one row per
       declared entity. Stated only for that case, because it is the only one
       where config alone fixes the answer: `PerEvent` draws its own row count
       and `coverage` below 1.0 keeps a hash-selected subset, so neither has a
       number to compare against. On any other table the clause is skipped,
       and an entity the generator silently omits there is invisible to it.

    Raises `GeneratorConformanceError` naming which clause broke, or returns
    `None` if the generator passes all five.

    `table` names which of `schema`'s tables to emit, and may be omitted only
    when the schema has exactly one. `split_chunk_size` must be small enough
    to split that table into several chunks: clauses 3 and 4 are about chunk
    boundaries, and a size that leaves the whole table in one chunk cannot
    show a boundary problem at all.
    """
    target = _target_table(schema, table, kind="generator")

    baseline = _generator_chunks(generator, target, schema, chunk_size)
    _check_generator_determinism(
        baseline, _generator_chunks(generator, target, schema, chunk_size)
    )

    split = _generator_chunks(generator, target, schema, split_chunk_size)
    _check_emitted_columns(split, target)
    _check_entity_non_straddling(split, split_chunk_size)
    _check_chunk_invariance(baseline, split, split_chunk_size)
    _check_row_count(split, target, schema)


# --- stage 1's individual checks --------------------------------------------


def _check_generator_determinism(
    first: list[pd.DataFrame], second: list[pd.DataFrame]
) -> None:
    """Clause 1. Two runs of the same table at the same chunk size agree."""
    _check_same_rows(
        "determinism", _collect(first), _collect(second), "a second emit() of the same table"
    )


def _check_emitted_columns(chunks: list[pd.DataFrame], table: Table) -> None:
    """Clause 2. Each chunk carries the bookkeeping keys and the declared columns.

    Both halves matter and neither is checked anywhere else. The reserved keys
    are what every later stage derives from and the pipeline strips them last,
    so a generator that omits one produces a table the linker cannot key. The
    declared columns are what the user asked for, in the order
    `Table.output_columns` promises. Identifiers are excluded because the
    linker attaches those, not the generator.
    """
    expected = table.output_columns(with_identifiers=False)
    for index, chunk in enumerate(chunks):
        for key in (ENTITY_KEY, ROW_KEY):
            if key not in chunk.columns:
                _fail_generator(
                    "emitted columns",
                    f"chunk {index} is missing the bookkeeping column {key!r} that every "
                    f"later stage derives from; it has {list(chunk.columns)}",
                )
        produced = [c for c in chunk.columns if not str(c).startswith(RESERVED_PREFIX)]
        if produced != expected:
            _fail_generator(
                "emitted columns",
                f"chunk {index} produced {produced}, but table {table.name!r} declares "
                f"{expected}",
            )


def _check_entity_non_straddling(chunks: list[pd.DataFrame], chunk_size: int) -> None:
    """Clause 3. Every row of an entity arrives in the same chunk.

    The guarantee `RuleGenerator._entities_per_chunk` states and nothing
    asserted until this existed. Straddling corrupts nothing derived from
    `(seed, key, salt)`, since every value is position independent, but it
    silently breaks any consumer that treats a chunk as self describing: a
    running total, a first-visit flag, a within-entity sort, a chunk written
    to its own file.
    """
    seen: dict[Any, int] = {}
    for index, chunk in enumerate(chunks):
        if ENTITY_KEY not in chunk.columns:
            _fail_generator(
                "entity non-straddling",
                f"chunk {index} has no {ENTITY_KEY!r} column, so an entity's rows "
                f"cannot be located; it has {list(chunk.columns)}",
            )
        for key in pd.unique(chunk[ENTITY_KEY]):
            if key in seen and seen[key] != index:
                _fail_generator(
                    "entity non-straddling",
                    f"entity {key!r} has rows in chunk {seen[key]} and again in chunk "
                    f"{index} at chunk_size={chunk_size}; an entity's rows must not be "
                    f"split across a chunk boundary",
                )
            seen[key] = index


def _check_chunk_invariance(
    baseline: list[pd.DataFrame], split: list[pd.DataFrame], split_chunk_size: int
) -> None:
    """Clause 4. The rows are the same however the stream was cut up.

    Compares the concatenation, not the chunks: how many chunks a size
    produces is exactly what `chunk_size` is allowed to change.
    """
    _check_same_rows(
        "chunk invariance",
        _collect(baseline),
        _collect(split),
        f"chunk_size={split_chunk_size}",
    )


def _check_row_count(chunks: list[pd.DataFrame], table: Table, schema: Schema) -> None:
    """Clause 5. A fully covered PerEntity table has one row per entity."""
    if not _row_count_is_fixed_by_config(table):
        return
    expected = schema.entity(table.entity).count.value
    rows = _collect(chunks)
    if len(rows) != expected:
        _fail_generator(
            "row count",
            f"table {table.name!r} emitted {len(rows)} rows for an entity declaring "
            f"count={expected} at coverage=1.0",
        )
    distinct = rows[ENTITY_KEY].nunique()
    if distinct != expected:
        _fail_generator(
            "row count",
            f"table {table.name!r} emitted {len(rows)} rows covering only {distinct} "
            f"distinct entities, but count={expected} at coverage=1.0 means one row each",
        )


def _row_count_is_fixed_by_config(table: Table) -> bool:
    """Whether config alone says how many rows this table must have.

    Only `PerEntity` at full coverage does. Named rather than inlined so the
    condition that decides whether clause 5 runs at all is something a test
    can assert directly, instead of a silent skip nobody can see.
    """
    return isinstance(table.grain, PerEntity) and table.coverage.value == 1.0


# --- stage 1's plumbing -----------------------------------------------------


def _generator_chunks(
    generator: Any, table: Table, schema: Schema, chunk_size: int
) -> list[pd.DataFrame]:
    """One emit() run's chunks, kept apart rather than concatenated.

    Chunk boundaries are the subject of two of the five clauses, so they have
    to survive collection. `_emit` above concatenates, which erases exactly
    the evidence those clauses read.
    """
    ctx = RunContext(schema=schema, chunk_size=chunk_size)
    return list(generator.emit(table, ctx))


def _check_same_rows(
    clause: str, expected: pd.DataFrame, actual: pd.DataFrame, what: str
) -> None:
    if len(expected) != len(actual):
        _fail_generator(
            clause, f"{what} produced {len(actual)} rows, expected {len(expected)}"
        )
    if list(expected.columns) != list(actual.columns):
        _fail_generator(
            clause,
            f"{what} produced columns {list(actual.columns)}, expected "
            f"{list(expected.columns)}",
        )
    for column in expected.columns:
        rows = _differing_rows(expected[column], actual[column])
        if len(rows):
            _fail_generator(
                clause,
                f"{what} changed column {column!r} in {len(rows)} of {len(expected)} "
                f"row(s), e.g. {_show(expected[column], actual[column], rows)}",
            )


def _fail_generator(clause: str, detail: str) -> None:
    """Raise naming the clause first.

    The clause prefix is load-bearing rather than cosmetic: a conformance
    check can go red for a reason other than the one it is about, and the
    prefix is what lets a test assert the failure is the property it meant to
    provoke instead of an incidental one.
    """
    raise GeneratorConformanceError(f"{clause}: {detail}")
