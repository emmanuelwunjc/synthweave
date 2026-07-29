"""The four stage interfaces.

These are the actual v0.1 deliverable. The implementations that ship alongside
them are preliminary on purpose and expected to be replaced; the interfaces
are the commitment, because replacing an implementation should be a version
bump and not a restructuring.

Every stage past the first has the same shape:

    run(chunks, table, ctx) -> chunks

Chunks in, chunks out. A stage never sees a whole table unless it chooses to
buffer one, which is what lets a forty million row run stream through a
fixed amount of memory. The generator is the only stage with a different
shape, because it is the source: it has no upstream chunks to consume.

Interfaces are Protocols rather than base classes, so an implementation does
not need to import or inherit from anything to be usable. Matching the shape
is enough.

Chunk ownership: a stage may mutate the chunk it is handed, but a chunk can be
a view onto a larger frame, so a stage that adds or replaces a column must copy
first. `own(chunk)` does this. Skipping it produces a pandas
SettingWithCopyWarning and, worse, a write that may not land.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Protocol, runtime_checkable

import pandas as pd

from ..context import RunContext
from ..schema import Table

Chunks = Iterator[pd.DataFrame]


@runtime_checkable
class Generator(Protocol):
    """Stage 1. Produces seed rows from the schema alone, with no input data."""

    def emit(self, table: Table, ctx: RunContext) -> Chunks: ...


@runtime_checkable
class Synthesizer(Protocol):
    """Stage 2. Adds statistical structure.

    Structure comes from a `StructureSource`, not from the input chunks. This
    matters: a generator drawing columns independently produces data with no
    inter-column structure, so a model fitted on its output would learn
    nothing and return independent columns. Taking structure from a separate
    source is what makes the no-real-data path actually produce structured
    output.
    """

    def run(self, chunks: Chunks, table: Table, ctx: RunContext) -> Chunks: ...


@runtime_checkable
class Noiser(Protocol):
    """Stage 3a. Applies messiness: typos, OCR confusions, missingness."""

    def run(self, chunks: Chunks, table: Table, ctx: RunContext) -> Chunks: ...


@runtime_checkable
class Linker(Protocol):
    """Stage 3b. Attaches deterministic cross-table identifiers.

    Runs after noise so identifiers are not themselves corrupted. Simulating
    dirty identifiers is a job for a noise config that names the identifier
    column, not something the linker should do by default.
    """

    def run(self, chunks: Chunks, table: Table, ctx: RunContext) -> Chunks: ...


@runtime_checkable
class StructureSource(Protocol):
    """Where a synthesizer's inter-column structure comes from.

    Three kinds ship: structure declared in config via conditional rules,
    structure learned from real data the user supplies, and structure taken
    from published aggregates such as marginals or a correlation matrix.
    """

    def training_frame(self, table: Table, ctx: RunContext) -> pd.DataFrame | None:
        """Rows to fit on, or None to fit on the incoming chunks themselves."""
        ...


def own(chunk: pd.DataFrame) -> pd.DataFrame:
    """A chunk the caller may safely write columns to.

    Copies only when the frame is a view onto something larger, so the common
    case where a stage already owns its chunk costs nothing.

    The check reads private pandas attributes through `getattr` defaults on
    purpose. Under pandas 3 copy-on-write these attributes go away and the
    hazard goes with them, since a write to a slice then lands on a copy
    automatically. Missing attributes must therefore read as "no copy needed"
    rather than raising.
    """
    if getattr(chunk, "_is_view", False) or getattr(chunk, "_is_copy", None) is not None:
        return chunk.copy()
    return chunk


def buffer_to(chunks: Iterable[pd.DataFrame], max_rows: int) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Pull chunks until `max_rows` rows are available, returning them and the rest.

    The fit cap in action. A table smaller than the cap is buffered whole and
    fitted on every row, exactly as a single-table tool would do. A larger one
    stops at the cap. Consumed chunks are returned alongside so the caller can
    replay them downstream without re-generating anything.

    The buffer is truncated to exactly `max_rows`, and that truncation is what
    makes fitting chunk invariant. Without it the buffer would hold whole
    chunks, so a chunk size of 113 would fit on 339 rows while a chunk size of
    100,000 fitted on all of them. Different training data means a different
    model and different output, which would break the guarantee that chunk
    size is only a memory knob. Since the stream is deterministically ordered,
    the first `max_rows` rows are identical however they were chunked.
    """
    seen: list[pd.DataFrame] = []
    total = 0
    for chunk in chunks:
        seen.append(chunk)
        total += len(chunk)
        if total >= max_rows:
            break
    if not seen:
        return pd.DataFrame(), []
    buffered = pd.concat(seen, ignore_index=True) if len(seen) > 1 else seen[0]
    if len(buffered) > max_rows:
        buffered = buffered.iloc[:max_rows]
    return buffered.reset_index(drop=True), seen
