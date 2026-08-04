"""The pipeline: composes four stages and runs a schema through them.

The pipeline owns none of the stages' internals. It resolves each stage from
the registry, hands chunks from one to the next, and collects what they
report. Adding a stage implementation never requires touching this file,
which is the whole point of the registry.

Stage order is fixed: generate, then synthesize, then link, then noise.

Synthesis precedes noise so the model never fits on corrupted values. Linking
precedes noise so that identifier columns exist by the time noise runs, which
is what lets a user dirty an identifier on purpose by naming it in the noise
config. Identifiers still come out clean unless explicitly named, since the
noiser only touches columns it was told about.

Every stage past the generator is optional. A pipeline with only a generator
is valid and produces clean, unmodeled, unlinked data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from .context import RunContext
from .provenance import ProvenanceRecord
from .registry import resolve
from .rules import as_declared
from .schema import Schema, Table
from .validation import RESERVED_PREFIX, validate_schema


@dataclass
class PipelineResult:
    """Output tables plus everything needed to explain how they were made."""

    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    provenance: ProvenanceRecord = field(default_factory=ProvenanceRecord)
    metadata: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)

    def __getitem__(self, name: str) -> pd.DataFrame:
        if name not in self.tables:
            raise KeyError(f"no table {name!r} in result; produced: {sorted(self.tables)}")
        return self.tables[name]

    def unjustified(self):
        """Config values that are library defaults or bare assumptions."""
        return self.provenance.unjustified()

    def summary(self) -> pd.DataFrame:
        """Row counts and column counts per table."""
        rows = [
            {"table": name, "rows": len(df), "columns": len(df.columns)}
            for name, df in self.tables.items()
        ]
        return pd.DataFrame(rows, columns=["table", "rows", "columns"])


class Pipeline:
    """Runs a schema through the four stages.

    Args:
        schema: entities, tables, and seed.
        generator: stage 1. Registered name or instance.
        synthesizer: stage 2, or None to skip.
        noiser: stage 3a, or None to skip.
        linker: stage 3b, or None to skip.
        chunk_size: rows per chunk. A memory knob only. Because every value
            derives from (seed, key, salt) rather than from RNG state,
            changing it cannot change the output.
    """

    def __init__(
        self,
        schema: Schema,
        *,
        generator: Any = "rules",
        synthesizer: Any = None,
        noiser: Any = None,
        linker: Any = "deterministic",
        chunk_size: int = 100_000,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        validate_schema(schema)
        self.schema = schema
        self.chunk_size = chunk_size
        self.generator = resolve("generator", generator)
        self.synthesizer = resolve("synthesizer", synthesizer) if synthesizer else None
        self.noiser = resolve("noiser", noiser) if noiser else None
        self.linker = resolve("linker", linker) if linker else None

    def _context(self) -> RunContext:
        return RunContext(schema=self.schema, chunk_size=self.chunk_size)

    def _stream(self, table: Table, ctx: RunContext) -> Iterator[pd.DataFrame]:
        """Chunks for one table, passed through every configured stage.

        Reserved bookkeeping columns (the entity key and row key that stages
        use to derive values) are stripped last, so they never reach the user
        but every stage can rely on them being present.
        """
        chunks = self.generator.emit(table, ctx)
        for stage in (self.synthesizer, self.linker, self.noiser):
            if stage is not None:
                chunks = stage.run(chunks, table, ctx)
        return _strip_reserved(chunks)

    def run(self) -> PipelineResult:
        """Run every table and materialize the results in memory.

        The convenience path, for data that fits in memory. Use `run_to` when
        it does not.
        """
        ctx = self._context()
        tables = {
            table.name: _concat(self._stream(table, ctx), self._empty_frame(table))
            for table in self.schema.tables
        }
        return PipelineResult(tables=tables, provenance=ctx.provenance, metadata=ctx.metadata)

    def run_to(self, directory: str | Path, *, format: str = "parquet") -> PipelineResult:
        """Run every table, writing chunks to disk as they are produced.

        Nothing accumulates in memory, so this is the path for output larger
        than RAM. The returned result carries provenance, metadata, and file
        paths, but no tables.
        """
        from .io import ChunkWriter

        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        ctx = self._context()
        paths: dict[str, str] = {}

        for table in self.schema.tables:
            # The same typed stand-in `run` hands back for a table that
            # produced nothing, so the file and the in-memory result cannot
            # disagree about the schema of a zero-row run.
            writer = ChunkWriter(
                out, table.name, format=format, empty=self._empty_frame(table)
            )
            rows = 0
            for chunk in self._stream(table, ctx):
                writer.write(chunk)
                rows += len(chunk)
            paths[table.name] = str(writer.close())
            ctx.report(table.name, "output", rows=rows, path=paths[table.name])

        return PipelineResult(provenance=ctx.provenance, metadata=ctx.metadata, paths=paths)

    def _columns_of(self, table) -> list[str]:
        """What this pipeline will emit for a table, identifiers included only
        when a linker is actually configured to attach them."""
        return table.output_columns(with_identifiers=self.linker is not None)

    def _empty_frame(self, table: Table) -> pd.DataFrame:
        """The frame a table stands in with when it produced no rows at all.

        Zero rows is data, not an error: coverage, presence, or an event count
        of zero can legitimately exclude every entity. What must not change is
        the table's *type*, because a caller writing the result to Parquet
        would otherwise get one schema on a run that produced rows and another
        on a run that did not.

        There is nothing to infer a type from here, so each column is built by
        the same `as_declared` call the generator ends every populated draw
        with. That keeps one source of truth, the rule's own `dtype()`: this
        path cannot drift from the populated one without the populated one
        moving too. A column whose rule declares nothing (an identifier, a
        grain column, a `Sequential`) is left `object`, exactly as before,
        since a rule that will not name its type cannot be second-guessed
        from zero rows.
        """
        entity = self.schema.entity(table.entity)
        rules = {name: entity.attributes[name] for name in table.carry}
        rules.update(table.columns)
        empty = np.array([], dtype=object)
        columns = self._columns_of(table)
        return pd.DataFrame(
            {
                name: as_declared(rules[name], empty) if name in rules else empty
                for name in columns
            },
            columns=columns,
        )

    def stream(self, table_name: str) -> Iterator[pd.DataFrame]:
        """Chunks for one table, for a caller doing its own streaming."""
        return self._stream(self.schema.table(table_name), self._context())


def _strip_reserved(chunks: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    for chunk in chunks:
        drop = [c for c in chunk.columns if str(c).startswith(RESERVED_PREFIX)]
        yield chunk.drop(columns=drop) if drop else chunk


def _concat(chunks: Iterator[pd.DataFrame], empty: pd.DataFrame) -> pd.DataFrame:
    collected = [c for c in chunks if len(c)]
    if not collected:
        # A table can legitimately end up with no rows, most often because
        # coverage excluded every entity. Handing back a frame with no columns
        # would mean downstream code cannot even read the schema it was
        # promised, so the declared shape stands in. `empty` carries the
        # declared dtypes too: see `Pipeline._empty_frame`.
        return empty
    if len(collected) == 1:
        return collected[0].reset_index(drop=True)
    return pd.concat(collected, ignore_index=True)
