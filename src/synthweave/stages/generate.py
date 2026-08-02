"""Stage 1: rule-driven generation from a schema alone.

No real data is involved. Entities are never materialized either: an entity is
an integer index, and its attributes are derived on demand from
(seed, entity key, attribute name). A population of forty million people costs
nothing until rows are produced, and a person's birth date is identical in
every table that carries it without any lookup.

The key convention set here is what every later stage depends on:

    entity key   person:417
    row key      person:417              (PerEntity)
                 person:417|2019         (PerPeriod)
                 person:417|2           (PerEvent)

Entity-level draws use the entity key, so they stay constant across a person's
rows. Row-level draws use the row key, so they vary within a person. That one
distinction is what makes panel data behave like panel data.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

from .. import _hash
from ..context import RunContext
from ..registry import register
from ..rules import as_declared, resolve_order
from ..schema import PerEntity, PerEvent, PerPeriod, Table
from ..validation import ENTITY_KEY
from .base import own


@register("generator", "rules")
class RuleGenerator:
    """Draws every column from its rule, chunk by chunk."""

    def emit(self, table: Table, ctx: RunContext) -> Iterator[pd.DataFrame]:
        entity = ctx.schema.entity(table.entity)
        total = entity.count.value
        coverage = ctx.provenance.add(
            f"{table.name}.coverage", table.coverage, default_origin="user-provided"
        )
        stride = self._entities_per_chunk(table, ctx.chunk_size)

        emitted = 0
        covered = 0
        for start in range(0, total, stride):
            stop = min(start + stride, total)
            keys = self._entity_keys(entity.name, start, stop)

            if coverage < 1.0:
                keep = _hash.unit(keys, ctx.seed, f"coverage\x00{table.name}") < coverage
                keys = keys[keep]
            covered += len(keys)
            if len(keys) == 0:
                continue

            chunk = self._expand(table, keys, ctx)
            if chunk.empty:
                continue
            chunk = self._draw_carried(table, entity, chunk, ctx)
            chunk = self._draw_columns(table, chunk, ctx)
            emitted += len(chunk)
            yield chunk

        ctx.report(
            table.name,
            "generate",
            entities_total=total,
            entities_covered=covered,
            rows=emitted,
        )

    # --- grain ------------------------------------------------------------

    def _entities_per_chunk(self, table: Table, chunk_size: int) -> int:
        """Entities per chunk, sized so output chunks land near `chunk_size`.

        Chunking happens over entities rather than rows so that an entity's
        rows never straddle a boundary. Nothing depends on that (every value
        is position independent), but it keeps chunks self describing.
        """
        grain = table.grain
        if isinstance(grain, PerPeriod):
            per_entity = max(1, int(len(grain.periods) * grain.presence.value))
        elif isinstance(grain, PerEvent):
            per_entity = max(1, (grain.low + grain.high) // 2)
        else:
            per_entity = 1
        return max(1, chunk_size // per_entity)

    def _expand(self, table: Table, keys: np.ndarray, ctx: RunContext) -> pd.DataFrame:
        """One row per entity, per entity-period, or per entity-event."""
        grain = table.grain

        if isinstance(grain, PerEntity):
            return pd.DataFrame({ENTITY_KEY: keys, "_sw_row": keys})

        if isinstance(grain, PerPeriod):
            presence = ctx.provenance.add(
                f"{table.name}.grain.presence", grain.presence, default_origin="user-provided"
            )
            entity_col = np.repeat(keys, len(grain.periods))
            period_col = np.tile(np.asarray(grain.periods, dtype=object), len(keys))
            row_keys = np.char.add(
                np.asarray(entity_col, dtype=str), np.char.add("|", np.asarray(period_col, dtype=str))
            ).astype(object)
            frame = pd.DataFrame(
                {ENTITY_KEY: entity_col, "_sw_row": row_keys, grain.period_column: period_col}
            )
            if presence < 1.0:
                keep = _hash.unit(row_keys, ctx.seed, f"presence\x00{table.name}") < presence
                frame = frame.loc[keep]
            return frame.reset_index(drop=True)

        if isinstance(grain, PerEvent):
            counts = _hash.integers(
                keys, ctx.seed, f"events\x00{table.name}", grain.low, grain.high
            )
            entity_col = np.repeat(keys, counts)
            if len(entity_col) == 0:
                return pd.DataFrame()
            occurrence = np.concatenate([np.arange(c) for c in counts]) if len(counts) else np.array([])
            row_keys = np.char.add(
                np.asarray(entity_col, dtype=str),
                np.char.add("|", np.asarray(occurrence, dtype=str)),
            ).astype(object)
            return pd.DataFrame(
                {
                    ENTITY_KEY: entity_col,
                    "_sw_row": row_keys,
                    grain.occurrence_column: occurrence,
                }
            )

        raise TypeError(f"unknown grain {type(grain).__name__}")

    # --- drawing ----------------------------------------------------------

    def _draw_carried(self, table: Table, entity, chunk: pd.DataFrame, ctx) -> pd.DataFrame:
        """Entity attributes, keyed on the entity so they repeat across its rows."""
        if not table.carry:
            return chunk
        order = resolve_order(entity.attributes)
        keys = chunk[ENTITY_KEY].to_numpy()
        drawn = pd.DataFrame(index=chunk.index)

        # A carried attribute's dependency is drawn too, and so is that
        # dependency's own dependency, transitively. `order` is topologically
        # sorted (dependencies before dependents), so walking it in reverse
        # and growing `needed` as each dependent is confirmed needed reaches
        # the full transitive closure in one pass.
        needed = set(table.carry)
        for name in reversed(order):
            if name in needed:
                needed.update(entity.attributes[name].depends_on())

        for name in order:
            if name not in needed:
                continue
            drawn[name] = _draw(
                entity.attributes[name], keys, ctx, f"entity\x00{entity.name}\x00{name}", drawn
            )
        return pd.concat([chunk, drawn[list(table.carry)]], axis=1)

    def _draw_columns(self, table: Table, chunk: pd.DataFrame, ctx) -> pd.DataFrame:
        """Table columns, keyed on the row so they vary within an entity."""
        if not table.columns:
            return chunk
        chunk = own(chunk)
        keys = chunk["_sw_row"].to_numpy()
        for name in resolve_order(table.columns, available=table.carry):
            chunk[name] = _draw(
                table.columns[name], keys, ctx, f"table\x00{table.name}\x00{name}", chunk
            )
        return chunk

    @staticmethod
    def _entity_keys(entity_name: str, start: int, stop: int) -> np.ndarray:
        return np.array([f"{entity_name}:{i}" for i in range(start, stop)], dtype=object)


def _draw(rule, keys, ctx, salt: str, frame: pd.DataFrame) -> np.ndarray:
    """One column's values, in the type its rule declares."""
    return as_declared(rule, rule.draw(keys, seed=ctx.seed, salt=salt, frame=frame))
