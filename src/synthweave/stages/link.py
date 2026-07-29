"""Stage 3b: deterministic cross-table identifiers.

The linking guarantee is a property of the derivation, not of bookkeeping.
An identifier is a pure function of (seed, entity key, identifier tag), so the
same person yields the same student id in every table carrying student ids,
with no lookup table, no join, and no coordination between tables. Two tables
generated years apart from the same config still line up.

That purity is also what makes linking chunk-safe. There is no state to carry
across chunks, so identifiers cannot drift when chunk size changes.

Tables that list no identifiers get none, which makes a table unlinkable by
construction. That is sometimes exactly the requirement.

Linking runs before noise so that identifier columns exist when noise is
applied. They stay clean unless a noise config names them, and naming one is
how you simulate a mistyped SSN in a matching benchmark.
"""

from __future__ import annotations

from typing import Iterator

import pandas as pd

from .. import _hash
from ..context import RunContext
from ..registry import register
from ..schema import Table
from ..validation import ENTITY_KEY
from .base import own


@register("linker", "deterministic")
class DeterministicLinker:
    """Attaches one column per identifier tag the table carries."""

    def run(
        self, chunks: Iterator[pd.DataFrame], table: Table, ctx: RunContext
    ) -> Iterator[pd.DataFrame]:
        if not table.identifiers:
            yield from chunks
            return

        entity = ctx.schema.entity(table.entity)
        specs = [entity.identifier(tag) for tag in table.identifiers]

        for chunk in chunks:
            chunk = own(chunk)
            keys = chunk[ENTITY_KEY].to_numpy()
            for spec in specs:
                chunk[spec.tag] = _hash.derive_id(
                    keys,
                    ctx.seed,
                    f"{entity.name}\x00{spec.tag}",
                    prefix=spec.prefix,
                    digits=spec.digits,
                )
            # Identifiers lead, matching how administrative extracts are laid out.
            ordered = [s.tag for s in specs] + [c for c in chunk.columns if c not in {s.tag for s in specs}]
            yield chunk[ordered]

        ctx.report(table.name, "link", identifiers=[s.tag for s in specs])
