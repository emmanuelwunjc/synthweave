"""Per-run state shared by every stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .provenance import ProvenanceRecord
from .schema import Schema


@dataclass
class RunContext:
    """What a stage needs to know about the run it is part of.

    Deliberately small. A stage gets the seed, the schema, somewhere to record
    provenance, and somewhere to report what it did. It does not get a handle
    on other stages, which is what keeps stages from growing dependencies on
    each other.
    """

    schema: Schema
    chunk_size: int = 100_000
    provenance: ProvenanceRecord = field(default_factory=ProvenanceRecord)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def seed(self) -> int | str:
        return self.schema.seed

    def report(self, table: str, stage: str, **facts: Any) -> None:
        """Record what a stage did to a table.

        Surfaces on the result so a user can check realized noise rates or the
        synthesizer's fit size without reaching into stage internals.
        """
        self.metadata.setdefault(table, {}).setdefault(stage, {}).update(facts)
