"""Entity-first schema: what exists, and at what grain each table records it.

Entities come first and tables are built over them. A person is defined once,
with their own attributes and their own identifier kinds, and then any number
of tables record that person at whatever grain suits the table. A roster is
one row per person; a wage file is one row per person per quarter; a
transaction file is one row per event.

Keeping entities separate from tables is what lets the linker stay ignorant of
table shape. Identifiers derive from entity identity alone, so the same
derivation serves a roster row and a panel row with no special cases.

Entities are never materialized. An entity is just an integer index plus a set
of rules for deriving its attributes on demand, so a population of forty
million people costs nothing until rows are actually produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .provenance import Tagged, as_tagged
from .rules import Rule


@dataclass(frozen=True)
class Identifier:
    """An identifier kind an entity carries.

    `tag` separates independent identifier streams for the same entity. A
    person's student id and tax id use different tags, so neither can be
    derived from the other, which is what makes them behave like real
    administrative identifiers rather than two views of one number.
    """

    tag: str
    prefix: str = ""
    digits: int = 10

    def __post_init__(self):
        if not self.tag:
            raise ValueError("Identifier needs a non-empty tag")
        if self.digits < 1:
            raise ValueError(f"Identifier {self.tag!r}: digits must be at least 1")


@dataclass
class Entity:
    """A population of things that appear across tables."""

    name: str
    count: int | Tagged
    attributes: Mapping[str, Rule] = field(default_factory=dict)
    identifiers: Sequence[Identifier] = field(default_factory=tuple)

    def __post_init__(self):
        self.count = as_tagged(self.count)
        if self.count.value < 1:
            raise ValueError(f"entity {self.name!r}: count must be at least 1")
        tags = [i.tag for i in self.identifiers]
        dupes = {t for t in tags if tags.count(t) > 1}
        if dupes:
            raise ValueError(f"entity {self.name!r}: duplicate identifier tags {sorted(dupes)}")

    def identifier(self, tag: str) -> Identifier:
        for ident in self.identifiers:
            if ident.tag == tag:
                return ident
        known = [i.tag for i in self.identifiers]
        raise KeyError(f"entity {self.name!r} has no identifier {tag!r}; it has {known}")


# --- Grain ------------------------------------------------------------------
# A grain answers: how many rows does one entity produce in this table, and
# what is the stable key of each of those rows?


@dataclass(frozen=True)
class PerEntity:
    """One row per covered entity. A roster or registry."""

    entity: str


@dataclass(frozen=True)
class PerPeriod:
    """One row per covered entity per period. Panel data.

    `presence` is the chance an entity appears in any given period. Below 1.0
    the panel is unbalanced, which is what real administrative panels look
    like: people enter, leave, and have gap quarters.
    """

    entity: str
    periods: Sequence[Any]
    presence: float | Tagged = 1.0
    period_column: str = "period"

    def __post_init__(self):
        if len(self.periods) == 0:
            raise ValueError("PerPeriod needs at least one period")
        object.__setattr__(self, "presence", as_tagged(self.presence))
        if not 0.0 < self.presence.value <= 1.0:
            raise ValueError("PerPeriod presence must be in (0, 1]")


@dataclass(frozen=True)
class PerEvent:
    """A variable number of rows per covered entity. Transactional records."""

    entity: str
    low: int = 1
    high: int = 5
    occurrence_column: str = "occurrence"

    def __post_init__(self):
        if self.low < 0:
            raise ValueError("PerEvent low must be non-negative")
        if self.high <= self.low:
            raise ValueError("PerEvent needs high > low")


Grain = PerEntity | PerPeriod | PerEvent


@dataclass
class Table:
    """One output table.

    Args:
        name: output table name.
        grain: how entities map to rows.
        columns: rules for columns generated at row level.
        carry: entity attribute names to copy onto every row. These stay
            identical everywhere the entity appears, which is what makes a
            person's birth date consistent across tables.
        identifiers: identifier tags this table carries. A table that lists
            none is unlinkable by construction, which is sometimes the point.
        coverage: share of the entity population appearing in this table.
            Below 1.0 produces realistic coverage gaps between tables rather
            than perfect overlap.
    """

    name: str
    grain: Grain
    columns: Mapping[str, Rule] = field(default_factory=dict)
    carry: Sequence[str] = field(default_factory=tuple)
    identifiers: Sequence[str] = field(default_factory=tuple)
    coverage: float | Tagged = 1.0

    def __post_init__(self):
        self.coverage = as_tagged(self.coverage)
        if not 0.0 < self.coverage.value <= 1.0:
            raise ValueError(f"table {self.name!r}: coverage must be in (0, 1]")

    @property
    def entity(self) -> str:
        return self.grain.entity


@dataclass
class Schema:
    """Entities, tables, and the run seed that makes it all reproducible."""

    entities: Sequence[Entity]
    tables: Sequence[Table]
    seed: int | str = 0

    def entity(self, name: str) -> Entity:
        for e in self.entities:
            if e.name == name:
                return e
        raise KeyError(f"no entity named {name!r}; known entities: {[e.name for e in self.entities]}")

    def table(self, name: str) -> Table:
        for t in self.tables:
            if t.name == name:
                return t
        raise KeyError(f"no table named {name!r}; known tables: {[t.name for t in self.tables]}")
