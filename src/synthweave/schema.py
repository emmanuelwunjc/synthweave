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
from .rules import coerce_rule

# 10**19 already exceeds the unsigned 64-bit range the hash reduces into, so
# the modulus stops bounding the value and identifiers come back at mixed
# widths. 18 is the largest width the derivation can actually honour.
MAX_DIGITS = 18


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
        if self.digits > MAX_DIGITS:
            raise ValueError(
                f"Identifier {self.tag!r}: digits must be at most {MAX_DIGITS}, because "
                f"10**{self.digits} does not fit the 64-bit hash the derivation uses. "
                f"Past that the values stop being the width you asked for."
            )


@dataclass
class Entity:
    """A population of things that appear across tables."""

    name: str
    count: int | Tagged
    attributes: Mapping[str, Any] = field(default_factory=dict)
    identifiers: Sequence[Identifier | str] = field(default_factory=tuple)

    def __post_init__(self):
        self.count = as_tagged(self.count)
        if self.count.value < 1:
            raise ValueError(f"entity {self.name!r}: count must be at least 1")
        self.attributes = {k: coerce_rule(v) for k, v in self.attributes.items()}
        # A bare tag string is shorthand for Identifier(tag=that_string); an
        # already-built Identifier passes through untouched.
        self.identifiers = [
            Identifier(tag=i) if isinstance(i, str) else i for i in self.identifiers
        ]
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

    def emitted_column(self) -> str | None:
        """The column this grain produces, beyond the entity's own."""
        return None


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

    def emitted_column(self) -> str | None:
        return self.period_column

    def __post_init__(self):
        if len(self.periods) == 0:
            raise ValueError("PerPeriod needs at least one period")
        repeated = sorted({p for p in self.periods if list(self.periods).count(p) > 1}, key=str)
        if repeated:
            # One row per entity per period is the whole promise of this grain.
            # A repeated period breaks the panel key silently, and it is always
            # a config slip rather than something anyone means.
            raise ValueError(f"PerPeriod has repeated period(s) {repeated}")
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

    def emitted_column(self) -> str | None:
        return self.occurrence_column

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
        grain: how entities map to rows. A bare entity name string is
            shorthand for `PerEntity(that_name)`.
        columns: rules for columns generated at row level. A value that is
            not already a `Rule` is coerced with `coerce_rule`.
        carry: entity attribute names to copy onto every row. These stay
            identical everywhere the entity appears, which is what makes a
            person's birth date consistent across tables. `"*"` carries
            every attribute the table's entity has.
        identifiers: identifier tags this table carries. A table that lists
            none is unlinkable by construction, which is sometimes the point.
        coverage: share of the entity population appearing in this table.
            Below 1.0 produces realistic coverage gaps between tables rather
            than perfect overlap.
    """

    name: str
    grain: Grain | str
    columns: Mapping[str, Any] = field(default_factory=dict)
    carry: Sequence[str] | str = field(default_factory=tuple)
    identifiers: Sequence[str] = field(default_factory=tuple)
    coverage: float | Tagged = 1.0

    def __post_init__(self):
        # A bare entity name is shorthand for the common case, one row per
        # covered entity. PerPeriod/PerEvent still need to be spelled out,
        # since a bare name can't say which of those is meant either.
        if isinstance(self.grain, str):
            self.grain = PerEntity(self.grain)
        self.columns = {k: coerce_rule(v) for k, v in self.columns.items()}
        self.coverage = as_tagged(self.coverage)
        if not 0.0 < self.coverage.value <= 1.0:
            raise ValueError(f"table {self.name!r}: coverage must be in (0, 1]")

    @property
    def entity(self) -> str:
        return self.grain.entity

    def output_columns(self, *, with_identifiers: bool = True) -> list[str]:
        """The columns this table produces, in the order a run emits them.

        Known from config alone, before a single row exists. That is what lets
        a table which ends up with no rows still hand back its shape instead of
        an empty frame with nothing on it.
        """
        columns = list(self.identifiers) if with_identifiers else []
        produced = self.grain.emitted_column()
        if produced is not None:
            columns.append(produced)
        return columns + list(self.carry) + list(self.columns)


@dataclass
class Schema:
    """Entities, tables, and the run seed that makes it all reproducible."""

    entities: Sequence[Entity]
    tables: Sequence[Table]
    seed: int | str = 0

    def __post_init__(self):
        # carry="*" needs the entity to resolve against, which only Schema
        # can see; Table alone only knows its entity by name.
        for table in self.tables:
            if table.carry == "*":
                table.carry = tuple(self.entity(table.entity).attributes.keys())

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
