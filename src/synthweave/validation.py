"""Config validation, run before any generation work starts.

At the target scale a config error found halfway through a run is expensive,
so everything checkable is checked up front: references that do not resolve,
rule dependency cycles, name collisions, and identifiers a table asks for that
its entity does not define.
"""

from __future__ import annotations

from .rules import resolve_order
from .schema import PerEvent, PerPeriod, Schema

RESERVED_PREFIX = "_sw_"
ENTITY_KEY = "_sw_entity"
ROW_KEY = "_sw_row"


class SchemaError(ValueError):
    """A config problem the user needs to fix before a run can start."""


def validate_schema(schema: Schema) -> None:
    """Raise SchemaError on the first problem found."""
    if not schema.entities:
        raise SchemaError("schema defines no entities")
    if not schema.tables:
        raise SchemaError("schema defines no tables")

    _check_unique([e.name for e in schema.entities], "entity")
    _check_unique([t.name for t in schema.tables], "table")

    for entity in schema.entities:
        for name in entity.attributes:
            _check_not_reserved(name, f"entity {entity.name!r} attribute")
        try:
            resolve_order(entity.attributes)
        except ValueError as exc:
            raise SchemaError(f"entity {entity.name!r}: {exc}") from exc

    for table in schema.tables:
        _validate_table(schema, table)


def _validate_table(schema: Schema, table) -> None:
    where = f"table {table.name!r}"

    try:
        entity = schema.entity(table.entity)
    except KeyError as exc:
        raise SchemaError(f"{where}: {exc}") from exc

    for name in table.columns:
        _check_not_reserved(name, f"{where} column")

    try:
        resolve_order(table.columns, available=table.carry)
    except ValueError as exc:
        raise SchemaError(f"{where}: {exc}") from exc

    for attr in table.carry:
        if attr not in entity.attributes:
            raise SchemaError(
                f"{where} carries attribute {attr!r}, which entity {entity.name!r} "
                f"does not define; it has {sorted(entity.attributes)}"
            )
        if attr in table.columns:
            raise SchemaError(
                f"{where}: {attr!r} is both a carried entity attribute and a table column"
            )

    known_tags = {i.tag for i in entity.identifiers}
    for tag in table.identifiers:
        if tag not in known_tags:
            raise SchemaError(
                f"{where} asks for identifier {tag!r}, which entity {entity.name!r} "
                f"does not define; it has {sorted(known_tags)}"
            )

    grain = table.grain
    if isinstance(grain, PerPeriod):
        if grain.period_column in table.columns or grain.period_column in table.carry:
            raise SchemaError(
                f"{where}: {grain.period_column!r} is produced by the grain and "
                f"cannot also be a column"
            )
    elif isinstance(grain, PerEvent):
        if grain.occurrence_column in table.columns or grain.occurrence_column in table.carry:
            raise SchemaError(
                f"{where}: {grain.occurrence_column!r} is produced by the grain and "
                f"cannot also be a column"
            )


def _check_unique(names: list[str], kind: str) -> None:
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise SchemaError(f"duplicate {kind} names: {dupes}")


def _check_not_reserved(name: str, where: str) -> None:
    if name.startswith(RESERVED_PREFIX):
        raise SchemaError(f"{where} {name!r} uses the reserved prefix {RESERVED_PREFIX!r}")
