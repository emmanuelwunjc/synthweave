"""Config validation, run before any generation work starts.

At the target scale a config error found halfway through a run is expensive,
so everything checkable is checked up front: references that do not resolve,
rule dependency cycles, name collisions, and identifiers a table asks for that
its entity does not define.
"""

from __future__ import annotations

from .rules import resolve_order
from .schema import MAX_DIGITS, PerEvent, PerPeriod, Schema

RESERVED_PREFIX = "_sw_"
# Below one expected collision, every entity is expected to keep its own
# identifier. At one or above, the run is expected to hand back at least one
# identifier that means two different entities, which the linker exists to
# prevent.
TOLERABLE_COLLISIONS = 1
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

    _check_unique(list(table.carry), f"{where} carried attribute")

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

    _check_unique(list(table.identifiers), f"{where} identifier")

    known_tags = {i.tag for i in entity.identifiers}
    for tag in table.identifiers:
        if tag not in known_tags:
            raise SchemaError(
                f"{where} asks for identifier {tag!r}, which entity {entity.name!r} "
                f"does not define; it has {sorted(known_tags)}"
            )
        # The linker assigns identifiers last, so without this check a tag
        # sharing a name with a column would overwrite it and the run would
        # report success while the modelled values were gone.
        if tag in table.columns:
            raise SchemaError(
                f"{where}: identifier {tag!r} has the same name as a table column, "
                f"and the identifier would overwrite it"
            )
        if tag in table.carry:
            raise SchemaError(
                f"{where}: identifier {tag!r} has the same name as the carried entity "
                f"attribute {tag!r}, and the identifier would overwrite it"
            )
        _check_identifier_width(entity, entity.identifier(tag), where)

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


def _expected_collisions(digits: int, population: int) -> float:
    """Roughly how many entities will share an identifier with another.

    The birthday bound, not the keyspace size. Nine digits looks roomy for
    400,000 people until the bound puts about 80 collisions in it, which is
    what the measurement in I14 found.
    """
    return population * population / (2 * 10**digits)


def _check_identifier_width(entity, identifier, where: str) -> None:
    """Refuse a keyspace too small to number the population it has to number.

    Judged on the birthday bound rather than the raw keyspace, because that is
    what the measurement showed: 9 digits over 400,000 entities looks roomy and
    still produces around 80 identifiers that refer to two different people. A
    duplicate identifier means two entities are indistinguishable in the
    output, which breaks the one guarantee the linker exists to make, so this
    is an error rather than a warning.
    """
    population = entity.count.value
    expected = _expected_collisions(identifier.digits, population)
    if expected < TOLERABLE_COLLISIONS:
        return
    # Invert the bound: the narrowest keyspace whose expectation stays under
    # one collision is 10**digits > population**2 / 2, so `needed` is just the
    # digit count of that threshold. A value with `d` digits is already less
    # than 10**d, so no "+1" belongs on top of it.
    needed = len(str(population * population // 2))
    if needed > MAX_DIGITS:
        raise SchemaError(
            f"{where}: identifier {identifier.tag!r} cannot stay under one expected "
            f"collision for {population:,} {entity.name!r} entities; that would need "
            f"{needed} digits, past the {MAX_DIGITS}-digit limit the derivation supports. "
            f"Split this population across more than one identifier stream."
        )
    raise SchemaError(
        f"{where}: identifier {identifier.tag!r} uses {identifier.digits} digits, which "
        f"is too narrow for {population:,} {entity.name!r} entities. Around "
        f"{expected:.0f} of them would share an identifier with someone else. "
        f"Use digits={needed} or more."
    )


def _check_unique(names: list[str], kind: str) -> None:
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise SchemaError(f"duplicate {kind} names: {dupes}")


def _check_not_reserved(name: str, where: str) -> None:
    if name.startswith(RESERVED_PREFIX):
        raise SchemaError(f"{where} {name!r} uses the reserved prefix {RESERVED_PREFIX!r}")
