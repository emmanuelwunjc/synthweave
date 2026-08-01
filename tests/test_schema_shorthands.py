"""Reduced-input shorthands: coerce_rule, string grain/identifiers, carry="*".

Every shorthand is additive. Each test proves the shorthand form produces the
exact same result as writing the equivalent explicit form out by hand.
"""

from __future__ import annotations

import pytest

import synthweave as sw
from synthweave.rules import coerce_rule


# --- coerce_rule --------------------------------------------------------


def test_a_rule_instance_passes_through_unchanged():
    rule = sw.Integer(1, 10)
    assert coerce_rule(rule) is rule


def test_a_list_becomes_an_equal_weight_choice():
    assert coerce_rule(["HS", "College"]) == sw.Choice(["HS", "College"])


def test_a_tuple_also_becomes_a_choice():
    assert coerce_rule(("HS", "College")) == sw.Choice(["HS", "College"])


@pytest.mark.parametrize("value", [1, 1.5, "HS", True])
def test_a_scalar_becomes_a_constant(value):
    assert coerce_rule(value) == sw.Constant(value)


def test_an_unsupported_type_raises_naming_the_alternatives():
    with pytest.raises(TypeError, match="sw.Integer"):
        coerce_rule({"not": "a rule"})


def test_conditional_case_values_are_coerced_too():
    """The same plain-value shorthand works inside a Conditional's branches."""
    rule = sw.Conditional("education", {"HS": "no credential", "College": "BA"})
    assert rule.cases == {"HS": sw.Constant("no credential"), "College": sw.Constant("BA")}


def test_conditional_default_is_coerced_too():
    rule = sw.Conditional("education", {"HS": "no credential"}, default="unknown")
    assert rule.default == sw.Constant("unknown")


# --- Entity/Table shorthands ---------------------------------------------


def test_entity_accepts_plain_attribute_values():
    entity = sw.Entity("person", 10, attributes={"education": ["HS", "College"], "active": True})
    assert entity.attributes == {
        "education": sw.Choice(["HS", "College"]),
        "active": sw.Constant(True),
    }


def test_entity_accepts_bare_identifier_tag_strings():
    entity = sw.Entity("person", 10, identifiers=["tax_id"])
    assert entity.identifiers == [sw.Identifier("tax_id")]


def test_entity_identifier_shorthand_still_catches_duplicate_tags():
    with pytest.raises(ValueError, match="duplicate"):
        sw.Entity("person", 10, identifiers=["tax_id", "tax_id"])


def test_table_accepts_a_bare_entity_name_as_grain():
    table = sw.Table("t", grain="person")
    assert table.grain == sw.PerEntity("person")
    assert table.entity == "person"


def test_table_accepts_plain_column_values():
    table = sw.Table("t", grain="person", columns={"wage": 0.0})
    assert table.columns == {"wage": sw.Constant(0.0)}


def test_carry_star_expands_to_every_entity_attribute():
    person = sw.Entity(
        "person", 10, attributes={"education": ["HS", "College"], "birth_year": sw.Integer(1960, 2005)}
    )
    table = sw.Table("t", grain="person", carry="*")
    schema = sw.Schema(entities=[person], tables=[table])
    assert schema.table("t").carry == ("education", "birth_year")


def test_carry_star_matches_naming_every_attribute_by_hand():
    person = sw.Entity("person", 500, attributes={"education": ["HS", "College"]})
    explicit = sw.Table("explicit", grain="person", carry=["education"])
    starred = sw.Table("starred", grain="person", carry="*")
    schema = sw.Schema(entities=[person], tables=[explicit, starred], seed=3)
    result = sw.Pipeline(schema).run()
    assert result["explicit"]["education"].tolist() == result["starred"]["education"].tolist()


# --- full pipeline: shorthand form matches the explicit form exactly ------


def test_shorthand_schema_matches_the_explicit_equivalent():
    explicit_person = sw.Entity(
        "person",
        1_000,
        attributes={"education": sw.Choice(["HS", "College"])},
        identifiers=[sw.Identifier("tax_id")],
    )
    explicit_table = sw.Table(
        "t",
        grain=sw.PerEntity("person"),
        carry=["education"],
        columns={"wage": sw.Constant(0.0)},
    )
    explicit_result = sw.Pipeline(
        sw.Schema(entities=[explicit_person], tables=[explicit_table], seed=42)
    ).run()["t"]

    shorthand_person = sw.Entity(
        "person", 1_000, attributes={"education": ["HS", "College"]}, identifiers=["tax_id"]
    )
    shorthand_table = sw.Table("t", grain="person", carry="*", columns={"wage": 0.0})
    shorthand_result = sw.Pipeline(
        sw.Schema(entities=[shorthand_person], tables=[shorthand_table], seed=42)
    ).run()["t"]

    assert explicit_result.equals(shorthand_result)


def test_carry_star_resolves_per_schema_not_once_per_table():
    """A `Table` reused across schemas must resolve `carry="*"` each time.

    Resolution used to write the expanded tuple back into the shared `Table`,
    so the second schema saw `carry` already resolved, skipped it, and
    silently inherited the first schema's attribute set. No exception, no
    warning: the second schema just quietly lost a column.
    """
    table = sw.Table("t", grain="person", carry="*")

    first = sw.Entity("person", 10, attributes={"education": ["HS", "College"]})
    schema_one = sw.Schema(entities=[first], tables=[table])

    second = sw.Entity(
        "person",
        10,
        attributes={"birth_year": sw.Integer(1990, 2000), "education": ["HS", "College"]},
    )
    schema_two = sw.Schema(entities=[second], tables=[table])

    assert schema_one.table("t").carry == ("education",)
    assert schema_two.table("t").carry == ("birth_year", "education")
    # The shared object itself must be left alone, or the bug simply moves.
    assert table.carry == "*"
