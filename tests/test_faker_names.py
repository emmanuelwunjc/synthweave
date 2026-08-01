"""synthweave.connectors.faker_names: deterministic Name and SSN rules.

Faker's own API is call-and-advance, so these are reimplemented on top of
synthweave's own hash-derived draws rather than calling Faker's generators.
Every test here is really testing that reimplementation stays honest: same
guarantees as any other Rule (chunk invariant, deterministic, consistent per
entity), plus the specific value constraints (real name pool, valid SSN
structure).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import invariants
import synthweave as sw
from synthweave.connectors.faker_names import Name, SSN

pytest.importorskip("faker")


@pytest.fixture
def people_with_pii() -> sw.Entity:
    return sw.Entity(
        "person",
        2_000,
        attributes={
            "first_name": Name("first_name"),
            "last_name": Name("last_name"),
            "ssn": SSN(),
        },
        identifiers=["tax_id"],
    )


@pytest.fixture
def schema_with_pii(people_with_pii) -> sw.Schema:
    roster = sw.Table("roster", grain="person", carry="*", identifiers=["tax_id"])
    wages = sw.Table("wages", grain="person", carry=["first_name", "ssn"], identifiers=["tax_id"])
    return sw.Schema(entities=[people_with_pii], tables=[roster, wages], seed=7)


def test_names_are_chunk_invariant(schema_with_pii):
    invariants.assert_chunk_invariant(schema_with_pii)


def test_names_are_deterministic(schema_with_pii):
    invariants.assert_deterministic(schema_with_pii)


def test_first_and_last_name_are_drawn_from_real_pools(schema_with_pii):
    result = sw.Pipeline(schema_with_pii).run()["roster"]
    first_values, _ = _pool("first_name")
    last_values, _ = _pool("last_name")
    assert set(result["first_name"]) <= set(first_values)
    assert set(result["last_name"]) <= set(last_values)


def test_name_pool_is_not_a_single_value(schema_with_pii):
    """A real spread, not a degenerate constant-looking pool."""
    result = sw.Pipeline(schema_with_pii).run()["roster"]
    assert result["first_name"].nunique() > 50


def test_male_and_female_pools_differ(schema_with_pii):
    female_values, _ = _pool("first_name_female")
    male_values, _ = _pool("first_name_male")
    assert set(female_values) != set(male_values)


def test_invalid_which_raises():
    with pytest.raises(ValueError, match="which must be one of"):
        Name("nickname")


def test_carried_name_and_ssn_are_consistent_across_tables(schema_with_pii):
    result = sw.Pipeline(schema_with_pii).run()
    invariants.assert_entity_attributes_consistent(
        result, ["roster", "wages"], key="tax_id", attribute="first_name"
    )
    invariants.assert_entity_attributes_consistent(
        result, ["roster", "wages"], key="tax_id", attribute="ssn"
    )


# --- SSN structural validity -------------------------------------------


def test_ssn_matches_the_formatted_pattern(schema_with_pii):
    result = sw.Pipeline(schema_with_pii).run()["roster"]
    assert result["ssn"].str.match(r"^\d{3}-\d{2}-\d{4}$").all()


def test_ssn_area_never_666(schema_with_pii):
    result = sw.Pipeline(schema_with_pii).run()["roster"]
    areas = result["ssn"].str.slice(0, 3).astype(int)
    assert not (areas == 666).any()


def test_ssn_area_within_valid_range(schema_with_pii):
    result = sw.Pipeline(schema_with_pii).run()["roster"]
    areas = result["ssn"].str.slice(0, 3).astype(int)
    assert areas.min() >= 1
    assert areas.max() <= 899


def test_ssn_area_is_not_skewed_by_the_666_exclusion():
    """`area == 666` must be resampled, not collapsed onto 667.

    `np.where(area == 666, 667, area)` reassigns every 666 draw to a fixed
    667 instead of drawing again, so 667 comes up roughly twice as often as
    any other valid area. Over a large draw, 667's share should sit with its
    neighbors, not roughly double them.
    """
    keys = np.arange(200_000).astype(str)
    ssns = SSN().draw(keys, seed=1, salt="ssn")
    areas = pd.Series([int(s[:3]) for s in ssns])
    counts = areas.value_counts()
    neighbor_avg = counts.reindex([663, 664, 665, 670, 671, 672]).mean()
    assert counts[667] < neighbor_avg * 1.5


def test_ssn_group_and_serial_are_never_zero(schema_with_pii):
    result = sw.Pipeline(schema_with_pii).run()["roster"]
    groups = result["ssn"].str.slice(4, 6).astype(int)
    serials = result["ssn"].str.slice(7, 11).astype(int)
    assert (groups >= 1).all()
    assert (serials >= 1).all()


def test_ssn_is_chunk_invariant_and_deterministic(schema_with_pii):
    invariants.assert_chunk_invariant(schema_with_pii)
    invariants.assert_deterministic(schema_with_pii)


def _pool(which: str):
    from synthweave.connectors.faker_names import _name_pool

    return _name_pool(which)
