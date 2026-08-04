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
from synthweave.connectors.faker_names import _FAKER_SUPPORTED, Name, SSN

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


def test_unweighted_provider_attribute_is_rejected_by_name(monkeypatch):
    """A Faker version that turns a weighted dict into a plain list must fail loudly.

    `Provider.first_names` and friends are Faker internals, not public API, so
    their shape can change with no deprecation. Silently falling back to
    unweighted picks would keep the pipeline green while quietly dropping the
    US name frequency weighting the whole module exists to provide.
    """
    from faker.providers.person.en_US import Provider

    monkeypatch.setattr(Provider, "first_names", ["Aaron", "Adam"], raising=True)
    with pytest.raises(RuntimeError, match="first_names"):
        Name("first_name")


def test_missing_provider_attribute_names_the_attribute_and_the_version(monkeypatch):
    """A removed internal must not surface as a bare AttributeError.

    Substitutes the whole provider class, because the attribute is also defined
    on Faker's base person provider: deleting it from `en_US` alone would fall
    back to the base one rather than being absent.
    """

    class ProviderWithoutLastNames:
        first_names = {"Aaron": 1.0}
        first_names_female = {"April": 1.0}
        first_names_male = {"Aaron": 1.0}

    monkeypatch.setattr("faker.providers.person.en_US.Provider", ProviderWithoutLastNames)
    with pytest.raises(RuntimeError) as excinfo:
        Name("last_name")
    message = str(excinfo.value)
    assert "last_names" in message
    # Not the literal range: asserting `"Faker>=20,<41"` here made this test
    # agree with a stale message instead of catching it, since bumping
    # `pyproject.toml` alone left both unchanged. `tests/test_faker_bound_sync.py`
    # owns the version half of this and pins it to `pyproject.toml`.
    assert _FAKER_SUPPORTED in message


def test_empty_provider_mapping_is_rejected(monkeypatch):
    """An empty pool would make every drawn name identical or crash inside `pick`."""
    from faker.providers.person.en_US import Provider

    monkeypatch.setattr(Provider, "last_names", {}, raising=True)
    with pytest.raises(RuntimeError, match="last_names"):
        Name("last_name")


@pytest.mark.parametrize(
    "weight, problem",
    [
        (0.0, "non-positive"),
        (-1.0, "non-positive"),
        ("0.01", "non-numeric"),
        (None, "non-numeric"),
        (True, "non-numeric"),
        (float("inf"), "non-finite"),
        (float("nan"), "non-finite"),
    ],
)
def test_unusable_weight_is_rejected_and_the_message_says_why(monkeypatch, weight, problem):
    """Weights must be finite positive numbers, and the error must say which rule broke.

    `pick` divides by their sum, so a non-finite weight is as unusable as a
    negative one: `_hash.pick` rejects it with a bare `ValueError` that names
    neither Faker nor the attribute, which is the failure mode this guard
    exists to replace. Reporting a string or `None` as a "non-positive weight"
    would be a factually wrong diagnostic in the one code path whose only job
    is to diagnose.
    """
    from faker.providers.person.en_US import Provider

    monkeypatch.setattr(Provider, "first_names", {"Aaron": 1.0, "Adam": weight}, raising=True)
    with pytest.raises(RuntimeError) as excinfo:
        Name("first_name")
    message = str(excinfo.value)
    assert "first_names" in message
    assert problem in message


@pytest.mark.parametrize("weight", [np.float32(0.5), np.int64(2), np.float64(0.25)])
def test_numpy_scalar_weights_are_accepted(monkeypatch, weight):
    """Real numeric types other than `float` must not be mistaken for a shape break.

    The guard's job is to catch Faker changing shape, not to reject a weight
    that is a perfectly usable number `numpy` happens to own.
    """
    from faker.providers.person.en_US import Provider

    monkeypatch.setattr(Provider, "first_names", {"Aaron": 1.0, "Adam": weight}, raising=True)
    values, weights = _pool("first_name")
    assert len(values) == len(weights) == 2
    assert (weights > 0).all()


def test_valid_provider_data_still_builds_a_weighted_pool():
    """The guard must not reject the real Faker data it is meant to protect."""
    values, weights = _pool("first_name")
    assert len(values) == len(weights) > 0
    assert (weights > 0).all()
