"""sw.Mode: the mode-based front door, starting with sw.Mode.metadata().

Grouped by the property being proved, not by the module doing the work. Every
test goes through the public API: build a mode, call attribute()/entity()/
table()/schema(), assert on the result or on a pipeline run.
"""

from __future__ import annotations

import pandas as pd
import pytest

import synthweave as sw


# --- construction -------------------------------------------------------


def test_metadata_returns_a_mode_instance():
    m = sw.Mode.metadata()
    assert isinstance(m, sw.Mode)


# --- attribute(): distribution dispatch ----------------------------------


def test_min_max_without_distribution_defaults_to_uniform():
    m = sw.Mode.metadata()
    rule = m.attribute("income", min=20_000, max=90_000)
    assert rule == sw.Uniform(20_000, 90_000)


def test_mean_sd_normal_with_low_bound():
    m = sw.Mode.metadata()
    rule = m.attribute("income", mean=45_000, sd=9_000, distribution="normal", min=0)
    assert rule == sw.Normal(45_000, 9_000, low=0, high=None)


def test_values_and_weights_become_a_choice():
    m = sw.Mode.metadata()
    rule = m.attribute("education", values=["HS", "College"], weights=[0.6, 0.4])
    assert rule == sw.Choice(["HS", "College"], [0.6, 0.4])


def test_normal_missing_sd_raises_naming_it():
    m = sw.Mode.metadata()
    with pytest.raises(ValueError, match="sd"):
        m.attribute("income", distribution="normal", mean=1)


# --- attribute(): noise-kwarg bookkeeping --------------------------------


def test_missing_rate_is_recorded_but_does_not_affect_the_returned_rule():
    m = sw.Mode.metadata()
    rule = m.attribute("income", min=0, max=1, missing_rate=0.1)
    assert rule == sw.Uniform(0, 1)
    assert m._noise_kwargs["income"] == {"missing_rate": 0.1}


# --- entity()/table(): passthrough to the real schema types --------------


def test_entity_returns_a_real_entity():
    m = sw.Mode.metadata()
    person = m.entity("person", count=100, attributes={"income": sw.Uniform(0, 1)})
    assert isinstance(person, sw.Entity)


def test_table_returns_a_real_table():
    m = sw.Mode.metadata()
    table = m.table("roster", grain="person", carry=["income"])
    assert isinstance(table, sw.Table)


# --- schema(): a full run end to end --------------------------------------


def test_schema_run_produces_null_values_near_the_configured_missing_rate():
    m = sw.Mode.metadata()
    m.attribute("income", min=0, max=100, missing_rate=0.3)
    person = m.entity(
        "person", count=20_000, attributes={"income": m.attribute("income", min=0, max=100)}
    )
    table = m.table("records", grain="person", carry=["income"])

    result = m.schema(entities=[person], tables=[table], seed=42).run()

    assert 0.29 < result["records"]["income"].isna().mean() < 0.31


def test_schema_run_matches_hand_built_pipeline_for_the_same_seed():
    person_direct = sw.Entity(
        "person", count=400, attributes={"income": sw.Uniform(0, 100)}
    )
    table_direct = sw.Table("records", grain="person", carry=["income"])
    expected = sw.Pipeline(
        sw.Schema(entities=[person_direct], tables=[table_direct], seed=7),
        noiser=sw.Noise({"records": {"income": [sw.Missing(0.2)]}}),
    ).run()

    m = sw.Mode.metadata()
    rule = m.attribute("income", min=0, max=100, missing_rate=0.2)
    person = m.entity("person", count=400, attributes={"income": rule})
    table = m.table("records", grain="person", carry=["income"])
    actual = m.schema(entities=[person], tables=[table], seed=7).run()

    pd.testing.assert_frame_equal(expected["records"], actual["records"])


def test_schema_run_to_writes_csv_like_a_hand_built_pipeline(tmp_path):
    person = sw.Entity("person", count=50, attributes={"income": sw.Uniform(0, 100)})
    table = sw.Table("records", grain="person", carry=["income"])

    m = sw.Mode.metadata()
    result = m.schema(entities=[person], tables=[table], seed=3).run_to(
        tmp_path, format="csv"
    )

    assert set(result.paths) == {"records"}
    written = pd.read_csv(result.paths["records"])
    assert len(written) == 50
