"""sw.Mode: the mode-based front door, starting with sw.Mode.metadata().

Grouped by the property being proved, not by the module doing the work. Every
test goes through the public API: build a mode, call attribute()/entity()/
table()/schema(), assert on the result or on a pipeline run.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

import invariants
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


# --- sw.Mode.real_data(): loading the source -----------------------------


@pytest.fixture
def donor_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "education": rng.choice(["HS", "College"], size=2_000, p=[0.6, 0.4]),
            "wage": rng.uniform(20_000, 90_000, size=2_000),
            "age": rng.integers(18, 70, size=2_000),
        }
    )


def test_real_data_returns_a_mode_instance_from_a_dataframe(donor_frame):
    m = sw.Mode.real_data(source=donor_frame, epsilon=1.0)
    assert isinstance(m, sw.Mode)


def test_real_data_loads_a_csv_file(donor_frame, tmp_path):
    path = tmp_path / "donor.csv"
    donor_frame.to_csv(path, index=False)
    m = sw.Mode.real_data(source=str(path))
    assert isinstance(m, sw.Mode)


def test_real_data_loads_a_parquet_file(donor_frame, tmp_path):
    path = tmp_path / "donor.parquet"
    donor_frame.to_parquet(path)
    m = sw.Mode.real_data(source=str(path))
    assert isinstance(m, sw.Mode)


def test_real_data_rejects_an_unsupported_extension(tmp_path):
    path = tmp_path / "donor.txt"
    path.write_text("not real data")
    with pytest.raises(ValueError, match=re.escape(str(path))):
        sw.Mode.real_data(source=str(path))


def test_real_datas_docstring_disclaims_differential_privacy():
    assert "differential privacy" in sw.Mode.real_data.__doc__.lower()


# --- sw.Mode.real_data(): attribute()'s epsilon dispatch -------------------


def test_attribute_without_epsilon_uses_the_mode_level_default(donor_frame):
    m = sw.Mode.real_data(source=donor_frame, epsilon=0.75)
    m.attribute("wage")
    assert m._real_data_epsilon["wage"] == 0.75


def test_attribute_epsilon_overrides_the_mode_level_default(donor_frame):
    m = sw.Mode.real_data(source=donor_frame, epsilon=0.75)
    m.attribute("wage", epsilon=0.5)
    assert m._real_data_epsilon["wage"] == 0.5


# --- sw.Mode.real_data(): one CARTSynthesizer per distinct epsilon ---------


def test_two_attributes_at_the_same_epsilon_share_one_synthesizer(donor_frame):
    m = sw.Mode.real_data(source=donor_frame, epsilon=1.0)
    m.attribute("wage")
    m.attribute("age")
    synthesizer = m._extra_pipeline_kwargs()["synthesizer"]
    assert isinstance(synthesizer, sw.CARTSynthesizer)
    assert set(synthesizer.columns) == {"wage", "age"}


def test_two_attributes_at_different_epsilons_get_two_synthesizers(donor_frame):
    m = sw.Mode.real_data(source=donor_frame, epsilon=1.0)
    m.attribute("wage", epsilon=0.5)
    m.attribute("age", epsilon=2.0)
    synthesizer = m._extra_pipeline_kwargs()["synthesizer"]
    assert not isinstance(synthesizer, sw.CARTSynthesizer)
    assert len(synthesizer.synthesizers) == 2
    assert {tuple(s.columns) for s in synthesizer.synthesizers} == {("wage",), ("age",)}


# --- sw.Mode.real_data(): a full run actually synthesizes from the donor ---


def test_schema_run_synthesizes_real_data_columns_from_the_donor_pool(donor_frame):
    m = sw.Mode.real_data(source=donor_frame, epsilon=1.0)
    education = m.attribute("education")
    wage = m.attribute("wage")
    person = m.entity(
        "person", count=500, attributes={"education": education, "wage": wage}
    )
    table = m.table("roster", grain="person", carry=["education", "wage"])

    result = m.schema(entities=[person], tables=[table], seed=11).run()

    invariants.assert_values_come_from(result["roster"], "education", donor_frame["education"])
    invariants.assert_values_come_from(result["roster"], "wage", donor_frame["wage"])
