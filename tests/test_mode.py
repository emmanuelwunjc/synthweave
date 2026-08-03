"""sw.Mode: the mode-based front door, starting with sw.Mode.metadata().

Grouped by the property being proved, not by the module doing the work. Every
test goes through the public API: build a mode, call attribute()/entity()/
table()/schema(), assert on the result or on a pipeline run.
"""

from __future__ import annotations

import json
import re
from unittest.mock import MagicMock, patch

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


# --- sw.Mode.scope(): wraps the ACS PUMS connector -------------------------

_ACS_PAYLOAD = [["AGEP", "PINCP", "state"]] + [
    [str(20 + i % 50), str(20_000 + i * 137), "36"] for i in range(200)
]


def _mock_acs_response(status: int = 200):
    response = MagicMock()
    response.status = status
    response.read.return_value = json.dumps(_ACS_PAYLOAD).encode()
    response.__enter__.return_value = response
    return response


@pytest.fixture(autouse=True)
def _acs_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CENSUS_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize("area_code", ["NY", "New York", "36"])
def test_scope_constructs_from_any_form_resolve_state_accepts(area_code):
    m = sw.Mode.scope(area_code=area_code)
    assert isinstance(m, sw.Mode)


def test_attribute_without_variable_raises_naming_it():
    m = sw.Mode.scope(area_code="NY")
    with pytest.raises(ValueError, match="wage"):
        m.attribute("wage")


def test_attribute_with_variable_registers_it():
    m = sw.Mode.scope(area_code="NY")
    m.attribute("wage", variable="PINCP")
    assert m._variables["wage"] == "PINCP"


def test_two_attributes_can_share_one_acs_variable():
    m = sw.Mode.scope(area_code="NY")
    wage = m.attribute("wage", variable="PINCP")
    earnings = m.attribute("earnings", variable="PINCP")
    person = m.entity(
        "person", count=50, attributes={"wage": wage, "earnings": earnings}
    )
    table = m.table("roster", grain="person", carry=["wage", "earnings"])

    with patch("urllib.request.urlopen", return_value=_mock_acs_response()):
        result = m.schema(entities=[person], tables=[table], seed=5).run()

    donor_wages = {20_000 + i * 137 for i in range(200)}
    assert set(result["roster"]["wage"]) <= donor_wages
    assert set(result["roster"]["earnings"]) <= donor_wages


def test_a_shared_acs_variable_is_requested_once():
    m = sw.Mode.scope(area_code="NY")
    wage = m.attribute("wage", variable="PINCP")
    earnings = m.attribute("earnings", variable="PINCP")
    person = m.entity(
        "person", count=10, attributes={"wage": wage, "earnings": earnings}
    )
    table = m.table("roster", grain="person", carry=["wage", "earnings"])

    with patch("urllib.request.urlopen", return_value=_mock_acs_response()) as urlopen:
        m.schema(entities=[person], tables=[table], seed=5).run()

    requested = urlopen.call_args.args[0]
    assert requested.count("PINCP") == 1


def test_scope_generalizes_the_fetched_rows_by_epsilon():
    m = sw.Mode.scope(area_code="NY", epsilon=0.5)
    m.attribute("wage", variable="PINCP")

    with patch("urllib.request.urlopen", return_value=_mock_acs_response()):
        synthesizer = m._extra_pipeline_kwargs()["synthesizer"]

    # epsilon 0.5 asks for a shallow tree, not CART's unbounded default.
    assert synthesizer.max_depth == 2


def test_scope_attribute_epsilon_overrides_the_mode_level_default():
    m = sw.Mode.scope(area_code="NY", epsilon=0.5)
    m.attribute("wage", variable="PINCP", epsilon=1.0)

    with patch("urllib.request.urlopen", return_value=_mock_acs_response()):
        synthesizer = m._extra_pipeline_kwargs()["synthesizer"]

    assert synthesizer.max_depth == 4


def test_scope_attributes_at_different_epsilons_get_two_synthesizers():
    m = sw.Mode.scope(area_code="NY")
    m.attribute("wage", variable="PINCP", epsilon=0.5)
    m.attribute("age", variable="AGEP", epsilon=2.0)

    with patch("urllib.request.urlopen", return_value=_mock_acs_response()):
        synthesizer = m._extra_pipeline_kwargs()["synthesizer"]

    assert not isinstance(synthesizer, sw.CARTSynthesizer)
    assert {tuple(s.columns) for s in synthesizer.synthesizers} == {("wage",), ("age",)}
    assert {s.max_depth for s in synthesizer.synthesizers} == {2, 8}


def test_scopes_docstring_first_sentence_states_state_level_only():
    first_sentence = sw.Mode.scope.__doc__.strip().split(".")[0]
    assert "state-level ACS geography only" in first_sentence


def test_scopes_docstring_disclaims_differential_privacy():
    assert "differential privacy" in sw.Mode.scope.__doc__.lower()


def test_schema_run_calls_fetch_pums_once_for_every_attribute_combined():
    m = sw.Mode.scope(area_code="NY")
    age = m.attribute("age", variable="AGEP")
    wage = m.attribute("wage", variable="PINCP")
    person = m.entity("person", count=50, attributes={"age": age, "wage": wage})
    table = m.table("roster", grain="person", carry=["age", "wage"])

    with patch("urllib.request.urlopen", return_value=_mock_acs_response()) as urlopen:
        m.schema(entities=[person], tables=[table], seed=5).run()

    assert urlopen.call_count == 1


def test_a_second_run_on_the_same_schema_does_not_refetch():
    m = sw.Mode.scope(area_code="NY")
    wage = m.attribute("wage", variable="PINCP")
    person = m.entity("person", count=50, attributes={"wage": wage})
    table = m.table("roster", grain="person", carry=["wage"])

    with patch("urllib.request.urlopen", return_value=_mock_acs_response()) as urlopen:
        schema = m.schema(entities=[person], tables=[table], seed=5)
        schema.run()
        first_count = urlopen.call_count
        schema.run()
        assert urlopen.call_count == first_count


def test_schema_defers_the_fetch_until_run():
    m = sw.Mode.scope(area_code="NY")
    wage = m.attribute("wage", variable="PINCP")
    person = m.entity("person", count=10, attributes={"wage": wage})
    table = m.table("roster", grain="person", carry=["wage"])

    with patch("urllib.request.urlopen", return_value=_mock_acs_response()) as urlopen:
        schema = m.schema(entities=[person], tables=[table], seed=5)
        assert urlopen.call_count == 0
        schema.run()
        assert urlopen.call_count == 1


def test_an_unrecognized_area_code_raises_on_run_not_on_schema():
    m = sw.Mode.scope(area_code="Nowhere")
    wage = m.attribute("wage", variable="PINCP")
    person = m.entity("person", count=10, attributes={"wage": wage})
    table = m.table("roster", grain="person", carry=["wage"])

    schema = m.schema(entities=[person], tables=[table], seed=5)
    with pytest.raises(ValueError, match="Nowhere"):
        schema.run()


def test_schema_run_synthesizes_from_real_acs_rows():
    m = sw.Mode.scope(area_code="NY")
    wage = m.attribute("wage", variable="PINCP")
    person = m.entity("person", count=50, attributes={"wage": wage})
    table = m.table("roster", grain="person", carry=["wage"])

    with patch("urllib.request.urlopen", return_value=_mock_acs_response()):
        result = m.schema(entities=[person], tables=[table], seed=5).run()

    donor_wages = {20_000 + i * 137 for i in range(200)}
    assert set(result["roster"]["wage"]) <= donor_wages


def test_fetch_pums_failure_propagates_unchanged():
    m = sw.Mode.scope(area_code="NY")
    wage = m.attribute("wage", variable="PINCP")
    person = m.entity("person", count=10, attributes={"wage": wage})
    table = m.table("roster", grain="person", carry=["wage"])

    with patch("urllib.request.urlopen", return_value=_mock_acs_response(status=500)):
        with pytest.raises(RuntimeError, match="HTTP 500"):
            m.schema(entities=[person], tables=[table], seed=5).run()
