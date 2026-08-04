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
from synthweave.mode import _cart_knobs
from synthweave.provenance import unwrap


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


def test_normal_missing_mean_raises_naming_it():
    m = sw.Mode.metadata()
    with pytest.raises(ValueError, match="mean"):
        m.attribute("income", distribution="normal", sd=1)


def test_an_unknown_kwarg_is_rejected_rather_than_dropped():
    m = sw.Mode.metadata()
    with pytest.raises(ValueError, match="typo"):
        m.attribute("income", min=0, max=1, typo=0.1)


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
    rule = m.attribute("income", min=0, max=100, missing_rate=0.3)
    person = m.entity("person", count=20_000, attributes={"income": rule})
    table = m.table("records", grain="person", carry=["income"])

    result = m.schema(entities=[person], tables=[table], seed=42).run()

    assert 0.29 < result["records"]["income"].isna().mean() < 0.31


def test_missing_rate_reaches_a_column_carried_by_wildcard():
    m = sw.Mode.metadata()
    rule = m.attribute("income", min=0, max=100, missing_rate=0.5)
    person = m.entity("person", count=20_000, attributes={"income": rule})
    table = m.table("records", grain="person", carry="*")

    result = m.schema(entities=[person], tables=[table], seed=42).run()

    assert 0.49 < result["records"]["income"].isna().mean() < 0.51


def test_missing_rate_reaches_a_column_declared_after_the_table():
    m = sw.Mode.metadata()
    table = m.table("records", grain="person", carry=["income"])
    rule = m.attribute("income", min=0, max=100, missing_rate=0.5)
    person = m.entity("person", count=20_000, attributes={"income": rule})

    result = m.schema(entities=[person], tables=[table], seed=42).run()

    assert 0.49 < result["records"]["income"].isna().mean() < 0.51


def test_a_noise_rate_matching_no_column_anywhere_raises():
    m = sw.Mode.metadata()
    rule = m.attribute("incme", min=0, max=100, missing_rate=0.3)
    person = m.entity("person", count=10, attributes={"income": rule})
    table = m.table("records", grain="person", carry=["income"])

    with pytest.raises(ValueError, match="would never be applied"):
        m.schema(entities=[person], tables=[table], seed=1)


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


# --- sw.Mode.real_data(): the epsilon -> CART knob mapping ----------------
#
# `Mode.real_data`'s docstring points callers at `_cart_knobs` as the source
# of truth for the mapping, so the numbers it names have to be the numbers it
# produces. These pin both halves so neither can drift alone.


def _knob_values(epsilon: float) -> dict:
    """The knobs with their provenance tags stripped.

    `min_samples_leaf` comes back `Tagged` so the epsilon a user set is not
    recorded as a library default (#145), and these two tests are about the
    numbers rather than about the tag.
    """
    return {key: unwrap(value) for key, value in _cart_knobs(epsilon).items()}


def test_cart_knobs_at_the_epsilon_ceiling_are_pinned():
    assert _knob_values(5.0) == {
        "max_depth": None,
        "min_samples_leaf": 20,
        "fit_cap": 200_000,
    }


def test_epsilon_past_the_ceiling_produces_the_ceiling_knobs():
    assert _knob_values(10.0) == _knob_values(5.0)


def test_cart_knobs_docstring_names_the_values_it_actually_produces():
    doc = _cart_knobs.__doc__
    for fragment in ("max_depth=None", "min_samples_leaf=20", "fit_cap=DEFAULT_FIT_CAP"):
        assert fragment in doc


# --- sw.Mode.real_data(): attribute()'s epsilon dispatch -------------------


def test_attribute_without_epsilon_uses_the_mode_level_default(donor_frame):
    m = sw.Mode.real_data(source=donor_frame, epsilon=0.75)
    m.attribute("wage")
    assert m._real_data_epsilon["wage"] == 0.75


def test_attribute_epsilon_overrides_the_mode_level_default(donor_frame):
    m = sw.Mode.real_data(source=donor_frame, epsilon=0.75)
    m.attribute("wage", epsilon=0.5)
    assert m._real_data_epsilon["wage"] == 0.5


# --- sw.Mode.real_data(): rejecting a caller error at the caller's line ----


@pytest.mark.parametrize("bad", [0, -1, -0.5])
def test_real_data_rejects_a_non_positive_epsilon(donor_frame, bad):
    with pytest.raises(ValueError, match=re.escape(repr(bad))):
        sw.Mode.real_data(source=donor_frame, epsilon=bad)


@pytest.mark.parametrize("bad", [0, -1])
def test_attribute_rejects_a_non_positive_epsilon_naming_the_attribute(donor_frame, bad):
    m = sw.Mode.real_data(source=donor_frame)
    with pytest.raises(ValueError, match="wage"):
        m.attribute("wage", epsilon=bad)


def test_attribute_rejects_an_unknown_kwarg_naming_the_attribute(donor_frame):
    m = sw.Mode.real_data(source=donor_frame)
    with pytest.raises(ValueError, match="wage"):
        m.attribute("wage", min=0, max=1)


# --- sw.Mode.real_data(): one CARTSynthesizer per distinct epsilon ---------


def test_two_attributes_at_the_same_epsilon_share_one_synthesizer(donor_frame):
    m = sw.Mode.real_data(source=donor_frame, epsilon=1.0)
    m.attribute("wage")
    m.attribute("age")
    synthesizer = m._extra_pipeline_kwargs({"roster": ["wage", "age"]})["synthesizer"]
    assert isinstance(synthesizer, sw.CARTSynthesizer)
    assert set(synthesizer.columns) == {"wage", "age"}


def test_two_attributes_at_different_epsilons_get_two_synthesizers(donor_frame):
    m = sw.Mode.real_data(source=donor_frame, epsilon=1.0)
    m.attribute("wage", epsilon=0.5)
    m.attribute("age", epsilon=2.0)
    synthesizer = m._extra_pipeline_kwargs({"roster": ["wage", "age"]})["synthesizer"]
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


@pytest.mark.parametrize("bad", [0, -1])
def test_scope_rejects_a_non_positive_epsilon(bad):
    """Scope reaches the same clamp real_data does, so it needs the same guard.

    `_cart_knobs` turns 0 or -1 into 0.01 and hands back the most generalized
    column the mapping can produce. That is a plausible-looking result for
    what is really a typo, and it is worse here than in real_data mode: the
    donor rows are real Census respondents, so a silently mangled epsilon is
    a silently wrong disclosure posture.
    """
    with pytest.raises(ValueError, match="positive"):
        sw.Mode.scope(area_code="NY", epsilon=bad)


@pytest.mark.parametrize("bad", [0, -1])
def test_scope_rejects_a_non_positive_per_attribute_epsilon(bad):
    """The per-column override reaches the same clamp, so it gets the same guard."""
    m = sw.Mode.scope(area_code="NY")
    with pytest.raises(ValueError, match="positive"):
        m.attribute("wage", variable="PINCP", epsilon=bad)


def test_scope_rejects_an_unknown_kwarg_naming_the_attribute():
    """docs/GUIDE.md Part 4 promises this in every mode, not two of three.

    A narrow keyword-only signature answered `m.attribute("wage", min=0)`
    with `TypeError: ScopeMode._build_rule() got an unexpected keyword
    argument 'min'`, which names a private method instead of the attribute
    the caller wrote. A misspelled kwarg is the case that matters: it has to
    fail loudly and say which attribute is at fault.
    """
    m = sw.Mode.scope(area_code="NY")
    with pytest.raises(ValueError, match="wage"):
        m.attribute("wage", variable="PINCP", min=0, max=1)


def test_scope_rejects_a_misspelled_noise_rate_rather_than_dropping_it():
    """`missing_rate=` is real in every mode, so `missng_rate=` is a typo and
    not a keyword this mode should quietly swallow."""
    m = sw.Mode.scope(area_code="NY")
    with pytest.raises(ValueError, match="missng_rate"):
        m.attribute("wage", variable="PINCP", missng_rate=0.1)


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


def test_scope_epsilon_sets_the_synthesizers_max_depth_knob():
    """Named for the knob, because the knob is all it checks.

    It was called `test_scope_generalizes_the_fetched_rows_by_epsilon`, and in
    this exact scenario -- one attribute, so one column with nothing to
    condition on -- `max_depth` changes nothing about the rows that come out
    (#143, decision in #163). The data-level counterpart is
    `test_epsilon_changes_the_synthesized_data_not_only_the_knob_values`.
    """
    m = sw.Mode.scope(area_code="NY", epsilon=0.5)
    m.attribute("wage", variable="PINCP")

    with patch("urllib.request.urlopen", return_value=_mock_acs_response()):
        synthesizer = m._extra_pipeline_kwargs({"roster": ["wage"]})["synthesizer"]

    # epsilon 0.5 asks for a shallow tree, not CART's unbounded default.
    assert synthesizer.max_depth == 2


def test_scope_attribute_epsilon_overrides_the_mode_level_default():
    m = sw.Mode.scope(area_code="NY", epsilon=0.5)
    m.attribute("wage", variable="PINCP", epsilon=1.0)

    with patch("urllib.request.urlopen", return_value=_mock_acs_response()):
        synthesizer = m._extra_pipeline_kwargs({"roster": ["wage"]})["synthesizer"]

    assert synthesizer.max_depth == 4


def test_scope_attributes_at_different_epsilons_get_two_synthesizers():
    m = sw.Mode.scope(area_code="NY")
    m.attribute("wage", variable="PINCP", epsilon=0.5)
    m.attribute("age", variable="AGEP", epsilon=2.0)

    with patch("urllib.request.urlopen", return_value=_mock_acs_response()):
        synthesizer = m._extra_pipeline_kwargs({"roster": ["wage", "age"]})["synthesizer"]

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


# --- sw.Mode.real_data(): mixed epsilons keep the donor's joint structure ---


@pytest.fixture
def linked_donor() -> pd.DataFrame:
    """A donor where wage is determined by education, plus noise.

    The point of the fixture is that the joint structure is the whole signal:
    the two education groups are 50k apart, so any test that draws the columns
    independently lands both group means on the pooled mean and cannot miss it.
    """
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {"education": rng.choice(["HS", "College"], size=800, p=[0.6, 0.4])}
    )
    frame["wage"] = np.where(frame["education"] == "College", 80_000.0, 30_000.0) + (
        rng.normal(0, 2_000, 800)
    )
    return frame


def _wage_by_education(frame: pd.DataFrame) -> dict[str, float]:
    return {k: float(v) for k, v in frame.groupby("education")["wage"].mean().items()}


def test_two_attributes_at_different_epsilons_keep_the_donors_joint_structure(
    linked_donor,
):
    """#139: mixed epsilons used to draw each group independently.

    Asserts the data, not the object graph: college graduates must still out-earn
    high school graduates by roughly the donor's own gap. Decorrelated output
    collapses both group means onto the pooled mean (~50k each) while every
    univariate summary stays correct, which is why the structural test that
    shipped alongside this could not see it.
    """
    m = sw.Mode.real_data(source=linked_donor, epsilon=1.0)
    education = m.attribute("education")
    wage = m.attribute("wage", epsilon=2.0)
    person = m.entity(
        "person", count=2_000, attributes={"education": education, "wage": wage}
    )
    table = m.table("roster", grain="person", carry=["education", "wage"])

    out = m.schema(entities=[person], tables=[table], seed=11).run()["roster"]

    donor = _wage_by_education(linked_donor)
    got = _wage_by_education(out)
    assert got["College"] == pytest.approx(donor["College"], rel=0.05), (
        f"mixed epsilons lost the education/wage relationship: {got} vs donor {donor}"
    )
    assert got["HS"] == pytest.approx(donor["HS"], rel=0.05), (
        f"mixed epsilons lost the education/wage relationship: {got} vs donor {donor}"
    )


# --- sw.Mode.real_data(): an attribute lands only where it was declared ----


def test_a_real_data_column_stays_out_of_a_table_that_did_not_declare_it(donor_frame):
    """#144a: the synthesizer used to be unscoped and `apply` creates columns.

    Asserts the output frames, not the synthesizer's `tables` set: a table that
    carries nothing from the donor used to come back holding every mode
    attribute, so a user exporting it shipped real-derived values they never
    asked for.
    """
    m = sw.Mode.real_data(source=donor_frame, epsilon=1.0)
    wage = m.attribute("wage")
    age = m.attribute("age")
    person = m.entity("person", count=50, attributes={"wage": wage, "age": age})
    wages = m.table("wages", grain="person", carry=["wage"])
    ages = m.table("ages", grain="person", carry=["age"])

    result = m.schema(entities=[person], tables=[wages, ages], seed=1).run()

    assert "age" not in result["wages"].columns
    assert "wage" not in result["ages"].columns


def test_a_real_data_attribute_no_table_carries_raises_naming_it(donor_frame):
    """#144a: declared and never carried used to be injected into every table."""
    m = sw.Mode.real_data(source=donor_frame, epsilon=1.0)
    wage = m.attribute("wage")
    age = m.attribute("age")
    person = m.entity("person", count=10, attributes={"wage": wage, "age": age})
    table = m.table("roster", grain="person", carry=["wage"])

    with pytest.raises(ValueError, match="age"):
        m.schema(entities=[person], tables=[table], seed=1).run()


def test_binding_a_real_data_attribute_under_another_name_raises_naming_both(
    donor_frame,
):
    """#144b: the mode keys on the source column, the schema on the bound name.

    The two used to disagree in silence: the user's column came back entirely
    null and a phantom column carrying the real-derived values appeared beside
    it. Asserts the raise, and the message has to name both halves so the fix
    is obvious from the traceback.
    """
    m = sw.Mode.real_data(source=donor_frame, epsilon=1.0)
    person = m.entity("person", count=10, attributes={"salary": m.attribute("wage")})
    table = m.table("roster", grain="person", carry=["salary"])

    with pytest.raises(ValueError, match="salary.*wage|wage.*salary"):
        m.schema(entities=[person], tables=[table], seed=1).run()


# --- sw.Mode.real_data(): the audit trail a chained run leaves ------------


def _two_epsilon_result(donor_frame):
    m = sw.Mode.real_data(source=donor_frame, epsilon=0.5)
    education = m.attribute("education")
    wage = m.attribute("wage", epsilon=4.0)
    person = m.entity(
        "person", count=200, attributes={"education": education, "wage": wage}
    )
    table = m.table("roster", grain="person", carry=["education", "wage"])
    return m.schema(entities=[person], tables=[table], seed=3).run()


def test_a_two_epsilon_run_records_both_generalization_levels(donor_frame):
    """#145: every group wrote the same provenance path, so one survived.

    epsilon 0.5 asks for leaves of 100/0.5 = 200 rows and epsilon 4.0 for
    100/4 = 25, per `_cart_knobs`. Both were applied to the output, so both
    have to appear: a user defending this table needs to see that two
    different generalization levels were used, not one.
    """
    result = _two_epsilon_result(donor_frame)

    leaves = {
        tagged.value
        for path, tagged in result.provenance.entries.items()
        if path.endswith("min_samples_leaf")
    }
    assert leaves == {200, 25}


def test_a_two_epsilon_run_reports_both_fits_without_colliding(donor_frame):
    """#145: `ctx.report` keyed on the table and the stage name alone."""
    result = _two_epsilon_result(donor_frame)

    reports = [
        facts
        for stage, facts in result.metadata["roster"].items()
        if stage.startswith("synthesize")
    ]
    assert len(reports) == 2
    assert {tuple(facts["columns"]) for facts in reports} == {("education",), ("wage",)}


def test_an_epsilon_derived_leaf_size_is_not_reported_as_a_library_default(donor_frame):
    """#145: the one number with a stated justification was the one flagged.

    `CARTSynthesizer` tags any plain int leaf size as a library default, and
    the epsilon-derived one arrived as a plain int, so `unjustified()` named
    the value the user chose by setting `epsilon=`.
    """
    result = _two_epsilon_result(donor_frame)

    assert not [path for path in result.unjustified() if "min_samples_leaf" in path]


# --- sw.Mode.real_data(): epsilon reaches the data, not just the knobs ----


@pytest.fixture
def stepped_donor() -> pd.DataFrame:
    """Four regions whose wages are 20k apart, plus noise.

    A conditioned column drawn under a tight epsilon cannot keep the steps: the
    leaf size epsilon 0.01 asks for is larger than the whole fit sample, so the
    tree has one leaf and wage stops depending on region. Under a loose epsilon
    it splits per region and the steps survive. That difference is visible in
    the output frame, which is the point of this fixture.
    """
    rng = np.random.default_rng(4)
    region = rng.choice(["r0", "r1", "r2", "r3"], size=2_000)
    frame = pd.DataFrame({"region": region})
    steps = {"r0": 20_000.0, "r1": 40_000.0, "r2": 60_000.0, "r3": 80_000.0}
    frame["wage"] = frame["region"].map(steps) + rng.normal(0, 1_000, 2_000)
    return frame


def _wage_spread_by_region(frame: pd.DataFrame) -> float:
    means = frame.groupby("region")["wage"].mean()
    return float(means.max() - means.min())


def test_epsilon_changes_the_synthesized_data_not_only_the_knob_values(stepped_donor):
    """#143: nothing in the suite checked that epsilon reaches the output.

    Stubbing `_cart_knobs` to constants used to fail exactly four tests, and
    every one of them asserted `synthesizer.max_depth` or the knob dict. This
    one asserts the frame: a tight epsilon flattens the wage-by-region steps a
    loose one keeps.
    """

    def run(epsilon: float) -> pd.DataFrame:
        m = sw.Mode.real_data(source=stepped_donor, epsilon=epsilon)
        region = m.attribute("region")
        wage = m.attribute("wage")
        person = m.entity(
            "person", count=2_000, attributes={"region": region, "wage": wage}
        )
        table = m.table("roster", grain="person", carry=["region", "wage"])
        return m.schema(entities=[person], tables=[table], seed=19).run()["roster"]

    donor_spread = _wage_spread_by_region(stepped_donor)
    loose = _wage_spread_by_region(run(5.0))
    tight = _wage_spread_by_region(run(0.01))

    assert loose > 0.9 * donor_spread, (
        f"epsilon 5.0 should keep the donor's wage-by-region steps: "
        f"spread {loose:.0f} vs donor {donor_spread:.0f}"
    )
    assert tight < 0.2 * donor_spread, (
        f"epsilon 0.01 should generalize the steps away: "
        f"spread {tight:.0f} vs donor {donor_spread:.0f}"
    )
