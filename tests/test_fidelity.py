"""fidelity_report: real-vs-synthesized comparison for the Empirical/CART path.

Marginal and association checks are proved against plain frames, no pipeline
needed. `empty_donor_leaves` wiring is proved two ways: a stub object (same
convention `test_synthesis_and_plugins.py` uses for plugin behavior) proves
`fidelity_report`'s own merge/filter logic, and a real pipeline run proves
`CARTSynthesizer.donor_diagnostics()` is actually reachable end to end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import synthweave as sw
from synthweave.fidelity import fidelity_report


# --- input validation --------------------------------------------------


def test_columns_must_be_explicit():
    frame = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="at least one column"):
        fidelity_report(frame, frame, columns=[])


def test_missing_column_raises_key_error():
    real = pd.DataFrame({"x": [1, 2, 3]})
    synth = pd.DataFrame({"y": [1, 2, 3]})
    with pytest.raises(KeyError):
        fidelity_report(synth, real, columns=["x"])


# --- numeric marginals (KS) ---------------------------------------------


def test_identical_numeric_distribution_scores_as_a_perfect_match():
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, 500)
    real = pd.DataFrame({"amount": values})
    synth = pd.DataFrame({"amount": values.copy()})

    report = fidelity_report(synth, real, columns=["amount"])
    row = report.to_frame().iloc[0]

    assert row["kind"] == "numeric"
    assert row["value"] == pytest.approx(0.0)
    assert row["pvalue"] == pytest.approx(1.0)


def test_shifted_numeric_distribution_is_flagged():
    rng = np.random.default_rng(1)
    real = pd.DataFrame({"amount": rng.normal(0, 1, 500)})
    synth = pd.DataFrame({"amount": rng.normal(6, 1, 500)})

    report = fidelity_report(synth, real, columns=["amount"])
    row = report.to_frame().iloc[0]

    assert row["value"] > 0.8
    assert row["pvalue"] < 0.01


# --- categorical marginals (share delta) --------------------------------


def test_matching_category_shares_score_zero_delta():
    real = pd.DataFrame({"education": ["HS"] * 60 + ["College"] * 40})
    synth = pd.DataFrame({"education": ["HS"] * 60 + ["College"] * 40})

    report = fidelity_report(synth, real, columns=["education"])
    row = report.to_frame().iloc[0]

    assert row["kind"] == "categorical"
    assert row["value"] == pytest.approx(0.0)


def test_differing_category_shares_score_positive_delta():
    real = pd.DataFrame({"education": ["HS"] * 80 + ["College"] * 20})
    synth = pd.DataFrame({"education": ["HS"] * 20 + ["College"] * 80})

    report = fidelity_report(synth, real, columns=["education"])
    row = report.to_frame().iloc[0]

    # Total variation distance between {HS: .8, College: .2} and
    # {HS: .2, College: .8} is 0.5 * (.6 + .6) = 0.6.
    assert row["value"] == pytest.approx(0.6)


def test_category_only_in_one_frame_is_treated_as_zero_share():
    real = pd.DataFrame({"sector": ["retail"] * 10})
    synth = pd.DataFrame({"sector": ["tech"] * 10})

    report = fidelity_report(synth, real, columns=["sector"])
    row = report.to_frame().iloc[0]

    # No overlap at all: shares are {retail: 1} vs {tech: 1}, TVD = 1.
    assert row["value"] == pytest.approx(1.0)


# --- thresholds: descriptive by default, opt-in verdicts ----------------


def test_no_thresholds_leaves_every_verdict_unset():
    real = pd.DataFrame({"amount": [1.0, 2.0, 3.0], "sector": ["a", "b", "a"]})
    synth = pd.DataFrame({"amount": [1.0, 2.0, 3.0], "sector": ["a", "b", "a"]})

    report = fidelity_report(synth, real, columns=["amount", "sector"])

    assert report.thresholds is None
    assert report.to_frame()["passed"].isna().all()
    assert report.associations.empty or report.associations["passed"].isna().all()


def test_thresholds_populate_boolean_verdicts():
    rng = np.random.default_rng(2)
    values = rng.normal(0, 1, 300)
    real = pd.DataFrame({"amount": values, "sector": ["a"] * 150 + ["b"] * 150})
    synth = pd.DataFrame({"amount": values.copy(), "sector": ["a"] * 150 + ["b"] * 150})

    report = fidelity_report(
        synth,
        real,
        columns=["amount", "sector"],
        thresholds={"ks_pvalue": 0.05, "category_share_delta": 0.1},
    )
    frame = report.to_frame()

    assert set(frame["passed"]) <= {True, False}
    assert frame["passed"].notna().all()
    assert bool(frame.loc[frame["column"] == "amount", "passed"].iloc[0]) is True
    assert bool(frame.loc[frame["column"] == "sector", "passed"].iloc[0]) is True


def test_threshold_only_gates_the_checks_it_names():
    """Supplying `ks_pvalue` alone leaves categorical rows unjudged."""
    real = pd.DataFrame({"amount": [1.0, 2.0], "sector": ["a", "b"]})
    synth = pd.DataFrame({"amount": [1.0, 2.0], "sector": ["a", "b"]})

    report = fidelity_report(
        synth, real, columns=["amount", "sector"], thresholds={"ks_pvalue": 0.05}
    )
    frame = report.to_frame()

    assert frame.loc[frame["column"] == "sector", "passed"].iloc[0] is None


# --- associations ---------------------------------------------------------


def test_pearson_association_for_numeric_pairs_preserved():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 400)
    y = x * 2 + rng.normal(0, 0.1, 400)
    real = pd.DataFrame({"x": x, "y": y})
    synth = pd.DataFrame({"x": x.copy(), "y": y.copy()})

    report = fidelity_report(synth, real, columns=["x", "y"])
    row = report.associations.iloc[0]

    assert row["metric"] == "pearson"
    assert row["real"] == pytest.approx(row["synth"], abs=1e-9)
    assert row["delta"] == pytest.approx(0.0, abs=1e-9)


def test_pearson_association_flags_a_destroyed_relationship():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, 400)
    y = x * 2 + rng.normal(0, 0.1, 400)
    real = pd.DataFrame({"x": x, "y": y})
    synth = pd.DataFrame({"x": x.copy(), "y": rng.normal(0, 1, 400)})

    report = fidelity_report(synth, real, columns=["x", "y"])
    row = report.associations.iloc[0]

    assert abs(row["real"]) > 0.9
    assert row["delta"] > 0.5


def test_cramers_v_association_for_categorical_pairs():
    rng = np.random.default_rng(5)
    education = rng.choice(["HS", "College"], size=400)
    # sector strongly tracks education in the real data...
    sector = np.where(
        education == "HS",
        rng.choice(["retail", "trades"], size=400, p=[0.8, 0.2]),
        rng.choice(["tech", "health"], size=400, p=[0.8, 0.2]),
    )
    real = pd.DataFrame({"education": education, "sector": sector})
    # ...and is independent of it in the synth data.
    synth = pd.DataFrame(
        {"education": education, "sector": rng.choice(["retail", "trades", "tech", "health"], size=400)}
    )

    report = fidelity_report(synth, real, columns=["education", "sector"])
    row = report.associations.iloc[0]

    assert row["metric"] == "cramers_v"
    assert row["real"] > row["synth"]
    assert row["delta"] > 0.2


def test_correlation_ratio_handles_more_than_two_categories():
    """eta, not point-biserial: sector here has 4 levels, not 2."""
    rng = np.random.default_rng(6)
    sector = rng.choice(["retail", "trades", "tech", "health"], size=400)
    base = {"retail": 30_000, "trades": 48_000, "tech": 95_000, "health": 72_000}
    wage = np.array([base[s] for s in sector]) + rng.normal(0, 2_000, 400)
    real = pd.DataFrame({"sector": sector, "wage": wage})
    synth = pd.DataFrame({"sector": sector, "wage": rng.normal(60_000, 20_000, 400)})

    report = fidelity_report(synth, real, columns=["sector", "wage"])
    row = report.associations.iloc[0]

    assert row["metric"] == "correlation_ratio"
    assert row["real"] > 0.9
    assert row["synth"] < row["real"]


def test_association_threshold_gates_pairwise_verdicts():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 300)
    y = x * 2 + rng.normal(0, 0.1, 300)
    real = pd.DataFrame({"x": x, "y": y})
    synth = pd.DataFrame({"x": x.copy(), "y": y.copy()})

    report = fidelity_report(
        synth, real, columns=["x", "y"], thresholds={"association_delta": 0.1}
    )

    assert report.associations.iloc[0]["passed"] == True  # noqa: E712 (np.True_, not bool)


# --- empty donor leaves: stub proves fidelity_report's own merge/filter ---


class _StubSynthesizer:
    """Duck-types CARTSynthesizer's donor_diagnostics() surface, nothing else."""

    def __init__(self, diagnostics):
        self._diagnostics = diagnostics

    def donor_diagnostics(self):
        return self._diagnostics


def test_no_synthesizer_means_empty_donor_leaves_is_empty():
    real = pd.DataFrame({"x": [1, 2, 3]})
    report = fidelity_report(real, real, columns=["x"])
    assert report.empty_donor_leaves == {}


def test_empty_donor_leaves_merges_across_tables_for_requested_columns():
    stub = _StubSynthesizer(
        {
            "roster": {"sector": 3, "wage": 0},
            "jobs": {"sector": 2, "tenure": 5},
        }
    )
    real = pd.DataFrame({"sector": ["a"], "wage": [1], "tenure": [1]})

    report = fidelity_report(real, real, columns=["sector", "wage"], synthesizer=stub)

    # sector's count is summed across both tables; tenure is dropped because
    # it was never in `columns`; wage's zero count still shows up explicitly.
    assert report.empty_donor_leaves == {"sector": 5, "wage": 0}


def test_synthesizer_without_donor_diagnostics_raises_type_error():
    real = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(TypeError, match="donor_diagnostics"):
        fidelity_report(real, real, columns=["x"], synthesizer=object())


# --- CARTSynthesizer.donor_diagnostics(): the real wiring -----------------


def test_cart_synthesizer_exposes_donor_diagnostics_after_a_real_run(schema):
    """End-to-end smoke: a real pipeline run leaves a readable, well-shaped
    diagnostics dict, whether or not any leaf actually ended up empty."""
    synth = sw.CARTSynthesizer(["wage"], tables=["wages"], predictors=["education"])
    sw.Pipeline(schema, synthesizer=synth).run()

    diagnostics = synth.donor_diagnostics()

    assert diagnostics == {"wages": {}} or set(diagnostics["wages"]) <= {"wage"}
    assert synth.donor_diagnostics("wages") == {"wages": diagnostics.get("wages", {})}
    assert synth.donor_diagnostics("no-such-table") == {}


def test_cart_synthesizer_donor_diagnostics_reports_what_the_fit_recorded():
    """`donor_diagnostics` is a thin, faithful passthrough of whatever the
    fitted model accumulated -- proved directly rather than by trying to
    coax an actual empty leaf out of sklearn, which the fit/apply symmetry
    in `_FittedCART` makes hard to force deterministically."""
    synth = sw.CARTSynthesizer(["wage"], tables=["wages"])

    class _FakeModel:
        empty_donor_counts = {"wage": 4}

    synth._fitted["wages"] = _FakeModel()
    synth._complete["wages"] = True

    assert synth.donor_diagnostics() == {"wages": {"wage": 4}}
    assert synth.donor_diagnostics("wages") == {"wages": {"wage": 4}}


def test_donor_diagnostics_reflects_only_the_last_run_when_synthesizer_is_reused(schema):
    """Bug hunt (2026-07-31, I18): reusing one `CARTSynthesizer` across two
    `Pipeline` runs for the same table silently replaced the first run's
    diagnostics with the second's, with nothing distinguishing "never run"
    from "run, then overwritten by an unrelated later run." Not fully fixed
    -- `_fitted` is still keyed by table name only -- but now at least
    honestly documented and pinned down by this test, rather than the
    docstring implying a 1:1 binding to a specific run that isn't real."""
    synth = sw.CARTSynthesizer(["wage"], tables=["wages"], predictors=["education"])
    sw.Pipeline(schema, synthesizer=synth).run()
    first_model = synth._fitted["wages"]

    sw.Pipeline(schema, synthesizer=synth).run()

    assert synth._fitted["wages"] is not first_model


def test_donor_diagnostics_excludes_a_table_still_mid_stream(schema):
    """Bug hunt (2026-07-31, I18): `donor_diagnostics` used to register a
    table as soon as fitting finished, before any chunk had actually been
    applied -- so a caller driving `Pipeline.stream()` by hand and checking
    diagnostics mid-stream got a real-looking but silently incomplete count
    instead of the true, fully-accumulated one. A table now stays absent
    until every chunk has actually been applied."""
    synth = sw.CARTSynthesizer(["wage"], tables=["wages"], predictors=["education"])
    pipe = sw.Pipeline(schema, synthesizer=synth, chunk_size=50)

    stream = pipe.stream("wages")
    next(stream)  # fitting has happened; not one chunk has been applied yet

    assert synth.donor_diagnostics("wages") == {}

    list(stream)  # drain the rest

    assert "wages" in synth.donor_diagnostics()
