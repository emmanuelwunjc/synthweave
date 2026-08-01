"""How closely `Empirical`/CART synthesis preserved the real data it fit on.

`Declared` and `Prior` structure sources have no original relationship to
check against: the schema author declared whatever structure they wanted, so
there is nothing "real" to be unfaithful to. `Empirical` is different. When
`CARTSynthesizer` fits on real microdata, the whole point is that the output
should carry the same statistical relationships the real input actually
had. Whether it did was previously only a prose claim about how the CART fit
mechanism works. `fidelity_report` turns it into numbers.

Three checks, each comparing real vs. synthesized:

    marginals      per column: a KS test for numeric columns, a
                   total-variation share delta for categorical columns.
    associations   per column pair: Pearson correlation (numeric-numeric),
                   Cramer's V (categorical-categorical), or the correlation
                   ratio / eta (numeric-categorical, any category count).
    empty donors   per column, only when a fitted `CARTSynthesizer` is
                   passed in: rows that landed in a decision-tree leaf with
                   no donor rows and kept their pre-synthesis placeholder
                   instead of a synthesized value. This is zero by
                   construction, not a measurement of the data: a leaf
                   exists only because training rows partitioned into it, so
                   every leaf has donors and no row can reach one without.
                   Read a non-zero value as a bug in synthweave rather than
                   anything about the input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.stats.contingency import association as _cramers_v_stat


@dataclass
class FidelityReport:
    """Real-vs-synthesized comparison. Build with `fidelity_report`, not directly.

    `to_frame()` returns the per-column marginal check. `associations` is the
    per-pair structure check, as its own frame rather than folded into
    `to_frame()` since it has a different shape (one row per column pair, not
    per column). `empty_donor_leaves` is a plain `{column: count}` dict,
    empty unless `fidelity_report` was given a `synthesizer=`.

    Every `passed` column is `None` unless the matching key was present in
    the `thresholds` given to `fidelity_report` — `fidelity_report` reports
    numbers by default and only judges them when explicitly asked to.
    """

    columns: list[str] = field(default_factory=list)
    column_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    associations: pd.DataFrame = field(default_factory=pd.DataFrame)
    empty_donor_leaves: dict[str, int] = field(default_factory=dict)
    thresholds: dict[str, float] | None = None

    def to_frame(self) -> pd.DataFrame:
        """The per-column marginal check: one row per column in `columns`."""
        return self.column_stats


def fidelity_report(
    synth_df: pd.DataFrame,
    real_df: pd.DataFrame,
    columns: Sequence[str],
    *,
    thresholds: Mapping[str, float] | None = None,
    synthesizer: Any = None,
) -> FidelityReport:
    """Compare synthesized output to the real data it was fitted on.

    Args:
        synth_df: a table `CARTSynthesizer` produced, fitted via `Empirical`.
        real_df: the real data given to `Empirical` (or the same real frame
            used elsewhere, for the same columns).
        columns: columns to check, in either frame. Required and never
            auto-detected: real and synth frames commonly share columns
            (identifiers, carried keys) nobody meant to hold a preserved
            statistical relationship.
        thresholds: off by default, meaning purely descriptive output (raw
            KS statistics, share deltas, association deltas), since this
            library has no built-in opinion about what counts as "close
            enough" for data it knows nothing about. Pass a dict to turn on
            pass/fail judgement, e.g.
            `{"ks_pvalue": 0.05, "category_share_delta": 0.1, "association_delta": 0.1}`.
            Only the keys supplied gate a verdict; every other check's
            `passed` column stays `None`.
        synthesizer: a `CARTSynthesizer` you fully ran (via `Pipeline.run()`
            or `run_to()`) to produce `synth_df`, if you want
            `empty_donor_leaves` populated. Without it that dict is empty
            rather than wrong: which leaves had no donors is state that
            lives inside the fit, not in the two output frames, so there is
            no way to recover it after the fact. Reflects that synthesizer's
            *last* completed run for each table, not necessarily the one
            that produced `synth_df` specifically — reusing one
            `CARTSynthesizer` instance across multiple `Pipeline` runs for
            the same table replaces the earlier run's counts. See
            `CARTSynthesizer.donor_diagnostics`.

    Returns:
        A `FidelityReport`.
    """
    columns = list(columns)
    if not columns:
        raise ValueError("fidelity_report needs at least one column")

    missing_real = [c for c in columns if c not in real_df.columns]
    missing_synth = [c for c in columns if c not in synth_df.columns]
    if missing_real or missing_synth:
        raise KeyError(
            f"columns not present in both frames (missing from real: {missing_real}, "
            f"missing from synth: {missing_synth})"
        )

    numeric_cols = {c for c in columns if pd.api.types.is_numeric_dtype(real_df[c])}
    thresholds = dict(thresholds) if thresholds else {}

    return FidelityReport(
        columns=columns,
        column_stats=_column_stats(real_df, synth_df, columns, numeric_cols, thresholds),
        associations=_association_matrix(real_df, synth_df, columns, numeric_cols, thresholds),
        empty_donor_leaves=_empty_donor_leaves(synthesizer, columns),
        thresholds=thresholds or None,
    )


# --- marginals ---------------------------------------------------------


def _column_stats(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    columns: list[str],
    numeric_cols: set[str],
    thresholds: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col in numeric_cols:
            rows.append(_numeric_stat(real_df[col], synth_df[col], thresholds))
        else:
            rows.append(_categorical_stat(real_df[col], synth_df[col], thresholds))
        rows[-1]["column"] = col
    return pd.DataFrame(rows, columns=["column", "kind", "metric", "value", "pvalue", "passed"])


def _numeric_stat(real: pd.Series, synth: pd.Series, thresholds: dict[str, float]) -> dict:
    real_vals = pd.to_numeric(real, errors="coerce").dropna()
    synth_vals = pd.to_numeric(synth, errors="coerce").dropna()
    if len(real_vals) and len(synth_vals):
        stat, pvalue = ks_2samp(real_vals, synth_vals)
    else:
        stat, pvalue = float("nan"), float("nan")
    passed = (
        bool(pvalue >= thresholds["ks_pvalue"])
        if "ks_pvalue" in thresholds and not np.isnan(pvalue)
        else None
    )
    return {
        "kind": "numeric",
        "metric": "ks_statistic",
        "value": float(stat),
        "pvalue": float(pvalue),
        "passed": passed,
    }


def _categorical_stat(real: pd.Series, synth: pd.Series, thresholds: dict[str, float]) -> dict:
    value = _share_delta(real, synth)
    passed = (
        bool(value <= thresholds["category_share_delta"])
        if "category_share_delta" in thresholds
        else None
    )
    return {
        "kind": "categorical",
        "metric": "category_share_delta",
        "value": value,
        "pvalue": float("nan"),
        "passed": passed,
    }


def _share_delta(real: pd.Series, synth: pd.Series) -> float:
    """Total variation distance between two categories' shares, in [0, 1]."""
    real_shares = real.value_counts(normalize=True, dropna=True)
    synth_shares = synth.value_counts(normalize=True, dropna=True)
    categories = real_shares.index.union(synth_shares.index)
    r = real_shares.reindex(categories, fill_value=0.0)
    s = synth_shares.reindex(categories, fill_value=0.0)
    return 0.5 * float((r - s).abs().sum())


# --- associations --------------------------------------------------------


def _association_matrix(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    columns: list[str],
    numeric_cols: set[str],
    thresholds: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1 :]:
            metric, real_val, synth_val = _association(
                real_df, synth_df, col_a, col_b, numeric_cols
            )
            delta = (
                float("nan")
                if np.isnan(real_val) or np.isnan(synth_val)
                else abs(real_val - synth_val)
            )
            passed = (
                bool(delta <= thresholds["association_delta"])
                if "association_delta" in thresholds and not np.isnan(delta)
                else None
            )
            rows.append(
                {
                    "column_a": col_a,
                    "column_b": col_b,
                    "metric": metric,
                    "real": real_val,
                    "synth": synth_val,
                    "delta": delta,
                    "passed": passed,
                }
            )
    return pd.DataFrame(
        rows, columns=["column_a", "column_b", "metric", "real", "synth", "delta", "passed"]
    )


def _association(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    col_a: str,
    col_b: str,
    numeric_cols: set[str],
) -> tuple[str, float, float]:
    a_numeric = col_a in numeric_cols
    b_numeric = col_b in numeric_cols
    if a_numeric and b_numeric:
        return (
            "pearson",
            _pearson(real_df[col_a], real_df[col_b]),
            _pearson(synth_df[col_a], synth_df[col_b]),
        )
    if not a_numeric and not b_numeric:
        return (
            "cramers_v",
            _cramers_v(real_df[col_a], real_df[col_b]),
            _cramers_v(synth_df[col_a], synth_df[col_b]),
        )
    cat_col, num_col = (col_a, col_b) if b_numeric else (col_b, col_a)
    return (
        "correlation_ratio",
        _correlation_ratio(real_df[cat_col], real_df[num_col]),
        _correlation_ratio(synth_df[cat_col], synth_df[num_col]),
    )


def _pearson(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    mask = a.notna() & b.notna()
    if mask.sum() < 2 or a[mask].std() == 0 or b[mask].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna()
    table = pd.crosstab(a[mask], b[mask])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan")
    return float(_cramers_v_stat(table.to_numpy(), method="cramer"))


def _correlation_ratio(categorical: pd.Series, numeric: pd.Series) -> float:
    """eta: generalizes point-biserial correlation past two categories.

    eta^2 is the share of the numeric column's variance explained by category
    membership (between-group sum of squares over total sum of squares); eta
    is its square root, put on the same [0, 1] scale as Cramer's V.
    """
    numeric = pd.to_numeric(numeric, errors="coerce")
    mask = categorical.notna() & numeric.notna()
    categorical = categorical[mask]
    numeric = numeric[mask]
    if categorical.nunique() < 2 or len(numeric) < 2:
        return float("nan")
    overall_mean = numeric.mean()
    ss_total = float(((numeric - overall_mean) ** 2).sum())
    if ss_total == 0:
        return float("nan")
    ss_between = sum(
        len(group) * (group.mean() - overall_mean) ** 2
        for _, group in numeric.groupby(categorical, observed=True)
    )
    return float(np.sqrt(max(ss_between, 0.0) / ss_total))


# --- empty donor leaves ----------------------------------------------------


def _empty_donor_leaves(synthesizer: Any, columns: list[str]) -> dict[str, int]:
    if synthesizer is None:
        return {}
    if not hasattr(synthesizer, "donor_diagnostics"):
        raise TypeError(
            f"synthesizer must expose donor_diagnostics() (a fitted CARTSynthesizer "
            f"does); got {type(synthesizer).__name__}"
        )
    merged: dict[str, int] = {}
    for table_diagnostics in synthesizer.donor_diagnostics().values():
        for column, count in table_diagnostics.items():
            if column in columns:
                merged[column] = merged.get(column, 0) + count
    return merged
