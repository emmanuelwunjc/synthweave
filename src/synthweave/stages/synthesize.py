"""Stage 2: sequential CART synthesis against a selectable structure source.

The algorithm is the synthpop one. Columns are synthesized one at a time in a
visit sequence. The first is drawn from its marginal; each subsequent column is
predicted by a tree fitted on the columns already synthesized, and the value is
sampled from the donor rows sitting in the predicted leaf. Sampling from real
donor leaves rather than from a fitted distribution is what keeps values
plausible and preserves the joint structure.

What differs from synthpop is where the structure being learned comes from.
A generator that draws columns independently produces a table with no
inter-column structure, so fitting on its output would learn nothing and give
independent columns back. The synthesizer therefore takes a `StructureSource`:

    Declared   structure is already in the data, put there by conditional
               rules in the schema. The model smooths and generalizes it.
    Empirical  fit on real microdata the user supplies. The classic case.
    Prior      fit on published aggregates (marginals, a correlation matrix)
               when the user has statistics but no rows.

The fit cap is a maximum, not a mandate. A table smaller than the cap is
fitted on every row, exactly as a single-table tool would do. Only above the
cap is a sample taken, because fitting a tree on forty million rows is neither
tractable nor necessary.
"""

from __future__ import annotations

import inspect
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .. import _hash
from ..context import RunContext
from ..provenance import as_tagged, modeled
from ..registry import register, registry, resolve
from ..schema import Table
from .base import buffer_to

DEFAULT_FIT_CAP = 200_000


# --- structure sources ------------------------------------------------------


@register("structure", "declared")
class Declared:
    """Structure comes from the generated data itself.

    Valid only because conditional rules can put real structure there. With a
    schema of independent rules there is nothing to learn and the synthesizer
    will say so rather than pretend otherwise.
    """

    def training_frame(self, table: Table, ctx: RunContext) -> pd.DataFrame | None:
        return None


@register("structure", "empirical")
class Empirical:
    """Structure learned from real microdata the user supplies."""

    def __init__(self, frame: pd.DataFrame):
        if frame is None or len(frame) == 0:
            raise ValueError("Empirical structure source needs a non-empty frame")
        self.frame = frame

    def training_frame(self, table: Table, ctx: RunContext) -> pd.DataFrame:
        return self.frame


# Marginals and joints are both figures the user cited from somewhere. A
# joint overwrites the columns it spans, so a marginal that disagrees is
# silently discarded and the output honours only one of the two published
# numbers, with nothing saying which. That is checked at construction rather
# than at generation, so the error names the config and not a frame.
_MARGIN_TOLERANCE = 1e-6


def _check_joints_agree_with_marginals(
    marginals: Mapping[str, Mapping[Any, float]],
    joints: Mapping[tuple[str, str], Mapping[tuple[Any, Any], float]],
) -> None:
    """Raise when a joint implies a different marginal than the declared one."""
    for (left, right), dist in joints.items():
        for position, column in enumerate((left, right)):
            declared = marginals.get(column)
            if not declared:
                # No declared marginal for this column, so nothing to disagree
                # with. The joint is the only statement about it.
                continue
            implied: dict[Any, float] = {}
            for pair, weight in dist.items():
                implied[pair[position]] = implied.get(pair[position], 0.0) + float(weight)
            total = sum(implied.values())
            if total > 0:
                implied = {k: v / total for k, v in implied.items()}

            declared_total = sum(float(v) for v in declared.values())
            normalised = {
                k: float(v) / declared_total if declared_total else 0.0
                for k, v in declared.items()
            }

            disagreements = [
                (value, share, implied.get(value, 0.0))
                for value, share in sorted(normalised.items(), key=str)
                if abs(share - implied.get(value, 0.0)) > _MARGIN_TOLERANCE
            ]
            if disagreements:
                detail = ", ".join(
                    f"{value!r}: marginal says {share:.4g}, joint implies {got:.4g}"
                    for value, share, got in disagreements[:3]
                )
                raise ValueError(
                    f"Prior: the joint {(left, right)!r} implies a different marginal for "
                    f"{column!r} than the one declared ({detail}). Both are figures you "
                    "cited, and applying the joint would silently discard the marginal. "
                    "Reconcile them, or drop the marginal for that column and let the "
                    "joint be the only statement about it."
                )


def _check_joints_do_not_share_a_column(
    joints: Mapping[tuple[str, str], Mapping[tuple[Any, Any], float]],
) -> None:
    """Raise when two joints both name the same column.

    `training_frame` applies joints in dict order, and each one overwrites
    both of the columns it spans. Two joints sharing a column can agree with
    every declared marginal and still not both survive: whichever runs last
    wins the shared column outright, and the row-level link the earlier
    joint was cited for is gone, replaced by an independently drawn value
    that only happens to share the same marginal shares. Nothing before this
    caught it, because `_check_joints_agree_with_marginals` only compares a
    joint against a *marginal*, never a joint against another joint.
    """
    seen: dict[str, tuple[str, str]] = {}
    for pair in joints:
        for column in pair:
            earlier = seen.get(column)
            if earlier is not None and earlier != pair:
                raise ValueError(
                    f"Prior: joints {earlier!r} and {pair!r} both name {column!r}. "
                    "Applying both would silently discard whichever ran first: "
                    "training_frame overwrites a joint's columns in full, so the "
                    "row-level link the earlier joint declared cannot survive. "
                    "Combine them into a single joint over every column you need "
                    "correlated together, or drop one."
                )
            seen[column] = pair


@register("structure", "prior")
class Prior:
    """Structure from published aggregates rather than rows.

    Marginals and an optional pairwise joint are expanded into a synthetic
    training frame, which then goes through the same CART path as real data.
    One code path serves all three sources, so a new source kind never means
    touching the synthesizer.

        Prior(
            marginals={"education": {"HS": 0.6, "College": 0.4}},
            joints={("education", "employed"): {("HS", True): 0.42, ...}},
        )
    """

    def __init__(
        self,
        marginals: Mapping[str, Mapping[Any, float]],
        joints: Mapping[tuple[str, str], Mapping[tuple[Any, Any], float]] | None = None,
        rows: int = 50_000,
    ):
        if not marginals:
            raise ValueError("Prior needs at least one marginal")
        self.marginals = marginals
        self.joints = joints or {}
        self.rows = rows
        _check_joints_do_not_share_a_column(self.joints)
        _check_joints_agree_with_marginals(self.marginals, self.joints)

    def training_frame(self, table: Table, ctx: RunContext) -> pd.DataFrame:
        keys = np.array([f"prior:{table.name}:{i}" for i in range(self.rows)], dtype=object)
        frame = pd.DataFrame(index=range(self.rows))

        for column, dist in self.marginals.items():
            values = np.array(list(dist.keys()), dtype=object)
            weights = np.array(list(dist.values()), dtype=float)
            picked = _hash.pick(keys, ctx.seed, f"prior\x00{column}", values, weights)
            # `pick` always returns object, since it must support arbitrary
            # hashable keys (strings, tuples). A marginal declared over
            # numbers -- income, age, the obvious `Prior` use case -- gets
            # its real dtype back so `_FittedCART._numeric` fits it as a
            # number rather than guessing a classifier from an object
            # column. A marginal over strings or anything else stays
            # object, same as every other categorical column here.
            natural = np.asarray(list(dist.keys())).dtype
            frame[column] = picked.astype(natural) if natural.kind in "iuf" else picked

        # A joint overrides the two marginals it spans, so declared pairwise
        # structure survives into the training frame.
        for (left, right), dist in self.joints.items():
            pairs = list(dist.keys())
            weights = np.array(list(dist.values()), dtype=float)
            # Pick over positions, not over the pairs themselves. Handing the
            # tuples to `pick` builds a 2-D array, and `pick` is a 1-D
            # primitive by contract, so it rejected every joint outright.
            positions = np.arange(len(pairs), dtype=object)
            chosen = _hash.pick(keys, ctx.seed, f"prior\x00{left}\x00{right}", positions, weights)
            frame[left] = [pairs[i][0] for i in chosen]
            frame[right] = [pairs[i][1] for i in chosen]

        return frame


class StructureConfigError(ValueError):
    """A structure source was named that cannot be built from a name alone."""


def _resolve_structure_name(name: str) -> Any:
    """Resolve a registered structure name, or say why the name can't work.

    Resolving a registered *class* means instantiating it with no arguments.
    That works for `Declared` and for any third-party source that needs no
    configuration, but `Empirical` needs a `frame` and `Prior` needs
    `marginals`, and a bare string has no channel to carry either. Those
    used to surface as `TypeError: Empirical.__init__() missing 1 required
    positional argument: 'frame'`, which names the mechanism rather than the
    fix. Check first and name the fix instead.
    """
    found = registry("structure").get(name)
    if isinstance(found, type):
        required = [
            p.name
            for p in inspect.signature(found).parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        ]
        if required:
            raise StructureConfigError(
                f"structure={name!r} needs configuration a bare name cannot carry "
                f"({', '.join(required)}). Pass a configured instance instead, "
                f"e.g. structure={found.__name__}({required[0]}=...)."
            )
    return resolve("structure", name)


def _coerce_structure(structure: Any) -> Any:
    """A structure source as given, or inferred from a plainer value.

    `None` -> `Declared()`. A registered name (`"declared"`, or a third
    party's own registration that needs no configuration) -> resolved
    through the same `"structure"` registry `Empirical`/`Prior`/`Declared`
    register themselves into. `"empirical"`/`"prior"` are registered there
    too but cannot be built from the name alone (they need a frame and
    marginals respectively), so naming them raises `StructureConfigError`
    pointing at the configured-instance form. Real data already in hand ->
    `Empirical(structure)`. Aggregate stats already in hand ->
    `Prior(marginals=structure)`. Anything else (an already-built structure
    source instance) passes through as-is.
    """
    if structure is None:
        return Declared()
    if isinstance(structure, str):
        return _resolve_structure_name(structure)
    if isinstance(structure, pd.DataFrame):
        return Empirical(structure)
    if isinstance(structure, Mapping):
        # Check the shape before committing to the guess. Prior only asserts
        # the dict is non-empty, so a Mapping that is not marginals (e.g.
        # {"table": some_frame}) was accepted here and failed much later
        # inside Prior.training_frame, with an error that never mentioned
        # the coercion that had guessed wrong.
        bad = [
            key
            for key, dist in structure.items()
            if not isinstance(dist, Mapping)
            or not all(isinstance(weight, (int, float)) for weight in dist.values())
        ]
        if bad:
            raise TypeError(
                f"structure= was given a dict, which is shorthand for Prior marginals shaped "
                f"{{column: {{value: probability}}}}, but {bad[:3]} "
                f"{'does' if len(bad) == 1 else 'do'} not have that shape. "
                "Pass real data as a DataFrame, or build the source explicitly with "
                "sw.Empirical(frame) or sw.Prior(marginals=..., joints=...)."
            )
        return Prior(marginals=structure)
    return structure


# --- the synthesizer --------------------------------------------------------


@register("synthesizer", "cart")
class CARTSynthesizer:
    """Sequential CART synthesis, fitted once and applied chunk-wise.

    Args:
        columns: columns to synthesize, in visit order. Order matters: each is
            conditioned on the ones before it.
        tables: tables to apply to. Other tables pass through untouched.
            Unset means every table, which then requires every table to carry
            the named columns.
        predictors: columns to condition on but not resynthesize, such as
            carried entity attributes.
        numeric: columns to fit as numbers even though their dtype does not
            say so, such as an object column holding numeric strings. Values
            that will not parse raise, naming the column.
        structure: where to learn from. Defaults to `Declared`. Also accepts
            a `pd.DataFrame` directly (wrapped in `Empirical`), a `Mapping`
            directly (wrapped in `Prior(marginals=...)`), or the name of a
            structure source registered under the `"structure"` kind that
            needs no configuration (`"declared"`, or a third party's own).
            `"empirical"`/`"prior"` cannot be named this way: they need a
            frame and marginals respectively, which a bare string cannot
            carry, so naming them raises `StructureConfigError`.
        fit_cap: maximum rows to fit on. Below this, every row is used.
        max_depth, min_samples_leaf: tree controls. Shallow trees generalize
            more and disclose less.
    """

    def __init__(
        self,
        columns: Sequence[str],
        *,
        tables: Sequence[str] | None = None,
        predictors: Sequence[str] = (),
        numeric: Sequence[str] = (),
        structure: Any = None,
        fit_cap: int | Any = None,
        max_depth: int | None = None,
        min_samples_leaf: int | Any = 5,
    ):
        if not columns:
            raise ValueError("CARTSynthesizer needs at least one column to synthesize")
        self.columns = list(columns)
        self.tables = set(tables) if tables is not None else None
        self.predictors = list(predictors)
        self.numeric = set(numeric)
        self.structure = _coerce_structure(structure)
        self.fit_cap = as_tagged(fit_cap) if fit_cap is not None else modeled(
            DEFAULT_FIT_CAP, "library default fit cap"
        )
        self.max_depth = max_depth
        self.min_samples_leaf = (
            as_tagged(min_samples_leaf)
            if not isinstance(min_samples_leaf, int)
            else modeled(min_samples_leaf, "library default leaf size")
        )
        # Populated per table by `run`, read back by `donor_diagnostics`.
        # `synthweave.fidelity.fidelity_report` is the reason this exists: it
        # takes a fitted synthesizer to surface leaves that fell back to a
        # placeholder value, something the two output frames alone can't show.
        # Keyed by table name, not by run: reusing one `CARTSynthesizer`
        # instance across more than one `Pipeline` run for the same table
        # overwrites the earlier fit's entry. `_complete` exists because
        # `run` is a generator — fitting (and registering `_fitted`) happens
        # before the first chunk is even pulled, but `empty_donor_counts`
        # only finishes accumulating once every chunk has actually been
        # applied. Without it, `donor_diagnostics` could hand back a
        # snapshot mid-stream and it would look identical to a complete one.
        self._fitted: dict[str, "_FittedCART"] = {}
        self._complete: dict[str, bool] = {}

    def donor_diagnostics(self, table: str | None = None) -> dict[str, dict[str, int]]:
        """Empty-donor-leaf fallback counts from the last fit, by table and column.

        A count above zero for a (table, column) pair means some rows in that
        table's synthesized `column` landed in a decision-tree leaf with no
        donor rows, and kept their pre-synthesis placeholder value instead of
        a real synthesized one. A table is absent until a pipeline has fully
        run this synthesizer for it: `Pipeline.run()`/`run_to()` always reach
        that point, but a caller manually driving `Pipeline.stream()` and
        inspecting diagnostics before consuming every chunk sees the table
        stay absent rather than get a real but incomplete count.
        `table=None` (the default) returns every table finished so far.

        Reflects the *last* fit only: reusing one `CARTSynthesizer` instance
        across more than one `Pipeline` run for the same table replaces the
        earlier run's counts rather than combining them.
        """
        if table is not None:
            model = self._fitted.get(table)
            complete = self._complete.get(table, False)
            return {table: dict(model.empty_donor_counts)} if model and complete else {}
        return {
            name: dict(model.empty_donor_counts)
            for name, model in self._fitted.items()
            if self._complete.get(name, False)
        }

    def run(
        self, chunks: Iterator[pd.DataFrame], table: Table, ctx: RunContext
    ) -> Iterator[pd.DataFrame]:
        # A pipeline has one synthesizer but many tables, and a synthesized
        # column rarely exists in all of them. Naming tables explicitly scopes
        # the stage; leaving it unset means every table, which then requires
        # every table to have the columns.
        if self.tables is not None and table.name not in self.tables:
            yield from chunks
            return

        cap = ctx.provenance.add(f"{table.name}.synth.fit_cap", self.fit_cap)
        if self.numeric:
            ctx.provenance.add(
                f"{table.name}.synth.numeric",
                modeled(sorted(self.numeric), "columns declared numeric by the user"),
            )
        leaf = ctx.provenance.add(f"{table.name}.synth.min_samples_leaf", self.min_samples_leaf)

        train = self.structure.training_frame(table, ctx)
        replay: list[pd.DataFrame] = []

        if train is None:
            train, replay = buffer_to(chunks, cap)
            if train.empty:
                return
            sampled = False
        else:
            sampled = len(train) > cap
            if sampled:
                keys = np.arange(len(train)).astype(str).astype(object)
                pick = _hash.unit(keys, ctx.seed, f"fitsample\x00{table.name}") < (cap / len(train))
                train = train.loc[pick]
                if len(train) > cap:
                    # The draw above targets `cap` in expectation but can
                    # overshoot it. A `RangeIndex` survives a boolean mask
                    # with its original positional labels, so keying a
                    # second Bernoulli pass by `train.index` reused the
                    # exact same keys, seed, and salt as the pass above and
                    # reproduced its exact same draw -- a strictly looser
                    # threshold applied to an identical draw keeps every
                    # survivor, making that "second pass" a no-op.
                    # Truncating by position instead, the same exact-cap
                    # discipline `buffer_to` already uses for the
                    # no-structure path, guarantees the cap here too.
                    train = train.iloc[:cap]

        missing = [c for c in self.columns + self.predictors if c not in train.columns]
        if missing:
            raise KeyError(
                f"table {table.name!r}: structure source has no column(s) {missing}; "
                f"it provides {sorted(train.columns)}"
            )

        model = _FittedCART(
            self.columns, self.predictors, self.max_depth, leaf, ctx.seed, self.numeric
        ).fit(train)
        self._fitted[table.name] = model
        self._complete[table.name] = False

        ctx.report(
            table.name,
            "synthesize",
            fit_rows=len(train),
            fit_cap=cap,
            sampled=sampled,
            columns=list(self.columns),
            numeric=sorted(self.numeric),
            source=type(self.structure).__name__,
        )

        for chunk in _chain(replay, chunks):
            yield model.apply(chunk, ctx.seed, table.name)
        self._complete[table.name] = True


def _chain(first: list[pd.DataFrame], rest: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    yield from first
    yield from rest


class _FittedCART:
    """Trees plus per-leaf donor pools, one per synthesized column."""

    def __init__(self, columns, predictors, max_depth, min_samples_leaf, seed, numeric=()):
        self.columns = columns
        self.predictors = predictors
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
        self.numeric = set(numeric)
        self.codes: dict[str, dict[Any, int]] = {}
        self.trees: dict[str, Any] = {}
        self.donors: dict[str, dict[int, np.ndarray]] = {}
        self.dtypes: dict[str, Any] = {}
        self.marginal: np.ndarray | None = None
        # Rows that landed in a leaf with no donors and kept their
        # pre-synthesis value, accumulated across every chunk `apply` sees.
        self.empty_donor_counts: dict[str, int] = {}

    def fit(self, train: pd.DataFrame) -> "_FittedCART":
        # Donor sampling runs through object arrays, because a leaf's pool is
        # sliced by a boolean mask and pandas has no dtype-preserving way to
        # do that per leaf. The column's type is a property of the fit, not of
        # a chunk, so recording it here is what lets `apply` put it back
        # without asking the chunk. Reading the dtype off the chunk instead
        # would make it chunk dependent, which is exactly what invariant 2
        # forbids.
        self.dtypes = {column: train[column].dtype for column in self.columns}

        for column in self.columns + self.predictors:
            if not self._numeric(column, train[column]):
                cats = pd.unique(train[column].dropna())
                self.codes[column] = {v: i for i, v in enumerate(cats)}

        self.marginal = train[self.columns[0]].to_numpy(dtype=object)

        # Every column with something to condition on gets a tree, including
        # the first when predictors are supplied. synthpop draws its first
        # variable unconditionally because nothing precedes it; here the
        # predictors do precede it, and ignoring them would throw away the
        # relationship the user asked to preserve.
        for i, target in enumerate(self.columns):
            features = self.predictors + self.columns[:i]
            if not features:
                continue
            X = self._encode(train, features)
            y = train[target]
            numeric = self._numeric(target, y)
            # sklearn's "best" splitter still consumes randomness to break
            # ties between equally-good splits, so an unseeded tree gives a
            # different fit (and so different donor groupings) on every
            # call, even from identical data. Deriving the seed from the run
            # seed and the target keeps that tie-break itself deterministic,
            # and salting per target keeps one column's tie-break from
            # correlating with another's.
            random_state = int(_hash.hash_key(self.seed, f"cart\x00{target}"), 16) % (2**32)
            tree = (
                DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=random_state,
                )
                if numeric
                else DecisionTreeClassifier(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=random_state,
                )
            )
            y_fit = self._as_numbers(target, y) if numeric else self._encode_column(target, y)
            tree.fit(X, y_fit)
            self.trees[target] = (tree, features)

            leaves = tree.apply(X)
            raw = y.to_numpy(dtype=object)
            self.donors[target] = {
                leaf: raw[leaves == leaf] for leaf in np.unique(leaves)
            }
        return self

    def apply(self, chunk: pd.DataFrame, seed, table_name: str) -> pd.DataFrame:
        if chunk.empty:
            return chunk
        keys = chunk["_sw_row"].to_numpy()
        out = chunk.copy()

        first = self.columns[0]
        if first not in self.trees:
            # Nothing to condition on, so the marginal is the whole story.
            idx = _hash.integers(
                keys, seed, f"synth\x00{table_name}\x00{first}", 0, len(self.marginal)
            )
            out[first] = self._restore(first, self.marginal[idx])

        for target in self.columns:
            if target not in self.trees:
                continue
            tree, features = self.trees[target]
            leaves = tree.apply(self._encode(out, features))
            pos = _hash.unit(keys, seed, f"synth\x00{table_name}\x00{target}")
            values = np.empty(len(out), dtype=object)
            for leaf in np.unique(leaves):
                mask = leaves == leaf
                pool = self.donors[target].get(leaf)
                if pool is None or len(pool) == 0:
                    values[mask] = out[target].to_numpy(dtype=object)[mask]
                    self.empty_donor_counts[target] = (
                        self.empty_donor_counts.get(target, 0) + int(mask.sum())
                    )
                    continue
                take = np.minimum((pos[mask] * len(pool)).astype(int), len(pool) - 1)
                values[mask] = pool[take]
            out[target] = self._restore(target, values)
        return out

    def _numeric(self, column: str, series: pd.Series) -> bool:
        """Whether to fit this column as numbers.

        The column's dtype answers this, which it can do now that rules
        declare what they produce. Guessing from the values used to be the
        answer, and it meant a column of amounts with a handful of `refused`
        entries was called numeric at 95% parseable and then handed to a
        regressor unconverted. `numeric=` is the explicit override for an
        object column that really does hold numbers.
        """
        return column in self.numeric or pd.api.types.is_numeric_dtype(series)

    def _as_numbers(self, column: str, series: pd.Series) -> np.ndarray:
        """Numbers for the fit, failing by name rather than from inside sklearn."""
        if pd.api.types.is_numeric_dtype(series):
            return series.to_numpy()
        converted = pd.to_numeric(series, errors="coerce")
        bad = series[converted.isna() & series.notna()]
        if len(bad):
            raise ValueError(
                f"column {column!r} was declared numeric but holds {len(bad)} value(s) "
                f"that are not numbers, e.g. {sorted(set(bad.astype(str)))[:3]}. "
                f"Clean them, or drop the column from `numeric` to fit it as categories."
            )
        return converted.to_numpy()

    def _restore(
        self, column: str, values: np.ndarray
    ) -> np.ndarray | pd.api.extensions.ExtensionArray:
        """Give sampled donor values back the dtype the fit saw.

        The dtype comes from the model, never from the chunk, so every chunk
        lands on the same type. An int64 column cannot hold nulls in the first
        place, so a cast back to it cannot be asked to represent one.

        `ndarray.astype` speaks numpy dtypes only, so a pandas ExtensionDtype
        (`category`, nullable `Int64`, and every text column under pandas 3)
        raises `TypeError` there rather than converting. Those go through
        `pd.array`, which is the only constructor that takes an
        `ExtensionDtype` and hands back the matching extension array.
        """
        dtype = self.dtypes.get(column)
        if dtype is None or dtype == object:
            return values
        if isinstance(dtype, pd.api.extensions.ExtensionDtype):
            return pd.array(values, dtype=dtype)
        return values.astype(dtype)

    def _encode(self, frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
        cols = [self._encode_column(f, frame[f]) for f in features]
        return np.column_stack(cols) if cols else np.zeros((len(frame), 1))

    def _encode_column(self, name: str, series: pd.Series) -> np.ndarray:
        if name not in self.codes:
            return pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        mapping = self.codes[name]
        return series.map(lambda v: mapping.get(v, -1)).to_numpy(dtype=float)
