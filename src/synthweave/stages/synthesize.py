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

from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .. import _hash
from ..context import RunContext
from ..provenance import as_tagged, modeled
from ..registry import register
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

    def training_frame(self, table: Table, ctx: RunContext) -> pd.DataFrame:
        keys = np.array([f"prior:{table.name}:{i}" for i in range(self.rows)], dtype=object)
        frame = pd.DataFrame(index=range(self.rows))

        for column, dist in self.marginals.items():
            values = np.array(list(dist.keys()), dtype=object)
            weights = np.array(list(dist.values()), dtype=float)
            frame[column] = _hash.pick(keys, ctx.seed, f"prior\x00{column}", values, weights)

        # A joint overrides the two marginals it spans, so declared pairwise
        # structure survives into the training frame.
        for (left, right), dist in self.joints.items():
            pairs = np.array(list(dist.keys()), dtype=object)
            weights = np.array(list(dist.values()), dtype=float)
            picked = _hash.pick(keys, ctx.seed, f"prior\x00{left}\x00{right}", pairs, weights)
            frame[left] = [p[0] for p in picked]
            frame[right] = [p[1] for p in picked]

        return frame


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
        structure: where to learn from. Defaults to `Declared`.
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
        self.structure = structure or Declared()
        self.fit_cap = as_tagged(fit_cap) if fit_cap is not None else modeled(
            DEFAULT_FIT_CAP, "library default fit cap"
        )
        self.max_depth = max_depth
        self.min_samples_leaf = (
            as_tagged(min_samples_leaf)
            if not isinstance(min_samples_leaf, int)
            else modeled(min_samples_leaf, "library default leaf size")
        )

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
            keys = np.asarray(train.index, dtype=str).astype(object)
            pick = _hash.unit(keys, ctx.seed, f"fitsample\x00{table.name}") < (cap / len(train))
            train = train.loc[pick]
            sampled = True

        missing = [c for c in self.columns + self.predictors if c not in train.columns]
        if missing:
            raise KeyError(
                f"table {table.name!r}: structure source has no column(s) {missing}; "
                f"it provides {sorted(train.columns)}"
            )

        model = _FittedCART(
            self.columns, self.predictors, self.max_depth, leaf
        ).fit(train)

        ctx.report(
            table.name,
            "synthesize",
            fit_rows=len(train),
            fit_cap=cap,
            sampled=sampled,
            columns=list(self.columns),
            source=type(self.structure).__name__,
        )

        for chunk in _chain(replay, chunks):
            yield model.apply(chunk, ctx.seed, table.name)


def _chain(first: list[pd.DataFrame], rest: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    yield from first
    yield from rest


class _FittedCART:
    """Trees plus per-leaf donor pools, one per synthesized column."""

    def __init__(self, columns, predictors, max_depth, min_samples_leaf):
        self.columns = columns
        self.predictors = predictors
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.codes: dict[str, dict[Any, int]] = {}
        self.trees: dict[str, Any] = {}
        self.donors: dict[str, dict[int, np.ndarray]] = {}
        self.dtypes: dict[str, Any] = {}
        self.marginal: np.ndarray | None = None

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
            if not _is_numeric(train[column]):
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
            numeric = _is_numeric(y)
            tree = (
                DecisionTreeRegressor(
                    max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf
                )
                if numeric
                else DecisionTreeClassifier(
                    max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf
                )
            )
            y_fit = y.to_numpy() if numeric else self._encode_column(target, y)
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
                    continue
                take = np.minimum((pos[mask] * len(pool)).astype(int), len(pool) - 1)
                values[mask] = pool[take]
            out[target] = self._restore(target, values)
        return out

    def _restore(self, column: str, values: np.ndarray) -> np.ndarray:
        """Give sampled donor values back the dtype the fit saw.

        The dtype comes from the model, never from the chunk, so every chunk
        lands on the same type. An int64 column cannot hold nulls in the first
        place, so a cast back to it cannot be asked to represent one.
        """
        dtype = self.dtypes.get(column)
        if dtype is None or dtype == object:
            return values
        return values.astype(dtype)

    def _encode(self, frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
        cols = [self._encode_column(f, frame[f]) for f in features]
        return np.column_stack(cols) if cols else np.zeros((len(frame), 1))

    def _encode_column(self, name: str, series: pd.Series) -> np.ndarray:
        if name not in self.codes:
            return pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        mapping = self.codes[name]
        return series.map(lambda v: mapping.get(v, -1)).to_numpy(dtype=float)


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(pd.to_numeric(series, errors="coerce").dropna()) and (
        pd.to_numeric(series, errors="coerce").notna().mean() > 0.9
    )
