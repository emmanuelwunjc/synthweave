"""The mode-based front door: sugar over the schema/pipeline API for a
declared shape of user, without hiding it.

`Mode` is not instantiated directly. `sw.Mode.metadata()` (and the
`real_data`/`scope` modes added on top of the same base) are the
constructors. Each holds the noise-rate bookkeeping `attribute()` collects,
so `schema()` can wire a `Noise` stage without the caller ever building one
by hand.
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .pipeline import Pipeline, PipelineResult
from .rules import Choice, Normal, Rule, Uniform, _BaseRule
from .schema import Entity, Schema, Table
from .stages.noise import Missing, Noise, NoiseOp, OCR, Typo
from .stages.synthesize import DEFAULT_FIT_CAP, CARTSynthesizer, Empirical

_NOISE_RATE_KWARGS = ("missing_rate", "typo_rate", "ocr_rate")
_NOISE_OP_BY_KWARG = {"missing_rate": Missing, "typo_rate": Typo, "ocr_rate": OCR}


class Mode:
    """Shared plumbing every concrete mode reuses."""

    def __init__(self) -> None:
        self._noise_kwargs: dict[str, dict[str, float]] = {}

    @staticmethod
    def metadata() -> "MetadataMode":
        return MetadataMode()

    @staticmethod
    def real_data(source: pd.DataFrame | str | Path, *, epsilon: float = 1.0) -> "RealDataMode":
        """Real microdata the user supplies, synthesized via CART.

        `source` is a `pd.DataFrame`, or a path to a `.csv`/`.parquet` file.

        `epsilon` is NOT differential privacy: no Laplace/Gaussian mechanism
        and no privacy accounting exist in this codebase. It is a convenience
        knob mapped onto `CARTSynthesizer`'s existing, already-correct
        generalization controls (`max_depth`, `min_samples_leaf`, `fit_cap`),
        which is real and effective but not a formal privacy guarantee. See
        `_cart_knobs` for the mapping.
        """
        _check_epsilon(epsilon, "real_data")
        return RealDataMode(_load_source(source), epsilon=epsilon)

    @staticmethod
    def scope(area_code: str, *, epsilon: float = 1.0) -> "ScopeMode":
        """Maps onto state-level ACS geography only, nothing finer than that.

        `fetch_pums` resolves geography to a US state and no further (no
        county, no PUMA); `area_code` is passed straight through to it as
        `state=`, so any form `_resolve_state` accepts works here too (a
        FIPS code, a USPS abbreviation, or a full state name).

        `epsilon` is NOT differential privacy: no Laplace/Gaussian mechanism
        and no privacy accounting exist in this codebase. It is a convenience
        knob mapped onto `CARTSynthesizer`'s existing, already-correct
        generalization controls (`max_depth`, `min_samples_leaf`, `fit_cap`),
        which is real and effective but not a formal privacy guarantee. See
        `_cart_knobs` for the mapping. It applies here for the same reason it
        applies in `real_data` mode, and more so: the donor rows are real
        Census respondent records, so leaving the synthesizer at its
        unbounded defaults would disclose more than the mode a user feeds
        their own data to. `attribute(name, variable=..., epsilon=...)`
        overrides it per attribute.
        """
        _check_epsilon(epsilon, "scope")
        return ScopeMode(area_code, epsilon=epsilon)

    def attribute(self, name: str, **kwargs: Any) -> Rule:
        noise_kwargs = {k: kwargs.pop(k) for k in _NOISE_RATE_KWARGS if k in kwargs}
        if noise_kwargs:
            self._noise_kwargs.setdefault(name, {}).update(noise_kwargs)
        return self._build_rule(name, **kwargs)

    def _build_rule(self, name: str, **kwargs: Any) -> Rule:
        raise NotImplementedError

    def entity(
        self,
        name: str,
        *,
        count: int,
        attributes: dict | None = None,
        identifiers: Sequence[str] | None = None,
    ) -> Entity:
        return Entity(
            name,
            count=count,
            attributes=attributes or {},
            identifiers=identifiers or (),
        )

    def table(
        self,
        name: str,
        *,
        grain: Any,
        columns: dict | None = None,
        carry: Any = (),
        identifiers: Sequence[str] | None = None,
        coverage: float = 1.0,
    ) -> Table:
        return Table(
            name,
            grain=grain,
            columns=columns or {},
            carry=carry,
            identifiers=identifiers or (),
            coverage=coverage,
        )

    def schema(self, *, entities: list[Entity], tables: list[Table], seed: int | str) -> "ModeSchema":
        schema = Schema(entities=entities, tables=tables, seed=seed)
        # The noise map resolves now: an unmatched rate is a declaration
        # mistake, so it should surface on the call that declared it. The
        # pipeline builder goes across uncalled. `ScopeMode` fetches real rows
        # off the network inside it, and a fetch (or a ValueError for an area
        # code the connector does not recognize) belongs on `.run()`, not on a
        # call that otherwise only assembles objects.
        return ModeSchema(schema, self._noise_for(schema), self._extra_pipeline_kwargs)

    def _noise_for(self, schema: Schema) -> dict[str, dict[str, list[NoiseOp]]]:
        """Match every recorded noise rate against the schema's real columns.

        Resolved here rather than in `table()` because only a `Schema` knows
        the full picture: it has already expanded `carry="*"` against the
        entity, and it holds every table at once, so a rate declared after a
        table was built still lands. Doing it per table as it was declared
        made both of those silently drop the rate.
        """
        noise: dict[str, dict[str, list[NoiseOp]]] = {}
        matched: set[str] = set()
        for table in schema.tables:
            for column_name in list(table.carry) + list(table.columns):
                rates = self._noise_kwargs.get(column_name)
                if not rates:
                    continue
                matched.add(column_name)
                noise.setdefault(table.name, {})[column_name] = [
                    _NOISE_OP_BY_KWARG[kwarg](rate) for kwarg, rate in rates.items()
                ]
        # A rate nobody applies is always a mistake (a typo'd name, a column
        # that never made it onto a table), and silence is what made the two
        # bugs above survive.
        unmatched = sorted(set(self._noise_kwargs) - matched)
        if unmatched:
            raise ValueError(
                f"noise rates were declared for {unmatched}, but no table carries or "
                "generates a column by that name, so they would never be applied"
            )
        return noise

    def _extra_pipeline_kwargs(self) -> dict[str, Any]:
        return {}


def _check_kwargs(
    name: str,
    kwargs: dict[str, Any],
    *,
    known: Sequence[str],
    context: str,
    required: Sequence[str] = (),
) -> None:
    """Gate one rule branch's kwargs before the rule is built.

    Both halves exist because the failure is otherwise silent or cryptic. An
    unknown kwarg used to be dropped, so a misspelled `missing_rate` produced
    a clean-looking rule and no noise. A missing one used to surface as a bare
    KeyError naming neither the attribute nor what it wanted.
    """
    for key in required:
        if key not in kwargs:
            raise ValueError(f"attribute {name!r}: {context} needs {key}")
    unknown = sorted(set(kwargs) - set(known))
    if unknown:
        raise ValueError(
            f"attribute {name!r}: {context} does not take {unknown}; "
            f"it takes {sorted(known)}"
        )


class MetadataMode(Mode):
    """Pure rule-based generation. No real data, no synthesizer."""

    def _build_rule(self, name: str, **kwargs: Any) -> Rule:
        if "values" in kwargs:
            _check_kwargs(name, kwargs, known=("values", "weights"), context="values")
            return Choice(kwargs["values"], kwargs.get("weights"))
        distribution = kwargs.get("distribution")
        if distribution is None and "min" in kwargs and "max" in kwargs:
            distribution = "uniform"
        if distribution == "uniform":
            _check_kwargs(
                name,
                kwargs,
                known=("distribution", "min", "max"),
                required=("min", "max"),
                context='distribution="uniform"',
            )
            return Uniform(kwargs["min"], kwargs["max"])
        if distribution == "normal":
            _check_kwargs(
                name,
                kwargs,
                known=("distribution", "mean", "sd", "min", "max"),
                required=("mean", "sd"),
                context='distribution="normal"',
            )
            return Normal(
                kwargs["mean"], kwargs["sd"], low=kwargs.get("min"), high=kwargs.get("max")
            )
        raise ValueError(
            f"attribute {name!r}: cannot resolve a rule from {sorted(kwargs)}; "
            "give min/max, mean/sd (distribution=\"normal\"), or values"
        )


class RealDataMode(Mode):
    """Real microdata the user supplies, synthesized via Empirical + CART."""

    def __init__(self, frame: pd.DataFrame, *, epsilon: float) -> None:
        super().__init__()
        self._frame = frame
        self._epsilon = epsilon
        self._real_data_epsilon: dict[str, float] = {}

    def _build_rule(self, name: str, **kwargs: Any) -> Rule:
        # Take **kwargs rather than a keyword-only `epsilon` so a stray kwarg
        # is answered by name, the way MetadataMode answers one. A narrow
        # signature would surface Python's own TypeError instead, which leaks
        # this private method's name at the caller's line.
        epsilon = kwargs.pop("epsilon", None)
        if kwargs:
            raise ValueError(
                f"attribute {name!r}: real_data mode takes only epsilon, got "
                f"{sorted(kwargs)}; the column's distribution comes from the "
                "donor frame, not from a declared rule"
            )
        if epsilon is not None:
            _check_epsilon(epsilon, f"attribute {name!r}")
        self._real_data_epsilon[name] = epsilon if epsilon is not None else self._epsilon
        return _RealDataColumn(name)

    def _extra_pipeline_kwargs(self) -> dict[str, Any]:
        if not self._real_data_epsilon:
            return {}
        return {"synthesizer": _epsilon_chain(self._real_data_epsilon, self._frame)}


class ScopeMode(Mode):
    """Real ACS PUMS population microdata for a locked-down US state,
    fetched automatically and synthesized the same way `real_data` mode is.
    """

    def __init__(self, area_code: str, *, epsilon: float) -> None:
        super().__init__()
        self.area_code = area_code
        self._epsilon = epsilon
        self._variables: dict[str, str] = {}
        self._scope_epsilon: dict[str, float] = {}
        self._fetched: pd.DataFrame | None = None

    def _build_rule(self, name: str, **kwargs: Any) -> Rule:
        # Take **kwargs rather than keyword-only parameters for the same
        # reason RealDataMode does: a narrow signature answers a stray kwarg
        # with Python's own TypeError, which names this private method rather
        # than the attribute, and contradicts GUIDE.md's promise that every
        # mode rejects an unknown keyword by name with a ValueError.
        variable = kwargs.pop("variable", None)
        epsilon = kwargs.pop("epsilon", None)
        if kwargs:
            raise ValueError(
                f"attribute {name!r}: scope mode takes only variable and "
                f"epsilon, got {sorted(kwargs)}; the column's distribution "
                "comes from the fetched ACS rows, not from a declared rule"
            )
        if variable is None:
            raise ValueError(f"attribute {name!r}: scope mode needs variable=")
        if epsilon is not None:
            _check_epsilon(epsilon, f"attribute {name!r}")
        self._variables[name] = variable
        self._scope_epsilon[name] = epsilon if epsilon is not None else self._epsilon
        return _RealDataColumn(name)

    def _extra_pipeline_kwargs(self) -> dict[str, Any]:
        if not self._variables:
            return {}
        if self._fetched is None:
            from .connectors.acs_pums import fetch_pums

            # Two attributes may name the same ACS variable (two views of one
            # source column, e.g. at different epsilons), so ask the connector
            # for each distinct code once rather than repeating it in the
            # request.
            requested = list(dict.fromkeys(self._variables.values()))
            fetched = fetch_pums(requested, state=self.area_code)
            # The mode's column names and the ACS variable codes they were
            # requested under can differ (`attribute("wage", variable="PINCP")`),
            # but the synthesizer stage matches by the mode's own column name.
            # Built per attribute name rather than renamed in place: a rename
            # map keyed on the variable would collapse those shared codes and
            # silently drop every attribute but the last one claiming it.
            self._fetched = pd.DataFrame(
                {name: fetched[variable] for name, variable in self._variables.items()}
            )
        return {"synthesizer": _epsilon_chain(self._scope_epsilon, self._fetched)}


class _RealDataColumn(_BaseRule):
    """Placeholder for a column sourced from real microdata, supplied
    (`real_data` mode) or fetched (`scope` mode).

    Sits in an `Entity`'s attributes or a `Table`'s columns like any other
    `Rule` so `entity()`/`table()` need no special casing, but its `draw()`
    is never meant to survive a real run: the `CARTSynthesizer` built by
    the owning mode's `_extra_pipeline_kwargs()` overwrites the column
    entirely once fitted on the donor frame.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def draw(self, keys, *, seed, salt, frame=None):
        return np.full(len(keys), None, dtype=object)


def _empirical_cart(columns: list[str], frame: pd.DataFrame, **knobs: Any) -> CARTSynthesizer:
    """One `CARTSynthesizer` fit against `frame` via `Empirical`.

    Shared by `RealDataMode` and `ScopeMode`: both wire real donor rows into
    the same structure-source-plus-synthesizer shape, differing only in
    where the frame came from (user-supplied vs. `fetch_pums`).

    No `tables=`, deliberately. `attribute()` names a column, and the same
    column can be carried into any number of tables declared later, so the
    mode has no table list to scope by at this point. Unscoped is the safe
    direction: the donor frame is the structure source for every table, so
    each one gets the column synthesized rather than some table silently
    keeping the placeholder.
    """
    return CARTSynthesizer(columns=columns, structure=Empirical(frame), **knobs)


def _epsilon_chain(
    epsilons: dict[str, float], frame: pd.DataFrame
) -> CARTSynthesizer | "_ChainedSynthesizer":
    """Columns grouped by their effective epsilon, one synthesizer per group.

    Shared by `RealDataMode` and `ScopeMode`: both let a caller set epsilon
    per attribute, and one tree cannot carry two generalization levels, so
    columns asking for the same level are fit together and the groups are
    chained. Only the donor frame differs between the two modes.
    """
    groups: dict[float, list[str]] = {}
    for name, epsilon in epsilons.items():
        groups.setdefault(epsilon, []).append(name)
    return _chain(
        [
            _empirical_cart(columns, frame, **_cart_knobs(epsilon))
            for epsilon, columns in groups.items()
        ]
    )


def _chain(synthesizers: list[CARTSynthesizer]) -> CARTSynthesizer | "_ChainedSynthesizer":
    """A single synthesizer's worth of stage, whether one or several groups."""
    return synthesizers[0] if len(synthesizers) == 1 else _ChainedSynthesizer(synthesizers)


def _check_epsilon(epsilon: float, where: str) -> None:
    """Reject a non-positive epsilon where the caller wrote it.

    `_cart_knobs` clamps its input, so 0 and -1 would otherwise become 0.01
    and produce the most generalized column the mapping can make. That is a
    plausible-looking result for what is really a typo, so it is caught at
    `real_data()`/`attribute()` instead: those are the lines the user wrote.
    """
    if epsilon <= 0:
        raise ValueError(f"{where}: epsilon must be positive, got {epsilon!r}")


def _cart_knobs(epsilon: float) -> dict[str, Any]:
    """`epsilon` -> `CARTSynthesizer`'s generalization knobs.

    Not differential privacy: no Laplace/Gaussian mechanism, no privacy
    accounting. Lower epsilon asks for more generalization (shallower trees,
    bigger leaves, a smaller fit sample), matching `CARTSynthesizer`'s own
    documented claim that shallow trees generalize more and disclose less.
    Monotonic in epsilon, clamped to (0, 5]. At 5 and above the knobs are
    `max_depth=None`, `min_samples_leaf=20`, `fit_cap=DEFAULT_FIT_CAP`. Depth
    and fit cap do reach `CARTSynthesizer`'s own defaults there, but the leaf
    size does not: real_data always fits on real microdata, so the most
    permissive setting still asks for leaves twenty rows wide rather than the
    library default of five. The `max(5, ...)` floor below is a guard against
    a leaf smaller than that library default; it never binds while the clamp
    ceiling is 5, since `100 / 5` is already 20.
    """
    capped = min(max(epsilon, 0.01), 5.0)
    return {
        "max_depth": None if capped >= 5.0 else max(1, round(capped * 4)),
        "min_samples_leaf": max(5, round(100 / capped)),
        "fit_cap": max(1_000, round(DEFAULT_FIT_CAP * capped / 5.0)),
    }


class _ChainedSynthesizer:
    """Applies more than one `CARTSynthesizer` to the same table in sequence.

    A `Pipeline` has one synthesizer slot; a mode with real_data columns at
    different epsilons needs one `CARTSynthesizer` per epsilon group; two
    generalization levels cannot be fit as a single tree, so this chains
    the resulting synthesizers instead.
    """

    def __init__(self, synthesizers: list[CARTSynthesizer]) -> None:
        self.synthesizers = list(synthesizers)

    def run(self, chunks, table, ctx):
        for synthesizer in self.synthesizers:
            chunks = synthesizer.run(chunks, table, ctx)
        return chunks


def _load_source(source: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source
    path = Path(source)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(
        f"{path}: real_data source must be a .csv or .parquet file, got {path.suffix!r}"
    )


class ModeSchema:
    """A schema plus the noise/pipeline wiring a mode collected for it."""

    def __init__(
        self,
        schema: Schema,
        table_noise: dict[str, dict[str, list[NoiseOp]]],
        extra_pipeline_kwargs: Callable[[], dict[str, Any]],
    ) -> None:
        self.schema = schema
        self._table_noise = dict(table_noise)
        self._build_extra_pipeline_kwargs = extra_pipeline_kwargs

    def _pipeline(self) -> Pipeline:
        noiser = Noise(self._table_noise) if self._table_noise else None
        # Called per run, not once at construction, so the mode's own
        # side effects (`ScopeMode`'s fetch) land on `.run()`. The mode
        # caches what it fetched, so a second run reuses those rows.
        return Pipeline(self.schema, noiser=noiser, **self._build_extra_pipeline_kwargs())

    def run(self) -> PipelineResult:
        return self._pipeline().run()

    def run_to(self, path: str | Path, *, format: str = "parquet") -> PipelineResult:
        return self._pipeline().run_to(path, format=format)
