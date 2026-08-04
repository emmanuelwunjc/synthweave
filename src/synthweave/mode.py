"""The mode-based front door: sugar over the schema/pipeline API for a
declared shape of user, without hiding it.

`Mode` is not instantiated directly. `sw.Mode.metadata()` (and the
`real_data`/`scope` modes added on top of the same base) are the
constructors. Each holds the noise-rate bookkeeping `attribute()` collects,
so `schema()` can wire a `Noise` stage without the caller ever building one
by hand.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .pipeline import Pipeline, PipelineResult
from .provenance import user
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

        Giving two attributes different epsilons does not decorrelate them.
        Each epsilon group is fit as its own tree, but every group after the
        first conditions on the columns the earlier groups produced, so the
        donor's joint structure between differently-generalized attributes
        survives. It survived nothing before #139: the groups were drawn
        independently, so mixed epsilons kept every marginal and destroyed
        every relationship, without a warning.
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
        overrides it per attribute, and mixing epsilons keeps the joint
        structure between the attributes for the same reason `real_data`
        does: later epsilon groups condition on the earlier ones.
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
        # The noise map and the real-column placement resolve now: an unmatched
        # name is a declaration mistake either way, so it should surface on the
        # call that declared it. The pipeline builder goes across uncalled.
        # `ScopeMode` fetches real rows off the network inside it, and a fetch
        # (or a ValueError for an area code the connector does not recognize)
        # belongs on `.run()`, not on a call that otherwise only assembles
        # objects.
        placement = _placement(self._declared_columns(), schema)
        return ModeSchema(
            schema,
            self._noise_for(schema),
            partial(self._extra_pipeline_kwargs, placement),
        )

    def _declared_columns(self) -> Sequence[str]:
        """Columns this mode synthesizes from real donor rows, in declaration order.

        Empty for `MetadataMode`, which has no donor and no synthesizer.
        """
        return ()

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

    def _extra_pipeline_kwargs(self, placement: dict[str, list[str]]) -> dict[str, Any]:
        return {}


def _placement(names: Sequence[str], schema: Schema) -> dict[str, list[str]]:
    """Which of `names` each table actually declares, keyed by table name.

    Two silent failures live here, and both are the same shape as the noise
    one `_noise_for` already catches: a name the user wrote that nothing in the
    schema answers to.

    - A real column used to be synthesized into *every* table, because the
      synthesizer was built unscoped and `_FittedCART.apply` creates a declared
      column on a chunk that lacks it rather than skipping. A table carrying
      nothing from the donor came back holding every mode attribute (#144).
    - An attribute nobody carried was synthesized anyway, into all of them.

    Returned in declaration order per table, which is the order `_epsilon_chain`
    then conditions in.
    """
    names = list(names)
    if not names:
        return {}
    _check_bound_names(schema)
    placement: dict[str, list[str]] = {}
    matched: set[str] = set()
    for table in schema.tables:
        declared = set(table.carry) | set(table.columns)
        columns = [name for name in names if name in declared]
        if columns:
            placement[table.name] = columns
            matched.update(columns)
    unmatched = sorted(set(names) - matched)
    if unmatched:
        raise ValueError(
            f"real columns were declared for {unmatched}, but no table carries or "
            "generates a column by that name. They used to be synthesized into "
            "every table anyway, which puts donor-derived values in an export "
            "nobody asked for; carry them somewhere, or drop the attribute"
        )
    return placement


def _check_bound_names(schema: Schema) -> None:
    """Reject a real column bound under a name other than its own.

    The mode keys on the name `attribute()` was called with; the schema keys on
    the name the rule was bound to. When they disagree the user's column comes
    back entirely null and a phantom column holding the real-derived values
    appears beside it, with nothing raised (#144). `real_data` mode has no
    rename channel at all, and `scope` mode's is `variable=`, so the fix is
    always to declare the attribute under the name it is bound to.
    """
    bindings = [
        (bound, rule)
        for entity in schema.entities
        for bound, rule in entity.attributes.items()
    ] + [
        (bound, rule) for table in schema.tables for bound, rule in table.columns.items()
    ]
    for bound, rule in bindings:
        if isinstance(rule, _RealDataColumn) and rule.name != bound:
            raise ValueError(
                f"attribute {rule.name!r} is bound as {bound!r}. The mode keys real "
                f"columns on the name attribute() was called with, so {bound!r} would "
                f"come back all null beside a {rule.name!r} column you never declared. "
                f"Declare it as attribute({bound!r}) in real_data mode, or as "
                f"attribute({bound!r}, variable=...) in scope mode"
            )


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

    def _declared_columns(self) -> Sequence[str]:
        return list(self._real_data_epsilon)

    def _extra_pipeline_kwargs(self, placement: dict[str, list[str]]) -> dict[str, Any]:
        if not self._real_data_epsilon:
            return {}
        return {
            "synthesizer": _epsilon_chain(self._real_data_epsilon, self._frame, placement)
        }


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

    def _build_rule(
        self, name: str, *, variable: str | None = None, epsilon: float | None = None
    ) -> Rule:
        if variable is None:
            raise ValueError(f"attribute {name!r}: scope mode needs variable=")
        if epsilon is not None:
            _check_epsilon(epsilon, f"attribute {name!r}")
        self._variables[name] = variable
        self._scope_epsilon[name] = epsilon if epsilon is not None else self._epsilon
        return _RealDataColumn(name)

    def _declared_columns(self) -> Sequence[str]:
        return list(self._variables)

    def _extra_pipeline_kwargs(self, placement: dict[str, list[str]]) -> dict[str, Any]:
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
        return {
            "synthesizer": _epsilon_chain(self._scope_epsilon, self._fetched, placement)
        }


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

    `tables=` is always passed, and always names exactly one table. It was
    unscoped once, on the reasoning that a column can be carried into any
    number of tables declared later, so the mode had no table list at that
    point. It does now: `schema()` resolves the placement before the pipeline
    is built. Unscoped was not the safe direction, because
    `_FittedCART.apply` *creates* a declared column on a chunk that lacks it,
    so every table got every mode attribute whether or not it declared one
    (#144).
    """
    return CARTSynthesizer(columns=columns, structure=Empirical(frame), **knobs)


def _epsilon_chain(
    epsilons: dict[str, float], frame: pd.DataFrame, placement: dict[str, list[str]]
) -> CARTSynthesizer | "_ChainedSynthesizer":
    """Columns grouped by their effective epsilon, one synthesizer per group.

    Shared by `RealDataMode` and `ScopeMode`: both let a caller set epsilon
    per attribute, and one tree cannot carry two generalization levels, so
    columns asking for the same level are fit together and the groups are
    chained. Only the donor frame differs between the two modes.

    Each group after the first conditions on every column the earlier groups
    already produced, via `predictors=`. Without that the groups were drawn
    independently and two attributes given different epsilons came out
    uncorrelated: every univariate summary of the output stayed right and
    every bivariate one was destroyed, silently (#139). Groups run in the
    order their first attribute was declared, so the conditioning order is
    the order the user wrote rather than an accident of float ordering.

    One synthesizer per (table, epsilon group) rather than per group: a group's
    columns need not all land on the same table, and a synthesizer carries one
    `tables=` scope and one predictor list. Splitting by table is what keeps a
    column out of a table that never declared it (#144) and keeps the
    predictors to columns that table actually has.
    """
    groups: dict[float, list[str]] = {}
    for name, epsilon in epsilons.items():
        groups.setdefault(epsilon, []).append(name)
    synthesizers = []
    for table_name, table_columns in placement.items():
        conditioned: list[str] = []
        for epsilon, columns in groups.items():
            here = [column for column in columns if column in table_columns]
            if not here:
                continue
            synthesizers.append(
                _empirical_cart(
                    here,
                    frame,
                    tables=[table_name],
                    predictors=list(conditioned),
                    label=_group_label(epsilon),
                    **_cart_knobs(epsilon),
                )
            )
            conditioned += here
    return _chain(synthesizers)


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
    # Tagged as user-provided, not left a plain int. `CARTSynthesizer` reads a
    # plain int leaf size as its own library default, so an epsilon-derived one
    # arrived in `unjustified()` as an undefended magic number -- the one value
    # in the run that traces straight back to something the user chose (#145).
    return {
        "max_depth": None if capped >= 5.0 else max(1, round(capped * 4)),
        "min_samples_leaf": user(
            max(5, round(100 / capped)), f"derived from epsilon={epsilon!r}"
        ),
        "fit_cap": max(1_000, round(DEFAULT_FIT_CAP * capped / 5.0)),
    }


def _group_label(epsilon: float) -> str:
    """The provenance/report label one epsilon group's synthesizer runs under.

    Several of them reach the same table, and everything `CARTSynthesizer.run`
    records keys on the table name, so without a label each group overwrote the
    last and the audit trail showed one generalization level where two were
    applied (#145). The epsilon is the label because it is what distinguishes
    the groups, and it is what a reader of the record needs to see.
    """
    return f"eps{epsilon:g}"


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
