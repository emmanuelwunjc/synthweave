"""The mode-based front door: sugar over the schema/pipeline API for a
declared shape of user, without hiding it.

`Mode` is not instantiated directly. `sw.Mode.metadata()` (and the
`real_data`/`scope` modes added on top of the same base) are the
constructors. Each holds the noise-rate bookkeeping `attribute()` collects
and the table-noise bookkeeping `table()` collects, so `schema()` can wire a
`Noise` stage without the caller ever building one by hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .pipeline import Pipeline, PipelineResult
from .rules import Choice, Normal, Rule, Uniform, _BaseRule
from .schema import Entity, Schema, Table
from .stages.noise import Missing, NoiseOp, OCR, Typo
from .stages.synthesize import DEFAULT_FIT_CAP, CARTSynthesizer, Empirical

_NOISE_RATE_KWARGS = ("missing_rate", "typo_rate", "ocr_rate")
_NOISE_OP_BY_KWARG = {"missing_rate": Missing, "typo_rate": Typo, "ocr_rate": OCR}


class Mode:
    """Shared plumbing every concrete mode reuses."""

    def __init__(self) -> None:
        self._noise_kwargs: dict[str, dict[str, float]] = {}
        self._table_noise: dict[str, dict[str, list[NoiseOp]]] = {}

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
        return RealDataMode(_load_source(source), epsilon=epsilon)

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
        identifiers: list | None = None,
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
        identifiers: list | None = None,
        coverage: float = 1.0,
    ) -> Table:
        table = Table(
            name,
            grain=grain,
            columns=columns or {},
            carry=carry,
            identifiers=identifiers or (),
            coverage=coverage,
        )
        noise = self._table_noise_for(table)
        if noise:
            self._table_noise[table.name] = noise
        return table

    def _table_noise_for(self, table: Table) -> dict[str, list[NoiseOp]]:
        carried = list(table.carry) if isinstance(table.carry, (list, tuple)) else []
        names = list(table.columns) + carried
        noise: dict[str, list[NoiseOp]] = {}
        for column_name in names:
            rates = self._noise_kwargs.get(column_name)
            if not rates:
                continue
            noise[column_name] = [
                _NOISE_OP_BY_KWARG[kwarg](rate) for kwarg, rate in rates.items()
            ]
        return noise

    def schema(self, *, entities: list[Entity], tables: list[Table], seed: int | str) -> "ModeSchema":
        schema = Schema(entities=entities, tables=tables, seed=seed)
        return ModeSchema(schema, self._table_noise, self._extra_pipeline_kwargs())

    def _extra_pipeline_kwargs(self) -> dict[str, Any]:
        return {}


class MetadataMode(Mode):
    """Pure rule-based generation. No real data, no synthesizer."""

    def _build_rule(self, name: str, **kwargs: Any) -> Rule:
        if "values" in kwargs:
            return Choice(kwargs["values"], kwargs.get("weights"))
        distribution = kwargs.get("distribution")
        if distribution is None and "min" in kwargs and "max" in kwargs:
            distribution = "uniform"
        if distribution == "uniform":
            return Uniform(kwargs["min"], kwargs["max"])
        if distribution == "normal":
            if "sd" not in kwargs:
                raise ValueError(f"attribute {name!r}: distribution=\"normal\" needs sd")
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

    def _build_rule(self, name: str, *, epsilon: float | None = None) -> Rule:
        self._real_data_epsilon[name] = epsilon if epsilon is not None else self._epsilon
        return _RealDataColumn(name)

    def _extra_pipeline_kwargs(self) -> dict[str, Any]:
        if not self._real_data_epsilon:
            return {}
        groups: dict[float, list[str]] = {}
        for name, eps in self._real_data_epsilon.items():
            groups.setdefault(eps, []).append(name)
        synthesizers = [
            CARTSynthesizer(columns=columns, structure=Empirical(self._frame), **_cart_knobs(eps))
            for eps, columns in groups.items()
        ]
        synthesizer = synthesizers[0] if len(synthesizers) == 1 else _ChainedSynthesizer(synthesizers)
        return {"synthesizer": synthesizer}


class _RealDataColumn(_BaseRule):
    """Placeholder for a real_data-sourced column.

    Sits in an `Entity`'s attributes or a `Table`'s columns like any other
    `Rule` so `entity()`/`table()` need no special casing, but its `draw()`
    is never meant to survive a real run: the `CARTSynthesizer` built by
    `RealDataMode._extra_pipeline_kwargs()` overwrites the column entirely
    once fitted on the donor frame.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def draw(self, keys, *, seed, salt, frame=None):
        return np.full(len(keys), None, dtype=object)


def _cart_knobs(epsilon: float) -> dict[str, Any]:
    """`epsilon` -> `CARTSynthesizer`'s generalization knobs.

    Not differential privacy: no Laplace/Gaussian mechanism, no privacy
    accounting. Lower epsilon asks for more generalization (shallower trees,
    bigger leaves, a smaller fit sample), matching `CARTSynthesizer`'s own
    documented claim that shallow trees generalize more and disclose less.
    Monotonic in epsilon, clamped to (0, 5]; at 5 and above every knob
    relaxes to "no limit" (`max_depth=None`, `min_samples_leaf=5`,
    `fit_cap=DEFAULT_FIT_CAP`).
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
        extra_pipeline_kwargs: dict[str, Any],
    ) -> None:
        self.schema = schema
        self._table_noise = dict(table_noise)
        self._extra_pipeline_kwargs = extra_pipeline_kwargs

    def _pipeline(self) -> Pipeline:
        from .stages.noise import Noise

        noiser = Noise(self._table_noise) if self._table_noise else None
        return Pipeline(self.schema, noiser=noiser, **self._extra_pipeline_kwargs)

    def run(self) -> PipelineResult:
        return self._pipeline().run()

    def run_to(self, path: str | Path, *, format: str = "parquet") -> PipelineResult:
        return self._pipeline().run_to(path, format=format)
