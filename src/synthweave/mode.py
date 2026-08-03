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

from .pipeline import Pipeline, PipelineResult
from .rules import Choice, Normal, Rule, Uniform
from .schema import Entity, Schema, Table
from .stages.noise import Missing, NoiseOp, OCR, Typo

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
