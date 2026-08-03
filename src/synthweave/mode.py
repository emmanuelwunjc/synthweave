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
from typing import Any, Sequence

from .pipeline import Pipeline, PipelineResult
from .rules import Choice, Normal, Rule, Uniform
from .schema import Entity, Schema, Table
from .stages.noise import Missing, Noise, NoiseOp, OCR, Typo

_NOISE_RATE_KWARGS = ("missing_rate", "typo_rate", "ocr_rate")
_NOISE_OP_BY_KWARG = {"missing_rate": Missing, "typo_rate": Typo, "ocr_rate": OCR}


class Mode:
    """Shared plumbing every concrete mode reuses."""

    def __init__(self) -> None:
        self._noise_kwargs: dict[str, dict[str, float]] = {}

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
        return ModeSchema(schema, self._noise_for(schema), self._extra_pipeline_kwargs())

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
        noiser = Noise(self._table_noise) if self._table_noise else None
        return Pipeline(self.schema, noiser=noiser, **self._extra_pipeline_kwargs)

    def run(self) -> PipelineResult:
        return self._pipeline().run()

    def run_to(self, path: str | Path, *, format: str = "parquet") -> PipelineResult:
        return self._pipeline().run_to(path, format=format)
