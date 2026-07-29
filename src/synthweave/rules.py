"""Column rules: how a single column's values are drawn.

A rule is a pure function from a set of stable keys to an array of values.
It never holds RNG state, so the same key always yields the same value no
matter when or in what chunk it is drawn.

`Conditional` is the important one. It is how structure gets into the
no-real-data path: a generator drawing every column independently produces a
table with no inter-column relationships, and a model fitted on that has
nothing to learn. Declaring that wage depends on education puts real
structure in the data before any model sees it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from . import _hash


@runtime_checkable
class Rule(Protocol):
    """Draws one column's worth of values.

    Args:
        keys: stable per-row keys. Identical keys must yield identical values.
        seed: the run seed.
        salt: names this draw, so two columns using the same keys stay
            uncorrelated. Callers pass the column's path.
        frame: columns already drawn for these rows, or None. Only conditional
            rules need it.
    """

    def draw(
        self,
        keys: np.ndarray,
        *,
        seed: int | str,
        salt: str,
        frame: pd.DataFrame | None = None,
    ) -> np.ndarray: ...

    def depends_on(self) -> tuple[str, ...]:
        """Columns that must be drawn before this one."""
        ...


class _BaseRule:
    """Default: depends on nothing."""

    def depends_on(self) -> tuple[str, ...]:
        return ()


@dataclass(frozen=True)
class Constant(_BaseRule):
    value: Any

    def draw(self, keys, *, seed, salt, frame=None):
        return np.full(len(keys), self.value, dtype=object)


@dataclass(frozen=True)
class Choice(_BaseRule):
    """Weighted pick from a fixed set. Equal weights when `weights` is None."""

    values: Sequence[Any]
    weights: Sequence[float] | None = None

    def __post_init__(self):
        if len(self.values) == 0:
            raise ValueError("Choice needs at least one value")
        if self.weights is not None and len(self.weights) != len(self.values):
            raise ValueError(
                f"Choice got {len(self.values)} values but {len(self.weights)} weights"
            )

    def draw(self, keys, *, seed, salt, frame=None):
        w = np.asarray(self.weights, dtype=float) if self.weights is not None else None
        return _hash.pick(keys, seed, salt, np.asarray(self.values, dtype=object), w)


@dataclass(frozen=True)
class Integer(_BaseRule):
    """Uniform integers in [low, high)."""

    low: int
    high: int

    def draw(self, keys, *, seed, salt, frame=None):
        return _hash.integers(keys, seed, salt, self.low, self.high)


@dataclass(frozen=True)
class Uniform(_BaseRule):
    low: float
    high: float

    def draw(self, keys, *, seed, salt, frame=None):
        return self.low + _hash.unit(keys, seed, salt) * (self.high - self.low)


@dataclass(frozen=True)
class Normal(_BaseRule):
    mean: float
    sd: float
    low: float | None = None
    high: float | None = None

    def draw(self, keys, *, seed, salt, frame=None):
        out = _hash.normal(keys, seed, salt, self.mean, self.sd)
        if self.low is not None or self.high is not None:
            out = np.clip(out, self.low, self.high)
        return out


@dataclass(frozen=True)
class Conditional(_BaseRule):
    """Pick a sub-rule based on another column's value.

    This is how declared structure is expressed without any real data:

        Conditional("education", {
            "HS":      Normal(38_000, 9_000, low=0),
            "College": Normal(64_000, 18_000, low=0),
        }, default=Normal(30_000, 8_000, low=0))

    Every branch draws under the same salt, so a row's value does not shift
    just because it landed in a different branch than it would have under a
    different config.
    """

    on: str
    cases: Mapping[Any, Rule]
    default: Rule | None = None

    def __post_init__(self):
        if not self.cases:
            raise ValueError("Conditional needs at least one case")

    def depends_on(self) -> tuple[str, ...]:
        return (self.on,)

    def draw(self, keys, *, seed, salt, frame=None):
        if frame is None or self.on not in frame.columns:
            raise ValueError(
                f"Conditional rule needs column {self.on!r}, which has not been drawn yet"
            )
        source = frame[self.on].to_numpy()
        out = np.empty(len(keys), dtype=object)
        unmatched = np.ones(len(keys), dtype=bool)

        for case_value, rule in self.cases.items():
            mask = source == case_value
            if not mask.any():
                continue
            out[mask] = rule.draw(keys[mask], seed=seed, salt=salt, frame=_subset(frame, mask))
            unmatched &= ~mask

        if unmatched.any():
            if self.default is None:
                missing = sorted({str(v) for v in source[unmatched]})[:5]
                raise ValueError(
                    f"Conditional on {self.on!r} has no case for {missing} and no default"
                )
            out[unmatched] = self.default.draw(
                keys[unmatched], seed=seed, salt=salt, frame=_subset(frame, unmatched)
            )
        return out


@dataclass(frozen=True)
class Sequential(_BaseRule):
    """A value derived from another column by a vectorized function.

    Escape hatch for relationships the other rules do not express, such as
    deriving a birth year from an age.
    """

    on: str
    fn: Any
    extra: tuple[str, ...] = field(default_factory=tuple)

    def depends_on(self) -> tuple[str, ...]:
        return (self.on, *self.extra)

    def draw(self, keys, *, seed, salt, frame=None):
        if frame is None or self.on not in frame.columns:
            raise ValueError(f"Sequential rule needs column {self.on!r}, which is not available")
        if self.extra:
            return np.asarray(self.fn(frame[self.on], *(frame[c] for c in self.extra)))
        return np.asarray(self.fn(frame[self.on]))


def _subset(frame: pd.DataFrame | None, mask: np.ndarray) -> pd.DataFrame | None:
    return None if frame is None else frame.loc[mask]


def resolve_order(rules: Mapping[str, Rule], available: Iterable[str] = ()) -> list[str]:
    """Order columns so each rule's dependencies are drawn first.

    `available` names columns already present before any rule runs, such as
    entity attributes a table carries. A rule may depend on those without them
    needing to be ordered, which is what lets a table column be conditional on
    a carried attribute (wage conditional on the person's education).

    Raises on a cycle or a reference to a name that does not exist, since both
    are config errors the user should see before a long run starts.
    """
    present = set(available)
    order: list[str] = []
    done: set[str] = set()
    visiting: set[str] = set()

    def visit(name: str, trail: tuple[str, ...]) -> None:
        if name in done:
            return
        if name in visiting:
            cycle = " -> ".join((*trail, name))
            raise ValueError(f"column rules form a cycle: {cycle}")
        visiting.add(name)
        for dep in rules[name].depends_on():
            if dep in present:
                continue
            if dep not in rules:
                raise ValueError(
                    f"column {name!r} depends on {dep!r}, which is not available here"
                )
            visit(dep, (*trail, name))
        visiting.discard(name)
        done.add(name)
        order.append(name)

    for name in rules:
        visit(name, ())
    return order
