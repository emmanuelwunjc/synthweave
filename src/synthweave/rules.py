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

    # A rule may also define `dtype() -> np.dtype | None`, naming the type its
    # column holds. Declared rather than inferred, because inference is chunk
    # dependent: a chunk of whole numbers looks like an integer column and the
    # next one looks like a float, so the type would follow `chunk_size`.
    #
    # It is deliberately not a member of this Protocol. `Rule` is
    # runtime_checkable, and adding it here would make `isinstance` reject
    # every rule written before it existed, which is the opposite of optional.
    # Read it with `declared_dtype`, which treats absence as "no declaration".


class _BaseRule:
    """Default: depends on nothing, declares no type."""

    def depends_on(self) -> tuple[str, ...]:
        return ()

    def dtype(self) -> np.dtype | None:
        return None


@dataclass(frozen=True)
class Constant(_BaseRule):
    value: Any

    def draw(self, keys, *, seed, salt, frame=None):
        return np.full(len(keys), self.value, dtype=object)

    def dtype(self):
        return _scalar_dtype(self.value)


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

    def dtype(self):
        return _widest_dtype([_scalar_dtype(v) for v in self.values])


def coerce_rule(value: Any) -> Rule:
    """A `Rule` as given, or inferred from a plain Python value.

    A `list` or `tuple` becomes `Choice(values)` (equal weight). A single
    `int`/`float`/`str`/`bool` becomes `Constant(value)`. Anything else
    raises, naming the rule types to reach for instead.

    Deliberately does not try to turn a bare pair of numbers into a range.
    `"wage": (38_000, 9_000)` reads like `Normal(mean=38_000, sd=9_000)` to a
    person and like `Integer(low=38_000, high=9_000)` (empty, since
    `low > high`) to a naive 2-tuple-means-range rule. Guessing between those
    is exactly the kind of silent-wrong-output bug worth never introducing,
    so a range always needs `sw.Integer`/`sw.Uniform` written explicitly.
    """
    if isinstance(value, Rule):
        return value
    # A namedtuple is a tuple, but it is a record, not a list of alternatives.
    # The plain-tuple branch below would turn Point(1, 2) into "pick 1 or 2
    # with equal odds", which nobody handing over a structured value means.
    # `_fields` is the standard way to tell one from a plain tuple.
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        raise TypeError(
            f"{type(value).__name__}{tuple(value)!r} is a namedtuple, which names fields "
            "rather than listing alternatives, so there is no one obvious rule it means. "
            "Say it explicitly: sw.Choice([...]) to pick between its values, or a rule "
            "per field if the fields are separate columns."
        )
    if isinstance(value, (list, tuple)):
        return Choice(list(value))
    # A numpy scalar means the same thing a Python one does, and pulling a
    # value out of a frame or array is the ordinary way to get one. Which of
    # them worked used to be an accident of numpy's type hierarchy: np.float64
    # subclasses float and so passed the branch below, np.int64 does not
    # subclass int and fell through to an error suggesting sw.Integer(low,
    # high), which is not the fix for a single fixed value. Unwrapped to a
    # plain Python scalar so the column does not inherit a surprising dtype.
    if isinstance(value, np.generic):
        return Constant(value.item())
    if isinstance(value, (int, float, str, bool)):
        return Constant(value)
    raise TypeError(
        f"{value!r} is not a Rule and cannot be inferred automatically. "
        "Wrap it explicitly: sw.Integer(low, high), sw.Uniform(low, high), "
        "sw.Normal(mean, sd), sw.Choice(values, weights), or sw.Constant(value)."
    )


@dataclass(frozen=True)
class Integer(_BaseRule):
    """Uniform integers in [low, high)."""

    low: int
    high: int

    def draw(self, keys, *, seed, salt, frame=None):
        return _hash.integers(keys, seed, salt, self.low, self.high)

    def dtype(self):
        return np.dtype(np.int64)


@dataclass(frozen=True)
class Uniform(_BaseRule):
    low: float
    high: float

    def __post_init__(self):
        if self.high <= self.low:
            raise ValueError(f"Uniform needs high > low, got low={self.low} high={self.high}")

    def draw(self, keys, *, seed, salt, frame=None):
        return self.low + _hash.unit(keys, seed, salt) * (self.high - self.low)

    def dtype(self):
        return np.dtype(np.float64)


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

    def dtype(self):
        return np.dtype(np.float64)


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
        object.__setattr__(
            self, "cases", {k: coerce_rule(v) for k, v in self.cases.items()}
        )
        if self.default is not None:
            object.__setattr__(self, "default", coerce_rule(self.default))

    def depends_on(self) -> tuple[str, ...]:
        return (self.on,)

    def dtype(self):
        """Resolved from the branches, so it cannot depend on which branch a
        chunk happened to contain. An integer branch beside a float branch
        gives float, and any branch that declares nothing gives nothing."""
        branches = list(self.cases.values()) + ([self.default] if self.default else [])
        return _widest_dtype([declared_dtype(rule) for rule in branches])

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


def _scalar_dtype(value: Any) -> np.dtype | None:
    """A numeric or boolean value's dtype, or None for anything else.

    Text and mixed values stay object. Narrowing those would gain nothing and
    would risk changing a value on the way.
    """
    dtype = np.asarray([value]).dtype
    return dtype if dtype.kind in "ifb" else None


def _widest_dtype(declared: list[np.dtype | None]) -> np.dtype | None:
    """The one type that holds all of them, or None if any declines to say.

    The `is None` test is deliberate. `np.dtype("float64") == None` is True,
    because numpy reads None as its default float dtype, so `None in declared`
    silently answers yes for any float column.
    """
    if not declared or any(dtype is None for dtype in declared):
        return None
    return np.result_type(*declared)


def declared_dtype(rule: Rule) -> np.dtype | None:
    """What a rule says its column holds.

    Read through `getattr` because `dtype` is an optional part of the protocol.
    A custom rule written before it existed simply declares nothing, and its
    column is left exactly as it was drawn.
    """
    declare = getattr(rule, "dtype", None)
    return declare() if callable(declare) else None


def as_declared(rule: Rule, values: np.ndarray) -> np.ndarray:
    """Values in the type their rule declares."""
    dtype = declared_dtype(rule)
    return values if dtype is None else values.astype(dtype)


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


class RuleConformanceError(AssertionError):
    """A custom Rule broke one of the two contracts `check_rule` verifies."""


def check_rule(
    rule: Rule,
    *,
    n: int = 64,
    seed: int | str = 0,
    salt: str = "check_rule",
    frame: pd.DataFrame | None = None,
) -> None:
    """Verify a custom `Rule` honours its two contracts.

    `Rule.draw`'s docstring promises "identical keys must yield identical
    values" regardless of when or in what chunk they are drawn. Nothing
    checks that promise today: `Rule` is a `runtime_checkable` Protocol
    matched on `draw`/`depends_on` alone, so a rule keyed on row position
    (`np.arange(len(keys))`) or reaching for `random.random()` passes
    `isinstance` and only breaks determinism once a run is already chunked.

    Draws a small key array three ways and requires the same values back:
    called twice (determinism), with the keys shuffled (values must follow
    the key, not its position), and split into two calls concatenated
    (chunk invariance). Raises `RuleConformanceError` naming which contract
    broke and showing the first differing keys and values, or returns
    `None` if the rule passes all three.

    `frame` is only needed for a rule that depends on other columns (e.g.
    `Conditional`); it is sliced alongside `keys` for the shuffled and split
    calls, so its rows must line up with `keys` positionally.
    """
    keys = np.array([f"check_rule:{i}" for i in range(n)], dtype=object)

    baseline = rule.draw(keys, seed=seed, salt=salt, frame=frame)

    repeat = rule.draw(keys, seed=seed, salt=salt, frame=frame)
    _assert_same_values(
        keys, baseline, repeat, "calling draw() twice with identical input gave different "
        "values back (the rule is not deterministic, e.g. it reaches for random state)"
    )

    order = np.arange(n)[::-1]
    shuffled_keys = keys[order]
    shuffled = rule.draw(shuffled_keys, seed=seed, salt=salt, frame=_reorder_frame(frame, order))
    by_key = dict(zip(keys, baseline))
    expected = np.array([by_key[k] for k in shuffled_keys], dtype=baseline.dtype)
    _assert_same_values(
        shuffled_keys, expected, shuffled, "shuffling the key array changed a key's value "
        "(the rule depends on position or order, not on the key itself)"
    )

    split = n // 2
    first = rule.draw(keys[:split], seed=seed, salt=salt, frame=_reorder_frame(frame, slice(0, split)))
    second = rule.draw(
        keys[split:], seed=seed, salt=salt, frame=_reorder_frame(frame, slice(split, None))
    )
    combined = np.concatenate([first, second])
    _assert_same_values(
        keys, baseline, combined, "splitting the keys across two calls changed a value "
        "(the rule is not chunk invariant, e.g. it reads chunk-level state)"
    )


def _reorder_frame(frame, index):
    return None if frame is None else frame.iloc[index].reset_index(drop=True)


def _assert_same_values(keys, expected, actual, reason: str) -> None:
    mismatched = [
        (k, e, a) for k, e, a in zip(keys, expected, actual) if not _values_equal(e, a)
    ]
    if mismatched:
        shown = ", ".join(f"{k!r}: expected {e!r}, got {a!r}" for k, e, a in mismatched[:3])
        raise RuleConformanceError(
            f"{reason}. {len(mismatched)} of {len(keys)} key(s) differed, e.g. {shown}"
        )


def _values_equal(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b
