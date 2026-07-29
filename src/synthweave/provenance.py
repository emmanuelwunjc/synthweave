"""Provenance tagging for config values.

Every value that shapes the output carries a tag saying where it came from.
The point is that a reader of a config can tell apart a number the user chose,
a number the library defaulted to, and a number taken from a published source.
Without this, a config is a wall of magic numbers and nobody can tell which
ones were ever justified.

Three origins, deliberately few:

- `user-provided`: the user set this explicitly.
- `modeled`: a library default or an assumption standing in for real knowledge.
  This is the one to audit. `PipelineResult.unjustified()` lists exactly these.
- `cited`: taken from a source, which is recorded in `note`.

This is a reduced form of the multi-registry approach used in EdSim (which
separates open placeholders, cited figures, deferred simplifications, and
shipped simplifications). One tag is cheap to add now and expensive to
retrofit onto user configs later, so it ships in v0.1; the finer split can
come once usage shows it is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Iterable, Literal, TypeVar

T = TypeVar("T")

Origin = Literal["user-provided", "modeled", "cited"]
ORIGINS: tuple[Origin, ...] = ("user-provided", "modeled", "cited")


@dataclass(frozen=True)
class Tagged(Generic[T]):
    """A config value plus where it came from."""

    value: T
    origin: Origin = "user-provided"
    note: str | None = None

    def __post_init__(self) -> None:
        if self.origin not in ORIGINS:
            raise ValueError(f"unknown origin {self.origin!r}; expected one of {ORIGINS}")
        if self.origin == "cited" and not self.note:
            raise ValueError("a cited value must carry a note naming the source")

    def __repr__(self) -> str:
        suffix = f", {self.note!r}" if self.note else ""
        return f"Tagged({self.value!r}, {self.origin!r}{suffix})"


def user(value: T, note: str | None = None) -> Tagged[T]:
    """A value the user chose."""
    return Tagged(value, "user-provided", note)


def modeled(value: T, note: str | None = None) -> Tagged[T]:
    """An assumption or library default. Shows up in `unjustified()`."""
    return Tagged(value, "modeled", note)


def cited(value: T, source: str) -> Tagged[T]:
    """A value from a named source. `source` should be traceable, e.g. a URL."""
    return Tagged(value, "cited", source)


def as_tagged(value: Any, default_origin: Origin = "user-provided") -> Tagged:
    """Accept either a raw value or an already-Tagged one.

    Lets users write `coverage=0.8` instead of `coverage=user(0.8)` everywhere,
    while library-supplied defaults pass `default_origin="modeled"` so they are
    correctly flagged as unjustified.
    """
    if isinstance(value, Tagged):
        return value
    return Tagged(value, default_origin)


def unwrap(value: Any) -> Any:
    """The underlying value, whether or not it is tagged."""
    return value.value if isinstance(value, Tagged) else value


@dataclass
class ProvenanceRecord:
    """Every tagged value in a run, addressed by a dotted config path."""

    entries: dict[str, Tagged] = field(default_factory=dict)

    def add(self, path: str, value: Any, default_origin: Origin = "user-provided") -> Any:
        """Record a value at `path` and return the raw value for immediate use."""
        tagged = as_tagged(value, default_origin)
        self.entries[path] = tagged
        return tagged.value

    def of_origin(self, origin: Origin) -> dict[str, Tagged]:
        return {p: t for p, t in self.entries.items() if t.origin == origin}

    def unjustified(self) -> dict[str, Tagged]:
        """Values that are library defaults or bare assumptions.

        The audit list: everything here is a number nobody has defended yet.
        """
        return self.of_origin("modeled")

    def to_frame(self):
        """The record as a DataFrame, for a methods section or an appendix."""
        import pandas as pd

        rows: Iterable[dict[str, Any]] = (
            {"path": p, "value": t.value, "origin": t.origin, "note": t.note}
            for p, t in sorted(self.entries.items())
        )
        return pd.DataFrame(rows, columns=["path", "value", "origin", "note"])

    def __len__(self) -> int:
        return len(self.entries)
