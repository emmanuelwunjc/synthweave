"""Deterministic, chunk-invariant name and SSN generation.

Optional install: `pip install "synthweave[pii]"`.

Faker's own API is call-and-advance (each call mutates internal RNG state),
which doesn't fit synthweave's contract that every value is a pure function
of `(seed, a stable key, a salt)`. So this module doesn't call Faker's
generation methods at all. For names, it reaches into Faker's underlying,
real frequency-weighted US name data and picks from it with synthweave's own
`_hash.pick` — the exact mechanism `Choice` already uses, so it's
chunk-invariant and reproducible the same way. For SSNs, Faker's own logic
is three constrained random integers (no name-frequency data involved), so
it's reimplemented natively with `_hash.integers`, matching the same
documented validity rule Faker's `ssn` provider uses (area 1-899 excluding
666, group 1-99, serial 1-9999) with no Faker dependency needed for that
part at all.

US English only (`en_US`). Other locales structure their provider data
differently (unweighted tuples instead of frequency-weighted dicts, and not
every locale's SSN-equivalent provider exists at all), so supporting them
would mean verifying each locale's data shape individually rather than
assuming they all look like en_US. Not attempted here.

Deliberately does not cover street/city/zip addresses: Faker's own address
provider composes fake, non-existent city names from templates and does not
tie a postal code to a city or state, so wrapping it would not actually
deliver the internally-consistent address this looks like it should. Real
address consistency needs a real US ZIP/city/state reference file, a
different kind of data source entirely — see `docs/NEXT_STEPS.md`.
"""

from __future__ import annotations

import math
import numbers
from typing import Any

import numpy as np
import pandas as pd

from .. import _hash

_FIELDS = ("first_name", "first_name_female", "first_name_male", "last_name")

# The Faker attributes read below are internals, not public API, so the shape
# they are expected to have is stated here and checked at pool-build time.
# Keep in step with the `Faker` bound in `pyproject.toml`.
_FAKER_SUPPORTED = "Faker>=20,<41"

_PROVIDER_ATTRS = {
    "first_name": "first_names",
    "first_name_female": "first_names_female",
    "first_name_male": "first_names_male",
    "last_name": "last_names",
}


class Name:
    """A real, US-frequency-weighted first or last name.

        sw.Entity("person", 10_000, attributes={
            "first_name": faker_names.Name("first_name"),
            "last_name": faker_names.Name("last_name"),
        })

    `which` is one of `"first_name"` (the general pool), `"first_name_female"`,
    `"first_name_male"`, or `"last_name"`.
    """

    def __init__(self, which: str = "first_name"):
        if which not in _FIELDS:
            raise ValueError(f"Name: which must be one of {_FIELDS}, got {which!r}")
        self.which = which
        values, weights = _name_pool(which)
        self._values = values
        self._weights = weights

    def depends_on(self) -> tuple[str, ...]:
        return ()

    def dtype(self):
        return None

    def draw(
        self, keys: np.ndarray, *, seed: int | str, salt: str, frame: pd.DataFrame | None = None
    ) -> np.ndarray:
        return _hash.pick(keys, seed, salt, self._values, self._weights)


class SSN:
    """A properly formatted (though not government-issued) US SSN.

    Unlike `sw.Identifier`, which is a plain hash-derived digit string with
    no format validity, this follows the real SSA structural rule: area
    1-899 excluding 666, group 1-99, serial 1-9999 (the same rule Faker's
    own `ssn` provider documents). It is still not a real SSN and carries no
    guarantee of not colliding with one; it is *format*-realistic, not
    registry-checked.
    """

    def depends_on(self) -> tuple[str, ...]:
        return ()

    def dtype(self):
        return None

    def draw(
        self, keys: np.ndarray, *, seed: int | str, salt: str, frame: pd.DataFrame | None = None
    ) -> np.ndarray:
        # 898 valid areas (1-899 minus 666). Draw over that many values and
        # shift past the gap, rather than remapping 666 onto a fixed
        # replacement: remapping would give the replacement double the
        # selection probability of every other area.
        area = _hash.integers(keys, seed, f"{salt}\x00area", 1, 899)
        area = np.where(area >= 666, area + 1, area)
        group = _hash.integers(keys, seed, f"{salt}\x00group", 1, 100)
        serial = _hash.integers(keys, seed, f"{salt}\x00serial", 1, 10000)
        return np.array(
            [f"{a:03d}-{g:02d}-{s:04d}" for a, g, s in zip(area, group, serial)], dtype=object
        )


def _name_pool(which: str) -> tuple[np.ndarray, np.ndarray]:
    """Values and weights for one of the four supported name pools.

    Imports Faker lazily and only here, so `synthweave.connectors` itself
    doesn't require the `pii` extra unless this specific module is used.
    """
    from faker.providers.person.en_US import Provider

    attr = _PROVIDER_ATTRS[which]
    pool: Any = _checked_provider_pool(Provider, attr)
    values = np.array(list(pool.keys()), dtype=object)
    weights = np.array(list(pool.values()), dtype=np.float64)
    return values, weights


def _checked_provider_pool(provider: Any, attr: str) -> dict:
    """`provider.attr`, confirmed to still be a non-empty weighted name mapping.

    `first_names` and friends are Faker internals. They can change shape in any
    release with no deprecation path, and the two bad outcomes differ: a missing
    attribute raises a bare `AttributeError` that names no cause, while a shape
    change from weighted mapping to plain sequence would keep working and
    silently drop the frequency weighting this module exists to provide. Both
    are turned into one error that names the attribute and the supported range.
    """

    def bad(problem: str) -> RuntimeError:
        return RuntimeError(
            f"faker_names: Faker's private attribute "
            f"faker.providers.person.en_US.Provider.{attr} {problem}. synthweave reads "
            f"this internal directly for deterministic, frequency-weighted picks, so a "
            f"change to its shape breaks it. Supported: {_FAKER_SUPPORTED}, installed: "
            f"{_faker_version()}."
        )

    if not hasattr(provider, attr):
        raise bad("is missing")
    pool = getattr(provider, attr)
    if not isinstance(pool, dict):
        raise bad(f"is a {type(pool).__name__}, not a name-to-weight mapping")
    if not pool:
        raise bad("is an empty mapping")
    for name, weight in pool.items():
        if not isinstance(name, str):
            raise bad(f"has a non-string name {name!r}")
        if not isinstance(weight, numbers.Real) or isinstance(weight, bool):
            raise bad(f"has a non-numeric weight {weight!r} for {name!r}")
        if not math.isfinite(weight):
            raise bad(f"has a non-finite weight {weight!r} for {name!r}")
        if not weight > 0:
            raise bad(f"has a non-positive weight {weight!r} for {name!r}")
    return pool


def _faker_version() -> str:
    """The installed Faker version, from packaging metadata rather than Faker itself.

    Reading `faker.VERSION` would be one more internal to depend on, in the one
    code path that only runs because an internal already moved.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("Faker")
    except PackageNotFoundError:
        return "unknown"
