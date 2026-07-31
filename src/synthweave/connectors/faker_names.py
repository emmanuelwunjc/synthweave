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

from typing import Any

import numpy as np
import pandas as pd

from .. import _hash

_FIELDS = ("first_name", "first_name_female", "first_name_male", "last_name")


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
        area = _hash.integers(keys, seed, f"{salt}\x00area", 1, 900)
        area = np.where(area == 666, 667, area)
        group = _hash.integers(keys, seed, f"{salt}\x00group", 1, 100)
        serial = _hash.integers(keys, seed, f"{salt}\x00serial", 1, 10000)
        return np.array(
            [f"{a:03d}-{g:02d}-{s:04d}" for a, g, s in zip(area, group, serial)], dtype=object
        )


def _name_pool(which: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Values and weights for one of the four supported name pools.

    Imports Faker lazily and only here, so `synthweave.connectors` itself
    doesn't require the `pii` extra unless this specific module is used.
    """
    from faker.providers.person.en_US import Provider

    attr = {
        "first_name": "first_names",
        "first_name_female": "first_names_female",
        "first_name_male": "first_names_male",
        "last_name": "last_names",
    }[which]
    pool: Any = getattr(Provider, attr)
    if isinstance(pool, dict):
        values = np.array(list(pool.keys()), dtype=object)
        weights = np.array(list(pool.values()), dtype=np.float64)
    else:
        values = np.array(list(pool), dtype=object)
        weights = None
    return values, weights
