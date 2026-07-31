"""Real, internally consistent US postal addresses (ZIP + city + state).

Solves a specific gap Faker doesn't: Faker's `city()` composes fake,
non-existent city names from templates, takes no state argument, and never
ties a postal code to a city, so wrapping it wouldn't have delivered real
address consistency. This module fetches GeoNames' real US postal code
table instead — ZIP, city, and state all come from the same real row, so
they're consistent by construction.

Nothing is bundled into this package. Like `acs_pums.fetch_pums`, the
reference table is fetched over HTTP on first use and cached locally
(`.synthweave_cache/` by default); the installed package carries none of it.
Bundling real, US-specific reference data into `src/synthweave/` would make
the core package itself US-specific, against synthweave's own
"generic, no-real-data-required" design.

Data: GeoNames postal code dataset (download.geonames.org), licensed
CC BY 4.0. Using it requires attribution to GeoNames
(https://www.geonames.org) in anything built on this data.

    from synthweave.connectors.geonames import USAddress

    people = sw.Entity("person", 10_000, attributes={
        "city": USAddress("city"),
        "state": USAddress("state_abbr"),
        "postal_code": USAddress("postal_code"),
    })
    # Same person -> same reference row -> city/state/zip actually agree.
"""

from __future__ import annotations

import csv
import io
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .. import _hash

_URL = "https://download.geonames.org/export/zip/US.zip"
_MEMBER = "US.txt"
_DEFAULT_CACHE_DIR = Path(".synthweave_cache") / "geonames"
_COLUMNS = (
    "country_code",
    "postal_code",
    "city",
    "state",
    "state_abbr",
    "county",
    "county_code",
    "admin_name3",
    "admin_code3",
    "latitude",
    "longitude",
    "accuracy",
)
_FIELDS = ("city", "state", "state_abbr", "postal_code", "county", "latitude", "longitude")

_cache: dict[str, pd.DataFrame] = {}


class USAddress:
    """One field of a real, internally consistent US postal address.

    `field` is one of `"city"`, `"state"`, `"state_abbr"`, `"postal_code"`,
    `"county"`, `"latitude"`, `"longitude"`.

    `group` ties related fields to the same underlying reference row, the
    way `Identifier`'s `tag` ties independent identifier streams apart. Two
    `USAddress` rules sharing a `group` (the default, `"address"`) always
    resolve to the same real GeoNames row for a given entity, so
    `USAddress("city")` and `USAddress("state_abbr")` used together describe
    one real place. A second, independent address on the same entity (e.g.
    a work address distinct from a home address) needs its own `group`, or
    the two would always coincide.
    """

    def __init__(self, field: str, group: str = "address", cache_dir: str | Path | None = _DEFAULT_CACHE_DIR):
        if field not in _FIELDS:
            raise ValueError(f"USAddress: field must be one of {_FIELDS}, got {field!r}")
        self.field = field
        self.group = group
        self._data = _postal_data(cache_dir)

    def depends_on(self) -> tuple[str, ...]:
        return ()

    def dtype(self):
        return None

    def draw(
        self, keys: np.ndarray, *, seed, salt: str, frame: pd.DataFrame | None = None
    ) -> np.ndarray:
        # Deliberately not `salt` (which is per-column, e.g. differs between
        # "city" and "state_abbr" draws): every field in the same `group`
        # must select the identical row, so the row-selection salt is fixed
        # to the group name instead of the column being drawn.
        idx = _hash.integers(keys, seed, f"usaddress\x00{self.group}", 0, len(self._data))
        return self._data[self.field].to_numpy(dtype=object)[idx]


def _postal_data(cache_dir: str | Path | None) -> pd.DataFrame:
    memo_key = "<none>" if cache_dir is None else str(Path(cache_dir))
    if memo_key in _cache:
        return _cache[memo_key]

    cache_path = None if cache_dir is None else Path(cache_dir) / "us_postal.csv"
    if cache_path is not None and cache_path.exists():
        frame = pd.read_csv(cache_path, dtype=str)
        _cache[memo_key] = frame
        return frame

    frame = _fetch_and_parse()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache_path, index=False)
    _cache[memo_key] = frame
    return frame


def _fetch_and_parse() -> pd.DataFrame:
    try:
        with urllib.request.urlopen(_URL, timeout=60) as response:
            if response.status != 200:
                raise RuntimeError(f"GeoNames request failed with HTTP {response.status}: {_URL}")
            body = response.read()
    except urllib.error.URLError as e:
        raise RuntimeError(f"GeoNames request failed: {_URL} ({e})") from e

    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        raw = archive.read(_MEMBER).decode("utf-8")

    rows = list(csv.reader(io.StringIO(raw), delimiter="\t"))
    frame = pd.DataFrame(rows, columns=_COLUMNS)
    if frame.empty:
        raise RuntimeError(f"GeoNames archive at {_URL} produced no rows")
    return frame
