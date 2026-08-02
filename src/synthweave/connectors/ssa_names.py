"""Real first names, jointly distributed with birth year (and optionally sex).

Faker's name pool (see `faker_names.py`) is a single flat, real-frequency
list with no awareness of *when* someone was born — a 1925 birth and a 2020
birth draw from the same pool, even though "Linda" and "Olivia" belong to
very different eras in reality. This connector fixes that using the Social
Security Administration's actual national baby-name-by-year data: one real
frequency table per year, 1880-2025.

Nothing is bundled into this package, same as every other connector here.
`fetch_pums`/`geonames.py` fetch over HTTP; this one also accepts a local
`source` zip path, because `www.ssa.gov` blocks some environments' automated
requests outright (confirmed 403 on every request, including the plain
landing page, from the environment this was built in) — a real person's
browser is not blocked, so a local copy is sometimes the only way in. Ships
with no data either way; whichever path is used, it's fetched or read at
call time and cached to `.synthweave_cache/`, never bundled.

Data: SSA national baby names by year (ssa.gov/oact/babynames), public
domain (a US federal government work).

    from synthweave.connectors.ssa_names import SSAFirstName

    people = sw.Entity("person", 10_000, attributes={
        "birth_year": sw.Integer(1980, 2010),
        "sex": sw.Choice(["M", "F"]),
        "first_name": SSAFirstName(on="birth_year", sex_on="sex"),
    })
"""

from __future__ import annotations

import hashlib
import io
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .. import _hash

_URL = "https://www.ssa.gov/oact/babynames/names.zip"
_DEFAULT_CACHE_DIR = Path(".synthweave_cache") / "ssa_names"

_cache: dict[str, pd.DataFrame] = {}


class SSAFirstName:
    """A first name jointly distributed with birth year (and optionally sex).

    Args:
        on: the schema's column name holding birth year. Must already be
            drawn (an `sw.Integer`/similar attribute), since this rule reads
            it from the frame rather than drawing it itself.
        sex_on: optional column name holding sex, as SSA's own `"M"`/`"F"`
            codes. If given, names are drawn from that year's *and* that
            sex's real distribution; if omitted, both sexes' names for that
            year are pooled together.
        source: a local `names.zip` path (SSA's own bundle format,
            `yob<year>.txt` files inside), for environments where
            `www.ssa.gov` is unreachable. `None` (default) fetches live.
    """

    def __init__(
        self,
        on: str = "birth_year",
        sex_on: str | None = None,
        source: str | Path | None = None,
        cache_dir: str | Path | None = _DEFAULT_CACHE_DIR,
    ):
        self.on = on
        self.sex_on = sex_on
        self._data = _ssa_data(source, cache_dir)
        self._min_year = int(self._data["year"].min())
        self._max_year = int(self._data["year"].max())

    def depends_on(self) -> tuple[str, ...]:
        deps = (self.on,)
        return deps + (self.sex_on,) if self.sex_on else deps

    def dtype(self):
        return None

    def draw(
        self, keys: np.ndarray, *, seed, salt: str, frame: pd.DataFrame | None = None
    ) -> np.ndarray:
        if frame is None or self.on not in frame.columns:
            raise ValueError(f"SSAFirstName rule needs column {self.on!r}, which is not available")
        if self.sex_on is not None and self.sex_on not in frame.columns:
            raise ValueError(
                f"SSAFirstName rule needs column {self.sex_on!r}, which is not available"
            )

        years = frame[self.on].to_numpy()
        # Check for missing years before the range check, not after: NaN is
        # False for both `< min` and `> max`, so it passes the range guard,
        # and then False for `== year` in the grouping loop below, so the row
        # matches no group and its slot in the output array is never written.
        # The result was an uninitialised None in the output rather than any
        # error, with nothing pointing at the missing input that caused it.
        missing = pd.isna(years)
        if missing.any():
            raise ValueError(
                f"SSAFirstName: {self.on!r} has {int(missing.sum())} missing value(s), which "
                "cannot select a birth-year cohort. Fill or drop them before synthesizing, "
                "e.g. with a Conditional rule or by filtering the frame."
            )
        bad = years[(years < self._min_year) | (years > self._max_year)]
        if len(bad):
            raise ValueError(
                f"SSAFirstName: {self.on!r} has year(s) outside the data's "
                f"{self._min_year}-{self._max_year} range, e.g. {sorted(set(bad))[:3]}"
            )
        sexes = frame[self.sex_on].to_numpy() if self.sex_on else None

        out = np.empty(len(keys), dtype=object)
        groups = zip(years, sexes) if sexes is not None else zip(years, [None] * len(years))
        for year, sex in set(groups):
            mask = years == year
            if sex is not None:
                mask = mask & (sexes == sex)
            if not mask.any():
                continue
            values, weights = self._pool(year, sex)
            out[mask] = _hash.pick(keys[mask], seed, salt, values, weights)
        return out

    def _pool(self, year, sex) -> tuple[np.ndarray, np.ndarray]:
        subset = self._data[self._data["year"] == year]
        if sex is not None:
            subset = subset[subset["sex"] == sex]
        if subset.empty:
            raise ValueError(f"SSAFirstName: no data for year={year!r} sex={sex!r}")
        pooled = subset.groupby("name", sort=False)["count"].sum()
        return pooled.index.to_numpy(dtype=object), pooled.to_numpy(dtype=np.float64)


def _cache_filename(source: str | Path | None) -> str:
    """Cache filename for one source, distinct per source.

    The name used to be a constant, with `source` present only in the
    in-memory memo key. A second `SSAFirstName` pointing at a different local
    zip therefore found the first source's file already on disk and silently
    returned its data, never reading the file it was given. The same
    collision hit switching between a local source and a live fetch.

    A live fetch keeps the original plain name, so existing caches stay
    valid. A local source is hashed rather than slugged, because two paths
    can share a basename and the full path is not filename-safe.
    """
    if source is None:
        return "ssa_names.csv"
    digest = hashlib.sha256(str(Path(source).resolve()).encode()).hexdigest()[:16]
    return f"ssa_names.{digest}.csv"


def _ssa_data(source: str | Path | None, cache_dir: str | Path | None) -> pd.DataFrame:
    memo_key = f"{source}|{cache_dir}"
    if memo_key in _cache:
        return _cache[memo_key]

    cache_path = None if cache_dir is None else Path(cache_dir) / _cache_filename(source)
    if cache_path is not None and cache_path.exists():
        frame = pd.read_csv(cache_path)
        _cache[memo_key] = frame
        return frame

    body = _read_local(source) if source is not None else _fetch()
    frame = _parse(body)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache_path, index=False)
    _cache[memo_key] = frame
    return frame


def _read_local(source: str | Path) -> bytes:
    path = Path(source)
    if not path.exists():
        raise RuntimeError(f"SSAFirstName: source file not found: {path}")
    return path.read_bytes()


def _fetch() -> bytes:
    try:
        with urllib.request.urlopen(_URL, timeout=60) as response:
            if response.status != 200:
                raise RuntimeError(f"SSA names request failed with HTTP {response.status}: {_URL}")
            return response.read()
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"SSA names request failed: {_URL} ({e}). If this environment can't reach "
            f"ssa.gov, download names.zip in a browser and pass source=<path> instead."
        ) from e


def _parse(body: bytes) -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        for member in archive.namelist():
            if not (member.startswith("yob") and member.endswith(".txt")):
                continue
            year = int(member[3:7])
            text = archive.read(member).decode("utf-8")
            for line in text.splitlines():
                if not line:
                    continue
                name, sex, count = line.split(",")
                rows.append((year, name, sex, int(count)))
    if not rows:
        raise RuntimeError("SSA names archive produced no yob*.txt rows")
    return pd.DataFrame(rows, columns=["year", "name", "sex", "count"])
