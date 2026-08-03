"""Real last names, jointly distributed with a race/ethnicity attribute.

The Census Bureau's 2010 surname file reports, per surname, the share of
people carrying it who reported each race/ethnicity category in the census
(`pctwhite`, `pctblack`, `pctapi`, `pctaian`, `pct2prace`, `pcthispanic`) and
the surname's total count. That's `P(race | surname)`, the reverse of what's
needed to draw a surname given a declared race. This connector inverts it
the standard way: `weight(surname, race) = count(surname) * pct[race](surname)`,
the expected number of real people with that surname reporting that race —
a defensible, commonly used approximation, not an exact P(surname | race).

Nothing is bundled into this package, same as every other connector here:
fetched over HTTP and cached locally on first use.

Data: US Census Bureau, Frequently Occurring Surnames from the 2010 Census
(www2.census.gov/topics/genealogy/2010surnames), a US federal government
work, public domain. Cells with too few people are marked `(S)` (suppressed
for privacy) in the source and treated as 0 here, which slightly
underweights rare surname/race combinations rather than guessing at them.

The five Census categories are not a match for every schema's own race/
ethnicity vocabulary (they're also not mutually exclusive with each other in
Census's own methodology — `pcthispanic` in particular overlaps with the
others). `categories` is deliberately a required mapping, not a guessed
default, so that alignment is explicit rather than silently assumed:

    from synthweave.connectors.census_surnames import Surname

    people = sw.Entity("person", 10_000, attributes={
        "race": sw.Choice(["white", "black", "asian", "hispanic"], [...]),
        "last_name": Surname(on="race", categories={
            "white": "pctwhite", "black": "pctblack",
            "asian": "pctapi", "hispanic": "pcthispanic",
        }),
    })
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .. import _hash
from ._fetch import cached_dataframe
from ._fetch import fetch_url as _fetch_url

_URL = "https://www2.census.gov/topics/genealogy/2010surnames/names.zip"
_MEMBER = "Names_2010Census.csv"
_DEFAULT_CACHE_DIR = Path(".synthweave_cache") / "census_surnames"
_PCT_COLUMNS = ("pctwhite", "pctblack", "pctapi", "pctaian", "pct2prace", "pcthispanic")

_cache: dict[str, pd.DataFrame] = {}


class Surname:
    """A last name jointly distributed with a declared race/ethnicity column.

    Args:
        on: the schema's column name holding race/ethnicity.
        categories: required mapping from that column's own values to one
            of `pctwhite`, `pctblack`, `pctapi`, `pctaian`, `pct2prace`,
            `pcthispanic` (the Census file's actual categories). No default,
            since guessing at this alignment would be silently wrong for
            most schemas.
    """

    def __init__(
        self,
        on: str,
        categories: Mapping[str, str],
        cache_dir: str | Path | None = _DEFAULT_CACHE_DIR,
    ):
        if not categories:
            raise ValueError("Surname needs a non-empty categories mapping")
        unknown = set(categories.values()) - set(_PCT_COLUMNS)
        if unknown:
            raise ValueError(
                f"Surname: categories map to unknown Census column(s) {sorted(unknown)}; "
                f"valid columns are {_PCT_COLUMNS}"
            )
        self.on = on
        self.categories = dict(categories)
        self._data = _surname_data(cache_dir)
        self._weights_by_column: dict[str, np.ndarray] = {}

    def depends_on(self) -> tuple[str, ...]:
        return (self.on,)

    def dtype(self):
        return None

    def draw(
        self, keys: np.ndarray, *, seed, salt: str, frame: pd.DataFrame | None = None
    ) -> np.ndarray:
        if frame is None or self.on not in frame.columns:
            raise ValueError(f"Surname rule needs column {self.on!r}, which is not available")

        values = frame[self.on].to_numpy()
        out = np.empty(len(keys), dtype=object)
        for category in set(values):
            mask = values == category
            if not mask.any():
                continue
            if category not in self.categories:
                raise KeyError(
                    f"Surname: no Census column mapped for category {category!r}; "
                    f"mapping covers {sorted(self.categories)}"
                )
            names, weights = self._pool(self.categories[category])
            out[mask] = _hash.pick(keys[mask], seed, salt, names, weights)
        return out

    def _pool(self, column: str) -> tuple[np.ndarray, np.ndarray]:
        if column not in self._weights_by_column:
            self._weights_by_column[column] = (
                self._data["count"].to_numpy(dtype=np.float64)
                * self._data[column].to_numpy(dtype=np.float64)
                / 100.0
            )
        return self._data["name"].to_numpy(dtype=object), self._weights_by_column[column]


def _surname_data(cache_dir: str | Path | None) -> pd.DataFrame:
    memo_key = "<none>" if cache_dir is None else str(Path(cache_dir))
    return cached_dataframe(_cache, memo_key, cache_dir, "surnames.csv", _fetch_and_parse)


def _fetch_and_parse() -> pd.DataFrame:
    body = _fetch_url(_URL, timeout=60, label="Census surnames")

    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        raw = archive.read(_MEMBER).decode("utf-8")

    frame = pd.read_csv(io.StringIO(raw))
    # "ALL OTHER NAMES" is an aggregate row (rank 0), not a real surname;
    # suppressed cells ("(S)", too few people for Census to report safely)
    # become 0 rather than a guess.
    frame = frame[frame["name"] != "ALL OTHER NAMES"].copy()
    for column in ("count",) + _PCT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    if frame.empty:
        raise RuntimeError(f"Census surnames archive at {_URL} produced no usable rows")
    return frame[["name", "count", *_PCT_COLUMNS]].reset_index(drop=True)
