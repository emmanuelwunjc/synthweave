"""Fetch real microdata rows from the Census Bureau's ACS PUMS API.

For the "not enough data" case: a caller who lacks a real microdata sample
can pull one from a public source instead, and hand it to the same
`Empirical` structure source that a caller with their own data would use.

    from synthweave.connectors.acs_pums import fetch_pums

    df = fetch_pums(["AGEP", "PINCP"], state="36")
    sw.Empirical(df)

Nothing here maps ACS variable codes to a schema's column names — that
mapping is specific to whatever the caller is modeling, so it lives at the
call site (see examples/three_layers_data_availability.py), not in this
generic fetch function.

The Census API requires a registered key for every request as of this
writing; there is no anonymous quota. Get one free at
https://api.census.gov/data/key_signup.html and either export
CENSUS_API_KEY, or drop it in a `.env` file at the repo root (gitignored,
never committed) as `CENSUS_API_KEY=...`.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

_API_BASE = "https://api.census.gov/data"
_KEY_SIGNUP_URL = "https://api.census.gov/data/key_signup.html"
_DEFAULT_CACHE_DIR = Path(".synthweave_cache") / "acs_pums"

# USPS abbreviation -> FIPS code. A closed, fixed federal reference standard
# (same category as a timezone or currency-code table), not schema-specific
# data, so it stays in the connector rather than in examples/.
_STATE_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
    "CT": "09", "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15",
    "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21",
    "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27",
    "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
    "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53",
    "WV": "54", "WI": "55", "WY": "56", "PR": "72",
}

_STATE_NAMES = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR",
    "CALIFORNIA": "CA", "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE",
    "DISTRICT OF COLUMBIA": "DC", "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI",
    "IDAHO": "ID", "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS",
    "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD",
    "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS",
    "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK",
    "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX",
    "UTAH": "UT", "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY", "PUERTO RICO": "PR",
}


def _resolve_state(state: str) -> str:
    """A 2-digit FIPS code, from a FIPS code, a USPS abbreviation, or a name.

    `"36"`, `"NY"`, and `"New York"` all resolve to the same code, so a
    caller never has to look up a FIPS table by hand.
    """
    if state.isdigit():
        # Validated like every other form. This branch used to zero-pad and
        # return without a membership check, so a code that is not a state
        # reached the API and came back as a confusing transport-level
        # failure naming a URL rather than the input that was wrong.
        code = state.zfill(2)
        if code not in set(_STATE_FIPS.values()):
            raise ValueError(
                f"{state!r} is not a recognized state FIPS code. Codes are 2 digits and "
                "not contiguous; a USPS abbreviation or full name works too, e.g. "
                '"NY" or "New York".'
            )
        return code
    abbrev = _STATE_NAMES.get(state.strip().upper(), state.strip().upper())
    if abbrev not in _STATE_FIPS:
        raise ValueError(
            f"{state!r} is not a recognized state FIPS code, USPS abbreviation, or name."
        )
    return _STATE_FIPS[abbrev]


def fetch_pums(
    variables: list[str],
    state: str,
    *,
    year: int = 2022,
    survey: str = "acs1",
    cache_dir: str | Path | None = _DEFAULT_CACHE_DIR,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Real ACS PUMS microdata rows for the given variables and state.

    Args:
        variables: ACS variable codes to fetch, e.g. `["AGEP", "PINCP"]`.
            Passed straight through to the API; not validated against the
            ACS data dictionary, so a typo surfaces as the API's own error.
        state: a state FIPS code, USPS abbreviation, or full name, e.g.
            `"36"`, `"NY"`, or `"New York"` all mean the same state.
        year: survey year.
        survey: `"acs1"` (1-year) or `"acs5"` (5-year, more geographies).
        cache_dir: where to cache raw responses, keyed by the request. Pass
            `None` to disable caching (every call hits the live API).
        api_key: overrides `CENSUS_API_KEY` from the environment/`.env`.

    Raises:
        RuntimeError: no API key available, or the API returned an error.
    """
    key = api_key or _resolve_api_key()
    if not key:
        raise RuntimeError(
            "CENSUS_API_KEY is not set. Sign up for a free key at "
            f"{_KEY_SIGNUP_URL}, then either `export CENSUS_API_KEY=...` or "
            "add `CENSUS_API_KEY=...` to a `.env` file at the repo root."
        )

    params = {"get": ",".join(variables), "for": f"state:{_resolve_state(state)}"}
    url = f"{_API_BASE}/{year}/acs/{survey}/pums?{urllib.parse.urlencode(params)}"

    cache_path = _cache_path(cache_dir, url)
    if cache_path is not None and cache_path.exists():
        payload = json.loads(cache_path.read_text())
    else:
        payload = _request(f"{url}&key={key}", url)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload))

    return _to_frame(payload, variables, url)


def _resolve_api_key() -> str | None:
    if "CENSUS_API_KEY" in os.environ:
        return os.environ["CENSUS_API_KEY"]
    return _read_dotenv().get("CENSUS_API_KEY")


def _read_dotenv() -> dict[str, str]:
    """A minimal `KEY=VALUE` reader for a local `.env`, stdlib only.

    Walks up from the current directory looking for `.env`, the way most
    dotenv tools do, since a script under `examples/` runs from the repo
    root but a test might run from elsewhere.
    """
    values: dict[str, str] = {}
    here = Path.cwd()
    for directory in (here, *here.parents):
        candidate = directory / ".env"
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, _, value = line.partition("=")
                values.setdefault(name.strip(), value.strip())
            break
    return values


def _cache_path(cache_dir: str | Path | None, url: str) -> Path | None:
    if cache_dir is None:
        return None
    digest = hashlib.sha256(url.encode()).hexdigest()[:32]
    return Path(cache_dir) / f"{digest}.json"


def _request(url_with_key: str, url_for_errors: str) -> list[list[str]]:
    try:
        with urllib.request.urlopen(url_with_key, timeout=30) as response:
            status = response.status
            body = response.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"ACS PUMS request failed with HTTP {e.code}: {url_for_errors}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"ACS PUMS request failed: {url_for_errors} ({e.reason})") from e

    if status != 200:
        raise RuntimeError(f"ACS PUMS request failed with HTTP {status}: {url_for_errors}")
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"ACS PUMS response was not valid JSON: {url_for_errors}"
        ) from e


def _to_frame(payload: list[list[str]], variables: list[str], url: str) -> pd.DataFrame:
    # A header row with nothing under it is a request that matched nothing,
    # not a successful fetch of zero people. The Census API returns exactly
    # that for a filter matching no records, or a subtly wrong
    # state/year/survey/variable combination. Letting it through produced an
    # empty frame that only failed later, inside Empirical, with an error
    # pointing nowhere near the request that was actually wrong.
    if not payload or len(payload) <= 1:
        raise RuntimeError(
            f"ACS PUMS response contained no rows, only a header if that: {url}. "
            "Usually the geography, year, survey or variable filter matched no "
            "records; check those before assuming the API is down."
        )
    header, *rows = payload
    frame = pd.DataFrame(rows, columns=header)
    for column in variables:
        if column not in frame.columns:
            continue
        # The API returns every value as a string; ACS variable codes are
        # numeric by convention (AGEP, PINCP, ...), so coerce what parses and
        # leave the rest alone rather than guessing.
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().all():
            frame[column] = numeric
    return frame[variables]
