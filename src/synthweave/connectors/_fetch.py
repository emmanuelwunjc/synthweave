"""Shared fetch/cache machinery for connector modules.

Every connector here fetches one reference table over HTTP and caches it
locally: an in-memory memo first, then an on-disk file, then a live fetch as
the last resort, with urllib errors wrapped into a plain `RuntimeError`
naming the request. Four connectors used to reimplement that shape
independently, drifting slightly each time; this module gives it one
implementation.

`cached_dataframe`'s memo is owned by the caller, not this module, so each
connector keeps its own inspectable, clearable cache dict rather than
sharing one global cache keyed across unrelated data sources.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import pandas as pd


def fetch_url(
    url: str,
    *,
    timeout: int,
    label: str,
    request_url: str | None = None,
    hint: str | None = None,
) -> bytes:
    """Bytes from `url`, wrapping urllib errors as a `RuntimeError` naming `label`.

    `request_url`, if given, is the URL actually requested (e.g. with a
    secret API key appended); `url` is what appears in the error message, so
    a caller can keep a secret out of an exception. `hint`, if given, is
    appended to a connectivity-failure message only (not an HTTP error one),
    for guidance specific to that data source (e.g. "download it by hand and
    pass a local path instead").
    """
    target = request_url if request_url is not None else url
    try:
        with urllib.request.urlopen(target, timeout=timeout) as response:
            status = response.status
            body = response.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{label} request failed with HTTP {e.code}: {url}") from e
    except urllib.error.URLError as e:
        suffix = f" {hint}" if hint else ""
        raise RuntimeError(f"{label} request failed: {url} ({e.reason}).{suffix}") from e

    if status != 200:
        raise RuntimeError(f"{label} request failed with HTTP {status}: {url}")
    return body


def cached_dataframe(
    memo: dict[str, pd.DataFrame],
    memo_key: str,
    cache_dir: str | Path | None,
    filename: str,
    load: Callable[[], pd.DataFrame],
    *,
    dtype=None,
) -> pd.DataFrame:
    """A DataFrame from `memo`, then an on-disk CSV, then `load()` as a last resort.

    A hit at any level is written back to every level above it (disk hit ->
    memoized; `load()` miss -> written to disk, when `cache_dir` is not
    `None`, then memoized), so the next call in this process or a later one
    skips straight to the cheapest hit.
    """
    if memo_key in memo:
        return memo[memo_key]

    cache_path = None if cache_dir is None else Path(cache_dir) / filename
    if cache_path is not None and cache_path.exists():
        frame = pd.read_csv(cache_path, dtype=dtype)
        memo[memo_key] = frame
        return frame

    frame = load()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache_path, index=False)
    memo[memo_key] = frame
    return frame
