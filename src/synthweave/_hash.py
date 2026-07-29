"""Deterministic, vectorized derivation primitives.

Every random-looking value in synthweave comes from here. Nothing uses a
stateful RNG, because a stateful RNG makes output depend on how many draws
happened before it, which means row order and chunk boundaries would change
results. Instead every value is a pure function of:

    (run seed, a stable row or entity key, a salt naming what is being drawn)

Two consequences follow, and they are the reason the whole library can be
chunked:

1. Order independence. Row 5 gets the same value whether it is processed
   first or last.
2. Chunk invariance. Splitting a stream into different chunk sizes cannot
   change any value, so `chunk_size` is purely a memory knob.

The salt separates independent draws for the same key. Drawing "height" and
"weight" for the same person uses the same key but different salts, so the
two are uncorrelated.

Hashing uses pandas' vectorized array hash rather than per-element hashlib.
At tens of millions of rows a Python-level hashlib loop is the bottleneck,
and the vectorized hash is stable for a fixed hash_key, which is all the
determinism contract requires.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

_U64 = np.uint64(2**64 - 1)
_SCALE = np.float64(2.0**64)


def hash_key(seed: int | str, salt: str) -> str:
    """A 16-character key for pandas' vectorized hash, derived from seed and salt.

    pandas requires exactly 16 characters. Deriving it via sha256 means every
    (seed, salt) pair produces an independent hash stream.
    """
    return hashlib.sha256(f"{seed}\x00{salt}".encode()).hexdigest()[:16]


def hash_u64(keys: np.ndarray | pd.Series, seed: int | str, salt: str) -> np.ndarray:
    """Vectorized uint64 hash of `keys` under (seed, salt)."""
    arr = pd.Index(np.asarray(keys, dtype=object))
    return pd.util.hash_array(arr.to_numpy(), encoding="utf8", hash_key=hash_key(seed, salt))


def unit(keys: np.ndarray | pd.Series, seed: int | str, salt: str) -> np.ndarray:
    """Deterministic floats in [0, 1), one per key."""
    return hash_u64(keys, seed, salt).astype(np.float64) / _SCALE


def integers(
    keys: np.ndarray | pd.Series, seed: int | str, salt: str, low: int, high: int
) -> np.ndarray:
    """Deterministic integers in [low, high), one per key."""
    if high <= low:
        raise ValueError(f"integers() needs high > low, got low={low} high={high}")
    span = np.uint64(high - low)
    return (hash_u64(keys, seed, salt) % span).astype(np.int64) + low


def normal(
    keys: np.ndarray | pd.Series, seed: int | str, salt: str, mean: float, sd: float
) -> np.ndarray:
    """Deterministic normal draws via Box-Muller on two independent hash streams.

    Box-Muller is used rather than an RNG because it is a pure function of the
    two uniforms, which preserves the order independence contract.
    """
    u1 = np.clip(unit(keys, seed, f"{salt}\x00bm1"), 1e-12, 1.0)
    u2 = unit(keys, seed, f"{salt}\x00bm2")
    z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    return mean + sd * z


def pick(
    keys: np.ndarray | pd.Series,
    seed: int | str,
    salt: str,
    values: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Deterministic weighted choice from `values`, one per key."""
    values = np.asarray(values, dtype=object)
    if len(values) == 0:
        raise ValueError("pick() needs at least one value")
    u = unit(keys, seed, salt)
    if weights is None:
        idx = np.minimum((u * len(values)).astype(np.int64), len(values) - 1)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != values.shape:
            raise ValueError(f"weights length {w.shape} does not match values {values.shape}")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        total = w.sum()
        if total <= 0:
            raise ValueError("weights must sum to a positive number")
        idx = np.searchsorted(np.cumsum(w / total), u, side="right")
        idx = np.minimum(idx, len(values) - 1)
    return values[idx]


def derive_id(
    keys: np.ndarray | pd.Series,
    seed: int | str,
    tag: str,
    *,
    prefix: str = "",
    digits: int = 10,
) -> np.ndarray:
    """Deterministic identifiers of the form `<prefix><zero-padded digits>`.

    This is the cross-table linking primitive. Because the value depends only
    on (seed, entity key, tag), the same entity yields the same identifier in
    every table that carries that identifier kind, with no lookup table and no
    coordination between tables. A different `tag` yields an unrelated
    identifier for the same entity, which is how one person carries both a
    student id and a tax id without the two being derivable from each other.
    """
    if digits < 1:
        raise ValueError("digits must be at least 1")
    modulus = 10**digits
    n = (hash_u64(keys, seed, f"id\x00{tag}") % np.uint64(modulus)).astype(np.int64)
    # Formatted with numpy string ops rather than a Python f-string loop.
    # Profiling a 960,000 row run put the f-string version at the top of
    # tottime by a wide margin; this is the same output built in C.
    text = np.char.zfill(n.astype("U"), digits)
    if prefix:
        text = np.char.add(prefix, text)
    return text.astype(object)
