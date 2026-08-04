"""Stage 3a: messiness.

Real administrative data is dirty in specific, patterned ways. Names get
mistyped, scanned forms confuse 0 with O, fields go blank. Record linkage
research needs that dirt, because an algorithm evaluated on clean data tells
you nothing about how it performs on the real thing.

Noise is applied to any column regardless of what the column means, so a user
is never limited to the field types some fixed schema anticipated.

Which rows get hit is deterministic: a row is corrupted when
unit(row key, seed, salt) < rate. So the same run always dirties the same
rows, and the realized rate is reported so a user can confirm they got the
mess they asked for.

`rate` is a flat probability, or a function of the chunk returning one
probability per row. The flat form can only express MCAR (missing completely
at random); the function form is what expresses differential nonresponse,
where the chance of a value going missing depends on the row it belongs to.
Either way the draw itself is the same hash-derived comparison: the rate
function chooses the threshold and never draws, which is what keeps a
per-row rate as deterministic and chunk invariant as a flat one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from .. import _hash
from ..context import RunContext
from ..provenance import as_tagged
from ..registry import register
from ..schema import Table
from .base import own

# Visually confusable pairs, the usual suspects from scanned forms.
_OCR_MAP = {
    "0": "O", "O": "0", "1": "I", "I": "1", "l": "1", "5": "S", "S": "5",
    "8": "B", "B": "8", "2": "Z", "Z": "2", "6": "G", "G": "6",
}
_KEYBOARD = {
    "a": "sq", "b": "vn", "c": "xv", "d": "sf", "e": "wr", "f": "dg", "g": "fh",
    "h": "gj", "i": "uo", "j": "hk", "k": "jl", "l": "k", "m": "n", "n": "bm",
    "o": "ip", "p": "o", "q": "wa", "r": "et", "s": "ad", "t": "ry", "u": "yi",
    "v": "cb", "w": "qe", "x": "zc", "y": "tu", "z": "x",
}


class NoiseOp:
    """Corrupts a subset of values. Subclasses implement `corrupt`.

    `rate` is either a flat probability applied to every row, or a vectorized
    function of the chunk returning one probability per row. The function form
    is what expresses differential nonresponse, e.g. missingness that differs
    by subgroup rather than being uniform across the column:

        Missing(lambda f: 0.05 + 0.25 * (f["education"] == "HS"))

    It follows the same contract as `Sequential.fn`: handed the frame, returns
    an array. A flat rate is range-checked here; a function's output can only
    be checked once it has been called, which happens per chunk in `Noise.run`.

    The function must be a pure function of each row's own values. It is
    handed one chunk at a time, so anything derived from the chunk as a whole
    (its length, a mean, a row's position in it) would make the output depend
    on `chunk_size`, which is meant to be a memory knob and nothing else. The
    frame it receives is the chunk as the noise stage holds it, which includes
    `_sw_`-prefixed bookkeeping columns; those are internal and must not be
    read.
    """

    def __init__(self, rate: float | Callable[[pd.DataFrame], Any]):
        self.rate = as_tagged(rate)
        if not callable(self.rate.value) and not 0.0 <= self.rate.value <= 1.0:
            raise ValueError(f"noise rate must be in [0, 1], got {self.rate.value}")

    @property
    def name(self) -> str:
        return type(self).__name__.lower()

    def corrupt(self, values: np.ndarray, keys: np.ndarray, seed, salt: str) -> np.ndarray:
        raise NotImplementedError


class Missing(NoiseOp):
    """Blanks the value. The most common real-world defect by far."""

    def corrupt(self, values, keys, seed, salt):
        return np.full(len(values), None, dtype=object)


class Typo(NoiseOp):
    """Mistypes one character: a keyboard slip, or a slip of the fingers.

    A keyboard slip needs a layout, and `_KEYBOARD` only knows a Latin one.
    The character picked for corruption used to be looked up there and the
    row left untouched when it had no neighbour, which for a value written
    entirely in a script the map does not cover (CJK, Greek, Cyrillic) is not
    a reduced rate but zero corruption, on every row of every run, while the
    reported realized rate still claimed the configured one. The declared
    rate is a promise, so silently under-delivering it is worse than the
    missing map.

    Filling the map in for every script is the wrong repair twice over: it is
    unbounded, and key adjacency is not how those scripts are typed in the
    first place (a CJK value is composed through an IME, not struck key by
    key). What is script-agnostic is the *other* everyday typo, the fingers
    arriving out of order or twice:

    - transposition, swapping the picked character with its neighbour;
    - duplication, when transposition would not change anything (a
      one-character value, or two identical characters side by side).

    A keyboard slip is still preferred wherever the layout knows the
    character, so Latin-script corruption is byte-for-byte what it was.
    Empty and null values are still passed through: there is no character to
    mistype, and `Missing` already owns the null case.
    """

    def corrupt(self, values, keys, seed, salt):
        pos = _hash.unit(keys, seed, f"{salt}\x00pos")
        out = np.empty(len(values), dtype=object)
        for i, raw in enumerate(values):
            text = "" if raw is None else str(raw)
            if not text:
                out[i] = raw
                continue
            j = min(int(pos[i] * len(text)), len(text) - 1)
            ch = text[j]
            options = _KEYBOARD.get(ch.lower())
            if options:
                repl = options[j % len(options)]
                out[i] = text[:j] + (repl.upper() if ch.isupper() else repl) + text[j + 1 :]
            else:
                out[i] = _slip(text, j)
        return out


def _slip(text: str, j: int) -> str:
    """Mistype character `j` without knowing what script it is written in.

    Swap it with the character next to it, preferring the one on the right so
    a mid-word slip reads the way a real one does. Two identical characters
    swap to themselves and a one-character value has nothing to swap with, so
    those double the character instead: a repeat is as ordinary a typo as a
    transposition, and it is the only other corruption that needs no map.
    """
    for k in (j + 1, j - 1):
        if 0 <= k < len(text) and text[k] != text[j]:
            lo, hi = min(j, k), max(j, k)
            return text[:lo] + text[hi] + text[lo] + text[hi + 1 :]
    return text[:j] + text[j] + text[j:]


class OCR(NoiseOp):
    """Swaps a character for its visual twin, as a scanner would."""

    def corrupt(self, values, keys, seed, salt):
        pos = _hash.unit(keys, seed, f"{salt}\x00pos")
        out = np.empty(len(values), dtype=object)
        for i, raw in enumerate(values):
            text = "" if raw is None else str(raw)
            candidates = [j for j, ch in enumerate(text) if ch in _OCR_MAP]
            if not candidates:
                out[i] = raw
                continue
            j = candidates[min(int(pos[i] * len(candidates)), len(candidates) - 1)]
            out[i] = text[:j] + _OCR_MAP[text[j]] + text[j + 1 :]
        return out


def _row_rates(fn: Callable[[pd.DataFrame], Any], chunk: pd.DataFrame, path: str) -> np.ndarray:
    """Per-row rates from a rate function, length- and range-checked.

    A flat rate is checked in `NoiseOp.__init__`; a function's output only
    exists once it has been called, so the same check has to happen here.
    Out of range is silent otherwise: above 1 corrupts every row, below 0
    corrupts none, and both look like a result rather than a config error.

    The length is checked for the same reason. `np.asarray` accepts a scalar
    or a length numpy can broadcast, so a rate derived from the chunk as a
    whole (a mean, a count, a row's position) applies cleanly and makes the
    output depend on `chunk_size`, which is meant to be a memory knob. A
    function is asked for one rate per row; anything else is a config error,
    not a rate. A genuinely flat rate is spelled as a number, not a function.
    """
    rates = np.asarray(fn(chunk), dtype=float)
    if rates.shape != (len(chunk),):
        got = "a scalar" if rates.ndim == 0 else f"shape {rates.shape}"
        raise ValueError(
            f"noise rate function for {path} must return one rate per row: "
            f"expected shape {(len(chunk),)}, got {got}. A rate computed from the "
            f"chunk as a whole is not a function of the row, and would make the "
            f"result depend on chunk_size."
        )
    if not np.all((rates >= 0.0) & (rates <= 1.0)):
        bad = rates[(rates < 0.0) | (rates > 1.0)]
        raise ValueError(
            f"noise rate function for {path} returned value(s) outside [0, 1], "
            f"e.g. {sorted(set(bad.tolist()))[:3]}"
        )
    return rates


def _restore_dtype(values: np.ndarray, dtype: Any) -> np.ndarray | pd.api.extensions.ExtensionArray:
    """Give a noised column back its original dtype where the values allow it.

    Corruption runs through an object array regardless of the column's real
    type, so writing it straight back promoted every noised column to
    `object`, even at rate 0.0 where nothing actually changed. Only `Missing`
    is safe to undo: it introduces `None`, which a float column already
    represents as NaN and a non-nullable int column cannot hold at all, so an
    int column widens to pandas' nullable `Int64` rather than losing its
    numeric type. `Typo`/`OCR` replace a value with different text, which is
    a real type change, not noise to paper over; the cast below fails for
    that case and the column is left as object, as it must be.

    `ndarray.astype` speaks numpy dtypes only, so a pandas ExtensionDtype
    (`category`, nullable `Int64`, and every text column under pandas 3)
    raises there rather than converting, and the column fell back to object
    even when the values allowed the dtype: a `category` column that a
    `Missing` pass had emptied came back as `object`, contradicting the
    promise above. Those go through `pd.array`, the same restore
    `CARTSynthesizer._restore` uses, rather than a second special case per
    dtype the way `Int64` alone once was.

    One difference from `astype` has to be undone here, though. Where
    `astype` rejects a value the dtype cannot hold, `pd.array` maps it to
    null: a `Typo` result outside a category's set would disappear into a
    null and the column would read as clean. That is exactly the papering
    over this function refuses to do, so a cast that nulls a value that was
    not already null is treated as the failure `astype` would have raised.
    """
    if dtype == object:
        return values
    try:
        if any(v is None for v in values) and pd.api.types.is_integer_dtype(dtype):
            return pd.array(values, dtype="Int64")
        if isinstance(dtype, pd.api.extensions.ExtensionDtype):
            restored = pd.array(values, dtype=dtype)
            if (pd.isna(restored) & ~pd.isna(values)).any():
                return values
            return restored
        return values.astype(dtype)
    except (TypeError, ValueError):
        return values


@dataclass
class _Applied:
    eligible: int = 0
    corrupted: int = 0


@register("noiser", "default")
class Noise:
    """Applies configured noise ops per table and column.

    Config is table, then column, then an ordered list of ops:

        Noise({"people": {"first_name": [Typo(0.05)], "ssn": [Missing(0.02)]}})
    """

    def __init__(self, config: Mapping[str, Mapping[str, Sequence[NoiseOp]]] | None = None):
        self.config = config or {}

    def run(self, chunks: Iterator[pd.DataFrame], table: Table, ctx: RunContext) -> Iterator[pd.DataFrame]:
        spec = self.config.get(table.name)
        if not spec:
            yield from chunks
            return

        tally: dict[tuple[str, str], _Applied] = {}
        for chunk in chunks:
            chunk = own(chunk)
            keys = chunk["_sw_row"].to_numpy()
            for column, ops in spec.items():
                if column not in chunk.columns:
                    raise KeyError(
                        f"noise config names column {column!r} of table {table.name!r}, "
                        f"which does not exist; available: {sorted(chunk.columns)}"
                    )
                original_dtype = chunk[column].dtype
                # copy=True is required: under pandas 3 copy-on-write, to_numpy on a
                # column that is already object dtype hands back a read-only view, so
                # the `values[hit] = ...` write below raises. It has no observable
                # effect on pandas 2, so the test-pandas3 CI job is its only guard.
                values = chunk[column].to_numpy(dtype=object, copy=True)
                for op in ops:
                    salt = f"noise\x00{table.name}\x00{column}\x00{op.name}"
                    rate = ctx.provenance.add(
                        f"{table.name}.noise.{column}.{op.name}.rate", op.rate
                    )
                    if callable(rate):
                        rate = _row_rates(rate, chunk, f"{table.name}.{column}.{op.name}")
                    hit = _hash.unit(keys, ctx.seed, salt) < rate
                    counter = tally.setdefault((column, op.name), _Applied())
                    counter.eligible += len(values)
                    counter.corrupted += int(hit.sum())
                    if hit.any():
                        values[hit] = op.corrupt(values[hit], keys[hit], ctx.seed, salt)
                chunk[column] = _restore_dtype(values, original_dtype)
            yield chunk

        ctx.report(
            table.name,
            "noise",
            realized={
                f"{col}.{op}": (a.corrupted / a.eligible if a.eligible else 0.0)
                for (col, op), a in tally.items()
            },
        )
