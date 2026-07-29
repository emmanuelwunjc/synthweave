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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

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
    """Corrupts a subset of values. Subclasses implement `corrupt`."""

    def __init__(self, rate: float):
        self.rate = as_tagged(rate)
        if not 0.0 <= self.rate.value <= 1.0:
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
    """Replaces one character with a keyboard neighbour."""

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
            if not options:
                out[i] = text
                continue
            repl = options[j % len(options)]
            out[i] = text[:j] + (repl.upper() if ch.isupper() else repl) + text[j + 1 :]
        return out


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
                values = chunk[column].to_numpy(dtype=object)
                for op in ops:
                    salt = f"noise\x00{table.name}\x00{column}\x00{op.name}"
                    rate = ctx.provenance.add(
                        f"{table.name}.noise.{column}.{op.name}.rate", op.rate
                    )
                    hit = _hash.unit(keys, ctx.seed, salt) < rate
                    counter = tally.setdefault((column, op.name), _Applied())
                    counter.eligible += len(values)
                    counter.corrupted += int(hit.sum())
                    if hit.any():
                        values[hit] = op.corrupt(values[hit], keys[hit], ctx.seed, salt)
                chunk[column] = values
            yield chunk

        ctx.report(
            table.name,
            "noise",
            realized={
                f"{col}.{op}": (a.corrupted / a.eligible if a.eligible else 0.0)
                for (col, op), a in tally.items()
            },
        )
