"""The noise stage: dtype preservation and rate accounting.

`noise.py` is Lane A's file. Its dedicated coverage lives here rather than
in `test_pipeline.py`, which Lane B owns, so this stage's tests never
collide with theirs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import synthweave as sw


def test_a_zero_rate_op_leaves_the_column_dtype_untouched(schema):
    """The bug: any noise op converted a column to object, even at rate 0.0.

    Nothing is corrupted at rate 0.0, so there is nothing that justifies the
    column losing its dtype.
    """
    raw = sw.Pipeline(schema).run()["wages"]
    assert raw["wage"].dtype == np.float64

    out = sw.Pipeline(
        schema, noiser=sw.Noise({"wages": {"wage": [sw.Missing(0.0)]}})
    ).run()["wages"]

    assert out["wage"].dtype == np.float64
    pd.testing.assert_series_equal(out["wage"], raw["wage"])


def test_missing_on_a_float_column_stays_float(schema):
    """A float column already has a representation for a missing value."""
    out = sw.Pipeline(
        schema, noiser=sw.Noise({"wages": {"wage": [sw.Missing(0.3)]}})
    ).run()["wages"]

    assert out["wage"].dtype == np.float64
    assert out["wage"].isna().any()


def test_missing_on_an_int_column_widens_to_nullable_int(schema):
    """An int64 column cannot hold a null, so it widens to pandas' Int64.

    The alternative, silently falling back to `object`, is what this issue
    is about in the first place.
    """
    out = sw.Pipeline(
        schema, noiser=sw.Noise({"roster": {"birth_year": [sw.Missing(0.3)]}})
    ).run()["roster"]

    assert str(out["birth_year"].dtype) == "Int64"
    assert out["birth_year"].isna().any()


def test_typo_still_produces_a_text_column(schema):
    """A real type change (number -> corrupted text) must still show up.

    Restoring dtype after corruption must not paper over a genuine change:
    only a no-op or a null-only change like `Missing` is safe to restore.
    """
    out = sw.Pipeline(
        schema, noiser=sw.Noise({"roster": {"education": [sw.Typo(0.5)]}})
    ).run()["roster"]

    # Spelled dtype-agnostically: a text column is `object` on pandas 2 and `str`
    # on pandas 3. What matters is that it is text and not the original number.
    assert pd.api.types.is_string_dtype(out["education"])
    assert not pd.api.types.is_numeric_dtype(out["education"])


def _script_corruption_rates(rate: float) -> dict[str, float]:
    """Share of rows actually changed, per source value, for a Typo run."""
    values = ["北京市", "Ωμέγα", "Ünüver", "Smith"]
    entity = sw.Entity(
        "person",
        count=4_000,
        attributes={"name": sw.Choice(values, [0.25, 0.25, 0.25, 0.25])},
        identifiers=[sw.Identifier("tax_id")],
    )
    table = sw.Table("t", grain=sw.PerEntity("person"), carry=["name"])
    schema = sw.Schema(entities=[entity], tables=[table], seed=1)

    clean = sw.Pipeline(schema).run()["t"]["name"]
    dirty = sw.Pipeline(
        schema, noiser=sw.Noise({"t": {"name": [sw.Typo(rate)]}})
    ).run()["t"]["name"]
    return {v: float((dirty[clean == v] != v).mean()) for v in values}


def test_typo_delivers_the_configured_rate_on_every_script(schema):
    """The declared rate is a promise, and it was broken outside Latin script.

    `Typo` picked one character position, looked it up in the Latin-only
    keyboard map, and gave up when there was no neighbour. For a value made
    entirely of CJK or Greek characters that is not a reduced rate, it is
    zero corruption on every row of every run, with the realized rate still
    reporting the configured one. Silent under-delivery of a declared rate is
    the defect; the missing keyboard map is only how it happens.
    """
    rates = _script_corruption_rates(0.5)
    for value, realized in rates.items():
        assert 0.45 < realized < 0.55, (
            f"{value!r} was corrupted at {realized:.3f}, not the configured 0.5"
        )


def test_a_latin_typo_is_still_a_keyboard_slip(schema):
    """Whatever handles other scripts must not change what ASCII already did.

    A Latin-script typo has to keep looking like a mistyped adjacent key:
    same length, exactly one character different, and the replacement a
    neighbour of the character it replaced.
    """
    entity = sw.Entity(
        "person",
        count=500,
        attributes={"name": sw.Choice(["Smith", "Jones"], [0.5, 0.5])},
        identifiers=[sw.Identifier("tax_id")],
    )
    table = sw.Table("t", grain=sw.PerEntity("person"), carry=["name"])
    schema = sw.Schema(entities=[entity], tables=[table], seed=7)

    clean = sw.Pipeline(schema).run()["t"]["name"]
    dirty = sw.Pipeline(
        schema, noiser=sw.Noise({"t": {"name": [sw.Typo(1.0)]}})
    ).run()["t"]["name"]

    neighbours = {
        "a": "sq", "b": "vn", "c": "xv", "d": "sf", "e": "wr", "f": "dg",
        "g": "fh", "h": "gj", "i": "uo", "j": "hk", "k": "jl", "l": "k",
        "m": "n", "n": "bm", "o": "ip", "p": "o", "q": "wa", "r": "et",
        "s": "ad", "t": "ry", "u": "yi", "v": "cb", "w": "qe", "x": "zc",
        "y": "tu", "z": "x",
    }
    for before, after in zip(clean, dirty):
        assert len(after) == len(before), f"{before!r} -> {after!r} changed length"
        differing = [i for i in range(len(before)) if before[i] != after[i]]
        assert len(differing) == 1, f"{before!r} -> {after!r} changed {len(differing)} chars"
        i = differing[0]
        assert after[i].lower() in neighbours[before[i].lower()], (
            f"{before!r} -> {after!r}: {after[i]!r} is not a keyboard neighbour of {before[i]!r}"
        )
def test_missing_on_a_category_column_stays_categorical(people):
    """A column of only nulls still allows a categorical dtype.

    `_restore_dtype` promises the original dtype back where the values still
    allow it, but only `Int64` was special-cased, so a `category` column came
    back as `object` after a null-only `Missing` pass. Fed through a custom
    generator because no built-in stage emits a categorical column.
    """

    @sw.register("generator", "test-categorical", overwrite=True)
    class CategoricalRows:
        def emit(self, table, ctx):
            yield pd.DataFrame(
                {
                    "_sw_entity": [f"person:{i}" for i in range(6)],
                    "_sw_row": [f"r{i}" for i in range(6)],
                    "grade": pd.Series(list("ABCABC"), dtype="category"),
                }
            )

    table = sw.Table("t", grain=sw.PerEntity("person"))
    schema = sw.Schema(entities=[people], tables=[table], seed=1)

    raw = sw.Pipeline(schema, generator="test-categorical").run()["t"]
    assert str(raw["grade"].dtype) == "category"

    out = sw.Pipeline(
        schema,
        generator="test-categorical",
        noiser=sw.Noise({"t": {"grade": [sw.Missing(1.0)]}}),
    ).run()["t"]

    assert out["grade"].isna().all()
    assert str(out["grade"].dtype) == "category"


def test_typo_on_a_category_column_falls_back_to_object(people):
    """Restoring an ExtensionDtype must not paper over a real value change.

    `pd.array` maps a value a `category` cannot hold to null rather than
    raising, unlike `astype`. Without a guard, a corrupted value would vanish
    into a null and the column would look clean.
    """

    @sw.register("generator", "test-categorical", overwrite=True)
    class CategoricalRows:
        def emit(self, table, ctx):
            yield pd.DataFrame(
                {
                    "_sw_entity": [f"person:{i}" for i in range(6)],
                    "_sw_row": [f"r{i}" for i in range(6)],
                    "grade": pd.Series(["aa"] * 6, dtype="category"),
                }
            )

    table = sw.Table("t", grain=sw.PerEntity("person"))
    schema = sw.Schema(entities=[people], tables=[table], seed=1)

    out = sw.Pipeline(
        schema,
        generator="test-categorical",
        noiser=sw.Noise({"t": {"grade": [sw.Typo(1.0)]}}),
    ).run()["t"]

    assert out["grade"].notna().all()
    assert (out["grade"] != "aa").all()
