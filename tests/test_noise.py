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
