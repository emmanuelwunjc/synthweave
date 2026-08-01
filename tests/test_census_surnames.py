"""synthweave.connectors.census_surnames: fetch/parse mechanics and race conditioning.

Fetch-shape tests are mocked (offline). The conditioning/fidelity tests hit
the real, live Census file, skipped if this environment has no network.
"""

from __future__ import annotations

import io
import zipfile
from unittest.mock import MagicMock, patch

import pytest

import invariants
import synthweave as sw
from synthweave.connectors.census_surnames import Surname, _fetch_and_parse

FAKE_CSV = """name,rank,count,prop100k,cum_prop100k,pctwhite,pctblack,pctapi,pctaian,pct2prace,pcthispanic
SMITH,1,1000,1,1,90,5,1,1,1,2
NGUYEN,2,500,1,2,2,0,95,0,1,1
GONZALEZ,3,500,1,3,5,0,0,0,1,90
SUPPRESSED,4,100,1,4,90,(S),0,0,(S),(S)
ALL OTHER NAMES,0,2000,1,5,60,20,10,2,3,15
"""


def _fake_zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("Names_2010Census.csv", FAKE_CSV)
    return buf.getvalue()


def _mock_response(body: bytes, status: int = 200):
    response = MagicMock()
    response.status = status
    response.read.return_value = body
    response.__enter__.return_value = response
    return response


# --- fetch/parse mechanics --------------------------------------------------


def test_fetch_and_parse_shapes_a_frame():
    with patch("urllib.request.urlopen", return_value=_mock_response(_fake_zip_bytes())):
        frame = _fetch_and_parse()
    assert set(frame["name"]) == {"SMITH", "NGUYEN", "GONZALEZ", "SUPPRESSED"}


def test_all_other_names_row_is_excluded():
    with patch("urllib.request.urlopen", return_value=_mock_response(_fake_zip_bytes())):
        frame = _fetch_and_parse()
    assert "ALL OTHER NAMES" not in set(frame["name"])


def test_suppressed_cells_become_zero():
    with patch("urllib.request.urlopen", return_value=_mock_response(_fake_zip_bytes())):
        frame = _fetch_and_parse()
    row = frame[frame["name"] == "SUPPRESSED"].iloc[0]
    assert row["pctblack"] == 0.0
    assert row["pct2prace"] == 0.0
    assert row["pcthispanic"] == 0.0


def test_empty_categories_raises():
    with pytest.raises(ValueError, match="non-empty categories"):
        Surname(on="race", categories={}, cache_dir=None)


def test_unknown_census_column_raises():
    with pytest.raises(ValueError, match="unknown Census column"):
        Surname(on="race", categories={"white": "not_a_real_column"}, cache_dir=None)


# --- race conditioning, against the real live Census data ------------------


@pytest.fixture(scope="module")
def real_surname_data():
    try:
        from synthweave.connectors.census_surnames import _surname_data

        return _surname_data(None)
    except RuntimeError as e:
        pytest.skip(f"Census surnames unreachable in this environment: {e}")


@pytest.fixture
def schema_with_surnames(real_surname_data) -> sw.Schema:
    person = sw.Entity(
        "person",
        2_000,
        attributes={
            "race": sw.Choice(["white", "black", "asian", "hispanic"], [0.6, 0.13, 0.06, 0.19]),
            "last_name": Surname(
                on="race",
                categories={
                    "white": "pctwhite",
                    "black": "pctblack",
                    "asian": "pctapi",
                    "hispanic": "pcthispanic",
                },
                cache_dir=None,
            ),
        },
    )
    table = sw.Table("roster", grain="person", carry="*")
    return sw.Schema(entities=[person], tables=[table], seed=3)


def test_surnames_are_chunk_invariant(schema_with_surnames):
    invariants.assert_chunk_invariant(schema_with_surnames)


def test_surnames_are_deterministic(schema_with_surnames):
    invariants.assert_deterministic(schema_with_surnames)


def test_surname_pool_differs_meaningfully_by_race(schema_with_surnames):
    """Real distributions, not the same pool reshuffled per race."""
    result = sw.Pipeline(schema_with_surnames).run()["roster"]
    top_asian = set(result[result["race"] == "asian"]["last_name"].value_counts().head(10).index)
    top_hispanic = set(
        result[result["race"] == "hispanic"]["last_name"].value_counts().head(10).index
    )
    # Real Asian and Hispanic surname pools barely overlap in reality.
    assert len(top_asian & top_hispanic) <= 1


def test_unmapped_category_raises(real_surname_data):
    person = sw.Entity(
        "person",
        50,
        attributes={
            "race": sw.Choice(["white", "other"]),
            "last_name": Surname(on="race", categories={"white": "pctwhite"}, cache_dir=None),
        },
    )
    table = sw.Table("t", grain="person", carry="*")
    with pytest.raises(KeyError, match="no Census column mapped"):
        sw.Pipeline(sw.Schema([person], [table], seed=1)).run()
