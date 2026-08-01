"""synthweave.connectors.geonames: fetch/cache mechanics and address consistency.

Network access is mocked for the fetch-shape tests. The consistency/
determinism tests need the real GeoNames row count and structure to mean
anything, so they run against the live cache built by earlier tests in this
file (skipped, not mocked, if no network is available in this environment).
"""

from __future__ import annotations

import io
import zipfile
from unittest.mock import MagicMock, patch

import pytest

import invariants
import synthweave as sw
from synthweave.connectors.geonames import USAddress, _fetch_and_parse, _postal_data

FAKE_ROWS = [
    "US\t99553\tAkutan\tAlaska\tAK\tAleutians East\t013\t\t\t54.143\t-165.7854\t1",
    "US\t99571\tCold Bay\tAlaska\tAK\tAleutians East\t013\t\t\t55.1858\t-162.7211\t1",
    "US\t10001\tNew York\tNew York\tNY\tNew York\t061\t\t\t40.7484\t-73.9967\t4",
]


def _fake_zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("US.txt", "\n".join(FAKE_ROWS) + "\n")
    return buf.getvalue()


def _mock_response(body: bytes, status: int = 200):
    response = MagicMock()
    response.status = status
    response.read.return_value = body
    response.__enter__.return_value = response
    return response


# --- fetch/cache mechanics --------------------------------------------------


def test_fetch_and_parse_shapes_a_frame():
    with patch("urllib.request.urlopen", return_value=_mock_response(_fake_zip_bytes())):
        frame = _fetch_and_parse()

    assert len(frame) == 3
    assert list(frame.loc[frame["postal_code"] == "10001", "city"])[0] == "New York"
    assert list(frame.loc[frame["postal_code"] == "10001", "state_abbr"])[0] == "NY"


def test_postal_data_caches_to_disk_and_skips_a_second_fetch(tmp_path):
    with patch(
        "urllib.request.urlopen", return_value=_mock_response(_fake_zip_bytes())
    ) as urlopen:
        _postal_data(tmp_path)
        assert urlopen.call_count == 1
        assert (tmp_path / "us_postal.csv").exists()

        # A fresh process wouldn't have the in-memory memo either; simulate
        # that by going through the module's cache dict directly cleared.
        from synthweave.connectors import geonames

        geonames._cache.clear()
        _postal_data(tmp_path)
        assert urlopen.call_count == 1  # still 1: read from the on-disk cache


def test_non_200_status_raises():
    with patch("urllib.request.urlopen", return_value=_mock_response(b"", status=500)):
        with pytest.raises(RuntimeError, match="HTTP 500"):
            _fetch_and_parse()


def test_invalid_field_raises():
    with pytest.raises(ValueError, match="field must be one of"):
        USAddress("street_name")


# --- address consistency, using the real cached data ------------------------


@pytest.fixture(scope="module")
def real_postal_data():
    """The real GeoNames table, fetched once and reused by every test below.

    Skips this whole block if there's no network reachable in the test
    environment, rather than failing tests that aren't about network access.
    """
    try:
        return _postal_data(None)
    except RuntimeError as e:
        pytest.skip(f"GeoNames unreachable in this environment: {e}")


@pytest.fixture
def people_with_address(real_postal_data) -> sw.Entity:
    return sw.Entity(
        "person",
        1_000,
        attributes={
            "city": USAddress("city", cache_dir=None),
            "state": USAddress("state_abbr", cache_dir=None),
            "postal_code": USAddress("postal_code", cache_dir=None),
        },
    )


@pytest.fixture
def schema_with_address(people_with_address) -> sw.Schema:
    table = sw.Table("roster", grain="person", carry="*")
    return sw.Schema(entities=[people_with_address], tables=[table], seed=4)


def test_address_fields_are_chunk_invariant(schema_with_address):
    invariants.assert_chunk_invariant(schema_with_address)


def test_address_fields_are_deterministic(schema_with_address):
    invariants.assert_deterministic(schema_with_address)


def test_city_state_zip_are_real_consistent_rows(schema_with_address, real_postal_data):
    result = sw.Pipeline(schema_with_address).run()["roster"]
    real_triples = set(
        zip(real_postal_data["city"], real_postal_data["state_abbr"], real_postal_data["postal_code"])
    )
    out_triples = set(zip(result["city"], result["state"], result["postal_code"]))
    assert out_triples <= real_triples


def test_independent_groups_do_not_always_coincide(real_postal_data):
    person = sw.Entity(
        "person",
        1_000,
        attributes={
            "home_city": USAddress("city", cache_dir=None),
            "work_city": USAddress("city", group="work", cache_dir=None),
        },
    )
    table = sw.Table("roster", grain="person", carry="*")
    result = sw.Pipeline(sw.Schema([person], [table], seed=1)).run()["roster"]
    # Not literally impossible to coincide by chance, but with 41k+ real
    # cities, an independent group matching the default group on most rows
    # would mean the two groups aren't actually independent.
    same_share = (result["home_city"] == result["work_city"]).mean()
    assert same_share < 0.05


def test_a_leading_quote_does_not_swallow_the_following_row():
    """US.txt is tab separated and was never quoted, so quotes are literal.

    `csv.reader`'s default dialect treats a `"` at the *start* of a field as
    an opening quote and keeps consuming lines until it finds a closing one.
    A field beginning with a quote therefore absorbs the next row into
    itself: two records become one, every column of it misaligned against
    _COLUMNS, and nothing raises.

    The quote must lead the field to trigger this. Mid-field quotes are
    literal even under the default dialect, which is why an earlier version
    of this test passed against the bug.
    """
    from synthweave.connectors import geonames

    leading_quote = 'US\t10001\t"Big Apple\tNY\tNY\tNew York\t061\t\t\t40.7\t-74.0\t4\n'
    following = "US\t10002\tPlainville\tNY\tNY\tNew York\t061\t\t\t40.8\t-74.1\t4\n"
    rows = geonames._parse_postal_tsv(leading_quote + following)

    assert len(rows) == 2, "the second row was absorbed into the first"
    assert rows[0][1] == "10001"
    assert rows[1][1] == "10002"
    assert rows[1][2] == "Plainville"
