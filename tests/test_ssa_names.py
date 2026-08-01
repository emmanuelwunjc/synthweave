"""synthweave.connectors.ssa_names: fetch/parse mechanics and year/sex conditioning.

Uses a small synthetic zip for the mechanics tests (offline). The
conditioning/fidelity tests need a real local copy of SSA's names.zip,
via the SSA_NAMES_ZIP environment variable pointing at one — ssa.gov
blocks automated fetches from some environments, so this can't assume
network access. Skipped cleanly if that variable isn't set.
"""

from __future__ import annotations

import io
import os
import zipfile
from unittest.mock import MagicMock, patch

import pytest

import invariants
import synthweave as sw
from synthweave.connectors.ssa_names import SSAFirstName, _cache, _fetch, _parse, _ssa_data

FAKE_YEARS = {
    1900: ["Mary,F,400", "John,M,350", "William,M,200"],
    2020: ["Noah,M,300", "Olivia,F,280", "Emma,F,260"],
}


def _zip_bytes_for(years: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for year, lines in years.items():
            archive.writestr(f"yob{year}.txt", "\n".join(lines) + "\n")
    return buf.getvalue()


def _fake_zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for year, lines in FAKE_YEARS.items():
            archive.writestr(f"yob{year}.txt", "\n".join(lines) + "\n")
    return buf.getvalue()


def _mock_response(body: bytes, status: int = 200):
    response = MagicMock()
    response.status = status
    response.read.return_value = body
    response.__enter__.return_value = response
    return response


# --- fetch/parse mechanics --------------------------------------------------


def test_parse_shapes_a_frame():
    frame = _parse(_fake_zip_bytes())
    assert set(frame["year"]) == {1900, 2020}
    assert int(frame.loc[(frame["year"] == 1900) & (frame["name"] == "Mary"), "count"].iloc[0]) == 400


def test_fetch_uses_the_mocked_url():
    with patch("urllib.request.urlopen", return_value=_mock_response(_fake_zip_bytes())):
        body = _fetch()
    assert _parse(body) is not None


def test_missing_local_source_raises():
    with pytest.raises(RuntimeError, match="not found"):
        SSAFirstName(source="/no/such/file.zip", cache_dir=None)


def test_year_outside_range_raises(tmp_path):
    zip_path = tmp_path / "names.zip"
    zip_path.write_bytes(_fake_zip_bytes())
    rule = SSAFirstName(on="birth_year", source=zip_path, cache_dir=None)
    person = sw.Entity("person", 10, attributes={"birth_year": sw.Constant(1500)})
    table = sw.Table("t", grain="person", carry=["birth_year"], columns={"first_name": rule})
    with pytest.raises(ValueError, match="outside the data's"):
        sw.Pipeline(sw.Schema([person], [table])).run()


# --- conditioning behavior, against a real local SSA file -------------------


@pytest.fixture(scope="module")
def ssa_zip_path():
    path = os.environ.get("SSA_NAMES_ZIP")
    if not path or not os.path.exists(path):
        pytest.skip("SSA_NAMES_ZIP not set to a real names.zip; skipping real-data tests")
    return path


@pytest.fixture
def schema_with_names(ssa_zip_path) -> sw.Schema:
    person = sw.Entity(
        "person",
        2_000,
        attributes={
            "birth_year": sw.Integer(1950, 2010),
            "sex": sw.Choice(["M", "F"]),
            "first_name": SSAFirstName(on="birth_year", sex_on="sex", source=ssa_zip_path, cache_dir=None),
        },
    )
    table = sw.Table("roster", grain="person", carry="*")
    return sw.Schema(entities=[person], tables=[table], seed=6)


def test_names_are_chunk_invariant(schema_with_names):
    invariants.assert_chunk_invariant(schema_with_names)


def test_names_are_deterministic(schema_with_names):
    invariants.assert_deterministic(schema_with_names)


def test_names_respect_sex_conditioning(schema_with_names, ssa_zip_path):
    result = sw.Pipeline(schema_with_names).run()["roster"]
    from synthweave.connectors.ssa_names import _ssa_data

    data = _ssa_data(ssa_zip_path, None)
    female_names = set(data.loc[data["sex"] == "F", "name"])
    male_names = set(data.loc[data["sex"] == "M", "name"])
    female_only = female_names - male_names
    male_only = male_names - female_names
    got_female_only = set(result.loc[result["sex"] == "F", "first_name"]) & female_only
    got_male_only = set(result.loc[result["sex"] == "M", "first_name"]) & male_only
    # Every unisex name is ambiguous, so check the unambiguous ones only.
    wrong = set(result.loc[result["sex"] == "F", "first_name"]) & male_only
    assert not wrong, f"female rows got male-only names: {wrong}"
    assert got_female_only or got_male_only  # sanity: the split isn't vacuous


def test_name_pool_shifts_by_era(ssa_zip_path):
    """Real names for 1900 and 2020 shouldn't look like the same distribution."""
    person = sw.Entity(
        "person",
        4_000,
        attributes={
            "birth_year": sw.Choice([1900, 2020]),
            "first_name": SSAFirstName(on="birth_year", source=ssa_zip_path, cache_dir=None),
        },
    )
    table = sw.Table("t", grain="person", carry="*")
    result = sw.Pipeline(sw.Schema([person], [table], seed=2)).run()["t"]
    top_1900 = set(result[result["birth_year"] == 1900]["first_name"].value_counts().head(5).index)
    top_2020 = set(result[result["birth_year"] == 2020]["first_name"].value_counts().head(5).index)
    assert len(top_1900 & top_2020) < 3


def test_a_nan_birth_year_is_rejected_rather_than_left_as_none(tmp_path):
    """NaN slips every comparison, so the row was silently never written.

    The range guard uses `years < min` and `years > max`, both False for
    NaN, so a NaN year passed validation. The draw loop then groups with
    `years == year`, also False for NaN, so the row matched no group and its
    slot in the output array was never assigned, surfacing as an
    uninitialised None rather than an error.
    """
    zip_path = tmp_path / "names.zip"
    zip_path.write_bytes(_fake_zip_bytes())
    rule = SSAFirstName(on="birth_year", source=zip_path, cache_dir=None)
    person = sw.Entity("person", 10, attributes={"birth_year": sw.Constant(float("nan"))})
    table = sw.Table("t", grain="person", carry=["birth_year"], columns={"first_name": rule})
    with pytest.raises(ValueError, match="(?i)nan|missing"):
        sw.Pipeline(sw.Schema([person], [table])).run()


def test_two_sources_do_not_share_one_cache_file(tmp_path):
    """The on-disk cache name must distinguish which source produced it.

    The filename was always `ssa_names.csv`, with `source` present only in
    the in-memory memo key. In a fresh process, a second `SSAFirstName`
    pointing at a different local zip found the first source's file already
    on disk and silently returned its data. The second source was never
    read and nothing was raised.
    """
    other_years = {1900: ["Zelda,F,999"]}
    first = tmp_path / "first.zip"
    first.write_bytes(_fake_zip_bytes())
    second = tmp_path / "second.zip"
    second.write_bytes(_zip_bytes_for(other_years))

    cache = tmp_path / "cache"
    # Bypass the in-memory memo, which already keys on source, so this
    # exercises the on-disk path the way a fresh process would.
    frame_one = _ssa_data(first, cache)
    _cache.clear()
    frame_two = _ssa_data(second, cache)

    assert set(frame_one["name"]) == {"Mary", "John", "William", "Noah", "Olivia", "Emma"}
    assert set(frame_two["name"]) == {"Zelda"}
