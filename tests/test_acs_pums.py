"""synthweave.connectors.acs_pums: request shape, caching, and failure modes.

All network access is mocked. Nothing here reaches the live Census API, so
the suite stays fast and offline like the rest of the tests.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from synthweave.connectors.acs_pums import _resolve_state, fetch_pums

PAYLOAD = [
    ["AGEP", "PINCP", "state"],
    ["19", "3800", "36"],
    ["56", "0", "36"],
    ["75", "12500", "36"],
]


def _mock_response(body: bytes, status: int = 200):
    response = MagicMock()
    response.status = status
    response.read.return_value = body
    response.__enter__.return_value = response
    return response


def test_fetches_and_shapes_a_frame():
    with patch("urllib.request.urlopen", return_value=_mock_response(json.dumps(PAYLOAD).encode())):
        frame = fetch_pums(["AGEP", "PINCP"], state="36", api_key="k", cache_dir=None)

    assert list(frame.columns) == ["AGEP", "PINCP"]
    assert len(frame) == 3
    assert pd.api.types.is_numeric_dtype(frame["AGEP"])
    assert frame["PINCP"].tolist() == [3800, 0, 12500]


def test_second_call_hits_the_cache_not_the_network(tmp_path):
    with patch(
        "urllib.request.urlopen", return_value=_mock_response(json.dumps(PAYLOAD).encode())
    ) as urlopen:
        fetch_pums(["AGEP", "PINCP"], state="36", api_key="k", cache_dir=tmp_path)
        assert urlopen.call_count == 1
        fetch_pums(["AGEP", "PINCP"], state="36", api_key="k", cache_dir=tmp_path)
        assert urlopen.call_count == 1  # no second network call


def test_api_key_reaches_the_request_url():
    captured = {}

    def fake_urlopen(url, timeout=30):
        captured["url"] = url
        return _mock_response(json.dumps(PAYLOAD).encode())

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        fetch_pums(["AGEP", "PINCP"], state="36", api_key="the-key", cache_dir=None)

    assert "key=the-key" in captured["url"]


def test_missing_key_raises_with_the_signup_url(monkeypatch, tmp_path):
    monkeypatch.delenv("CENSUS_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here to fall back to
    with pytest.raises(RuntimeError, match="key_signup"):
        fetch_pums(["AGEP", "PINCP"], state="36", cache_dir=None)


def test_env_var_key_is_used_when_none_passed(monkeypatch, tmp_path):
    monkeypatch.setenv("CENSUS_API_KEY", "env-key")
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_urlopen(url, timeout=30):
        captured["url"] = url
        return _mock_response(json.dumps(PAYLOAD).encode())

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        fetch_pums(["AGEP", "PINCP"], state="36", cache_dir=None)

    assert "key=env-key" in captured["url"]


def test_dotenv_key_is_used_when_env_var_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("CENSUS_API_KEY", raising=False)
    (tmp_path / ".env").write_text("CENSUS_API_KEY=dotenv-key\n")
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_urlopen(url, timeout=30):
        captured["url"] = url
        return _mock_response(json.dumps(PAYLOAD).encode())

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        fetch_pums(["AGEP", "PINCP"], state="36", cache_dir=None)

    assert "key=dotenv-key" in captured["url"]


# --- state name/abbreviation shorthand -----------------------------------


@pytest.mark.parametrize("state", ["36", "NY", "ny", "New York", "new york"])
def test_state_forms_resolve_to_the_same_fips_code(state):
    assert _resolve_state(state) == "36"


def test_a_single_digit_fips_code_is_zero_padded():
    assert _resolve_state("6") == "06"


def test_an_unrecognized_state_raises():
    with pytest.raises(ValueError, match="Atlantis"):
        _resolve_state("Atlantis")


@pytest.mark.parametrize("state", ["36", "NY", "New York"])
def test_state_forms_produce_the_same_request_url(state):
    captured = {}

    def fake_urlopen(url, timeout=30):
        captured["url"] = url
        return _mock_response(json.dumps(PAYLOAD).encode())

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        fetch_pums(["AGEP", "PINCP"], state=state, api_key="k", cache_dir=None)

    assert "state%3A36" in captured["url"]


def test_non_200_status_raises():
    with patch("urllib.request.urlopen", return_value=_mock_response(b"", status=500)):
        with pytest.raises(RuntimeError, match="HTTP 500"):
            fetch_pums(["AGEP", "PINCP"], state="36", api_key="k", cache_dir=None)


def test_malformed_json_raises():
    with patch("urllib.request.urlopen", return_value=_mock_response(b"not json")):
        with pytest.raises(RuntimeError, match="not valid JSON"):
            fetch_pums(["AGEP", "PINCP"], state="36", api_key="k", cache_dir=None)


def test_a_header_only_response_is_a_failure_not_an_empty_frame():
    """Zero data rows means the request matched nothing, not that it worked.

    The guard was `len(payload) < 1`, so a header-only response (length 1)
    passed and produced a zero-row frame that looked like a successful
    fetch. The Census API returns exactly that shape for a filter matching
    nothing, or a subtly wrong state/year/survey combination. Handing the
    empty frame onward failed much later in `Empirical`, far from the
    request that actually went wrong.
    """
    header_only = [["AGEP", "PINCP", "ST"]]
    with patch(
        "urllib.request.urlopen", return_value=_mock_response(json.dumps(header_only).encode())
    ):
        with pytest.raises(RuntimeError, match="no rows"):
            fetch_pums(["AGEP", "PINCP"], state="36", api_key="k", cache_dir=None)
