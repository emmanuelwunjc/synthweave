"""`pyproject.toml` is the only place the supported Faker range is written.

The bound appeared in four places at once (the `pii` extra, the `dev` extra,
`_FAKER_SUPPORTED` in `connectors/faker_names.py`, and a literal asserted in
`tests/test_faker_names.py`). Bumping the extras alone left the runtime error
message telling users "Supported: Faker>=20,<41" after that stopped being
true, and the test that looked like it guarded the message asserted the same
stale literal, so it stayed green through the drift. A guard that can only
agree with what it guards is worse than no guard.

Here `pyproject.toml` is the single authority and everything else is derived
from it: this module reads the declared spec once and asserts the source
constant and the message users actually see both match it.

Two roads were not taken:

  - `tomllib` parses `pyproject.toml` properly but is stdlib only from 3.11,
    and CI's gating leg is `test (3.10)`. `tests/test_ci_docs_sync.py` reads
    the same file with a narrow regex for the same reason; this follows it.
  - `importlib.metadata.requires("synthweave")` would give the declared specs
    from installed distribution metadata. Rejected on two counts. That
    metadata is written at install time, so editing `pyproject.toml` without
    reinstalling leaves it stale -- it would not notice the exact bump this
    test exists to catch. And the suite runs as `PYTHONPATH=src pytest`
    without the package installed at all, where it yields nothing.

Deliberately does not import Faker. The bound has to be checkable on a
minimal install, which is precisely where the `pii` extra is absent.
"""

import pathlib
import re

import pytest

from synthweave.connectors.faker_names import _FAKER_SUPPORTED, _checked_provider_pool

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _optional_dependencies(pyproject_text: str) -> str:
    """The body of `[project.optional-dependencies]`, comments stripped.

    Scoping to that one table keeps an unrelated table from contributing a
    spec, and dropping full-line comments keeps the rationale comment sitting
    above `pii` from being read as a declaration.
    """
    match = re.search(
        r"^\[project\.optional-dependencies\]\n(.*?)(?=^\[|\Z)",
        pyproject_text,
        re.MULTILINE | re.DOTALL,
    )
    assert match, "pyproject.toml has no [project.optional-dependencies] table"
    return "\n".join(
        line for line in match.group(1).splitlines() if not line.lstrip().startswith("#")
    )


def declared_faker_bound(pyproject_text: str | None = None) -> str:
    """The one Faker version spec declared in `pyproject.toml`'s extras.

    Every extra that names Faker has to name the same range: `pii` is what a
    user installs and `dev` is what CI resolves, so letting them differ would
    mean the suite validates a range no user gets.
    """
    if pyproject_text is None:
        pyproject_text = (ROOT / "pyproject.toml").read_text()
    specs = re.findall(r'"(Faker[^"]*)"', _optional_dependencies(pyproject_text))
    assert specs, "no extra in pyproject.toml declares a Faker dependency"
    assert len(set(specs)) == 1, (
        f"pyproject.toml's extras declare more than one Faker range: {sorted(set(specs))!r}. "
        "The `pii` extra is what users install and `dev` is what CI resolves, so a "
        "difference means the suite validates a range nobody gets."
    )
    return specs[0]


# --- the parser itself, on synthetic input ------------------------------


def test_extras_declaring_different_faker_ranges_is_an_error():
    """The `pii`/`dev` half of the drift. Two extras naming Faker differently
    is the same defect one step earlier, so it fails rather than silently
    picking whichever came first."""
    text = '[project.optional-dependencies]\npii = ["Faker>=20,<41"]\ndev = ["Faker>=20,<42"]\n'
    with pytest.raises(AssertionError, match="more than one Faker range"):
        declared_faker_bound(text)


def test_a_commented_out_bound_is_not_read_as_a_declaration():
    text = (
        "[project.optional-dependencies]\n"
        '# was: pii = ["Faker>=20,<40"]\n'
        'pii = ["Faker>=20,<41"]\n'
    )
    assert declared_faker_bound(text) == "Faker>=20,<41"


def test_only_the_optional_dependencies_table_is_read():
    """A Faker spec in another table is not what the `pii` extra installs."""
    text = (
        "[project]\n"
        'dependencies = ["Faker>=1,<2"]\n'
        "\n"
        "[project.optional-dependencies]\n"
        'pii = ["Faker>=20,<41"]\n'
    )
    assert declared_faker_bound(text) == "Faker>=20,<41"


def test_extras_without_faker_is_an_error():
    """Reading no bound must fail loudly. Returning a default would let the
    guard pass while checking nothing."""
    text = '[project.optional-dependencies]\nparquet = ["pyarrow>=12"]\n'
    with pytest.raises(AssertionError, match="declares a Faker dependency"):
        declared_faker_bound(text)


# --- the real tree ------------------------------------------------------


def test_source_constant_matches_the_bound_declared_in_pyproject():
    declared = declared_faker_bound()
    assert _FAKER_SUPPORTED == declared, (
        f"connectors/faker_names.py declares _FAKER_SUPPORTED = {_FAKER_SUPPORTED!r}, but "
        f"pyproject.toml's extras declare {declared!r}. pyproject.toml is the single "
        "source; update _FAKER_SUPPORTED to match it."
    )


def test_the_runtime_error_names_the_bound_declared_in_pyproject():
    """End to end: what a user is told to install is what `pyproject.toml` says.

    Calls the guard against a stand-in provider rather than monkeypatching
    Faker's real one, so this holds without the `pii` extra installed.
    """

    class ProviderWithoutLastNames:
        pass

    with pytest.raises(RuntimeError) as excinfo:
        _checked_provider_pool(ProviderWithoutLastNames, "last_names")
    message = str(excinfo.value)
    assert declared_faker_bound() in message, (
        f"the error users see says {message!r}, which does not name the Faker range "
        f"{declared_faker_bound()!r} that pyproject.toml actually declares."
    )
