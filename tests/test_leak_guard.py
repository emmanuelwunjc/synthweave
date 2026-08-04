"""The repo is public, and files the .gitignore header says "stay on this
machine" were committed to it anyway before those rules existed. .gitignore
cannot fix that on its own: `git add -f` bypasses it, and a file that is
already tracked stays tracked no matter what the ignore rules say. So the
guard is tools/check_no_private_leak.py, and this is its test.

Written red first, per CLAUDE.md: each assertion below was confirmed to fail
against a stub checker that returned "clean" before the real one existed.
"""

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load():
    """tools/ is not an importable package, same problem (and same fix) as
    tests/test_docs_map_sync.py has with tools/check_docs_map.py."""
    spec = importlib.util.spec_from_file_location(
        "check_no_private_leak", ROOT / "tools" / "check_no_private_leak.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _load()

# The path half asks the repo's own .gitignore through `git check-ignore`, so
# it needs a checkout. tools/mutation_check.py copies the tree without .git,
# and an sdist has no .git either; test_docs_map_sync self-skips for the same
# reason. The guard itself raises rather than passing in that case, which is
# what test_no_git_checkout_fails_closed pins down.
needs_git = pytest.mark.skipif(
    not guard.is_git_checkout(), reason="not a git checkout, so .gitignore cannot be consulted"
)


# --- (a) private-category paths ------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "CLAUDE.md",
        "CLAUDE.local.md",
        "docs/HANDOFF.md",
        "docs/ISSUES.md",
        "docs/NEXT_STEPS.md",
        "docs/specs/noise-model.md",
        "docs/brainstorms/idea.md",
        ".claude/skills/pm/SKILL.md",
        ".env",
        ".env.local",
        ".env.production",
    ],
)
@needs_git
def test_private_paths_are_rejected(path):
    """These are the categories the .gitignore header names as local-only.
    Every one of them has to be refused even when it reaches the index, which
    is exactly what `git add -f` does.
    """
    assert guard.private_path_reason(path) is not None


@pytest.mark.parametrize(
    "path",
    [
        "src/synthweave/mode.py",
        "tests/test_noise.py",
        "examples/three_modes.py",
        "tools/mutation_check.py",
        "docs/GUIDE.md",
        "docs/MAP.md",
        "README.md",
        "CONTRIBUTING.md",
        "pyproject.toml",
        ".github/workflows/ci.yml",
    ],
)
@needs_git
def test_published_paths_are_accepted(path):
    """docs/GUIDE.md and docs/MAP.md are the two negated exceptions to
    `docs/*`. A guard that rejected them would block the documentation the
    README points at.
    """
    assert guard.private_path_reason(path) is None


@needs_git
def test_private_paths_come_from_gitignore_not_a_hardcoded_list():
    """The categories are derived from the repo's own ignore rules, so adding
    a rule to .gitignore extends the guard without editing the guard."""
    assert guard.private_path_reason("docs/A_FILE_NOBODY_HAS_NAMED_YET.md")


# --- (b) PII-shaped content ----------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "subject ssn is 123-45-6789 per the intake form",
        "contact: jane.doe@example.org",
        "call 415-555-0142 to confirm",
        "CENSUS_API_KEY=0123456789abcdef0123456789abcdef01234567",  # leak-guard: allow (fixture)
        "loaded from /Users/yw1084/code/synthweave/private.csv",  # leak-guard: allow (fixture)
    ],
)
def test_pii_shapes_are_caught_in_a_data_shaped_file(line):
    assert guard.pii_findings("docs/GUIDE.md", line)


def test_findings_report_the_line_number():
    findings = guard.pii_findings("notes.md", "clean\nssn 123-45-6789\n")
    assert findings and findings[0][0] == 2


# --- the false positives that would get the guard turned off -------------


def test_synthetic_fixtures_in_the_package_are_not_flagged():
    """This repo generates fake SSNs, names and emails on purpose
    (src/synthweave/stages, tests/test_faker_names.py, and issue I28 is
    literally about SSN area codes). Flagging those makes the guard noise,
    and a noisy guard gets disabled, which protects nothing.
    """
    body = 'SSN_FORMAT = "123-45-6789"\nemail = "a.person@example.com"\nphone = "415-555-0142"\n'
    for path in ("src/synthweave/stages/identity.py", "tests/test_faker_names.py",
                 "examples/three_modes.py"):
        assert guard.pii_findings(path, body) == []


def test_secrets_are_still_caught_inside_the_package():
    """Synthetic *data* is expected under src/, tests/ and examples/. A real
    credential or a personal absolute path never is, so those two shapes are
    not suppressed there."""
    assert guard.pii_findings("src/synthweave/connectors/acs_pums.py",
                              "CENSUS_API_KEY=0123456789abcdef0123456789abcdef01234567")  # leak-guard: allow (fixture)
    assert guard.pii_findings("tests/test_acs_pums.py", "path = '/Users/someone/data'")  # leak-guard: allow (fixture)


def test_placeholder_credentials_in_fixtures_are_not_flagged():
    """tests/test_acs_pums.py writes `CENSUS_API_KEY=dotenv-key` into a temp
    .env, and acs_pums.py's docstring says "api_key: overrides ...". Both are
    credential *shapes* with no credential in them. The value has to look like
    a key (20+ characters, at least one digit) before this fires."""
    assert guard.pii_findings("tests/test_acs_pums.py",
                              '(tmp_path / ".env").write_text("CENSUS_API_KEY=dotenv-key\\n")') == []
    assert guard.pii_findings("src/synthweave/connectors/acs_pums.py",
                              "api_key: overrides `CENSUS_API_KEY` from the environment") == []


def test_a_suppression_comment_silences_one_line():
    line = "example ssn 123-45-6789  # leak-guard: allow (documented sample)"
    assert guard.pii_findings("docs/GUIDE.md", line) == []


def test_no_git_checkout_fails_closed(tmp_path, monkeypatch):
    """Without .git, `git check-ignore` exits 128. Reading that as "not
    ignored" would silently disable the whole path half of this guard, which
    is the one failure mode a guard must never have.
    """
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    with pytest.raises(guard.NoGitCheckout):
        guard.private_path_reason("docs/HANDOFF.md")
    assert guard.main(["docs/HANDOFF.md"]) == 1


def test_binary_and_unscanned_file_types_are_skipped():
    assert guard.pii_findings("assets/logo.png", "123-45-6789") == []


# --- the whole point: the current tree must be quiet ---------------------


@needs_git
def test_the_committed_tree_passes_its_own_guard():
    """A guard that fires on the repo as it stands blocks all legitimate work
    and gets removed. `pre-commit run --all-files` runs exactly this."""
    assert guard.main(guard.tracked_files()) == 0
