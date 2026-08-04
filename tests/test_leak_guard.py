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
        "subject ssn is 123-45-6789 per the intake form",  # leak-guard: allow (this guard's own SSN fixture)
        "contact: jane.doe@example.org",  # leak-guard: allow (example.org is the reserved documentation domain)
        "call 415-555-0142 to confirm",  # leak-guard: allow (555-01xx is the reserved fictional range)
        "CENSUS_API_KEY=0123456789abcdef0123456789abcdef01234567",  # leak-guard: allow (fixture)
        "loaded from /Users/yw1084/code/synthweave/private.csv",  # leak-guard: allow (fixture)
    ],
)
def test_pii_shapes_are_caught_in_a_data_shaped_file(line):
    assert guard.pii_findings("docs/GUIDE.md", line)


@pytest.mark.parametrize(
    "line",
    [
        # The keyword sits in the middle of the name, not at the end. `\b`
        # after the stem required it to end the identifier, so every one of
        # these passed clean through the guard as merged (issue #153).
        "aws_secret_access_key=wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY",  # leak-guard: allow (fake AWS example key from AWS' own docs)
        "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY",  # leak-guard: allow (fake AWS example key from AWS' own docs)
        "secret_key_base=0123456789abcdef0123456789abcdef01234567",  # leak-guard: allow (fixture)
        "census_key=0123456789abcdef0123456789abcdef01234567",  # leak-guard: allow (fixture)
        # Short-but-random, and long-but-wordy. The 20-character-with-a-digit
        # value bound waved both of these through.
        "API_KEY=abc123def456ghi7",  # leak-guard: allow (fixture)
        "password: correcthorsebatterystaple",  # leak-guard: allow (fixture)
        # .json and .yaml are both scanned suffixes, and in both the name is
        # quoted, so the `=`/`:` does not follow the identifier directly.
        '  "aws_secret_access_key": "wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY",',  # leak-guard: allow (fake AWS example key from AWS' own docs)
    ],
)
def test_compound_credential_names_are_caught(line):
    """`aws_secret_access_key` is the canonical AWS variable name, and the
    single most commonly leaked credential shape in public repositories."""
    assert guard.pii_findings("docs/GUIDE.md", line)


def test_findings_report_the_line_number():
    findings = guard.pii_findings("notes.md", "clean\nssn 123-45-6789\n")  # leak-guard: allow (this guard's own SSN fixture)
    assert findings and findings[0][0] == 2


# --- the false positives that would get the guard turned off -------------


@pytest.mark.parametrize(
    "path",
    ["tests/test_intake.py", "src/synthweave/stages/identity.py",
     "examples/three_modes.py"],
)
def test_a_real_person_in_the_package_is_caught(path):
    """src/, tests/ and examples/ were blanket-exempted from the SSN, email and
    phone shapes because "the package generates these, so it would fire
    constantly". Measured across every tracked file it does not: the tree
    yields eight findings, all this file's own fixtures. The exemption bought
    that much quiet and blinded the guard across the three largest directories,
    so it is gone (issue #154). The package generates these values at runtime;
    a literal one written into a file is a person, and is now caught.
    """
    body = (
        'CONTACT = "marcus.delacroix@realmail.example"\n'  # leak-guard: allow (the invented person this test exists to catch)
        'SSN = "078-05-1120"\n'  # leak-guard: allow (078-05-1120 is the void Woolworth wallet SSN, issued to nobody)
        'PHONE = "415-555-0142"\n'  # leak-guard: allow (555-01xx is the reserved fictional range)
    )
    assert {label for _, label, _ in guard.pii_findings(path, body)} == {
        "SSN", "email address", "phone number"
    }


def test_secrets_are_still_caught_inside_the_package():
    """A credential or a personal absolute path is never legitimate anywhere,
    including under src/ and tests/."""
    assert guard.pii_findings("src/synthweave/connectors/acs_pums.py",
                              "CENSUS_API_KEY=0123456789abcdef0123456789abcdef01234567")  # leak-guard: allow (fixture)
    assert guard.pii_findings("tests/test_acs_pums.py", "path = '/Users/someone/data'")  # leak-guard: allow (fixture)


@pytest.mark.parametrize(
    ("path", "line"),
    [
        # Every credential-shaped line that exists in the tree today, verbatim.
        # These are what the value bound is for. A guard that fires on a
        # deliberately fake fixture gets switched off, and then it guards
        # nothing, so widening the *name* half must not cost any of them.
        ("tests/test_acs_pums.py",
         '    (tmp_path / ".env").write_text("CENSUS_API_KEY=dotenv-key\\n")'),
        ("tests/test_acs_pums.py",
         '    assert "key=dotenv-key" in captured["url"]'),
        ("tests/test_acs_pums.py",
         '    (outer / ".env").write_text("CENSUS_API_KEY=ancestor-key\\n")'),
        ("tests/test_acs_pums.py",
         '    (tmp_path / ".env").write_text("CENSUS_API_KEY=root-key\\n")'),
        ("tests/test_acs_pums.py",
         '    assert "key=root-key" in captured["url"]'),
        ("src/synthweave/connectors/acs_pums.py", "    api_key: str | None = None,"),
        ("src/synthweave/connectors/acs_pums.py",
         "        api_key: overrides `CENSUS_API_KEY` from the environment/`.env`."),
        # The `_`-delimited part rule is what keeps the widened name half from
        # matching any identifier that merely contains "key".
        ("src/synthweave/pipeline.py", "    keyword = 'a_long_enough_value_here'"),
        ("src/synthweave/pipeline.py", "    rows = sorted(rows, key=operator.itemgetter(0))"),
    ],
)
def test_placeholder_credentials_in_fixtures_are_not_flagged(path, line):
    """Credential *shapes* with no credential in them. `dotenv-key` is 10
    characters, `ancestor-key` is 12 with no digit, `root-key` is 8, and the
    docstring's value stops at the first space. All stay under both bars."""
    assert guard.pii_findings(path, line) == []


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


def test_no_git_binary_reports_a_usable_message_not_a_traceback(capsys, monkeypatch):
    """With no paths given the entry point calls `staged_files()`, which shells
    out to git. When git is absent that raised an uncaught FileNotFoundError:
    fail-closed, but with a traceback instead of an explanation (issue #153).
    """
    def no_git(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory: 'git'")

    monkeypatch.setattr(guard.subprocess, "run", no_git)
    assert guard.cli([]) == 1
    out = capsys.readouterr().out
    assert "Traceback" not in out
    assert "git" in out


def test_binary_and_unscanned_file_types_are_skipped():
    assert guard.pii_findings("assets/logo.png", "123-45-6789") == []  # leak-guard: allow (this guard's own SSN fixture)


# --- the whole point: the current tree must be quiet ---------------------


@needs_git
def test_the_committed_tree_passes_its_own_guard():
    """A guard that fires on the repo as it stands blocks all legitimate work
    and gets removed. `pre-commit run --all-files` runs exactly this."""
    assert guard.main(guard.tracked_files()) == 0
