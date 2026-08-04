"""The `conventional` gate's allowed types are written down three times: the
`grep -Eq` regex that actually decides, the `::error::` message that tells a
contributor what went wrong, and CONTRIBUTING.md's commit-subject rule.
Nothing kept them in sync, so adding a type to the regex and forgetting the
other two left the error message lying to whoever hit it.

This is the same drift class `test_ci_docs_sync.py` guards for required
checks, handled the same way: derive the truth from the workflow, assert the
prose agrees.

The regex is parsed out of the workflow rather than copied here -- a copy in
the test would just be a fourth thing to drift -- and the behaviour cases run
it through `grep -E` the way the workflow does, so what is tested is the gate
itself and not a Python-flavoured approximation of it.
"""

import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github/workflows/pr-title.yml"


def _gate_regex() -> str:
    """The ERE the workflow hands to `grep -Eq`, verbatim."""
    match = re.search(r"grep -Eq '(.*)'; then", WORKFLOW.read_text())
    assert match, "pr-title.yml has no `grep -Eq '...'` gate to derive from"
    return match.group(1)


def _types_in_regex() -> set[str]:
    """The alternation at the head of the gate regex: `^(feat|fix|...)`."""
    match = re.match(r"\^\(([a-z|]+)\)", _gate_regex())
    assert match, "the gate regex does not start with a `^(a|b|c)` type alternation"
    return set(match.group(1).split("|"))


def _types_in_error_message() -> set[str]:
    """The types the failure message names, so a contributor is told the truth."""
    match = re.search(r"Allowed types: ([^.]+)\.", WORKFLOW.read_text())
    assert match, "pr-title.yml has no 'Allowed types: ...' message to check"
    return {t.strip() for t in match.group(1).split(",")}


def _types_in_contributing() -> set[str]:
    """CONTRIBUTING.md's commit-subject rule, the third copy."""
    contributing = (ROOT / "CONTRIBUTING.md").read_text()
    match = re.search(r"where type is\s*(.+?)\.\s", contributing, re.DOTALL)
    assert match, "CONTRIBUTING.md has no 'where type is ...' sentence to check"
    return set(re.findall(r"`([^`]+)`", match.group(1)))


def _gate_accepts(title: str) -> bool:
    """Run the real gate: same regex, same `grep -E`, same no-trailing-newline
    input the workflow's `printf '%s'` produces."""
    return (
        subprocess.run(
            ["grep", "-Eq", _gate_regex()],
            input=title,
            text=True,
        ).returncode
        == 0
    )


def test_error_message_lists_the_types_the_regex_accepts():
    assert _types_in_error_message() == _types_in_regex(), (
        f"pr-title.yml's error message names {sorted(_types_in_error_message())!r} "
        f"but its regex accepts {sorted(_types_in_regex())!r}. A contributor who "
        "hits the gate would be told the wrong rule."
    )


def test_contributing_lists_the_types_the_regex_accepts():
    assert _types_in_contributing() == _types_in_regex(), (
        f"CONTRIBUTING.md documents {sorted(_types_in_contributing())!r} as the "
        f"allowed commit types but pr-title.yml enforces "
        f"{sorted(_types_in_regex())!r}."
    )


def test_gate_accepts_the_conventional_commits_recommended_types():
    """`ci` and `build` are in the spec's recommended set alongside the ones
    already allowed. Rejecting them forces a genuine CI change to be filed as
    `chore`, which is the catch-all, and the squash-merged title is permanent.
    """
    for type_ in ("feat", "fix", "chore", "docs", "refactor", "test", "perf", "ci", "build"):
        assert _gate_accepts(f"{type_}: x"), f"gate rejects `{type_}: x`"


def test_gate_accepts_scopes_and_breaking_change_markers():
    assert _gate_accepts("ci(workflows): make test-pandas3 a gate")
    assert _gate_accepts("build(deps): bump pandas floor")
    assert _gate_accepts("feat!: drop python 3.9")
    assert _gate_accepts("feat(api)!: rename Pipeline.run")


def test_gate_rejects_an_untyped_title():
    """The exact title from #24 that is baked into the v0.2.0 release notes
    with no type prefix. It is why this workflow exists."""
    assert not _gate_accepts("PII generators, real-data connectors")


def test_gate_rejects_an_unrecognised_type():
    assert not _gate_accepts("wibble: x")
    assert not _gate_accepts("feat x")
    assert not _gate_accepts("feat:")
