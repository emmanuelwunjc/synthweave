"""CONTRIBUTING.md's required-checks list must match the CI jobs that actually
exist. This drifted once already (PR #67 added the `conventional` job to
branch protection without CONTRIBUTING.md ever mentioning it); this test
makes that drift fail the suite instead of waiting for someone to notice.
"""

import re
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _job_blocks(workflow_text: str) -> list[str]:
    """Each top-level job's header line plus its body, split on the next
    job header (a line indented by exactly two spaces)."""
    jobs_section = workflow_text.split("\njobs:\n", 1)[1]
    return re.split(r"\n(?=  [a-zA-Z][a-zA-Z0-9_-]*:\n)", "\n" + jobs_section)[1:]


def _expected_required_checks() -> set[str]:
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    pr_title = (ROOT / ".github/workflows/pr-title.yml").read_text()

    checks = set()
    for block in _job_blocks(ci) + _job_blocks(pr_title):
        job_id = re.match(r"  ([a-zA-Z][a-zA-Z0-9_-]*):\n", block).group(1)
        match = re.search(r'python-version:\s*\[([^\]]+)\]', block)
        if match:
            versions = [v.strip().strip('"') for v in match.group(1).split(",")]
            checks.update(f"{job_id} ({v})" for v in versions)
        else:
            checks.add(job_id)
    return checks


def _documented_required_checks() -> set[str]:
    contributing = (ROOT / "CONTRIBUTING.md").read_text()
    # Non-greedy, stops at the first ". "/".\n" (a sentence boundary), not at
    # the "." inside "3.10" (never followed by whitespace).
    match = re.search(r"Required checks:\s*(.+?)\.\s", contributing, re.DOTALL)
    assert match, "CONTRIBUTING.md has no 'Required checks: ...' sentence to check"
    return set(re.findall(r"`([^`]+)`", match.group(1)))


def test_documented_required_checks_match_actual_ci_jobs():
    expected = _expected_required_checks()
    documented = _documented_required_checks()
    assert documented == expected, (
        f"CONTRIBUTING.md documents {documented!r} as required checks, but "
        f"the workflows in .github/workflows/ actually define {expected!r}. "
        "Update CONTRIBUTING.md (and branch protection) to match."
    )
