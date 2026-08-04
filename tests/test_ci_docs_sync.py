"""CONTRIBUTING.md's required-checks list must match the checks CI actually
gates on. This drifted once already (PR #67 added the `conventional` job to
branch protection without CONTRIBUTING.md ever mentioning it); this test
makes that drift fail the suite instead of waiting for someone to notice.

"A job exists" and "a job is required" are not the same thing, so the
required set is derived from the workflow files rather than assumed:

  - A job with `continue-on-error: true` cannot fail the build, so it can
    never be a gate no matter what branch protection says.
  - A matrix job reports one check per interpreter, but protection waits on
    the minimum supported one only. The rest run for information.
  - A job that another job `needs` gates through that dependent rather than on
    its own. `mutation-shard` is fanned out across runners and rolled up by
    `mutation-check`; protection lists the roll-up, not the legs, so the leg
    count can change without touching branch protection.

The parsing here is deliberately regex-based: PyYAML is not a dependency of
this package, and adding one so a test can read two small workflow files is
a worse trade than a narrow regex.
"""

import re
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _job_blocks(workflow_text: str) -> list[str]:
    """Each top-level job's header line plus its body, split on the next
    job header (a line indented by exactly two spaces).

    Full-line comments are dropped first. The workflows put a rationale
    comment at job indentation above most jobs, and leaving those in lets a
    commented-out key be read as a real one.
    """
    jobs_section = workflow_text.split("\njobs:\n", 1)[1]
    jobs_section = "\n".join(
        line for line in jobs_section.splitlines() if not line.lstrip().startswith("#")
    )
    return re.split(r"\n(?=  [a-zA-Z][a-zA-Z0-9_-]*:\n)", "\n" + jobs_section)[1:]


def _jobs_without_timeout(workflow_text: str) -> list[str]:
    """Job ids in one workflow that declare no `timeout-minutes`.

    Anchored at four spaces, the job-key level where `runs-on` lives. A
    step's own `timeout-minutes` is indented deeper and bounds one step
    rather than the job, so it must not count as the job having a cap.
    """
    missing = []
    for block in _job_blocks(workflow_text):
        job_id = re.match(r"  ([a-zA-Z][a-zA-Z0-9_-]*):\n", block).group(1)
        if not re.search(r"^    timeout-minutes:\s*\d+\s*$", block, re.MULTILINE):
            missing.append(job_id)
    return missing


def _gating_python_version() -> str:
    """The project's minimum supported interpreter, read from pyproject. That
    is the matrix leg branch protection waits on, so raising the floor has to
    move the required check with it."""
    pyproject = (ROOT / "pyproject.toml").read_text()
    match = re.search(r'requires-python\s*=\s*"[^0-9]*([0-9]+\.[0-9]+)"', pyproject)
    assert match, "pyproject.toml has no requires-python floor to derive from"
    return match.group(1)


def _jobs_depended_on(blocks: list[str]) -> set[str]:
    """Job ids named in some other job's `needs:`. They report their own check
    run, but branch protection waits on the job that rolls them up."""
    depended = set()
    for block in blocks:
        match = re.search(r"^    needs:\s*(.+)$", block, re.MULTILINE)
        if match:
            depended.update(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", match.group(1)))
    return depended


def _required_checks(workflow_text: str, gating_python: str) -> set[str]:
    """The checks in one workflow that can actually block a merge."""
    checks = set()
    blocks = _job_blocks(workflow_text)
    depended = _jobs_depended_on(blocks)
    for block in blocks:
        job_id = re.match(r"  ([a-zA-Z][a-zA-Z0-9_-]*):\n", block).group(1)
        # Four spaces: a job-level key. A step's own continue-on-error is
        # indented deeper and does not make the whole job advisory.
        if re.search(r"^    continue-on-error:\s*true\s*$", block, re.MULTILINE):
            continue
        if job_id in depended:
            continue
        match = re.search(r"python-version:\s*\[([^\]]+)\]", block)
        if match:
            versions = [v.strip().strip('"') for v in match.group(1).split(",")]
            assert gating_python in versions, (
                f"job {job_id!r} runs {versions!r}, which does not include the "
                f"gating interpreter {gating_python!r}. The required check named "
                "in branch protection would not exist."
            )
            checks.add(f"{job_id} ({gating_python})")
        else:
            checks.add(job_id)
    return checks


def _expected_required_checks() -> set[str]:
    gating_python = _gating_python_version()
    checks = set()
    for name in ("ci.yml", "pr-title.yml"):
        text = (ROOT / ".github/workflows" / name).read_text()
        checks |= _required_checks(text, gating_python)
    return checks


def _documented_required_checks() -> set[str]:
    contributing = (ROOT / "CONTRIBUTING.md").read_text()
    # Non-greedy, stops at the first ". "/".\n" (a sentence boundary), not at
    # the "." inside "3.10" (never followed by whitespace).
    match = re.search(r"Required checks:\s*(.+?)\.\s", contributing, re.DOTALL)
    assert match, "CONTRIBUTING.md has no 'Required checks: ...' sentence to check"
    return set(re.findall(r"`([^`]+)`", match.group(1)))


def test_continue_on_error_job_is_never_required():
    """A job that cannot fail the build cannot be a required check. This is
    asserted on the property, not on a job name, so making today's advisory
    job gating (or adding a new advisory one) is caught either way.
    """
    workflow = (
        "name: X\n"
        "\njobs:\n"
        "  gating:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: true\n"
        "\n"
        "  advisory:\n"
        "    runs-on: ubuntu-latest\n"
        "    continue-on-error: true\n"
        "    steps:\n"
        "      - run: true\n"
    )
    assert _required_checks(workflow, "3.10") == {"gating"}


def test_only_the_gating_interpreter_of_a_matrix_job_is_required():
    """Branch protection gates one leg of the matrix. The other interpreters
    run on every PR but no rule waits on them, so counting them as required
    would document a gate that does not exist.
    """
    workflow = (
        "name: X\n"
        "\njobs:\n"
        "  test:\n"
        "    runs-on: ubuntu-latest\n"
        "    strategy:\n"
        "      matrix:\n"
        '        python-version: ["3.10", "3.11", "3.12"]\n'
        "    steps:\n"
        "      - run: true\n"
    )
    assert _required_checks(workflow, "3.10") == {"test (3.10)"}


def test_a_job_another_job_needs_is_not_itself_required():
    """A fanned-out job rolled up by an aggregator gates through the
    aggregator. Counting the legs as required would document gates branch
    protection does not have, and would make the leg count impossible to
    change without editing branch protection.
    """
    workflow = (
        "name: X\n"
        "\njobs:\n"
        "  leg:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: true\n"
        "\n"
        "  rollup:\n"
        "    runs-on: ubuntu-latest\n"
        "    needs: leg\n"
        "    if: always()\n"
        "    steps:\n"
        "      - run: true\n"
    )
    assert _required_checks(workflow, "3.10") == {"rollup"}


def test_matrix_without_the_gating_interpreter_is_an_error():
    """If the matrix stops running the minimum supported interpreter, the
    required check named in branch protection no longer exists. Fail loudly
    instead of quietly deriving a set with one fewer gate in it.
    """
    workflow = (
        "name: X\n"
        "\njobs:\n"
        "  test:\n"
        "    strategy:\n"
        "      matrix:\n"
        '        python-version: ["3.11", "3.12"]\n'
    )
    try:
        _required_checks(workflow, "3.10")
    except AssertionError:
        return
    raise AssertionError("expected a missing gating interpreter to raise")


def test_gating_interpreter_is_the_projects_minimum_supported_python():
    assert _gating_python_version() == "3.10"


def test_a_job_without_timeout_minutes_is_detected():
    """The rule as a property, on a workflow written here rather than on the
    repo's own files, so the detector is proven independently of whatever
    .github/workflows currently happens to contain.
    """
    workflow = (
        "name: X\n"
        "\njobs:\n"
        "  capped:\n"
        "    runs-on: ubuntu-latest\n"
        "    timeout-minutes: 10\n"
        "    steps:\n"
        "      - run: true\n"
        "\n"
        "  uncapped:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: true\n"
        "        timeout-minutes: 5\n"
    )
    assert _jobs_without_timeout(workflow) == ["uncapped"]


def test_every_workflow_job_declares_a_timeout():
    """A job with no `timeout-minutes` runs to GitHub's six-hour default. With
    branch protection on `strict: true`, one hung required job stalls the
    whole merge queue for six hours, since every queued PR must be up to date
    with `main` and cannot get there while the queue is blocked.

    PR #72 added timeouts to the four jobs that existed then and wrote the
    rule down in prose. `test-pandas3` was added afterwards without one, and
    became a required check in PR #111. Prose is not a mechanism; this is.
    """
    missing = {}
    for path in sorted((ROOT / ".github/workflows").glob("*.yml")):
        jobs = _jobs_without_timeout(path.read_text())
        if jobs:
            missing[path.name] = jobs
    assert not missing, (
        f"these workflow jobs declare no `timeout-minutes`: {missing!r}. "
        "Without one a hang runs for GitHub's six-hour default, and a "
        "required check that hangs blocks every merge in the queue for that "
        "whole time. Add a `timeout-minutes:` matching the sibling jobs."
    )


def test_documented_required_checks_match_actual_ci_jobs():
    expected = _expected_required_checks()
    documented = _documented_required_checks()
    assert documented == expected, (
        f"CONTRIBUTING.md documents {documented!r} as required checks, but "
        f"the workflows in .github/workflows/ actually gate on {expected!r}. "
        "Update CONTRIBUTING.md (and branch protection) to match."
    )
