"""`tools/mutation_check.py --shard i/N` splits the mutation list across CI
runners. The split is only safe if every entry lands in exactly one shard, so
that is asserted here rather than trusted.

Two ways it could go wrong, both silent, both fatal to the guarantee the
harness exists to provide:

  - The slicing arithmetic drops an entry when the count does not divide the
    list evenly. A skipped mutation reports nothing, and "nothing was checked"
    reads exactly like "the check passed".
  - The `N` in the CI command and the length of the CI matrix disagree. The
    harness cannot see the matrix, so it cannot catch that itself; this test
    can, and does.

Workflow parsing is regex-based for the same reason as `test_ci_docs_sync.py`:
PyYAML is not a dependency, and adding one to read one small file is a worse
trade than a narrow regex.

The rest of the file pins the harness's other two "silence reads as a pass"
holes: a suite run that never finished must not count as coverage (#130), and
the sandbox's blind spot must be declared and no wider than declared (#131).

Nothing here runs the whole suite. The harness already runs `pytest tests/`
inside its sandbox, so a test that did the same would nest one full suite run
inside another every time the harness ran, without bound.
"""

import os
import pathlib
import queue
import re
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import mutation_check  # noqa: E402


def _ci_text() -> str:
    return (ROOT / ".github/workflows/ci.yml").read_text()


def test_shards_cover_every_mutation_exactly_once():
    """Across every plausible shard count, including ones that do not divide
    the list evenly, the shards partition the list."""
    total = len(mutation_check.MUTATIONS)
    for count in range(1, 13):
        seen = []
        for index in range(1, count + 1):
            seen.extend(mutation_check.shard(mutation_check.MUTATIONS, f"{index}/{count}"))
        names = [entry[0] for entry in seen]
        assert len(names) == total, f"N={count} produced {len(names)} entries, not {total}"
        assert sorted(names) == sorted(entry[0] for entry in mutation_check.MUTATIONS)


def test_no_shard_spec_runs_everything():
    """The plain `python3 tools/mutation_check.py` invocation people run
    locally must not quietly become a subset."""
    assert mutation_check.shard(mutation_check.MUTATIONS, None) == list(mutation_check.MUTATIONS)


def test_a_shard_index_outside_the_count_is_refused():
    for spec in ("0/6", "7/6", "1/0", "one/6", "1"):
        try:
            mutation_check.shard(mutation_check.MUTATIONS, spec)
        except SystemExit:
            continue
        raise AssertionError(f"--shard {spec} should have been refused")


def test_ci_shard_matrix_matches_the_shard_count_in_the_command():
    """The matrix legs and the `/N` in the command are written in two places
    in ci.yml. If they drift, some mutations never run and the required check
    still goes green. This is the mechanism that stops that."""
    text = _ci_text()
    matrix = re.search(r"^\s*shard:\s*\[([^\]]+)\]", text, re.MULTILINE)
    assert matrix, "ci.yml declares no `shard:` matrix"
    legs = sorted(int(v.strip()) for v in matrix.group(1).split(","))

    commands = re.findall(r"--shard \$\{\{ matrix\.shard \}\}/(\d+)", text)
    assert commands, "ci.yml runs no sharded mutation_check command"
    assert len(set(commands)) == 1, f"ci.yml declares more than one shard count: {commands!r}"
    count = int(commands[0])

    assert legs == list(range(1, count + 1)), (
        f"ci.yml runs `--shard i/{count}` but its matrix legs are {legs!r}. "
        f"They must be exactly 1..{count}, or some mutations never run."
    )


def test_the_shard_matrix_gates_through_a_job_named_mutation_check():
    """Branch protection lists `mutation-check` by name and is deliberately
    not being changed, so the aggregator job has to keep that exact id and
    has to actually depend on the shards."""
    text = _ci_text()
    assert re.search(r"^  mutation-check:\s*$", text, re.MULTILINE), (
        "ci.yml has no `mutation-check` job; branch protection requires that "
        "check by name and would wait forever for it"
    )
    aggregator = text.split("\n  mutation-check:\n", 1)[1]
    assert re.search(r"^    needs:.*mutation-shard", aggregator, re.MULTILINE), (
        "`mutation-check` must `needs: mutation-shard`, or it reports success "
        "without the shards having run"
    )
    assert re.search(r"^    if: always\(\)\s*$", aggregator, re.MULTILINE), (
        "`mutation-check` must carry `if: always()`. Without it a failing "
        "shard leaves the aggregator *skipped*, and a skipped required check "
        "may not block merging."
    )


# --- #130 an unfinished suite run must not count as coverage ---------------
#
# `run_suite` used to answer one boolean, so "the suite ran and went red" and
# "the suite never finished" were the same answer, and `check()` mapped both to
# CAUGHT. That is the one direction this harness must never fail in: it claims
# a fix is locked down when nothing checked it, and it exits 0 while doing so.

# A probe entry pointed at this harness's own file. Using a real file with a
# real snippet keeps `check()` on its normal path (neither STALE branch fires),
# and using *this* file rather than a src/ one means the probe does not go
# stale when someone else edits the library.
PROBE = ("probe entry", "tools/mutation_check.py", "MUTATIONS = [", "MUTATIONS = [  # probe\n")


def _sandbox_for_probe(tmp_path):
    """A directory `check()` can write the probe's mutated file into."""
    box = tmp_path / "w"
    (box / "tools").mkdir(parents=True, exist_ok=True)
    boxes: "queue.Queue" = queue.Queue()
    boxes.put(box)
    return boxes


def _fake_pytest(monkeypatch, outcomes):
    """Replace the pytest subprocess with a scripted sequence of outcomes.

    Each outcome is either an exit code, or the string "timeout" meaning the
    run never finished. Patching at the subprocess boundary rather than
    stubbing `run_suite` keeps `run_suite`'s own timeout handling under test.
    """
    completed = subprocess.CompletedProcess
    remaining = list(outcomes)

    def fake_run(cmd, **kwargs):
        outcome = remaining.pop(0) if remaining else 0
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=300)
        # An (exit code, summary line) pair where the summary matters.
        if isinstance(outcome, tuple):
            outcome, summary = outcome
        else:
            summary = "1 passed in 0.01s\n" if outcome == 0 else "1 failed in 0.01s\n"
        return completed(cmd, outcome, summary, "")

    monkeypatch.setattr(mutation_check.subprocess, "run", fake_run)


def test_a_suite_run_that_never_finished_is_not_reported_as_coverage(monkeypatch, tmp_path):
    """The property: an inconclusive run is inconclusive, not covered."""
    _fake_pytest(monkeypatch, ["timeout"])
    verdict, _, detail = mutation_check.check(PROBE, _sandbox_for_probe(tmp_path))
    assert verdict != "CAUGHT", (
        "a suite run that never finished was reported as coverage; a timeout "
        "proves nothing about the revert"
    )
    assert verdict == mutation_check.INCONCLUSIVE
    assert "timed out" in detail


def test_a_run_containing_an_unfinished_suite_exits_non_zero(monkeypatch, tmp_path, capsys):
    """The verdict has to reach the exit status too. A named verdict that
    still exits 0 leaves CI green over an unverified entry."""
    # Baseline passes, then the one mutation's suite never finishes.
    _fake_pytest(monkeypatch, [0, "timeout"])
    monkeypatch.setattr(mutation_check, "MUTATIONS", [PROBE])
    monkeypatch.setattr(
        mutation_check.shutil,
        "copytree",
        lambda src, dst, **kw: (pathlib.Path(dst) / "tools").mkdir(parents=True) or dst,
    )

    status = mutation_check.main([])

    out = capsys.readouterr().out
    assert mutation_check.INCONCLUSIVE in out, out
    assert status != 0, f"an unfinished suite run exited {status}:\n{out}"


def test_a_suite_that_did_finish_still_decides_caught_versus_missed(monkeypatch, tmp_path):
    """Guard against buying #130 by weakening the verdicts that already work."""
    _fake_pytest(monkeypatch, [1])
    assert mutation_check.check(PROBE, _sandbox_for_probe(tmp_path))[0] == "CAUGHT"
    _fake_pytest(monkeypatch, [0])
    assert mutation_check.check(PROBE, _sandbox_for_probe(tmp_path))[0] == "MISSED"


# --- #158 a catch has to be a catch *of the named property* ----------------
#
# Until now a mutation counted as CAUGHT whenever the suite went red, for any
# reason at all. Two entries on main were pinned by reds that said nothing
# about the property they name: one survived on whether a particular integer
# was divisible by 199, another on a crash in three unrelated tests that
# stayed red with the whole guarded clause deleted. Red-for-any-reason is the
# same failure shape as the ones already fixed here: silence reading as a pass.
#
# The mechanism is a fifth tuple element naming the test(s) that must be among
# the failures. These tests pin its three edges: a named catcher that passes,
# a named catcher that fails, and a named catcher that no longer exists.

PINNED = PROBE + (("tests/test_probe.py::test_probe",),)


def test_an_entry_may_name_the_tests_that_must_catch_it():
    """Four-element entries stay legal; a fifth element names the catchers."""
    assert mutation_check.catchers(PROBE) == ()
    assert mutation_check.catchers(PINNED) == ("tests/test_probe.py::test_probe",)


def test_a_red_suite_whose_named_catcher_passed_is_not_reported_as_coverage(monkeypatch, tmp_path):
    """The #158 property. The suite went red, but the test that owns the
    property did not, so the red is incidental and proves nothing."""
    # Full suite red, then the targeted catcher run green.
    _fake_pytest(monkeypatch, [1, 0])
    verdict, _, detail = mutation_check.check(PINNED, _sandbox_for_probe(tmp_path))
    assert verdict != mutation_check.CAUGHT, (
        "a red that no named catcher accounted for was reported as coverage"
    )
    assert verdict == mutation_check.INCIDENTAL
    assert "test_probe" in detail


def test_a_red_suite_whose_named_catcher_failed_is_caught(monkeypatch, tmp_path):
    _fake_pytest(monkeypatch, [1, 1])
    assert mutation_check.check(PINNED, _sandbox_for_probe(tmp_path))[0] == mutation_check.CAUGHT


def test_a_named_catcher_that_only_skipped_is_inconclusive_not_incidental(monkeypatch, tmp_path):
    """A pinned test can self-skip: `test_own_makes_a_view_safe_to_write_to`
    does exactly that under pandas 3, and `test-pandas3` is a required check.
    pytest exits 0 for an all-skipped selection, which would read as "the
    catcher passed", i.e. as INCIDENTAL. Nothing ran, so nothing was verified,
    and that is INCONCLUSIVE."""
    _fake_pytest(monkeypatch, [1, (0, "1 skipped in 0.01s\n")])
    verdict, _, detail = mutation_check.check(PINNED, _sandbox_for_probe(tmp_path))
    assert verdict == mutation_check.INCONCLUSIVE, f"got {verdict}: {detail}"


def test_a_named_catcher_that_no_longer_exists_is_stale_not_caught(monkeypatch, tmp_path):
    """pytest answers a renamed or deleted nodeid with exit 4, which the old
    `returncode != 0` read as a failing test, i.e. as a catch. A pin that
    points at nothing verified nothing, which is exactly STALE."""
    _fake_pytest(monkeypatch, [1, 4])
    verdict, _, detail = mutation_check.check(PINNED, _sandbox_for_probe(tmp_path))
    assert verdict == mutation_check.STALE, f"got {verdict}: {detail}"
    assert "test_probe" in detail


def test_an_unpinned_entry_keeps_the_old_behaviour(monkeypatch, tmp_path):
    """The 90-odd entries with no fifth element must not change meaning, and
    must not spend a second pytest run."""
    _fake_pytest(monkeypatch, [1])
    assert mutation_check.check(PROBE, _sandbox_for_probe(tmp_path))[0] == mutation_check.CAUGHT
    _fake_pytest(monkeypatch, [0])
    assert mutation_check.check(PROBE, _sandbox_for_probe(tmp_path))[0] == mutation_check.MISSED


def test_an_incidental_catch_exits_non_zero(monkeypatch, tmp_path, capsys):
    """A named verdict that still exits 0 leaves CI green over an entry that
    verified nothing, the same hole #130 closed for timeouts."""
    _fake_pytest(monkeypatch, [0, 1, 0])  # baseline, mutated suite red, catcher green
    monkeypatch.setattr(mutation_check, "MUTATIONS", [PINNED])
    monkeypatch.setattr(mutation_check, "report_blind_spot", lambda output: 0)
    monkeypatch.setattr(
        mutation_check.shutil,
        "copytree",
        lambda src, dst, **kw: (pathlib.Path(dst) / "tools").mkdir(parents=True) or dst,
    )

    status = mutation_check.main([])

    out = capsys.readouterr().out
    assert mutation_check.INCIDENTAL in out, out
    assert status != 0, f"an incidental catch exited {status}:\n{out}"


def test_a_pytest_usage_error_on_the_full_suite_is_not_a_catch(monkeypatch, tmp_path):
    """Same guard one level up: exit 4 on `pytest tests/` means no test ran,
    which is not evidence that the revert was noticed."""
    _fake_pytest(monkeypatch, [4])
    assert mutation_check.check(PROBE, _sandbox_for_probe(tmp_path))[0] != mutation_check.CAUGHT


def test_every_declared_catcher_names_a_test_that_exists():
    """The rename hazard, caught here in a one-second collection rather than
    only in a 100-entry harness run: a pinned nodeid that no longer resolves
    is a pin that verifies nothing."""
    declared = sorted(
        {nodeid for entry in mutation_check.MUTATIONS for nodeid in mutation_check.catchers(entry)}
    )
    if not declared:
        pytest.skip("no entry pins a catcher yet")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "--no-header",
         "-p", "no:cacheprovider", *declared],
        cwd=ROOT, env={**os.environ, "PYTHONPATH": "src"}, capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        "at least one pinned catcher no longer resolves to a test:\n"
        + proc.stdout + proc.stderr
    )


def test_every_entry_names_a_catcher():
    """An unpinned entry silently reverts to the pre-#158 behaviour.

    `catchers` has to stay syntactically optional, so an entry can be added and
    pinned in two steps and so `check()` keeps a defined answer for a
    four-element tuple. That leaves nothing stopping an entry from shipping
    unpinned, at which point it is CAUGHT by whatever went red first and its
    name never has to match what broke. That is exactly the hole #158 was
    filed to close, and a comment saying "pin your entry" does not close it.

    This is the mechanism rather than the request. It is a list comprehension
    rather than a harness run, so it costs milliseconds and fails in the
    `test` job long before the eight-minute `mutation-shard` job starts.
    """
    unpinned = [entry[0] for entry in mutation_check.MUTATIONS if not mutation_check.catchers(entry)]
    assert not unpinned, (
        "every MUTATIONS entry must name the test that owns its property:\n  "
        + "\n  ".join(unpinned)
        + "\nRun `python3 tools/mutation_check.py --audit` to see every test the "
        "revert breaks, then pin the one that asserts the property."
    )


def test_failing_tests_reads_the_node_ids_out_of_a_short_summary():
    """`--audit` reports which tests a mutation actually broke, so an
    incidental red can be seen rather than inferred."""
    output = (
        "FAILED tests/test_pipeline.py::test_a - ValueError: zero-size array\n"
        "FAILED tests/test_noise.py::test_b[Int64]\n"
        "2 failed, 507 passed in 14.02s\n"
    )
    assert mutation_check.failing_tests(output) == [
        "tests/test_noise.py::test_b[Int64]",
        "tests/test_pipeline.py::test_a",
    ]


# --- #131 the sandbox's blind spot is declared, and no wider ---------------

# Captured verbatim from `pytest tests/ -q --no-header -rs` inside a sandbox
# copy on main, so the expected values below come from real output rather than
# from re-deriving the parser's own logic.
_REAL_SHORT_SUMMARY = """\
SKIPPED [1] tests/test_docs_map_sync.py:51: not a git checkout (an sdist, say)
SKIPPED [11] tests/test_leak_guard.py:45: not a git checkout
SKIPPED [10] tests/test_leak_guard.py:70: not a git checkout
SKIPPED [1] tests/test_leak_guard.py:94: not a git checkout
SKIPPED [1] tests/test_leak_guard.py:181: not a git checkout
SKIPPED [1] tests/test_ssa_names.py:119: SSA_NAMES_ZIP not set
409 passed, 28 skipped in 7.81s
"""


def test_skips_by_module_totals_the_short_summary_per_file():
    assert mutation_check.skips_by_module(_REAL_SHORT_SUMMARY) == {
        "tests/test_docs_map_sync.py": 1,
        "tests/test_leak_guard.py": 23,
        "tests/test_ssa_names.py": 1,
    }


def test_blind_spot_names_only_modules_that_skip_more_in_the_sandbox():
    """A module that skips for the same reason in both trees is not a blind
    spot; only the extra skips the sandbox causes are."""
    checkout = "SKIPPED [1] tests/test_ssa_names.py:119: SSA_NAMES_ZIP not set\n"
    assert mutation_check.blind_spot(_REAL_SHORT_SUMMARY, checkout) == {
        "tests/test_docs_map_sync.py": 1,
        "tests/test_leak_guard.py": 23,
    }


def test_an_undeclared_blind_spot_is_reported_and_a_declared_one_is_not():
    """The gate. A blind spot nobody wrote down is how the gap widens in
    silence, which is the whole complaint in #131."""
    declared = {module: 1 for module in mutation_check.SANDBOX_BLIND_SPOT}
    assert mutation_check.undeclared_blind_spot(declared) == {}
    assert mutation_check.undeclared_blind_spot({**declared, "tests/test_new.py": 2}) == {
        "tests/test_new.py": 2
    }


def test_every_declared_blind_spot_module_exists():
    for module in mutation_check.SANDBOX_BLIND_SPOT:
        assert (ROOT / module).exists(), f"{module} is declared blind but is not a test module"


def test_the_declared_blind_spot_is_the_one_the_sandbox_actually_has(tmp_path):
    """Measured, not asserted from the source comment: each declared module
    runs in a real checkout and self-skips in a sandbox copy.

    Only the declared modules are run, not the whole suite. This module is
    excluded from that list even though it is declared, and that exclusion is
    load-bearing: without it, running this file spawns a run of this file,
    which spawns another, without bound. Confirmed the hard way, by a probe
    run that hung until it was killed.
    """
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout, so there is no skew to measure")

    own = str(pathlib.Path(__file__).resolve().relative_to(ROOT))
    modules = sorted(set(mutation_check.SANDBOX_BLIND_SPOT) - {own})
    box = tmp_path / "box"
    shutil.copytree(ROOT, box, ignore=shutil.ignore_patterns(*mutation_check.SANDBOX_SKIP), symlinks=True)

    def skips(cwd):
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", *modules, "-q", "--no-header", "-rs",
             "-p", "no:cacheprovider"],
            cwd=cwd, env={**os.environ, "PYTHONPATH": "src"}, capture_output=True, text=True,
        )
        return mutation_check.skips_by_module(proc.stdout + proc.stderr)

    in_checkout, in_sandbox = skips(ROOT), skips(box)
    assert in_checkout == {}, f"these modules skip in a real checkout too: {in_checkout}"
    assert sorted(in_sandbox) == modules, (
        f"the sandbox skips {sorted(in_sandbox)}, but the harness declares {modules}"
    )
