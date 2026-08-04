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

The rest of the file pins another "silence reads as a pass" hole: a suite run
that never finished must not count as coverage (#130).

Nothing here runs the whole suite. The harness already runs `pytest tests/`
inside its sandbox, so a test that did the same would nest one full suite run
inside another every time the harness ran, without bound.
"""

import pathlib
import queue
import re
import subprocess
import sys

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
