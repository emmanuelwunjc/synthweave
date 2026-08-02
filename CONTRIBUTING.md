# Contributing

## Setup

```bash
git clone https://github.com/emmanuelwunjc/synthweave.git
cd synthweave
pip install -e ".[dev]"
pre-commit install
```

`pre-commit install` is worth the one command: the same hooks run in CI, so
wiring them locally means you find formatting and lint problems in a second
rather than after a push.

## The four commands

```bash
PYTHONPATH=src python3 -m pytest tests/ -q                       # suite
PYTHONPATH=src python3 -m pyflakes src/ tests/ examples/ tools/  # lint
python3 tools/mutation_check.py                                  # coverage gate
pre-commit run --all-files                                       # hooks
```

Some connector tests need network access or a local data file. They skip
cleanly when it is unavailable, so an offline run is still a valid run.

## What "done" means here

**A fix is not done until reverting it turns a test red.**

`tools/mutation_check.py` enforces that. It reverts each logged fix one at a
time, runs the suite, and reports whether anything noticed. A fix nothing
catches is not a fix, and this is not hypothetical: two issues were once logged
as fixed while having no regression coverage at all, and reverting either left
the suite fully green.

So when you fix something:

1. Write the failing test first. Confirm it fails **for the reason you think**,
   not incidentally.
2. Fix it.
3. Add an entry to `MUTATIONS` in `tools/mutation_check.py` and confirm it
   reports `CAUGHT`.
4. Close the GitHub issue with a comment naming the fix and the test that
   covers it, so the reasoning survives outside the diff.

A `STALE` entry fails the run too. A mutation whose snippet no longer matches
the code verified nothing, and silently reads as a pass.

## Tests

Tests go through the public API only. Nothing reaches into a stage
implementation, which is what lets internals be rewritten without touching the
suite.

Two invariants are worth knowing because they catch the worst bugs:

- **Determinism.** Same schema and seed produce identical output, always.
- **Chunk invariance.** `chunk_size` is a memory knob and nothing else. If it
  can change a single value, every claim about scaling is unsound.

Helpers for both live in `tests/invariants.py`. New surfaces get one of each,
whether or not a chunking bug looks likely there.

If a test passes the moment you write it, make it fail on purpose before
trusting it. A test that cannot fail is not coverage.

## Branches, commits, PRs

- Branch from `main` as `type/short-description`, e.g. `fix/pii-cache-key`.
  Short-lived: hours or days, not weeks.
- [Conventional Commits](https://www.conventionalcommits.org/) for subjects:
  `type(scope): description`, where type is `feat`, `fix`, `chore`, `docs`,
  `refactor`, `test`, or `perf`. The body explains *why*, not *what*.
- One logical change per commit and per PR. Keep PRs to a few hundred lines
  across a handful of files. Pushing back on an oversized PR is correct.
- PRs squash-merge, so the PR title becomes the commit on `main` and appears
  verbatim in the generated release notes. Write it accordingly.

Required checks: `test (3.10)`, `pre-commit`, `mutation-check`,
`conventional`. Branch protection blocks merging until they pass, and a PR
must be up to date with `main` first.

Never push directly to `main`. Branch protection blocks it for everyone
except repo admins, but the rule holds regardless of who is technically able
to bypass it: every change goes through a PR, always.

## Working in parallel

Several sessions may work this repo at once. The rules above still hold; two
more apply.

**Own files, not features.** Before starting, agree which files each session
owns, and never edit a file you do not own. Reaching across is the failure
mode that corrupts a repo, and it is not caught by tests — both sides pass
locally and conflict at merge.

**`tools/mutation_check.py` is the one file everybody touches.** Every fix
appends to `MUTATIONS`. **Append at the end of the list, never in the
middle**, so a conflict resolves as one mechanical hunk instead of a
semantic merge. Merge those PRs one at a time.

**`src/synthweave/__init__.py` is shared and load-bearing** — it is the
public API surface. Say so in the PR title if you touch it, and expect to
serialise with anything else that does.

If a change needs a file you do not own, say so on the issue and stop.
Asking costs less than an untangling.

## Releases

Versions live in `pyproject.toml` and nowhere else. `__version__` reads it back
from installed metadata, so there is no second copy to drift.

Releasing is one push:

```bash
# bump version in pyproject.toml, merge that PR, then:
git tag v0.3.0
git push origin v0.3.0
```

`release.yml` builds the sdist and wheel from the tagged tree, publishes to PyPI
via OIDC Trusted Publishing, and creates the GitHub release with notes
generated from the PR titles since the previous tag. The workflow refuses to
publish if the tag and `pyproject.toml` disagree about the version.

This is the only path to PyPI on purpose: v0.1.0 was published from a folder
outside this repo, so the released artifact had no corresponding commit here.
Building from the tag makes that impossible to repeat.

## Reporting a bug

Open an issue. The most valuable ones describe **wrong output that raised no
error**, since a crash is cheap to find and a plausible-looking wrong number is
not.
