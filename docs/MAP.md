# Docs map

This is the index. If you are not sure which file has what you need, start
here instead of opening files to check. It does not replace any file's own
reading-order instructions (`AUTOPILOT.md` and `HANDOFF.md` each already say
what to read and in what order for their audience) -- it exists for the
question those don't answer: "which file, out of all of them, covers topic X."

Two kinds of file live under `docs/`. **Tracked** (in git, present in any
clone): `GUIDE.md`, this file, and `.gitignore`'s exceptions in general.
**Untracked** (gitignored, exist only on the maintainer's machine, absent
from a fresh clone): everything else here -- `AUTOPILOT.md`, `HANDOFF.md`,
`ISSUES.md`, `NEXT_STEPS.md`, `brainstorms/`, `reference-data/`, `research/`,
`specs/`. If you cloned this repo fresh and a path below is missing, that is
why -- it is not broken, it is a maintainer working file that was never
meant to ship.

## Quick lookup: "I need to know about X"

| You want... | Read |
|---|---|
| the public API, how to use synthweave | `README.md` -> `docs/GUIDE.md` |
| how to contribute code, the PR process | `CONTRIBUTING.md` |
| what to do *right now* as an agent picking up an issue | `docs/AUTOPILOT.md`, and stop there |
| what happened last session, current live state/hazards | `docs/HANDOFF.md` |
| what's queued next, the roadmap | `docs/research/BLUEPRINT.md` (`NEXT_STEPS.md` is ~70% stale and being folded in -- see "Known overlaps" below) |
| live, actionable, labeled work items | GitHub Issues (`gh issue list`) -- the tracker of record, not a file in this repo |
| the narrative "why a past bug happened and how the fix works" | `docs/ISSUES.md` |
| deep technical research behind a design decision (constraints, calibration, utility metrics, combining rules, panel dynamics) | `docs/research/METHODOLOGY.md` |
| why synthweave is positioned the way it is vs. SDV/synthcity/synthpop/etc. | `docs/research/FIELD_SURVEY.md` |
| how the roadmap breaks into session-sized chunks | `docs/research/WORK_PACKAGES.md` |
| how a feature was originally scoped and why | `docs/brainstorms/` (currently `2026-07-29-synthweave-scope-requirements.md`) |
| the original v0.1 product spec | `docs/specs/synthweave-v0.1.md` |
| the spec that seeded the 2026-08 bug hunt (I1-I15 in `ISSUES.md`) | `docs/specs/synthweave-v0.1-bug-hunt.md` |
| reporting a security issue | `SECURITY.md` |
| the checklist a PR is expected to satisfy | `.github/pull_request_template.md` |
| what an `area/*` GitHub label (e.g. `area/connectors`) actually covers, or which lane owns a given file | `docs/AUTOPILOT.md` -> "Your lane" (the ownership table; `area/*` labels are for browsing, not lane assignment) |

## Reading order by role

- **Agent picking up a queued issue**: `docs/AUTOPILOT.md` only. It explicitly tells you not to read `NEXT_STEPS.md`, `HANDOFF.md`, or `docs/research/` -- follow that; they are maintainer strategy docs and reading them slows you down without changing what you need to do.
- **Maintainer or anyone planning the next phase of work**: `docs/HANDOFF.md` -> `docs/research/BLUEPRINT.md` -> `gh issue list`.
- **New contributor**: `README.md` -> `docs/GUIDE.md` -> `CONTRIBUTING.md`.
- **Anyone wanting the history of how synthweave got here**: `docs/brainstorms/` -> `docs/specs/` -> `docs/research/` -> `docs/ISSUES.md`, roughly in that chronological order.

## The full tree, one line each

```
README.md                          public pitch + quickstart          tracked
CONTRIBUTING.md                    contributor process                tracked
SECURITY.md                        vulnerability disclosure           tracked
docs/
  MAP.md                           this file                          tracked
  GUIDE.md                         glossary + API reference + tutorial  tracked
  AUTOPILOT.md                     agent dispatch: read first, stop there  untracked
  HANDOFF.md                       point-in-time session state/hazards   untracked
  ISSUES.md                        narrative bug-postmortem log (I1-I40) untracked
  NEXT_STEPS.md                    backlog, ~70% stale, folding into BLUEPRINT  untracked
  brainstorms/                     original scope/requirements decisions  untracked
  specs/                           v0.1 product spec + bug-hunt spec      untracked
  research/
    BLUEPRINT.md                   execution plan: what to build, what order, what "done" means  untracked
    WORK_PACKAGES.md                the plan broken into session-sized chunks with a dependency graph  untracked
    METHODOLOGY.md                 sourced technical research backing BLUEPRINT's decisions  untracked
    FIELD_SURVEY.md                competitive positioning research  untracked
  reference-data/                  data files (names/ etc.), not documentation  untracked
```

## Known overlaps -- acknowledged, not contradictions

These are real duplications the docs themselves already partly flag. Listed
here so nobody re-discovers them from scratch or assumes they are an
oversight; they are maintainer-owned reconciliation work
(`AUTOPILOT.md`: "the maintainer reconciles `docs/`"), tracked under GitHub
issue #57 ("Tracking: documentation consistency") unless noted otherwise.

- **`NEXT_STEPS.md` vs `docs/research/BLUEPRINT.md` vs GitHub milestones.**
  All three now encode the same version-gated roadmap (v0.2->v1.0+ / Phase
  1-3 / M1-M4). `HANDOFF.md` and `BLUEPRINT.md` both already say
  `NEXT_STEPS.md` is stale and slated to collapse into `BLUEPRINT.md`;
  `NEXT_STEPS.md` itself doesn't know that yet, and nothing links the
  GitHub milestones back to either doc.
- **`docs/ISSUES.md` vs GitHub Issues.** Different in kind, not duplicated:
  GitHub is the live, labeled, actionable tracker; `ISSUES.md` is a
  narrative postmortem log of bugs already found and fixed, with full
  reasoning GitHub doesn't retain once an issue closes. Nothing currently
  cross-references one from the other.
- **`docs/brainstorms/` vs `docs/specs/synthweave-v0.1.md`.** Near-duplicate
  problem framing (same competitor comparisons, same three-stage pipeline
  decision) written twice in two genres -- decision record, then spec --
  with no cross-reference between them.
- **The 2026-08-02 mutation-harness incident** (a concurrent `git checkout`
  aborting a running `tools/mutation_check.py`, and the reverted snippet
  getting committed as if it were real code) is told nearly verbatim in
  three places: `docs/AUTOPILOT.md`'s stop banner, `docs/HANDOFF.md`'s
  "Live state" section, and `~/CLAUDE.md`'s "Concurrent sessions" section.
  Same anecdote, same causal chain, same remediation, restated three times
  with no link between them.

None of the above is a contradiction -- each file's own content is internally
correct. The gap is that the *relationship* between them is only partly
written down. This map exists to close that gap for lookup purposes; actually
merging/deprecating the overlapping content is issue #57's job, not this
file's.

## Keeping this map from going stale

Two mechanisms, matching the two kinds of file above:

- **Tracked docs**: `tests/test_docs_map_sync.py` runs in CI on every push
  and PR. It fails if a git-tracked `.md` file isn't mentioned somewhere in
  this file -- add a new tracked doc, forget to link it here, CI catches it.
- **Untracked docs**: CI can't see gitignored files at all, so nothing here
  can be enforced by a required check. `python3 tools/check_docs_map.py`
  does the same comparison against the real filesystem instead of git, but
  by the same token it only ever runs locally -- run it yourself after
  adding a file under `docs/`, the same discipline as
  `tools/mutation_check.py` after a fix.
