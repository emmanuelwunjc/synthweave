"""Group commit subjects into release notes, by Conventional Commit type.

`.github/workflows/release.yml` used to hand GitHub `generate_release_notes:
true`, which emits a flat list of squash-merged PR titles. The titles are
typed, so the information is there, but nothing groups it: a reader cannot
tell a new feature from a CI tweak without reading every line.

GitHub's own `.github/release.yml` categorisation is the obvious fix and does
not work here. It groups by PR *label*, and no merged PR in this repo carries
one (checked across #84 through #107: every label array is empty), so every
entry would land in the catch-all. The titles are the only structured signal
that actually exists, so the grouping is generated from those.

    python3 tools/release_notes.py               # since the previous v* tag
    python3 tools/release_notes.py v0.2.0..HEAD  # over an explicit range

The parsing rules and the section order are exercised by
tests/test_release_notes.py against real subjects from this repo's history.

`git log --no-merges` is deliberate, and it is the one place a subject really
is dropped. A merge commit's own subject ("Merge branch 'side'") is
bookkeeping: it names a branch, not a change, and it would land under "Other"
next to the untyped subjects that "never dropped" is about. Nothing else goes
with it, because the commits the merge joins are themselves inside the range
and still reach the notes. So the work is never lost, only the join is.
"""

import dataclasses
import re
import subprocess
import sys

# Section order is by what a reader of a release wants first, not alphabetical
# and not the order the types happen to appear. What changed for a user leads
# (breaking, then features and fixes, then performance and docs); what changed
# for a contributor trails. "Other" is the residue: subjects with no type at
# all, which must never be dropped -- the largest change in this project's
# history, "PII generators, real-data connectors, ..." (#24), landed untyped
# and is baked into the v0.2.0 notes.
SECTIONS = [
    ("breaking", "Breaking changes"),
    ("feat", "Features"),
    ("fix", "Fixes"),
    ("perf", "Performance"),
    ("docs", "Documentation"),
    ("style", "Style"),
    ("refactor", "Refactoring"),
    ("test", "Tests"),
    ("build", "Build"),
    ("ci", "CI"),
    ("chore", "Chores"),
    ("revert", "Reverts"),
    ("other", "Other"),
]

# `type(scope)!: description`. The `!` is captured rather than allowed inside
# the type, so `feat!:` is a breaking `feat` and not an unknown type called
# `feat!` that would quietly fall through to "Other".
_SUBJECT = re.compile(
    r"^(?P<type>[a-z]+)(?:\((?P<scope>[^)]*)\))?(?P<breaking>!)?: (?P<description>.+)$"
)

# The `(#NN)` GitHub appends when it squash-merges. Only a trailing one is the
# PR: `feat(mode): ... (#87) (#92)` cites issue #87 and was merged as #92.
_TRAILING_PR = re.compile(r"\s*\(#(?P<pr>\d+)\)$")

# A release tag, with an optional PEP 440 prerelease suffix. `release.yml`
# triggers on `tags: ["v*"]`, so `v0.3.0rc1` starts a real release run; a tag
# shape refused here stops that run at the step before `Publish to PyPI`, and
# publishing a release candidate becomes impossible until the tag is deleted.
_VERSION_TAG = re.compile(
    r"^v(?P<version>\d+(?:\.\d+)*)(?:(?P<phase>a|b|rc)(?P<serial>\d+))?$"
)

# a1 < b2 < rc1 < the final release, per PEP 440. A final release has no
# suffix, so it takes the rank above every prerelease of the same version.
_PHASES = {"a": 0, "b": 1, "rc": 2}
_FINAL = len(_PHASES)

_TYPES = {key for key, _ in SECTIONS}

EMPTY = "_No changes since the previous release._"


@dataclasses.dataclass(frozen=True)
class Entry:
    """One commit subject, taken apart."""

    type: str
    scope: str | None
    breaking: bool
    description: str
    pr: str | None


def parse(subject: str) -> Entry:
    """Take a commit subject apart. An unrecognised shape is type "other",
    never an error: a subject that does not parse still happened."""
    subject = subject.strip()
    pr = None
    match = _TRAILING_PR.search(subject)
    if match:
        pr = match.group("pr")
        subject = subject[: match.start()]

    match = _SUBJECT.match(subject)
    if not match or match.group("type") not in _TYPES:
        return Entry(
            type="other", scope=None, breaking=False, description=subject, pr=pr
        )
    return Entry(
        type=match.group("type"),
        scope=match.group("scope") or None,
        breaking=bool(match.group("breaking")),
        description=match.group("description"),
        pr=pr,
    )


def _line(entry: Entry, *, label_type: bool) -> str:
    """One bullet.

    The `(#NN)` reference is kept rather than stripped: GitHub autolinks a
    bare `#NN` in a release body, so it costs nothing, needs no repository
    URL hardcoded here, and gives the reader the way back to the discussion
    that a bare sentence does not.
    """
    prefix = entry.scope
    if label_type:
        prefix = f"{entry.type}({entry.scope})" if entry.scope else entry.type
    text = f"**{prefix}:** {entry.description}" if prefix else entry.description
    return f"- {text} (#{entry.pr})" if entry.pr else f"- {text}"


def group(subjects) -> dict[str, list[Entry]]:
    """Entries by section key. A breaking change is filed under "breaking"
    only, not also under its own type: one upgrade-blocking list a reader can
    scan beats the same line printed twice."""
    grouped: dict[str, list[Entry]] = {}
    for subject in subjects:
        if not subject.strip():
            continue
        entry = parse(subject)
        key = "breaking" if entry.breaking else entry.type
        grouped.setdefault(key, []).append(entry)
    return grouped


def render(subjects) -> str:
    """Markdown for the release body. A section with no entries emits no
    heading, so an empty range produces no headings at all."""
    grouped = group(subjects)
    blocks = []
    for key, heading in SECTIONS:
        entries = grouped.get(key)
        if not entries:
            continue
        lines = [_line(entry, label_type=key == "breaking") for entry in entries]
        blocks.append(f"### {heading}\n\n" + "\n".join(lines))
    if not blocks:
        return EMPTY + "\n"
    return "\n\n".join(blocks) + "\n"


def _version(tag: str):
    """A sort key for a release tag, or None if it is not one."""
    match = _VERSION_TAG.match(tag)
    if not match:
        return None
    release = tuple(int(part) for part in match.group("version").split("."))
    phase = match.group("phase")
    if phase is None:
        return (release, _FINAL, 0)
    return (release, _PHASES[phase], int(match.group("serial")))


def previous_tag(tags, current: str) -> str | None:
    """The highest release tag below `current`, or None if it is the first.

    Only `v`-prefixed version tags count: this repo also carries
    `archive/bug-hunt` and `wip/pre-worktree-split`, which are not releases.
    Ordering is numeric per component, so v0.10.0 sorts above v0.9.0 the way
    a string sort would not, and a prerelease sorts below the final release of
    the same version, so v0.3.0's notes cover only what landed after v0.3.0rc1.

    The tag returned is one taken from `tags`, never a name rebuilt from the
    parsed components: a tag spelled `v0.02.0` rebuilds as `v0.2.0`, which no
    ref matches, and `git log` then fails about an unknown revision rather
    than about the tag spelling.
    """
    here = _version(current)
    if here is None:
        raise ValueError(f"{current!r} is not a vN.N.N release tag")
    versioned = ((_version(tag), tag) for tag in tags)
    earlier = [(v, tag) for v, tag in versioned if v is not None and v < here]
    if not earlier:
        return None
    return max(earlier, key=lambda pair: pair[0])[1]


def _git(*args) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout


def _default_range() -> str:
    """Everything since the previous release tag, for the tag being built.

    `--match` is not decoration. `previous_tag` already filters non-release
    tags out of the earlier side of the range; the current side needs the same
    filter for the same reason. `git describe --exact-match` returns whichever
    tag on the commit sorts first by refname, so a housekeeping tag like
    `archive/bug-hunt` beats `v0.2.0` when both point at the release commit,
    and `previous_tag` then raises. That step runs before the PyPI publish, so
    a tag with nothing to do with releasing would block the release outright.
    """
    current = _git("describe", "--tags", "--exact-match", "--match", "v[0-9]*").strip()
    earlier = previous_tag(_git("tag", "--list").split(), current)
    return f"{earlier}..{current}" if earlier else current


def main(argv) -> int:
    commit_range = argv[0] if argv else _default_range()
    subjects = _git("log", "--no-merges", "--format=%s", commit_range).splitlines()
    sys.stdout.write(render(subjects))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
