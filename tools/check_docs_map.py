"""Local-only companion to tests/test_docs_map_sync.py.

That test can only see git-tracked files, so it can never check the
gitignored majority of docs/ (AUTOPILOT.md, HANDOFF.md, ISSUES.md,
NEXT_STEPS.md, brainstorms/, specs/, research/) -- a CI checkout doesn't
have them, so CI cannot enforce anything about them. This script fills that
gap by scanning the real filesystem instead of git, but by the same token
it can only ever run locally: run it yourself after adding a new file under
docs/, the way you'd run tools/mutation_check.py after a fix.

    python3 tools/check_docs_map.py
"""

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
EXEMPT = {"docs/MAP.md"}

# Markdown structures that make a filename an entry a reader can follow: a
# table row, a list item, or a fenced block (the map's one-line tree).
_ENTRY_PREFIXES = ("|", "-", "*", "+")
_LINK = re.compile(r"\[[^\]]*\]\([^)]*\)")


def _entry_text(map_text: str):
    """Yield only the parts of the map that index something.

    A bare substring search over the whole file counts any prose sentence
    that happens to name a file, which reads as covered while leaving the
    reader nothing to follow. Prose is what this filters out.
    """
    yield from (match.group(0) for match in _LINK.finditer(map_text))
    in_fence = False
    for raw in map_text.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or line.startswith(_ENTRY_PREFIXES):
            yield line


def is_indexed(map_text: str, doc_path: str) -> bool:
    """Whether docs/MAP.md actually indexes `doc_path`, not merely names it.

    The whole repo-relative path has to appear, bounded on both sides so it
    is the entry's own target and not part of a longer one. Matching the
    bare filename by substring was wrong twice over: `docs/GUIDE.md`'s row
    then covered any future `docs/<dir>/GUIDE.md`, and an entry for
    `MYGUIDE.md` marked `GUIDE.md` indexed while pointing somewhere else.

    Shared with tests/test_docs_map_sync.py on purpose: the test and this
    script check the same repo from different angles (git vs the filesystem),
    and two copies of the rule would drift.
    """
    # A path character on either side means the match is a fragment of some
    # longer path, so it indexes a different file.
    pattern = re.compile(rf"(?<![\w./-]){re.escape(doc_path)}(?![\w./-])")
    return any(pattern.search(entry) for entry in _entry_text(map_text))


# A target a reader could follow: a token with a directory component and
# either a file extension or a trailing slash. That deliberately excludes
# bare names (`NEXT_STEPS.md` in a sentence about it, the relative leaves of
# the map's one-line tree) and non-paths that contain a slash (`area/*`
# label globs), neither of which names a location to resolve.
_TARGET = re.compile(r"^[\w.-]+(?:/[\w.-]+)*/(?:[\w.-]+\.[a-zA-Z0-9]+|)$")


def referenced_paths(map_text: str) -> list[str]:
    """Repo-relative paths the map's entries point at, in order, deduped."""
    found = []
    for entry in _entry_text(map_text):
        candidates = re.findall(r"`([^`]*)`", entry) + [
            match.group(1) for match in re.finditer(r"\]\(([^)]*)\)", entry)
        ]
        for candidate in candidates:
            for token in candidate.split():
                # Trailing sentence punctuation only. A leading dot is part
                # of the path (`.github/pull_request_template.md`).
                token = token.rstrip(".,;:!?'\")")
                if _TARGET.match(token) and token not in found:
                    found.append(token)
    return found


def _git_ignores(root: pathlib.Path):
    """Predicate for "git deliberately does not track this path".

    Most of docs/ is gitignored maintainer notes that the map indexes on
    purpose, and they are absent from a fresh clone. Their absence is
    expected, so it must not be reported as a broken entry. If git is not
    available, nothing can be told apart, so treat every target as ignored
    rather than inventing failures.
    """
    def ignored(path: str) -> bool:
        try:
            result = subprocess.run(
                ["git", "check-ignore", "-q", path],
                cwd=root, capture_output=True,
            )
        except FileNotFoundError:
            return True
        # 0 = ignored, 1 = not ignored, anything else = git could not tell.
        return result.returncode != 1

    return ignored


def missing_targets(map_text: str, root: pathlib.Path, is_ignored=None) -> list[str]:
    """Entries whose target does not exist and is not gitignored.

    An entry pointing at a deleted or misspelled file used to fail nothing at
    all: the check only ran forwards, from file to map, so the map could
    promise a document that was not there.
    """
    if is_ignored is None:
        is_ignored = _git_ignores(root)
    return [
        path
        for path in referenced_paths(map_text)
        if not (root / path).exists() and not is_ignored(path)
    ]


def main() -> int:
    map_text = (DOCS / "MAP.md").read_text()
    unindexed = [
        str(path.relative_to(ROOT).as_posix())
        for path in sorted(DOCS.rglob("*.md"))
        if path.relative_to(ROOT).as_posix() not in EXEMPT
        and not is_indexed(map_text, path.relative_to(ROOT).as_posix())
    ]
    dangling = missing_targets(map_text, ROOT)
    if not unindexed and not dangling:
        print("Every .md file under docs/ is indexed in docs/MAP.md.")
        return 0
    if unindexed:
        print("Not indexed in docs/MAP.md:")
        for path in unindexed:
            print(f"  {path}")
    if dangling:
        print("Indexed in docs/MAP.md but not present (and not gitignored):")
        for path in dangling:
            print(f"  {path}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
