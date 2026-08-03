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
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
EXEMPT = {"MAP.md"}

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

    Shared with tests/test_docs_map_sync.py on purpose: the test and this
    script check the same repo from different angles (git vs the filesystem),
    and two copies of the rule would drift.
    """
    name = pathlib.PurePosixPath(doc_path).name
    return any(name in entry for entry in _entry_text(map_text))


def main() -> int:
    map_text = (DOCS / "MAP.md").read_text()
    missing = [
        str(path.relative_to(ROOT))
        for path in sorted(DOCS.rglob("*.md"))
        if path.name not in EXEMPT and not is_indexed(map_text, path.name)
    ]
    if not missing:
        print("Every .md file under docs/ is indexed in docs/MAP.md.")
        return 0
    print("Not indexed in docs/MAP.md:")
    for path in missing:
        print(f"  {path}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
