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
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
EXEMPT = {"MAP.md"}


def main() -> int:
    map_text = (DOCS / "MAP.md").read_text()
    missing = [
        str(path.relative_to(ROOT))
        for path in sorted(DOCS.rglob("*.md"))
        if path.name not in EXEMPT and path.name not in map_text
    ]
    if not missing:
        print("Every .md file under docs/ is mentioned in docs/MAP.md.")
        return 0
    print("Not mentioned in docs/MAP.md:")
    for path in missing:
        print(f"  {path}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
