"""Every tracked markdown doc must be mentioned in docs/MAP.md, or it's
invisible to the index the moment it's added. This only reaches files git
actually tracks: docs/AUTOPILOT.md, docs/HANDOFF.md, docs/ISSUES.md, and
friends are gitignored on purpose (maintainer working notes, see
docs/MAP.md's own "tracked vs untracked" section) and never exist in a CI
checkout at all, so there is no way for this test -- or anything running in
CI -- to enforce their presence in the map. Keeping those cross-linked is a
local, manual discipline; tools/check_docs_map.py is the local-only helper
for that half.
"""

import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Doc files that exist for a reason other than "read me": don't expect a
# cross-link to itself, and a license isn't a doc in the sense MAP.md indexes.
EXEMPT = {"docs/MAP.md", "LICENSE"}


def _tracked_markdown_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "*.md", "docs/*.md"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return [line for line in out.splitlines() if line and line not in EXEMPT]


def test_every_tracked_doc_is_mentioned_in_the_map():
    map_text = (ROOT / "docs/MAP.md").read_text()
    missing = [
        f for f in _tracked_markdown_files()
        if pathlib.Path(f).name not in map_text
    ]
    assert not missing, (
        f"docs/MAP.md never mentions {missing!r}. A tracked doc that isn't "
        "in the index is invisible the moment someone opens docs/MAP.md "
        "expecting it to be complete -- add a row/line for it there."
    )
