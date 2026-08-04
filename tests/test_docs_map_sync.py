"""Every tracked markdown doc must be indexed in docs/MAP.md, or it's
invisible to the index the moment it's added. This only reaches files git
actually tracks: docs/AUTOPILOT.md, docs/HANDOFF.md, docs/ISSUES.md, and
friends are gitignored on purpose (maintainer working notes, see
docs/MAP.md's own "tracked vs untracked" section) and never exist in a CI
checkout at all, so there is no way for this test -- or anything running in
CI -- to enforce their presence in the map. Keeping those cross-linked is a
local, manual discipline; tools/check_docs_map.py is the local-only helper
for that half.
"""

import importlib.util
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_check_docs_map():
    """tools/ is not an importable package, but the rule for "is this doc
    indexed?" must have exactly one definition or this test and the tool
    silently drift apart. Load the script by path rather than restating it.
    """
    spec = importlib.util.spec_from_file_location(
        "check_docs_map", ROOT / "tools" / "check_docs_map.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_docs_map = _load_check_docs_map()

# Doc files that exist for a reason other than "read me": don't expect a
# cross-link to itself, and a license isn't a doc in the sense MAP.md indexes.
EXEMPT = {"docs/MAP.md", "LICENSE"}


def _tracked_markdown_files() -> list[str]:
    """`*.md` already matches at any depth, so it covers docs/ too."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "*.md"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout
    except FileNotFoundError:
        pytest.skip("git is not installed, so the tracked-file list is unavailable")
    except subprocess.CalledProcessError:
        pytest.skip(
            "not a git checkout (an sdist, say), so there is no tracked-file "
            "list to compare the map against"
        )
    return [line for line in out.splitlines() if line and line not in EXEMPT]


def test_every_tracked_doc_is_indexed_in_the_map():
    map_text = (ROOT / "docs/MAP.md").read_text()
    missing = [
        f for f in _tracked_markdown_files()
        if not check_docs_map.is_indexed(map_text, f)
    ]
    assert not missing, (
        f"docs/MAP.md never indexes {missing!r}. A tracked doc that isn't "
        "in the index is invisible the moment someone opens docs/MAP.md "
        "expecting it to be complete -- add a row/line for it there."
    )


def test_a_doc_only_named_in_prose_is_not_indexed():
    """A filename can show up in a sentence without the map ever pointing at
    it. That reads as covered while leaving the reader no entry to follow, so
    the mention alone must not satisfy the check.
    """
    prose_only = (
        "# Docs map\n\nReport vulnerabilities the way SECURITY.md describes.\n"
    )
    assert not check_docs_map.is_indexed(prose_only, "SECURITY.md")


def test_a_doc_in_a_subdirectory_is_not_covered_by_its_namesake():
    """Matching on the bare filename made any new `docs/<dir>/GUIDE.md` pass
    on the strength of the existing `docs/GUIDE.md` row. The map indexes a
    file, not a name, so the whole path has to appear.
    """
    map_text = "| the API | `docs/GUIDE.md` |\n"
    assert check_docs_map.is_indexed(map_text, "docs/GUIDE.md")
    assert not check_docs_map.is_indexed(map_text, "docs/tutorial/GUIDE.md")


def test_a_longer_filename_does_not_index_the_shorter_one():
    """A substring test let an entry for `MYGUIDE.md` mark `GUIDE.md` as
    indexed. The reader following that entry lands on the wrong document.
    """
    map_text = "- **The other guide**: `docs/MYGUIDE.md`\n"
    assert not check_docs_map.is_indexed(map_text, "docs/GUIDE.md")


def test_an_entry_pointing_at_a_missing_file_is_reported(tmp_path):
    """The check only ever ran forwards, from file to map, so an entry whose
    target was deleted or misspelled stayed in the map indefinitely: it reads
    as an index and leads nowhere.
    """
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/GUIDE.md").write_text("# guide\n")
    map_text = "| the API | `docs/GUIDE.md` |\n| gone | `docs/OLD.md` |\n"
    missing = check_docs_map.missing_targets(
        map_text, tmp_path, is_ignored=lambda path: False
    )
    assert missing == ["docs/OLD.md"]


def test_a_gitignored_target_may_be_absent(tmp_path):
    """Most of docs/ is gitignored maintainer notes, absent from a fresh
    clone and from CI. The map indexes them on purpose and says so, so their
    absence is expected rather than a broken entry.
    """
    (tmp_path / "docs").mkdir()
    map_text = "| session state | `docs/HANDOFF.md` |\n"
    assert check_docs_map.missing_targets(
        map_text, tmp_path, is_ignored=lambda path: True
    ) == []


@pytest.mark.parametrize(
    "entry",
    [
        "| reporting a security issue | `SECURITY.md` |",
        "- **Reporting a vulnerability**: `SECURITY.md`",
        "```\nSECURITY.md    vulnerability disclosure    tracked\n```",
        "See [the disclosure policy](SECURITY.md) before filing.",
    ],
    ids=["table-row", "list-item", "tree-block", "link"],
)
def test_the_shapes_the_map_actually_uses_count_as_indexed(entry):
    """The map indexes in four shapes. All four have to keep passing, or the
    tightened check would just be a rename of "always fails".
    """
    assert check_docs_map.is_indexed(entry, "SECURITY.md")
