"""tools/release_notes.py turns a range of commit subjects into grouped
release notes. The subjects here are real ones from this repo's own
`git log`, not invented strings: the generator only has to work on what
actually gets merged, and the one commit with no type prefix ("PII
generators, real-data connectors, ..." (#24), baked into the v0.2.0 notes)
is the case a naive parser silently drops.
"""

import importlib.util
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_release_notes():
    """tools/ is not an importable package. Load the script by path, the way
    tests/test_docs_map_sync.py loads tools/check_docs_map.py."""
    spec = importlib.util.spec_from_file_location(
        "release_notes", ROOT / "tools" / "release_notes.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release_notes = _load_release_notes()


def _sections(markdown: str) -> list[str]:
    """The section headings, in the order they appear."""
    return [
        line.removeprefix("### ").strip()
        for line in markdown.splitlines()
        if line.startswith("### ")
    ]


@pytest.mark.parametrize(
    "subject, section",
    [
        ("feat: ship sw.check_rule() as a public rule conformance harness (#79)",
         "Features"),
        ("fix(noise): require one rate per row from a noise rate function (#101)",
         "Fixes"),
        ("perf(ci): shard and copy-isolate mutation-check (#105)",
         "Performance"),
        ("docs: add docs/MAP.md, an index of what covers what (#78)",
         "Documentation"),
        ("refactor(connectors): unify fetch/cache into a shared _fetch.py module (#91)",
         "Refactoring"),
        ("test: strengthen two of five weak assertions from #49 (#83)",
         "Tests"),
        ("ci: require Conventional Commit PR titles (#67)",
         "CI"),
        ("build: bump the wheel metadata",
         "Build"),
        ("chore: keep .claude/skills/ local, like the other agent instructions (#99)",
         "Chores"),
    ],
)
def test_each_type_lands_in_its_own_section(subject, section):
    markdown = release_notes.render([subject])
    assert _sections(markdown) == [section]


def test_an_untyped_subject_reaches_other_instead_of_vanishing():
    """The real #24 case. A generator that only understands `type: desc`
    would drop the single largest change in the project's history."""
    subject = (
        "PII generators, real-data connectors, fidelity reporting, "
        "and docs overhaul (#24)"
    )
    markdown = release_notes.render([subject])
    assert _sections(markdown) == ["Other"]
    assert "PII generators, real-data connectors" in markdown


def test_a_scoped_type_parses_as_that_type_and_keeps_its_scope():
    entry = release_notes.parse(
        "fix(noise): require one rate per row from a noise rate function (#101)"
    )
    assert entry.type == "fix"
    assert entry.scope == "noise"
    assert entry.description == "require one rate per row from a noise rate function"


def test_a_breaking_marker_is_not_swallowed():
    """`feat!:` must not parse as a type of `feat!` (which would fall to
    Other) nor as a plain `feat` (which would hide the break)."""
    entry = release_notes.parse("feat!: drop the legacy Rule tuple form (#120)")
    assert entry.type == "feat"
    assert entry.breaking is True

    markdown = release_notes.render(
        [
            "feat!: drop the legacy Rule tuple form (#120)",
            "fix(io)!: reject a CSV chunk with mismatched columns (#70)",
        ]
    )
    assert _sections(markdown) == ["Breaking changes"]
    assert "drop the legacy Rule tuple form" in markdown
    assert "reject a CSV chunk with mismatched columns" in markdown


def test_a_breaking_change_is_listed_once_not_also_under_its_type():
    markdown = release_notes.render(
        [
            "feat!: drop the legacy Rule tuple form (#120)",
            "feat: ship sw.check_rule() as a public rule conformance harness (#79)",
        ]
    )
    assert _sections(markdown) == ["Breaking changes", "Features"]
    assert markdown.count("drop the legacy Rule tuple form") == 1


def test_an_empty_range_emits_no_headers():
    markdown = release_notes.render([])
    assert _sections(markdown) == []
    assert "###" not in markdown


def test_an_absent_type_emits_no_header():
    markdown = release_notes.render(["docs: add docs/MAP.md (#78)"])
    assert _sections(markdown) == ["Documentation"]


def test_section_order_is_by_reader_interest_not_alphabetical():
    """Every type at once, deliberately shuffled on input. Features and fixes
    lead; contributor-facing housekeeping and the untyped residue trail."""
    subjects = [
        "chore: pin pandas<3 (#84)",
        "docs: add docs/MAP.md (#78)",
        "PII generators, real-data connectors, fidelity reporting (#24)",
        "fix(noise): require one rate per row (#101)",
        "ci: require Conventional Commit PR titles (#67)",
        "feat!: drop the legacy Rule tuple form (#120)",
        "test: strengthen two of five weak assertions (#83)",
        "perf(ci): shard and copy-isolate mutation-check (#105)",
        "refactor(connectors): unify fetch/cache (#91)",
        "build: bump the wheel metadata",
        "feat: ship sw.check_rule() (#79)",
    ]
    assert _sections(release_notes.render(subjects)) == [
        "Breaking changes",
        "Features",
        "Fixes",
        "Performance",
        "Documentation",
        "Refactoring",
        "Tests",
        "Build",
        "CI",
        "Chores",
        "Other",
    ]


def test_the_trailing_pr_number_is_kept_as_a_reference_not_left_inline():
    """GitHub autolinks a bare `#NN` in a release body, so the reference is
    worth keeping. It is parsed out so an entry that carries both an issue
    and a PR number keeps them distinguishable."""
    entry = release_notes.parse(
        "feat(mode): sw.Mode base class + Mode.metadata() end to end (#87) (#92)"
    )
    assert entry.pr == "92"
    assert entry.description == "sw.Mode base class + Mode.metadata() end to end (#87)"
    assert release_notes.render([
        "feat(mode): sw.Mode base class + Mode.metadata() end to end (#87) (#92)"
    ]).rstrip().endswith("(#92)")


def test_a_subject_with_no_pr_number_renders_without_a_dangling_reference():
    markdown = release_notes.render(["build: bump the wheel metadata"])
    assert "(#" not in markdown
    assert "bump the wheel metadata" in markdown


def test_the_previous_tag_is_the_nearest_earlier_version_tag():
    """`git tag` in this repo also lists `archive/bug-hunt` and
    `wip/pre-worktree-split`, which are not releases. Picking the wrong one
    would generate notes over the wrong range."""
    tags = ["archive/bug-hunt", "v0.1.0", "v0.2.0", "wip/pre-worktree-split"]
    assert release_notes.previous_tag(tags, "v0.3.0") == "v0.2.0"
    assert release_notes.previous_tag(tags, "v0.2.0") == "v0.1.0"
    assert release_notes.previous_tag(tags, "v0.1.0") is None


def test_version_tags_are_ordered_numerically_not_as_strings():
    """A plain string sort puts v0.10.0 before v0.9.0, which would silently
    generate notes over a range that has already been released."""
    tags = ["v0.9.0", "v0.10.0"]
    assert release_notes.previous_tag(tags, "v1.0.0") == "v0.10.0"


def _run(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def test_the_range_ignores_a_non_release_tag_on_the_release_commit(tmp_path, monkeypatch):
    """`previous_tag` already filters non-release tags out of the *earlier*
    side of the range. The current side needs the same filter: this repo
    carries `archive/bug-hunt` and `wip/pre-worktree-split`, and a
    `git describe --exact-match` that is allowed to return one of those makes
    the release job die at the notes step, which sits before the PyPI publish.
    A tag that has nothing to do with releasing then blocks a release.
    """
    _run(tmp_path, "init", "-q", "-b", "main")
    # `.test` is an RFC 2606 reserved TLD, so this cannot be a real mailbox.
    _run(tmp_path, "config", "user.email", "t@example.test")
    _run(tmp_path, "config", "user.name", "t")
    (tmp_path / "a.txt").write_text("one\n")
    _run(tmp_path, "add", "a.txt")
    _run(tmp_path, "commit", "-qm", "feat: the first thing")
    _run(tmp_path, "tag", "v0.1.0")
    (tmp_path / "a.txt").write_text("two\n")
    _run(tmp_path, "commit", "-qam", "fix: the second thing")
    _run(tmp_path, "tag", "v0.2.0")
    # A non-release tag sharing the release commit. `git describe` picks the
    # tag whose refname sorts first, so `archive/bug-hunt` beats `v0.2.0`.
    _run(tmp_path, "tag", "archive/bug-hunt")

    monkeypatch.chdir(tmp_path)

    assert release_notes._default_range() == "v0.1.0..v0.2.0"


def _init_repo(tmp_path):
    """A real one-commit-per-call git repo, for the tests that exercise the
    git layer rather than the string layer."""
    _run(tmp_path, "init", "-q", "-b", "main")
    # `.test` is an RFC 2606 reserved TLD, so this cannot be a real mailbox.
    _run(tmp_path, "config", "user.email", "t@example.test")  # leak-guard: allow
    _run(tmp_path, "config", "user.name", "t")


def _commit(tmp_path, subject, text):
    (tmp_path / "a.txt").write_text(text)
    _run(tmp_path, "add", "a.txt")
    _run(tmp_path, "commit", "-qm", subject)


def test_a_prerelease_tag_on_the_release_commit_still_yields_a_range(
    tmp_path, monkeypatch
):
    """`release.yml` triggers on `tags: ["v*"]`, so `v0.3.0rc1` starts a real
    release run. A tag shape the notes generator refuses stops that run at the
    step before `Publish to PyPI`, which makes publishing a release candidate
    impossible until someone deletes the tag."""
    _init_repo(tmp_path)
    _commit(tmp_path, "feat: the first thing", "one\n")
    _run(tmp_path, "tag", "v0.2.0")
    _commit(tmp_path, "fix: the second thing", "two\n")
    _run(tmp_path, "tag", "v0.3.0rc1")

    monkeypatch.chdir(tmp_path)

    assert release_notes._default_range() == "v0.2.0..v0.3.0rc1"


def test_a_prerelease_orders_below_the_final_release_of_the_same_version():
    """v0.3.0rc1 precedes v0.3.0, so the notes for the final release cover
    only what landed after the candidate, not the whole range again."""
    tags = ["v0.2.0", "v0.3.0a1", "v0.3.0b2", "v0.3.0rc1", "v0.3.0"]
    assert release_notes.previous_tag(tags, "v0.3.0") == "v0.3.0rc1"
    assert release_notes.previous_tag(tags, "v0.3.0rc1") == "v0.3.0b2"
    assert release_notes.previous_tag(tags, "v0.3.0b2") == "v0.3.0a1"
    assert release_notes.previous_tag(tags, "v0.3.0a1") == "v0.2.0"


def test_the_previous_tag_is_a_tag_that_exists_rather_than_a_rebuilt_name():
    """A name rebuilt from parsed components loses the spelling: `v0.02.0`
    comes back as `v0.2.0`, which no ref matches, so `git log` then fails
    about an unknown revision instead of about the tag."""
    assert release_notes.previous_tag(["v0.02.0"], "v0.3.0") == "v0.02.0"


def test_main_writes_the_rendered_body_for_an_explicit_range(
    tmp_path, monkeypatch, capsys
):
    """The whole reason the tool exists is what lands on stdout during
    `Generate the release notes`. Nothing asserted that end to end."""
    _init_repo(tmp_path)
    _commit(tmp_path, "chore: set the base", "base\n")
    _run(tmp_path, "tag", "v0.1.0")
    _commit(tmp_path, "feat: ship the thing (#1)", "one\n")
    _commit(tmp_path, "fix(io): stop dropping a column (#2)", "two\n")

    monkeypatch.chdir(tmp_path)

    assert release_notes.main(["v0.1.0..HEAD"]) == 0
    assert capsys.readouterr().out == (
        "### Features\n"
        "\n"
        "- ship the thing (#1)\n"
        "\n"
        "### Fixes\n"
        "\n"
        "- **io:** stop dropping a column (#2)\n"
    )


def test_a_merge_commits_own_work_still_reaches_the_notes(
    tmp_path, monkeypatch, capsys
):
    """`--no-merges` drops the merge commit itself, and only that. The branch
    commits it joins are still in the range, so the work reaches the notes and
    only the bookkeeping subject is absent. This is what keeps #114's "never
    dropped" claim about untyped subjects true."""
    _init_repo(tmp_path)
    _commit(tmp_path, "chore: set the base", "base\n")
    _run(tmp_path, "tag", "v0.1.0")
    _run(tmp_path, "checkout", "-q", "-b", "side")
    _commit(tmp_path, "an untyped thing done on a branch", "side\n")
    _run(tmp_path, "checkout", "-q", "main")
    _run(tmp_path, "merge", "-q", "--no-ff", "-m", "Merge branch 'side'", "side")

    monkeypatch.chdir(tmp_path)

    release_notes.main(["v0.1.0..HEAD"])
    body = capsys.readouterr().out
    assert "an untyped thing done on a branch" in body
    assert "Merge branch" not in body
