"""Refuse to commit local-only files, or PII-shaped content, to a public repo.

github.com/emmanuelwunjc/synthweave is public. The .gitignore header names
what is meant to "stay on this machine": the issue log, specs, brainstorms,
session handoffs, the agent instructions, and secrets. Several of those were
committed anyway, before the ignore rules existed, and are permanently in
public history.

.gitignore does not prevent a repeat. It is advisory in two ways that both
apply here: `git add -f` bypasses it outright, and a file that is already
tracked keeps being tracked no matter what the ignore rules say. This script
is the mechanism that closes both, per CLAUDE.md ("a hook that blocks it, CI
that fails on it, ... then prose").

It runs as a `local` pre-commit hook over the staged files, and in CI through
`pre-commit run --all-files` over every tracked file. Run it by hand with:

    python3 tools/check_no_private_leak.py <paths...>
    python3 tools/check_no_private_leak.py            # every staged file

Suppressing a line that is a deliberate example: append a comment containing
`leak-guard: allow` with the reason. Suppression is per line on purpose;
there is no file-level or global off switch.
"""

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# `.env` and `.env.local` are the only two spellings .gitignore lists, so
# check-ignore would wave through a `.env.production`. Secrets are the one
# category where a near-miss filename is worth catching directly.
_SECRET_FILENAME = re.compile(r"(^|/)\.env($|\.)")

# Where synthetic data is expected. This package's whole job is generating
# fake SSNs, names, emails and phone numbers, so those shapes under these
# roots are the product working, not a leak. Flagging them would make the
# guard noise, and a noisy guard gets removed.
_SYNTHETIC_ROOTS = ("src/", "tests/", "examples/")

# Extensions worth reading. Real data hides in text; a .png or a .parquet is
# not something this check can read usefully anyway.
_SCANNED_SUFFIXES = {
    ".md", ".txt", ".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml",
    ".toml", ".cfg", ".ini", ".py", ".sh", ".env", ".log", ".rst",
}

_SUPPRESSION = re.compile(r"leak-guard:\s*allow")

# Shapes that mean real people. Suppressed under _SYNTHETIC_ROOTS.
_PERSONAL_PATTERNS = (
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("email address", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("phone number", re.compile(r"\b\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b")),
)

# Shapes that are never legitimate anywhere, including in the package. A
# credential is not synthetic test data, and a path rooted in someone's home
# directory names the machine's owner and cannot work on anyone else's.
_UNIVERSAL_PATTERNS = (
    # The value has to look like a key, not like prose or a placeholder: 20+
    # characters with at least one digit. Without that bound this fires on
    # `api_key: overrides ...` in a docstring, and on the deliberately fake
    # `CENSUS_API_KEY=dotenv-key` fixtures in tests/test_acs_pums.py. A real
    # Census key is 40 hex characters and still matches.
    (
        "credential",
        re.compile(
            r"(?i)\b[A-Z0-9_]*(?:API_KEY|APIKEY|TOKEN|SECRET|PASSWORD)\b"
            r"\s*[=:]\s*[\"']?(?=[A-Za-z0-9_\-]*\d)[A-Za-z0-9_\-]{20,}"
        ),
    ),
    # /home/runner is GitHub Actions' own working directory and names nobody.
    ("personal absolute path", re.compile(r"/(?:Users|home)/(?!runner\b)[A-Za-z0-9._-]+/")),
)


class NoGitCheckout(Exception):
    """Raised instead of guessing when the ignore rules cannot be consulted.

    Failing closed matters more here than anywhere else in this file. This
    guard's whole job is to be the thing that does not quietly wave a private
    file through, and `git check-ignore` exits 128 both when git is missing
    and when the directory is not a checkout. Treating that like "not
    ignored" would turn the guard off in exactly the conditions where nobody
    would notice.
    """


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )


def is_git_checkout() -> bool:
    try:
        return _git("rev-parse", "--git-dir").returncode == 0
    except FileNotFoundError:
        return False


def tracked_files() -> list[str]:
    return _git("ls-files").stdout.split()


def staged_files() -> list[str]:
    return _git("diff", "--cached", "--name-only", "--diff-filter=ACMR").stdout.split()


def private_path_reason(path: str):
    """Why `path` must not be committed, or None if it may be.

    The categories are not a list invented here: they come from the repo's
    own .gitignore, asked through `git check-ignore --no-index`. `--no-index`
    is the load-bearing flag. Without it git reports an already-tracked file
    as not ignored, which is exactly the case that put CLAUDE.md and
    docs/ISSUES.md into public history. It also means adding a rule to
    .gitignore extends this guard without editing this file.
    """
    if _SECRET_FILENAME.search(path):
        return "a .env file. Secrets are read at runtime, never committed"
    try:
        code = _git("check-ignore", "--no-index", "-q", "--", path).returncode
    except FileNotFoundError as error:
        raise NoGitCheckout("git is not installed") from error
    if code == 0:
        return (
            ".gitignore marks it local-only (working notes, specs, handoffs "
            "or agent instructions), and this repo is public"
        )
    if code != 1:
        raise NoGitCheckout(f"`git check-ignore` failed on {path!r} (exit {code})")
    return None


def pii_findings(path: str, text: str) -> list:
    """(line number, what matched, the line) for each PII shape in `text`."""
    if pathlib.PurePosixPath(path).suffix not in _SCANNED_SUFFIXES:
        return []
    patterns = _UNIVERSAL_PATTERNS
    if not path.startswith(_SYNTHETIC_ROOTS):
        patterns = patterns + _PERSONAL_PATTERNS
    findings = []
    for number, line in enumerate(text.splitlines(), start=1):
        if _SUPPRESSION.search(line):
            continue
        for label, pattern in patterns:
            if pattern.search(line):
                findings.append((number, label, line.strip()))
    return findings


def _read(path: str) -> str:
    try:
        return (ROOT / path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def main(paths: list) -> int:
    try:
        private = [(path, private_path_reason(path)) for path in paths]
    except NoGitCheckout as error:
        print(f"leak-guard cannot read the ignore rules: {error}.")
        print("Refusing to pass rather than guess. Run this inside a git checkout.")
        return 1
    private = [(path, reason) for path, reason in private if reason]
    blocked = {path for path, _ in private}
    leaks = [
        (path, finding)
        for path in paths
        if path not in blocked
        for finding in pii_findings(path, _read(path))
    ]
    if not private and not leaks:
        return 0
    print("This repo is public. Refusing the following:\n")
    for path, reason in private:
        print(f"  {path}: {reason}.")
    if private:
        print(
            "\n  Unstage with `git restore --staged <path>`. If a file genuinely\n"
            "  belongs in the public package, add it to .gitignore's exception\n"
            "  list first, so the rule and the guard agree.\n"
        )
    for path, (number, label, line) in leaks:
        print(f"  {path}:{number}: looks like a {label}: {line}")
    if leaks:
        print(
            "\n  Remove it, or if it is a deliberate example, append a comment\n"
            "  containing `leak-guard: allow` and the reason to that line.\n"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or staged_files()))
