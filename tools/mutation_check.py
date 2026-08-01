"""Mutation harness: revert each logged fix, confirm the suite goes red.

A fix with no test that catches its absence is not locked down. This applies
each revert one at a time, runs the suite, restores the file, and reports
whether the suite noticed.
"""

import atexit
import os
import pathlib
import signal
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Guards against the 2026-07-31 false alarm: an external timeout killed this
# process mid-mutation and left a reverted snippet in the working tree,
# which then looked like a real regression. Each mutation's own try/finally
# already restores on a normal exception; this covers the process itself
# being torn down (SIGTERM, or the interpreter's own atexit) between the
# write and the finally block running.
_pending_restore: tuple[pathlib.Path, str] | None = None


def _restore_pending(*_args) -> None:
    if _pending_restore is not None:
        path, text = _pending_restore
        path.write_text(text)


atexit.register(_restore_pending)
signal.signal(signal.SIGTERM, lambda *a: (_restore_pending(), sys.exit(1)))

# (issue, file, original_snippet, reverted_snippet)
MUTATIONS = [
    (
        "I1 synthesizer table scoping",
        "src/synthweave/stages/synthesize.py",
        """        if self.tables is not None and table.name not in self.tables:
            yield from chunks
            return
""",
        "",
    ),
    (
        "I3 fit buffer truncation (chunk invariance)",
        "src/synthweave/stages/base.py",
        """    if len(buffered) > max_rows:
        buffered = buffered.iloc[:max_rows]
""",
        "",
    ),
    (
        "I4 first column conditions on predictors",
        "src/synthweave/stages/synthesize.py",
        """        for i, target in enumerate(self.columns):
            features = self.predictors + self.columns[:i]
            if not features:
                continue
""",
        """        for i, target in enumerate(self.columns):
            features = self.predictors + self.columns[:i]
            if i == 0:
                continue
""",
    ),
    (
        "I5 link before noise (identifier noise possible)",
        "src/synthweave/pipeline.py",
        "for stage in (self.synthesizer, self.linker, self.noiser):",
        "for stage in (self.synthesizer, self.noiser, self.linker):",
    ),
    (
        "I6 own() defensive copy",
        "src/synthweave/stages/base.py",
        """    if getattr(chunk, "_is_view", False) or getattr(chunk, "_is_copy", None) is not None:
        return chunk.copy()
    return chunk""",
        """    return chunk""",
    ),
    (
        "I9 synthesized column keeps its fitted dtype",
        "src/synthweave/stages/synthesize.py",
        """        dtype = self.dtypes.get(column)
        if dtype is None or dtype == object:
            return values
        return values.astype(dtype)""",
        """        return values""",
    ),
    (
        "I11 identifier tag vs column/carry collision",
        "src/synthweave/validation.py",
        """        if tag in table.columns:
            raise SchemaError(
                f"{where}: identifier {tag!r} has the same name as a table column, "
                f"and the identifier would overwrite it"
            )
""",
        "",
    ),
    (
        "I11 identifier tag vs carried attribute",
        "src/synthweave/validation.py",
        """        if tag in table.carry:
            raise SchemaError(
                f"{where}: identifier {tag!r} has the same name as the carried entity "
                f"attribute {tag!r}, and the identifier would overwrite it"
            )
""",
        "",
    ),
    (
        "I13 empty table still writes a file",
        "src/synthweave/io.py",
        """        elif not self._wrote_header:
            self.write_empty()""",
        """        elif self.format == "csv" and not self._wrote_header:
            self.path.write_text("")""",
    ),
    (
        "I12 writer reconciles a chunk against the file schema",
        "src/synthweave/io.py",
        """        elif batch.schema != self._writer.schema:
            batch = self._reconcile(batch, pa)""",
        "",
    ),
    (
        "I17 declared-numeric coercion names the column",
        "src/synthweave/stages/synthesize.py",
        """        converted = pd.to_numeric(series, errors="coerce")
        bad = series[converted.isna() & series.notna()]""",
        """        converted = pd.to_numeric(series, errors="coerce")
        bad = []""",
    ),
    (
        "I13 empty table keeps its declared columns",
        "src/synthweave/pipeline.py",
        "        return pd.DataFrame(columns=columns)",
        "        return pd.DataFrame()",
    ),
    (
        "I14 identifier width vs population",
        "src/synthweave/validation.py",
        """    if expected < TOLERABLE_COLLISIONS:
        return""",
        """    if True:
        return""",
    ),
    (
        "I14 digits past the hash width",
        "src/synthweave/schema.py",
        "        if self.digits > MAX_DIGITS:",
        "        if self.digits > 10**9:",
    ),
    (
        "I15 joint prior picks over positions",
        "src/synthweave/stages/synthesize.py",
        "            positions = np.arange(len(pairs), dtype=object)",
        "            positions = np.array(pairs, dtype=object)",
    ),
    (
        "I16 repeated period rejected",
        "src/synthweave/schema.py",
        """        if repeated:""",
        """        if False:""",
    ),
    (
        "I10/I12 rules declare their dtype",
        "src/synthweave/rules.py",
        """    dtype = declared_dtype(rule)
    return values if dtype is None else values.astype(dtype)""",
        """    return values""",
    ),
    (
        "I17 numeric decided by dtype, not by guessing",
        "src/synthweave/stages/synthesize.py",
        "        return column in self.numeric or pd.api.types.is_numeric_dtype(series)",
        """        return column in self.numeric or (
            pd.to_numeric(series, errors="coerce").notna().mean() > 0.9
        )""",
    ),
    (
        "I29 donor diagnostics silently stale on synthesizer reuse",
        "src/synthweave/stages/synthesize.py",
        "        self._fitted[table.name] = model",
        "        self._fitted.setdefault(table.name, model)",
    ),
    (
        "I30 donor diagnostics readable mid-stream, before every chunk applied",
        "src/synthweave/stages/synthesize.py",
        """            complete = self._complete.get(table, False)
            return {table: dict(model.empty_donor_counts)} if model and complete else {}""",
        """            return {table: dict(model.empty_donor_counts)} if model else {}""",
    ),
    (
        "I28 SSN area 666 remapped onto 667 instead of skipped",
        "src/synthweave/connectors/faker_names.py",
        """        area = _hash.integers(keys, seed, f"{salt}\\x00area", 1, 899)
        area = np.where(area >= 666, area + 1, area)""",
        """        area = _hash.integers(keys, seed, f"{salt}\\x00area", 1, 900)
        area = np.where(area == 666, 667, area)""",
    ),
    (
        "I31 structure name needing config raises a bare TypeError",
        "src/synthweave/stages/synthesize.py",
        "        return _resolve_structure_name(structure)",
        '        return resolve("structure", structure)',
    ),
    (
        "#10 per-row rate function is actually applied",
        "src/synthweave/stages/noise.py",
        """                    if callable(rate):
                        rate = _row_rates(rate, chunk, f"{table.name}.{column}.{op.name}")""",
        "",
    ),
    (
        "#17 header-only ACS response rejected",
        "src/synthweave/connectors/acs_pums.py",
        "    if not payload or len(payload) <= 1:",
        "    if not payload or len(payload) < 1:",
    ),
    (
        "#19 GeoNames TSV parsed without CSV quote handling",
        "src/synthweave/connectors/geonames.py",
        '    return list(csv.reader(io.StringIO(raw), delimiter="\\t", quoting=csv.QUOTE_NONE))',
        '    return list(csv.reader(io.StringIO(raw), delimiter="\\t"))',
    ),
    (
        "#18 missing birth year rejected before the range guard",
        "src/synthweave/connectors/ssa_names.py",
        "        missing = pd.isna(years)\n        if missing.any():",
        "        missing = pd.isna(years)\n        if False:",
    ),
    (
        "#16 SSA cache filename distinguishes the source",
        "src/synthweave/connectors/ssa_names.py",
        "    cache_path = None if cache_dir is None else Path(cache_dir) / _cache_filename(source)",
        '    cache_path = None if cache_dir is None else Path(cache_dir) / "ssa_names.csv"',
    ),
    (
        "#11 carry=* resolves per schema, not once per table",
        "src/synthweave/schema.py",
        """        self.tables = tuple(
            replace(table, carry=tuple(self.entity(table.entity).attributes.keys()))
            if table.carry == "*"
            else table
            for table in self.tables
        )""",
        """        for table in self.tables:
            if table.carry == "*":
                table.carry = tuple(self.entity(table.entity).attributes.keys())""",
    ),
    (
        "#10 per-row rate range check",
        "src/synthweave/stages/noise.py",
        "    if not np.all((rates >= 0.0) & (rates <= 1.0)):",
        "    if False:",
    ),
]


def run_suite() -> tuple[bool, str, str]:
    # Inherit the real environment and override only PYTHONPATH. Replacing it
    # outright used to work on a dev machine and fail on a CI runner, where
    # the interpreter lives outside a hardcoded PATH and the suite needs
    # variables (HOME among them) that a stripped env does not carry.
    env = {**os.environ, "PYTHONPATH": "src"}
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--no-header", "-x", "-W", "error::UserWarning"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return False, "suite timed out after 300s", ""
    output = proc.stdout + proc.stderr
    tail = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    return proc.returncode == 0, (tail[-1] if tail else "no output"), output


print("baseline: ", end="", flush=True)
ok, msg, output = run_suite()
print(f"{'PASS' if ok else 'FAIL'}  {msg}")
if not ok:
    # The whole run stops here, so print what actually failed. Reporting only
    # "baseline must pass" leaves no way to diagnose it from a CI log, where
    # nobody can re-run the suite by hand.
    print("\n--- baseline failure output ---")
    print(output)
    sys.exit("baseline must pass before mutating")

gaps = []
stale = []
for name, relpath, original, reverted in MUTATIONS:
    path = ROOT / relpath
    if not path.exists():
        # A missing file verifies exactly as much as a missing snippet does:
        # nothing. It used to crash the whole run instead, which meant one
        # entry naming a file absent on the current branch took down every
        # later entry's check too.
        print(f"  STALE  {name}: {relpath} does not exist, so nothing was verified")
        stale.append(name)
        continue
    text = path.read_text()
    if original not in text:
        print(f"  STALE  {name}: snippet not found, so nothing was verified")
        stale.append(name)
        continue
    path.write_text(text.replace(original, reverted, 1))
    _pending_restore = (path, text)
    try:
        ok, msg, _ = run_suite()
    finally:
        path.write_text(text)
        _pending_restore = None
    caught = not ok
    print(f"  {'CAUGHT ' if caught else 'MISSED '} {name}\n           -> {msg}")
    if not caught:
        gaps.append(name)

print()
if stale:
    print(f"{len(stale)} mutation(s) no longer match the code and verified nothing:")
    for s in stale:
        print(f"  - {s}")
    print("  fix the snippet; a stale mutation reads as a pass and is not one")
if gaps:
    print(f"{len(gaps)} fix(es) with NO regression coverage:")
    for g in gaps:
        print(f"  - {g}")
if not gaps and not stale:
    print("every logged fix is covered by a failing test")
sys.exit(1 if (gaps or stale) else 0)
