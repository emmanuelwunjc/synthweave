"""Mutation harness: revert each logged fix, confirm the suite goes red.

A fix with no test that catches its absence is not locked down. Each revert is
applied to a throwaway copy of the checkout, the suite runs against that copy,
and the harness reports whether the suite noticed.

Three properties are load-bearing and none is traded for speed:

- MISSED: a reverted fix that no test catches fails the run.
- STALE: an entry whose file or snippet no longer matches verified nothing, so
  it fails the run too.
- INCONCLUSIVE: a suite run that never finished verified nothing either, so it
  is neither a pass nor a catch, and it fails the run as well.

The last one is the direction this harness must never fail in. "I could not
tell" resolving to "covered" is worse than a crash, because a crash gets
noticed.

Every mutation is independent, so they run concurrently, one sandbox per
worker. Sandboxing is what makes that safe: the real working tree is never
written to, which also retires the 2026-07-31 false alarm (a killed process
leaving a reverted snippet behind) and the 2026-08-02 hazard where a
concurrent `git checkout` tripped over a mid-run mutation.

    python3 tools/mutation_check.py                # every entry
    python3 tools/mutation_check.py --shard 2/6    # one CI runner's share

CI fans the shards out across runners and rolls them up into the
`mutation-check` gate. `tests/test_mutation_shards.py` asserts the shards
partition the list and that the workflow's matrix matches the `N` it passes;
a shard that quietly skips entries would report a pass it never earned.
"""

import concurrent.futures
import os
import pathlib
import queue
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Copied per worker: everything except the version-control metadata and caches.
#
# Skipping `.git` is a deliberate narrowing, not an oversight, and it is not
# free: the suite the sandbox runs is smaller than the one CI's `test` job
# gates on. The tests that go missing are named in SANDBOX_BLIND_SPOT below,
# and `report_blind_spot()` measures the gap on every run so it cannot widen
# in silence.
#
# Why not carry `.git`. In a linked worktree `.git` is a *file* pointing at
# another directory, so a plain copy either fails or produces a repo that
# resolves back into the real checkout: local and CI behaviour would then
# differ, which is the opposite of what this harness is for. Resolving
# `--git-common-dir` and copying it is possible but pays ~12 MB per worker on
# a 41 MB tree, once per `os.cpu_count()`. And it would not buy honesty: the
# `@needs_git` tests ask `git check-ignore` about the *tracked* tree, so a
# copied index describing the original checkout would answer about files that
# are not the ones under test. `git init` in the sandbox is worse still (see
# issue #131): a repo with no commits makes `tracked_files()` empty, and
# `test_the_committed_tree_passes_its_own_guard` would pass while checking
# nothing, the exact "test that cannot fail" shape CLAUDE.md warns about.
SANDBOX_SKIP = (".git", "__pycache__", ".pytest_cache")

# Test modules that cannot run inside a sandbox, and why. Declared here rather
# than described in prose so `report_blind_spot()` can check the claim against
# what the sandbox actually skips: a module that goes blind without being
# listed here fails the run.
SANDBOX_BLIND_SPOT = {
    "tests/test_leak_guard.py": (
        "the public-repo leak guard asks `git check-ignore` about the tracked "
        "tree; every @needs_git test self-skips without a checkout"
    ),
    "tests/test_docs_map_sync.py": (
        "compares docs/MAP.md against git's tracked-file list, which a "
        "checkout-less tree cannot produce"
    ),
    "tests/test_mutation_shards.py": (
        "one test measures this very blind spot by diffing a sandbox against "
        "a real checkout, and a sandbox has no checkout to diff against"
    ),
}

# Wall-clock bound on one suite run. A run that hits it is inconclusive, never
# a pass and never a catch.
SUITE_TIMEOUT = 300

# How a suite run ended. Three outcomes, not two: "ran and went red" and "never
# finished" used to share one `False`, which made a timed-out run indexed as
# proof that a revert was caught.
PASSED, FAILED, DID_NOT_FINISH = "passed", "failed", "did not finish"

# Verdicts. CAUGHT is the only one that counts as coverage; the rest fail the
# run, each for its own reason.
CAUGHT, MISSED, STALE, INCONCLUSIVE = "CAUGHT", "MISSED", "STALE", "INCONCLUSIVE"

# Printed above the names when a verdict fails the run.
FAILURE_HEADLINE = {
    STALE: "mutation(s) no longer match the code and verified nothing:",
    MISSED: "fix(es) with NO regression coverage:",
    INCONCLUSIVE: "mutation(s) whose suite run never finished, so nothing was verified:",
}

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
        if isinstance(dtype, pd.api.extensions.ExtensionDtype):
            return pd.array(values, dtype=dtype)
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
        # `_concat`'s stand-in frame moved out to `Pipeline._empty_frame` with
        # the #82 dtype fix. Same fix, same revert: hand back a frame with no
        # columns and see whether the suite notices.
        "        return empty\n",
        "        return pd.DataFrame()\n",
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
        "        _cache_filename(source),",
        '        "ssa_names.csv",',
    ),
    (
        "#13 namedtuple refused rather than flattened into a Choice",
        "src/synthweave/rules.py",
        '    if isinstance(value, tuple) and hasattr(value, "_fields"):',
        "    if False:",
    ),
    (
        "offline: surname weight is count * pct/100",
        "src/synthweave/connectors/census_surnames.py",
        """                self._data["count"].to_numpy(dtype=np.float64)
                * self._data[column].to_numpy(dtype=np.float64)
                / 100.0""",
        """                self._data["count"].to_numpy(dtype=np.float64)""",
    ),
    (
        "offline: address fields in a group share one row",
        "src/synthweave/connectors/geonames.py",
        'idx = _hash.integers(keys, seed, f"usaddress\\x00{self.group}", 0, len(self._data))',
        'idx = _hash.integers(keys, seed, f"usaddress\\x00{salt}", 0, len(self._data))',
    ),
    (
        "offline: SSA pool narrows to the requested sex",
        "src/synthweave/connectors/ssa_names.py",
        """        if sex is not None:
            subset = subset[subset["sex"] == sex]""",
        "",
    ),
    (
        "#37 joint disagreeing with a declared marginal is rejected",
        "src/synthweave/stages/synthesize.py",
        "        _check_joints_agree_with_marginals(self.marginals, self.joints)",
        "        pass",
    ),
    (
        "empty-donor invariant: donor map covers every leaf",
        "src/synthweave/stages/synthesize.py",
        """            self.donors[target] = {
                leaf: raw[leaves == leaf] for leaf in np.unique(leaves)
            }""",
        """            self.donors[target] = {
                leaf: raw[leaves == leaf] for leaf in np.unique(leaves)[1:]
            }""",
    ),
    (
        "#12 carry=* unknown entity raises SchemaError with table context",
        "src/synthweave/schema.py",
        '            raise SchemaError(f"table {table.name!r}: {e.args[0]}") from e',
        "            raise",
    ),
    (
        "#14 numpy scalars coerce symmetrically",
        "src/synthweave/rules.py",
        "    if isinstance(value, np.generic):\n        return Constant(value.item())",
        "    if False:\n        return Constant(value.item())",
    ),
    (
        "#20 unknown numeric state FIPS code rejected",
        "src/synthweave/connectors/acs_pums.py",
        "        if code not in set(_STATE_FIPS.values()):",
        "        if False:",
    ),
    (
        "#15 malformed structure dict rejected at the coercion",
        "src/synthweave/stages/synthesize.py",
        "        if bad:",
        "        if False:",
    ),
    (
        "#11 carry=* resolves per schema, not once per table",
        "src/synthweave/schema.py",
        """        self.tables = tuple(
            replace(table, carry=self._every_attribute_of(table)) if table.carry == "*" else table
            for table in self.tables
        )""",
        """        for table in self.tables:
            if table.carry == "*":
                table.carry = self._every_attribute_of(table)""",
    ),
    (
        "#10 per-row rate range check",
        "src/synthweave/stages/noise.py",
        "    if not np.all((rates >= 0.0) & (rates <= 1.0)):",
        "    if False:",
    ),
    (
        "#41 NaN/inf Choice weight rejected instead of collapsing the column",
        "src/synthweave/_hash.py",
        "        if not np.all(np.isfinite(w)):",
        "        if False:",
    ),
    (
        "#43 CSV chunk writer guards column order/shape between chunks",
        "src/synthweave/io.py",
        "            elif list(chunk.columns) != self._csv_columns:",
        "            elif False:",
    ),
    (
        "#44 a carried attribute's transitive dependency chain is drawn in full",
        "src/synthweave/stages/generate.py",
        "            if name in needed:\n                needed.update(entity.attributes[name].depends_on())",
        "            pass",
    ),
    (
        "#40 CART trees seeded so tied splits break the same way every fit",
        "src/synthweave/stages/synthesize.py",
        "            random_state = int(_hash.hash_key(self.seed, f\"cart\\x00{target}\"), 16) % (2**32)",
        "            random_state = None",
    ),
    (
        "#42 a noised column keeps its dtype where the values still allow it",
        "src/synthweave/stages/noise.py",
        "                chunk[column] = _restore_dtype(values, original_dtype)",
        "                chunk[column] = values",
    ),
    (
        "#45.1a duplicate table carry names are rejected",
        "src/synthweave/validation.py",
        '    _check_unique(list(table.carry), f"{where} carried attribute")',
        "    pass",
    ),
    (
        "#45.1b duplicate table identifier names are rejected",
        "src/synthweave/validation.py",
        '    _check_unique(list(table.identifiers), f"{where} identifier")',
        "    pass",
    ),
    (
        "#45.2a identifier-width fix drops the erroneous extra digit",
        "src/synthweave/validation.py",
        "    needed = len(str(population * population // 2))",
        "    needed = len(str(population * population // 2)) + 1",
    ),
    (
        "#45.2b a population past the digit limit says so instead of recommending it",
        "src/synthweave/validation.py",
        "    if needed > MAX_DIGITS:",
        "    if False:",
    ),
    (
        "#45.3 Uniform rejects high <= low instead of silently descending",
        "src/synthweave/rules.py",
        "        if self.high <= self.low:",
        "        if False:",
    ),
    (
        "#46.1 two joints sharing a column are rejected",
        "src/synthweave/stages/synthesize.py",
        "        _check_joints_do_not_share_a_column(self.joints)\n        _check_joints_agree_with_marginals(self.marginals, self.joints)",
        "        _check_joints_agree_with_marginals(self.marginals, self.joints)",
    ),
    (
        "#46.2 a numeric Prior marginal carries its natural dtype",
        "src/synthweave/stages/synthesize.py",
        "            frame[column] = picked.astype(natural) if natural.kind in \"iuf\" else picked",
        "            frame[column] = picked",
    ),
    (
        "#46.3 fit_cap holds for a supplied structure source across seeds",
        "src/synthweave/stages/synthesize.py",
        "                    train = train.iloc[:cap]",
        "                    keys = np.asarray(train.index, dtype=str).astype(object)\n                    pick = _hash.unit(keys, ctx.seed, f\"fitsample\\x00{table.name}\") < (cap / len(train))\n                    train = train.loc[pick]",
    ),
    (
        "#64 check_rule catches a non-deterministic rule",
        "src/synthweave/rules.py",
        '    repeat = rule.draw(keys, seed=seed, salt=salt, frame=frame)\n    _assert_same_values(\n        keys, baseline, repeat, "calling draw() twice with identical input gave different "\n        "values back (the rule is not deterministic, e.g. it reaches for random state)"\n    )',
        "    pass",
    ),
    (
        "#64 check_rule catches a position-keyed rule via a shuffled key array",
        "src/synthweave/rules.py",
        "    order = np.arange(n)[::-1]",
        "    order = np.arange(n)",
    ),
    (
        "#64 check_rule catches a chunk-size-dependent rule via a split call",
        "src/synthweave/rules.py",
        '    combined = np.concatenate([first, second])\n    _assert_same_values(\n        keys, baseline, combined, "splitting the keys across two calls changed a value "\n        "(the rule is not chunk invariant, e.g. it reads chunk-level state)"\n    )',
        "    pass",
    ),
    (
        "#89.1 two attributes sharing one ACS variable both keep a column",
        "src/synthweave/mode.py",
        "            self._fetched = pd.DataFrame(\n                {name: fetched[variable] for name, variable in self._variables.items()}\n            )",
        "            self._fetched = fetched.rename(\n                columns={variable: name for name, variable in self._variables.items()}\n            )",
    ),
    (
        "#89.2 scope mode generalizes by epsilon instead of CART's defaults",
        "src/synthweave/mode.py",
        '            "synthesizer": _epsilon_chain(self._scope_epsilon, self._fetched, placement)',
        '            "synthesizer": _empirical_cart(list(self._variables), self._fetched)',
    ),
    (
        "#89.3 scope epsilon is validated, not clamped (mode level)",
        "src/synthweave/mode.py",
        '        _check_epsilon(epsilon, "scope")\n',
        "",
    ),
    (
        "#89.4 scope epsilon is validated, not clamped (per attribute)",
        "src/synthweave/mode.py",
        '        if epsilon is not None:\n            _check_epsilon(epsilon, f"attribute {name!r}")\n        self._variables[name] = variable',
        "        self._variables[name] = variable",
    ),
    (
        "#88 non-positive real_data epsilon rejected instead of clamped to 0.01",
        "src/synthweave/mode.py",
        '    if epsilon <= 0:\n        raise ValueError(f"{where}: epsilon must be positive, got {epsilon!r}")',
        "    pass",
    ),
    (
        "#88 a stray attribute kwarg in real_data mode is named, not a bare TypeError",
        "src/synthweave/mode.py",
        """        if kwargs:
            raise ValueError(
                f"attribute {name!r}: real_data mode takes only epsilon, got "
                f"{sorted(kwargs)}; the column's distribution comes from the "
                "donor frame, not from a declared rule"
            )
""",
        "",
    ),
    (
        "N2 a stray attribute kwarg in scope mode is named, not a bare TypeError",
        "src/synthweave/mode.py",
        """        if kwargs:
            raise ValueError(
                f"attribute {name!r}: scope mode takes only variable and "
                f"epsilon, got {sorted(kwargs)}; the column's distribution "
                "comes from the fetched ACS rows, not from a declared rule"
            )
""",
        "",
    ),
    (
        "#87 mode noise resolves against the schema's expanded carry (carry=*)",
        "src/synthweave/mode.py",
        "            for column_name in list(table.carry) + list(table.columns):",
        """            for column_name in list(table.columns) + (
                list(table.carry) if isinstance(table.carry, list) else []
            ):""",
    ),
    (
        "#87 a mode noise rate declared after table() still reaches the table",
        "src/synthweave/mode.py",
        """        return Table(
            name,
            grain=grain,
            columns=columns or {},
            carry=carry,
            identifiers=identifiers or (),
            coverage=coverage,
        )""",
        """        # Reverted: freeze the rates known now, which is what resolving
        # the noise map inside table() amounted to.
        known = set(self._noise_kwargs)

        class _AsOfNow(dict):
            def get(self, key, default=None):
                return dict.get(self, key, default) if key in known else default

        self._noise_kwargs = _AsOfNow(self._noise_kwargs)
        return Table(
            name,
            grain=grain,
            columns=columns or {},
            carry=carry,
            identifiers=identifiers or (),
            coverage=coverage,
        )""",
    ),
    (
        "#48 unregister actually removes the entry",
        "src/synthweave/registry.py",
        "    del table[name]",
        "    pass",
    ),
    (
        "#48 the autouse registry-reset fixture restores the pre-test snapshot",
        "tests/conftest.py",
        "    registry._restore(snapshot)",
        "    pass",
    ),
    (
        "#60 PackageNotFoundError stays out of the public namespace",
        "src/synthweave/__init__.py",
        # Not a literal revert, deliberately. Before #60 the module both
        # imported `PackageNotFoundError` un-aliased *and* caught it un-aliased
        # forty lines later, so undoing the fix for real spans two hunks and an
        # entry here carries exactly one contiguous snippet. The alias
        # assignment keeps the `except _PackageNotFoundError` clause bound
        # while restoring the public leak, which is the property under test.
        # An earlier form appended a duplicate un-aliased import instead; it
        # was CAUGHT too, but left the sandbox tree failing pyflakes with
        # "imported but unused", i.e. a mutation no reviewer would accept as a
        # plausible edit.
        "from importlib.metadata import PackageNotFoundError as _PackageNotFoundError\n"
        "from importlib.metadata import version as _installed_version",
        "from importlib.metadata import PackageNotFoundError\n"
        "from importlib.metadata import version as _installed_version\n"
        "\n"
        "_PackageNotFoundError = PackageNotFoundError",
    ),
    (
        "#63 faker_names validates Faker's private provider shape",
        "src/synthweave/connectors/faker_names.py",
        "    pool: Any = _checked_provider_pool(Provider, attr)",
        "    pool: Any = getattr(Provider, attr)",
    ),
    (
        "#63 faker_names weight check separates non-numeric, non-finite and non-positive",
        "src/synthweave/connectors/faker_names.py",
        '        if not isinstance(weight, numbers.Real) or isinstance(weight, bool):\n'
        '            raise bad(f"has a non-numeric weight {weight!r} for {name!r}")\n'
        "        if not math.isfinite(weight):\n"
        '            raise bad(f"has a non-finite weight {weight!r} for {name!r}")\n'
        "        if not weight > 0:",
        "        if not isinstance(weight, (int, float)) or isinstance(weight, bool)"
        " or not weight > 0:",
    ),
    (
        "#62 I32 the ACS response is parsed before it is cached",
        "src/synthweave/connectors/acs_pums.py",
        """    frame = _to_frame(payload, variables, url)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload))
    return frame""",
        """    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload))
    return _to_frame(payload, variables, url)""",
    ),
    (
        "#62 I33 the .env walk stops at the project root",
        "src/synthweave/connectors/acs_pums.py",
        """        if any((directory / marker).exists() for marker in _PROJECT_ROOT_MARKERS):
            break
""",
        "",
    ),
    (
        "#62 I33 the project root's own .env is read before the walk stops",
        "src/synthweave/connectors/acs_pums.py",
        """    for directory in (here, *here.parents):
        candidate = directory / ".env\"""",
        """    for directory in (here, *here.parents):
        if any((directory / marker).exists() for marker in _PROJECT_ROOT_MARKERS):
            break
        candidate = directory / ".env\"""",
    ),
    (
        "#47 a noise rate function must return one rate per row",
        "src/synthweave/stages/noise.py",
        """    if rates.shape != (len(chunk),):
""",
        """    if False:
""",
    ),
    (
        "#96.2 an ExtensionDtype is restored through pandas, not ndarray.astype",
        "src/synthweave/stages/synthesize.py",
        "        if isinstance(dtype, pd.api.extensions.ExtensionDtype):\n            return pd.array(values, dtype=dtype)\n",
        "",
    ),
    # --- #65 the derivation layer ------------------------------------------
    # Every value in the library routes through `_hash`, so a corruption here
    # is silently wrong everywhere at once. These deliberately are not
    # "delete the guard" mutations: each one is the shape a real arithmetic
    # slip would take, which is what the existing entries did not cover.
    (
        "#65 hash_key keeps seed and salt separated",
        "src/synthweave/_hash.py",
        'return hashlib.sha256(f"{seed}\\x00{salt}".encode()).hexdigest()[:16]',
        'return hashlib.sha256(f"{seed}{salt}".encode()).hexdigest()[:16]',
    ),
    (
        "#65 the salt reaches the hash key (independent draws per salt)",
        "src/synthweave/_hash.py",
        'hash_key=hash_key(seed, salt)',
        'hash_key=hash_key(seed, "")',
    ),
    (
        "#65 unit() divides by the full uint64 width",
        "src/synthweave/_hash.py",
        "_SCALE = np.float64(2.0**64)",
        "_SCALE = np.float64(2.0**63)",
    ),
    (
        "#65 integers() span excludes `high`",
        "src/synthweave/_hash.py",
        "    span = np.uint64(high - low)",
        "    span = np.uint64(high - low + 1)",
    ),
    (
        "#65 normal() Box-Muller radius keeps the -2 factor",
        "src/synthweave/_hash.py",
        "    z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)",
        "    z = np.sqrt(-1.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)",
    ),
    (
        "#65 unweighted pick() spreads over every value",
        "src/synthweave/_hash.py",
        "        idx = np.minimum((u * len(values)).astype(np.int64), len(values) - 1)",
        "        idx = np.minimum((u * (len(values) - 1)).astype(np.int64), len(values) - 1)",
    ),
    (
        "#65 weighted pick() normalizes the weights before the cumsum",
        "src/synthweave/_hash.py",
        "        idx = np.searchsorted(np.cumsum(w / total), u, side=\"right\")",
        "        idx = np.searchsorted(np.cumsum(w), u, side=\"right\")",
    ),
    (
        "#65 derive_id() zero-pads to the declared width",
        "src/synthweave/_hash.py",
        '    text = np.char.zfill(n.astype("U"), digits)',
        '    text = n.astype("U")',
    ),
    (
        "#65 derive_id() uses the full declared keyspace",
        "src/synthweave/_hash.py",
        "    modulus = 10**digits",
        "    modulus = 10 ** (digits - 1)",
    ),
    (
        "#65 derive_id() namespaces by tag, so two tags are unrelated",
        "src/synthweave/_hash.py",
        '    n = (hash_u64(keys, seed, f"id\\x00{tag}") % np.uint64(modulus)).astype(np.int64)',
        '    n = (hash_u64(keys, seed, "id") % np.uint64(modulus)).astype(np.int64)',
    ),
    # --- #65 subtler corruptions of already-mutated code --------------------
    # The existing entries for these lines only delete the guard. A wrong
    # constant or a dropped term leaves the guard in place and still produces
    # wrong output, which is what a real bug looks like.
    (
        "#65 TOLERABLE_COLLISIONS stays at one expected collision",
        "src/synthweave/validation.py",
        "TOLERABLE_COLLISIONS = 1",
        "TOLERABLE_COLLISIONS = 1000",
    ),
    (
        "#65 the collision bound is quadratic in population (birthday bound)",
        "src/synthweave/validation.py",
        "    return population * population / (2 * 10**digits)",
        "    return population / (2 * 10**digits)",
    ),
    (
        "#65 own() copies a frame pandas flagged as a copy, not only a view",
        "src/synthweave/stages/base.py",
        '    if getattr(chunk, "_is_view", False) or getattr(chunk, "_is_copy", None) is not None:',
        '    if getattr(chunk, "_is_view", False):',
    ),
    (
        # Retargeted 2026-08-04: #82 moved the empty-table build into
        # `_empty_frame`, so the old snippet stopped matching and the entry went
        # STALE. Same guarantee, same test, new line.
        "#65 an empty table keeps its declared column order, not a sorted one",
        "src/synthweave/pipeline.py",
        "            columns=columns,\n        )",
        "            columns=sorted(columns),\n        )",
    ),
    (
        "#142 check_synthesizer names the dropped row key instead of raising KeyError",
        "src/synthweave/conformance.py",
        "    if ROW_KEY not in got.columns:",
        "    if False:",
    ),
    # Bumps `pyproject.toml` and nothing else -- the exact drift this guard
    # exists for. Like every entry here the snippet is a literal snapshot, so a
    # real bound change makes this STALE, which fails the run and points at the
    # line to update. That is the intended way to find out, not a defect.
    (
        "N1 the Faker bound is single-sourced to pyproject.toml",
        "pyproject.toml",
        'pii = ["Faker>=20,<41"]',
        'pii = ["Faker>=20,<42"]',
    ),
    (
        # Not a fix, a guarantee. `_entities_per_chunk` chunks over entities so
        # that an entity's rows never straddle a boundary, and until #53 nothing
        # asserted it. This entry breaks the guarantee in the shipped generator
        # (one row, then the rest, so a chunk boundary lands inside the first
        # entity) while leaving determinism, chunk invariance, the emitted
        # columns and the row count untouched. Only the non-straddling check can
        # notice, which is the point of logging it here.
        #
        # The `len(chunk) > 1` guard is what keeps that true. Splitting
        # unconditionally makes `chunk.iloc[1:]` empty for a one-row chunk,
        # which is a *second* broken invariant (line 63 above refuses to emit an
        # empty chunk) and one that crashes three tests in `test_pipeline.py`
        # and `test_synthesis_and_plugins.py` on `ValueError: zero-size array to
        # reduction operation maximum`. Those reds say nothing about
        # straddling, and they are enough on their own to report this entry
        # CAUGHT with the whole non-straddling clause deleted, which would make
        # the entry worthless as evidence for the thing it names. That is the
        # #19 trap: plausible, green, proving nothing.
        "#53 generator chunks whole entities (non-straddling)",
        "src/synthweave/stages/generate.py",
        """            emitted += len(chunk)
            yield chunk
""",
        """            emitted += len(chunk)
            if len(chunk) > 1:
                yield chunk.iloc[:1]
                yield chunk.iloc[1:]
            else:
                yield chunk
""",
    ),
    (
        "#61 find_stack_level walks out of the package instead of guessing",
        "src/synthweave/_deprecation.py",
        """    frame = sys._getframe(1)
    level = 1
    while frame is not None and _is_ours(frame.f_code.co_filename):
        frame = frame.f_back
        level += 1
    return level""",
        """    return 2""",
    ),
    (
        "#81 Typo corrupts a value whose script has no keyboard map",
        "src/synthweave/stages/noise.py",
        """            if options:
                repl = options[j % len(options)]
                out[i] = text[:j] + (repl.upper() if ch.isupper() else repl) + text[j + 1 :]
            else:
                out[i] = _slip(text, j)
""",
        """            if not options:
                out[i] = text
                continue
            repl = options[j % len(options)]
            out[i] = text[:j] + (repl.upper() if ch.isupper() else repl) + text[j + 1 :]
""",
    ),
    (
        "I41 a noised ExtensionDtype column keeps its dtype",
        "src/synthweave/stages/noise.py",
        """        if isinstance(dtype, pd.api.extensions.ExtensionDtype):
            restored = pd.array(values, dtype=dtype)
            if (pd.isna(restored) & ~pd.isna(values)).any():
                return values
            return restored
""",
        "",
    ),
    (
        "#82 an empty table keeps the dtypes its rules declare",
        "src/synthweave/pipeline.py",
        """        return pd.DataFrame(
            {
                name: as_declared(rules[name], empty) if name in rules else empty
                for name in columns
            },
            columns=columns,
        )""",
        "        return pd.DataFrame(columns=columns)",
    ),
    # Reverting either of these puts the leak guard back exactly as PR #106
    # merged it, which is the state where `aws_secret_access_key=...` and a
    # real person in a tests/ fixture both pass clean. The failure they cause
    # is invisible and permanent -- a credential in public git history, with
    # pre-commit printing `Passed` -- so the tests that catch them are worth
    # pinning here.
    (
        "#153 a bare KEY is a credential stem, so `census_key=` is a credential",
        "tools/check_no_private_leak.py",
        r'    r"(?:KEY|APIKEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?)"',
        r'    r"(?:API_KEY|APIKEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?)"',
    ),
    (
        "#153 name parts after the stem, so `secret_key_base=` is a credential",
        "tools/check_no_private_leak.py",
        r'    r"(?:_[A-Z0-9]+)*\b"',
        r'    r"\b"',
    ),
    (
        "#153 a 12-character value with a digit is credential-shaped",
        "tools/check_no_private_leak.py",
        r"[A-Za-z0-9_\-]{12,}|",
        r"[A-Za-z0-9_\-]{20,}|",
    ),
    (
        "#154 src/, tests/ and examples/ are not exempt from the personal shapes",
        "tools/check_no_private_leak.py",
        "    patterns = _UNIVERSAL_PATTERNS + _PERSONAL_PATTERNS",
        "    patterns = _UNIVERSAL_PATTERNS",
    ),
    (
        "#139 later epsilon groups condition on the columns earlier ones produced",
        "src/synthweave/mode.py",
        "                    predictors=list(conditioned),\n",
        "",
    ),
    (
        "#144 a real column is scoped to the tables that declared it",
        "src/synthweave/mode.py",
        "                    tables=[table_name],\n",
        "",
    ),
    (
        "#144 a real attribute no table carries raises instead of reaching all of them",
        "src/synthweave/mode.py",
        """    unmatched = sorted(set(names) - matched)
    if unmatched:
        raise ValueError(""",
        """    unmatched = []
    if unmatched:
        raise ValueError(""",
    ),
    (
        "#144 a real attribute bound under another name raises, not an all-null column",
        "src/synthweave/mode.py",
        "        if isinstance(rule, _RealDataColumn) and rule.name != bound:",
        "        if False:",
    ),
    (
        "#145 each epsilon group records under its own provenance/report key",
        "src/synthweave/stages/synthesize.py",
        '        prefix = f"{table.name}.synth" + (f".{self.label}" if self.label else "")\n'
        '        stage = "synthesize" + (f".{self.label}" if self.label else "")',
        '        prefix = f"{table.name}.synth"\n'
        '        stage = "synthesize"',
    ),
    (
        "#145 an epsilon-derived leaf size is user-provided, not a library default",
        "src/synthweave/mode.py",
        """        "min_samples_leaf": user(
            max(5, round(100 / capped)), f"derived from epsilon={epsilon!r}"
        ),""",
        '        "min_samples_leaf": max(5, round(100 / capped)),',
    ),
    (
        # The exact mutation #143 reported as invisible: epsilon becomes a total
        # no-op. It used to fail four tests, every one of them asserting a knob
        # value on the synthesizer object rather than anything in the output.
        "#143 epsilon reaches the synthesized data, not only the knob values",
        "src/synthweave/mode.py",
        "    capped = min(max(epsilon, 0.01), 5.0)",
        "    capped = 5.0",
    ),
    # A non-release tag sharing the release commit (this repo had
    # `archive/bug-hunt`) makes `git describe --exact-match` return it, which
    # `previous_tag` rejects. That step runs before the PyPI publish, so the
    # whole release stops for a tag that has nothing to do with releasing.
    (
        "R1 release notes derive the current tag from v* tags only",
        "tools/release_notes.py",
        '    current = _git("describe", "--tags", "--exact-match", "--match", "v[0-9]*").strip()',
        '    current = _git("describe", "--tags", "--exact-match").strip()',
    ),
    # The exact line #125 reported: `write_empty` rebuilding its own stand-in
    # frame instead of writing the typed one the pipeline handed it. Invisible
    # on CSV, baked into the file on Parquet, where every column comes back as
    # the `null` type on a run that happened to cover no entity.
    (
        "#125 an empty table's Parquet file keeps the schema its rules declare",
        "src/synthweave/io.py",
        "        empty = self.empty\n",
        "        empty = pd.DataFrame({name: pd.Series(dtype=object) for name in self.columns})\n",
    ),
    # #137's occurrence half: the grain column loses its stand-in rule and goes
    # back to `object` on a zero-row run against `int64` populated.
    (
        "#137 an empty event table keeps its grain column's declared dtype",
        "src/synthweave/pipeline.py",
        "    return _GrainColumn(np.arange(0).dtype) if isinstance(grain, PerEvent) else None",
        "    return None",
    ),
]


def run_suite(
    cwd: pathlib.Path,
    targets: tuple[str, ...] = ("tests/",),
    extra: tuple[str, ...] = (),
) -> tuple[str, str, str]:
    """Run the suite in `cwd`. Returns (outcome, summary line, full output)."""
    # Inherit the real environment and override only PYTHONPATH. Replacing it
    # outright used to work on a dev machine and fail on a CI runner, where
    # the interpreter lives outside a hardcoded PATH and the suite needs
    # variables (HOME among them) that a stripped env does not carry.
    #
    # PYTHONPATH is relative on purpose: it resolves against `cwd`, so the
    # suite imports the sandbox's copy of the package rather than the one in
    # the checkout. CI's `pip install -e .` must not win here, which is what
    # the baseline run below actually proves -- an editable install shadowing
    # the sandbox would make the very first mutation report MISSED.
    env = {**os.environ, "PYTHONPATH": "src"}
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", *targets, "-q", "--no-header", "-x",
             "-W", "error::UserWarning", *extra],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=SUITE_TIMEOUT,
        )
    except subprocess.TimeoutExpired as expired:
        # Not FAILED. The suite never reached a verdict, so this run says
        # nothing at all about the mutation it was checking.
        return DID_NOT_FINISH, f"suite timed out after {expired.timeout}s", ""
    output = proc.stdout + proc.stderr
    tail = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    outcome = PASSED if proc.returncode == 0 else FAILED
    return outcome, (tail[-1] if tail else "no output"), output


_SKIPPED_LINE = re.compile(r"^SKIPPED \[(\d+)\] ([^:\s]+):")


def skips_by_module(output: str) -> dict[str, int]:
    """Total the `-rs` short summary's skips per test module."""
    counts: dict[str, int] = {}
    for line in output.splitlines():
        found = _SKIPPED_LINE.match(line.strip())
        if found:
            counts[found.group(2)] = counts.get(found.group(2), 0) + int(found.group(1))
    return counts


def blind_spot(sandbox_output: str, checkout_output: str) -> dict[str, int]:
    """Tests the sandbox skips that a real checkout runs, per module.

    A module that skips for the same reason in both trees (SSA_NAMES_ZIP being
    unset, say) is not a blind spot; only the extra skips the sandbox itself
    causes are.
    """
    checkout = skips_by_module(checkout_output)
    extra = {}
    for module, count in skips_by_module(sandbox_output).items():
        lost = count - checkout.get(module, 0)
        if lost > 0:
            extra[module] = lost
    return extra


def undeclared_blind_spot(measured: dict[str, int]) -> dict[str, int]:
    """The measured blind spot minus what SANDBOX_BLIND_SPOT admits to."""
    return {m: n for m, n in measured.items() if m not in SANDBOX_BLIND_SPOT}


def report_blind_spot(baseline_output: str) -> int:
    """Print the sandbox-versus-checkout skip delta; 1 if it is wider than declared.

    The baseline run's own `-rs` summary names every module that skipped in the
    sandbox. Only those modules are re-run against the real checkout, so the
    measurement costs one short pytest run rather than a second full suite.
    """
    suspects = sorted(skips_by_module(baseline_output))
    if not suspects:
        print("sandbox skew: nothing skipped in the sandbox")
        return 0

    outcome, msg, checkout_output = run_suite(
        ROOT, tuple(suspects), ("-rs", "-p", "no:cacheprovider")
    )
    if outcome != PASSED:
        # Not fatal here: the `test` job gates the real tree. Say so rather
        # than print a delta measured against a run that did not complete.
        print(f"sandbox skew: not measured, the checkout probe run {outcome} ({msg})")
        return 0

    measured = blind_spot(baseline_output, checkout_output)
    if not measured:
        print("sandbox skew: none, the sandbox runs every test the checkout does")
        return 0

    total = sum(measured.values())
    detail = ", ".join(f"{module} ({count})" for module, count in sorted(measured.items()))
    print(f"sandbox skew: {total} test(s) skip in the sandbox but run in the checkout: {detail}")

    undeclared = undeclared_blind_spot(measured)
    if undeclared:
        print("\nthese modules go blind in the sandbox and are not declared in SANDBOX_BLIND_SPOT:")
        for module, count in sorted(undeclared.items()):
            print(f"  - {module} ({count} test(s))")
        print("  declare them with a reason, or make the sandbox carry what they need")
        return 1
    return 0


def check(entry: tuple[str, str, str, str], sandboxes: "queue.Queue[pathlib.Path]") -> tuple[str, str, str]:
    """Verify one entry. Returns (verdict, name, detail); verdict is the label."""
    name, relpath, original, reverted = entry
    source = ROOT / relpath
    if not source.exists():
        # A missing file verifies exactly as much as a missing snippet does:
        # nothing. It used to crash the whole run instead, which meant one
        # entry naming a file absent on the current branch took down every
        # later entry's check too.
        return STALE, name, f"{relpath} does not exist, so nothing was verified"
    text = source.read_text()
    if original not in text:
        return STALE, name, "snippet not found, so nothing was verified"

    sandbox = sandboxes.get()
    try:
        path = sandbox / relpath
        path.write_text(text.replace(original, reverted, 1))
        try:
            outcome, msg, _ = run_suite(sandbox)
        finally:
            # The sandbox is reused by the next entry, so put the file back
            # even though nothing outside this directory can see it.
            path.write_text(text)
    finally:
        sandboxes.put(sandbox)
    # Anything that is not a completed pass or a completed failure is
    # inconclusive. The mapping is deliberately explicit rather than a
    # `not ok`: only a suite that ran to a red verdict earns CAUGHT.
    verdict = {FAILED: CAUGHT, PASSED: MISSED}.get(outcome, INCONCLUSIVE)
    return verdict, name, msg


def shard(entries: list, spec: str | None) -> list:
    """Return the `i/N` slice of `entries`, or all of them when spec is None.

    Strided, not chunked, and deliberately so: `entries[i-1::N]` covers every
    index exactly once for *any* N, so no arithmetic can drop an entry the way
    a start/stop chunk calculation can when the count does not divide evenly.
    That property is the whole point. A shard that silently skips a mutation
    reports a pass it never earned, which is the one failure mode this harness
    exists to prevent.
    """
    if spec is None:
        return list(entries)
    index, _, count = spec.partition("/")
    if not count.isdigit() or not index.isdigit():
        raise SystemExit(f"--shard wants i/N with both parts numeric, got {spec!r}")
    index, count = int(index), int(count)
    if count < 1 or not 1 <= index <= count:
        raise SystemExit(f"--shard {spec}: need 1 <= i <= N and N >= 1")
    return list(entries)[index - 1 :: count]


def main(argv: list[str]) -> int:
    spec = None
    if argv:
        if len(argv) != 2 or argv[0] != "--shard":
            raise SystemExit(f"usage: {sys.argv[0]} [--shard i/N]")
        spec = argv[1]
    entries = shard(MUTATIONS, spec)
    # Printed so the sum across a CI matrix is auditable from the logs alone:
    # the counts have to add up to the total, or a shard was left unrun.
    print(f"shard {spec or 'all'}: {len(entries)} of {len(MUTATIONS)} mutations")
    if not entries:
        raise SystemExit(f"--shard {spec}: no mutations in this shard, so nothing would be verified")

    with tempfile.TemporaryDirectory(prefix="mutation-check-") as tmp:
        # One sandbox per worker, not per mutation: the copy is reused, so the
        # copy cost is paid `workers` times rather than once per entry.
        workers = min(len(entries), os.cpu_count() or 1)
        sandboxes: "queue.Queue[pathlib.Path]" = queue.Queue()
        for i in range(workers):
            box = pathlib.Path(tmp) / f"w{i}"
            shutil.copytree(ROOT, box, ignore=shutil.ignore_patterns(*SANDBOX_SKIP), symlinks=True)
            sandboxes.put(box)

        print(f"baseline ({workers} workers): ", end="", flush=True)
        first = sandboxes.get()
        outcome, msg, output = run_suite(first, extra=("-rs",))
        sandboxes.put(first)
        print(f"{'PASS' if outcome == PASSED else 'FAIL'}  {msg}")
        if outcome != PASSED:
            # The whole run stops here, so print what actually failed.
            # Reporting only "baseline must pass" leaves no way to diagnose it
            # from a CI log, where nobody can re-run the suite by hand.
            print("\n--- baseline failure output ---")
            print(output)
            print("baseline must pass before mutating")
            return 1

        # The baseline PASS above is earned over the sandbox's suite, which is
        # not quite the checkout's. Say by how much, and refuse to run if the
        # difference is wider than SANDBOX_BLIND_SPOT admits.
        if report_blind_spot(output):
            return 1

        failures: dict[str, list[str]] = {verdict: [] for verdict in FAILURE_HEADLINE}
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            # `map` yields in submission order, so the log reads the same way
            # it did when this ran one entry at a time.
            for verdict, name, detail in pool.map(lambda e: check(e, sandboxes), entries):
                if verdict == STALE:
                    print(f"  STALE  {name}: {detail}")
                else:
                    print(f"  {verdict:<12} {name}\n                -> {detail}")
                if verdict in failures:
                    failures[verdict].append(name)

    print()
    for verdict, names in failures.items():
        if not names:
            continue
        print(f"{len(names)} {FAILURE_HEADLINE[verdict]}")
        for name in names:
            print(f"  - {name}")
        if verdict == STALE:
            print("  fix the snippet; a stale mutation reads as a pass and is not one")
        if verdict == INCONCLUSIVE:
            print("  re-run these; an unfinished run is not evidence of anything")
    if not any(failures.values()):
        print("every logged fix is covered by a failing test")
    return 1 if any(failures.values()) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
