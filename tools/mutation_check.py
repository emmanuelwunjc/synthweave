"""Mutation harness: revert each logged fix, confirm the suite goes red.

A fix with no test that catches its absence is not locked down. Each revert is
applied to a throwaway copy of the checkout, the suite runs against that copy,
and the harness reports whether the suite noticed.

Four properties are load-bearing and none is traded for speed:

- MISSED: a reverted fix that no test catches fails the run.
- STALE: an entry whose file or snippet no longer matches verified nothing, so
  it fails the run too.
- INCONCLUSIVE: a suite run that never finished verified nothing either, so it
  is neither a pass nor a catch, and it fails the run as well.
- INCIDENTAL: a red suite is not automatically a catch. An entry may name the
  test(s) that own the property it claims to pin, and a named entry is CAUGHT
  only if one of them is red. A red anywhere else fails the run instead.

INCONCLUSIVE is the direction this harness must never fail in. "I could not
tell" resolving to "covered" is worse than a crash, because a crash gets
noticed. INCIDENTAL is the same hazard one level in: a red for the wrong
reason resolving to "covered" reads exactly like coverage and is not.

    python3 tools/mutation_check.py --audit        # which tests each revert breaks

`--audit` runs every entry without `-x` and prints the whole failure set, which
is how an entry worth pinning gets found rather than guessed. It is a
diagnostic and always exits 0; the gate is the pin.

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

# Two more ways a pytest invocation ends without saying anything about the
# code. Both used to be folded into `returncode != 0`, i.e. into FAILED, i.e.
# into CAUGHT: a nodeid that no longer resolves and an internal pytest error
# each read as "a test noticed the revert".
NOT_COLLECTED = "collected no such test"
BROKE = "ended without a verdict"

# pytest's documented exit codes. 0 and 1 are the only two that mean a suite
# ran and reached a verdict.
_EXIT_OUTCOME = {0: PASSED, 1: FAILED, 4: NOT_COLLECTED, 5: NOT_COLLECTED}

# Verdicts. CAUGHT is the only one that counts as coverage; the rest fail the
# run, each for its own reason.
CAUGHT, MISSED, STALE, INCONCLUSIVE = "CAUGHT", "MISSED", "STALE", "INCONCLUSIVE"
# A red suite that no test naming the property accounted for. Not MISSED (the
# suite genuinely went red) and not CAUGHT (the red was about something else).
INCIDENTAL = "INCIDENTAL"

# Printed above the names when a verdict fails the run.
FAILURE_HEADLINE = {
    STALE: "mutation(s) no longer match the code and verified nothing:",
    MISSED: "fix(es) with NO regression coverage:",
    INCIDENTAL: "mutation(s) caught only by a red that says nothing about the property named:",
    INCONCLUSIVE: "mutation(s) whose suite run never finished, so nothing was verified:",
}

# (issue, file, original_snippet, reverted_snippet[, catchers])
#
# `catchers` is the #158 mechanism: a tuple of pytest node ids naming the
# test(s) that own the property the entry claims to pin. When it is present the
# entry is CAUGHT only if at least one of them is red under the revert, so a
# suite that went red somewhere else entirely no longer counts. See `check()`
# for why a missing node id is STALE rather than a catch.
#
# It stays syntactically optional here so an entry can be added and pinned in
# two steps and so `check()` keeps a defined answer for a four-element tuple.
# It is not optional in practice: `test_every_entry_names_a_catcher` fails the
# `test` job on an unpinned entry, naming it. That is deliberate. Asking in a
# comment is what left 109 of 113 entries unpinned after #158 landed.
#
# The pins were chosen from a full `--audit` run, which lists every test each
# revert breaks; the named catcher is the one that asserts the property, not
# merely one that happened to be red.
#
# Two known limitations, both recorded rather than silently lived with:
#
# - `test_own_makes_a_view_safe_to_write_to`, the sole catcher for the two
#   `own()` entries, self-skips when `pd.errors.SettingWithCopyWarning` is
#   absent, i.e. under pandas 3. That is latent, not live: this harness runs
#   only in the `mutation-shard` job, which installs the pinned pandas 2. The
#   `test-pandas3` job runs pytest and never this file. Under pandas 3's
#   copy-on-write the hazard genuinely does not exist, so there is no test to
#   write there.
# - A few entries are pinned to the best catcher that exists rather than to a
#   test that asserts the property outright. Each has its own issue.
MUTATIONS = [
    (
        "I1 synthesizer table scoping",
        "src/synthweave/stages/synthesize.py",
        """        if self.tables is not None and table.name not in self.tables:
            yield from chunks
            return
""",
        "",
        (
            "tests/test_synthesis_and_plugins.py::test_a_table_outside_the_synthesizer_scope_passes_through",
            "tests/test_mode.py::test_a_real_data_column_stays_out_of_a_table_that_did_not_declare_it",
        ),
    ),
    (
        "I3 fit buffer truncation (chunk invariance)",
        "src/synthweave/stages/base.py",
        """    if len(buffered) > max_rows:
        buffered = buffered.iloc[:max_rows]
""",
        "",
        (
            "tests/test_synthesis_and_plugins.py::test_above_the_cap_the_fit_is_capped",
            "tests/test_pipeline.py::test_chunk_size_cannot_change_output_with_every_stage_active",
        ),
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
        (
            "tests/test_synthesis_and_plugins.py::test_declared_structure_survives_synthesis",
            "tests/test_synthesis_and_plugins.py::test_empirical_structure_is_learned_from_supplied_data",
        ),
    ),
    (
        "I5 link before noise (identifier noise possible)",
        "src/synthweave/pipeline.py",
        "for stage in (self.synthesizer, self.linker, self.noiser):",
        "for stage in (self.synthesizer, self.noiser, self.linker):",
        (
            "tests/test_pipeline.py::test_an_identifier_column_can_be_noised_on_purpose",
        ),
    ),
    (
        "I6 own() defensive copy",
        "src/synthweave/stages/base.py",
        """    if getattr(chunk, "_is_view", False) or getattr(chunk, "_is_copy", None) is not None:
        return chunk.copy()
    return chunk""",
        """    return chunk""",
        (
            "tests/test_synthesis_and_plugins.py::test_own_makes_a_view_safe_to_write_to",
        ),
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
        (
            "tests/test_synthesis_and_plugins.py::test_a_synthesized_column_keeps_the_dtype_it_had",
            "tests/test_synthesis_and_plugins.py::test_a_synthesized_column_keeps_an_extension_dtype[Int64]",
        ),
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
        (
            "tests/test_pipeline.py::test_an_identifier_tag_cannot_overwrite_a_table_column",
        ),
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
        (
            "tests/test_pipeline.py::test_an_identifier_tag_cannot_overwrite_a_carried_attribute",
        ),
    ),
    (
        "I13 empty table still writes a file",
        "src/synthweave/io.py",
        """        elif not self._wrote_header:
            self.write_empty()""",
        """        elif self.format == "csv" and not self._wrote_header:
            self.path.write_text("")""",
        (
            "tests/test_synthesis_and_plugins.py::test_a_table_with_no_rows_still_writes_a_readable_file",
            "tests/test_synthesis_and_plugins.py::test_an_empty_table_writes_the_same_columns_as_a_full_one",
        ),
    ),
    (
        "I12 writer reconciles a chunk against the file schema",
        "src/synthweave/io.py",
        """        elif batch.schema != self._writer.schema:
            batch = self._reconcile(batch, pa)""",
        "",
        (
            "tests/test_synthesis_and_plugins.py::test_a_widening_type_shift_between_chunks_is_absorbed",
            "tests/test_synthesis_and_plugins.py::test_an_undeclared_column_type_that_shifts_between_chunks_names_itself",
        ),
    ),
    (
        "I17 declared-numeric coercion names the column",
        "src/synthweave/stages/synthesize.py",
        """        converted = pd.to_numeric(series, errors="coerce")
        bad = series[converted.isna() & series.notna()]""",
        """        converted = pd.to_numeric(series, errors="coerce")
        bad = []""",
        (
            "tests/test_synthesis_and_plugins.py::test_declaring_an_unparseable_column_numeric_says_which_column",
        ),
    ),
    (
        "I13 empty table keeps its declared columns",
        "src/synthweave/pipeline.py",
        # `_concat`'s stand-in frame moved out to `Pipeline._empty_frame` with
        # the #82 dtype fix. Same fix, same revert: hand back a frame with no
        # columns and see whether the suite notices.
        "        return empty\n",
        "        return pd.DataFrame()\n",
        (
            "tests/test_pipeline.py::test_a_table_that_emits_no_rows_still_has_its_columns",
        ),
    ),
    (
        "I14 identifier width vs population",
        "src/synthweave/validation.py",
        """    if expected < TOLERABLE_COLLISIONS:
        return""",
        """    if True:
        return""",
        (
            "tests/test_pipeline.py::test_a_digit_count_too_small_for_the_population_is_rejected",
            "tests/test_pipeline.py::test_the_recommended_digit_count_is_the_tightest_one_that_works",
        ),
    ),
    (
        "I14 digits past the hash width",
        "src/synthweave/schema.py",
        "        if self.digits > MAX_DIGITS:",
        "        if self.digits > 10**9:",
        (
            "tests/test_pipeline.py::test_a_digit_count_past_the_hash_width_is_rejected",
        ),
    ),
    (
        "I15 joint prior picks over positions",
        "src/synthweave/stages/synthesize.py",
        "            positions = np.arange(len(pairs), dtype=object)",
        "            positions = np.array(pairs, dtype=object)",
        (
            "tests/test_synthesis_and_plugins.py::test_a_joint_prior_shapes_the_relationship_it_declares",
            "tests/test_synthesis_and_plugins.py::test_a_joint_prior_is_chunk_invariant",
        ),
    ),
    (
        "I16 repeated period rejected",
        "src/synthweave/schema.py",
        """        if repeated:""",
        """        if False:""",
        (
            "tests/test_pipeline.py::test_a_repeated_period_is_rejected",
        ),
    ),
    (
        "I10/I12 rules declare their dtype",
        "src/synthweave/rules.py",
        """    dtype = declared_dtype(rule)
    return values if dtype is None else values.astype(dtype)""",
        """    return values""",
        (
            "tests/test_noise.py::test_a_zero_rate_op_leaves_the_column_dtype_untouched",
            "tests/test_pipeline.py::test_a_table_that_emits_no_rows_keeps_its_non_empty_dtypes",
        ),
    ),
    (
        "I17 numeric decided by dtype, not by guessing",
        "src/synthweave/stages/synthesize.py",
        "        return column in self.numeric or pd.api.types.is_numeric_dtype(series)",
        """        return column in self.numeric or (
            pd.to_numeric(series, errors="coerce").notna().mean() > 0.9
        )""",
        (
            "tests/test_synthesis_and_plugins.py::test_a_numeric_column_with_a_few_sentinel_values_still_fits",
        ),
    ),
    (
        "I29 donor diagnostics silently stale on synthesizer reuse",
        "src/synthweave/stages/synthesize.py",
        "        self._fitted[table.name] = model",
        "        self._fitted.setdefault(table.name, model)",
        (
            "tests/test_fidelity.py::test_donor_diagnostics_reflects_only_the_last_run_when_synthesizer_is_reused",
        ),
    ),
    (
        "I30 donor diagnostics readable mid-stream, before every chunk applied",
        "src/synthweave/stages/synthesize.py",
        """            complete = self._complete.get(table, False)
            return {table: dict(model.empty_donor_counts)} if model and complete else {}""",
        """            return {table: dict(model.empty_donor_counts)} if model else {}""",
        (
            "tests/test_fidelity.py::test_donor_diagnostics_excludes_a_table_still_mid_stream",
        ),
    ),
    (
        "I28 SSN area 666 remapped onto 667 instead of skipped",
        "src/synthweave/connectors/faker_names.py",
        """        area = _hash.integers(keys, seed, f"{salt}\\x00area", 1, 899)
        area = np.where(area >= 666, area + 1, area)""",
        """        area = _hash.integers(keys, seed, f"{salt}\\x00area", 1, 900)
        area = np.where(area == 666, 667, area)""",
        (
            "tests/test_faker_names.py::test_ssn_area_is_not_skewed_by_the_666_exclusion",
        ),
    ),
    (
        "I31 structure name needing config raises a bare TypeError",
        "src/synthweave/stages/synthesize.py",
        "        return _resolve_structure_name(structure)",
        '        return resolve("structure", structure)',
        (
            "tests/test_synthesis_and_plugins.py::test_structure_by_name_needing_config_names_the_missing_argument",
        ),
    ),
    (
        "#10 per-row rate function is actually applied",
        "src/synthweave/stages/noise.py",
        """                    if callable(rate):
                        rate = _row_rates(rate, chunk, f"{table.name}.{column}.{op.name}")""",
        "",
        (
            "tests/test_pipeline.py::test_missingness_rate_can_vary_by_row",
            "tests/test_pipeline.py::test_a_row_varying_rate_stays_deterministic_and_chunk_invariant",
        ),
    ),
    (
        "#17 header-only ACS response rejected",
        "src/synthweave/connectors/acs_pums.py",
        "    if not payload or len(payload) <= 1:",
        "    if not payload or len(payload) < 1:",
        (
            "tests/test_acs_pums.py::test_a_header_only_response_is_a_failure_not_an_empty_frame",
        ),
    ),
    (
        "#19 GeoNames TSV parsed without CSV quote handling",
        "src/synthweave/connectors/geonames.py",
        '    return list(csv.reader(io.StringIO(raw), delimiter="\\t", quoting=csv.QUOTE_NONE))',
        '    return list(csv.reader(io.StringIO(raw), delimiter="\\t"))',
        (
            "tests/test_geonames.py::test_a_leading_quote_does_not_swallow_the_following_row",
        ),
    ),
    (
        "#18 missing birth year rejected before the range guard",
        "src/synthweave/connectors/ssa_names.py",
        "        missing = pd.isna(years)\n        if missing.any():",
        "        missing = pd.isna(years)\n        if False:",
        (
            "tests/test_ssa_names.py::test_a_nan_birth_year_is_rejected_rather_than_left_as_none",
        ),
    ),
    (
        "#16 SSA cache filename distinguishes the source",
        "src/synthweave/connectors/ssa_names.py",
        "        _cache_filename(source),",
        '        "ssa_names.csv",',
        (
            "tests/test_ssa_names.py::test_two_sources_do_not_share_one_cache_file",
        ),
    ),
    (
        "#13 namedtuple refused rather than flattened into a Choice",
        "src/synthweave/rules.py",
        '    if isinstance(value, tuple) and hasattr(value, "_fields"):',
        "    if False:",
        (
            "tests/test_schema_shorthands.py::test_coerce_rule_refuses_a_namedtuple_instead_of_flattening_it",
        ),
    ),
    (
        "offline: surname weight is count * pct/100",
        "src/synthweave/connectors/census_surnames.py",
        """                self._data["count"].to_numpy(dtype=np.float64)
                * self._data[column].to_numpy(dtype=np.float64)
                / 100.0""",
        """                self._data["count"].to_numpy(dtype=np.float64)""",
        (
            "tests/test_census_surnames.py::test_pool_weights_are_count_times_percent",
        ),
    ),
    (
        "offline: address fields in a group share one row",
        "src/synthweave/connectors/geonames.py",
        'idx = _hash.integers(keys, seed, f"usaddress\\x00{self.group}", 0, len(self._data))',
        'idx = _hash.integers(keys, seed, f"usaddress\\x00{salt}", 0, len(self._data))',
        (
            "tests/test_geonames.py::test_fields_in_one_group_come_from_the_same_real_row",
        ),
    ),
    (
        "offline: SSA pool narrows to the requested sex",
        "src/synthweave/connectors/ssa_names.py",
        """        if sex is not None:
            subset = subset[subset["sex"] == sex]""",
        "",
        (
            "tests/test_ssa_names.py::test_pooling_a_year_and_sex_narrows_to_that_sex",
        ),
    ),
    (
        "#37 joint disagreeing with a declared marginal is rejected",
        "src/synthweave/stages/synthesize.py",
        "        _check_joints_agree_with_marginals(self.marginals, self.joints)",
        "        pass",
        (
            "tests/test_synthesis_and_plugins.py::test_a_joint_disagreeing_with_a_declared_marginal_is_rejected",
        ),
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
        (
            "tests/test_fidelity.py::test_no_leaf_is_ever_donorless_however_hard_you_push",
        ),
    ),
    (
        "#12 carry=* unknown entity raises SchemaError with table context",
        "src/synthweave/schema.py",
        '            raise SchemaError(f"table {table.name!r}: {e.args[0]}") from e',
        "            raise",
        (
            "tests/test_schema_shorthands.py::test_carry_star_with_an_unknown_entity_raises_schema_error_naming_the_table",
        ),
    ),
    (
        "#14 numpy scalars coerce symmetrically",
        "src/synthweave/rules.py",
        "    if isinstance(value, np.generic):\n        return Constant(value.item())",
        "    if False:\n        return Constant(value.item())",
        (
            "tests/test_schema_shorthands.py::test_numpy_scalars_coerce_symmetrically",
        ),
    ),
    (
        "#20 unknown numeric state FIPS code rejected",
        "src/synthweave/connectors/acs_pums.py",
        "        if code not in set(_STATE_FIPS.values()):",
        "        if False:",
        (
            "tests/test_acs_pums.py::test_an_unknown_numeric_state_code_is_rejected",
        ),
    ),
    (
        "#15 malformed structure dict rejected at the coercion",
        "src/synthweave/stages/synthesize.py",
        "        if bad:",
        "        if False:",
        (
            "tests/test_synthesis_and_plugins.py::test_a_malformed_structure_dict_is_rejected_at_the_coercion",
        ),
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
        (
            "tests/test_schema_shorthands.py::test_carry_star_resolves_per_schema_not_once_per_table",
        ),
    ),
    (
        "#10 per-row rate range check",
        "src/synthweave/stages/noise.py",
        "    if not np.all((rates >= 0.0) & (rates <= 1.0)):",
        "    if False:",
        (
            "tests/test_pipeline.py::test_a_row_varying_rate_outside_zero_to_one_fails_loudly",
        ),
    ),
    (
        "#41 NaN/inf Choice weight rejected instead of collapsing the column",
        "src/synthweave/_hash.py",
        "        if not np.all(np.isfinite(w)):",
        "        if False:",
        (
            "tests/test_pipeline.py::test_a_nan_choice_weight_is_rejected_instead_of_collapsing_the_column",
            "tests/test_pipeline.py::test_an_infinite_choice_weight_is_rejected_instead_of_collapsing_the_column",
        ),
    ),
    (
        "#43 CSV chunk writer guards column order/shape between chunks",
        "src/synthweave/io.py",
        "            elif list(chunk.columns) != self._csv_columns:",
        "            elif False:",
        (
            "tests/test_io.py::test_a_csv_chunk_missing_a_column_is_rejected",
            "tests/test_io.py::test_csv_chunks_with_reordered_columns_still_land_under_the_right_header",
        ),
    ),
    (
        "#44 a carried attribute's transitive dependency chain is drawn in full",
        "src/synthweave/stages/generate.py",
        "            if name in needed:\n                needed.update(entity.attributes[name].depends_on())",
        "            pass",
        (
            "tests/test_pipeline.py::test_carrying_a_leaf_draws_its_whole_transitive_dependency_chain",
        ),
    ),
    (
        "#40 CART trees seeded so tied splits break the same way every fit",
        "src/synthweave/stages/synthesize.py",
        "            random_state = int(_hash.hash_key(self.seed, f\"cart\\x00{target}\"), 16) % (2**32)",
        "            random_state = None",
        (
            "tests/test_synthesis_and_plugins.py::test_synthesis_is_deterministic_under_tied_splits",
            "tests/test_synthesis_and_plugins.py::test_multi_column_synthesis_is_deterministic",
        ),
    ),
    (
        "#42 a noised column keeps its dtype where the values still allow it",
        "src/synthweave/stages/noise.py",
        "                chunk[column] = _restore_dtype(values, original_dtype)",
        "                chunk[column] = values",
        (
            "tests/test_noise.py::test_missing_on_an_int_column_widens_to_nullable_int",
            "tests/test_noise.py::test_missing_on_a_category_column_stays_categorical",
        ),
    ),
    (
        "#45.1a duplicate table carry names are rejected",
        "src/synthweave/validation.py",
        '    _check_unique(list(table.carry), f"{where} carried attribute")',
        "    pass",
        (
            "tests/test_pipeline.py::test_a_table_carrying_the_same_attribute_twice_is_rejected",
        ),
    ),
    (
        "#45.1b duplicate table identifier names are rejected",
        "src/synthweave/validation.py",
        '    _check_unique(list(table.identifiers), f"{where} identifier")',
        "    pass",
        (
            "tests/test_pipeline.py::test_a_table_asking_for_the_same_identifier_twice_is_rejected",
        ),
    ),
    (
        "#45.2a identifier-width fix drops the erroneous extra digit",
        "src/synthweave/validation.py",
        "    needed = len(str(population * population // 2))",
        "    needed = len(str(population * population // 2)) + 1",
        (
            "tests/test_pipeline.py::test_the_recommended_digit_count_is_the_tightest_one_that_works",
        ),
    ),
    (
        "#45.2b a population past the digit limit says so instead of recommending it",
        "src/synthweave/validation.py",
        "    if needed > MAX_DIGITS:",
        "    if False:",
        (
            "tests/test_pipeline.py::test_a_population_needing_more_than_18_digits_says_so_instead_of_recommending_one",
        ),
    ),
    (
        "#45.3 Uniform rejects high <= low instead of silently descending",
        "src/synthweave/rules.py",
        "        if self.high <= self.low:",
        "        if False:",
        (
            "tests/test_pipeline.py::test_uniform_with_high_not_above_low_is_rejected",
        ),
    ),
    (
        "#46.1 two joints sharing a column are rejected",
        "src/synthweave/stages/synthesize.py",
        "        _check_joints_do_not_share_a_column(self.joints)\n        _check_joints_agree_with_marginals(self.marginals, self.joints)",
        "        _check_joints_agree_with_marginals(self.marginals, self.joints)",
        (
            "tests/test_synthesis_and_plugins.py::test_two_joints_sharing_a_column_are_rejected",
        ),
    ),
    (
        "#46.2 a numeric Prior marginal carries its natural dtype",
        "src/synthweave/stages/synthesize.py",
        "            frame[column] = picked.astype(natural) if natural.kind in \"iuf\" else picked",
        "            frame[column] = picked",
        (
            "tests/test_synthesis_and_plugins.py::test_a_numeric_prior_marginal_stays_numeric",
        ),
    ),
    (
        "#46.3 fit_cap holds for a supplied structure source across seeds",
        "src/synthweave/stages/synthesize.py",
        "                    train = train.iloc[:cap]",
        "                    keys = np.asarray(train.index, dtype=str).astype(object)\n                    pick = _hash.unit(keys, ctx.seed, f\"fitsample\\x00{table.name}\") < (cap / len(train))\n                    train = train.loc[pick]",
        (
            "tests/test_synthesis_and_plugins.py::test_fit_cap_holds_across_seeds_for_a_supplied_structure_source",
        ),
    ),
    (
        "#64 check_rule catches a non-deterministic rule",
        "src/synthweave/rules.py",
        '    repeat = rule.draw(keys, seed=seed, salt=salt, frame=frame)\n    _assert_same_values(\n        keys, baseline, repeat, "calling draw() twice with identical input gave different "\n        "values back (the rule is not deterministic, e.g. it reaches for random state)"\n    )',
        "    pass",
        (
            "tests/test_conformance.py::test_a_non_deterministic_rule_is_rejected",
        ),
    ),
    (
        "#64 check_rule catches a position-keyed rule via a shuffled key array",
        "src/synthweave/rules.py",
        "    order = np.arange(n)[::-1]",
        "    order = np.arange(n)",
        (
            "tests/test_conformance.py::test_a_row_position_keyed_rule_is_rejected",
        ),
    ),
    (
        "#64 check_rule catches a chunk-size-dependent rule via a split call",
        "src/synthweave/rules.py",
        '    combined = np.concatenate([first, second])\n    _assert_same_values(\n        keys, baseline, combined, "splitting the keys across two calls changed a value "\n        "(the rule is not chunk invariant, e.g. it reads chunk-level state)"\n    )',
        "    pass",
        (
            "tests/test_conformance.py::test_a_chunk_size_dependent_rule_is_rejected",
        ),
    ),
    (
        "#89.1 two attributes sharing one ACS variable both keep a column",
        "src/synthweave/mode.py",
        "            self._fetched = pd.DataFrame(\n                {name: fetched[variable] for name, variable in self._variables.items()}\n            )",
        "            self._fetched = fetched.rename(\n                columns={variable: name for name, variable in self._variables.items()}\n            )",
        (
            "tests/test_mode.py::test_two_attributes_can_share_one_acs_variable",
        ),
    ),
    (
        "#89.2 scope mode generalizes by epsilon instead of CART's defaults",
        "src/synthweave/mode.py",
        '            "synthesizer": _epsilon_chain(self._scope_epsilon, self._fetched, placement)',
        '            "synthesizer": _empirical_cart(list(self._variables), self._fetched)',
        (
            "tests/test_mode.py::test_scope_epsilon_sets_the_synthesizers_max_depth_knob",
            "tests/test_mode.py::test_scope_attributes_at_different_epsilons_get_two_synthesizers",
        ),
    ),
    (
        "#89.3 scope epsilon is validated, not clamped (mode level)",
        "src/synthweave/mode.py",
        '        _check_epsilon(epsilon, "scope")\n',
        "",
        (
            "tests/test_mode.py::test_scope_rejects_a_non_positive_epsilon[-1]",
            "tests/test_mode.py::test_scope_rejects_a_non_positive_epsilon[0]",
        ),
    ),
    (
        "#89.4 scope epsilon is validated, not clamped (per attribute)",
        "src/synthweave/mode.py",
        '        if epsilon is not None:\n            _check_epsilon(epsilon, f"attribute {name!r}")\n        self._variables[name] = variable',
        "        self._variables[name] = variable",
        (
            "tests/test_mode.py::test_scope_rejects_a_non_positive_per_attribute_epsilon[0]",
            "tests/test_mode.py::test_scope_rejects_a_non_positive_per_attribute_epsilon[-1]",
        ),
    ),
    (
        "#88 non-positive real_data epsilon rejected instead of clamped to 0.01",
        "src/synthweave/mode.py",
        '    if epsilon <= 0:\n        raise ValueError(f"{where}: epsilon must be positive, got {epsilon!r}")',
        "    pass",
        (
            "tests/test_mode.py::test_real_data_rejects_a_non_positive_epsilon[0]",
            "tests/test_mode.py::test_real_data_rejects_a_non_positive_epsilon[-1]",
        ),
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
        (
            "tests/test_mode.py::test_attribute_rejects_an_unknown_kwarg_naming_the_attribute",
        ),
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
        (
            "tests/test_mode.py::test_scope_rejects_an_unknown_kwarg_naming_the_attribute",
            "tests/test_mode.py::test_scope_rejects_a_misspelled_noise_rate_rather_than_dropping_it",
        ),
    ),
    (
        "#87 mode noise resolves against the schema's expanded carry (carry=*)",
        "src/synthweave/mode.py",
        "            for column_name in list(table.carry) + list(table.columns):",
        """            for column_name in list(table.columns) + (
                list(table.carry) if isinstance(table.carry, list) else []
            ):""",
        (
            "tests/test_mode.py::test_missing_rate_reaches_a_column_carried_by_wildcard",
        ),
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
        (
            "tests/test_mode.py::test_missing_rate_reaches_a_column_declared_after_the_table",
        ),
    ),
    (
        "#48 unregister actually removes the entry",
        "src/synthweave/registry.py",
        "    del table[name]",
        "    pass",
        (
            "tests/test_registry.py::test_unregister_removes_an_entry",
        ),
    ),
    (
        "#48 the autouse registry-reset fixture restores the pre-test snapshot",
        "tests/conftest.py",
        "    registry._restore(snapshot)",
        "    pass",
        (
            "tests/test_registry.py::test_a_registration_here_does_not_survive_to_the_next_test",
            "tests/test_registry.py::test_the_previous_tests_registration_did_not_leak",
        ),
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
        (
            "tests/test_public_api.py::test_no_public_name_is_a_foreign_import",
        ),
    ),
    (
        "#63 faker_names validates Faker's private provider shape",
        "src/synthweave/connectors/faker_names.py",
        "    pool: Any = _checked_provider_pool(Provider, attr)",
        "    pool: Any = getattr(Provider, attr)",
        (
            "tests/test_faker_names.py::test_unweighted_provider_attribute_is_rejected_by_name",
            "tests/test_faker_names.py::test_missing_provider_attribute_names_the_attribute_and_the_version",
        ),
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
        (
            "tests/test_faker_names.py::test_unusable_weight_is_rejected_and_the_message_says_why[inf-non-finite]",
            "tests/test_faker_names.py::test_unusable_weight_is_rejected_and_the_message_says_why[0.01-non-numeric]",
        ),
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
        (
            "tests/test_acs_pums.py::test_a_response_that_fails_to_parse_is_not_cached",
        ),
    ),
    (
        "#62 I33 the .env walk stops at the project root",
        "src/synthweave/connectors/acs_pums.py",
        """        if any((directory / marker).exists() for marker in _PROJECT_ROOT_MARKERS):
            break
""",
        "",
        (
            "tests/test_acs_pums.py::test_dotenv_lookup_stops_at_the_project_root",
        ),
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
        (
            "tests/test_acs_pums.py::test_dotenv_at_the_project_root_is_read_before_the_walk_stops",
        ),
    ),
    (
        "#47 a noise rate function must return one rate per row",
        "src/synthweave/stages/noise.py",
        """    if rates.shape != (len(chunk),):
""",
        """    if False:
""",
        (
            "tests/test_pipeline.py::test_a_row_varying_rate_stays_deterministic_and_chunk_invariant",
        ),
    ),
    (
        "#96.2 an ExtensionDtype is restored through pandas, not ndarray.astype",
        "src/synthweave/stages/synthesize.py",
        "        if isinstance(dtype, pd.api.extensions.ExtensionDtype):\n            return pd.array(values, dtype=dtype)\n",
        "",
        (
            "tests/test_synthesis_and_plugins.py::test_a_synthesized_column_keeps_an_extension_dtype[Int64]",
            "tests/test_synthesis_and_plugins.py::test_a_synthesized_column_keeps_an_extension_dtype[category]",
        ),
    ),
    # --- #65 the derivation layer ------------------------------------------
    # Every value in the library routes through `_hash`, so a corruption here
    # is silently wrong everywhere at once. These deliberately are not
    # "delete the guard" mutations: each one is the shape a real arithmetic
    # slip would take, which is what the existing entries did not cover.
    (
        # Pinned (#156). This entry used to report CAUGHT on the strength of a
        # dtype-narrowing ValueError in a chunk-shift test: the mutation moves
        # every derived value in the library, so *something* somewhere was
        # always going to notice, and nothing asserted the separation itself.
        # One fixture edit away from MISSED, silently.
        "#65 hash_key keeps seed and salt separated",
        "src/synthweave/_hash.py",
        'return hashlib.sha256(f"{seed}\\x00{salt}".encode()).hexdigest()[:16]',
        'return hashlib.sha256(f"{seed}{salt}".encode()).hexdigest()[:16]',
        ("tests/test_hash_invariants.py::test_seed_and_salt_stay_separated_in_the_hash_key",),
    ),
    (
        "#65 the salt reaches the hash key (independent draws per salt)",
        "src/synthweave/_hash.py",
        'hash_key=hash_key(seed, salt)',
        'hash_key=hash_key(seed, "")',
        (
            "tests/test_geonames.py::test_a_separate_group_is_independent_of_the_first",
            "tests/test_pipeline.py::test_identifier_kinds_are_independent",
        ),
    ),
    (
        "#65 unit() divides by the full uint64 width",
        "src/synthweave/_hash.py",
        "_SCALE = np.float64(2.0**64)",
        "_SCALE = np.float64(2.0**63)",
        (
            "tests/test_hash_invariants.py::test_an_unweighted_choice_reaches_every_value_it_was_given",
            "tests/test_pipeline.py::test_missingness_lands_near_the_configured_rate",
        ),
    ),
    (
        "#65 integers() span excludes `high`",
        "src/synthweave/_hash.py",
        "    span = np.uint64(high - low)",
        "    span = np.uint64(high - low + 1)",
        (
            "tests/test_pipeline.py::test_event_grain_varies_row_count_per_entity",
            "tests/test_faker_names.py::test_ssn_area_within_valid_range",
        ),
    ),
    (
        "#65 normal() Box-Muller radius keeps the -2 factor",
        "src/synthweave/_hash.py",
        "    z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)",
        "    z = np.sqrt(-1.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)",
        (
            "tests/test_hash_invariants.py::test_normal_draws_have_the_requested_spread",
        ),
    ),
    (
        # Pinned (#156). The sole catcher was a KeyError for an unmapped
        # Census category, which is a test about a missing mapping and says
        # nothing about whether a pick reaches every value it was given.
        "#65 unweighted pick() spreads over every value",
        "src/synthweave/_hash.py",
        "        idx = np.minimum((u * len(values)).astype(np.int64), len(values) - 1)",
        "        idx = np.minimum((u * (len(values) - 1)).astype(np.int64), len(values) - 1)",
        ("tests/test_hash_invariants.py::test_an_unweighted_choice_reaches_every_value_it_was_given",),
    ),
    (
        "#65 weighted pick() normalizes the weights before the cumsum",
        "src/synthweave/_hash.py",
        "        idx = np.searchsorted(np.cumsum(w / total), u, side=\"right\")",
        "        idx = np.searchsorted(np.cumsum(w), u, side=\"right\")",
        (
            "tests/test_census_surnames.py::test_the_drawn_name_actually_follows_the_race_column",
        ),
    ),
    (
        "#65 derive_id() zero-pads to the declared width",
        "src/synthweave/_hash.py",
        '    text = np.char.zfill(n.astype("U"), digits)',
        '    text = n.astype("U")',
        (
            "tests/test_pipeline.py::test_identifiers_all_have_the_requested_width",
        ),
    ),
    (
        "#65 derive_id() uses the full declared keyspace",
        "src/synthweave/_hash.py",
        "    modulus = 10**digits",
        "    modulus = 10 ** (digits - 1)",
        (
            "tests/test_hash_invariants.py::test_identifiers_use_the_whole_declared_keyspace",
        ),
    ),
    (
        "#65 derive_id() namespaces by tag, so two tags are unrelated",
        "src/synthweave/_hash.py",
        '    n = (hash_u64(keys, seed, f"id\\x00{tag}") % np.uint64(modulus)).astype(np.int64)',
        '    n = (hash_u64(keys, seed, "id") % np.uint64(modulus)).astype(np.int64)',
        (
            "tests/test_pipeline.py::test_identifier_kinds_are_independent",
        ),
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
        (
            "tests/test_pipeline.py::test_a_digit_count_too_small_for_the_population_is_rejected",
        ),
    ),
    (
        "#65 the collision bound is quadratic in population (birthday bound)",
        "src/synthweave/validation.py",
        "    return population * population / (2 * 10**digits)",
        "    return population / (2 * 10**digits)",
        (
            "tests/test_pipeline.py::test_a_population_needing_more_than_18_digits_says_so_instead_of_recommending_one",
        ),
    ),
    (
        "#65 own() copies a frame pandas flagged as a copy, not only a view",
        "src/synthweave/stages/base.py",
        '    if getattr(chunk, "_is_view", False) or getattr(chunk, "_is_copy", None) is not None:',
        '    if getattr(chunk, "_is_view", False):',
        (
            "tests/test_synthesis_and_plugins.py::test_own_makes_a_view_safe_to_write_to",
        ),
    ),
    (
        # Retargeted 2026-08-04: #82 moved the empty-table build into
        # `_empty_frame`, so the old snippet stopped matching and the entry went
        # STALE. Same guarantee, same test, new line.
        "#65 an empty table keeps its declared column order, not a sorted one",
        "src/synthweave/pipeline.py",
        "            columns=columns,\n        )",
        "            columns=sorted(columns),\n        )",
        (
            "tests/test_pipeline.py::test_a_table_that_emits_no_rows_still_has_its_columns",
        ),
    ),
    (
        "#142 check_synthesizer names the dropped row key instead of raising KeyError",
        "src/synthweave/conformance.py",
        "    if ROW_KEY not in got.columns:",
        "    if False:",
        (
            "tests/test_synthesizer_conformance.py::test_a_synthesizer_dropping_the_row_key_is_rejected",
        ),
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
        (
            "tests/test_faker_bound_sync.py::test_source_constant_matches_the_bound_declared_in_pyproject",
            "tests/test_faker_bound_sync.py::test_the_runtime_error_names_the_bound_declared_in_pyproject",
        ),
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
        # Pinned (#158). The trap described above is exactly what the pin
        # mechanism is for, so this entry names the test that asserts the
        # guarantee rather than trusting the suite to be red for the right
        # reason. Confirmed: with the guard in place every remaining failure
        # reads `entity non-straddling: ...`.
        (
            "tests/test_generator_conformance.py::test_every_registered_generator_conforms",
        ),
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
        (
            "tests/test_deprecation.py::test_find_stack_level_is_one_when_called_from_outside_the_package",
            "tests/test_deprecation.py::test_nested_package_helpers_and_comprehensions_are_walked_over",
        ),
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
        (
            "tests/test_pipeline.py::test_typo_corrupts_a_value_with_no_latin_characters",
            "tests/test_noise.py::test_typo_delivers_the_configured_rate_on_every_script",
        ),
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
        (
            "tests/test_noise.py::test_missing_on_a_category_column_stays_categorical",
        ),
    ),
    (
        # #134. The entry above deletes the whole ExtensionDtype block, so the
        # null guard *inside* it never gets reverted on its own and nothing
        # says whether it is load-bearing. It is: `pd.array` maps a value the
        # dtype cannot hold to null where `astype` would have raised, so
        # without this clause a `Typo` result outside a category's set
        # disappears into a null and the column reads as clean -- the exact
        # papering over `_restore_dtype` exists to refuse.
        "#134 a cast that nulls a live value is a failed restore, not a clean column",
        "src/synthweave/stages/noise.py",
        """            if (pd.isna(restored) & ~pd.isna(values)).any():
                return values
""",
        "",
        ("tests/test_noise.py::test_typo_on_a_category_column_falls_back_to_object",),
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
        (
            "tests/test_pipeline.py::test_a_table_that_emits_no_rows_keeps_its_non_empty_dtypes",
        ),
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
        (
            "tests/test_leak_guard.py::test_compound_credential_names_are_caught[census_key=0123456789abcdef0123456789abcdef01234567]",  # leak-guard: allow (a pytest node id, and the value is the leak guard's own fake fixture)
        ),
    ),
    (
        "#153 name parts after the stem, so `secret_key_base=` is a credential",
        "tools/check_no_private_leak.py",
        r'    r"(?:_[A-Z0-9]+)*\b"',
        r'    r"\b"',
        (
            "tests/test_leak_guard.py::test_compound_credential_names_are_caught[secret_key_base=0123456789abcdef0123456789abcdef01234567]",  # leak-guard: allow (a pytest node id, and the value is the leak guard's own fake fixture)
        ),
    ),
    (
        "#153 a 12-character value with a digit is credential-shaped",
        "tools/check_no_private_leak.py",
        r"[A-Za-z0-9_\-]{12,}|",
        r"[A-Za-z0-9_\-]{20,}|",
        (
            "tests/test_leak_guard.py::test_compound_credential_names_are_caught[API_KEY=abc123def456ghi7]",  # leak-guard: allow (a pytest node id, and the value is the leak guard's own fake fixture)
        ),
    ),
    (
        "#154 src/, tests/ and examples/ are not exempt from the personal shapes",
        "tools/check_no_private_leak.py",
        "    patterns = _UNIVERSAL_PATTERNS + _PERSONAL_PATTERNS",
        "    patterns = _UNIVERSAL_PATTERNS",
        (
            "tests/test_leak_guard.py::test_a_real_person_in_the_package_is_caught",
        ),
    ),
    (
        "#139 later epsilon groups condition on the columns earlier ones produced",
        "src/synthweave/mode.py",
        "                    predictors=list(conditioned),\n",
        "",
        (
            "tests/test_mode.py::test_two_attributes_at_different_epsilons_keep_the_donors_joint_structure",
        ),
    ),
    (
        "#144 a real column is scoped to the tables that declared it",
        "src/synthweave/mode.py",
        "                    tables=[table_name],\n",
        "",
        (
            "tests/test_mode.py::test_a_real_data_column_stays_out_of_a_table_that_did_not_declare_it",
        ),
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
        (
            "tests/test_mode.py::test_a_real_data_attribute_no_table_carries_raises_naming_it",
        ),
    ),
    (
        "#144 a real attribute bound under another name raises, not an all-null column",
        "src/synthweave/mode.py",
        "        if isinstance(rule, _RealDataColumn) and rule.name != bound:",
        "        if False:",
        (
            "tests/test_mode.py::test_binding_a_real_data_attribute_under_another_name_raises_naming_both",
        ),
    ),
    (
        "#145 each epsilon group records under its own provenance/report key",
        "src/synthweave/stages/synthesize.py",
        '        prefix = f"{table.name}.synth" + (f".{self.label}" if self.label else "")\n'
        '        stage = "synthesize" + (f".{self.label}" if self.label else "")',
        '        prefix = f"{table.name}.synth"\n'
        '        stage = "synthesize"',
        (
            "tests/test_mode.py::test_a_two_epsilon_run_records_both_generalization_levels",
            "tests/test_mode.py::test_a_two_epsilon_run_reports_both_fits_without_colliding",
        ),
    ),
    (
        "#145 an epsilon-derived leaf size is user-provided, not a library default",
        "src/synthweave/mode.py",
        """        "min_samples_leaf": user(
            max(5, round(100 / capped)), f"derived from epsilon={epsilon!r}"
        ),""",
        '        "min_samples_leaf": max(5, round(100 / capped)),',
        (
            "tests/test_mode.py::test_an_epsilon_derived_leaf_size_is_not_reported_as_a_library_default",
        ),
    ),
    (
        # The exact mutation #143 reported as invisible: epsilon becomes a total
        # no-op. It used to fail four tests, every one of them asserting a knob
        # value on the synthesizer object rather than anything in the output.
        "#143 epsilon reaches the synthesized data, not only the knob values",
        "src/synthweave/mode.py",
        "    capped = min(max(epsilon, 0.01), 5.0)",
        "    capped = 5.0",
        (
            "tests/test_mode.py::test_epsilon_changes_the_synthesized_data_not_only_the_knob_values",
        ),
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
        (
            "tests/test_release_notes.py::test_the_range_ignores_a_non_release_tag_on_the_release_commit",
        ),
    ),
    # #150: both checkers refuse to certify a clause that had no chunk
    # boundary to inspect. Mutated at the call sites rather than inside
    # `_require_a_chunk_boundary`, because one mutation of the shared helper
    # would disable both refusals at once and could then be caught by either
    # test. Feeding a hardcoded 2 leaves the helper intact and switches off
    # exactly one checker's refusal, so each entry is pinned by the test for
    # that checker and nothing else.
    (
        "#150 check_generator refuses a split_chunk_size that produced no boundary",
        "src/synthweave/conformance.py",
        "        sum(1 for chunk in split if len(chunk)),",
        "        2,",
        (
            "tests/test_generator_conformance.py::test_a_split_size_that_does_not_split_is_refused",
        ),
    ),
    (
        "#150 check_synthesizer refuses a split_chunk_size that produced no boundary",
        "src/synthweave/conformance.py",
        "    _require_a_chunk_boundary(\n"
        '        split_chunks, split_chunk_size, "clause 5 (chunk invariance)"\n'
        "    )",
        "    _require_a_chunk_boundary(\n"
        '        2, split_chunk_size, "clause 5 (chunk invariance)"\n'
        "    )",
        (
            "tests/test_synthesizer_conformance.py::test_a_split_size_that_does_not_split_is_refused",
        ),
    ),
    # The exact line #125 reported: `write_empty` rebuilding its own stand-in
    # frame instead of writing the typed one the pipeline handed it. Invisible
    # on CSV, baked into the file on Parquet, where every column comes back as
    # the `null` type on a run that happened to cover no entity.
    (
        "#125 an empty table's Parquet file keeps the schema its rules declare",
        "src/synthweave/io.py",
        "        empty = self.empty\n",
        "        empty = pd.DataFrame(\n"
        "            {name: pd.Series(dtype=object) for name in self.empty.columns}\n"
        "        )\n",
        (
            "tests/test_io.py::test_a_parquet_file_for_a_table_with_no_rows_keeps_its_declared_schema",
        ),
    ),
    # #137's occurrence half: the grain column loses its stand-in rule and goes
    # back to `object` on a zero-row run against `int64` populated.
    (
        "#137 an empty event table keeps its grain column's declared dtype",
        "src/synthweave/pipeline.py",
        "    return _GrainColumn(np.arange(0).dtype) if isinstance(grain, PerEvent) else None",
        "    return None",
        (
            "tests/test_pipeline.py::test_an_empty_event_table_keeps_the_occurrence_dtype_of_a_populated_one",
        ),
    ),
    # #101's shape check passes a chunk-derived rate that was broadcast back to
    # one value per row, so without the behavioural split every row silently
    # gets the mean of whichever rows shared its chunk and the output depends
    # on chunk_size.
    (
        "#126 a chunk-derived noise rate is refused, not just a misshapen one",
        "src/synthweave/stages/noise.py",
        "    _check_row_wise(fn, chunk, rates, path)\n",
        "",
        (
            "tests/test_pipeline.py::test_a_chunk_derived_rate_of_the_right_length_is_refused",
        ),
    ),
    # The two directions this exemption can fail in, pinned separately.
    # Reverting the first puts the email shape back as PR #164 merged it, where
    # `t@example.test` is a finding and every future example address pays an
    # annotation. A guard people annotate reflexively is a guard nobody reads.
    (
        "#167 an address at an RFC 2606 reserved domain is not a finding",
        "tools/check_no_private_leak.py",
        r'    r"(?!" + _RESERVED_DOMAIN + r"(?![A-Za-z0-9-]|\.[A-Za-z0-9]))"' + "\n",
        "",
        ("tests/test_leak_guard.py::test_reserved_example_domains_are_not_addresses",),
    ),
    # Reverting this drops the tail anchor on the reserved-domain exemption,
    # so any domain merely *containing* a reserved label passes as unreal.
    # `e@example.com.co` is then a real mailbox the guard waves through while  # leak-guard: allow (an invented address, named because it is the case this entry pins)
    # printing `Passed`, which is the failure the guard exists to prevent.
    (
        "#167 the reserved-domain exemption binds to the end of the domain",
        "tools/check_no_private_leak.py",
        r'    r"(?!" + _RESERVED_DOMAIN + r"(?![A-Za-z0-9-]|\.[A-Za-z0-9]))"',
        r'    r"(?!" + _RESERVED_DOMAIN + r")"',
        ("tests/test_leak_guard.py::test_a_resolvable_address_is_still_caught_in_any_directory",),
    ),
    # The other side of that anchor. Both of these leave the exemption working
    # everywhere except where example addresses actually get written: the end
    # of an English sentence, and a capitalised spelling. The guard then still
    # reads as fixed while a doc author reaches for the escape hatch, which is
    # the habit #167 exists to end rather than relocate.
    (
        "#167 a full stop after a reserved domain is punctuation, not a label",
        "tools/check_no_private_leak.py",
        r'r"(?![A-Za-z0-9-]|\.[A-Za-z0-9]))"',
        r'r"(?![A-Za-z0-9.-]))"',
        ("tests/test_leak_guard.py::test_a_reserved_domain_is_reserved_in_prose_and_in_any_case",),
    ),
    (
        "#167 a reserved domain is reserved in any case, as domains are",
        "tools/check_no_private_leak.py",
        r'    r"(?i)\b[A-Za-z0-9._%+-]+@"',
        r'    r"\b[A-Za-z0-9._%+-]+@"',
        ("tests/test_leak_guard.py::test_a_reserved_domain_is_reserved_in_prose_and_in_any_case",),
    ),
]


def catchers(entry: tuple) -> tuple[str, ...]:
    """The test node ids an entry pins itself to, or `()` if it pins none.

    Entries are four-element tuples by default and five-element ones when
    pinned, so that adding the mechanism did not require touching a hundred
    existing entries at once.
    """
    return tuple(entry[4]) if len(entry) > 4 else ()


def run_suite(
    cwd: pathlib.Path,
    targets: tuple[str, ...] = ("tests/",),
    extra: tuple[str, ...] = (),
    stop_early: bool = True,
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
            [sys.executable, "-m", "pytest", *targets, "-q", "--no-header",
             *(("-x",) if stop_early else ()),
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
    # Not `returncode != 0`. pytest exits 4 for a node id that does not
    # resolve and 2/3 when it broke before running anything, and reading
    # either as "a test went red" is how a pin that points at nothing would
    # report as coverage.
    outcome = _EXIT_OUTCOME.get(proc.returncode, BROKE)
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


_FAILED_LINE = re.compile(r"^(?:FAILED|ERROR) (\S+::\S+)")


def failing_tests(output: str) -> list[str]:
    """The node ids in a run's short summary, sorted.

    Used by `--audit` to show *which* tests a revert broke, which is the
    evidence an entry needs before its catchers can be pinned honestly.
    """
    found = set()
    for line in output.splitlines():
        match = _FAILED_LINE.match(line.strip())
        if match:
            found.add(match.group(1).split(" - ")[0])
    return sorted(found)


def confirm_catchers(
    sandbox: pathlib.Path, named: tuple[str, ...]
) -> tuple[str, str]:
    """Re-run only the tests an entry pinned itself to. Returns (verdict, detail).

    The full-suite run already went red by the time this is called; the only
    question left is whether the red had anything to do with the property the
    entry names. Three outcomes matter and each maps to a different verdict:

    - a pinned test is red     -> CAUGHT, the red is the one that was claimed
    - every pinned test passes -> INCIDENTAL, the suite broke somewhere else
    - a pinned test is missing -> STALE, the pin verified nothing

    The last one is why this cannot be a plain `returncode != 0`. pytest
    answers a renamed or deleted node id with exit 4, which the old mapping
    read as a failing test, so a rename would have turned a broken pin into a
    silent pass. `run_suite` now reports that as NOT_COLLECTED instead.
    """
    outcome, msg, _ = run_suite(sandbox, named, ("-p", "no:cacheprovider"))
    listed = ", ".join(named)
    if outcome == NOT_COLLECTED:
        return STALE, f"pinned catcher(s) {listed} no longer resolve to a test ({msg})"
    if outcome == PASSED and " passed" not in msg:
        # An all-skipped selection also exits 0. Nothing ran, so this says
        # nothing either way -- and it is not hypothetical: pinned tests do
        # self-skip, `test_own_makes_a_view_safe_to_write_to` under pandas 3
        # among them.
        return INCONCLUSIVE, f"every pinned catcher was skipped, not run ({msg})"
    if outcome == PASSED:
        return INCIDENTAL, (
            f"the suite went red, but the pinned catcher(s) {listed} all passed, "
            "so the red says nothing about the property this entry names"
        )
    if outcome != FAILED:
        return INCONCLUSIVE, f"the catcher run {outcome} ({msg})"
    return CAUGHT, msg


def check(entry: tuple, sandboxes: "queue.Queue[pathlib.Path]") -> tuple[str, str, str]:
    """Verify one entry. Returns (verdict, name, detail); verdict is the label."""
    name, relpath, original, reverted = entry[:4]
    named = catchers(entry)
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
            if outcome == FAILED and named:
                # Second, much smaller run: the pinned tests only. Paid solely
                # by pinned entries that already went red, so the cost is a
                # few seconds per entry rather than a second full suite.
                verdict, msg = confirm_catchers(sandbox, named)
                return verdict, name, msg
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


def audit(entry: tuple, sandboxes: "queue.Queue[pathlib.Path]") -> tuple[str, list[str]]:
    """Name every test a revert breaks, so an incidental red can be *seen*.

    `check()` stops at the first failure, which is fast but tells you only
    that something went red. Reviewing whether a red is the right red needs
    the whole failure set, so this runs without `-x` and parses the short
    summary. It is a diagnostic, not a gate: it is what you run once over the
    list to decide which entries need pinning.
    """
    name, relpath, original, reverted = entry[:4]
    source = ROOT / relpath
    if not source.exists() or original not in (text := source.read_text()):
        return name, []
    sandbox = sandboxes.get()
    try:
        path = sandbox / relpath
        path.write_text(text.replace(original, reverted, 1))
        try:
            _, _, output = run_suite(sandbox, extra=("-rf",), stop_early=False)
        finally:
            path.write_text(text)
    finally:
        sandboxes.put(sandbox)
    return name, failing_tests(output)


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
    spec, auditing = None, False
    argv = list(argv)
    if "--audit" in argv:
        auditing = True
        argv.remove("--audit")
    if argv:
        if len(argv) != 2 or argv[0] != "--shard":
            raise SystemExit(f"usage: {sys.argv[0]} [--shard i/N] [--audit]")
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

        if auditing:
            print("audit: every test each revert breaks (diagnostic, not a gate)\n")
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                for name, broken in pool.map(lambda e: audit(e, sandboxes), entries):
                    print(f"  {name}")
                    for nodeid in broken or ["(nothing broke, or the entry is stale)"]:
                        print(f"      {nodeid}")
            return 0

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
