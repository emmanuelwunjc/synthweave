# Issues log

Problems found while building, recorded when encountered. The first five were
logged retroactively, after the practice was adopted; everything later was
logged before its fix.

Status: `open`, `fixed`, `not a defect`.

## How these are verified

A fix is only considered closed once a test goes red when the fix is reverted.
`tools/mutation_check.py` automates that: it undoes each logged fix one at a
time, runs the suite, and reports whether anything noticed.

That check earned its keep. I5 and I6 were both marked fixed while having no
regression coverage at all, so reverting either left the suite fully green. It
also corrected the stated root cause of I6 and most of I8. Run it after
touching any of the code these entries cover.


A skipped mutation used to print `SKIP` and still let the run finish with
"every logged fix is covered", which is the same false confidence the tool
exists to remove. A mutation whose snippet no longer matches the code now
reports `STALE` and fails the run, because it verified nothing. Found when a
snippet went stale during the I14 fix.
---

## I1. Synthesizer applied to every table regardless of its columns
**Status:** fixed
**Found:** running the first test suite.

A pipeline has one synthesizer but many tables, and a synthesized column
rarely exists in all of them. `CARTSynthesizer(["wage"])` was applied to the
roster table too, which has no wage column, and raised.

**Fix:** added a `tables` argument scoping the stage. Tables not named pass
through untouched. Leaving it unset still means every table, which then
requires every table to carry the columns, so a typo still fails loudly
rather than silently skipping.

---

## I2. Rate assertions were too tight for the fixture population
**Status:** fixed
**Found:** `test_realized_noise_rate_is_reported` failed at 0.1475 against a
0.2 target.

Not a library bug. At 400 entities a 20% rate has a 3-sigma spread of roughly
plus or minus 6 points, so the assertion was a coin flip. Confirmed the hash
was not biased before touching anything: over 200,000 keys the mean was
0.50025 and the deciles were flat to within 0.5%.

**Fix:** added a 20,000-entity fixture for tests that assert on realized
rates, and tightened the bounds rather than loosening them. Sizing the sample
to the assertion is the fix; widening the tolerance would have hidden a real
regression later.

---

## I3. Chunk size changed the output when the synthesizer was active
**Status:** fixed
**Severity:** high. This one invalidated the core scaling claim.
**Found:** `test_chunk_size_cannot_change_output_with_every_stage_active`.

The fit buffer accumulated whole chunks until it reached the cap, so a chunk
size of 113 fitted on 339 rows while a chunk size of 100,000 fitted on all
1,200. Different training data, different tree, different output. Chunk size
is supposed to be a memory knob with no effect on results, and it was not.

**Fix:** truncate the fit buffer to exactly the cap. The stream is
deterministically ordered, so the first N rows are identical however they
were chunked.

**Note:** every other stage was already chunk invariant, because values derive
from (seed, key, salt) rather than from position or RNG state. Fitting was the
only stage that looked at more than one row at a time, and so the only one
that could break the property.

---

## I4. Synthesizer ignored its predictors for the first column
**Status:** fixed
**Severity:** high. Silently destroyed the relationship it was asked to model.
**Found:** `test_empirical_structure_is_learned_from_supplied_data`. Output had
HS earners at 86k and College at 25k, the relationship inverted and scrambled.

The first column in the visit sequence was always drawn from its unconditional
marginal. synthpop does this because nothing precedes the first variable, but
here the predictors do precede it. Conditioning was skipped, so `education`
never influenced `wage`.

**Fix:** fit a tree for every column that has something to condition on,
including the first when predictors are supplied. Fall back to the marginal
only when there is genuinely nothing to condition on.

---

## I5. Identifier columns could not be noised
**Status:** fixed
**Found:** running `examples/linked_admin_records.py`, which asked for
missingness on the SSN column.

Stage order was generate, synthesize, noise, link. Identifier columns are
created by the linker, so they did not exist when noise ran. The linker's own
docstring claimed a user could dirty an identifier by naming it in the noise
config, which was impossible under that ordering.

The stated requirement is that identifiers stay clean unless explicitly
named, not that noise cannot reach them. Linking before noise satisfies the
requirement and enables the documented capability.

**Fix:** order is now generate, synthesize, link, noise. Synthesis still
precedes noise so no model fits on corrupted values. Identifiers still come
out clean unless named, because the noiser only touches columns it was given.
Spec updated to match.

---

## I6. Chunk ownership was an unstated contract
**Status:** fixed
**Found:** two warnings in the test run.
**Diagnosis corrected** after mutation testing, see below.

Stages assign columns onto the chunk they receive. When that chunk is a slice
of a larger frame, pandas warns that the write may not propagate.

The first diagnosis was that library stages were mutating views. That was
wrong. Neutering `own()` and running the suite under
`-W error::pandas.errors.SettingWithCopyWarning` showed only **one** failure,
and it was a test's own custom stage, not library code. Built-in stages always
construct the frames they write to, so the hazard never reaches them.

The real problem was that a **third-party** stage has no such guarantee and
there was nothing telling its author to copy. The bug was a missing contract,
not a missing copy.

**Fix:** `own()` in `stages/base.py`, documented as part of the plugin
contract and shown in the README extension example. Built-in stages call it
too, which costs nothing and keeps one rule rather than two.

**Regression test:** `test_own_makes_a_view_safe_to_write_to` exercises `own`
directly, since it is public API for extension authors even though it is not
part of the pipeline seam.

---

## I7. Wage means shifted after synthesis in the example
**Status:** not a defect, verified
**Found:** comparing declared normals against post-pipeline means in
`examples/linked_admin_records.py`.

Declared means were 41,000 / 33,000 / 27,000 by education. After CART
synthesis and 8% missingness the observed means were 41,055 / 32,836 / 26,934.

Originally filed as "wontfix, probably inherent to donor resampling", which was
an assumption. Checked properly on 30,000 rows, comparing pre-synthesis and
post-synthesis distributions within each group:

| group | pre mean | post mean | shift | pre sd | post sd | KS p |
|---|---|---|---|---|---|---|
| HS | 38,007 | 38,092 | +0.22% | 8,979 | 8,982 | 0.896 |
| College | 63,780 | 63,951 | +0.27% | 17,867 | 17,968 | 0.648 |

Means move by roughly a quarter of a percent, standard deviations are
essentially unchanged, and two-sample KS tests come nowhere near rejecting.
The synthesized distribution is statistically indistinguishable from its
source. This is a fidelity result worth keeping, not a defect.

---

## I8. Throughput: no single dominant bottleneck
**Status:** fixed (partially), diagnosis corrected
**Found:** benchmarking a 3.2 million row streaming run.

Measured: 3,200,000 rows streamed to disk, peak RSS 0.43 GB. Memory stays flat
as rows grow, so the streaming design holds and row count is bounded by time
rather than RAM. That part of the original finding stands.

**The original diagnosis was wrong and was stated without profiling.** It
claimed throughput was limited by two per-row Python loops, in
`_hash.derive_id` and `RuleGenerator._entity_keys`. A cProfile run over
960,000 rows showed:

- `derive_id` was the single largest `tottime` entry, so that half was right.
- `_entity_keys` did not appear in the top 14 at all. It was never a
  bottleneck; naming it was a guess presented as a fact.

**Fix applied:** `derive_id` now formats identifiers with `np.char.zfill` and
`np.char.add` instead of a Python f-string loop. Verified byte-identical
output across prefix and digit variants before and after, so no generated
value can change.

**Honest result:** 2.3x faster in isolation, but only **1.09x end-to-end**,
measured by an interleaved A/B of both implementations in one session (loop
median 8.30s, vectorized median 7.61s over 1.6M rows). Identifier formatting
is only about a tenth of the run.

There is no single dominant bottleneck left. The remaining cost is spread
across pandas' vectorized array hashing, dataframe construction, and I/O.
Further speedup means attacking several things at once, which is not worth
doing until a real workload demands it.

**Method note:** the first end-to-end comparison appeared to show a slowdown
(15.0s before, 22.7s after). That was cross-session machine load, not a
regression. Within a session, repeat runs vary by only 1.1x. Perf claims here
should come from interleaved A/B in a single session; sequential
before-and-after numbers across sessions are not trustworthy.

---

---

## I9. Synthesis dropped a column's dtype
**Status:** fixed
**Found:** first multi-column synthesis test, during the v0.1 bug hunt.

Donor sampling builds each column in an object array, because a leaf's pool is
selected with a boolean mask and there is no dtype-preserving way to do that
per leaf. The object array was then assigned straight back to the frame, so an
`int64` column came out of the synthesizer as `object`. Values were correct.
The type was not.

That is not cosmetic. An object column is invisible to `select_dtypes`, costs
roughly four times the memory (measured: 64,132 bytes against 16,132 for 2,000
float values), and leaves Parquet with nothing to infer from.

**Fix:** `_FittedCART` records each synthesized column's dtype at fit time and
casts back to it in `apply`. The dtype comes from the model, never from the
chunk, which is what keeps it chunk invariant. Reading it off the chunk would
have made a column's type depend on which values happened to land in that
chunk.

**Covered by:** `test_a_synthesized_column_keeps_the_dtype_it_had` and
`test_synthesis_does_not_hide_a_numeric_column_behind_object_dtype`.
Registered in `tools/mutation_check.py` as I9.

---

## I10. Conditional columns are object dtype from generation
**Status:** fixed
**Tracked:** [#2](https://github.com/emmanuelwunjc/synthweave/issues/2)
**Found:** while fixing I9, which only restores what generation produced.

`Conditional.draw` fills `np.empty(len(keys), dtype=object)` because its
branches may return different types. The generator assigns that array straight
into the chunk, so every conditionally-drawn column is `object` even when every
value in it is a float. Declared conditional structure is the flagship
no-real-data feature, so this is the common case, not an edge one: in the bug
hunt fixture the entire table came back with zero numeric columns.

Cost is the same as I9: four times the memory, invisible to numeric selection,
and nothing for Parquet to infer from.

**Why it is not just fixed.** The obvious fix is to narrow the array after
drawing. Narrowing per chunk is not safe, and this was measured rather than
assumed. For a `Conditional` with an `Integer` branch and a `Normal` branch,
`pd.api.types.infer_dtype` over the drawn column gives:

| chunk_size | kinds inferred across chunks |
|---|---|
| 7 | `floating`, `integer`, `mixed-integer-float` |
| 500 | `mixed-integer-float` |

So a column's type would depend on chunk size, which breaks invariant 2
directly and would break chunked Parquet writes, where the writer's schema is
fixed by the first chunk.

A safe fix has to derive the dtype from config rather than from data, which
means a rule declaring the type it produces. That is an addition to the public
`Rule` protocol, and the bug hunt spec says to raise an API change rather than
make one.

**Options:**
1. Optional `dtype()` on the `Rule` protocol, defaulting to object. `Conditional`
   resolves it from its branches. Chunk invariant by construction, and custom
   rules keep working unchanged.
2. Narrow to `float64` whenever a column is entirely numeric and non-null.
   Chunk invariant, since any all-numeric chunk gives the same answer. Costs
   integer columns their integer type.
3. Leave it. Correct today, just wasteful and awkward downstream.


**Fixed by:** an optional `dtype()` on the `Rule` protocol (option 1). Each rule
declares the type it produces and `Conditional` resolves its own from its
branches, so an integer branch beside a float branch gives float everywhere. The
declaration comes from config, never from the values, which is what makes it
chunk invariant. Rules that do not implement `dtype()` declare nothing and their
columns are left exactly as drawn, so custom rules written against v0.1 keep
working. This also fixed I12.
---

## I11. An identifier tag silently overwrites a column of the same name
**Status:** fixed
**Tracked:** [#3](https://github.com/emmanuelwunjc/synthweave/issues/3)
**Found:** bug hunt, identifier collision probe. Suspected in the handoff, confirmed here.

`validate_schema` checks carried attributes against table columns, and grain
columns against both, but never checks identifier tags against either. The
linker runs last and assigns `chunk[spec.tag]`, so the identifier wins and the
user's data is gone. No error, no warning.

Both variants reproduce:

| collision | result |
|---|---|
| identifier tag `wage` vs declared column `wage` | the `Normal` draws are replaced by `W866357799`-style strings |
| identifier tag `education` vs carried attribute `education` | the carried attribute is replaced the same way |

Worst class of failure in this codebase: the run succeeds, the file looks
right, and a column of real modelled values has been destroyed.

**Fix direction:** extend `_validate_table` to check every tag in
`table.identifiers` against `table.columns`, `table.carry`, and the grain's
produced column, alongside the checks already there. Cheap and local.

**Covered by:** `test_an_identifier_tag_cannot_overwrite_a_table_column` and
`test_an_identifier_tag_cannot_overwrite_a_carried_attribute`, both strict
xfail until fixed.


**Fixed by:** `_validate_table` now checks every identifier tag against the
table's columns and carried attributes, alongside the checks already there. The
run stops before generating anything.
---

## I12. Chunked Parquet aborts on a column whose type varies between chunks
**Status:** fixed
**Tracked:** [#4](https://github.com/emmanuelwunjc/synthweave/issues/4)
**Found:** bug hunt, Parquet probe.

`ChunkWriter` opens a `ParquetWriter` with the first chunk's arrow schema, so
every later chunk must match it. A `Conditional` with an `Integer` branch and a
`Normal` branch produces a column that infers as `int64` in a chunk holding
only the integer branch and `double` in one holding a float. The run then dies
partway through:

    ValueError: Table schema does not match schema used to create file

Reproduced with a 97/3 split at `chunk_size=7`, 400 rows. The same config at
`chunk_size=400` writes cleanly, so this is a chunk-invariance violation
(invariant 2) as well as a crash, and `chunk_size` stops being only a memory
knob. A truncated `t.parquet` is left behind.

Related to I10: a rule-declared dtype would fix both, since the column type
would stop depending on which values landed in a chunk.

**Covered by:** `test_parquet_survives_a_column_whose_type_varies_between_chunks`,
strict xfail until fixed.


**Fixed by:** the I10 dtype declaration, plus a reconciliation step in
`ChunkWriter`. A column produced by a rule that declares its type is now
identical in every chunk, so the writer has nothing to disagree with.

A rule that declares nothing is the residual case, and it is real: `Sequential`
wraps an arbitrary function, so it can hand back whole numbers in one chunk and
fractions in the next. The writer now casts such a chunk onto the schema the
file was opened with. Widening is silent and lossless. Narrowing would change
the value, so the run stops with the column named and the fix stated, rather
than with pyarrow's schema dump. Found during code review of the first fix,
which claimed more than it delivered.

**Covered by:** `test_parquet_survives_a_column_whose_type_varies_between_chunks`,
`test_a_widening_type_shift_between_chunks_is_absorbed`, and
`test_an_undeclared_column_type_that_shifts_between_chunks_names_itself`.
---

## I13. A table that emits no rows produces nothing usable
**Status:** fixed
**Tracked:** [#5](https://github.com/emmanuelwunjc/synthweave/issues/5)
**Found:** bug hunt, empty output probe.

With coverage low enough that no entity is selected, three things go wrong at
once and none of them raises:

- the in-memory frame has 0 rows **and 0 columns**, so downstream code cannot
  read the schema it was promised
- the CSV file is 0 bytes; `pd.read_csv` fails with `EmptyDataError`
- the Parquet file is never created at all; `pd.read_parquet` fails with
  `FileNotFoundError`

The run reports success in every case.

**Fix direction:** the pipeline knows a table's columns from the schema before
generation, so an empty result should carry them. Writers should emit a header
or an empty file with a schema rather than nothing.

**Covered by:** `test_a_table_that_emits_no_rows_still_has_its_columns` and
`test_a_table_with_no_rows_still_writes_a_readable_file`, both strict xfail.


**Fixed by:** `Table.output_columns()`, which derives a table's columns from
config alone. An empty in-memory result carries them, and `ChunkWriter` writes a
header-only CSV or a zero-row Parquet file with a schema. Both read back as an
empty frame.
---

## I14. Identifier `digits` is unvalidated at both ends
**Status:** fixed
**Tracked:** [#6](https://github.com/emmanuelwunjc/synthweave/issues/6)
**Found:** bug hunt, digit extremes probe. Measured across digits 1, 9, 18, 19, 20, 25.

| digits | behaviour |
|---|---|
| 1 | 50 entities collapse onto 10 identifiers. Silent identity loss. |
| 9, 18 | correct |
| 19 | widths come out as a mix of 19 and 20, because 10**19 exceeds the uint64 modulus |
| 20, 25 | `OverflowError: Python int too large to convert to C long`, raised from numpy |

Three faces of one cause: `digits` is accepted without a bound and without any
relation to the population it has to number.

The collision end was then quantified against the default of 10 digits and the
9 digits the fixtures use. Measured collisions track the birthday bound almost
exactly, which confirms the hash is uniform and the risk is inherent to the
keyspace rather than a hashing defect:

| entities | collisions at 9 digits | birthday expectation |
|---|---|---|
| 10,000 | 0 | 0.1 |
| 100,000 | 1 | 5.0 |
| 400,000 | 76 | 80.0 |

So a 400,000 person run at 9 digits hands back 76 identifiers that refer to two
different people, silently. Nothing in the library says so.

**Fix direction:** reject `digits` above the uint64 width in `Identifier`, and
warn when `10**digits` is small relative to `entity.count`, using the birthday
bound rather than the raw keyspace size.

**Covered by:** `test_a_digit_count_too_small_for_the_population_is_rejected`,
`test_identifiers_all_have_the_requested_width`, and
`test_a_digit_count_past_the_hash_width_is_rejected`, all strict xfail.


**Fixed by:** `Identifier` rejects `digits` above 18, the widest the 64-bit hash
can honour. Validation additionally rejects a keyspace too narrow for the
population, judged on the birthday bound rather than the raw keyspace, and the
message names the digit count that would work. `Identifier.expected_collisions()`
exposes the same number.
---

## I15. `Prior(joints=...)` has never worked
**Status:** fixed
**Tracked:** [#7](https://github.com/emmanuelwunjc/synthweave/issues/7)
**Found:** bug hunt, joint prior probe.

`Prior.training_frame` builds `np.array(list(dist.keys()))` from the joint's
tuple keys, which gives a 2-D array of shape `(n, 2)`. `_hash.pick` compares
that against a 1-D weights array and raises:

    ValueError: weights length (4,) does not match values (4, 2)

Every call with `joints=` set fails this way, so the path cannot ever have
run. It is documented in the `Prior` docstring and exported from the package
top level.

**Fix direction:** pick over an index rather than over the tuples themselves,
then map the chosen index back to the pair. Keeps `_hash.pick` 1-D, which is
its contract.

**Covered by:** `test_a_joint_prior_shapes_the_relationship_it_declares`,
strict xfail until fixed.


**Fixed by:** picking over positions and mapping back to the pair, so
`_hash.pick` keeps its 1-D contract.
---

## Suspicions the hunt cleared

Recorded so the next session does not re-investigate them. Each now has a test.

| Suspicion | Verdict |
|---|---|
| Multi-column synthesis conditions on the wrong columns | Sound. The chain education to sector to wage survives, both links intact. |
| Multi-column synthesis is not chunk invariant | Sound at chunk sizes 13, 97, 100,000. |
| Multi-column synthesis invents values | Sound. Every value traced to a donor row. |
| Two entities share an identifier space | Sound. Identical tag, prefix, and digit count across two entities produced zero shared values in 500 each. |
| Entity order in the schema changes output | Sound. Reversing the entity list is byte-identical. |
| `PerEvent(low=0)` breaks on empty concatenation | Sound. 117 of 200 entities had rows, no error. |
| `Missing` before `Typo` stringifies `None` | Sound. Typo skips nulls. |
| `Typo` and `OCR` break on non-ASCII | Sound. Typo corrupts by character. OCR is a no-op on scripts with no confusion pairs, which is correct but worth documenting. |
| `Sequential` is untested and possibly order-dependent | Sound. Derives correctly and is chunk invariant. |

---

## I16. A repeated period silently duplicates panel keys
**Status:** fixed
**Tracked:** [#8](https://github.com/emmanuelwunjc/synthweave/issues/8)
**Found:** bug hunt, grain edge case probe.

`PerPeriod("person", periods=[2020, 2020, 2021])` is accepted. The result has
300 rows for 100 entities, of which 100 are duplicate `(tax_id, period)` pairs.
The grain's whole promise is one row per entity per period, so this hands back
a panel whose key is not a key, and any join on it fans out.

Unsorted periods are fine by contrast: rows are emitted in the declared order,
but every `(entity, period)` value matches, so it is presentation rather than a
data change. That is now asserted rather than assumed.

**Fix direction:** reject duplicate periods in `PerPeriod.__post_init__`,
alongside the checks already there. A repeated period is always a config slip.

**Covered by:** `test_a_repeated_period_is_rejected` (strict xfail) and
`test_period_order_does_not_change_the_panel` (passing).


**Fixed by:** `PerPeriod.__post_init__` rejects repeated periods, next to the
presence check already there.
---

## I17. The numeric heuristic crashes on lightly contaminated columns
**Status:** fixed
**Tracked:** [#9](https://github.com/emmanuelwunjc/synthweave/issues/9)
**Found:** bug hunt, `_is_numeric` boundary probe.

`_is_numeric` calls a column numeric when more than 90% of its values parse as
numbers. The column then goes to a `DecisionTreeRegressor` as raw values, so
any non-numeric value in it reaches sklearn unconverted:

    ValueError: could not convert string to float: 'refused'

Measured on an amount column with `refused` sentinels:

| share of sentinels | outcome |
|---|---|
| 5% | crash, opaque sklearn error |
| 11% | works, treated as categorical |

The threshold is backwards in effect: more contamination is what makes the fit
succeed. The failing case is also the realistic one, since survey amounts
usually carry a small number of refusals rather than a large number.

**Fix direction:** once a column is judged numeric, coerce it before fitting
and decide explicitly what happens to the values that do not parse, rather than
handing them to sklearn. Whichever way that goes, it needs to be a documented
choice.

**Covered by:** `test_a_numeric_column_with_a_few_sentinel_values_still_fits`,
strict xfail.


**Fixed by:** deleting the heuristic. A column is fitted as numbers when its
dtype says so, which it now can because rules declare their type (I10).
`CARTSynthesizer(numeric=[...])` is the explicit override for an object column
that really does hold numbers, and a value that will not parse raises an error
naming the column instead of surfacing from inside sklearn.
---

## I18. Schema carry="*" mutates the shared Table object
**Status:** open
**Tracked:** [#11](https://github.com/emmanuelwunjc/synthweave/issues/11)
**Found:** code-review pass, 2026-07-31, targeting the session's schema-shorthand additions.

`Schema.__post_init__` resolves `carry="*"` by writing directly into the
`Table` instance rather than a copy. Reusing the same `Table` object across
two `Schema`s means the second `Schema` silently keeps the first schema's
resolved carry set, since `table.carry` is no longer the string `"*"` by
the time the second `Schema` checks it.

**Fix direction:** resolve into a local copy rather than mutating the
shared `Table` instance.

**Covered by:** not yet — no test reuses a `Table` object across schemas.
---

## I19. carry="*" raises a bare KeyError instead of SchemaError
**Status:** open
**Tracked:** [#12](https://github.com/emmanuelwunjc/synthweave/issues/12)
**Found:** code-review pass, 2026-07-31, same sweep as I18.

`carry="*"`'s entity lookup in `Schema.__post_init__` isn't wrapped the way
`validation.py`'s identical lookup is, so a typo'd entity name fails
differently (bare `KeyError`, no table context) depending on whether
`carry="*"` is used.

**Fix direction:** wrap the lookup the same way `validation.py` does.

**Covered by:** not yet.
---

## I20. coerce_rule silently treats namedtuple as Choice
**Status:** open
**Tracked:** [#13](https://github.com/emmanuelwunjc/synthweave/issues/13)
**Found:** code-review pass, 2026-07-31, same sweep as I18.

`coerce_rule`'s `isinstance(value, (list, tuple))` check matches
`namedtuple` instances too, silently flattening a structured record into
an equal-weight `Choice` over its field values.

**Fix direction:** exclude `namedtuple` (e.g. `hasattr(value, "_fields")`)
from the tuple branch.

**Covered by:** not yet.
---

## I21. coerce_rule is asymmetric for numpy scalars
**Status:** open
**Tracked:** [#14](https://github.com/emmanuelwunjc/synthweave/issues/14)
**Found:** code-review pass, 2026-07-31, same sweep as I18.

`np.float64` passes the `isinstance(value, float)` check (subclasses
Python's `float`) and silently becomes `Constant`; `np.int64` doesn't
subclass `int` and fails with an error suggesting a fix (wrap in a `Rule`)
that isn't actually the right one (cast to a plain scalar).

**Fix direction:** handle numpy scalar types explicitly and consistently
either way, and fix the error message if the fail-loud path is kept.

**Covered by:** not yet.
---

## I22. CARTSynthesizer's dict-to-Prior coercion gives a confusing error for a malformed dict
**Status:** open
**Tracked:** [#15](https://github.com/emmanuelwunjc/synthweave/issues/15)
**Found:** code-review pass, 2026-07-31, same sweep as I18.

Any `Mapping` passed to `structure=` is wrapped as `Prior(marginals=...)`
unconditionally. A dict not actually shaped like marginals fails much
later inside `Prior.training_frame` with an unrelated error
(`'numpy.ndarray' object is not callable`), no indication the root cause
was the auto-coercion guessing wrong.

**Fix direction:** validate the dict's shape before wrapping, or fail
earlier and more specifically inside `Prior`.

**Covered by:** not yet.
---

## I23. SSAFirstName's on-disk cache filename ignores source
**Status:** open
**Tracked:** [#16](https://github.com/emmanuelwunjc/synthweave/issues/16)
**Found:** bug hunt, 2026-07-31, targeting the session's new connector modules.

The in-memory memo key includes `source`, but the on-disk cache filename
(`ssa_names.csv`) doesn't. Two different `source` files sharing a
`cache_dir` silently reuse whichever data cached first; the second
`source` is never actually read.

**Fix direction:** include something that distinguishes `source` in the
cache filename.

**Covered by:** not yet.
---

## I24. fetch_pums accepts a header-only ACS response as success
**Status:** open
**Tracked:** [#17](https://github.com/emmanuelwunjc/synthweave/issues/17)
**Found:** bug hunt, 2026-07-31, same sweep as I23.

`_to_frame` only raises on a fully empty payload; a header-only response
(zero data rows, a reachable case for a filter matching nothing) passes
the check and silently produces an empty `DataFrame`.

**Fix direction:** treat `len(payload) <= 1` as the same failure case.

**Covered by:** not yet.
---

## I25. SSAFirstName silently drops rows with a NaN birth_year
**Status:** open
**Tracked:** [#18](https://github.com/emmanuelwunjc/synthweave/issues/18)
**Found:** bug hunt, 2026-07-31, same sweep as I23.

The year-range validation is false for `NaN` on both bounds, so `NaN`
passes it; the per-year grouping mask (`years == year`) is also always
false for `NaN`, so the row is skipped entirely and left as an
uninitialized `None` rather than raising.

**Fix direction:** check for `NaN` explicitly alongside the range check.

**Covered by:** not yet.
---

## I26. geonames.py parses a TSV with CSV quote-handling enabled
**Status:** open
**Tracked:** [#19](https://github.com/emmanuelwunjc/synthweave/issues/19)
**Found:** bug hunt, 2026-07-31, same sweep as I23.

`csv.reader` on GeoNames' `US.txt` uses the default `QUOTE_MINIMAL`
dialect on a file that was never quoted; a literal `"` in a place/county
name would cause rows to merge and columns to misalign silently.

**Fix direction:** `quoting=csv.QUOTE_NONE`.

**Covered by:** not yet.
---

## I27. fetch_pums's numeric state codes bypass validation
**Status:** open
**Tracked:** [#20](https://github.com/emmanuelwunjc/synthweave/issues/20)
**Found:** bug hunt, 2026-07-31, same sweep as I23.

`_resolve_state`'s digit-string branch returns `state.zfill(2)`
unconditionally, with no check against `_STATE_FIPS.values()`, unlike the
name/abbreviation branch. A bogus numeric code only fails later as an
opaque Census API error.

**Fix direction:** validate the digit-string branch against
`_STATE_FIPS.values()` too.

**Covered by:** not yet.
---

## I28. faker_names.SSN skews the distribution by remapping area==666
**Status:** fixed
**Tracked:** [#21](https://github.com/emmanuelwunjc/synthweave/issues/21)
**Found:** bug hunt, 2026-07-31, same sweep as I23.

`area = np.where(area == 666, 667, area)` reassigns every `666` draw to a
fixed `667` instead of resampling, giving `667` roughly double the
selection probability of every other valid area code. Lower severity:
still deterministic, chunk-invariant, and validly formatted — a
distributional-fidelity issue, not a contract violation or a crash.

**Fix:** draw over the 898 valid areas directly (`_hash.integers(..., 1,
899)`) and shift values `>= 666` up by one, so the excluded value is skipped
rather than folded onto a neighbor. Uniform across all 898, and still
deterministic and chunk invariant since it is the same single hash draw.

**Covered by:** `test_ssn_area_is_not_skewed_by_the_666_exclusion` in
`tests/test_faker_names.py`. Registered in `tools/mutation_check.py` as I28.
---

## I29. Donor diagnostics silently go stale when a synthesizer is reused across runs
**Status:** fixed (partially — see below)
**Tracked:** [#22](https://github.com/emmanuelwunjc/synthweave/issues/22)
**Found:** targeted bug hunt on the same-day `fidelity_report`/
`donor_diagnostics` addition, 2026-07-31, by an independent reviewer agent
given the diff and asked to find reachable bugs, not style nits.

`CARTSynthesizer._fitted` is keyed by table name only, with nothing scoping
an entry to a particular `Pipeline` run. `fidelity_report`'s own docstring
claimed the `synthesizer=` argument was "the fitted `CARTSynthesizer` from
the pipeline run that produced `synth_df`" — a 1:1 binding that isn't real.
Reusing one `CARTSynthesizer` instance for a second `Pipeline` run against
the same table name silently overwrites the first run's `empty_donor_counts`
with the second's, with nothing distinguishing "never run" from "run, then
quietly replaced by an unrelated later run."

**Fix:** not a behavior change — "last fit wins" is a reasonable, simple
design for a synthesizer meant to be reusable config, not a bug to engineer
away with run-tracking machinery. The actual fix is honesty: both
`CARTSynthesizer.donor_diagnostics`'s and `fidelity_report`'s docstrings now
say plainly that diagnostics reflect the last completed run for a table,
not necessarily the run that produced whatever `synth_df` you're holding.

**Covered by:**
`test_donor_diagnostics_reflects_only_the_last_run_when_synthesizer_is_reused`
in `tests/test_fidelity.py`. Registered in `tools/mutation_check.py` as I29.

---

## I30. donor_diagnostics readable — and silently wrong — before a stream finishes
**Status:** fixed
**Tracked:** [#23](https://github.com/emmanuelwunjc/synthweave/issues/23)
**Found:** same bug hunt as I29.

`CARTSynthesizer.run()` is a generator: `self._fitted[table.name] = model`
was set right after fitting, before the `for chunk in ...: yield
model.apply(...)` loop that actually accumulates
`_FittedCART.empty_donor_counts` had processed a single chunk.
`Pipeline.run()`/`run_to()` always fully drain that generator before
returning, so they were never affected, but a caller using the public
`Pipeline.stream()` API and checking `donor_diagnostics()` (directly or via
`fidelity_report(synthesizer=...)`) partway through consuming the stream got
a real-looking dict with a silently incomplete count — exactly the "silent
and plausible" failure mode this project's bug-hunt spec singles out as the
one that matters most, since a crash is cheap and a wrong-but-plausible
number is not.

**Fix:** `CARTSynthesizer` now tracks per-table completion (`self._complete`),
set `False` when a model is fitted and `True` only after every chunk has
actually been applied. `donor_diagnostics` omits a table entirely rather
than returning a partial count for it.

**Covered by:** `test_donor_diagnostics_excludes_a_table_still_mid_stream` in
`tests/test_fidelity.py`. Registered in `tools/mutation_check.py` as I30.

---

## I31. structure="empirical"/"prior" raised a TypeError naming the mechanism
**Status:** fixed
**Found:** code review of PR #24, 2026-07-31, verifying the branch's own
claim that the `"structure"` registry bug was fixed.

Wiring `resolve()` into `_coerce_structure` made registry names resolve, but
`resolve()` builds a registered *class* by calling it with no arguments.
`Declared()` takes none, so `structure="declared"` worked. `Empirical`
requires `frame` and `Prior` requires `marginals`, so `structure="empirical"`
and `structure="prior"` still failed — now with `TypeError:
Empirical.__init__() missing 1 required positional argument: 'frame'`, which
names the internal mechanism and not the thing the caller should do instead.
Both `CARTSynthesizer`'s docstring and `docs/HANDOFF.md` claimed these names
worked. The existing regression test used a throwaway zero-arg stub and its
docstring said outright that it avoided depending on `"empirical"`, so the
suite stayed green over a claim it never checked.

**Fix:** `_resolve_structure_name` inspects the registered class's signature
first. A source with required constructor parameters raises
`StructureConfigError` (exported from the package root) naming them and the
configured-instance form to use instead. Configuration-free names, including
third-party registrations, resolve exactly as before. Docstring and
`HANDOFF.md` corrected to match.

**Covered by:**
`test_structure_by_name_needing_config_names_the_missing_argument` in
`tests/test_synthesis_and_plugins.py`. Registered in
`tools/mutation_check.py` as I31.

---

## I32. fetch_pums caches the raw response before validating it parses
**Status:** open
**Found:** code review of PR #24, 2026-07-31 (non-blocking).

`src/synthweave/connectors/acs_pums.py` writes the API response to the cache
file before `_to_frame` has confirmed it is the shape a PUMS response should
be. A "200 OK but semantically malformed" reply — e.g. the Census API
returning a valid-JSON error object rather than data rows — is therefore
cached permanently, and every later call reads the bad cache and fails the
same way until someone deletes `.synthweave_cache/` by hand. The failure is
loud rather than silent, which is why this is not blocking, but it is
self-perpetuating.

**Fix direction:** write the cache only after a successful parse, or treat a
cache entry that fails to parse as a miss and re-fetch.

**Covered by:** not yet.

---

## I33. _read_dotenv can pick up an unrelated ancestor .env
**Status:** open
**Found:** code review of PR #24, 2026-07-31 (non-blocking).

`_read_dotenv` in `src/synthweave/connectors/acs_pums.py` walks up from the
current working directory to the filesystem root and stops at the first
`.env` it finds. Run from a nested directory whose repo has no `.env`, it
can silently read one belonging to an unrelated ancestor directory (up to
and including the user's home directory). Low severity: the only key read is
`CENSUS_API_KEY`, and a wrong key fails loudly at the API rather than
corrupting output.

**Fix direction:** anchor the search at the repo/project root rather than
walking to `/`, or document the walk-up explicitly as intended behavior.

**Covered by:** not yet.

---

## I34. faker_names reads Faker's private provider attributes
**Status:** open
**Found:** code review of PR #24, 2026-07-31 (non-blocking).

`_name_pool` in `src/synthweave/connectors/faker_names.py` reaches into
`faker.providers.person.en_US.Provider`'s class attributes (`first_names`,
`last_names`, and friends) directly. These are not part of Faker's public
API. The module's own docstring already explains why Faker's call-and-advance
generators are bypassed (determinism), so the approach is deliberate, but the
dependency is on internals that can change shape across Faker versions
without a deprecation path. `pyproject.toml` pins only `Faker>=20`.

**Fix direction:** assert the expected attribute shape at import and fail
with a clear message when it changes, and/or add an upper bound to the
`Faker` pin.

**Covered by:** not yet.

---

## I35. mutation_check.py crashed on a file absent from the current branch
**Status:** fixed
**Found:** running the harness on a branch off `main`, 2026-08-01.

The run resolved each mutation's path and called `read_text()` on it
unconditionally. An entry naming a file that does not exist on the checked-out
branch therefore raised `FileNotFoundError` and killed the process, taking
every later entry's check down with it. Hit because the I28 entry names
`src/synthweave/connectors/faker_names.py`, which only exists on the unmerged
PII branch.

Worse than an ordinary crash: the harness exists to answer "is every fix
covered," and dying partway through means most entries were never checked at
all, while the output still looks like a run that happened.

**Fix:** a missing file verifies exactly as much as a missing snippet does,
so it now reports `STALE` and continues, the same as a snippet that no longer
matches. `STALE` already fails the run, so nothing is silently excused.

**Covered by:** not directly. The harness cannot mutate itself, so this is
verified by the run completing on a branch where four entries name files or
snippets that do not exist there.

---

## I36. PR #24 sat on GitHub without its own review fixes
**Status:** fixed
**Found:** branch and PR audit, 2026-08-01.

The two blocking defects found reviewing PR #24 (I28, I31) were fixed and
committed locally as `91dd085`, but never pushed. For some hours the open PR
therefore contained the original bugs while the local branch contained the
fixes, and merging the PR in that state would have merged the defects back in
under a title claiming they were fixed.

Not a code defect, but exactly the class of thing this log exists for: the
review, the fix, and the artifact everyone else sees had drifted apart with
nothing surfacing it.

**Fix:** pushed. Verified `91dd085` now appears in `gh pr view 24 --json
commits`. Worth a habit rather than a patch: after fixing review findings on a
branch that already has an open PR, confirm the PR itself shows the fix commit
before considering the review closed.

**Covered by:** not applicable (process, not code).

---

## Blocked while I15 is open

`Prior` precedence between an explicit marginal and a joint that spans the same
column (user story 31 in the bug hunt spec) cannot be tested through the public
API, because every joint call raises before precedence matters. Revisit once
I15 is fixed.
