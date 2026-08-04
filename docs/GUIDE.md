# The synthweave guide: glossary, syntax reference, and tutorial

*Looking for a different doc? [docs/MAP.md](MAP.md) is the index.*

This is the plain-language companion to the README. The README is a quick
pitch for someone who already knows the vocabulary. This document is for
anyone who does not, yet, and wants to understand every piece of syntax
before writing their own schema.

Four parts:

1. **Glossary** — what each word means, in plain English, before you see any code.
2. **Syntax reference** — every function and class you can write, what each argument does, and what type it expects.
3. **Tutorial** — building one config from nothing, step by step, adding one idea at a time.
4. **The front door**: `sw.Mode`, a shorter way in once the pieces make sense.

If a word confuses you while reading the tutorial, it is defined in the
glossary. If you already know what you want to build and just need to look up
one piece of syntax, skip to the reference. If you want the shortest path to
working output and are happy to come back for the details, skip to Part 4.

---

## Part 1: Glossary

### The shape of the data

**Entity** — a *kind of thing* that shows up in your data, most often a
person, but it could be a business, a household, a vehicle, anything that
gets recorded more than once. You describe one entity once (its attributes,
its identifiers), and every table that mentions that entity reuses the same
description. Written as `sw.Entity(...)`.

**Population** — how many of that entity exist. Set with `count=` on an
`Entity`. synthweave never actually builds a list of a million people in
memory; a population is just a number until rows are asked for.

**Attribute** — a piece of information that belongs to the *entity itself*,
not to any one table. A person's `education` level is an attribute: it is
true about the person, and it can show up unchanged in three different
tables. Written inside `Entity(attributes={...})`.

**Identifier** — an ID number or code that names one specific entity, like a
student ID or a tax ID. An entity can carry more than one identifier
(`student_id` and `tax_id` are independent; knowing one tells you nothing
about the other). Written as `sw.Identifier(...)`, or its shorthand, a bare
string.

**Table** — one output file or dataframe. A table records some or all of an
entity's population, at some *grain* (see below), with some *columns*.
Written as `sw.Table(...)`.

**Grain** — how many rows one entity produces in a given table, and what
makes each of those rows unique. Three kinds:
  - `PerEntity` — one row per entity. A roster or registry.
  - `PerPeriod` — one row per entity per time period (year, quarter). A wage
    file, a panel dataset.
  - `PerEvent` — a random number of rows per entity. A transaction log.

**Carry** — attribute names to copy from the entity onto every row of a
table. If `education` is a carried attribute, every row about that person in
that table shows the same `education` value, because it was never redrawn —
it was copied straight from the entity.

**Column** — a value generated fresh *at the row level*, inside one specific
table, as opposed to an attribute (which belongs to the entity and can be
carried into many tables). A wage amount is usually a column: it makes sense
only inside the wage table, once per row.

**Coverage** — the share of the entity population that shows up in a given
table at all. `coverage=0.7` means 70% of people appear in this table and
30% do not — the way a real administrative file never contains everyone.

**Schema** — the whole config: every entity plus every table plus a seed
number. One `sw.Schema(...)` object describes an entire dataset.

**Seed** — a number (or string) that fully determines the output. Same
schema, same seed, same data, every single time, on any machine. Change the
seed and you get a different (but equally valid) random draw. (One honest
caveat: the `synthesize` stage's model-fitting step uses scikit-learn's
decision trees, which don't take an explicit seed from synthweave. This
hasn't produced any observed non-determinism across repeated fresh-process
runs, but unlike literally everything else in synthweave, it isn't derived
from the seed by construction. Worth knowing if bit-for-bit reproducibility
matters to you as a hard guarantee.)

### Rules: how one value gets chosen

A **Rule** is the thing that decides what value goes into one attribute or
column. Every rule is deterministic: given the same row and the same seed, a
rule always produces the same value. The built-in rules:

- **Constant** — always the same value. `sw.Constant(0)`.
- **Choice** — pick one value from a fixed list, optionally with different
  odds for each. `sw.Choice(["HS", "College"], [0.6, 0.4])`.
- **Integer** — a whole number, evenly likely anywhere in a range.
  `sw.Integer(1960, 2005)`.
- **Uniform** — same idea as Integer, but decimals instead of whole numbers.
  `sw.Uniform(0.0, 1.0)`.
- **Normal** — a bell-curve distribution: a mean, a spread (standard
  deviation), and optional floor/ceiling. `sw.Normal(38_000, 9_000, low=0)`.
- **Conditional** — pick a *different rule* depending on another column's
  value. This is how you say "wage depends on education." `sw.Conditional("education", {"HS": sw.Normal(...), "College": sw.Normal(...)})`.
- **Sequential** — compute a value from other already-drawn columns using
  your own function. An escape hatch for anything the rules above can't
  express, like deriving a birth year from an age.

### Making the data messier and more connected

**Pipeline** — the thing that actually runs a schema and produces data. It
strings together up to four stages in a fixed order:

1. **Generate** — draws every attribute and column from its rule. Always
   runs.
2. **Synthesize** — optional. Refits some columns using a statistical model
   (see "structure source" below), instead of drawing them independently.
   Used when you want realistic *relationships between columns*, learned
   rather than declared by hand.
3. **Link** — optional. Attaches identifier columns (like `tax_id`) so the
   same person's ID matches across every table that carries it.
4. **Noise** — optional. Deliberately makes some values messy (typos, blanks,
   scanned-form errors) the way real data is messy.

**Structure source** — where the synthesizer's statistical model learns
*relationships between columns* from. Three choices:
  - **Declared** — nothing extra; the model learns from the relationships
    your `Conditional` rules already put in the generated data. Use this
    when you have no real data at all, only domain knowledge.
  - **Empirical** — the model learns from a real sample of data you already
    have (a `pandas.DataFrame`).
  - **Prior** — the model learns from published statistics (percentage
    breakdowns, i.e. "marginals") when you have no rows of real data, only
    summary numbers from a report or table.

**CARTSynthesizer** — the actual statistical model used by the "synthesize"
stage. CART stands for Classification And Regression Tree — you don't need
to understand the algorithm to use it, only that it is what turns a
structure source into synthesized column values that keep the relationships
found in that source.

**Noise op** — one specific kind of messiness applied to one column:
  - **Missing** — blank the value out entirely.
  - **Typo** — swap one character for a nearby key on a keyboard.
  - **OCR** — swap one character for one that looks similar on a scanned
    form (0 and O, 1 and I, and so on).

Each noise op takes a `rate`: the share of values it corrupts, e.g.
`sw.Typo(0.05)` corrupts about 5% of that column's values.

### Trust and honesty about the numbers

**Provenance** — a record of *where every config number came from*.
synthweave tags every number as one of three origins:
  - **user-provided** — you typed this number yourself. The default tag; you
    rarely write this by hand.
  - **modeled** — a library default, or a number you made up as a
    placeholder. Write it as `sw.modeled(0.08, "just a guess for now")`.
  - **cited** — a number taken from a real source. Write it as
    `sw.cited(0.82, "NCES Table 219.10, retrieved 2026-07-29")`.

**`result.unjustified()`** — after running a pipeline, this lists every
number that is tagged `modeled` (a guess, not sourced) so you can see at a
glance which parts of your config still need a real source behind them.

### Getting data in from the outside world

**Connector** — code that fetches real data from an external source, for
when you have neither your own real data nor published aggregate stats. The
one that exists today is `synthweave.connectors.acs_pums.fetch_pums`, which
pulls real demographic microdata rows from the US Census Bureau's American
Community Survey (ACS), Public Use Microdata Sample (PUMS). "PUMS" just means
individual, anonymized survey responses, as opposed to a table of totals.

**Shorthand** — a shorter way to write something that means exactly the same
as a longer, more explicit form. synthweave has several (see the syntax
reference below); every shorthand is optional, and the longer form always
still works.

---

## Part 2: Syntax reference

For every entry: the full explicit form first, then any shorthand.

### `sw.Entity(name, count, attributes={}, identifiers=[])`

| Argument | Type | Meaning |
|---|---|---|
| `name` | string | What this entity is called, e.g. `"person"`. Tables refer back to it by this name. |
| `count` | integer | How many of this entity exist. |
| `attributes` | `{name: Rule}` | What this entity knows about itself. Each value is a `Rule` (see above), or a shorthand: a plain list/tuple becomes `Choice`, a plain number/string/bool becomes `Constant`. |
| `identifiers` | list of `Identifier` | The ID types this entity carries. Shorthand: a plain string `"tax_id"` becomes `Identifier(tag="tax_id")`. |

```python
people = sw.Entity(
    "person",
    count=10_000,
    attributes={"education": ["HS", "College"], "birth_year": sw.Integer(1960, 2005)},
    identifiers=["student_id", "tax_id"],
)
```

### `sw.Identifier(tag, prefix="", digits=10)`

| Argument | Type | Meaning |
|---|---|---|
| `tag` | string | Names this identifier stream. Two identifiers with different tags can never be derived from each other. |
| `prefix` | string | Text stuck on the front of every value, e.g. `prefix="SID"` gives `SID000482913`. |
| `digits` | integer, 1-18 | How many digits the number part has. |

### `sw.Table(name, grain, columns={}, carry=[], identifiers=[], coverage=1.0)`

| Argument | Type | Meaning |
|---|---|---|
| `name` | string | Output table name. |
| `grain` | `PerEntity`/`PerPeriod`/`PerEvent` | How entities map to rows (see glossary). Shorthand: a plain string `"person"` becomes `PerEntity("person")`. |
| `columns` | `{name: Rule}` | Row-level values, same coercion rules as `Entity.attributes`. |
| `carry` | list of strings, or `"*"` | Which entity attributes to copy onto every row. `"*"` means every attribute the entity has. |
| `identifiers` | list of strings | Which of the entity's identifier tags this table carries (and therefore can be linked on). |
| `coverage` | number, 0 to 1 | Share of the population that appears in this table at all. |

```python
roster = sw.Table("roster", grain="person", carry="*", identifiers=["student_id"])

wages = sw.Table(
    "wages",
    grain=sw.PerPeriod("person", periods=range(2018, 2026), presence=0.8),
    carry=["education"],
    identifiers=["tax_id"],
    coverage=0.7,
    columns={"wage": sw.Conditional("education", {
        "HS": sw.Normal(38_000, 9_000, low=0),
        "College": sw.Normal(64_000, 18_000, low=0),
    })},
)
```

### `sw.PerEntity(entity)`

One row per covered entity. Only argument: the entity's name.

### `sw.PerPeriod(entity, periods, presence=1.0, period_column="period")`

| Argument | Type | Meaning |
|---|---|---|
| `entity` | string | Which entity this grain is for. |
| `periods` | list of anything (usually years) | The set of periods, one column value per period. |
| `presence` | number, 0 to 1 | Chance an entity shows up in any *given* period — below 1.0, some entities have gap periods, the way real panels do. |
| `period_column` | string | What to name the column that says which period a row is. |

### `sw.PerEvent(entity, low=1, high=5, occurrence_column="occurrence")`

A random number of rows per entity, between `low` and `high`. Good for
transaction-style records.

### `sw.Schema(entities, tables, seed=0)`

| Argument | Type | Meaning |
|---|---|---|
| `entities` | list of `Entity` | Every entity in the dataset. |
| `tables` | list of `Table` | Every table in the dataset. |
| `seed` | number or string | Determines every value in the output. Same seed always gives the same data. |

### `sw.Pipeline(schema, generator="rules", synthesizer=None, noiser=None, linker="deterministic", chunk_size=100_000)`

| Argument | Type | Meaning |
|---|---|---|
| `schema` | `Schema` | What to run. |
| `synthesizer` | `CARTSynthesizer` or `None` | Stage 2, skipped if `None`. |
| `noiser` | `Noise` or `None` | Stage 4, skipped if `None`. |
| `linker` | `"deterministic"` or `None` | Stage 3, on by default. |
| `chunk_size` | integer | Rows processed at a time. A memory setting only — it never changes the output. |

Two ways to get output:

- `.run()` — everything in memory, returns a `PipelineResult`. Good up to
  however much RAM you have.
- `.run_to(directory, format="parquet")` — streams straight to disk,
  never holding more than one chunk in memory. Good for output far larger
  than RAM.

### `sw.CARTSynthesizer(columns, tables=None, predictors=[], numeric=[], structure=None, fit_cap=None, max_depth=None, min_samples_leaf=5)`

| Argument | Type | Meaning |
|---|---|---|
| `columns` | list of strings | Which columns to synthesize, in order. Each is conditioned on the ones before it. |
| `tables` | list of strings, or `None` | Which tables this applies to. `None` means every table (and then every table must have these columns). |
| `predictors` | list of strings | Columns to condition on but never resynthesize themselves, e.g. a carried attribute. |
| `structure` | see below | Where to learn column relationships from. |
| `fit_cap` | integer | Maximum rows the model actually trains on, for speed on huge tables. |

`structure` accepts, from most to least explicit:

| Form | Meaning |
|---|---|
| `sw.Declared()` | Learn from the generated data's own declared structure (the default if `structure` is left out). |
| `sw.Empirical(df)` | Learn from a real `pandas.DataFrame` you supply. |
| `sw.Prior(marginals={...}, joints=None, rows=50_000)` | Learn from published percentage breakdowns, no real rows needed. |
| a bare `pandas.DataFrame` | Shorthand for `sw.Empirical(that_dataframe)`. |
| a bare `dict` | Shorthand for `sw.Prior(marginals=that_dict)`. |

`Prior`'s `marginals` argument is a dict of dicts: `{"education": {"HS": 0.6, "College": 0.4}}` means 60% HS, 40% College. Its optional
`joints` argument does the same thing for a *pair* of columns at once, when
you know how two columns relate to each other, not just each one alone.

**How a value actually gets chosen, once fitted.** For each row, the fitted
decision tree assigns it to a leaf, then the synthesized value is drawn from
the *real values that landed in that same leaf during fitting* — not a
resampled or smoothed distribution. This matters two ways:

- **Privacy:** when `structure=` is `Empirical` (real data), a synthesized
  value is a literal value copied from one of your real rows, chosen from
  whichever real rows shared a leaf with the row being synthesized. A narrow
  leaf (few training rows in it) means few real values it could have come
  from — `min_samples_leaf` (default `5`) is the main lever to keep that
  count from getting too small if the columns you condition on could be
  identifying.
- **Empty leaves cannot happen, and the counter proves it rather than
  measures it.** Earlier versions of this guide described a leaf with no
  donor rows as ordinary behaviour to tune around. It isn't reachable.
  `fit` builds each column's donor pools from the leaves the training data
  actually lands in, and scikit-learn creates a leaf only by partitioning
  training samples, so every leaf holds at least one row. The donor map
  therefore covers every leaf the tree can produce, and no input row can
  reach one that is missing, however unlike the training data it is.

  The fallback branch still exists, and `fidelity_report`'s
  `empty_donor_leaves` still reports it, but read it as an assertion about
  the library rather than a property of your data: it is zero by
  construction, and a non-zero value means the fit/apply symmetry has been
  broken and you have found a bug. A test pins this
  (`test_no_leaf_is_ever_donorless_however_hard_you_push`), so if it ever
  becomes reachable it fails there first.

  `min_samples_leaf` still matters, just not for this. It controls how many
  real values a leaf can draw from, which is a disclosure concern (above),
  not a correctness one.

### `sw.fidelity_report(synth_df, real_df, columns=[...], thresholds=None, synthesizer=None)`

Compares synthesized output against the real data it was fitted on, for the
`Empirical` structure source specifically. `Declared` and `Prior` have
nothing to check against — a schema's declared structure is whatever the
author wanted it to be — but `Empirical`'s whole premise is that the output
preserves relationships that actually existed in real input, and until this
existed there was no way to check that beyond trusting the CART fit
mechanism's own claims.

```python
report = sw.fidelity_report(
    synth_df, real_df,
    columns=["education", "sector", "wage"],
    thresholds={"ks_pvalue": 0.05, "category_share_delta": 0.1, "association_delta": 0.1},
    synthesizer=synth,   # the fitted CARTSynthesizer, for empty-donor-leaf counts
)
report.to_frame()           # one row per column: KS stat (numeric) or share delta (categorical)
report.associations         # one row per column pair: Pearson / Cramer's V / correlation ratio
report.empty_donor_leaves   # {column: count of rows that kept a pre-synthesis placeholder}
```

- **`columns`** is required and never auto-detected: real and synth frames
  commonly share columns (identifiers, carried keys) nobody meant to have a
  preserved statistical relationship.
- **`thresholds`** is `None` by default, meaning purely descriptive numbers —
  synthweave has no built-in opinion about what counts as "close enough" for
  data it knows nothing about. Pass a dict to turn on pass/fail verdicts;
  only the keys you supply gate a `passed` column, the rest stay `None`.
  Recognized keys: `ks_pvalue` (numeric columns pass when the KS p-value is
  at or above it), `category_share_delta` and `association_delta` (pass when
  the delta is at or below it).
- **Association metric** depends on the pair's types: Pearson correlation
  for two numeric columns, Cramer's V for two categorical columns, and the
  correlation ratio (eta) for a numeric-categorical pair — eta generalizes
  point-biserial correlation to any number of categories, not just two.
- **`synthesizer`** is the fitted `CARTSynthesizer` instance from the
  pipeline run that produced `synth_df`. Without it, `empty_donor_leaves` is
  empty rather than wrong: which leaves had no donors is state that lives
  inside the fit, not in the two output frames, so there's no way to recover
  it from `synth_df`/`real_df` alone after the fact.

### `sw.Noise({table: {column: [NoiseOp(...), ...]}})`

One `Noise` object configures every table and column that should get dirty.
Each column gets an ordered list of noise ops, applied in that order.

```python
sw.Noise({
    "roster": {"education": [sw.Typo(0.05)]},
    "wages": {"wage": [sw.Missing(0.08)]},
})
```

- `sw.Missing(rate)` — blank out `rate` share of values.

`rate` is either a flat probability, or a **function of the chunk returning
one probability per row**. The flat form can only express MCAR (missing
completely at random). The function form expresses *differential*
nonresponse, where the chance of a value going missing depends on the row it
belongs to:

```python
# 30% missing for HS, 5% for College.
sw.Missing(lambda f: 0.05 + 0.25 * (f["education"] == "HS"))
```

It is vectorized, the same contract `sw.Sequential` uses: handed the frame,
returns an array. The corruption decision still routes through
`unit(key, seed, salt) < rate`, so the function only chooses the threshold
and never draws — which is what keeps a per-row rate exactly as
deterministic and chunk invariant as a flat one. A rate outside `[0, 1]`
raises, naming the column and op.
- `sw.Typo(rate)` — one nearby-key typo in `rate` share of values.
- `sw.OCR(rate)` — one visually-confusable-character swap in `rate` share of values.

### `sw.modeled(value, note)`, `sw.cited(value, source)`, `sw.user(value, note=None)`

Wrap any number to tag where it came from (see "Provenance" in the
glossary). Most numbers don't need this — plain numbers default to
`user-provided` automatically.

### `result.summary()`, `result.unjustified()`, `result["table_name"]`

After `pipeline.run()`, the `PipelineResult` you get back:

- `result["table_name"]` — the dataframe for one table.
- `result.summary()` — a small table of row/column counts per output table.
- `result.unjustified()` — every config number still tagged `modeled`, i.e.
  everything worth double-checking before you trust the output.

### `synthweave.connectors.acs_pums.fetch_pums(variables, state, year=2022, survey="acs1", cache_dir=..., api_key=None)`

| Argument | Type | Meaning |
|---|---|---|
| `variables` | list of ACS variable codes | e.g. `["AGEP", "PINCP"]` for age and personal income. |
| `state` | string | A FIPS code, USPS abbreviation, or full name — `"36"`, `"NY"`, and `"New York"` all work. |
| `year` | integer | Survey year. |
| `survey` | `"acs1"` or `"acs5"` | 1-year (fresher, fewer geographies) or 5-year (more geographies, less current) survey. |
| `api_key` | string, or `None` | Overrides the `CENSUS_API_KEY` environment variable / `.env` file. Required one way or another; there is no free anonymous access anymore. |

Returns a `pandas.DataFrame` with one column per requested variable, ready to
hand to `sw.Empirical(that_dataframe)` or straight to `CARTSynthesizer(structure=that_dataframe)`.

---

## Part 3: Tutorial

Building up one dataset from nothing, adding one new idea per step. Every
step below has actually been run; the output shown is real.

### Step 1: the smallest possible dataset

```python
import synthweave as sw

people = sw.Entity("person", count=100, attributes={"education": ["HS", "College"]})
table = sw.Table("roster", grain="person")
result = sw.Pipeline(sw.Schema([people], [table])).run()
print(result["roster"].head(3))
```

```
Empty DataFrame
Columns: []
Index: [0, 1, 2]
```

This is the entire grammar of synthweave in five lines: describe a
population (`Entity`), describe one table recording that population
(`Table`), wrap them in a `Schema`, and run a `Pipeline`. No identifiers, no
seed given (it defaults to `0`), no synthesizer, no noise — and the output
shows exactly what you asked for: nothing. `people` has an attribute, but
nothing tells `roster` to carry it; the table has no identifiers and no
columns of its own either. A table only ever shows what you explicitly gave
it. Step 2 fixes that.

### Step 2: give people an ID, and copy attributes onto the table

```python
people = sw.Entity("person", count=100, attributes={"education": ["HS", "College"]},
                    identifiers=["student_id"])
table = sw.Table("roster", grain="person", carry="*", identifiers=["student_id"])
```

`identifiers=["student_id"]` on the entity says people *have* a student ID.
`identifiers=["student_id"]` on the table says this table *carries* it —
without that second line, the table would have no ID column at all, on
purpose (a table with no identifiers is unlinkable, which is sometimes
exactly what you want to simulate). `carry="*"` copies every attribute
(here, just `education`) onto every row.

### Step 3: give one attribute real-looking spread, and add a second table

```python
people = sw.Entity(
    "person", count=10_000,
    attributes={"education": sw.Choice(["HS", "College"], [0.6, 0.4])},
    identifiers=["student_id", "tax_id"],
)
roster = sw.Table("roster", grain="person", carry="*", identifiers=["student_id"])
wages = sw.Table(
    "wages",
    grain=sw.PerEntity("person"),
    carry=["education"],
    identifiers=["tax_id"],
    coverage=0.8,
    columns={"wage": sw.Conditional("education", {
        "HS": sw.Normal(38_000, 9_000, low=0),
        "College": sw.Normal(64_000, 18_000, low=0),
    })},
)
result = sw.Pipeline(sw.Schema([people], [roster, wages], seed=7)).run()
```

Now `education` isn't 50/50, it's 60/40 (`sw.Choice` with weights). `wages`
has its own column, `wage`, that *depends on* `education` — that's what
`Conditional` does. `coverage=0.8` means only 80% of people show up in the
wage table at all. Two identifier tags on the person (`student_id`,
`tax_id`) mean the two tables use *different, unrelated* ID numbers for the
same person — you cannot guess someone's tax ID from their student ID.

Same person, same `tax_id`, in both tables that carry it — always, with no
lookup table involved.

### Step 4: make it messy

```python
result = sw.Pipeline(
    sw.Schema([people], [roster, wages], seed=7),
    noiser=sw.Noise({"wages": {"wage": [sw.Missing(0.05)]}}),
).run()
```

Adding a `noiser=` deliberately blanks out 5% of the `wage` column, the way
real administrative data has gaps. Everything else about the config is
unchanged.

### Step 5: learn structure from real data instead of declaring it

Suppose instead of guessing at `sw.Normal(38_000, 9_000, ...)` by hand, you
have an actual sample of real wage data sitting in a CSV. Read it into a
`pandas.DataFrame`, and hand it to the synthesizer directly:

```python
import pandas as pd

real_sample = pd.read_csv("my_real_wages.csv")  # columns: education, wage

result = sw.Pipeline(
    sw.Schema([people], [wages], seed=7),
    synthesizer=sw.CARTSynthesizer(["wage"], tables=["wages"], predictors=["education"],
                                    structure=real_sample),
).run()
```

The `wage` column in `wages` no longer needs a hand-picked `Normal` — it
gets replaced (`structure=real_sample`) with values statistically learned
from your real sample, keeping whatever real relationship your data actually
has between education and wage.

### Step 6: no real data, only a published statistic

Say you have no rows at all, just a report that says "65% own their home,
35% rent." That's enough:

```python
survey = sw.Table("survey", grain="person", columns={"tenure": "unknown"})
result = sw.Pipeline(
    sw.Schema([people], [survey], seed=11),
    synthesizer=sw.CARTSynthesizer(
        ["tenure"], tables=["survey"],
        structure={"tenure": {"own": 0.65, "rent": 0.35}},
    ),
).run()
```

`columns={"tenure": "unknown"}` is a placeholder value (shorthand for
`sw.Constant("unknown")`) that the synthesizer will overwrite entirely. The
bare dict passed to `structure=` is shorthand for `sw.Prior(marginals=...)`.

### Step 7: not even a published statistic — borrow real government data

```python
from synthweave.connectors.acs_pums import fetch_pums

acs = fetch_pums(["AGEP", "PINCP"], state="NY")
acs = acs.rename(columns={"AGEP": "age", "PINCP": "income"})

table = sw.Table("residents", grain="person", columns={"age": 0, "income": 0.0})
result = sw.Pipeline(
    sw.Schema([people], [table], seed=3),
    synthesizer=sw.CARTSynthesizer(["age", "income"], tables=["residents"], structure=acs),
).run()
```

`fetch_pums` pulls real, individual (anonymized) survey responses from the
US Census Bureau. Rename the columns to match your schema's names, and hand
the result to `structure=` exactly like the CSV in Step 5 — real government
data standing in for data you don't have. (Needs a free Census API key; see
the syntax reference above.)

**This step is not statistically the same as Step 5.** In Step 5, the real
sample *is* your target population. Here, it's someone else's: a Census
survey sample for whatever `state`/`year`/`survey` you asked for, which may
or may not resemble your actual population's demographics. Treat it as a
reasonable stand-in for shape and plausibility, not as evidence your target
population actually has this age/income distribution. If you need the real
distribution for your specific population, that's exactly the case Step 5
or Step 6 is for.

### Step 8: check your work

```python
print(result.summary())        # rows and columns per table
print(result.unjustified())    # every number you haven't sourced yet
```

Always worth running before you trust output: `summary()` catches an
obviously wrong row count at a glance, and `unjustified()` shows you exactly
which numbers in your config are still guesses (tagged `modeled`) rather
than something you chose deliberately or sourced from real data.

---

## Part 4: The front door (`sw.Mode`)

Everything above is the explicit form: you build the rules, the entities,
the tables, and the pipeline yourself, and you wire the synthesizer and the
noise config by hand. That is worth understanding, and it is what you will
reach for when a schema gets specific.

`sw.Mode` is a shorter way in for the common cases. It is sugar, not a
wrapper: `entity()` and `table()` hand back real `sw.Entity` and `sw.Table`
objects, and `schema().run()` runs a real `sw.Pipeline`. Nothing is hidden
and nothing is locked away. What it saves you is the wiring.

You pick a mode by what you are starting with:

| You have | Mode | What it does |
|---|---|---|
| The numbers, but no rows | `sw.Mode.metadata()` | Builds rules. No model at all. |
| A real microdata sample | `sw.Mode.real_data(source=...)` | Fits CART on your rows. |
| Neither, but a place | `sw.Mode.scope(area_code=..., epsilon=...)` | Fetches a real population from ACS PUMS. |

All three share the same four methods.

### The four methods every mode has

**`attribute(name, **kwargs)`** describes one column. What the keyword
arguments mean depends on the mode (see below). Any of `missing_rate=`,
`typo_rate=`, and `ocr_rate=` can be passed in *any* mode: they are recorded
rather than applied, and become the matching `sw.Missing`/`sw.Typo`/`sw.OCR`
noise ops on every table that carries that attribute. The rule you get back
is clean; the mess is added at the end of the pipeline, as it always was.

**`entity(name, count=, attributes=, identifiers=)`** is a plain
`sw.Entity`, same arguments as the reference above.

**`table(name, grain=, columns=, carry=, identifiers=, coverage=)`** is a
plain `sw.Table`, same arguments as the reference above.

**`schema(entities=, tables=, seed=)`** returns a small object with
`.run()` and `.run_to(path, format=...)` on it, which build and run a real
`sw.Pipeline` with the synthesizer and noiser the mode assembled for you.
This is also where your noise rates are matched against the schema's real
columns, once every table is known and `carry="*"` has been expanded.

### What every mode refuses

Failures are named, and they are named the same way in all three modes: a
`ValueError` at the line you wrote, saying which attribute or column is at
fault.

- An unknown keyword to `attribute()` is rejected by name, in every mode. A
  misspelled `missing_rate=` is a typo, not a silent no-op.
- A noise rate declared for a column no table carries or generates raises at
  the `schema()` call, naming the unmatched column.
- A non-positive `epsilon` raises at the `real_data()`, `scope()`, or
  `attribute()` call that set it, in both modes that take one. It is not
  quietly clamped into a plausible-looking value.

### `sw.Mode.metadata()`: you know the numbers

`attribute()` here takes the shape of a distribution and gives you back a
`Rule`:

- `min=` and `max=` → `sw.Uniform(min, max)`
- `mean=`, `sd=`, `distribution="normal"`, optional `min=`/`max=` as
  clipping bounds → `sw.Normal(...)`
- `values=`, optional `weights=` → `sw.Choice(values, weights)`

Anything that does not resolve (e.g. `distribution="normal"` with no `sd=`)
raises `ValueError` naming what is missing, at the `attribute()` call, before
any generation work starts.

```python
import synthweave as sw

m = sw.Mode.metadata()
education = m.attribute("education", values=["HS", "College"], weights=[0.6, 0.4])
income = m.attribute("income", mean=45_000, sd=12_000, distribution="normal",
                     min=0, missing_rate=0.05)

people = m.entity("person", count=2_000,
                  attributes={"education": education, "income": income},
                  identifiers=["tax_id"])
roster = m.table("roster", grain="person", carry=["education", "income"],
                 identifiers=["tax_id"])

result = m.schema(entities=[people], tables=[roster], seed=42).run()
print(result["roster"].head())
print(result["roster"]["income"].isna().mean())   # about 0.05
```

### `sw.Mode.real_data(source=..., epsilon=...)`: you have real rows

`source` is a `pandas.DataFrame`, or a path to a `.csv` or `.parquet` file.
Any other extension raises `ValueError` naming the path. `attribute(name)`
here needs no distribution: the column's values come from the donor rows,
sampled the way `sw.CARTSynthesizer` samples them.

**`epsilon` is not differential privacy.** There is no Laplace mechanism,
no Gaussian mechanism, and no privacy accounting anywhere in this library.
`epsilon` maps onto `sw.CARTSynthesizer`'s existing generalization
controls (`max_depth`, `min_samples_leaf`, `fit_cap`), where a lower value
fits shallower trees on fewer rows. That genuinely generalizes more and
discloses less, which is worth having, but it is not a formal guarantee of
anything and must not be reported as one. Set it per mode, or override it
for a single column.

```python
import numpy as np
import pandas as pd
import synthweave as sw

rng = np.random.default_rng(0)
age = rng.integers(18, 85, size=2_000)
sample = pd.DataFrame({"age": age,
                       "income": np.clip(age * 900 + rng.normal(0, 8_000, 2_000), 0, None)})

m = sw.Mode.real_data(source=sample, epsilon=1.0)
a = m.attribute("age")
i = m.attribute("income", epsilon=0.5)     # generalize this column harder

people = m.entity("person", count=2_000, attributes={"age": a, "income": i})
table = m.table("residents", grain="person", carry=["age", "income"])

result = m.schema(entities=[people], tables=[table], seed=7).run()
print(result["residents"].head())
```

Two attributes at the same `epsilon` are fitted by one synthesizer; two at
different values get one each, since two generalization levels cannot be
fitted as a single tree.

### `sw.Mode.scope(area_code=..., epsilon=...)`: you have neither, but you have a place

Fetches real (anonymized) ACS PUMS survey responses from the US Census
Bureau and synthesizes from those, the same way `real_data` mode does with
your own rows.

**`area_code` locks to a US state and nothing finer.** The connector
resolves state-level geography only: no county, no PUMA. `"NY"`,
`"New York"`, and `"36"` all mean the same whole state.

`attribute()` here requires `variable=`, the ACS variable code the column
should be filled from. There is no name guessing: `"income"` will not be
matched to `PINCP` for you, because a wrong guess would silently produce
plausible data from the wrong column. A missing `variable=` raises
`ValueError` naming the attribute.

**`epsilon` is not differential privacy here either.** There is no Laplace
mechanism, no Gaussian mechanism, and no privacy accounting anywhere in this
library. It is the same knob as in `real_data` mode: `sw.CARTSynthesizer`'s
generalization controls (`max_depth`, `min_samples_leaf`, `fit_cap`), where
a lower value fits shallower trees on fewer rows. That generalizes more and
discloses less, which is worth having, but it is not a formal guarantee of
anything and must not be reported as one.

It matters more here than it does with your own data, not less. The donor
rows are real Census respondent records. Left at the synthesizer's unbounded
defaults, a scope run would disclose more about those respondents than a
`real_data` run discloses about rows you already hold. `epsilon` defaults to
`1.0`, and `attribute(name, variable=..., epsilon=...)` overrides it for one
column. The grouping rule is the same as `real_data`'s: columns at the same
epsilon are fitted by one synthesizer, columns at different values get one
each.

```python
import synthweave as sw

m = sw.Mode.scope(area_code="NY", epsilon=1.0)
age = m.attribute("age", variable="AGEP")
income = m.attribute("income", variable="PINCP", epsilon=0.5)  # generalize this one harder

people = m.entity("person", count=2_000, attributes={"age": age, "income": income})
table = m.table("ny_residents", grain="person", carry=["age", "income"])

result = m.schema(entities=[people], tables=[table], seed=3).run()
print(result["ny_residents"].head())
```

The fetch happens once, for every scope attribute at the same time, and is
cached on the mode, so a second `.run()` does not go back to the network.
Needs a free Census API key as `CENSUS_API_KEY`. See the syntax reference
entry for `fetch_pums` above.

The same caveat from Step 7 applies here, and is worth repeating: an ACS
sample for a state is *someone else's* population, not yours. Treat it as a
stand-in for shape and plausibility, not as evidence about your actual
target population.

---

## Extending it

Everything synthweave ships with (`CARTSynthesizer`, `Declared`/`Empirical`/`Prior`,
`Typo`/`OCR`/`Missing`, the deterministic linker) is a plain Python class
matching a small `Protocol` (an expected shape: certain methods with certain
arguments), registered under a name. Adding your own noise op, structure
source, or synthesizer never means editing synthweave itself:

```python
from synthweave.stages.base import own

@sw.register("noiser", "scanner-artifacts")
class ScannerArtifacts:
    def run(self, chunks, table, ctx):
        for chunk in chunks:
            chunk = own(chunk)     # a chunk may be a view; copy before writing
            # ... your corruption logic ...
            yield chunk

sw.Pipeline(schema, noiser="scanner-artifacts").run()
```

`sw.available("noiser")` lists every noise op currently registered, built-in
or your own. The same pattern (`sw.register(kind, name)`, then pass that
name as a string) works for `"synthesizer"`, `"structure"`, `"generator"`,
and `"linker"`.

A custom `Rule` can also declare the type its column holds, so synthesized
output keeps a consistent dtype no matter what chunk size it was drawn in:

```python
class Age(sw.Rule):
    def draw(self, keys, *, seed, salt, frame=None): ...
    def depends_on(self): return ()
    def dtype(self): return np.dtype(np.int64)   # optional
```

This is declared rather than inferred on purpose: inferring a column's type
from its values is chunk dependent (a chunk of whole numbers looks like an
integer column, and the next one looks like a float), which would make a
column's type silently depend on `chunk_size`. Leave `dtype` out and the
column is passed through exactly as drawn.

## Where to go next

- `README.md` — the short version, once this vocabulary is familiar.
- `examples/linked_admin_records.py` — a full three-table worked example,
  explicit form throughout.
- `examples/three_layers_data_availability.py` — Steps 5, 6, and 7 above,
  as one runnable script.
- `examples/three_modes.py`: Part 4's three modes, as one runnable script.
