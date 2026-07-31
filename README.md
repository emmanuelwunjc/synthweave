# synthweave

**Synthetic tabular data that does not require real microdata.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Status: v0.1](https://img.shields.io/badge/status-v0.1-orange.svg)](#status)

Most synthetic data tools fit a model to data you already have. That's no
help when you have none, can't get access, or aren't allowed to put what you
have into a model. synthweave generates from a schema you declare by hand,
then optionally layers in statistical structure, realistic messiness, and
identifiers that link correctly across tables, all without a single real row
required to start.

It works the other way round too. Have real microdata? Hand it over and
synthweave fits on it like any other synthesis tool. Have only published
statistics, no rows at all? It works from those instead. Have neither? It can
pull real reference data from a public source (the US Census) to stand in.

New here? Every term below is defined in **[the full guide](docs/GUIDE.md)** —
glossary, complete syntax reference, and an 8-step tutorial. This README is
the fast tour.

## Contents

- [Install](#install)
- [Quickstart](#quickstart)
- [Why synthweave](#why-synthweave)
- [How it works](#how-it-works)
- [Three ways to use it](#three-ways-to-use-it)
- [FAQ](#faq)
- [Scale](#scale)
- [Status](#status)
- [Development](#development)

## Install

```bash
pip install synthweave
```

Or install straight from this repo (the latest commit, ahead of the last
PyPI release):

```bash
pip install "git+https://github.com/emmanuelwunjc/synthweave.git#subdirectory=package"
```

Or clone and install locally for development:

```bash
git clone https://github.com/emmanuelwunjc/synthweave.git
cd synthweave/package
pip install -e .
```

## Quickstart

```python
import synthweave as sw

people = sw.Entity("person", count=10_000, attributes={"education": ["HS", "College"]})
table = sw.Table("roster", grain="person", carry="*")
result = sw.Pipeline(sw.Schema([people], [table])).run()

result["roster"].head()
```

Ten thousand rows, no data file, no API key, no setup beyond `pip install`.
That's the entire shape of synthweave: describe a population once
(`Entity`), describe a table that records it (`Table`), run it
(`Pipeline`). Everything else in this README is a variation on those three
lines.

## Why synthweave

|  | synthweave | A GAN/VAE-based synthesizer (a deep learning model trained on your real data) | A fake-data generator (e.g. Faker) |
|---|---|---|---|
| Needs real data to start? | No | Yes, always | No |
| Preserves relationships between columns? | Yes, if declared or learned | Yes, if you have training data | No, columns are independent |
| Cross-table IDs that actually link? | Yes, derived, no lookup table | Not built in | Not built in |
| Deterministic, reproducible output? | Yes, for a given seed, regardless of chunk size | Usually not (model training has randomness) | Usually yes |

Config transparency is a synthweave-specific feature, not a category
comparison: every number in a config can be tagged with where it came from
(see [Every config number says where it came from](#how-it-works) below),
which the other two approaches don't attempt to solve.

The gap synthweave targets: you need data that *looks and behaves* like a
real linked administrative dataset (a school roster, a wage file, a benefits
record, all describing the same people), and you either can't access real
microdata, or you're only prototyping and don't want to touch it yet.

## How it works

A pipeline is up to four stages, each optional past the first:

| Stage | Does what | Needs real data? |
|---|---|---|
| **Generate** | Draws every attribute and column from its rule | No |
| **Synthesize** | Refits some columns using a statistical model, against a chosen *structure source* — see below | No |
| **Link** | Attaches identifier columns that match the same person across tables | No |
| **Noise** | Deliberately corrupts values (typos, blanks, scanned-form errors) | No |

Three properties make the whole thing hold together:

- **Nothing is random.** Every value is a pure function of `(seed, a stable
  key, a salt)`. Same seed, same output, forever, regardless of chunk size —
  verified across fresh process runs, and locked in by the test suite. That's
  what lets two tables generated years apart still link correctly, with no
  lookup table involved. (One caveat for the statistically minded: the
  underlying model-fitting step uses scikit-learn's decision trees, which
  don't take an explicit seed from synthweave. This hasn't produced any
  observed non-determinism, but unlike everything else in synthweave, it
  isn't derived from the seed by construction — worth knowing if you're
  relying on bit-for-bit reproducibility as a hard guarantee rather than an
  extremely well-tested one.)
- **Structure has to come from somewhere.** A model fit on independently
  generated columns learns nothing. So the synthesizer always names a
  **structure source** — where it learns relationships *between* columns
  from:
  - `Declared` — nothing extra; learn from the relationships your own
    `Conditional` rules already put in the data. Use this when you have no
    real data, only domain knowledge.
  - `Empirical` — learn from a real sample you supply.
  - `Prior` — learn from published statistics (percentage breakdowns), no
    rows needed.

  See "Three ways to use it" below for all three in code.
- **Every config number says where it came from.** Wrap a guess in
  `sw.modeled(0.08, "not yet sourced")`, a real figure in
  `sw.cited(0.82, "NCES Table 219.10")`. `result.unjustified()` lists every
  number nobody has defended yet, so a config is auditable instead of a wall
  of magic numbers.

There's also a reduced-typing shorthand for the common cases (a bare list
instead of `sw.Choice([...])`, `grain="person"` instead of
`sw.PerEntity("person")`, and more) — see [the fast path](docs/GUIDE.md#part-2-syntax-reference)
in the full guide.

## Three ways to use it

Pick the one that matches what you actually have. All three build on this
shared setup:

```python
people = sw.Entity("person", count=10_000, attributes={"education": ["HS", "College"]})
wages = sw.Table("wages", grain="person", carry=["education"], columns={"wage": 0.0})
schema = sw.Schema([people], [wages])
```

### "I have no data, only domain knowledge"

Declare the relationship yourself. `Conditional` puts real structure into the
data before anything is fitted, so wage genuinely depends on education
because you said so, not because a model guessed it — no `synthesizer=`
needed at all, this is what `wages` above already does once you replace its
placeholder `"wage": 0.0` with:

```python
wages = sw.Table("wages", grain="person", carry=["education"], columns={
    "wage": sw.Conditional("education", {
        "HS": sw.Normal(38_000, 9_000, low=0),
        "College": sw.Normal(64_000, 18_000, low=0),
    }),
})
schema = sw.Schema([people], [wages])
result = sw.Pipeline(schema).run()
```

### "I have some real data, and want more like it"

Keep `wages` as shown in the shared setup (a `0.0` placeholder — the
synthesizer overwrites it), and hand your real sample straight to it. It
learns the actual statistical relationships in your data and generates new
rows that carry them, without exposing any single real row verbatim as a
whole — though see the FAQ below on what "verbatim" does and doesn't cover.

```python
import pandas as pd

real_sample = pd.read_csv("my_real_wages.csv")  # columns: education, wage
result = sw.Pipeline(
    schema,
    synthesizer=sw.CARTSynthesizer(["wage"], predictors=["education"], structure=real_sample),
).run()
```

### "I have neither real data nor a real sample"

If you have published statistics (a report says "60% HS, 40% College"), use
those directly, no rows needed:

```python
result = sw.Pipeline(
    schema,
    synthesizer=sw.CARTSynthesizer(["education"], structure={"education": {"HS": 0.6, "College": 0.4}}),
).run()
```

That resynthesizes `education` in place, even though it's a *carried*
attribute rather than a column declared directly on `wages` — the
synthesizer doesn't care which, it only needs the column to exist by the
time it runs.

If you don't even have that, borrow real government survey data as a
stand-in. **Worth knowing:** this is real data, but not *your* population —
it's a Census survey sample for a given state/year, so treat it as a
plausible reference distribution, not a guarantee that it matches your
actual target population's demographics.

```python
from synthweave.connectors.acs_pums import fetch_pums

acs = fetch_pums(["AGEP", "PINCP"], state="NY").rename(columns={"AGEP": "age", "PINCP": "income"})
synthesizer = sw.CARTSynthesizer(["age", "income"], structure=acs)
```

Needs a free Census API key ([sign up here](https://api.census.gov/data/key_signup.html))
set as `CENSUS_API_KEY`. Full working version of all three: `examples/three_layers_data_availability.py`.

## FAQ

**Is this like Faker?**
No. Faker generates realistic-looking individual values (names, addresses,
emails) with no relationship between columns or rows. synthweave generates
whole linked datasets: multiple tables, the same person's ID matching
correctly across all of them, and real statistical relationships between
columns, either declared or learned.

**Does my real data ever leave my machine?**
No. `Empirical(df)` and `fetch_pums(...)` both run entirely locally; nothing
is uploaded anywhere. The Census connector calls the public Census API to
*fetch* government-published data, it never sends anything.

**Is the output actually private / safe to share?**
synthweave does not (yet) include formal disclosure-risk metrics or
differential privacy. Treat v0.1 output as "structurally realistic, not
formally privacy-audited," with one specific mechanism worth naming: when a
synthesized column's structure source is `Empirical` (real data), each
output value for that column is sampled from the *actual real values* that
landed in the same decision-tree leaf as the row being synthesized, not a
resampled or perturbed value. With few real rows behind a narrow leaf, that
is a real attribute-disclosure vector if the columns you condition on are
also identifying. See `CARTSynthesizer`'s `min_samples_leaf` in
[the full guide](docs/GUIDE.md) to control leaf size, and [Status](#status).

**How do I check that `Empirical`/CART synthesis actually preserved my real
data's structure?**
`sw.fidelity_report(synth_df, real_df, columns=[...])`. `Declared` and
`Prior` have no original relationship to check against — the schema declares
whatever structure it wants — but `Empirical` fits on real data, and the
whole point is that the output should carry the same statistical
relationships (marginal distributions, category shares, pairwise
associations) the real input had. The report compares both, plus how many
rows silently kept a pre-synthesis placeholder value because their
decision-tree leaf had no donors, if you pass the fitted `synthesizer=`. See
[the full guide](docs/GUIDE.md#swfidelity_reportsynth_df-real_df-columns-thresholdsnone-synthesizernone).

**Can I trust the numbers in someone else's config?**
Run `result.unjustified()`. Anything not tagged `cited` with a real source is
either something the author chose deliberately (`user-provided`, the
default) or an unverified guess (`modeled`) — you can see which is which at
a glance.

**How big can the output get?**
Memory stays flat as row count grows; see [Scale](#scale). Tested to several
million rows streamed to disk.

**Do I need a Census API key for anything other than the ACS PUMS connector?**
No. Everything else (declared structure, your own real data, published
aggregates) needs no external service or key at all.

**What if my dataset doesn't fit any of the three built-in structure sources?**
Every stage is a `Protocol` with a registry: implement the shape, register
it, no library edits or inheritance required. See "Extending it" in
[the full guide](docs/GUIDE.md).

## Scale

Stages pass chunks, not whole tables, so memory stays flat as row counts
grow. Measured at roughly **200,000 rows/s with peak RSS around 0.44 GB**,
streamed straight to disk, on a 3.2 million row run. The memory figure is the
load-bearing one: it doesn't grow with row count, so tens of millions of rows
is a matter of time, not RAM. Throughput varies with machine load, so treat
it as an order of magnitude, not a benchmark.

```python
sw.Pipeline(schema, chunk_size=200_000).run_to("out/", format="parquet")
```

## Status

v0.1. Every capability described above is present and tested; several are
deliberately preliminary. One implementation ships per stage; the interfaces
are the real commitment, so adding implementations shouldn't require
restructuring.

Not yet included: R interop, deep generative synthesizers (GAN/VAE-based),
disclosure-risk metrics, differential privacy.

Known issues and the design reasoning behind them are tracked in
[the issue tracker](https://github.com/emmanuelwunjc/synthweave/issues), not
in the repo.

## Development

```bash
pip install -e ".[dev]"
PYTHONPATH=src python -m pytest tests/ -q
```

Tests run offline and go through the public API only, so internal stage code
can be rewritten without touching them. Every bug fix is checked by
reverting it and confirming a test goes red; a fix nothing catches isn't a
fix.

## Where this came from

The design generalizes two patterns proven in a real engagement building
linked synthetic administrative data: deterministic hash-derived
identifiers, and provenance-tagged config. It draws on
[pseudopeople](https://pseudopeople.readthedocs.io/) for realistic noise
models, and [synthpop](https://cran.r-project.org/package=synthpop) /
[tidysynthesis](https://github.com/UrbanInstitute/tidysynthesis) for
sequential CART synthesis. What's new is combining them on a schema you
define yourself, with no real data required to start.

Also runs in a browser:
[synthweave-frontend](https://github.com/emmanuelwunjc/synthweave-frontend)
is a local front end that runs this pipeline, checks the linking guarantee
against the produced data, and re-runs your config at three chunk sizes to
prove the output is identical. Ships a three-table worked example and a
walkthrough deck.

## License

MIT
