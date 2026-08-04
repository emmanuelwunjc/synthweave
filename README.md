<div align="center">

# synthweave

**Synthetic tabular data that does not require real microdata.**

[![PyPI](https://img.shields.io/pypi/v/synthweave?color=blue)](https://pypi.org/project/synthweave/)
[![CI](https://github.com/emmanuelwunjc/synthweave/actions/workflows/ci.yml/badge.svg)](https://github.com/emmanuelwunjc/synthweave/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[Quickstart](#quickstart) · [Why](#why-synthweave) · [How it works](#how-it-works) · [Three ways to use it](#three-ways-to-use-it) · [FAQ](#faq) · [Full guide](docs/GUIDE.md) · [Doc index](docs/MAP.md)

</div>

---

Most synthetic data tools fit a model to data you already have. That is no help
when you have none, cannot get access, or are not allowed to put what you have
into a model.

synthweave generates from a schema you declare by hand, then optionally layers
in statistical structure, realistic messiness, and identifiers that link
correctly across tables. It works the other way round too:

| What you have | What synthweave does |
|---|---|
| Nothing but domain knowledge | Generates from rules you declare |
| Real microdata | Fits on it, like any other synthesis tool |
| Only published statistics | Expands those into a training frame |
| None of the above | Pulls public US Census data as a stand-in |

> **New here?** Every term below is defined in **[the full guide](docs/GUIDE.md)**:
> glossary, complete syntax reference, and an 8-step tutorial. This README is the
> fast tour.

## Install

```bash
pip install synthweave
```

<details>
<summary>Other install options</summary>

Latest commit, ahead of the last PyPI release:

```bash
pip install "git+https://github.com/emmanuelwunjc/synthweave.git"
```

Local development:

```bash
git clone https://github.com/emmanuelwunjc/synthweave.git
cd synthweave
pip install -e ".[dev]"
```

Optional extras: `[parquet]` for Parquet output, `[pii]` for Faker-backed names
and SSNs.

</details>

## Quickstart

```python
import synthweave as sw

people = sw.Entity("person", count=10_000,
                   attributes={"education": ["HS", "College"],
                               "birth_year": sw.Integer(1960, 2005)},
                   identifiers=["tax_id"])

roster = sw.Table("roster", grain="person", carry="*", identifiers=["tax_id"])

result = sw.Pipeline(sw.Schema([people], [roster], seed=7)).run()
print(result["roster"].head(4))
```

```text
       tax_id education  birth_year
0  9509912293        HS        1983
1  7347177053        HS        1990
2  0931814884        HS        1969
3  2374398424   College        1963
```

Ten thousand rows. No data file, no API key, no setup beyond `pip install`.

That is the whole shape of synthweave: describe a population once (`Entity`),
describe a table that records it (`Table`), run it (`Pipeline`). Everything
below is a variation on those three lines.

Run it again with `seed=7` on any machine, at any `chunk_size`, and you get
those same four rows. Add a second table carrying `tax_id` and the same
person keeps the same id in both, with no lookup table anywhere.

## Why synthweave

|  | synthweave | GAN/VAE synthesizer | Fake-data generator (Faker) |
|---|:---:|:---:|:---:|
| Needs real data to start | No | Always | No |
| Preserves column relationships | Yes, declared or learned | Yes, with training data | No |
| Cross-table IDs that link | Yes, derived, no lookup table | Not built in | Not built in |
| Deterministic output | Yes, per seed, any chunk size | Usually not | Usually yes |
| Config provenance tracking | Yes | Not attempted | Not attempted |

**The gap it targets:** you need data that looks and behaves like a real linked
administrative dataset (a school roster, a wage file, a benefits record, all
describing the same people), and you either cannot access real microdata or are
only prototyping and do not want to touch it yet.

## How it works

A pipeline is up to four stages, each optional past the first:

| Stage | Does what | Needs real data |
|---|---|:---:|
| **Generate** | Draws every attribute and column from its rule | No |
| **Synthesize** | Refits columns with a statistical model against a chosen *structure source* | No |
| **Link** | Attaches identifiers that match the same person across tables | No |
| **Noise** | Corrupts values deliberately (typos, blanks, scan errors) | No |

Three properties hold it together.

**Nothing is random.** Every value is a pure function of `(seed, stable key,
salt)`. Same seed, same output, forever, regardless of chunk size. That is what
lets two tables generated years apart still link correctly with no lookup table
involved.

<details>
<summary>How far the determinism guarantee reaches</summary>

All the way, including model fitting. scikit-learn's decision trees are given a
`random_state` derived from the run seed and the name of the column being
fitted, so even a tie-break between two equally good splits reproduces. Nothing
in synthweave draws from an unseeded source. Bit-for-bit reproducibility is a
guarantee, not an observation.

</details>

**Structure has to come from somewhere.** A model fit on independently generated
columns learns nothing, so the synthesizer always names a *structure source*:

- **`Declared`** learns from relationships your own `Conditional` rules already
  put in the data. Use when you have domain knowledge but no data.
- **`Empirical`** learns from a real sample you supply.
- **`Prior`** learns from published statistics. No rows needed.

**Every config number says where it came from.** Wrap a guess in
`sw.modeled(0.08, "not yet sourced")` and a real figure in
`sw.cited(0.82, "NCES Table 219.10")`. Then `result.unjustified()` lists every
number nobody has defended yet, so a config is auditable instead of a wall of
magic numbers.

There is also a reduced-typing shorthand for common cases (a bare list instead
of `sw.Choice([...])`, `grain="person"` instead of `sw.PerEntity("person")`).
See [the fast path](docs/GUIDE.md#part-2-syntax-reference).

## Three ways to use it

Pick the one matching what you actually have. All three share this setup:

```python
people = sw.Entity("person", count=10_000, attributes={"education": ["HS", "College"]})
wages  = sw.Table("wages", grain="person", carry=["education"], columns={"wage": 0.0})
schema = sw.Schema([people], [wages])
```

### 1. Domain knowledge only

Declare the relationship yourself. `Conditional` puts real structure in before
anything is fitted, so wage depends on education because you said so, not
because a model guessed. No `synthesizer=` needed.

```python
wages = sw.Table("wages", grain="person", carry=["education"], columns={
    "wage": sw.Conditional("education", {
        "HS":      sw.Normal(38_000, 9_000, low=0),
        "College": sw.Normal(64_000, 18_000, low=0),
    }),
})
result = sw.Pipeline(sw.Schema([people], [wages])).run()
```

### 2. A real sample you want more of

Keep the `0.0` placeholder (the synthesizer overwrites it) and hand over your
real data. It learns the actual relationships and generates rows that carry
them.

```python
real_sample = pd.read_csv("my_real_wages.csv")   # columns: education, wage

result = sw.Pipeline(
    schema,
    synthesizer=sw.CARTSynthesizer(["wage"], predictors=["education"], structure=real_sample),
).run()
```

### 3. Published statistics, or nothing at all

A report says "60% HS, 40% College"? Use it directly, no rows needed:

```python
result = sw.Pipeline(
    schema,
    synthesizer=sw.CARTSynthesizer(["education"], structure={"education": {"HS": 0.6, "College": 0.4}}),
).run()
```

If you do not even have that, borrow real government survey data as a stand-in:

```python
from synthweave.connectors.acs_pums import fetch_pums

acs = fetch_pums(["AGEP", "PINCP"], state="NY").rename(columns={"AGEP": "age", "PINCP": "income"})
synthesizer = sw.CARTSynthesizer(["age", "income"], structure=acs)
```

> **Worth knowing:** this is real data, but not *your* population. It is a Census
> survey sample for a given state and year, so treat it as a plausible reference
> distribution, not a guarantee it matches your target demographics. Needs a free
> [Census API key](https://api.census.gov/data/key_signup.html) as
> `CENSUS_API_KEY`.

All three run end to end in `examples/three_layers_data_availability.py`.

## FAQ

<details>
<summary><b>Is the output actually private, or safe to share?</b></summary>

synthweave does not yet include formal disclosure-risk metrics or differential
privacy. Treat output as "structurally realistic, not formally
privacy-audited."

One mechanism is worth naming explicitly. When a synthesized column's structure
source is `Empirical`, each output value is sampled from the *actual real
values* that landed in the same decision-tree leaf as the row being
synthesized, not a perturbed or resampled one. With few real rows behind a
narrow leaf, that is a genuine attribute-disclosure vector if the columns you
condition on are also identifying. Control leaf size with `min_samples_leaf`
(see [the guide](docs/GUIDE.md)).

</details>

<details>
<summary><b>Does my real data ever leave my machine?</b></summary>

No. `Empirical(df)` and `fetch_pums(...)` run entirely locally. The Census
connector calls the public Census API to *fetch* government-published data. It
never sends anything.

</details>

<details>
<summary><b>How do I check synthesis preserved my real data's structure?</b></summary>

`sw.fidelity_report(synth_df, real_df, columns=[...])`.

`Declared` and `Prior` have no original to check against, since the schema
declares whatever structure it wants. `Empirical` is different: it fits on real
data, and the point is that output carries the same marginal distributions,
category shares, and pairwise associations the input had. The report compares
both, and with `synthesizer=` also reports `empty_donor_leaves`. That last one
is zero by construction rather than a measurement of your data: every
decision-tree leaf holds at least one training row, so no row can land in a
donorless one. A non-zero value means a bug in synthweave, not a property of
your input.

</details>

<details>
<summary><b>Is this like Faker?</b></summary>

No. Faker generates realistic individual values with no relationship between
columns or rows. synthweave generates whole linked datasets: multiple tables,
the same person's ID matching across all of them, and real statistical
relationships between columns.

</details>

<details>
<summary><b>Can I trust the numbers in someone else's config?</b></summary>

Run `result.unjustified()`. Anything not tagged `cited` with a real source is
either a deliberate choice (`user-provided`) or an unverified guess
(`modeled`), and you can see which at a glance.

</details>

<details>
<summary><b>What if none of the three structure sources fit my case?</b></summary>

Every stage is a `Protocol` with a registry. Implement the shape, register it,
no library edits or inheritance required. See "Extending it" in
[the guide](docs/GUIDE.md).

</details>

## Scale

Stages pass chunks, not whole tables, so memory stays flat as row counts grow.
Measured at roughly **200,000 rows/s with peak RSS around 0.44 GB**, streamed
straight to disk, on a 3.2 million row run.

The memory figure is the load-bearing one: it does not grow with row count, so
tens of millions of rows is a matter of time, not RAM. Throughput varies with
machine load, so treat it as an order of magnitude, not a benchmark.

```python
sw.Pipeline(schema, chunk_size=200_000).run_to("out/", format="parquet")
```

## Status

**v0.2. Pre-1.0: the public API may still change between minor versions.**

Every capability described above is present and tested. One implementation
ships per stage, and the interfaces are the real commitment, so adding
implementations should not require restructuring.

The API freezes at v1.0, which also requires zero known bugs in the
"produces plausible-looking wrong data without raising" class. Those are
tracked openly in [the issue tracker](https://github.com/emmanuelwunjc/synthweave/issues)
along with the design reasoning behind them.

Not yet included: disclosure-risk metrics, differential privacy, R interop,
deep generative synthesizers.

## Development

```bash
pip install -e ".[dev]"
pre-commit install
PYTHONPATH=src python -m pytest tests/ -q
```

Tests run offline and go through the public API only, so internal stage code can
be rewritten without touching them. Every bug fix is checked by reverting it and
confirming a test goes red: a fix nothing catches is not a fix. See
[CONTRIBUTING.md](CONTRIBUTING.md). For everything else in `docs/` and which
file covers what, see [docs/MAP.md](docs/MAP.md).

## Where this came from

The design generalizes two patterns proven in a real engagement building linked
synthetic administrative data: deterministic hash-derived identifiers, and
provenance-tagged config. It draws on
[pseudopeople](https://pseudopeople.readthedocs.io/) for realistic noise models,
and [synthpop](https://cran.r-project.org/package=synthpop) /
[tidysynthesis](https://github.com/UrbanInstitute/tidysynthesis) for sequential
CART synthesis. What is new is combining them on a schema you define yourself,
with no real data required to start.

**Also runs in a browser.**
[synthweave-frontend](https://github.com/emmanuelwunjc/synthweave-frontend) runs
this pipeline locally, checks the linking guarantee against the produced data,
and re-runs your config at three chunk sizes to prove the output is identical.

## License

MIT
