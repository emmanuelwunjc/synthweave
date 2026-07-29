# synthweave

Synthetic tabular data that does not require real microdata.

Most synthetic data tools fit a model to data you already have. That is no help
when you have none, cannot get access, or are not allowed to put what you have
into a model. synthweave generates from a schema you declare, then optionally
adds statistical structure, realistic messiness, and identifiers that link
across tables.

It works the other way round too. If you do have real microdata, hand it over
and synthweave fits on it like any other tool.

```bash
pip install synthweave
```

## What it does

Three stages, each optional, composed into a pipeline.

| Stage | What it does | Needs real data? |
|---|---|---|
| **Generate** | Draws rows from declared entities, grains, and column rules | No |
| **Synthesize** | Sequential CART synthesis against a selectable structure source | No |
| **Finish** | Deterministic cross-table identifiers, then configurable messiness | No |

## Example

```python
import synthweave as sw

people = sw.Entity(
    "person",
    count=50_000,
    attributes={
        "education": sw.Choice(["HS", "College"], [0.6, 0.4]),
        "birth_year": sw.Integer(1960, 2005),
    },
    identifiers=[
        sw.Identifier("student_id", prefix="SID"),
        sw.Identifier("tax_id", prefix="TIN"),
    ],
)

roster = sw.Table(
    "roster",
    grain=sw.PerEntity("person"),
    carry=["education", "birth_year"],
    identifiers=["student_id", "tax_id"],
)

wages = sw.Table(
    "wages",
    grain=sw.PerPeriod("person", periods=range(2018, 2026), presence=0.8),
    carry=["education"],
    identifiers=["tax_id"],
    coverage=0.7,
    columns={
        "wage": sw.Conditional("education", {
            "HS":      sw.Normal(38_000, 9_000, low=0),
            "College": sw.Normal(64_000, 18_000, low=0),
        }),
    },
)

result = sw.Pipeline(
    sw.Schema(entities=[people], tables=[roster, wages], seed=42),
    noiser=sw.Noise({"roster": {"education": [sw.Typo(0.05)]}}),
).run()

result["wages"].head()
```

A person's `tax_id` is the same string in `roster` and in `wages`. Nothing was
looked up to make that true.

There is a fuller, three-table example in
[`examples/linked_admin_records.py`](examples/linked_admin_records.py).

## Core ideas

### Entities come first, tables record them at a grain

You define a person once. Tables then say how they record that person:

```python
sw.PerEntity("person")                                  # a roster
sw.PerPeriod("person", periods=[2020, 2021], presence=0.8)   # panel data
sw.PerEvent("person", low=1, high=5)                    # transactions
```

`coverage` controls what share of the population reaches a table, so tables
overlap partially the way real administrative files do rather than matching
perfectly.

### Nothing is random, everything is derived

There is no RNG anywhere. Every value is a pure function of
`(seed, a stable key, a salt naming the draw)`. Three things follow:

- **Reproducible.** Same seed, same output, forever.
- **Order independent.** A row's value does not depend on what was processed
  before it.
- **Chunk invariant.** `chunk_size` is purely a memory knob. The test suite
  proves this by running the same config at several chunk sizes and asserting
  the outputs are byte-identical.

That last property is what makes identifiers link without any lookup table. An
identifier is a function of the entity and a tag, so two tables generated years
apart from the same seed still join.

### Structure has to come from somewhere

This is the part most worth understanding.

If a generator draws every column independently, the data has no inter-column
relationships. Fitting a model on that output teaches it nothing, and it hands
back independent columns. So the synthesizer does not learn from its input. It
takes a **structure source**:

```python
sw.Declared()          # conditional rules in your schema put structure there
sw.Empirical(df)       # learn from real microdata you supply
sw.Prior(marginals=…)  # fit against published aggregates, no rows needed
```

`Declared` is what makes the no-real-data path produce structured output rather
than noise. `sw.Conditional("education", {...})` is how you say wage depends on
education, and the synthesizer generalizes that relationship rather than
inventing one.

### Every config value says where it came from

```python
coverage = sw.cited(0.82, "NCES Table 219.10, retrieved 2026-07-29")
gap_rate = sw.modeled(0.08, "assumption, not yet sourced")

result.unjustified()   # every value nobody has defended yet
result.provenance.to_frame()   # the whole record, for a methods section
```

A config full of bare numbers is impossible to review. Tagging makes the
difference between a figure you chose, a figure you took from a source, and a
placeholder visible at a glance.

## Scale

Stages pass chunks, not tables, so memory stays flat as row counts grow.

Measured at roughly **200,000 rows/s with peak RSS around 0.44 GB**, streamed
straight to disk, on a 3.2 million row run. The memory figure is the load
bearing one: it does not grow with row count, so tens of millions of rows is a
matter of time, not RAM. Throughput varies with machine load, so treat it as an
order of magnitude rather than a benchmark.

```python
sw.Pipeline(schema, chunk_size=200_000).run_to("out/", format="parquet")
```

Fitting is the only stage that must see more than one row at a time. It uses a
cap, which is a maximum rather than a mandate: below the cap every row is
fitted, exactly as a single-table tool would do. Only above it is a
deterministic sample taken.

## Extending it

Every stage is a Protocol with a registry. Match the shape and register it; no
inheritance, no library edits.

```python
from synthweave.stages.base import own

@sw.register("noiser", "scanner-artifacts")
class ScannerArtifacts:
    def run(self, chunks, table, ctx):
        for chunk in chunks:
            chunk = own(chunk)     # a chunk may be a view; copy before writing
            ...
            yield chunk

sw.Pipeline(schema, noiser="scanner-artifacts").run()
```

`sw.available("noiser")` lists what is registered.

## Where this came from

The design generalizes two patterns proven in a real engagement building linked
synthetic administrative data: deterministic hash-derived identifiers, and
provenance-tagged config. It draws on
[pseudopeople](https://pseudopeople.readthedocs.io/) for the idea of
realistic noise models,
[synthpop](https://cran.r-project.org/package=synthpop) and
[tidysynthesis](https://github.com/UrbanInstitute/tidysynthesis) for sequential
CART synthesis. What is new is combining them on a schema you define yourself,
with no real data required as a starting point.

## Status

v0.1. Every capability is present, and several are deliberately preliminary.
One implementation ships per stage; the interfaces are the commitment, so
adding implementations should not require restructuring.

Not yet included: R interop, deep generative synthesizers, disclosure-risk
metrics, differential privacy. See
[`docs/specs/synthweave-v0.1.md`](docs/specs/synthweave-v0.1.md) for full scope
and [`docs/ISSUES.md`](docs/ISSUES.md) for known issues.

## Development

```bash
pip install -e ".[dev]"
PYTHONPATH=src python -m pytest tests/ -q     # 61 tests, a few seconds
python tools/mutation_check.py                # see below
```

Tests run offline and go through the public API only, so stage internals can be
rewritten without touching them.

`tools/mutation_check.py` reverts each fix recorded in
[`docs/ISSUES.md`](docs/ISSUES.md) one at a time and checks that a test
actually goes red. A fix nothing catches is not a fix, and this has already
found two issues that were marked resolved while having no coverage at all.
Run it after fixing anything.

## License

MIT
