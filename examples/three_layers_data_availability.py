"""Three ways to synthesize the same kind of column, by what data you start with.

1. Real data.       You have a real microdata sample. `sw.Empirical(df)`.
2. Aggregate stats.  You have published shares, no rows. `sw.Prior(marginals=...)`.
3. Neither.          Pull real reference microdata from a public source (here,
                      the Census Bureau's ACS PUMS API) and treat it the same
                      way layer 1 would: `sw.Empirical(fetched_df)`.

Layers 1 and 2 are existing synthweave capabilities (`Empirical`, `Prior`);
this script exists to prove they actually work end to end, since until now
they were only exercised by the unit test suite. Layer 3 is net new: the ACS
PUMS connector at `synthweave.connectors.acs_pums`. Mapping ACS variable
codes (AGEP, PINCP) to this schema's column names (age, income) happens here,
not in the connector, since that mapping is specific to this demo.

Layer 3 needs a Census API key (free, https://api.census.gov/data/key_signup.html)
as `CENSUS_API_KEY`, in the environment or a `.env` file at the repo root. If
that's not set up, this script says so and moves on rather than failing.

Also uses the reduced-input shorthand throughout (a bare entity name for
`grain=`, plain Python values instead of `sw.Constant(0)`/`sw.Empirical(df)`
wrapping) -- see README's "the fast path" section for what each shorthand
expands to.

Run it:  python examples/three_layers_data_availability.py
"""

import numpy as np
import pandas as pd

import synthweave as sw
from synthweave.connectors.acs_pums import fetch_pums

resident = sw.Entity("resident", count=5_000, identifiers=["resident_id"])


def run_layer(table_name: str, columns: dict, synthesizer: sw.CARTSynthesizer) -> pd.DataFrame:
    table = sw.Table(table_name, grain="resident", columns=columns)
    schema = sw.Schema(entities=[resident], tables=[table], seed=7)
    return sw.Pipeline(schema, synthesizer=synthesizer).run()[table_name]


def working_age_sample(acs: pd.DataFrame) -> pd.DataFrame:
    """ACS PUMS rows, renamed to this schema's columns and adults only.

    PINCP is only meaningful for people old enough to have income; ACS marks
    the rest with sentinel values well outside any real income. The rename
    itself is schema-specific (which is why it lives here, in the example,
    and not in the connector).
    """
    sample = acs.rename(columns={"AGEP": "age", "PINCP": "income"})
    return sample[sample["age"] >= 18]


# --- layer 1: real data ------------------------------------------------------

print("=== Layer 1: real data (Empirical) ===")

# Stands in for a real microdata sample the caller already has on disk.
rng = np.random.default_rng(0)
age = rng.integers(18, 85, size=1_000)
income = np.clip(age * 900 + rng.normal(0, 8_000, size=1_000), 0, None)
real_sample = pd.DataFrame({"age": age, "income": income.round(0)})

layer1 = run_layer(
    "from_real_data",
    columns={"age": 0, "income": 0.0},
    synthesizer=sw.CARTSynthesizer(["age", "income"], tables=["from_real_data"], structure=real_sample),
)
print(layer1.head(4).to_string(index=False))
print(f"rows: {len(layer1)}, age range: {layer1['age'].min()}-{layer1['age'].max()}\n")


# --- layer 2: aggregate metadata, no rows ------------------------------------

print("=== Layer 2: aggregate metadata (Prior) ===")

# Stands in for published statistics: an income-bracket breakdown with no
# underlying microdata, the kind of thing a summary table gives you.
income_bracket_shares = {
    "under_25k": 0.22,
    "25k_to_50k": 0.28,
    "50k_to_100k": 0.31,
    "over_100k": 0.19,
}
layer2 = run_layer(
    "from_aggregates",
    columns={"income_bracket": "unknown"},
    synthesizer=sw.CARTSynthesizer(
        ["income_bracket"],
        tables=["from_aggregates"],
        structure=sw.Prior(marginals={"income_bracket": income_bracket_shares}, rows=5_000),
    ),
)
print(layer2.head(4).to_string(index=False))
print("share by bracket:")
print((layer2["income_bracket"].value_counts(normalize=True).round(3)).to_string())
print()


# --- layer 3: not enough data, so borrow real reference data -----------------

print("=== Layer 3: low/no data (ACS PUMS -> Empirical) ===")

try:
    acs_sample = working_age_sample(fetch_pums(["AGEP", "PINCP"], state="NY"))

    layer3 = run_layer(
        "from_acs_pums",
        columns={"age": 0, "income": 0.0},
        synthesizer=sw.CARTSynthesizer(["age", "income"], tables=["from_acs_pums"], structure=acs_sample),
    )
    print(f"fetched {len(acs_sample):,} real ACS PUMS rows for reference")
    print(layer3.head(4).to_string(index=False))
    print(f"rows: {len(layer3)}, age range: {layer3['age'].min()}-{layer3['age'].max()}")
except RuntimeError as e:
    print(f"skipped: {e}")
