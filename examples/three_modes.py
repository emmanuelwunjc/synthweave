"""The three modes, end to end, by what you start with.

`sw.Mode` is a front door over the same schema/pipeline API the rest of the
library exposes. It does not hide anything: `entity()` and `table()` hand
back real `sw.Entity`/`sw.Table` objects, and `schema().run()` runs a real
`sw.Pipeline`. What it saves you is the wiring, mostly the synthesizer and
the noise config.

1. metadata   You know the numbers but have no rows. Rules, no model.
2. real_data  You have a real microdata sample. Empirical + CART.
3. scope      You have neither, so borrow a real population, locked to a
              US state, from the Census Bureau's ACS PUMS API.

Mode 3 needs a Census API key (free, https://api.census.gov/data/key_signup.html)
as `CENSUS_API_KEY`, in the environment or a `.env` file at the repo root. If
that's not set up, this script says so and moves on rather than failing.

Run it:  python3 examples/three_modes.py
"""

import numpy as np
import pandas as pd

import synthweave as sw

# --- mode 1: metadata -------------------------------------------------------

print("=== Mode 1: metadata (you know the numbers, you have no rows) ===")

meta = sw.Mode.metadata()
education = meta.attribute("education", values=["HS", "College"], weights=[0.6, 0.4])
# missing_rate is bookkeeping, not part of the rule: the rule still draws a
# clean value, and the mode wires an sw.Missing(0.05) noise op for any table
# that carries this attribute.
income = meta.attribute("income", mean=45_000, sd=12_000, distribution="normal",
                        min=0, missing_rate=0.05)
age = meta.attribute("age", min=18, max=80)

people = meta.entity("person", count=2_000,
                     attributes={"education": education, "income": income, "age": age},
                     identifiers=["tax_id"])
roster = meta.table("roster", grain="person", carry=["education", "income", "age"],
                    identifiers=["tax_id"])

result = meta.schema(entities=[people], tables=[roster], seed=42).run()
frame = result["roster"]
print(frame.head(4).to_string(index=False))
print(f"rows: {len(frame)}, income missing: {frame['income'].isna().mean():.1%} "
      f"(asked for 5%)")
print(f"education shares: {frame['education'].value_counts(normalize=True).round(2).to_dict()}\n")


# --- mode 2: real_data ------------------------------------------------------

print("=== Mode 2: real_data (you have a real microdata sample) ===")

# Stands in for a real sample the caller already has on disk. A path to a
# .csv or .parquet works here just as well as a DataFrame.
rng = np.random.default_rng(0)
sample_age = rng.integers(18, 85, size=2_000)
sample_income = np.clip(sample_age * 900 + rng.normal(0, 8_000, size=2_000), 0, None)
real_sample = pd.DataFrame({"age": sample_age, "income": sample_income.round(0)})

# epsilon is NOT differential privacy. There is no Laplace or Gaussian
# mechanism here and no privacy accounting. It maps onto CARTSynthesizer's
# existing generalization knobs (max_depth, min_samples_leaf, fit_cap):
# lower epsilon fits shallower trees on fewer rows, which generalizes more
# and discloses less, but is not a formal guarantee of anything.
rd = sw.Mode.real_data(source=real_sample, epsilon=1.0)
rd_age = rd.attribute("age")
rd_income = rd.attribute("income", epsilon=0.5)   # generalize this one harder

rd_people = rd.entity("person", count=2_000,
                      attributes={"age": rd_age, "income": rd_income},
                      identifiers=["tax_id"])
rd_table = rd.table("residents", grain="person", carry=["age", "income"],
                    identifiers=["tax_id"])

rd_result = rd.schema(entities=[rd_people], tables=[rd_table], seed=7).run()
rd_frame = rd_result["residents"]
print(rd_frame.head(4).to_string(index=False))
print(f"rows: {len(rd_frame)}, age range: {rd_frame['age'].min()}-{rd_frame['age'].max()}")
# Donor sampling: every synthesized value is a value some real row held.
print(f"every age came from the donor sample: "
      f"{set(rd_frame['age']) <= set(real_sample['age'])}\n")


# --- mode 3: scope ----------------------------------------------------------

print("=== Mode 3: scope (borrow a real population, locked to a US state) ===")

# area_code locks to a US STATE and nothing finer. fetch_pums resolves no
# county and no PUMA, so "NY" here means the whole state.
scope = sw.Mode.scope(area_code="NY")
scope_age = scope.attribute("age", variable="AGEP")
scope_income = scope.attribute("income", variable="PINCP")

scope_people = scope.entity("person", count=2_000,
                            attributes={"age": scope_age, "income": scope_income},
                            identifiers=["tax_id"])
scope_table = scope.table("ny_residents", grain="person", carry=["age", "income"],
                          identifiers=["tax_id"])

try:
    scope_result = scope.schema(entities=[scope_people], tables=[scope_table], seed=3).run()
    scope_frame = scope_result["ny_residents"]
    print(scope_frame.head(4).to_string(index=False))
    print(f"rows: {len(scope_frame)}, "
          f"age range: {scope_frame['age'].min()}-{scope_frame['age'].max()}")
except RuntimeError as e:
    print(f"skipped: {e}")
