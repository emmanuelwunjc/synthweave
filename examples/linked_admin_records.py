"""A linked administrative dataset, no real microdata involved.

The shape follows a real engagement: a K12 graduating cohort, quarterly wage
records from a labour department, and a motor vehicle file, all describing the
same people and joinable on identifiers that were never looked up anywhere.

Three things worth watching:

1. No input data. Every value comes from the schema.
2. `education` is identical for a person in all three tables, and their
   `ssn` is identical in the two tables that carry it, because both derive
   from (seed, entity, tag).
3. Wage depends on education by declaration, so the table has real
   inter-column structure before any model runs.

On the rates below: they are shaped after figures used in a real project,
which attributed them to DHS OHSS, BLS CPS, and Benchimol et al. Those
attributions have not been verified here, so they are tagged `modeled` with a
note rather than `cited`. That is the honest tag, and `result.unjustified()`
lists them precisely so nobody mistakes an illustrative number for a
defended one.

Run it:  python examples/linked_admin_records.py
"""

import synthweave as sw

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

students = sw.Entity(
    "student",
    count=20_000,
    attributes={
        "education": sw.Choice(["HS diploma", "GED", "no credential"], [0.83, 0.06, 0.11]),
        "birth_year": sw.Integer(2002, 2005),
        "county": sw.Choice(["Albany", "Bronx", "Erie", "Kings", "Queens"]),
    },
    identifiers=[
        sw.Identifier("student_id", prefix="SID", digits=9),
        # A separate tag, so the SSN cannot be derived from the student id.
        sw.Identifier("ssn", digits=9),
    ],
)

k12 = sw.Table(
    "k12_cohort",
    grain=sw.PerEntity("student"),
    carry=["education", "birth_year", "county"],
    identifiers=["student_id", "ssn"],
)

wages = sw.Table(
    "dol_wages",
    grain=sw.PerPeriod("student", periods=YEARS, presence=0.78),
    carry=["education"],
    identifiers=["ssn"],
    coverage=0.72,  # not everyone shows up in the wage file
    columns={
        # Declared structure. Without this the columns would be independent
        # and stage 2 would have nothing to learn.
        "annual_wage": sw.Conditional(
            "education",
            {
                "HS diploma": sw.Normal(41_000, 12_000, low=0),
                "GED": sw.Normal(33_000, 10_000, low=0),
                "no credential": sw.Normal(27_000, 9_000, low=0),
            },
        ),
        "employer_naics": sw.Choice(["44", "62", "72", "23", "54"]),
    },
)

dmv = sw.Table(
    "dmv_records",
    grain=sw.PerEntity("student"),
    carry=["birth_year", "county"],
    identifiers=["ssn"],
    coverage=0.61,
    columns={"license_class": sw.Choice(["D", "DJ", "E"], [0.8, 0.15, 0.05])},
)

schema = sw.Schema(entities=[students], tables=[k12, wages, dmv], seed=42)

pipeline = sw.Pipeline(
    schema,
    # Smooth the declared education-wage relationship into something less
    # obviously parametric than the normals that produced it.
    synthesizer=sw.CARTSynthesizer(
        ["annual_wage"],
        tables=["dol_wages"],
        predictors=["education"],
        fit_cap=sw.modeled(50_000, "illustrative cap for this example"),
    ),
    noiser=sw.Noise(
        {
            "dol_wages": {
                "annual_wage": [
                    sw.Missing(
                        sw.modeled(0.08, "employment gaps; shaped after a BLS CPS figure, unverified here")
                    )
                ]
            },
            "dmv_records": {
                # A scanned form: digits get read wrong.
                "birth_year": [sw.OCR(sw.modeled(0.018, "cross-system DOB divergence, unverified here"))]
            },
            "k12_cohort": {
                "ssn": [
                    sw.Missing(
                        sw.modeled(0.021, "share without an SSN; shaped after a DHS OHSS figure, unverified here")
                    )
                ]
            },
        }
    ),
)

result = pipeline.run()

print("Tables produced")
print(result.summary().to_string(index=False))

print("\nK12 cohort")
print(result["k12_cohort"].head(4).to_string(index=False))

print("\nDOL wages")
print(result["dol_wages"].head(4).to_string(index=False))

# The linking guarantee, checked rather than asserted.
k12_ssns = set(result["k12_cohort"]["ssn"].dropna())
dmv_ssns = set(result["dmv_records"]["ssn"])
print(f"\nDMV records whose SSN also appears in the K12 file: {len(dmv_ssns & k12_ssns):,}")
print(f"DMV records with no K12 match: {len(dmv_ssns - k12_ssns):,} (all from suppressed SSNs)")

# Declared structure survived generation, synthesis, and noise.
by_education = result["dol_wages"].groupby("education")["annual_wage"].mean().round(0)
print("\nMean wage by education, after all three stages")
print(by_education.to_string())

print("\nValues nobody has justified yet")
for path, tagged in sorted(result.unjustified().items()):
    print(f"  {path:<45} {tagged.value!s:<10} {tagged.note or ''}")
