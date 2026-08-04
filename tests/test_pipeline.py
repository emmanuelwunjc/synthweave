"""End-to-end behavior at the public API.

Grouped by the property being proved, not by the module doing the work.
"""

from __future__ import annotations

import re
import warnings

import pandas as pd
import pytest

import invariants
import synthweave as sw


# --- schema conformance and grain ------------------------------------------


def test_per_entity_grain_gives_one_row_per_entity(schema):
    result = sw.Pipeline(schema).run()
    assert len(result["roster"]) == 400


def test_per_period_grain_multiplies_by_period_count(schema):
    result = sw.Pipeline(schema).run()
    assert len(result["wages"]) == 400 * 3


def test_declared_columns_and_carried_attributes_appear(schema):
    result = sw.Pipeline(schema).run()
    assert set(result["roster"].columns) == {"student_id", "tax_id", "education", "birth_year"}
    assert set(result["wages"].columns) == {"tax_id", "period", "education", "wage"}


def test_reserved_bookkeeping_columns_never_reach_the_user(schema):
    result = sw.Pipeline(schema).run()
    for frame in result.tables.values():
        assert not [c for c in frame.columns if c.startswith("_sw_")]


def test_event_grain_varies_row_count_per_entity(people):
    table = sw.Table(
        "visits",
        grain=sw.PerEvent("person", low=1, high=6),
        identifiers=["student_id"],
        columns={"amount": sw.Uniform(0, 100)},
    )
    result = sw.Pipeline(sw.Schema(entities=[people], tables=[table], seed=1)).run()
    counts = result["visits"].groupby("student_id").size()
    assert counts.min() >= 1
    assert counts.max() <= 5
    assert counts.nunique() > 1


def test_presence_below_one_produces_an_unbalanced_panel(people):
    table = sw.Table(
        "panel",
        grain=sw.PerPeriod("person", periods=[2020, 2021, 2022, 2023], presence=0.5),
        identifiers=["tax_id"],
    )
    result = sw.Pipeline(sw.Schema(entities=[people], tables=[table], seed=3)).run()
    per_person = result["panel"].groupby("tax_id").size()
    assert per_person.nunique() > 1
    assert 0.35 < len(result["panel"]) / (400 * 4) < 0.65


def test_coverage_limits_entities_in_a_table(people, roster):
    partial = sw.Table(
        "subset", grain=sw.PerEntity("person"), identifiers=["student_id"], coverage=0.5
    )
    schema = sw.Schema(entities=[people], tables=[roster, partial], seed=7)
    result = sw.Pipeline(schema).run()

    assert 0.4 < len(result["subset"]) / 400 < 0.6
    # Coverage gaps are real gaps: the subset is a strict subset of the roster.
    assert set(result["subset"]["student_id"]) < set(result["roster"]["student_id"])


def test_carrying_a_leaf_draws_its_whole_transitive_dependency_chain():
    """education -> sector -> wage, carrying only the leaf.

    A direct-dependency check draws sector (wage's dependency) but not
    education (sector's dependency), so wage's Conditional draw fails
    looking for a column that validation already declared present.
    """
    person = sw.Entity(
        "person",
        count=20,
        attributes={
            "education": sw.Choice(["HS", "College"], [0.5, 0.5]),
            "sector": sw.Conditional(
                "education", {"HS": sw.Constant("retail"), "College": sw.Constant("tech")}
            ),
            "wage": sw.Conditional(
                "sector", {"retail": sw.Constant(30_000), "tech": sw.Constant(90_000)}
            ),
        },
        identifiers=[sw.Identifier("tax_id", prefix="TIN", digits=9)],
    )
    table = sw.Table("t", grain=sw.PerEntity("person"), carry=["wage"], identifiers=["tax_id"])
    result = sw.Pipeline(sw.Schema(entities=[person], tables=[table], seed=1)).run()

    assert set(result["t"].columns) == {"tax_id", "wage"}
    assert set(result["t"]["wage"]) <= {30_000, 90_000}


# --- entity consistency and relational integrity ---------------------------


def test_entity_attributes_are_identical_wherever_the_entity_appears(schema):
    result = sw.Pipeline(schema).run()
    roster = result["roster"][["tax_id", "education"]].drop_duplicates()
    wages = result["wages"][["tax_id", "education"]].drop_duplicates()
    merged = roster.merge(wages, on="tax_id", suffixes=("_roster", "_wages"))

    assert len(merged) == 400
    assert (merged["education_roster"] == merged["education_wages"]).all()


def test_the_same_identifier_links_across_tables(schema):
    result = sw.Pipeline(schema).run()
    assert set(result["wages"]["tax_id"]) == set(result["roster"]["tax_id"])


def test_identifier_kinds_are_independent(schema):
    """A person's two identifiers must not be derivable from each other.

    Comparing suffixes for inequality would pass even for `tax_id =
    student_id + 1`: the two numeric parts almost never land on the exact
    same digits, yet one is trivially derivable from the other. Checking
    correlation between the two numeric sequences catches a fixed
    relationship like that, not just literal equality.
    """
    roster = sw.Pipeline(schema).run()["roster"]
    student_nums = roster["student_id"].str[3:].astype(int)
    tax_nums = roster["tax_id"].str[3:].astype(int)
    assert abs(student_nums.corr(tax_nums)) < 0.1


def test_identifiers_are_unique_per_entity(schema):
    roster = sw.Pipeline(schema).run()["roster"]
    assert roster["student_id"].is_unique
    assert roster["tax_id"].is_unique


def test_a_table_listing_no_identifiers_is_unlinkable(people):
    anonymous = sw.Table(
        "anonymous", grain=sw.PerEntity("person"), carry=["education"], identifiers=[]
    )
    result = sw.Pipeline(sw.Schema(entities=[people], tables=[anonymous], seed=1)).run()
    assert set(result["anonymous"].columns) == {"education"}


def test_identifiers_survive_a_separate_run_of_a_different_schema(people, roster):
    """The linking guarantee holds across runs, not just across tables in one run."""
    first = sw.Pipeline(sw.Schema(entities=[people], tables=[roster], seed=42)).run()

    later = sw.Table("later", grain=sw.PerEntity("person"), identifiers=["tax_id"])
    second = sw.Pipeline(sw.Schema(entities=[people], tables=[later], seed=42)).run()

    assert set(first["roster"]["tax_id"]) == set(second["later"]["tax_id"])


# --- determinism, order and chunk invariance -------------------------------


def test_same_seed_reproduces_the_run_exactly(schema):
    a = sw.Pipeline(schema).run()
    b = sw.Pipeline(schema).run()
    for name in a.tables:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_a_different_seed_changes_the_data(people, roster):
    a = sw.Pipeline(sw.Schema(entities=[people], tables=[roster], seed=1)).run()
    b = sw.Pipeline(sw.Schema(entities=[people], tables=[roster], seed=2)).run()
    assert not a["roster"]["tax_id"].equals(b["roster"]["tax_id"])


@pytest.mark.parametrize("chunk_size", [7, 50, 1_000, 10_000])
def test_chunk_size_cannot_change_the_output(schema, chunk_size):
    """The load-bearing test for the chunked design.

    chunk_size is a memory knob. If it could change a value, every claim about
    scaling to tens of millions of rows would be unsound.
    """
    reference = sw.Pipeline(schema, chunk_size=100_000).run()
    other = sw.Pipeline(schema, chunk_size=chunk_size).run()
    for name in reference.tables:
        pd.testing.assert_frame_equal(reference[name], other[name])


def test_chunk_size_cannot_change_output_with_every_stage_active(schema):
    kwargs = dict(
        synthesizer=sw.CARTSynthesizer(
            ["wage"], tables=["wages"], predictors=["education"], fit_cap=300
        ),
        noiser=sw.Noise({"wages": {"education": [sw.Typo(0.2)]}}),
    )
    a = sw.Pipeline(schema, chunk_size=113, **kwargs).run()
    b = sw.Pipeline(schema, chunk_size=100_000, **kwargs).run()
    pd.testing.assert_frame_equal(a["wages"], b["wages"])


# --- stage skipping ---------------------------------------------------------


def test_generator_alone_is_a_valid_pipeline(schema):
    result = sw.Pipeline(schema, linker=None).run()
    assert "tax_id" not in result["roster"].columns
    assert len(result["roster"]) == 400


def test_skipping_noise_leaves_data_clean(schema):
    result = sw.Pipeline(schema).run()
    assert result["roster"]["education"].isin(["HS", "College"]).all()


# --- noise ------------------------------------------------------------------


def test_missingness_lands_near_the_configured_rate(many_people):
    table = sw.Table(
        "records", grain=sw.PerEntity("person"), columns={"amount": sw.Uniform(0, 100)}
    )
    result = sw.Pipeline(
        sw.Schema(entities=[many_people], tables=[table], seed=42),
        noiser=sw.Noise({"records": {"amount": [sw.Missing(0.3)]}}),
    ).run()
    assert 0.29 < result["records"]["amount"].isna().mean() < 0.31


def test_missingness_rate_can_vary_by_row(many_people):
    """Differential nonresponse: the rate itself conditions on another column.

    A flat `rate` can only express MCAR, but real nonresponse correlates with
    other attributes. `rate` therefore also accepts a vectorized function of
    the chunk, the same escape hatch `Sequential.fn` already uses: it is
    handed the frame and returns one rate per row.
    """
    table = sw.Table(
        "survey",
        grain=sw.PerEntity("person"),
        carry=["education"],
        columns={"amount": sw.Uniform(0, 100)},
    )
    result = sw.Pipeline(
        sw.Schema(entities=[many_people], tables=[table], seed=42),
        noiser=sw.Noise(
            {
                "survey": {
                    "amount": [sw.Missing(lambda f: 0.05 + 0.25 * (f["education"] == "HS"))]
                }
            }
        ),
    ).run()["survey"]

    missing = result.groupby("education")["amount"].apply(lambda s: s.isna().mean())
    assert 0.28 < missing["HS"] < 0.32
    assert 0.04 < missing["College"] < 0.06


def test_a_row_varying_rate_stays_deterministic_and_chunk_invariant(people):
    """A per-row rate must not weaken the two standing guarantees.

    The corruption decision still routes through `unit(key, seed, salt) <
    rate`, so the rate function only chooses the threshold; it never draws.
    Chunking is the real risk here: the function is handed each chunk
    separately, so a rate that depended on chunk-level state (a mean, a
    position) rather than on the row would silently break this.

    A row-wise function alone cannot show that, because it has no chunk-level
    state to get wrong. So this test also runs an aggregate rate function, the
    exact shape the docstring warns about, and requires it to be refused.
    """
    table = sw.Table(
        "survey",
        grain=sw.PerEntity("person"),
        carry=["education"],
        columns={"amount": sw.Uniform(0, 100)},
    )
    schema = sw.Schema(entities=[people], tables=[table], seed=3)
    noiser = sw.Noise(
        {"survey": {"amount": [sw.Missing(lambda f: 0.05 + 0.25 * (f["education"] == "HS"))]}}
    )
    invariants.assert_deterministic(schema, noiser=noiser)
    invariants.assert_chunk_invariant(schema, noiser=noiser)

    aggregate = sw.Noise(
        {"survey": {"amount": [sw.Missing(lambda f: float((f["education"] == "HS").mean()))]}}
    )
    with pytest.raises(ValueError, match=r"survey\.amount\.missing"):
        sw.Pipeline(schema, chunk_size=7, noiser=aggregate).run()


def test_a_rate_function_returning_the_wrong_length_fails_loudly(people):
    """One rate per row, or an error. Broadcasting is the trap.

    `np.asarray(fn(chunk))` accepts a length the chunk never had, and numpy
    broadcasts a length-1 result over every row without complaint. That reads
    as a working per-row rate while being a flat one, so the mistake survives
    to the output rather than to a traceback.
    """
    table = sw.Table(
        "survey",
        grain=sw.PerEntity("person"),
        carry=["education"],
        columns={"amount": sw.Uniform(0, 100)},
    )
    schema = sw.Schema(entities=[people], tables=[table], seed=3)
    noiser = sw.Noise({"survey": {"amount": [sw.Missing(lambda f: [0.3])]}})
    with pytest.raises(ValueError, match=r"survey\.amount\.missing"):
        sw.Pipeline(schema, chunk_size=7, noiser=noiser).run()


def test_a_row_varying_rate_outside_zero_to_one_fails_loudly(people):
    """A flat rate is range-checked at construction. A function cannot be.

    Without a check after the call, a function returning 1.5 silently
    corrupts every row and one returning -0.1 silently corrupts none. Both
    produce a plausible-looking table rather than an error, which is the
    failure mode this library treats as the expensive one.
    """
    table = sw.Table(
        "survey",
        grain=sw.PerEntity("person"),
        carry=["education"],
        columns={"amount": sw.Uniform(0, 100)},
    )
    schema = sw.Schema(entities=[people], tables=[table], seed=3)
    noiser = sw.Noise(
        {"survey": {"amount": [sw.Missing(lambda f: 0.05 + 2.0 * (f["education"] == "HS"))]}}
    )
    with pytest.raises(ValueError, match=r"survey\.amount\.missing"):
        sw.Pipeline(schema, noiser=noiser).run()


def test_realized_noise_rate_is_reported(many_people):
    table = sw.Table("roster", grain=sw.PerEntity("person"), carry=["education"])
    result = sw.Pipeline(
        sw.Schema(entities=[many_people], tables=[table], seed=42),
        noiser=sw.Noise({"roster": {"education": [sw.Typo(0.2)]}}),
    ).run()
    realized = result.metadata["roster"]["noise"]["realized"]["education.typo"]
    assert 0.19 < realized < 0.21


def test_typos_corrupt_only_the_targeted_share(many_people):
    table = sw.Table("roster", grain=sw.PerEntity("person"), carry=["education"])
    schema = sw.Schema(entities=[many_people], tables=[table], seed=42)
    clean = sw.Pipeline(schema).run()["roster"]
    dirty = sw.Pipeline(
        schema, noiser=sw.Noise({"roster": {"education": [sw.Typo(0.2)]}})
    ).run()["roster"]
    assert 0.19 < (clean["education"] != dirty["education"]).mean() < 0.21


def test_ocr_noise_applies_to_identifier_digits(schema):
    dirty = sw.Pipeline(
        schema, noiser=sw.Noise({"roster": {"birth_year": [sw.OCR(1.0)]}})
    ).run()["roster"]
    # Every birth year contains a digit with a visual twin, so all get hit.
    assert dirty["birth_year"].astype(str).str.contains("[A-Za-z]").any()


def test_noise_on_an_unknown_column_fails_loudly(schema):
    with pytest.raises(KeyError, match="nonexistent"):
        sw.Pipeline(schema, noiser=sw.Noise({"roster": {"nonexistent": [sw.Typo(0.1)]}})).run()


def test_an_identifier_column_can_be_noised_on_purpose(schema):
    """Locks the stage order: linking must precede noise.

    Identifier columns are created by the linker. If noise ran first they
    would not exist yet and this config would raise, which is exactly the
    bug this test exists to catch. Dirtying an identifier is the normal way
    to build a record-linkage benchmark, so it has to be reachable.
    """
    result = sw.Pipeline(
        schema, noiser=sw.Noise({"roster": {"tax_id": [sw.Missing(0.25)]}})
    ).run()
    assert 0.15 < result["roster"]["tax_id"].isna().mean() < 0.35


def test_identifiers_stay_clean_unless_the_noise_config_names_them(schema):
    """The other half of the contract: reachable, but never touched by default."""
    result = sw.Pipeline(
        schema, noiser=sw.Noise({"roster": {"education": [sw.Typo(0.5)]}})
    ).run()
    assert result["roster"]["tax_id"].notna().all()
    assert result["roster"]["tax_id"].str.fullmatch(r"TIN\d{9}").all()


def test_a_pipeline_run_raises_no_pandas_copy_warnings(schema):
    """Stages must own a chunk before writing to it.

    A chunk can be a view onto a larger frame. Writing to one without copying
    warns and, worse, may not land. `own()` is the guard; this test is what
    notices if a stage stops using it.
    """
    copy_warning = getattr(pd.errors, "SettingWithCopyWarning", None)
    if copy_warning is None:  # pandas 3 copy-on-write removed the hazard
        pytest.skip("pandas no longer defines SettingWithCopyWarning")

    with warnings.catch_warnings():
        warnings.simplefilter("error", copy_warning)
        sw.Pipeline(
            schema,
            synthesizer=sw.CARTSynthesizer(
                ["wage"], tables=["wages"], predictors=["education"]
            ),
            noiser=sw.Noise({"wages": {"wage": [sw.Missing(0.1)]}}),
        ).run()


def test_noise_can_target_any_column_regardless_of_meaning(schema):
    """Not limited to the field types a fixed schema anticipated."""
    result = sw.Pipeline(
        schema, noiser=sw.Noise({"wages": {"period": [sw.Missing(0.5)]}})
    ).run()
    assert 0.45 < result["wages"]["period"].isna().mean() < 0.55


# --- provenance -------------------------------------------------------------


def test_library_defaults_are_flagged_as_unjustified(schema):
    result = sw.Pipeline(
        schema,
        synthesizer=sw.CARTSynthesizer(["wage"], tables=["wages"], predictors=["education"]),
    ).run()
    unjustified = result.unjustified()
    assert any("fit_cap" in path for path in unjustified)


def test_a_user_supplied_value_is_not_flagged(people):
    table = sw.Table(
        "roster", grain=sw.PerEntity("person"), identifiers=["tax_id"], coverage=sw.user(0.8)
    )
    result = sw.Pipeline(sw.Schema(entities=[people], tables=[table], seed=1)).run()
    assert "roster.coverage" not in result.unjustified()


def test_a_cited_value_records_its_source(people):
    table = sw.Table(
        "roster",
        grain=sw.PerEntity("person"),
        identifiers=["tax_id"],
        coverage=sw.cited(0.82, "NCES Table 219.10, retrieved 2026-07-29"),
    )
    result = sw.Pipeline(sw.Schema(entities=[people], tables=[table], seed=1)).run()
    entry = result.provenance.entries["roster.coverage"]
    assert entry.origin == "cited"
    assert "NCES" in entry.note


def test_provenance_exports_as_a_frame(schema):
    result = sw.Pipeline(schema).run()
    frame = result.provenance.to_frame()
    assert list(frame.columns) == ["path", "value", "origin", "note"]
    assert len(frame) > 0


def test_a_cited_value_must_name_its_source():
    with pytest.raises(ValueError, match="note"):
        sw.Tagged(0.5, "cited")


# --- validation -------------------------------------------------------------


def test_a_table_carrying_an_unknown_attribute_is_rejected(people):
    table = sw.Table("bad", grain=sw.PerEntity("person"), carry=["height"])
    with pytest.raises(sw.SchemaError, match="height"):
        sw.Pipeline(sw.Schema(entities=[people], tables=[table], seed=1))


def test_a_table_asking_for_an_unknown_identifier_is_rejected(people):
    table = sw.Table("bad", grain=sw.PerEntity("person"), identifiers=["passport"])
    with pytest.raises(sw.SchemaError, match="passport"):
        sw.Pipeline(sw.Schema(entities=[people], tables=[table], seed=1))


def test_a_table_carrying_the_same_attribute_twice_is_rejected(people):
    table = sw.Table(
        "bad", grain=sw.PerEntity("person"), carry=["education", "education"]
    )
    with pytest.raises(sw.SchemaError, match="duplicate"):
        sw.Pipeline(sw.Schema(entities=[people], tables=[table], seed=1))


def test_a_table_asking_for_the_same_identifier_twice_is_rejected(people):
    table = sw.Table(
        "bad", grain=sw.PerEntity("person"), identifiers=["tax_id", "tax_id"]
    )
    with pytest.raises(sw.SchemaError, match="duplicate"):
        sw.Pipeline(sw.Schema(entities=[people], tables=[table], seed=1))


def test_uniform_with_high_not_above_low_is_rejected():
    """Integer raises via _hash.integers on the same condition; Uniform did
    not, and silently returned values descending into (high, low] instead."""
    with pytest.raises(ValueError, match="high > low"):
        sw.Uniform(10, 5)


def test_an_unknown_entity_is_rejected(people):
    table = sw.Table("bad", grain=sw.PerEntity("ghost"))
    with pytest.raises(sw.SchemaError, match="ghost"):
        sw.Pipeline(sw.Schema(entities=[people], tables=[table], seed=1))


def test_a_rule_dependency_cycle_is_rejected(people):
    table = sw.Table(
        "bad",
        grain=sw.PerEntity("person"),
        columns={
            "a": sw.Conditional("b", {1: sw.Constant(1)}, default=sw.Constant(0)),
            "b": sw.Conditional("a", {1: sw.Constant(1)}, default=sw.Constant(0)),
        },
    )
    with pytest.raises(sw.SchemaError, match="cycle"):
        sw.Pipeline(sw.Schema(entities=[people], tables=[table], seed=1))


def test_validation_runs_before_any_generation_work(people):
    """Construction fails, so no rows are ever produced for a bad config."""
    table = sw.Table("bad", grain=sw.PerEntity("person"), carry=["height"])
    schema = sw.Schema(entities=[people], tables=[table], seed=1)
    with pytest.raises(sw.SchemaError):
        sw.Pipeline(schema)


def test_duplicate_table_names_are_rejected(people, roster):
    with pytest.raises(sw.SchemaError, match="duplicate"):
        sw.Pipeline(sw.Schema(entities=[people], tables=[roster, roster], seed=1))


def test_reserved_column_names_are_rejected(people):
    table = sw.Table("bad", grain=sw.PerEntity("person"), columns={"_sw_row": sw.Constant(1)})
    with pytest.raises(sw.SchemaError, match="reserved"):
        sw.Pipeline(sw.Schema(entities=[people], tables=[table], seed=1))


def test_a_nan_choice_weight_is_rejected_instead_of_collapsing_the_column():
    """A NaN passes `< 0` and `<= 0` checks silently: every comparison is False.

    Weights derived from `value_counts(normalize=True).reindex(categories)`
    produce exactly this for any category absent from the data, so this is an
    ordinary input, not a contrived one.
    """
    person = sw.Entity(
        "person",
        count=20,
        attributes={"education": sw.Choice(["HS", "College"], [float("nan"), 1.0])},
    )
    table = sw.Table("t", grain=sw.PerEntity("person"), carry=["education"])
    with pytest.raises(ValueError, match="finite"):
        sw.Pipeline(sw.Schema(entities=[person], tables=[table], seed=1)).run()


def test_an_infinite_choice_weight_is_rejected_instead_of_collapsing_the_column():
    person = sw.Entity(
        "person",
        count=20,
        attributes={"education": sw.Choice(["HS", "College"], [float("inf"), 1.0])},
    )
    table = sw.Table("t", grain=sw.PerEntity("person"), carry=["education"])
    with pytest.raises(ValueError, match="finite"):
        sw.Pipeline(sw.Schema(entities=[person], tables=[table], seed=1)).run()


# --- multiple entities ------------------------------------------------------
#
# Every test above this line uses a single entity. These drive the cross-entity
# paths: separate identifier spaces, separate attribute draws, and a schema
# whose entity order should not matter.


@pytest.fixture
def two_entity_schema() -> sw.Schema:
    """Two entities that collide on purpose.

    Same attribute name, same identifier tag, same prefix, same digit count.
    Anything that keys on those names alone rather than on the entity will
    show up as one entity's data appearing in the other's table.
    """
    make = lambda name: sw.Entity(  # noqa: E731
        name,
        count=500,
        attributes={"region": sw.Choice(["N", "S"], [0.5, 0.5])},
        identifiers=[sw.Identifier("id", prefix="X", digits=9)],
    )
    return sw.Schema(
        entities=[make("person"), make("firm")],
        tables=[
            sw.Table("people", grain=sw.PerEntity("person"), carry=["region"], identifiers=["id"]),
            sw.Table("firms", grain=sw.PerEntity("firm"), carry=["region"], identifiers=["id"]),
        ],
        seed=42,
    )


def test_two_entities_do_not_share_an_identifier_space(two_entity_schema):
    """Invariant 4 read the other way: identity must not merge across entities."""
    result = sw.Pipeline(two_entity_schema).run()
    shared = set(result["people"]["id"]) & set(result["firms"]["id"])
    assert not shared, f"{len(shared)} identifier(s) refer to both a person and a firm"


def test_two_entities_draw_their_attributes_independently(two_entity_schema):
    result = sw.Pipeline(two_entity_schema).run()
    assert list(result["people"]["region"][:50]) != list(result["firms"]["region"][:50])


def test_entity_order_in_the_schema_does_not_change_the_output(two_entity_schema):
    """Invariant 3 at schema level. Reordering config is not a data change."""
    forward = sw.Pipeline(two_entity_schema).run()
    reversed_schema = sw.Schema(
        entities=list(reversed(two_entity_schema.entities)),
        tables=two_entity_schema.tables,
        seed=two_entity_schema.seed,
    )
    backward = sw.Pipeline(reversed_schema).run()
    for name in forward.tables:
        pd.testing.assert_frame_equal(backward[name], forward[name])


def test_a_two_entity_schema_is_chunk_invariant(two_entity_schema):
    invariants.assert_chunk_invariant(two_entity_schema)


# --- identifier tag collisions ----------------------------------------------


def test_an_identifier_tag_cannot_overwrite_a_table_column():
    person = sw.Entity(
        "person",
        count=50,
        attributes={"education": sw.Choice(["HS", "College"], [0.5, 0.5])},
        identifiers=[sw.Identifier("wage", prefix="W", digits=9)],
    )
    table = sw.Table(
        "t",
        grain=sw.PerEntity("person"),
        carry=["education"],
        identifiers=["wage"],
        columns={"wage": sw.Normal(50_000, 5_000, low=0)},
    )
    with pytest.raises(sw.SchemaError, match="wage"):
        sw.Pipeline(sw.Schema(entities=[person], tables=[table], seed=1)).run()


def test_an_identifier_tag_cannot_overwrite_a_carried_attribute():
    person = sw.Entity(
        "person",
        count=50,
        attributes={"education": sw.Choice(["HS", "College"], [0.5, 0.5])},
        identifiers=[sw.Identifier("education", prefix="E", digits=9)],
    )
    table = sw.Table(
        "t", grain=sw.PerEntity("person"), carry=["education"], identifiers=["education"]
    )
    with pytest.raises(sw.SchemaError, match="education"):
        sw.Pipeline(sw.Schema(entities=[person], tables=[table], seed=1)).run()


# --- identifier digit extremes ----------------------------------------------


def _one_entity(**identifier_kwargs) -> sw.Schema:
    person = sw.Entity(
        "person",
        count=50,
        attributes={"a": sw.Constant("x")},
        identifiers=[sw.Identifier("id", **identifier_kwargs)],
    )
    return sw.Schema(
        entities=[person],
        tables=[sw.Table("t", grain=sw.PerEntity("person"), identifiers=["id"])],
        seed=1,
    )


def test_a_digit_count_too_small_for_the_population_is_rejected():
    """The birthday bound, not the raw keyspace, decides what is too narrow."""
    with pytest.raises(sw.SchemaError, match="share an identifier"):
        sw.Pipeline(_one_entity(prefix="I", digits=1)).run()


def test_identifiers_all_have_the_requested_width():
    values = sw.Pipeline(_one_entity(prefix="I", digits=18)).run()["t"]["id"]
    widths = {len(v) - 1 for v in values}
    assert widths == {18}, f"asked for 18 digits, got widths {sorted(widths)}"


def test_the_recommended_digit_count_is_the_tightest_one_that_works():
    """The narrowest keyspace whose collision expectation stays under one is
    population**2 / 2, so the recommendation should be exactly that value's
    digit count. A prior version added one digit too many: still usable, so
    only checking that `Identifier` accepts the number would have missed it."""
    population = 400_000
    person = sw.Entity(
        "person", count=population, attributes={"a": sw.Constant("x")},
        identifiers=[sw.Identifier("id", prefix="I", digits=1)],
    )
    schema = sw.Schema(
        entities=[person],
        tables=[sw.Table("t", grain=sw.PerEntity("person"), identifiers=["id"])],
        seed=1,
    )
    with pytest.raises(sw.SchemaError, match=r"Use digits=(\d+) or more") as exc:
        sw.Pipeline(schema)
    needed = int(re.search(r"Use digits=(\d+) or more", str(exc.value)).group(1))
    assert needed == len(str(population * population // 2))
    sw.Identifier("id", prefix="I", digits=needed)  # must not raise


def test_a_population_needing_more_than_18_digits_says_so_instead_of_recommending_one():
    """Past ~1.4 billion entities the birthday-bound-safe width exceeds
    MAX_DIGITS, so recommending it would send a user straight into a second,
    unreachable error."""
    person = sw.Entity(
        "person", count=2_000_000_000, attributes={"a": sw.Constant("x")},
        identifiers=[sw.Identifier("id", prefix="I", digits=18)],
    )
    schema = sw.Schema(
        entities=[person],
        tables=[sw.Table("t", grain=sw.PerEntity("person"), identifiers=["id"])],
        seed=1,
    )
    with pytest.raises(sw.SchemaError, match="past the 18-digit limit") as exc:
        sw.Pipeline(schema)
    assert "Use digits=" not in str(exc.value)


def test_a_digit_count_past_the_hash_width_is_rejected():
    """Rejected where it is declared, before anything runs."""
    with pytest.raises(ValueError, match="digits must be at most 18"):
        sw.Identifier("id", prefix="I", digits=20)


# --- empty output -----------------------------------------------------------


def test_a_table_that_emits_no_rows_still_has_its_columns():
    """Column order matters too: it is what `run_to(format="csv")` writes as
    the header, so a set comparison would miss a column landing in the wrong
    position even though its name is still present."""
    person = sw.Entity(
        "person",
        count=5,
        attributes={"education": sw.Choice(["HS", "College"], [0.6, 0.4])},
        identifiers=[sw.Identifier("tax_id")],
    )
    table = sw.Table(
        "t",
        grain=sw.PerEntity("person"),
        carry=["education"],
        identifiers=["tax_id"],
        coverage=0.001,
    )
    schema = sw.Schema(entities=[person], tables=[table], seed=4)
    empty = sw.Pipeline(schema).run()["t"]

    non_empty_table = sw.Table(
        "t", grain=sw.PerEntity("person"), carry=["education"], identifiers=["tax_id"]
    )
    non_empty = sw.Pipeline(sw.Schema(entities=[person], tables=[non_empty_table], seed=4)).run()["t"]

    assert len(empty) == 0
    assert list(empty.columns) == list(non_empty.columns)


# --- edge cases that turned out to be sound ---------------------------------


def test_event_grain_allows_entities_with_no_events(people):
    """PerEvent(low=0) means some entities genuinely never appear."""
    table = sw.Table("ev", grain=sw.PerEvent("person", low=0, high=3), identifiers=["tax_id"])
    result = sw.Pipeline(sw.Schema(entities=[people], tables=[table], seed=4)).run()["ev"]

    assert result["tax_id"].nunique() < 400
    assert len(result) > 0


def test_typos_leave_missing_values_alone(many_people):
    """Missing runs first, so Typo must skip nulls rather than stringify them."""
    noiser = sw.Noise({"roster": {"education": [sw.Missing(0.5), sw.Typo(0.9)]}})
    roster = sw.Table("roster", grain=sw.PerEntity("person"), carry=["education"])
    schema = sw.Schema(entities=[many_people], tables=[roster], seed=1)

    values = sw.Pipeline(schema, noiser=noiser).run()["roster"]["education"]
    # pd.notna, not `is not None`: pandas 3 spells a null in a text column as
    # float nan, which `is not None` would let through into the `in` check below.
    written = {v for v in values if pd.notna(v)}
    assert not any("None" in v for v in written)


def test_typo_corrupts_non_ascii_values_without_breaking(people):
    """Character indexing must be by character, not by byte."""
    entity = sw.Entity(
        "person",
        count=200,
        attributes={"name": sw.Choice(["北京市", "Ünüver", "Ωμέγα"], [0.34, 0.33, 0.33])},
        identifiers=[sw.Identifier("tax_id")],
    )
    table = sw.Table("t", grain=sw.PerEntity("person"), carry=["name"])
    result = sw.Pipeline(
        sw.Schema(entities=[entity], tables=[table], seed=1),
        noiser=sw.Noise({"t": {"name": [sw.Typo(0.5)]}}),
    ).run()["t"]

    assert result["name"].notna().all()
    assert (~result["name"].isin(["北京市", "Ünüver", "Ωμέγα"])).any()


def test_sequential_derives_a_column_from_another(people):
    table = sw.Table(
        "t",
        grain=sw.PerEntity("person"),
        carry=["birth_year"],
        columns={"age": sw.Sequential("birth_year", lambda year: 2026 - year)},
    )
    schema = sw.Schema(entities=[people], tables=[table], seed=1)
    result = sw.Pipeline(schema).run()["t"]

    assert (result["birth_year"] + result["age"] == 2026).all()
    invariants.assert_chunk_invariant(schema)


# --- grain edge cases -------------------------------------------------------


def test_a_repeated_period_is_rejected():
    """One row per entity per period is the grain's whole promise."""
    with pytest.raises(ValueError, match="repeated period"):
        sw.PerPeriod("person", periods=[2020, 2020, 2021])


def test_period_order_does_not_change_the_panel(people):
    """Unsorted periods are a config nicety, not a data change.

    Rows come out in the order the periods were declared, so the comparison
    sorts first. Emission order is presentation; what invariant 3 protects is
    the value attached to each (entity, period) pair.
    """
    def run(periods):
        table = sw.Table(
            "panel", grain=sw.PerPeriod("person", periods=periods), identifiers=["tax_id"]
        )
        frame = sw.Pipeline(sw.Schema(entities=[people], tables=[table], seed=1)).run()["panel"]
        return frame.sort_values(["tax_id", "period"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(run([2022, 2020, 2021]), run([2020, 2021, 2022]))
