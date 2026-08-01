"""Structure sources, the fit cap, plugin resolution, and incremental output.

Still one seam. Plugin behavior is proved by registering a stub implementation
and running the pipeline, never by reaching into the registry.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import invariants
import synthweave as sw


def _wage_gap(frame: pd.DataFrame) -> float:
    """College mean minus HS mean. The relationship every source should carry."""
    means = frame.groupby("education")["wage"].mean()
    return means["College"] - means["HS"]


# --- structure sources ------------------------------------------------------


def test_declared_structure_survives_synthesis(schema):
    """The no-real-data path.

    Conditional rules put the education-wage relationship into the generated
    data, so the model has something real to learn. Without conditional rules
    there would be no structure and this gap would collapse to noise.
    """
    raw = sw.Pipeline(schema).run()["wages"]
    synthesized = sw.Pipeline(
        schema,
        synthesizer=sw.CARTSynthesizer(
            ["wage"], tables=["wages"], predictors=["education"], structure=sw.Declared()
        ),
    ).run()["wages"]

    assert _wage_gap(raw) > 15_000
    assert _wage_gap(synthesized) > 15_000


def test_independent_rules_leave_no_structure_to_learn(people):
    """The failure mode the structure source exists to prevent.

    With independent rules there is no relationship in the data, so a model
    fitted on it cannot invent one. Documented as a test so the reason
    `StructureSource` exists is not lost.
    """
    table = sw.Table(
        "flat",
        grain=sw.PerEntity("person"),
        carry=["education"],
        columns={"wage": sw.Normal(50_000, 10_000, low=0)},
    )
    schema = sw.Schema(entities=[people], tables=[table], seed=5)
    out = sw.Pipeline(
        schema,
        synthesizer=sw.CARTSynthesizer(["wage"], tables=["flat"], predictors=["education"]),
    ).run()["flat"]

    assert abs(_wage_gap(out)) < 5_000


def test_empirical_structure_is_learned_from_supplied_data(people):
    """The classic fit-on-real-data case."""
    real = pd.DataFrame(
        {
            "education": ["HS"] * 300 + ["College"] * 300,
            "wage": list(np.linspace(20_000, 30_000, 300))
            + list(np.linspace(80_000, 90_000, 300)),
        }
    )
    table = sw.Table(
        "earnings",
        grain=sw.PerEntity("person"),
        carry=["education"],
        columns={"wage": sw.Constant(0.0)},
    )
    out = sw.Pipeline(
        sw.Schema(entities=[people], tables=[table], seed=9),
        synthesizer=sw.CARTSynthesizer(
            ["wage"],
            tables=["earnings"],
            predictors=["education"],
            structure=sw.Empirical(real),
        ),
    ).run()["earnings"]

    # The generator wrote zeros. Every non-zero value came from the real data.
    assert (out["wage"] > 0).all()
    assert _wage_gap(out) > 40_000


def test_prior_structure_comes_from_published_aggregates(people):
    """The case where a user has statistics but no rows at all."""
    table = sw.Table(
        "survey",
        grain=sw.PerEntity("person"),
        columns={"tenure": sw.Constant("unknown")},
    )
    out = sw.Pipeline(
        sw.Schema(entities=[people], tables=[table], seed=11),
        synthesizer=sw.CARTSynthesizer(
            ["tenure"],
            tables=["survey"],
            structure=sw.Prior(marginals={"tenure": {"own": 0.65, "rent": 0.35}}, rows=5_000),
        ),
    ).run()["survey"]

    share_own = (out["tenure"] == "own").mean()
    assert 0.60 < share_own < 0.70


def test_a_structure_source_missing_a_column_fails_loudly(people):
    table = sw.Table("t", grain=sw.PerEntity("person"), columns={"x": sw.Constant(1)})
    real = pd.DataFrame({"something_else": [1, 2, 3]})
    with pytest.raises(KeyError, match="x"):
        sw.Pipeline(
            sw.Schema(entities=[people], tables=[table], seed=1),
            synthesizer=sw.CARTSynthesizer(["x"], tables=["t"], structure=sw.Empirical(real)),
        ).run()


# --- structure= shorthand coercion -------------------------------------


def test_structure_accepts_a_bare_dataframe_same_as_empirical(people):
    """A DataFrame handed to structure= is wrapped in Empirical automatically."""
    real = pd.DataFrame(
        {
            "education": ["HS"] * 300 + ["College"] * 300,
            "wage": list(np.linspace(20_000, 30_000, 300))
            + list(np.linspace(80_000, 90_000, 300)),
        }
    )
    table = sw.Table(
        "earnings",
        grain=sw.PerEntity("person"),
        carry=["education"],
        columns={"wage": sw.Constant(0.0)},
    )
    explicit = sw.Pipeline(
        sw.Schema(entities=[people], tables=[table], seed=9),
        synthesizer=sw.CARTSynthesizer(
            ["wage"], tables=["earnings"], predictors=["education"], structure=sw.Empirical(real)
        ),
    ).run()["earnings"]
    bare = sw.Pipeline(
        sw.Schema(entities=[people], tables=[table], seed=9),
        synthesizer=sw.CARTSynthesizer(
            ["wage"], tables=["earnings"], predictors=["education"], structure=real
        ),
    ).run()["earnings"]
    assert explicit.equals(bare)


def test_structure_accepts_a_bare_mapping_same_as_prior(people):
    """A dict handed to structure= is wrapped in Prior(marginals=...) automatically."""
    table = sw.Table(
        "survey", grain=sw.PerEntity("person"), columns={"tenure": sw.Constant("unknown")}
    )
    marginals = {"tenure": {"own": 0.65, "rent": 0.35}}
    explicit = sw.Pipeline(
        sw.Schema(entities=[people], tables=[table], seed=11),
        synthesizer=sw.CARTSynthesizer(
            ["tenure"], tables=["survey"], structure=sw.Prior(marginals=marginals)
        ),
    ).run()["survey"]
    bare = sw.Pipeline(
        sw.Schema(entities=[people], tables=[table], seed=11),
        synthesizer=sw.CARTSynthesizer(["tenure"], tables=["survey"], structure=marginals),
    ).run()["survey"]
    assert explicit.equals(bare)


def test_structure_resolves_a_registered_name(people):
    """The "structure" registry kind is actually resolved now, not dead weight.

    Empirical/Prior/Declared register themselves under "structure" but
    nothing looked names up in that registry before this fix, so a caller
    passing structure="empirical" silently got the string back rather than
    an Empirical instance. Proven here with a throwaway registered stub
    instead of "empirical" itself, so this test does not depend on how the
    built-in structure sources happen to be named.
    """

    class _Stub:
        def training_frame(self, table, ctx):
            return pd.DataFrame({"x": [1, 2, 3] * 100})

    sw.register("structure", "test-stub-structure-source")(_Stub)
    table = sw.Table("t", grain=sw.PerEntity("person"), columns={"x": sw.Constant(0)})
    result = sw.Pipeline(
        sw.Schema(entities=[people], tables=[table], seed=1),
        synthesizer=sw.CARTSynthesizer(
            ["x"], tables=["t"], structure="test-stub-structure-source"
        ),
    ).run()["t"]
    assert set(result["x"].unique()) <= {1, 2, 3}


def test_structure_by_name_needing_config_names_the_missing_argument():
    """`structure="empirical"`/`"prior"` can never resolve by bare name.

    `Empirical` requires `frame`, `Prior` requires `marginals`; a string has
    no channel to carry either. Resolving the name used to hit `resolve()`'s
    generic zero-arg instantiation and raise a bare `TypeError` ("missing 1
    required positional argument: 'frame'") with no hint of what to do
    instead. It should raise a `StructureConfigError` naming the source and
    telling the caller to pass a configured instance.
    """
    with pytest.raises(sw.StructureConfigError, match="empirical"):
        sw.CARTSynthesizer(["x"], structure="empirical")
    with pytest.raises(sw.StructureConfigError, match="prior"):
        sw.CARTSynthesizer(["x"], structure="prior")


# --- the fit cap ------------------------------------------------------------


def test_below_the_cap_every_row_is_used(schema):
    """A small table is fitted whole, as a single-table tool would do."""
    result = sw.Pipeline(
        schema,
        synthesizer=sw.CARTSynthesizer(
            ["wage"], tables=["wages"], predictors=["education"], fit_cap=10_000
        ),
    ).run()
    report = result.metadata["wages"]["synthesize"]
    assert report["fit_rows"] == 1_200
    assert report["sampled"] is False


def test_above_the_cap_the_fit_is_capped(schema):
    result = sw.Pipeline(
        schema,
        synthesizer=sw.CARTSynthesizer(
            ["wage"], tables=["wages"], predictors=["education"], fit_cap=250
        ),
    ).run()
    assert result.metadata["wages"]["synthesize"]["fit_rows"] == 250


def test_a_table_outside_the_synthesizer_scope_passes_through(schema):
    """The roster has no wage column, so naming only wages must leave it alone."""
    plain = sw.Pipeline(schema).run()["roster"]
    scoped = sw.Pipeline(
        schema,
        synthesizer=sw.CARTSynthesizer(["wage"], tables=["wages"], predictors=["education"]),
    ).run()["roster"]
    pd.testing.assert_frame_equal(plain, scoped)


# --- plugin resolution ------------------------------------------------------


def test_a_registered_custom_noiser_is_the_one_that_runs(schema):
    """Proves registration, resolution, and composition through the top seam."""

    @sw.register("noiser", "test-shout", overwrite=True)
    class Shout:
        def run(self, chunks, table, ctx):
            for chunk in chunks:
                if "education" in chunk.columns:
                    chunk["education"] = chunk["education"].str.upper()
                yield chunk

    result = sw.Pipeline(schema, noiser="test-shout").run()
    assert set(result["roster"]["education"]) == {"HS", "COLLEGE"}


def test_a_custom_stage_instance_can_be_passed_directly(schema):
    from synthweave.stages.base import own

    class Tagger:
        def run(self, chunks, table, ctx):
            for chunk in chunks:
                # A chunk may be a view, so a stage adding a column copies
                # first. This is the contract third-party stages follow.
                chunk = own(chunk)
                chunk["source"] = table.name
                yield chunk

    result = sw.Pipeline(schema, noiser=Tagger()).run()
    assert (result["roster"]["source"] == "roster").all()


def test_a_custom_generator_replaces_stage_one(people):
    @sw.register("generator", "test-fixed", overwrite=True)
    class FixedRows:
        def emit(self, table, ctx):
            yield pd.DataFrame(
                {"_sw_entity": ["person:0", "person:1"], "_sw_row": ["r0", "r1"], "x": [10, 20]}
            )

    table = sw.Table("t", grain=sw.PerEntity("person"), identifiers=["tax_id"])
    result = sw.Pipeline(
        sw.Schema(entities=[people], tables=[table], seed=1), generator="test-fixed"
    ).run()

    assert list(result["t"]["x"]) == [10, 20]
    # The linker still runs, so a custom generator composes with the rest.
    assert result["t"]["tax_id"].str.startswith("TIN").all()


def test_own_makes_a_view_safe_to_write_to():
    """`own` is part of the plugin contract, so it is tested directly.

    A stage receives whatever frame the previous stage yielded, which may be a
    view onto something larger. Writing to a view warns and the write may not
    land. Built-in stages happen to construct their own frames, so this never
    bites them; a third-party stage has no such guarantee, which is why the
    contract and this test exist.
    """
    from synthweave.stages.base import own

    frame = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]})
    view = frame[frame["a"] > 2]

    copy_warning = getattr(pd.errors, "SettingWithCopyWarning", None)
    if copy_warning is None:  # pandas 3 copy-on-write removed the hazard
        pytest.skip("pandas no longer defines SettingWithCopyWarning")

    with warnings.catch_warnings():
        warnings.simplefilter("error", copy_warning)
        owned = own(view)
        owned["c"] = "written"

    assert list(owned["c"]) == ["written", "written"]
    assert "c" not in frame.columns  # the original is left alone


def test_an_unknown_stage_name_fails_with_the_registered_options(schema):
    with pytest.raises(KeyError, match="no noiser named"):
        sw.Pipeline(schema, noiser="does-not-exist")


def test_registering_over_an_existing_name_is_refused():
    @sw.register("noiser", "test-collision", overwrite=True)
    class First:
        def run(self, chunks, table, ctx):
            yield from chunks

    with pytest.raises(ValueError, match="already registered"):

        @sw.register("noiser", "test-collision")
        class Second:
            def run(self, chunks, table, ctx):
                yield from chunks


def test_built_in_stages_are_discoverable():
    assert "rules" in sw.available("generator")
    assert "cart" in sw.available("synthesizer")
    assert "deterministic" in sw.available("linker")


# --- incremental output -----------------------------------------------------


def test_run_to_writes_each_table_to_disk(schema, tmp_path):
    result = sw.Pipeline(schema, chunk_size=97).run_to(tmp_path, format="csv")

    assert set(result.paths) == {"roster", "wages"}
    written = pd.read_csv(result.paths["wages"])
    assert len(written) == 1_200
    assert result.metadata["wages"]["output"]["rows"] == 1_200


def test_incremental_output_matches_the_in_memory_run(schema, tmp_path):
    in_memory = sw.Pipeline(schema).run()["roster"]
    sw.Pipeline(schema, chunk_size=31).run_to(tmp_path, format="csv")
    on_disk = pd.read_csv(tmp_path / "roster.csv")

    assert list(on_disk.columns) == list(in_memory.columns)
    assert list(on_disk["tax_id"]) == list(in_memory["tax_id"])


def test_run_to_keeps_provenance_without_materializing_tables(schema, tmp_path):
    result = sw.Pipeline(schema).run_to(tmp_path, format="csv")
    assert result.tables == {}
    assert len(result.provenance) > 0


# --- multi-column synthesis -------------------------------------------------
#
# Every test above synthesizes exactly one column. These drive the sequential
# visit order: column n is fitted on the predictors plus columns 0..n-1, each
# encoded on its own, each with its own donor pools.


def _multi(**overrides):
    """The synthesizer under test in this section."""
    kwargs = dict(
        columns=["sector", "wage", "hours", "tenure"],
        tables=["jobs"],
        predictors=["education"],
        structure=sw.Declared(),
    )
    kwargs.update(overrides)
    return sw.CARTSynthesizer(**kwargs)


def test_a_chain_of_declared_relationships_survives_multi_column_synthesis(careers):
    """Both links of the chain, not just the first.

    education decides sector, and sector decides wage. A synthesizer that
    conditioned every column on the predictors alone would keep the first link
    and lose the second, which is the failure this catches.
    """
    out = sw.Pipeline(careers, synthesizer=_multi()).run()["jobs"]

    college = out.loc[out["education"] == "College", "sector"]
    assert (college.isin(["tech", "health"])).mean() > 0.9

    means = out.groupby("sector")["wage"].mean()
    assert means["tech"] > means["trades"] > means["retail"]


def test_multi_column_synthesis_is_chunk_invariant(careers):
    invariants.assert_chunk_invariant(careers, synthesizer=_multi())


def test_multi_column_synthesis_is_deterministic(careers):
    invariants.assert_deterministic(careers, synthesizer=_multi())


def test_every_synthesized_value_comes_from_a_donor_row(careers):
    """CART samples donors. It never interpolates a new value."""
    raw = sw.Pipeline(careers).run()["jobs"]
    out = sw.Pipeline(careers, synthesizer=_multi()).run()["jobs"]

    for column in ("sector", "wage", "hours"):
        invariants.assert_values_come_from(out, column, raw[column])


def test_a_synthesized_column_keeps_the_dtype_it_had(careers):
    """Choosing to synthesize must not change a column's type.

    Donor values are sampled through an object array, which is an
    implementation detail of the sampling and not something the user asked
    for. The column's type is fixed by the fit, so it is known and can be
    restored. Downstream code that sums or compares a column should not have
    to care whether a synthesizer ran.
    """
    raw = sw.Pipeline(careers).run()["jobs"]
    out = sw.Pipeline(careers, synthesizer=_multi()).run()["jobs"]

    for column in ("wage", "hours", "tenure"):
        assert out[column].dtype == raw[column].dtype, (
            f"{column} was {raw[column].dtype} before synthesis and "
            f"{out[column].dtype} after"
        )


def test_synthesis_does_not_hide_a_numeric_column_behind_object_dtype(careers):
    """The consequence that makes the dtype worth keeping.

    An object column is invisible to numeric selection, costs eight bytes a
    value, and gives Parquet nothing to infer from. tenure is int64 when the
    generator produces it and must still be numeric after synthesis.
    """
    out = sw.Pipeline(careers, synthesizer=_multi()).run()["jobs"]

    assert "tenure" in out.select_dtypes("number").columns, (
        f"tenure came back as {out['tenure'].dtype}, so no numeric operation sees it"
    )


# --- parquet output ---------------------------------------------------------
#
# Only CSV was covered before. Parquet is the format that matters at the scale
# this library targets, and it is stricter: a ParquetWriter fixes its schema
# from the first chunk, so every later chunk has to agree with it.


def _small_schema(**column_overrides) -> sw.Schema:
    person = sw.Entity(
        "person",
        count=600,
        attributes={"education": sw.Choice(["HS", "College"], [0.6, 0.4])},
        identifiers=[sw.Identifier("tax_id")],
    )
    columns = {"hours": sw.Integer(10, 50)}
    columns.update(column_overrides)
    table = sw.Table(
        "t",
        grain=sw.PerEntity("person"),
        carry=["education"],
        identifiers=["tax_id"],
        columns=columns,
    )
    return sw.Schema(entities=[person], tables=[table], seed=4)


def test_parquet_output_carries_the_same_values_as_the_in_memory_run(tmp_path):
    """Choosing a format must not choose a dataset.

    Dtypes are compared loosely on purpose. Conditional columns arrive as
    object dtype from the generator (I10) and Parquet resolves them to a real
    type on the way back, so a strict comparison would be testing I10 rather
    than the writer.
    """
    pytest.importorskip("pyarrow")
    schema = _small_schema()
    expected = sw.Pipeline(schema).run()["t"]

    sw.Pipeline(schema).run_to(tmp_path / "out", format="parquet")
    written = pd.read_parquet(tmp_path / "out" / "t.parquet")

    pd.testing.assert_frame_equal(written, expected, check_dtype=False)


def test_missing_values_round_trip_through_parquet(tmp_path):
    """Missing noise puts None in an object column. pyarrow must keep it null."""
    pytest.importorskip("pyarrow")
    schema = _small_schema()
    noiser = sw.Noise({"t": {"hours": [sw.Missing(0.2)], "education": [sw.Missing(0.2)]}})
    expected = sw.Pipeline(schema, noiser=noiser).run()["t"]

    sw.Pipeline(schema, noiser=noiser, chunk_size=97).run_to(tmp_path / "out", format="parquet")
    written = pd.read_parquet(tmp_path / "out" / "t.parquet")

    assert written["hours"].isna().sum() == expected["hours"].isna().sum()
    assert written["education"].isna().sum() == expected["education"].isna().sum()


def test_parquet_survives_a_column_whose_type_varies_between_chunks(tmp_path):
    pytest.importorskip("pyarrow")
    # Skewed so most chunks hold only the integer branch and a later one
    # carries a float. That is enough to change the inferred arrow type.
    person = sw.Entity(
        "person",
        count=400,
        attributes={"education": sw.Choice(["HS", "College"], [0.97, 0.03])},
        identifiers=[sw.Identifier("tax_id")],
    )
    table = sw.Table(
        "t",
        grain=sw.PerEntity("person"),
        carry=["education"],
        identifiers=["tax_id"],
        columns={
            "mixed": sw.Conditional(
                "education", {"HS": sw.Integer(0, 10), "College": sw.Normal(5.0, 1.0)}
            )
        },
    )
    schema = sw.Schema(entities=[person], tables=[table], seed=4)

    sw.Pipeline(schema, chunk_size=7).run_to(tmp_path / "small", format="parquet")
    assert len(pd.read_parquet(tmp_path / "small" / "t.parquet")) == 400


def test_a_table_with_no_rows_still_writes_a_readable_file(tmp_path):
    pytest.importorskip("pyarrow")
    person = sw.Entity(
        "person",
        count=5,
        attributes={"education": sw.Choice(["HS", "College"], [0.6, 0.4])},
        identifiers=[sw.Identifier("tax_id")],
    )
    table = sw.Table(
        "t", grain=sw.PerEntity("person"), carry=["education"], coverage=0.001
    )
    schema = sw.Schema(entities=[person], tables=[table], seed=4)

    sw.Pipeline(schema).run_to(tmp_path / "out", format="parquet")
    assert len(pd.read_parquet(tmp_path / "out" / "t.parquet")) == 0


# --- prior joints -----------------------------------------------------------


def test_a_joint_prior_shapes_the_relationship_it_declares():
    """The documented reason joints exist: a published cross-tabulation.

    The marginals alone say nothing about how education and employment move
    together. The joint says College is almost always employed, and that is
    what the synthesized data should show.
    """
    person = sw.Entity(
        "person",
        count=800,
        attributes={"education": sw.Choice(["HS", "College"], [0.6, 0.4])},
        identifiers=[sw.Identifier("tax_id")],
    )
    table = sw.Table(
        "t",
        grain=sw.PerEntity("person"),
        carry=["education"],
        columns={"employed": sw.Choice([True, False], [0.5, 0.5])},
    )
    prior = sw.Prior(
        marginals={"education": {"HS": 0.6, "College": 0.4}, "employed": {True: 0.5, False: 0.5}},
        joints={
            ("education", "employed"): {
                ("HS", True): 0.30,
                ("HS", False): 0.30,
                ("College", True): 0.38,
                ("College", False): 0.02,
            }
        },
    )
    result = sw.Pipeline(
        sw.Schema(entities=[person], tables=[table], seed=1),
        synthesizer=sw.CARTSynthesizer(
            ["employed"], tables=["t"], predictors=["education"], structure=prior
        ),
    ).run()["t"]

    rates = result.groupby("education")["employed"].mean()
    assert rates["College"] > rates["HS"] + 0.2


# --- the numeric heuristic --------------------------------------------------


def test_a_numeric_column_with_a_few_sentinel_values_still_fits(people):
    """Survey data: an amount column with a handful of 'refused' answers.

    The old heuristic called this column numeric because 95% of it parsed,
    then handed the raw strings to a regressor. Nothing guesses now: an object
    column is fitted as categories unless the user says otherwise.
    """
    rows = 2_000
    refused = 100  # 5%, comfortably under the 0.9 numeric threshold
    real = pd.DataFrame(
        {
            "education": ["HS", "College"] * (rows // 2),
            "amount": [str(v) for v in np.linspace(10, 100, rows - refused)] + ["refused"] * refused,
        }
    )
    table = sw.Table(
        "t", grain=sw.PerEntity("person"), carry=["education"], columns={"amount": sw.Constant(0)}
    )
    result = sw.Pipeline(
        sw.Schema(entities=[people], tables=[table], seed=1),
        synthesizer=sw.CARTSynthesizer(
            ["amount"], tables=["t"], predictors=["education"], structure=sw.Empirical(real)
        ),
    ).run()["t"]

    assert len(result) == 400


def test_a_column_can_be_declared_numeric_when_its_dtype_does_not_say_so(people):
    """The explicit override, for an object column that really does hold numbers."""
    real = pd.DataFrame(
        {
            "education": ["HS", "College"] * 500,
            "amount": [str(v) for v in np.linspace(10, 100, 1_000)],
        }
    )
    table = sw.Table(
        "t", grain=sw.PerEntity("person"), carry=["education"], columns={"amount": sw.Constant("0")}
    )
    result = sw.Pipeline(
        sw.Schema(entities=[people], tables=[table], seed=1),
        synthesizer=sw.CARTSynthesizer(
            ["amount"],
            tables=["t"],
            predictors=["education"],
            numeric=["amount"],
            structure=sw.Empirical(real),
        ),
    ).run()["t"]

    assert pd.to_numeric(result["amount"]).between(10, 100).all()


def test_declaring_an_unparseable_column_numeric_says_which_column(people):
    """The failure names the column instead of surfacing from inside sklearn."""
    real = pd.DataFrame(
        {"education": ["HS", "College"] * 500, "amount": ["12"] * 999 + ["refused"]}
    )
    table = sw.Table(
        "t", grain=sw.PerEntity("person"), carry=["education"], columns={"amount": sw.Constant("0")}
    )
    with pytest.raises(ValueError, match="'amount'.*not numbers"):
        sw.Pipeline(
            sw.Schema(entities=[people], tables=[table], seed=1),
            synthesizer=sw.CARTSynthesizer(
                ["amount"],
                tables=["t"],
                predictors=["education"],
                numeric=["amount"],
                structure=sw.Empirical(real),
            ),
        ).run()


def test_a_rule_that_declares_no_dtype_still_works(people):
    """The dtype hook is optional, so a rule written before it keeps working."""

    class LegacyRule:
        """No dtype method at all, exactly like a third-party rule from v0.1."""

        def draw(self, keys, *, seed, salt, frame=None):
            return np.array(["x"] * len(keys), dtype=object)

        def depends_on(self):
            return ()

    table = sw.Table("t", grain=sw.PerEntity("person"), columns={"legacy": LegacyRule()})
    result = sw.Pipeline(sw.Schema(entities=[people], tables=[table], seed=1)).run()["t"]

    assert set(result["legacy"]) == {"x"}


def test_a_joint_prior_is_chunk_invariant(people):
    """Invariant 2 on the surface I15 opened up."""
    table = sw.Table(
        "t",
        grain=sw.PerEntity("person"),
        carry=["education"],
        columns={"employed": sw.Choice([True, False], [0.5, 0.5])},
    )
    schema = sw.Schema(entities=[people], tables=[table], seed=1)
    prior = sw.Prior(
        marginals={"education": {"HS": 0.6, "College": 0.4}, "employed": {True: 0.5, False: 0.5}},
        joints={
            ("education", "employed"): {
                ("HS", True): 0.30,
                ("HS", False): 0.30,
                ("College", True): 0.38,
                ("College", False): 0.02,
            }
        },
        rows=2_000,
    )
    invariants.assert_chunk_invariant(
        schema,
        synthesizer=sw.CARTSynthesizer(
            ["employed"], tables=["t"], predictors=["education"], structure=prior
        ),
    )


def test_declaring_a_column_numeric_is_recorded_in_provenance(people):
    """It decides regressor against classifier, so it shapes output and is tagged."""
    real = pd.DataFrame(
        {"education": ["HS", "College"] * 500, "amount": [str(v) for v in np.linspace(1, 9, 1_000)]}
    )
    table = sw.Table(
        "t", grain=sw.PerEntity("person"), carry=["education"], columns={"amount": sw.Constant("0")}
    )
    result = sw.Pipeline(
        sw.Schema(entities=[people], tables=[table], seed=1),
        synthesizer=sw.CARTSynthesizer(
            ["amount"],
            tables=["t"],
            predictors=["education"],
            numeric=["amount"],
            structure=sw.Empirical(real),
        ),
    ).run()

    record = result.provenance.to_frame()
    assert "t.synth.numeric" in set(record["path"])
    assert result.metadata["t"]["synthesize"]["numeric"] == ["amount"]


def test_an_undeclared_column_type_that_shifts_between_chunks_names_itself(tmp_path):
    """The residual case rules cannot cover.

    Sequential wraps an arbitrary function, so it cannot say in advance what
    it produces. A widening shift is absorbed silently. A narrowing one would
    have to change the value, so the write stops with the column named rather
    than with a pyarrow schema dump. Fractions are rare here on purpose, so
    the first chunk is whole numbers and the file schema is fixed as integer.
    """
    pytest.importorskip("pyarrow")
    person = sw.Entity(
        "person", count=400, attributes={"a": sw.Constant("x")}, identifiers=[sw.Identifier("id")]
    )
    table = sw.Table(
        "t",
        grain=sw.PerEntity("person"),
        identifiers=["id"],
        columns={
            "n": sw.Integer(1, 1_000),
            "half": sw.Sequential(
                "n", lambda s: np.asarray([v / 2 if v % 199 == 0 else v for v in s])
            ),
        },
    )
    schema = sw.Schema(entities=[person], tables=[table], seed=4)

    with pytest.raises(ValueError, match="'half'.*changed type|changed type.*'half'"):
        sw.Pipeline(schema, chunk_size=7).run_to(tmp_path / "out", format="parquet")


def test_an_empty_table_writes_the_same_columns_as_a_full_one(tmp_path):
    """The empty file is the same shape as the file that would have been."""
    pytest.importorskip("pyarrow")
    person = sw.Entity(
        "person",
        count=5,
        attributes={"education": sw.Choice(["HS", "College"], [0.6, 0.4])},
        identifiers=[sw.Identifier("tax_id")],
    )
    full = sw.Table("t", grain=sw.PerEntity("person"), carry=["education"], identifiers=["tax_id"])
    empty = sw.Table(
        "t",
        grain=sw.PerEntity("person"),
        carry=["education"],
        identifiers=["tax_id"],
        coverage=0.001,
    )

    sw.Pipeline(sw.Schema(entities=[person], tables=[full], seed=4)).run_to(
        tmp_path / "full", format="parquet"
    )
    sw.Pipeline(sw.Schema(entities=[person], tables=[empty], seed=4)).run_to(
        tmp_path / "empty", format="parquet"
    )

    written = pd.read_parquet(tmp_path / "empty" / "t.parquet")
    assert list(written.columns) == list(pd.read_parquet(tmp_path / "full" / "t.parquet").columns)
    assert len(written) == 0


def test_a_widening_type_shift_between_chunks_is_absorbed(tmp_path):
    """The other direction: integers arriving after a float column started it."""
    pytest.importorskip("pyarrow")
    person = sw.Entity(
        "person", count=400, attributes={"a": sw.Constant("x")}, identifiers=[sw.Identifier("id")]
    )
    table = sw.Table(
        "t",
        grain=sw.PerEntity("person"),
        identifiers=["id"],
        columns={
            "n": sw.Integer(1, 100),
            "half": sw.Sequential(
                "n", lambda s: np.asarray([v / 2 if v % 7 == 0 else v for v in s])
            ),
        },
    )
    schema = sw.Schema(entities=[person], tables=[table], seed=4)

    sw.Pipeline(schema, chunk_size=7).run_to(tmp_path / "out", format="parquet")
    written = pd.read_parquet(tmp_path / "out" / "t.parquet")

    assert len(written) == 400
    assert written["half"].dtype == "float64"
