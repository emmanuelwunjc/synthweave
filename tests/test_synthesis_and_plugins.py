"""Structure sources, the fit cap, plugin resolution, and incremental output.

Still one seam. Plugin behavior is proved by registering a stub implementation
and running the pipeline, never by reaching into the registry.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

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
