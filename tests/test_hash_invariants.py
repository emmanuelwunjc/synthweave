"""Properties of the derivation layer that only a numeric slip would break.

`src/synthweave/_hash.py` is the floor every value in the library stands on,
so a corruption there is silently wrong everywhere at once rather than loudly
wrong in one place. The mutation harness covers the guards in that module, but
a guard being present says nothing about the arithmetic around it being right.

Each test here exists because a specific mutation in `tools/mutation_check.py`
was reported MISSED without it. They assert through the public API, so they
stay valid if the derivation is rewritten, and they compare against the
parameters the user declared rather than against anything recomputed the way
the implementation computes it.
"""

from __future__ import annotations

import synthweave as sw


def test_normal_draws_have_the_requested_spread():
    """`#65 normal() Box-Muller radius keeps the -2 factor`.

    Box-Muller only yields a unit normal because of the -2 in
    `sqrt(-2 * log(u1))`. Drop it to -1 and the draws are still centred, still
    bell shaped and still deterministic, so every existing test passes: only
    the spread is wrong, by a factor of sqrt(2). Checking the mean alone would
    not have noticed, which is why the standard deviation is asserted against
    the declared `sd` and not against anything the module computes.
    """
    population = 20_000
    mean, sd = 170.0, 10.0
    person = sw.Entity(
        "person",
        count=population,
        attributes={"height": sw.Normal(mean, sd)},
    )
    table = sw.Table("t", grain=sw.PerEntity("person"), carry=["height"])
    schema = sw.Schema(entities=[person], tables=[table], seed=7)

    values = sw.Pipeline(schema).run()["t"]["height"]

    # At 20,000 draws the sampling error on either statistic is well under
    # 0.2, while the sqrt(2) corruption moves sd by about 3. The tolerances
    # sit between the two so the test is neither flaky nor blind.
    assert abs(values.mean() - mean) < 0.5, f"mean {values.mean()} is not near {mean}"
    assert abs(values.std(ddof=0) - sd) < 0.5, f"sd {values.std(ddof=0)} is not near {sd}"


def test_identifiers_use_the_whole_declared_keyspace():
    """`#65 derive_id() uses the full declared keyspace`.

    Identifier width is a collision budget: `validate_schema` refuses a width
    whose birthday bound puts more than one expected collision in the
    population. That budget assumes the derivation actually spends the width
    it was given. Take the modulus down one decade and every identifier still
    has the right number of characters, still starts with the right prefix and
    is still stable across runs; it just has a leading zero forever, silently
    making collisions ten times more likely than the schema check promised.
    """
    digits = 10
    person = sw.Entity(
        "person",
        count=2_000,
        attributes={"a": sw.Constant("x")},
        identifiers=[sw.Identifier("id", digits=digits)],
    )
    table = sw.Table("t", grain=sw.PerEntity("person"), identifiers=["id"])
    schema = sw.Schema(entities=[person], tables=[table], seed=3)

    values = sw.Pipeline(schema).run()["t"]["id"]

    assert {len(v) for v in values} == {digits}, "identifiers are not the declared width"
    # With 2,000 draws over a full 10-digit keyspace, the chance that not one
    # of them reaches the top decade is 0.9 ** 2000, i.e. never. A modulus one
    # decade short cannot produce such a value at all.
    widest = max(int(v) for v in values)
    assert widest >= 10 ** (digits - 1), (
        f"no identifier reached the top decade of {digits} digits (widest {widest}); "
        "the derivation is not spending the width the schema check budgeted for"
    )
