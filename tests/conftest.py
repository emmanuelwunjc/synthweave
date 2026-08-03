"""Shared fixtures.

Every test goes through the public API: build a schema, run a pipeline, assert
on the result. Nothing reaches into a stage implementation, so the stages can
be rewritten without touching these tests.
"""

from __future__ import annotations

import pytest

import synthweave as sw
from synthweave import registry


@pytest.fixture(autouse=True)
def _reset_registries():
    """Undo any `register()`/`unregister()` a test performs.

    Without this, a test that registers a plugin under a fixed name (a stub
    structure source, a test noiser) either leaks it into `sw.available()`
    for every later test in the same process, or raises "already
    registered" the next time that same test runs in a warm interpreter
    (`pytest-repeat`, `--lf`, a notebook).
    """
    snapshot = registry._snapshot()
    yield
    registry._restore(snapshot)


@pytest.fixture
def people() -> sw.Entity:
    return sw.Entity(
        "person",
        count=400,
        attributes={
            "education": sw.Choice(["HS", "College"], [0.6, 0.4]),
            "birth_year": sw.Integer(1960, 2005),
        },
        identifiers=[
            sw.Identifier("student_id", prefix="SID", digits=9),
            sw.Identifier("tax_id", prefix="TIN", digits=9),
        ],
    )


@pytest.fixture
def many_people() -> sw.Entity:
    """A larger population for tests that assert on realized rates.

    At 400 entities a 20% rate has a 3-sigma spread of roughly plus or minus
    6 points, which is too wide for an assertion to mean anything. At 20,000 it
    is under 1 point, so a tight bound is a real check rather than a coin flip.
    """
    return sw.Entity(
        "person",
        count=20_000,
        attributes={"education": sw.Choice(["HS", "College"], [0.6, 0.4])},
        identifiers=[sw.Identifier("tax_id", prefix="TIN", digits=9)],
    )


@pytest.fixture
def roster() -> sw.Table:
    return sw.Table(
        "roster",
        grain=sw.PerEntity("person"),
        carry=["education", "birth_year"],
        identifiers=["student_id", "tax_id"],
    )


@pytest.fixture
def wages() -> sw.Table:
    return sw.Table(
        "wages",
        grain=sw.PerPeriod("person", periods=[2020, 2021, 2022]),
        carry=["education"],
        identifiers=["tax_id"],
        columns={
            "wage": sw.Conditional(
                "education",
                {
                    "HS": sw.Normal(38_000, 5_000, low=0),
                    "College": sw.Normal(64_000, 8_000, low=0),
                },
            )
        },
    )


@pytest.fixture
def schema(people, roster, wages) -> sw.Schema:
    return sw.Schema(entities=[people], tables=[roster, wages], seed=42)


@pytest.fixture
def jobs() -> sw.Table:
    """Three columns whose declared structure is chained.

    sector depends on the carried education, and wage and hours both depend on
    sector. A single synthesized column proves nothing about visit order or
    about conditioning on already-synthesized columns. A chain does.
    """
    return sw.Table(
        "jobs",
        grain=sw.PerEntity("person"),
        carry=["education"],
        columns={
            "sector": sw.Conditional(
                "education",
                {
                    "HS": sw.Choice(["retail", "trades"], [0.7, 0.3]),
                    "College": sw.Choice(["tech", "health"], [0.5, 0.5]),
                },
            ),
            "wage": sw.Conditional(
                "sector",
                {
                    "retail": sw.Normal(30_000, 4_000, low=0),
                    "trades": sw.Normal(48_000, 5_000, low=0),
                    "tech": sw.Normal(95_000, 9_000, low=0),
                    "health": sw.Normal(72_000, 7_000, low=0),
                },
            ),
            "hours": sw.Conditional(
                "sector",
                {
                    "retail": sw.Integer(10, 30),
                    "trades": sw.Integer(35, 50),
                    "tech": sw.Integer(35, 45),
                    "health": sw.Integer(30, 60),
                },
            ),
            # Unconditional on purpose: it is the one column the generator
            # hands over as a real int64, so it is the only one that can show
            # whether a later stage preserves a dtype.
            "tenure": sw.Integer(0, 40),
        },
    )


@pytest.fixture
def careers(people, jobs) -> sw.Schema:
    return sw.Schema(entities=[people], tables=[jobs], seed=11)
