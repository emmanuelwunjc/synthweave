"""Shared fixtures.

Every test goes through the public API: build a schema, run a pipeline, assert
on the result. Nothing reaches into a stage implementation, so the stages can
be rewritten without touching these tests.
"""

from __future__ import annotations

import pytest

import synthweave as sw


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
