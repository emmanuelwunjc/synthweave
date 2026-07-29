"""The design invariants, as assertions.

`docs/HANDOFF.md` lists nine properties the design guarantees. Prose cannot
fail a build, so each one that a test needs lives here as a function instead.

Every helper enters through the public API, exactly as the tests do. Nothing
here reaches into a stage, so these helpers stay valid when a stage is
rewritten.

Helpers are added when a test needs one, not in advance. An unused assertion
is as untested as the invariant it claims to protect.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

import synthweave as sw

from synthweave.validation import RESERVED_PREFIX

# A size well above any fixture, so one run is a single chunk, and a size small
# enough to split every fixture into many. Chunking must not matter.
WHOLE = 100_000
SPLIT = 13


def assert_chunk_invariant(schema: sw.Schema, *, sizes=(SPLIT, 97, WHOLE), **stages: Any) -> None:
    """Invariant 2. chunk_size is a memory knob and nothing else.

    The highest-value oracle in the suite: it caught the worst bug found so
    far. Every new surface gets one of these, whether or not a chunking bug
    looks likely there.
    """
    reference = sw.Pipeline(schema, chunk_size=sizes[-1], **stages).run()
    for size in sizes[:-1]:
        other = sw.Pipeline(schema, chunk_size=size, **stages).run()
        assert set(other.tables) == set(reference.tables), (
            f"chunk_size={size} produced tables {sorted(other.tables)}, "
            f"expected {sorted(reference.tables)}"
        )
        for name in reference.tables:
            pd.testing.assert_frame_equal(
                other[name], reference[name], obj=f"table {name!r} at chunk_size={size}"
            )


def assert_deterministic(schema: sw.Schema, **stages: Any) -> None:
    """Invariant 1. The same schema and seed produce the same output, always."""
    first = sw.Pipeline(schema, **stages).run()
    second = sw.Pipeline(schema, **stages).run()
    for name in first.tables:
        pd.testing.assert_frame_equal(second[name], first[name], obj=f"table {name!r} on rerun")


def assert_no_reserved_columns(result: sw.PipelineResult) -> None:
    """Invariant 9. Nothing prefixed `_sw_` reaches the user."""
    for name in result.tables:
        leaked = [c for c in result[name].columns if str(c).startswith(RESERVED_PREFIX)]
        assert not leaked, f"table {name!r} leaked bookkeeping columns {leaked}"


def assert_entity_attributes_consistent(
    result: sw.PipelineResult, tables: list[str], *, key: str, attribute: str
) -> None:
    """Invariant 6. A carried attribute is identical everywhere its entity appears."""
    seen: dict[Any, Any] = {}
    for name in tables:
        frame = result[name]
        for entity_key, value in zip(frame[key], frame[attribute]):
            if entity_key in seen:
                assert seen[entity_key] == value, (
                    f"entity {entity_key!r} has {attribute}={seen[entity_key]!r} in an earlier "
                    f"table but {value!r} in table {name!r}"
                )
            else:
                seen[entity_key] = value


def assert_values_come_from(frame: pd.DataFrame, column: str, source: pd.Series) -> None:
    """Donor sampling draws real values, never invented ones.

    Not one of the nine, but the property that makes CART output plausible:
    a synthesized value is a value some donor row actually held.
    """
    unseen = set(pd.Series(frame[column]).dropna()) - set(source.dropna())
    assert not unseen, (
        f"column {column!r} contains {len(unseen)} value(s) absent from the donor pool, "
        f"e.g. {sorted(unseen, key=str)[:3]}"
    )
