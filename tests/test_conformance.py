"""`sw.check_rule`: the conformance harness for a custom `Rule`.

`Rule` is a `runtime_checkable` Protocol matched on `draw`/`depends_on` only,
so nothing stops a plugin author from writing one that breaks the two
contracts the docstring promises. These tests are the harness's own proof
that it actually catches that.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

import synthweave as sw


class RowIndexRule:
    """Values follow row position, not the key. isinstance(_, sw.Rule) is
    True; the docstring's promise is broken anyway."""

    def draw(self, keys, *, seed, salt, frame=None):
        return np.arange(len(keys))

    def depends_on(self):
        return ()


class RandomRule:
    """Reaches for hidden RNG state instead of deriving from the key."""

    def draw(self, keys, *, seed, salt, frame=None):
        return np.array([random.random() for _ in keys])

    def depends_on(self):
        return ()


class ChunkCountRule:
    """Chunk invariance broken: the value depends on how many keys arrived
    together, not on any one key."""

    def draw(self, keys, *, seed, salt, frame=None):
        return np.full(len(keys), len(keys))

    def depends_on(self):
        return ()


def test_a_correct_rule_passes():
    sw.check_rule(sw.Choice(["a", "b"], [0.5, 0.5]))
    sw.check_rule(sw.Integer(0, 100))


def test_a_row_position_keyed_rule_is_rejected():
    with pytest.raises(sw.RuleConformanceError, match="position or order"):
        sw.check_rule(RowIndexRule())


def test_a_non_deterministic_rule_is_rejected():
    with pytest.raises(sw.RuleConformanceError, match="not deterministic"):
        sw.check_rule(RandomRule())


def test_a_chunk_size_dependent_rule_is_rejected():
    with pytest.raises(sw.RuleConformanceError, match="chunk invariant"):
        sw.check_rule(ChunkCountRule())
