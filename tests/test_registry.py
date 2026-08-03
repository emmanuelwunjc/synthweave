"""Registry lifecycle: register/unregister, and the autouse fixture that
undoes both between tests.

A test that registers a plugin under a fixed name either leaks it into
`sw.available()` for every later test in the same process, or raises
"already registered" the next time that test runs in a warm interpreter
(`pytest-repeat`, `--lf`, a notebook). `conftest.py`'s `_reset_registries`
fixture exists to make both impossible.
"""

from __future__ import annotations

import pytest

import synthweave as sw


def test_registering_the_same_name_twice_is_rejected():
    sw.register("noiser", "test-registry-dup")(object)
    with pytest.raises(ValueError, match="already registered"):
        sw.register("noiser", "test-registry-dup")(object)


def test_overwrite_true_replaces_an_existing_registration():
    sw.register("noiser", "test-registry-overwrite")(object)
    sw.register("noiser", "test-registry-overwrite", overwrite=True)(dict)
    assert sw.resolve("noiser", "test-registry-overwrite") == {}


def test_unregister_removes_an_entry():
    sw.register("noiser", "test-registry-remove")(object)
    assert "test-registry-remove" in sw.available("noiser")

    sw.unregister("noiser", "test-registry-remove")
    assert "test-registry-remove" not in sw.available("noiser")
    with pytest.raises(KeyError, match="test-registry-remove"):
        sw.resolve("noiser", "test-registry-remove")


def test_unregistering_an_unknown_name_is_rejected():
    with pytest.raises(KeyError, match="test-registry-never-registered"):
        sw.unregister("noiser", "test-registry-never-registered")


def test_a_registration_here_does_not_survive_to_the_next_test():
    """Half of the pair below. Registers something with no matching
    unregister() call, relying entirely on the autouse fixture to clean up."""
    sw.register("noiser", "test-registry-leak-check")(object)
    assert "test-registry-leak-check" in sw.available("noiser")


def test_the_previous_tests_registration_did_not_leak():
    """The other half. If `_reset_registries` did not restore the registry
    after the previous test, this name would still be here."""
    assert "test-registry-leak-check" not in sw.available("noiser")
