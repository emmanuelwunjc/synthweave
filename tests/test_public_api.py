"""The public API surface is frozen: nothing enters or leaves it silently.

`import synthweave as sw` is the whole contract. Every name a user can reach
through `sw.` is either a documented export or an accident, and until this
file existed nothing told the two apart. Adding a name to `__all__`, dropping
one, or letting an import leak a third-party object into the top level all
used to pass CI without comment.

The negative assertion is the point of the file. `PackageNotFoundError` was
re-exported from `importlib.metadata` purely because the version lookup
imported it un-aliased, so `sw.PackageNotFoundError` was a real, usable,
entirely unintended part of the API. Nothing had noticed.

The ten submodules bound by importing their contents (`sw.schema`,
`sw.rules`, ...) are asserted as they are, not as they should be. Whether
they belong in the surface is a separate decision (#52); this file only makes
the current answer impossible to change by accident.

Everything about the *namespace* is measured in a fresh interpreter, not this
one. `import synthweave.io` anywhere in the process binds `io` onto the parent
package, so by the time the full suite reaches this file, `dir(synthweave)`
also carries `io` and `connectors`. Asserting it in-process passes alone and
fails under `pytest tests/`, which is a test that reports test-ordering rather
than API surface. The subprocess measures what a user actually gets from a
bare `import synthweave`.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import sys

import synthweave as sw

# The frozen surface. Changing this tuple is the deliberate act of changing
# the public API, and should be visible in a diff as exactly that.
EXPECTED = (
    # schema
    "Schema", "Entity", "Table", "Identifier",
    "PerEntity", "PerPeriod", "PerEvent",
    # rules
    "Rule", "Constant", "Choice", "Integer", "Uniform", "Normal",
    "Conditional", "Sequential", "check_rule", "RuleConformanceError",
    # conformance
    "check_synthesizer", "SynthesizerConformanceError",
    # mode
    "Mode", "ModeSchema",
    # pipeline
    "Pipeline", "PipelineResult", "RunContext",
    # fidelity
    "fidelity_report", "FidelityReport",
    # stages
    "RuleGenerator", "CARTSynthesizer", "Noise", "DeterministicLinker",
    "Declared", "Empirical", "Prior", "StructureConfigError",
    "NoiseOp", "Typo", "OCR", "Missing",
    # provenance
    "Tagged", "ProvenanceRecord", "user", "modeled", "cited",
    # extension
    "register", "unregister", "resolve", "available", "derive_id",
    "SchemaError",
)

# Bound as a side effect of `from .schema import ...` and friends. Asserted
# as-is; see the module docstring.
EXPECTED_SUBMODULES = (
    "conformance",
    "context",
    "fidelity",
    # `from .mode import Mode, ModeSchema` binds `mode` here too, exactly as
    # every other `from .<module> import ...` above does.
    "mode",
    "pipeline",
    "provenance",
    "registry",
    "rules",
    "schema",
    "stages",
    "validation",
)


_PROBE = """
import json, types, synthweave
public = [n for n in dir(synthweave) if not n.startswith("_")]
modules, foreign = [], {}
for name in public:
    obj = getattr(synthweave, name)
    if isinstance(obj, types.ModuleType):
        modules.append(name)
        origin = obj.__name__
    else:
        origin = getattr(obj, "__module__", None)
    if origin is not None and origin.split(".")[0] != "synthweave":
        foreign[name] = origin
print(json.dumps({"public": public, "modules": modules, "foreign": foreign}))
"""


@functools.lru_cache(maxsize=1)
def _fresh_namespace() -> dict:
    """What `import synthweave` alone exposes, measured in a new interpreter."""
    src = os.path.dirname(os.path.dirname(os.path.abspath(sw.__file__)))
    # Extend the environment, never replace it. A helper here once built a
    # bare env, which held on macOS and broke on the CI runner.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [src] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout)


def _public_names() -> set[str]:
    return set(_fresh_namespace()["public"])


def test_expected_lists_each_name_once():
    # Guards the guard: a duplicate in EXPECTED would make the set comparison
    # below pass while the length comparison silently disagreed with __all__.
    assert len(EXPECTED) == len(set(EXPECTED))


def test_all_matches_the_frozen_surface():
    assert set(sw.__all__) == set(EXPECTED)
    # By length too, so a duplicate entry in __all__ is caught. A set
    # comparison alone cannot see one.
    assert len(sw.__all__) == len(EXPECTED)


def test_every_exported_name_resolves():
    missing = [name for name in EXPECTED if not hasattr(sw, name)]
    assert missing == []
    for name in EXPECTED:
        assert getattr(sw, name) is not None


def test_no_public_name_is_a_foreign_import():
    """No top-level public name may come from outside synthweave.

    An un-aliased `from importlib.metadata import PackageNotFoundError`, or
    any `import pandas`-style leak, publishes someone else's object under our
    name. Private aliases (`_PackageNotFoundError`) are the fix: they keep the
    import usable and out of the surface.
    """
    assert _fresh_namespace()["foreign"] == {}


def test_only_the_known_submodules_are_bound():
    assert set(_fresh_namespace()["modules"]) == set(EXPECTED_SUBMODULES)


def test_nothing_public_is_undeclared():
    """Every public name is either an export or a known bound submodule."""
    assert _public_names() == set(EXPECTED) | set(EXPECTED_SUBMODULES)
