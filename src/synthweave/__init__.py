"""synthweave: synthetic tabular data that does not require real microdata.

Declare entities and the tables that record them, then run a pipeline:

    import synthweave as sw

    people = sw.Entity(
        "person",
        count=10_000,
        attributes={
            "education": sw.Choice(["HS", "College"], [0.6, 0.4]),
            "birth_year": sw.Integer(1960, 2005),
        },
        identifiers=[sw.Identifier("student_id", prefix="SID"),
                     sw.Identifier("tax_id", prefix="TIN")],
    )

    roster = sw.Table(
        "roster",
        grain=sw.PerEntity("person"),
        carry=["education", "birth_year"],
        identifiers=["student_id"],
    )

    wages = sw.Table(
        "wages",
        grain=sw.PerPeriod("person", periods=range(2018, 2026), presence=0.8),
        carry=["education"],
        identifiers=["tax_id"],
        coverage=0.7,
        columns={
            # Declared structure: wage depends on education, so the table has
            # real inter-column relationships before any model sees it.
            "wage": sw.Conditional("education", {
                "HS":      sw.Normal(38_000, 9_000, low=0),
                "College": sw.Normal(64_000, 18_000, low=0),
            }),
        },
    )

    result = sw.Pipeline(
        sw.Schema(entities=[people], tables=[roster, wages], seed=42),
        noiser=sw.Noise({"roster": {"education": [sw.Typo(0.05)]}}),
    ).run()

    result["wages"].head()
    result.unjustified()     # config values nobody has justified yet

The same person's `tax_id` is identical in every table carrying it, because
identifiers are derived from (seed, entity, tag) rather than looked up.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _installed_version

from ._hash import derive_id
from .context import RunContext
from .fidelity import FidelityReport, fidelity_report
from .pipeline import Pipeline, PipelineResult
from .provenance import ProvenanceRecord, Tagged, cited, modeled, user
from .registry import available, register, resolve
from .rules import (
    Choice,
    Conditional,
    Constant,
    Integer,
    Normal,
    Rule,
    Sequential,
    Uniform,
)
from .schema import Entity, Identifier, PerEntity, PerEvent, PerPeriod, Schema, Table
from .stages.generate import RuleGenerator
from .stages.link import DeterministicLinker
from .stages.noise import OCR, Missing, Noise, NoiseOp, Typo
from .stages.synthesize import (
    CARTSynthesizer,
    Declared,
    Empirical,
    Prior,
    StructureConfigError,
)
from .validation import SchemaError

# `pyproject.toml` is the single source of the version. Read it back from the
# installed metadata rather than repeating it here: two copies drift the first
# time one is bumped alone, and the installed package can then disagree with
# its own metadata about what it is.
try:
    __version__ = _installed_version("synthweave")
except PackageNotFoundError:
    # Running from a source tree that was never installed, e.g. via
    # PYTHONPATH=src. There is no metadata to read, and guessing a number
    # here would recreate exactly the drift this avoids.
    __version__ = "0.0.0+unknown"

__all__ = [
    # schema
    "Schema", "Entity", "Table", "Identifier",
    "PerEntity", "PerPeriod", "PerEvent",
    # rules
    "Rule", "Constant", "Choice", "Integer", "Uniform", "Normal",
    "Conditional", "Sequential",
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
    "register", "resolve", "available", "derive_id",
    "SchemaError",
]
