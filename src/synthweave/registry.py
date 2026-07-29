"""Named registries for pluggable stage implementations.

Each stage kind (generator, synthesizer, noiser, linker, structure source) has
its own registry. A third party registers an implementation by name and the
pipeline resolves it without importing it directly, so adding a stage
implementation never means editing library code.

    @register("noiser", "my-scanner-artifacts")
    class ScannerArtifacts:
        def run(self, chunks, table, ctx): ...
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")

_REGISTRIES: dict[str, dict[str, Any]] = {}


def registry(kind: str) -> dict[str, Any]:
    """The registry for a stage kind, created on first use."""
    return _REGISTRIES.setdefault(kind, {})


def register(kind: str, name: str, *, overwrite: bool = False) -> Callable[[T], T]:
    """Register an implementation under `kind`/`name`.

    Refuses to shadow an existing name unless `overwrite=True`, so two plugins
    claiming the same name fail loudly instead of one silently winning.
    """

    def decorator(obj: T) -> T:
        table = registry(kind)
        if name in table and not overwrite:
            raise ValueError(
                f"{kind} {name!r} is already registered by {table[name]!r}; "
                f"pass overwrite=True to replace it"
            )
        table[name] = obj
        return obj

    return decorator


def resolve(kind: str, name_or_obj: Any) -> Any:
    """Look up a registered implementation, or pass an instance straight through.

    Users can name an implementation (`noiser="default"`) or hand over a
    configured instance. Both reach the pipeline the same way.
    """
    if not isinstance(name_or_obj, str):
        return name_or_obj
    table = registry(kind)
    if name_or_obj not in table:
        known = sorted(table)
        raise KeyError(f"no {kind} named {name_or_obj!r}; registered: {known}")
    found = table[name_or_obj]
    return found() if isinstance(found, type) else found


def available(kind: str) -> list[str]:
    return sorted(registry(kind))
