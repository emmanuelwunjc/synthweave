"""Deprecation machinery: a warning that points at the caller, not at us.

A deprecation warning exists to make somebody change a line of code. It can
only do that if it names *their* line. Python attributes a warning to a frame
chosen by `stacklevel`, counted outward from whoever calls `warnings.warn`, so
the number is a claim about how deep the call stack is at that moment. Any
fixed number is that claim frozen at one call site, and it is wrong the moment
the shape of the stack changes.

That is not hypothetical here. `deprecated()` is a decorator: applying it adds
a frame. Stacking two adds two. With a hardcoded `stacklevel=2` a singly
decorated function happens to be attributed correctly and a doubly decorated
one is blamed on this file, which is both useless to the user (it names a
module they have never opened) and unfilterable, since the `module=` argument
to `warnings.filterwarnings` matches on exactly that attribution.

`find_stack_level()` replaces the guess with a measurement: walk outward from
the frame that is about to warn and stop at the first frame whose file lives
outside this package. Whatever is between (decorator wrappers, private
helpers, and on Python < 3.12 the implicit frame a comprehension creates) is
skipped because it belongs to us, not because anyone counted it.
"""

from __future__ import annotations

import functools
import os
import sys
import warnings
from typing import Any, Callable, TypeVar

# The directory this package occupies. A frame is "ours" when its file lives
# under it, which covers subpackages (`stages/`, `connectors/`) for free.
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep

_F = TypeVar("_F", bound=Callable[..., Any])


class SynthweaveDeprecationWarning(DeprecationWarning):
    """A deprecation that says when it started and when it ends.

    A bare `DeprecationWarning` carries prose. Prose is unparseable, so a
    downstream project cannot answer "does anything I use disappear before I
    upgrade?" without reading every message by hand. `since` and
    `expected_removal` are the two dates that question needs, kept as
    attributes so a `pytest.warns(...)` assertion or a migration script can
    read them directly instead of matching on a sentence.
    """

    def __init__(
        self,
        message: str,
        *,
        since: str,
        expected_removal: str,
    ) -> None:
        super().__init__(message)
        self.since = since
        self.expected_removal = expected_removal

    def __reduce__(self) -> tuple:
        """Stay reconstructible despite the keyword-only metadata.

        `BaseException.__reduce__` rebuilds an exception by calling the class
        with `self.args`, which here is the message alone. That would raise
        `TypeError: missing 2 required keyword-only arguments` for anything
        that pickles or copies this -- and it is a public, exported class, so
        crossing a process boundary is somebody's normal Tuesday.
        """
        return (
            _rebuild,
            (str(self), self.since, self.expected_removal),
        )


def _rebuild(
    message: str, since: str, expected_removal: str
) -> SynthweaveDeprecationWarning:
    """Module-level so `__reduce__` has something picklable to name."""
    return SynthweaveDeprecationWarning(
        message, since=since, expected_removal=expected_removal
    )


def _is_ours(filename: str) -> bool:
    """Does this frame's file live inside the package?"""
    return os.path.abspath(filename).startswith(_PACKAGE_DIR)


def find_stack_level() -> int:
    """The `stacklevel` that attributes a warning to the caller's own line.

    Counted from the frame that calls this function, which must also be the
    frame that calls `warnings.warn` -- `stacklevel=1` means exactly that
    frame. Walk outward while the frames belong to this package and stop at
    the first one that does not; the number of steps taken is the level that
    names it.

    Returns 1 when called from outside the package, which is the truthful
    answer: the caller is already the frame to blame.
    """
    frame = sys._getframe(1)
    level = 1
    while frame is not None and _is_ours(frame.f_code.co_filename):
        frame = frame.f_back
        level += 1
    return level


def _deprecation_message(
    name: str,
    since: str,
    expected_removal: str,
    instead: str | None,
) -> str:
    message = (
        f"{name} is deprecated since synthweave {since} "
        f"and is expected to be removed in {expected_removal}."
    )
    if instead is not None:
        message += f" Use {instead} instead."
    return message


def deprecated(
    *,
    since: str,
    expected_removal: str,
    instead: str | None = None,
) -> Callable[[_F], _F]:
    """Mark a callable deprecated, warning at the caller's line when it runs.

    Private on purpose: nothing is deprecated yet, and an unused public
    decorator is a promise about a shape nobody has tested against a real
    deprecation. It becomes public when the first caller needs it.
    """

    def decorate(func: _F) -> _F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(
                SynthweaveDeprecationWarning(
                    _deprecation_message(
                        f"{func.__qualname__}()", since, expected_removal, instead
                    ),
                    since=since,
                    expected_removal=expected_removal,
                ),
                stacklevel=find_stack_level(),
            )
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate
