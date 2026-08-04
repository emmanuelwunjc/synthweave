"""A deprecation warning is only useful if it blames the right line.

Every assertion about `w.filename` in this file is the same assertion: the
warning names *this* test file and never `_deprecation.py`. That is not a
cosmetic preference. `warnings.filterwarnings(module=...)` and every "show me
what my own code has to change" workflow key on the attributed module, so a
warning attributed to library internals is one the user can neither locate nor
filter, and the deprecation may as well not have been announced.

The case that decides the design is two stacked decorators. A single decorator
happens to work under a hardcoded `stacklevel=2` -- the wrapper warns, level 2
is the wrapper's caller, and that is the user. Stack a second decorator and the
inner wrapper's caller is the outer wrapper, so the same hardcoded 2 now names
`_deprecation.py`. `find_stack_level()` counts frames instead of assuming how
many there are.
"""

from __future__ import annotations

import inspect
import os
import pickle
import warnings

import pytest

from synthweave import SynthweaveDeprecationWarning
from synthweave._deprecation import deprecated, find_stack_level

THIS_FILE = os.path.abspath(__file__)


@deprecated(since="0.4.0", expected_removal="1.0.0", instead="new_way()")
def once(a, b=2):
    """A docstring that must survive the decorator."""
    return a + b


@deprecated(since="0.4.0", expected_removal="1.0.0")
@deprecated(since="0.4.0", expected_removal="1.0.0")
def twice(a, b=2):
    """Two wrappers between the caller and the function."""
    return a * b


def _record(call):
    """Run `call` with warnings captured, returning the recorded warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    return result, caught


# --------------------------------------------------------------------------
# The warning class
# --------------------------------------------------------------------------


def test_the_warning_is_a_deprecation_warning():
    """Existing `-W` filters and `ignore::DeprecationWarning` must still bite."""
    assert issubclass(SynthweaveDeprecationWarning, DeprecationWarning)


def test_the_warning_carries_since_and_expected_removal():
    warning = SynthweaveDeprecationWarning(
        "gone soon", since="0.4.0", expected_removal="1.0.0"
    )
    assert warning.since == "0.4.0"
    assert warning.expected_removal == "1.0.0"
    # The message is still the message: a migration script reads the
    # attributes, a human reads `str(...)`, and neither costs the other.
    assert str(warning) == "gone soon"


def test_the_warning_survives_a_round_trip_through_pickle():
    """Keyword-only `__init__` breaks the default exception reconstruction.

    `BaseException.__reduce__` calls the class with `self.args` only, so
    without an override this raises `TypeError` the first time an exception
    crosses a process boundary or is deep-copied.
    """
    original = SynthweaveDeprecationWarning(
        "gone soon", since="0.4.0", expected_removal="1.0.0"
    )
    restored = pickle.loads(pickle.dumps(original))
    assert isinstance(restored, SynthweaveDeprecationWarning)
    assert str(restored) == "gone soon"
    assert restored.since == "0.4.0"
    assert restored.expected_removal == "1.0.0"


def test_since_and_expected_removal_survive_the_warnings_machinery():
    """Attributes must be readable off the caught warning, not just the object.

    `warnings` re-raises and re-wraps; if the metadata only existed on the
    instance we constructed, nothing downstream could act on it.
    """
    _, caught = _record(lambda: once(1))
    assert len(caught) == 1
    assert caught[0].message.since == "0.4.0"
    assert caught[0].message.expected_removal == "1.0.0"


# --------------------------------------------------------------------------
# The decorator
# --------------------------------------------------------------------------


def test_the_decorator_warns_with_our_own_class():
    with pytest.warns(SynthweaveDeprecationWarning):
        once(1)


def test_the_decorator_preserves_name_docstring_and_return_value():
    assert once.__name__ == "once"
    assert once.__doc__ == "A docstring that must survive the decorator."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SynthweaveDeprecationWarning)
        assert once(1) == 3
        assert once(1, b=10) == 11


def test_instead_reaches_the_message():
    _, caught = _record(lambda: once(1))
    message = str(caught[0].message)
    assert "new_way()" in message
    assert "0.4.0" in message
    assert "1.0.0" in message
    assert "once()" in message


def test_the_message_omits_the_replacement_clause_when_there_is_none():
    """`instead=None` must not produce a dangling "Use None instead"."""
    _, caught = _record(lambda: twice(3))
    message = str(caught[0].message)
    assert "None" not in message
    assert "instead" not in message


# --------------------------------------------------------------------------
# The load-bearing assertion: whose file gets blamed
# --------------------------------------------------------------------------


def test_one_decorator_blames_the_callers_file():
    _, caught = _record(lambda: once(1))
    assert len(caught) == 1
    assert os.path.abspath(caught[0].filename) == THIS_FILE


def test_two_stacked_decorators_blame_the_callers_file():
    """The case a hardcoded `stacklevel` cannot survive.

    Two wrappers fire two warnings. Under `stacklevel=2` the outer one still
    lands here by luck and the inner one lands on `_deprecation.py`, because
    the inner wrapper's caller *is* the outer wrapper. Both must name this
    file, so the assertion is over every warning recorded, not the first.
    """
    result, caught = _record(lambda: twice(3))
    assert result == 6
    assert len(caught) == 2
    blamed = {os.path.abspath(w.filename) for w in caught}
    assert blamed == {THIS_FILE}


def test_two_stacked_decorators_blame_the_callers_line():
    """The file is necessary but not sufficient: the line must be the call."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        expected_line = inspect.currentframe().f_lineno + 1
        twice(3)
    assert [w.lineno for w in caught] == [expected_line, expected_line]


def test_a_caller_side_comprehension_still_blames_the_caller():
    """A user calling a deprecated function inside their own comprehension.

    On Python < 3.12 a list comprehension gets its own frame; from 3.12 it is
    inlined (PEP 709). Either way the frame that exists belongs to the caller,
    so the answer must not depend on the interpreter version.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = [once(n) for n in range(3)]
    assert results == [2, 3, 4]
    assert {os.path.abspath(w.filename) for w in caught} == {THIS_FILE}


# --------------------------------------------------------------------------
# find_stack_level itself
# --------------------------------------------------------------------------


def test_find_stack_level_is_one_when_called_from_outside_the_package():
    """`stacklevel=1` blames the frame that calls `warn`, which is the caller.

    Called from a test, that is the truthful answer, and it is what makes the
    walk terminate rather than run off the top of the stack.
    """
    assert find_stack_level() == 1


# Compiled with a filename inside the package directory so its frames read as
# ours, without shipping a module in `src/` that exists only to be walked over.
# `find_stack_level` decides by `co_filename` alone, which is exactly the rule
# under test: a private helper and the implicit frame a comprehension creates
# are skipped because the file is ours, not because anything counted them.
_PACKAGE_INTERNAL_SOURCE = '''
import warnings

from synthweave._deprecation import SynthweaveDeprecationWarning, find_stack_level


def entry():
    return _helper()


def _helper():
    return [_emit() for _ in range(1)][0]


def _emit():
    warnings.warn(
        SynthweaveDeprecationWarning("probe", since="0.4.0", expected_removal="1.0.0"),
        stacklevel=find_stack_level(),
    )
    return "emitted"
'''


def _package_internal_entry():
    import synthweave._deprecation as dep

    filename = os.path.join(os.path.dirname(dep.__file__), "_probe_not_a_module.py")
    namespace: dict = {}
    exec(compile(_PACKAGE_INTERNAL_SOURCE, filename, "exec"), namespace)
    return namespace["entry"]


def test_nested_package_helpers_and_comprehensions_are_walked_over():
    """Three package frames deep, one of them a comprehension, still blames here."""
    entry = _package_internal_entry()
    result, caught = _record(entry)
    assert result == "emitted"
    assert len(caught) == 1
    assert os.path.abspath(caught[0].filename) == THIS_FILE
