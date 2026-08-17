"""What "this engine cannot encode that task" means, in one place.

A backend refuses a task for two different sorts of reason and they deserve one
vocabulary:

  * the encoder reads the task and declines a shape it cannot express -- a
    product of two fluents, a coefficient no rescaling makes whole, a numeric
    over-all condition. These are found by inspection, before clingo is asked
    anything;
  * clingo discovers, while grounding, that the program the encoding built is
    not representable -- a ``#sum`` whose weights leave the signed 32-bit range.
    That one cannot be predicted from the task alone: the encoding states a
    comparison as a difference ``#sum`` whose weights are ``value * coefficient``,
    and on a task whose fluents declare no bounds the reachable values are what
    grounding discovers. ``numeric-domains/satellite`` reaches the limit with
    every stated coefficient equal to 1, so there is nothing in the task to
    check against.

Both mean the same thing to a caller: the task is outside this engine's
fragment, which is not a crash and not a planning failure. UP already has a
status for it (``UNSUPPORTED_PROBLEM``), and the engine adapters translate
:class:`UnsupportedTaskError` into it.

:class:`UnsupportedTaskError` subclasses ``NotImplementedError`` deliberately.
That was the convention before this module existed -- every refusal site raised
a bare ``NotImplementedError`` and the benchmark harness caught it -- so
subclassing keeps every existing caller and test working while the sites are
migrated one at a time.
"""

from typing import Optional

__all__ = ['UnsupportedTaskError', 'refusal_from_clingo_error']


class UnsupportedTaskError(NotImplementedError):
    """The task is outside the fragment this engine can encode.

    Not a bug and not an unsolvable task: a statement about this engine's
    reach. Subclasses ``NotImplementedError`` for backwards compatibility with
    callers written against the older convention.
    """


# clingo reports a non-representable program by raising RuntimeError with the
# C++ assertion text. There is no error code to match on, so the message is all
# there is; both spellings clasp uses for the range check are listed. The match
# is deliberately narrow -- anything unrecognized stays an error, because
# calling a genuine crash "unsupported" would quietly inflate the fragment.
_OVERFLOW_MARKERS = (
    'integer overflow',
    'value too large',
)


def refusal_from_clingo_error(error: BaseException) -> Optional[UnsupportedTaskError]:
    """The refusal a clingo error represents, or ``None`` if it is not one.

    Only the representability limit is a refusal. ``std::bad_alloc`` is a memory
    limit -- a different outcome with a different fix -- and is left alone, as is
    anything unrecognized.
    """
    message = str(error)
    lowered = message.lower()
    if not any(marker in lowered for marker in _OVERFLOW_MARKERS):
        return None
    return UnsupportedTaskError(
        "The ASP encoding of this task is not representable by clingo: it built "
        "an aggregate whose weights leave the signed 32-bit term range, and "
        "clingo reported it while grounding. The encoding states a numeric "
        "comparison as a difference #sum over the reachable values of its "
        "fluents, so this is reached by tasks whose values grow without a "
        "declared bound rather than by any single large constant in the task. "
        f"clingo said: {message}")
