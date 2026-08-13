"""Refusals are part of the engine's contract, not an exception that escapes it.

Both backends refuse tasks they cannot encode. Until now the only place that
turned such a refusal into an outcome was the *benchmark harness*, which caught
``NotImplementedError`` and recorded ``UNSUPPORTED``; a library caller got the
bare exception and the engine never returned UP's own
``UNSUPPORTED_PROBLEM``. These tests pin the contract the other way round: the
engine answers "outside my fragment" through the result it returns, and says
why in its log messages.

Two properties are load-bearing and easy to break later:

  * :class:`UnsupportedTaskError` subclasses ``NotImplementedError``, so every
    caller written against the old convention -- including the ~17 direct
    ``pytest.raises(NotImplementedError)`` tests in this suite -- keeps working;
  * a limit clingo only discovers while *grounding* is a refusal too. The
    encoding states a numeric comparison as a difference ``#sum`` whose weights
    are ``value * coefficient``; on a task whose fluents carry no declared
    bounds the reachable values are what grounding discovers, so no encode-time
    check can predict the overflow (``numeric-domains/satellite`` reaches it
    with coefficients of 1). It has to be classified when it surfaces, which is
    what :func:`refusal_from_clingo_error` does -- and it must not swallow
    ``std::bad_alloc``, which is a memory limit and a different outcome.
"""

import pytest

import aspplanners  # noqa: F401 -- registers the engines
from unified_planning.engines import PlanGenerationResultStatus as Status
from unified_planning.shortcuts import (
    GE,
    Fluent,
    InstantaneousAction,
    OneshotPlanner,
    Problem,
    RealType,
    Times,
)

from aspplanners.common.errors import UnsupportedTaskError, refusal_from_clingo_error


def nonlinear_task() -> Problem:
    """A task outside the fragment: a precondition multiplying two fluents.

    The encoder refuses this by inspection, well before clingo is involved --
    it is the largest refusal family in the benchmark (287 tasks).
    """
    x = Fluent("x", RealType())
    y = Fluent("y", RealType())
    done = Fluent("done")

    act = InstantaneousAction("act")
    act.add_precondition(GE(Times(x, y), 1))
    act.add_effect(done, True)

    problem = Problem("nonlinear")
    problem.add_fluent(x, default_initial_value=2)
    problem.add_fluent(y, default_initial_value=3)
    problem.add_fluent(done, default_initial_value=False)
    problem.add_action(act)
    problem.add_goal(done)
    return problem


# ---------------------------------------------------------------------------
# The refusal reaches the caller as a result, not as an exception
# ---------------------------------------------------------------------------

def test_a_task_outside_the_fragment_comes_back_as_unsupported_problem():
    with OneshotPlanner(name="PLASPPlanner") as planner:
        result = planner.solve(nonlinear_task())
    assert result.status == Status.UNSUPPORTED_PROBLEM
    assert result.plan is None


def test_the_refusal_says_why_in_the_log_messages():
    with OneshotPlanner(name="PLASPPlanner") as planner:
        result = planner.solve(nonlinear_task())
    said = " ".join(message.message for message in result.log_messages)
    assert "cannot encode this task" in said
    assert "not linear" in said


def test_solving_a_refused_task_does_not_raise():
    """The point of the contract: no caller has to know which exception type
    means "outside the fragment"."""
    with OneshotPlanner(name="PLASPPlanner") as planner:
        planner.solve(nonlinear_task())   # must not raise


# ---------------------------------------------------------------------------
# Backwards compatibility of the refusal type
# ---------------------------------------------------------------------------

def test_the_refusal_type_is_still_a_not_implemented_error():
    """Existing callers catch ``NotImplementedError``; keep them working."""
    assert issubclass(UnsupportedTaskError, NotImplementedError)


def test_a_refusal_raised_directly_is_still_catchable_the_old_way():
    from aspplanners.plasp.planner import PLASPPlanner
    with pytest.raises(NotImplementedError, match="not linear"):
        PLASPPlanner(nonlinear_task(), "seq")


# ---------------------------------------------------------------------------
# Limits clingo only finds while grounding
# ---------------------------------------------------------------------------

def test_a_grounding_overflow_is_classified_as_a_refusal():
    error = RuntimeError(
        "bool Clasp::Asp::LogicProgram::simplifySum(Head_t, const Potassco::AtomSpan &, "
        "const Potassco::Sum_t &, RuleBuilder &, SRule &)@1854: Value too large to be "
        "stored in data type: Integer overflow!")
    refusal = refusal_from_clingo_error(error)
    assert isinstance(refusal, UnsupportedTaskError)
    assert "32-bit" in str(refusal)


def test_an_out_of_memory_error_is_not_a_refusal():
    """``std::bad_alloc`` is a memory limit, and a memout is not an unsupported
    task -- classifying it as one would quietly shrink the encodable set."""
    assert refusal_from_clingo_error(RuntimeError("std::bad_alloc")) is None


def test_an_unrecognized_clingo_error_is_not_a_refusal():
    assert refusal_from_clingo_error(RuntimeError("something else entirely")) is None
