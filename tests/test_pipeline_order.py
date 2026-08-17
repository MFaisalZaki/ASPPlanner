"""The compilation pipeline's stage order, checked rather than only documented.

``PLASPEncoder.compile`` runs its stages in an order that is load-bearing, and
every constraint has the same shape: getting it wrong produces a *wrong plan*
rather than an error. Rescaling after the durative split is "silently dropped on
every temporal task"; filling fluent defaults after the initial-state facts
"silently disables every numeric precondition that reads it"; applying the
duration scales before the split feeds the integer stand-ins into the time grid
instead of the durations the task states.

Those consequences are why the order is checked as the pipeline runs. The check
is on stage sequence only -- one list lookup per stage -- so it stays cheap on
the instances where grounding is already the binding cost.
"""

import pytest

from aspplanners.plasp.encoder import PipelineOrderError, _StageOrder


# ---------------------------------------------------------------------------
# The mechanism
# ---------------------------------------------------------------------------

def test_a_stage_may_run_once_its_dependencies_have():
    order = _StageOrder()
    order.run("first")
    order.run("second", after=("first",))
    order.run("third", after=("first", "second"))


def test_a_stage_that_runs_too_early_is_refused():
    order = _StageOrder()
    order.run("first")
    with pytest.raises(PipelineOrderError, match="ran before"):
        order.run("third", after=("first", "second"))


def test_the_error_names_the_missing_stage_and_what_did_run():
    order = _StageOrder()
    order.run("rescale")
    with pytest.raises(PipelineOrderError) as raised:
        order.run("facts", after=("rescale", "defaults"))
    message = str(raised.value)
    assert "'defaults'" in message      # what was missing
    assert "'rescale'" in message       # what had run


def test_a_pipeline_order_error_is_not_a_refusal():
    """The distinction matters now that the engines translate every
    ``NotImplementedError`` into UNSUPPORTED_PROBLEM: a reordered pipeline is a
    bug in this package and has to stay an error, not become a statement that
    the user's task is outside the fragment."""
    assert issubclass(PipelineOrderError, RuntimeError)
    assert not issubclass(PipelineOrderError, NotImplementedError)


# ---------------------------------------------------------------------------
# The pipeline actually declares the constraints its comments state
# ---------------------------------------------------------------------------

def test_the_real_pipeline_runs_its_stages_in_a_consistent_order():
    """Compiling any task exercises every `order.run` call, so a stage moved
    above one it depends on fails here rather than in a plan.

    Both a classical and a temporal task are compiled: the duration-scale
    stages only have work to do on the latter, but the order is asserted for
    both because the constraint holds either way.
    """
    from test_planner import counter_problem, numeric_counter_problem
    from aspplanners.plasp.planner import PLASPPlanner

    for problem in (counter_problem(), numeric_counter_problem()):
        PLASPPlanner(problem, "seq")    # must not raise PipelineOrderError


def test_a_temporal_task_compiles_in_a_consistent_order():
    import os
    from unified_planning.io import PDDLReader
    from aspplanners.plasp.planner import PLASPPlanner

    pddl_dir = os.path.join(os.path.dirname(__file__), "pddl", "matchcellar")
    problem = PDDLReader().parse_problem(os.path.join(pddl_dir, "domain.pddl"),
                                         os.path.join(pddl_dir, "problem.pddl"))
    PLASPPlanner(problem, "seq")        # must not raise PipelineOrderError
