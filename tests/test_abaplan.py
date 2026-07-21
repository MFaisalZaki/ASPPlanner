"""End-to-end tests for the ABAPlan backend (STRIPS-to-ABA, solved with
aspforaba).

The ABA backend is optional (the ``aba`` extra); skip the whole module cleanly
when its solver dependency is not installed.
"""

import pytest

pytest.importorskip("aspforaba", reason="ABA backend requires the optional `aba` extra")

import aspplanner  # noqa: F401,E402 -- registers the ABAPlanner engine
from unified_planning.engines import PlanGenerationResultStatus as Status  # noqa: E402
from unified_planning.shortcuts import (  # noqa: E402
    BoolType,
    Fluent,
    InstantaneousAction,
    Not,
    OneshotPlanner,
)

from aspplanner.abaplan.planner import ABAPlan  # noqa: E402

from test_planner import (  # noqa: E402
    assert_plan_is_over_original_problem,
    numeric_counter_problem,
    robot_line_problem,
)


# ---------------------------------------------------------------------------
# Direct API
# ---------------------------------------------------------------------------

def test_direct_solves_and_maps_back_to_original_problem():
    problem = robot_line_problem(n_locations=4)
    plan = ABAPlan(problem).plan(max_horizon=6)
    assert [ai.action.name for ai in plan.actions] == ["move", "move", "move"]
    assert_plan_is_over_original_problem(problem, plan)


def test_hyphenated_names_map_back():
    problem = robot_line_problem(n_locations=4, sep="-")
    plan = ABAPlan(problem).plan(max_horizon=6)
    assert_plan_is_over_original_problem(problem, plan)
    # the original vocabulary is hyphenated; the mapped-back plan must be too
    plan_str = str(plan)
    assert "loc-0" in plan_str and "loc_0" not in plan_str


def test_goal_already_satisfied_is_empty_plan():
    problem = robot_line_problem(n_locations=4)
    problem.clear_goals()
    robot_at = problem.fluent("robot_at")
    problem.add_goal(robot_at(problem.object("loc_0")))  # holds initially
    plan = ABAPlan(problem).plan(max_horizon=6)
    assert plan.actions == []


def test_multiple_goals():
    problem = robot_line_problem(n_locations=4)
    robot_at = problem.fluent("robot_at")
    connected = problem.fluent("connected")
    problem.clear_goals()
    problem.add_goal(robot_at(problem.object("loc_3")))
    problem.add_goal(connected(problem.object("loc_0"), problem.object("loc_1")))
    plan = ABAPlan(problem).plan(max_horizon=6)
    assert_plan_is_over_original_problem(problem, plan)
    assert [ai.action.name for ai in plan.actions] == ["move", "move", "move"]


def test_negative_precondition():
    """A `Not(...)` precondition is compiled away by the pipeline's
    negative-conditions remover before the STRIPS-to-ABA reduction."""
    problem = robot_line_problem(n_locations=4)
    robot_at = problem.fluent("robot_at")
    charged = Fluent("charged", BoolType())
    problem.add_fluent(charged, default_initial_value=False)
    charge = InstantaneousAction("charge")
    charge.add_precondition(Not(robot_at(problem.object("loc_3"))))
    charge.add_effect(charged(), True)
    problem.add_action(charge)
    problem.add_goal(charged())

    plan = ABAPlan(problem).plan(max_horizon=6)
    assert_plan_is_over_original_problem(problem, plan)
    assert "charge" in [ai.action.name for ai in plan.actions]


def test_unsolvable_within_horizon_returns_empty_plan():
    problem = robot_line_problem(n_locations=4, connected_line=False)
    plan = ABAPlan(problem).plan(max_horizon=4)
    assert plan.actions == []


def test_numeric_task_is_unsupported():
    """ABAPlan is STRIPS only; a numeric task fails to compile (the reachability
    grounder rejects it) rather than silently producing a bogus encoding."""
    with pytest.raises(Exception):
        ABAPlan(numeric_counter_problem(threshold=2))


def test_status_reflects_outcome():
    """plan() records the outcome in `status`/`logs` (like ASPPlanner)."""
    solved = ABAPlan(robot_line_problem(n_locations=4))
    solved.plan(max_horizon=6)
    assert solved.status == Status.SOLVED_SATISFICING

    unsolved = ABAPlan(robot_line_problem(n_locations=4, connected_line=False))
    unsolved.plan(max_horizon=4)
    assert unsolved.status == Status.UNSOLVABLE_INCOMPLETELY
    assert isinstance(unsolved.logs, list) and unsolved.logs


# ---------------------------------------------------------------------------
# UP engine interface
# ---------------------------------------------------------------------------

def test_up_engine_solves():
    problem = robot_line_problem(n_locations=4)
    with OneshotPlanner(name="ABAPlanner") as planner:
        result = planner.solve(problem)
    assert result.status == Status.SOLVED_SATISFICING
    assert len(result.plan.actions) == 3
    assert_plan_is_over_original_problem(problem, result.plan)


def test_up_engine_unsolvable_reports_status():
    problem = robot_line_problem(n_locations=4, connected_line=False)
    with OneshotPlanner(name="ABAPlanner", params={"max_horizon": 4}) as planner:
        result = planner.solve(problem)
    assert result.status == Status.UNSOLVABLE_INCOMPLETELY
    assert result.plan is None
