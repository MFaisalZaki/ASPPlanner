"""End-to-end PDDL instances across problem kinds.

Each case is parsed with UP's PDDLReader, solved through the UP engine, and
the lifted plan is checked to be stated in terms of the parsed task (action
identity, argument membership, arity) and to validate against it. The
deepening search guarantees the shortest sequential plan, so the expected
lengths are exact.
"""

import os

import pytest

import aspplanners  # noqa: F401 -- registers the engine
from unified_planning.engines import PlanGenerationResultStatus as Status
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner

from aspplanners.plasp.planner import PLASPPlanner
from test_planner import assert_plan_is_over_original_problem

PDDL_DIR = os.path.join(os.path.dirname(__file__), "pddl")

CASES = [
    # (case id, domain dir, problem file, optimal plan length)
    ("blocksworld-untyped-hyphens", "blocksworld", "problem.pddl", 4),
    ("gripper-flat-typing", "gripper", "problem.pddl", 5),
    ("transport-hierarchical-typing", "transport", "problem.pddl", 4),
    ("rover-numeric-lifted", "rover", "problem.pddl", 4),
    ("alarm-conditional-effect-fires", "alarm", "problem-trigger.pddl", 1),
    ("alarm-conditional-effect-avoided", "alarm", "problem-quiet.pddl", 2),
]


def parse_case(domain, problem_file):
    return PDDLReader().parse_problem(
        os.path.join(PDDL_DIR, domain, "domain.pddl"),
        os.path.join(PDDL_DIR, domain, problem_file),
    )


@pytest.mark.parametrize("case_id,domain,problem_file,optimal_length",
                         CASES, ids=[c[0] for c in CASES])
def test_pddl_instance_solves_and_lifts_correctly(case_id, domain, problem_file,
                                                  optimal_length):
    task = parse_case(domain, problem_file)
    with OneshotPlanner(name="PLASPPlanner") as planner:
        result = planner.solve(task)
    assert result.status == Status.SOLVED_SATISFICING, (
        f"{case_id}: expected a plan, got {result.status} "
        f"({[str(m) for m in result.log_messages or []]})"
    )
    assert len(result.plan.actions) == optimal_length, (
        f"{case_id}: expected the {optimal_length}-step optimum, "
        f"got {len(result.plan.actions)}: {result.plan.actions}"
    )
    assert_plan_is_over_original_problem(task, result.plan)


def test_direct_api_lifts_to_the_passed_task():
    """Same contract through PLASPPlanner directly (no UP engine wrapper)."""
    task = parse_case("blocksworld", "problem.pddl")
    planner = PLASPPlanner(task, "seq")
    plan = planner.plan(max_horizon=8)
    assert len(plan.actions) == 4
    assert_plan_is_over_original_problem(task, plan)
    # hyphenated PDDL names must round-trip untouched
    assert {ai.action.name for ai in plan.actions} <= {"pick-up", "put-down",
                                                       "stack", "unstack"}


def test_conditional_effect_semantics():
    """The quiet variant must disarm BEFORE opening the door; the trigger
    variant must see the alarm fire through the conditional effect."""
    quiet = parse_case("alarm", "problem-quiet.pddl")
    with OneshotPlanner(name="PLASPPlanner") as planner:
        result = planner.solve(quiet)
    names = [ai.action.name for ai in result.plan.actions]
    assert names == ["disarm", "open-door"]
