"""Products that are linear *in the state* because one factor never changes.

``(* (distance ?c1 ?c2) (burn ?a))`` is not a linear expression, and no integer
encoding states a product of two fluents. But neither of these is state: no
action of the task writes them, so their values are whatever the initial state
says and the grounder can read them out of it once per parameter binding. What
is left is an ordinary coefficient.

This is the largest refusal family in the benchmark, and the reason it is worth
its own module is that it is *lifted*: the fold has to instantiate per parameter
binding, carry its ``initialState`` lookup in the body of every rule that names
it, and survive the per-fluent storage factors of
:mod:`aspplanners.plasp.rescale`. Every plan below is validated against the
original problem, which is what makes these more than assertions about fact
text -- UP evaluates the product in the task's own units.
"""

import os

import pytest

import aspplanners  # noqa: F401 -- registers the engine
from unified_planning.engines import PlanGenerationResultStatus as Status
from unified_planning.io import PDDLReader

from aspplanners.plasp.planner import PLASPPlanner
from test_planner import assert_plan_is_over_original_problem

PDDL_DIR = os.path.join(os.path.dirname(__file__), "pddl")


def parse_case(problem_file):
    return PDDLReader().parse_problem(
        os.path.join(PDDL_DIR, "zenoflight", "domain.pddl"),
        os.path.join(PDDL_DIR, "zenoflight", problem_file),
    )


def facts(planner, *prefixes):
    return sorted(line for line in planner.compiled_task.fact_lines
                  if line.startswith(prefixes))


# ---------------------------------------------------------------------------
# The plan, which is where a mis-scaled product would show
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("problem_file,scale", [
    ("problem.pddl", 1),
    ("problem-fractional.pddl", 2),
], ids=["integral", "fractional"])
def test_a_static_product_prices_each_leg_and_forces_the_refuel(problem_file, scale):
    """Two legs the plane cannot fly on one tank, at prices only the product
    gives. Get the product wrong by any factor and either the refuel is
    unnecessary (so the 3-step plan is not shortest) or the second leg is
    unaffordable (so there is no plan at all) -- and UP's validator, which
    evaluates the task in its own units, rejects either."""
    task = parse_case(problem_file)
    planner = PLASPPlanner(task)
    assert planner.compiled_task.numeric_scale == scale

    plan = planner.plan(max_horizon=8)
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert sorted(ai.action.name for ai in plan.actions) == ["fly", "fly", "refuel"]
    assert_plan_is_over_original_problem(task, plan)


def test_the_product_is_read_out_of_the_initial_state_per_binding():
    """The fold's shape: the coefficient is a clingo term over two lookups, and
    every rule naming them carries the ``initialState`` atoms that bind them --
    which is what makes it one value per ``(?a, ?c1, ?c2)`` rather than one for
    the whole task."""
    planner = PLASPPlanner(parse_case("problem.pddl"))
    folded = [line for line in facts(planner, "numEffect(", "numPrecondition(")
              if '"fly"' in line]
    assert len(folded) == 3, folded   # two deltas and the fuel guard
    for line in folded:
        head, _, body = line.partition(":-")
        assert "RAWSTAT0*RAWSTAT1" in head, f"the product did not fold: {line}"
        for variable in ("RAWSTAT0", "RAWSTAT1"):
            assert f"value(variable((" in body and variable in body, (
                f"{variable} is not bound by an initialState lookup: {line}")
        assert 'initialState(variable(("burn",A))' in body
        assert 'initialState(variable(("distance",C1,C2))' in body


def test_a_static_fluent_read_linearly_keeps_its_numval_reading():
    """Only a product folds. ``capacity`` is static too, but ``refuel`` reads it
    linearly, so it stays an ordinary term over its ``numval`` -- a task that
    was encodable before this existed emits what it always did."""
    planner = PLASPPlanner(parse_case("problem.pddl"))
    assigns = facts(planner, "numAssignExpr(")
    assert len(assigns) == 1, assigns
    assert 'expr(variable(("capacity",A)), 0)' in assigns[0], assigns[0]
    assert "RAWSTAT" not in assigns[0], assigns[0]


def test_the_static_product_needs_no_compiler_and_no_extra_layer():
    planner = PLASPPlanner(parse_case("problem.pddl"))
    assert planner.compilationlist == []
    assert planner.layers == ("core", "numeric")
