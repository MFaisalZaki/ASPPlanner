"""A static value as the *coefficient* of a fluent the task writes.

The other half of the fold. :mod:`test_static_linearisation` covers a product
whose factors are all static, which collapses to a number; here one factor is
static and the other is state, so what the grounder resolves is the coefficient
and the fluent stays an ordinary term over its ``numval``. TPP-Metric's
``(* (- (request ?g) (bought ?g)) (price ?g ?m))`` is the shape.

The same fixture carries the coefficient that *cannot* be looked up --
fo-sailing's ``(* (rate ?t) 1.5)`` on a fluent an action writes -- because the
two are settled by opposite means and a task routinely wants both: one by
reading the initial state, the other by storing the effect's target twice as
fine. Both are stated with parameters, which is what makes them exercise the
per-binding path rather than the constant-folding one.
"""

import os

import aspplanners  # noqa: F401 -- registers the engine
from unified_planning.engines import PlanGenerationResultStatus as Status
from unified_planning.io import PDDLReader

from aspplanners.plasp.planner import PLASPPlanner
from test_planner import assert_plan_is_over_original_problem

PDDL_DIR = os.path.join(os.path.dirname(__file__), "pddl")


def tradepost():
    return PDDLReader().parse_problem(
        os.path.join(PDDL_DIR, "tradepost", "domain.pddl"),
        os.path.join(PDDL_DIR, "tradepost", "problem.pddl"),
    )


def effect(planner, action):
    lines = [line for line in planner.compiled_task.fact_lines
             if line.startswith(("numEffect(", "numEffectExpr(")) and f'"{action}"' in line]
    assert lines, f"{action} emitted no numeric effect"
    return lines


def test_the_task_solves_and_the_plan_prices_the_order_correctly():
    """`spent <= 12` is only reachable by buying at the cheaper market, and
    `moved >= 3` only by getting the 3/2 exactly right -- rounded down it is
    unreachable, rounded up it fails UP's validator, which evaluates the task in
    its own units."""
    task = tradepost()
    planner = PLASPPlanner(task)
    assert planner.compiled_task.numeric_scale == 2

    plan = planner.plan(max_horizon=8)
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    names = [ai.action.name for ai in plan.actions]
    assert "drive" in names and "buy" in names, names
    assert_plan_is_over_original_problem(task, plan)


def test_a_static_price_becomes_the_coefficient_of_a_written_fluent():
    """`price` is resolved into the coefficient; `bought`, which the task
    writes, stays a term the encoding reads off its numval at the step before."""
    planner = PLASPPlanner(tradepost())
    spent = [line for line in effect(planner, "buy") if 'variable(("spent"))' in line]
    assert len(spent) == 1, spent
    assert 'variable(("bought",G)),-RAWSTAT0' in spent[0], spent[0]
    assert 'initialState(variable(("price",G,M))' in spent[0], spent[0]


def test_a_coefficient_that_cannot_be_looked_up_is_cleared_by_the_target():
    """`rate` is written, so nothing can be read out of the initial state. The
    3/2 becomes the whole coefficient 3 because `moved` is stored twice as
    fine -- a factor on that fluent alone, which is what one task-wide factor
    could not have given (it would have scaled `rate` with it)."""
    planner = PLASPPlanner(tradepost())
    haul = effect(planner, "haul")
    assert len(haul) == 1, haul
    assert 'expr(sum((variable(("rate",T)),3)), 0)' in haul[0], haul[0]
    assert "RAWSTAT" not in haul[0], haul[0]
    # `rate` itself keeps the task's own units: only `moved` needed the grid.
    assert any('value(variable(("rate",constant("van"))), 1))' in line
               for line in planner.compiled_task.fact_lines)
