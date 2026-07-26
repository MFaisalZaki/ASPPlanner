"""Numeric comparisons over *linear* expressions.

A comparison side used to hold one fluent and a constant, because the fact
`expr(V, C)` had room for exactly that. It now holds any number of `K*V` terms,
carried in numTerm/3 facts and summed by the encoding's numValue/3 aggregate --
so `(+ (x ?b) (y ?b))`, `(- (y ?b) (x ?b))` and `(* 2 (x ?f))` are ordinary
expressions rather than the three separate NotImplementedErrors they were.

The single-fluent and constant-only shapes still render exactly as they did, so
those facts are unchanged for every task that never needed more.
"""

from fractions import Fraction

import pytest

import aspplanners  # noqa: F401 -- registers the engine
from unified_planning.engines import PlanGenerationResultStatus as Status
from unified_planning.shortcuts import (
    BoolType,
    Fluent,
    GE,
    InstantaneousAction,
    IntType,
    LE,
    Minus,
    Object,
    Plus,
    Problem,
    RealType,
    Times,
    UserType,
)

from aspplanners.plasp.facts import ASPNumComparison
from aspplanners.plasp.planner import PLASPPlanner
from tests.test_planner import assert_plan_is_over_original_problem


def comparison(build, *fluents, fluent_type=None):
    """`build(*fluent_exps)` rendered as the encoder would render it.

    Returns the ASPNumComparison's payload text -- `op, expr(...), expr(...)` --
    which is what goes inside a numPrecondition/numGoal fact.
    """
    problem = Problem("cmp")
    exps = []
    for name in fluents:
        fluent = Fluent(name, fluent_type or RealType())
        problem.add_fluent(fluent, default_initial_value=Fraction(0))
        exps.append(fluent())
    expression = build(*exps)
    op = "lt" if expression.is_lt() else "le"
    return ASPNumComparison(expression, op)


def facts_of(compiled, *prefixes):
    return sorted(line for line in compiled.fact_lines if line.startswith(prefixes))


# ---------------------------------------------------------------------------
# Rendering: the shapes a side can take
# ---------------------------------------------------------------------------

def test_a_constant_only_side_still_renders_as_expr_none():
    cmp = comparison(lambda x: LE(2, x), "x")
    assert str(cmp) == 'le, expr(none, 2), expr(variable(("x")), 0)'
    assert 'numConst(expr(none, 2), 2)' in cmp.declaration_heads()


def test_a_single_fluent_side_still_renders_as_expr_v_c():
    """The shape every task that worked before produces, unchanged."""
    cmp = comparison(lambda x: LE(2, Plus(x, 1)), "x")
    assert str(cmp) == 'le, expr(none, 2), expr(variable(("x")), 1)'
    assert cmp.declaration_heads() == [
        'numConst(expr(none, 2), 2)',
        'numConst(expr(variable(("x")), 1), 1)',
        'numTerm(expr(variable(("x")), 1), 1, variable(("x")))',
    ]


def test_two_fluents_on_one_side_become_a_sum_of_terms():
    """sailing's `(>= (+ (x ?b) (y ?b)) (d ?t))`."""
    cmp = comparison(lambda x, y, d: GE(Plus(x, y), d), "x", "y", "d")
    assert str(cmp) == (
        'le, expr(variable(("d")), 0), '
        'expr(sum((variable(("x")),1),(variable(("y")),1)), 0)')
    assert 'numTerm(expr(sum((variable(("x")),1),(variable(("y")),1)), 0), 1, variable(("x")))' \
        in cmp.declaration_heads()
    assert 'numTerm(expr(sum((variable(("x")),1),(variable(("y")),1)), 0), 1, variable(("y")))' \
        in cmp.declaration_heads()


def test_a_subtracted_fluent_becomes_a_negative_coefficient():
    """sailing's `(>= (- (y ?b) (x ?b)) (d ?t))`."""
    cmp = comparison(lambda x, y, d: GE(Minus(y, x), d), "x", "y", "d")
    assert 'sum((variable(("x")),-1),(variable(("y")),1))' in str(cmp)


def test_a_constant_coefficient_is_kept_as_the_coefficient():
    """farmland's `(* 2 (x ?f))`: the 2 multiplies the fluent, it is not a value
    of its own, so it lands in the term rather than being scaled with one."""
    cmp = comparison(lambda x: GE(Times(2, x), 6), "x")
    assert str(cmp) == 'le, expr(none, 6), expr(sum((variable(("x")),2)), 0)'


def test_a_fractional_coefficient_multiplies_its_own_comparison_out():
    """`(>= (* 1/2 x) 3)` is `x >= 6`: the comparison is multiplied through by
    the denominator, which is exact and preserves the direction."""
    cmp = comparison(lambda x: GE(Times(Fraction(1, 2), x), 3), "x")
    assert str(cmp) == 'le, expr(none, 6), expr(variable(("x")), 0)'


def test_a_fluent_occurring_twice_on_a_side_accumulates():
    cmp = comparison(lambda x: GE(Plus(x, x), 4), "x")
    assert str(cmp) == 'le, expr(none, 4), expr(sum((variable(("x")),2)), 0)'


def test_terms_that_cancel_leave_a_constant_only_side():
    cmp = comparison(lambda x: GE(Minus(x, x), 0), "x")
    assert str(cmp) == 'le, expr(none, 0), expr(none, 0)'


def test_the_rendering_does_not_depend_on_the_order_of_the_terms():
    """Sides are sorted, so the same expression written two ways is the same
    fact -- which is what lets two owners share one numValue."""
    assert str(comparison(lambda x, y: GE(Plus(x, y), 1), "x", "y")) == \
        str(comparison(lambda x, y: GE(Plus(y, x), 1), "x", "y"))


# ---------------------------------------------------------------------------
# What is genuinely non-linear still raises
# ---------------------------------------------------------------------------

def test_a_product_of_two_fluents_is_rejected():
    with pytest.raises(NotImplementedError, match="product of two numeric fluents"):
        comparison(lambda x, y: GE(Times(x, y), 1), "x", "y")


def test_dividing_by_a_fluent_is_rejected():
    from unified_planning.shortcuts import Div

    with pytest.raises(NotImplementedError, match="Dividing by a numeric fluent"):
        comparison(lambda x, y: GE(Div(x, y), 1), "x", "y")


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def two_counter_problem(guard):
    """Two counters, one action stepping each, and a goal `guard(x, y)`."""
    x = Fluent("x", IntType())
    y = Fluent("y", IntType())

    bump_x = InstantaneousAction("bump_x")
    bump_x.add_increase_effect(x(), 1)
    bump_y = InstantaneousAction("bump_y")
    bump_y.add_increase_effect(y(), 1)

    problem = Problem("two_counters")
    problem.add_fluent(x, default_initial_value=0)
    problem.add_fluent(y, default_initial_value=0)
    problem.add_action(bump_x)
    problem.add_action(bump_y)
    problem.add_goal(guard(x(), y()))
    return problem


def test_a_two_fluent_goal_is_summed_by_the_encoding():
    """`x + y >= 3` needs three bumps in any mix -- the aggregate has to add the
    two fluents, not read either one alone."""
    problem = two_counter_problem(lambda x, y: GE(Plus(x, y), 3))
    planner = PLASPPlanner(problem, "seq")
    plan = planner.plan()
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert len(plan.actions) == 3
    assert_plan_is_over_original_problem(problem, plan)


def test_a_two_fluent_goal_that_needs_both_fluents():
    """`x + 2y >= 4` with only `+1` steps: the cheapest plan uses the coefficient,
    so a solver that ignored it would return the wrong length."""
    problem = two_counter_problem(lambda x, y: GE(Plus(x, Times(2, y)), 4))
    planner = PLASPPlanner(problem, "seq")
    plan = planner.plan()
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert [ai.action.name for ai in plan.actions] == ["bump_y", "bump_y"]
    assert_plan_is_over_original_problem(problem, plan)


def test_a_two_fluent_precondition_is_summed_by_the_encoding():
    problem = two_counter_problem(lambda x, y: GE(Plus(x, y), 0))
    problem.clear_goals()
    done = Fluent("done", BoolType())
    problem.add_fluent(done, default_initial_value=False)
    finish = InstantaneousAction("finish")
    finish.add_precondition(GE(Plus(problem.fluent("x")(), problem.fluent("y")()), 2))
    finish.add_effect(done(), True)
    problem.add_action(finish)
    problem.add_goal(done())

    planner = PLASPPlanner(problem, "seq")
    plan = planner.plan()
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert [ai.action.name for ai in plan.actions][-1] == "finish"
    assert len(plan.actions) == 3
    assert_plan_is_over_original_problem(problem, plan)


def test_a_multi_fluent_literal_in_a_disjunctive_goal():
    """The goalOrDisjunctNum path, which reads numValue the same way."""
    from unified_planning.shortcuts import Or

    problem = two_counter_problem(
        lambda x, y: Or(GE(Plus(x, y), 2), GE(Minus(x, y), 5)))
    planner = PLASPPlanner(problem, "seq")
    assert any("goalOrDisjunctNum(" in line for line in planner.compiled_task.fact_lines)

    plan = planner.plan()
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert len(plan.actions) == 2       # the x+y >= 2 disjunct is the cheap one
    assert_plan_is_over_original_problem(problem, plan)


def test_a_lifted_multi_fluent_precondition_binds_its_parameters():
    """sailing's actual shape: the terms are lifted, so the numTerm rules have to
    carry the action's own body or the grounder could not bind them."""
    Boat = UserType("boat")
    x = Fluent("x", IntType(), b=Boat)
    y = Fluent("y", IntType(), b=Boat)
    saved = Fluent("saved", BoolType(), b=Boat)

    north = InstantaneousAction("north", b=Boat)
    north.add_increase_effect(y(north.parameter("b")), 1)

    save = InstantaneousAction("save", b=Boat)
    b = save.parameter("b")
    save.add_precondition(GE(Minus(y(b), x(b)), 2))
    save.add_effect(saved(b), True)

    problem = Problem("lifted_sailing")
    for fluent in (x, y):
        problem.add_fluent(fluent, default_initial_value=0)
    problem.add_fluent(saved, default_initial_value=False)
    problem.add_action(north)
    problem.add_action(save)
    boat = Object("b0", Boat)
    problem.add_object(boat)
    problem.add_goal(saved(boat))

    planner = PLASPPlanner(problem, "seq")
    lifted = [line for line in facts_of(planner.compiled_task, "numTerm(") if ",B)" in line]
    assert lifted, facts_of(planner.compiled_task, "numTerm(")
    assert all(":- action(action((\"save\",B)))." in line for line in lifted), lifted

    plan = planner.plan()
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert [ai.action.name for ai in plan.actions] == ["north", "north", "save"]
    assert_plan_is_over_original_problem(problem, plan)
