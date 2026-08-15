"""Tasks whose numeric *values* are fractional.

clingo terms are integers, so `(increase (x) 0.5)` cannot be encoded as it
stands; `aspplanners.plasp.rescale` multiplies the task's values by the least
factor that makes them whole. Every plan here is validated against the
*original*, unscaled problem, which is what makes the rescaling more than an
assertion about the fact text: UP evaluates the plan in the task's own units.
"""

from fractions import Fraction

import pytest

import aspplanners  # noqa: F401 -- registers the engine
from unified_planning.engines import PlanGenerationResultStatus as Status
from unified_planning.shortcuts import (
    BoolType,
    Div,
    Fluent,
    GE,
    InstantaneousAction,
    IntType,
    LE,
    OneshotPlanner,
    Problem,
    RealType,
)

from aspplanners.plasp.encoder import PLASPEncoder
from aspplanners.plasp.planner import PLASPPlanner
from aspplanners.plasp.rescale import MAX_NUMERIC_SCALE, scale_numeric_constants
from tests.test_planner import (
    assert_plan_is_over_original_problem,
    counter_problem,
    numeric_counter_problem,
    robot_line_problem,
    rover_recharge_problem,
)


def fractional_counter_problem(step=Fraction(1, 2), threshold=Fraction(3, 2),
                               start=Fraction(0), fluent_type=None):
    """`tick` moves a real-valued counter by `step`; `finish` needs it past
    `threshold`. The integral version of this is `numeric_counter_problem`."""
    counter = Fluent("counter", fluent_type or RealType())
    done = Fluent("done", BoolType())

    tick = InstantaneousAction("tick")
    if step >= 0:
        tick.add_increase_effect(counter(), step)
        finish_guard = GE(counter(), threshold)
    else:
        tick.add_decrease_effect(counter(), -step)
        finish_guard = LE(counter(), threshold)

    finish = InstantaneousAction("finish")
    finish.add_precondition(finish_guard)
    finish.add_effect(done(), True)

    problem = Problem("fractional_counter")
    problem.add_fluent(counter, default_initial_value=start)
    problem.add_fluent(done, default_initial_value=False)
    problem.add_action(tick)
    problem.add_action(finish)
    problem.add_goal(done())
    return problem


def numeric_facts(compiled, *prefixes):
    return sorted(line for line in compiled.fact_lines if line.startswith(prefixes))


# ---------------------------------------------------------------------------
# The reported failure: a fractional effect delta
# ---------------------------------------------------------------------------

def test_a_fractional_increase_becomes_an_integer_delta():
    """`increase 1/2` towards a `>= 3/2` guard: the task is multiplied by 2, so
    the delta is 1 and the guard 3, and it still takes three ticks."""
    problem = fractional_counter_problem()
    planner = PLASPPlanner(problem, "seq")
    assert planner.compiled_task.numeric_scale == 2

    assert numeric_facts(planner.compiled_task, "numEffect(") == [
        'numEffect(action(("tick")), variable(("counter")), 1) :- action(action(("tick"))).'
    ]
    assert any(", le, expr(none, 3)," in line
               for line in numeric_facts(planner.compiled_task, "numPrecondition("))

    plan = planner.plan()
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert [ai.action.name for ai in plan.actions] == ["tick", "tick", "tick", "finish"]
    # The real check: UP evaluates this against the task's own 1/2 and 3/2.
    assert_plan_is_over_original_problem(problem, plan)


def test_a_fractional_decrease_is_scaled_and_keeps_its_direction():
    """Sailing's shape: a fractional `decrease` towards a `<=` guard."""
    problem = fractional_counter_problem(step=Fraction(-3, 2), threshold=Fraction(-3),
                                         start=Fraction(0))
    planner = PLASPPlanner(problem, "seq")
    assert planner.compiled_task.numeric_scale == 2
    # -3/2 doubled is -3, and the sign survives -- clingo's own `/` would have
    # truncated -3/2 towards zero, to -1.
    assert numeric_facts(planner.compiled_task, "numEffect(") == [
        'numEffect(action(("tick")), variable(("counter")), -3) :- action(action(("tick"))).'
    ]

    plan = planner.plan()
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert [ai.action.name for ai in plan.actions] == ["tick", "tick", "finish"]
    assert_plan_is_over_original_problem(problem, plan)


def test_the_scale_factor_is_the_least_common_denominator():
    """Halves and thirds together need sixths, and nothing finer."""
    counter = Fluent("counter", RealType())
    done = Fluent("done", BoolType())

    half = InstantaneousAction("half")
    half.add_increase_effect(counter(), Fraction(1, 2))
    third = InstantaneousAction("third")
    third.add_increase_effect(counter(), Fraction(1, 3))
    finish = InstantaneousAction("finish")
    finish.add_precondition(GE(counter(), Fraction(5, 6)))
    finish.add_effect(done(), True)

    problem = Problem("sixths")
    problem.add_fluent(counter, default_initial_value=Fraction(0))
    problem.add_fluent(done, default_initial_value=False)
    for action in (half, third, finish):
        problem.add_action(action)
    problem.add_goal(done())

    planner = PLASPPlanner(problem, "seq")
    assert planner.compiled_task.numeric_scale == 6
    deltas = {line.split("variable")[0]: line for line in numeric_facts(planner.compiled_task, "numEffect(")}
    assert any('("half")), variable(("counter")), 3)' in line for line in deltas.values())
    assert any('("third")), variable(("counter")), 2)' in line for line in deltas.values())

    plan = planner.plan()
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert_plan_is_over_original_problem(problem, plan)


def test_a_fractional_initial_value_is_scaled_not_truncated():
    """The silent half of the bug: an initial value used to be interpolated as
    UP's `str()`, and `value(V, 7/2)` is integer *division* to clingo -- it
    grounds to 3, so the task quietly started somewhere else."""
    problem = fractional_counter_problem(start=Fraction(7, 2), threshold=Fraction(9, 2))
    planner = PLASPPlanner(problem, "seq")

    initial = numeric_facts(planner.compiled_task, "initialState(")
    assert any('value(variable(("counter")), 7))' in line for line in initial), initial
    assert not any("/" in line for line in planner.compiled_task.fact_lines)

    # 7/2 + 1/2 == 4 < 9/2, so it takes two ticks, not one.
    plan = planner.plan()
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert [ai.action.name for ai in plan.actions] == ["tick", "tick", "finish"]
    assert_plan_is_over_original_problem(problem, plan)


def test_a_non_integral_initial_value_reaching_the_fact_builder_raises():
    """Whatever else changes, the builder must never emit `value(V, 3/2)` again:
    clingo reads it as integer division rather than as a rational."""
    from aspplanners.plasp.facts import ASPInitialState

    problem = fractional_counter_problem()
    em = problem.environment.expression_manager
    counter = problem.fluent("counter")()

    with pytest.raises(NotImplementedError, match="Non-integral"):
        str(ASPInitialState(counter, em.Real(Fraction(3, 2))))


# ---------------------------------------------------------------------------
# The pass is invisible to a task that does not need it
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("build", [
    robot_line_problem,
    numeric_counter_problem,
    rover_recharge_problem,
    counter_problem,
], ids=["robot_line", "numeric_counter", "rover_recharge", "counter"])
def test_an_integral_task_is_left_exactly_alone(build):
    """No fractional value means no factor, and no rewrite at all: UP compares
    a problem by its initial state, goals and actions, so this pins that the
    pass did not touch a single expression."""
    problem = build()
    before = problem.clone()

    assert scale_numeric_constants(problem) == 1
    assert problem == before


def test_the_encoder_reports_a_scale_of_one_for_an_integral_task():
    assert PLASPPlanner(numeric_counter_problem(), "seq").compiled_task.numeric_scale == 1


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_a_bounded_numeric_type_with_a_fractional_value_is_rejected():
    """`integer[0, 10]` cannot be scaled with the values it bounds -- the bound
    is part of a type UP keys the fluent by -- so say so rather than letting
    `set_initial_value` fail with an unrelated type error."""
    problem = fractional_counter_problem(fluent_type=RealType(0, 10))
    with pytest.raises(NotImplementedError, match="bounded numeric type"):
        PLASPPlanner(problem, "seq")


def test_the_scale_factor_is_capped():
    """Two effects that each ask for a finer grid than the other leaves: `x`
    needs twice `y`'s resolution and `y` twice `x`'s, so the factors chase each
    other upwards and the cap is what stops them."""
    x = Fluent("x", RealType())
    y = Fluent("y", RealType())
    done = Fluent("done", BoolType())

    stir = InstantaneousAction("stir")
    stir.add_increase_effect(x(), Div(y(), 2))
    stir.add_increase_effect(y(), Div(x(), 2))

    finish = InstantaneousAction("finish")
    finish.add_precondition(GE(x(), 1))
    finish.add_effect(done(), True)

    problem = Problem("chasing")
    problem.add_fluent(x, default_initial_value=Fraction(1))
    problem.add_fluent(y, default_initial_value=Fraction(1))
    problem.add_fluent(done, default_initial_value=False)
    problem.add_action(stir)
    problem.add_action(finish)
    problem.add_goal(done())

    with pytest.raises(NotImplementedError, match="caps it at"):
        PLASPPlanner(problem, "seq")


def test_a_comparison_constant_does_not_force_a_scale_factor():
    """The other side of the same coin: a fractional constant in a *comparison*
    puts no factor on anything, because both sides are multiplied through
    instead. Denominators this coprime used to blow the cap between them; now
    only the effect's own 1/999983 is a factor, and the guard is stated as the
    whole-number comparison `999979 * counter >= 3 * 999983`."""
    problem = fractional_counter_problem(step=Fraction(1, 999983),
                                         threshold=Fraction(3, 999979))
    assert MAX_NUMERIC_SCALE < 999983 * 999979
    planner = PLASPPlanner(problem, "seq")
    assert planner.compiled_task.numeric_scale == 999983

    plan = planner.plan()
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert [ai.action.name for ai in plan.actions] == ["tick"] * 4 + ["finish"]
    assert_plan_is_over_original_problem(problem, plan)


def test_a_value_too_large_for_a_clingo_term_is_rejected():
    """clingo terms are signed 32-bit and wrap round *silently*, so an oversized
    constant would return a wrong plan rather than fail."""
    counter = Fluent("counter", IntType())
    done = Fluent("done", BoolType())
    finish = InstantaneousAction("finish")
    finish.add_precondition(GE(counter(), 2 ** 40))
    finish.add_effect(done(), True)

    problem = Problem("huge")
    problem.add_fluent(counter, default_initial_value=0)
    problem.add_fluent(done, default_initial_value=False)
    problem.add_action(finish)
    problem.add_goal(done())

    with pytest.raises(NotImplementedError, match="32-bit"):
        PLASPPlanner(problem, "seq")


def test_the_engine_solves_a_fractional_task_end_to_end():
    problem = fractional_counter_problem()
    with OneshotPlanner(name="PLASPPlanner") as planner:
        assert planner.supports(problem.kind)
        result = planner.solve(problem)
    assert result.status == Status.SOLVED_SATISFICING, \
        [str(m) for m in result.log_messages or []]
    assert_plan_is_over_original_problem(problem, result.plan)
