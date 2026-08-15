"""Numeric effects whose value reads the state.

``(increase (x) (y))`` -- a delta that is a linear expression over fluents
rather than a constant -- is what UP flags as FLUENTS_IN_NUMERIC_ASSIGNMENTS,
and any such effect tips a task's kind from SIMPLE_ to GENERAL_NUMERIC_PLANNING.
The encoding evaluates the expression at t-1 (through numValue/3, exactly as a
precondition's sides are) and applies it at t. These tests pin the emitted
facts, the plans -- validated against the original task, so the semantics is
UP's and not just the encoding's -- and the boundary: anything past linear is
still rejected, now at encoding time rather than by the kind check.
"""

from fractions import Fraction

import pytest

import aspplanners  # noqa: F401 -- registers the engine
from unified_planning.engines import PlanGenerationResultStatus as Status
from unified_planning.engines import CompilationKind
from unified_planning.shortcuts import (
    Equals,
    BoolType,
    Div,
    Fluent,
    GE,
    InstantaneousAction,
    IntType,
    LE,
    OneshotPlanner,
    Plus,
    Problem,
    RealType,
    Times,
)

from aspplanners.plasp.planner import PLASPPlanner
from tests.test_planner import assert_plan_is_over_original_problem


def numeric_facts(compiled, *prefixes):
    return sorted(line for line in compiled.fact_lines if line.startswith(prefixes))


def tank_problem(goal_level=6, bucket_size=3):
    """`pour` adds the bucket's current content to the tank and empties it;
    `refill` sets it back. The bucket is written as well as read, so the task's
    kind carries FLUENTS_IN_NUMERIC_ASSIGNMENTS, not just STATIC_..."""
    tank = Fluent("tank", IntType())
    bucket = Fluent("bucket", IntType())
    done = Fluent("done", BoolType())

    pour = InstantaneousAction("pour")
    pour.add_increase_effect(tank(), bucket())
    pour.add_effect(bucket(), 0)

    refill = InstantaneousAction("refill")
    refill.add_effect(bucket(), bucket_size)

    finish = InstantaneousAction("finish")
    finish.add_precondition(GE(tank(), goal_level))
    finish.add_effect(done(), True)

    problem = Problem("tank")
    problem.add_fluent(tank, default_initial_value=0)
    problem.add_fluent(bucket, default_initial_value=bucket_size)
    problem.add_fluent(done, default_initial_value=False)
    for action in (pour, refill, finish):
        problem.add_action(action)
    problem.add_goal(done())
    return problem


def delta_problem(build_delta, start=0, threshold=3, threshold_op=GE,
                  helper_value=1, fluent_type=None):
    """`tick` moves `x` by `build_delta(x, y)`; `finish` needs `x` past the
    threshold. `y` is never written, so it exercises the static-fluent read."""
    x = Fluent("x", fluent_type or IntType())
    y = Fluent("y", fluent_type or IntType())
    done = Fluent("done", BoolType())

    tick = InstantaneousAction("tick")
    build_delta(tick, x, y)

    finish = InstantaneousAction("finish")
    finish.add_precondition(threshold_op(x(), threshold))
    finish.add_effect(done(), True)

    problem = Problem("delta")
    problem.add_fluent(x, default_initial_value=start)
    problem.add_fluent(y, default_initial_value=helper_value)
    problem.add_fluent(done, default_initial_value=False)
    problem.add_action(tick)
    problem.add_action(finish)
    problem.add_goal(done())
    return problem


# ---------------------------------------------------------------------------
# Increase / decrease by a fluent
# ---------------------------------------------------------------------------

def test_a_fluent_delta_is_encoded_as_an_expression_and_solved():
    problem = tank_problem()
    planner = PLASPPlanner(problem, "seq")

    assert numeric_facts(planner.compiled_task, "numEffectExpr(") == [
        'numEffectExpr(action(("pour")), variable(("tank")), '
        'expr(variable(("bucket")), 0)) :- action(action(("pour"))).'
    ]
    # The expression's sides are declared like a comparison's: a numConst for
    # the additive constant, one numTerm per K*V.
    declarations = numeric_facts(planner.compiled_task, "numConst(", "numTerm(")
    assert 'numConst(expr(variable(("bucket")), 0), 0) :- action(action(("pour"))).' \
        in declarations
    assert 'numTerm(expr(variable(("bucket")), 0), 1, variable(("bucket"))) ' \
        ':- action(action(("pour"))).' in declarations

    plan = planner.plan()
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert [ai.action.name for ai in plan.actions] == \
        ["pour", "refill", "pour", "finish"]
    assert_plan_is_over_original_problem(problem, plan)


def test_a_decrease_by_a_fluent_keeps_its_sign():
    def drain(tick, x, y):
        tick.add_decrease_effect(x(), y())

    problem = delta_problem(drain, start=10, threshold=2, threshold_op=LE,
                            helper_value=4)
    planner = PLASPPlanner(problem, "seq")
    # A lone fluent with coefficient -1 cannot use the bare-variable shape,
    # which is reserved for coefficient 1; it goes through sum((V, K)).
    assert numeric_facts(planner.compiled_task, "numEffectExpr(") == [
        'numEffectExpr(action(("tick")), variable(("x")), '
        'expr(sum((variable(("y")),-1)), 0)) :- action(action(("tick"))).'
    ]

    plan = planner.plan()
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert [ai.action.name for ai in plan.actions] == ["tick", "tick", "finish"]
    assert_plan_is_over_original_problem(problem, plan)


def test_a_self_referential_delta_reads_the_step_before():
    """`x += x` doubles: the delta is x's own value at t-1, not at t."""
    def double(tick, x, _y):
        tick.add_increase_effect(x(), x())

    problem = delta_problem(double, start=1, threshold=4)
    planner = PLASPPlanner(problem, "seq")

    plan = planner.plan()
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert [ai.action.name for ai in plan.actions] == ["tick", "tick", "finish"]
    assert_plan_is_over_original_problem(problem, plan)


def test_constant_and_fluent_deltas_on_one_fluent_are_one_expression():
    """`x += y` and `x += 2` in the same action accumulate into a single
    update, exactly as two constant deltas always have."""
    def boost(tick, x, y):
        tick.add_increase_effect(x(), y())
        tick.add_increase_effect(x(), 2)

    problem = delta_problem(boost, threshold=6)
    planner = PLASPPlanner(problem, "seq")

    assert numeric_facts(planner.compiled_task, "numEffectExpr(") == [
        'numEffectExpr(action(("tick")), variable(("x")), '
        'expr(variable(("y")), 2)) :- action(action(("tick"))).'
    ]
    assert numeric_facts(planner.compiled_task, "numEffect(") == []

    plan = planner.plan()
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    # y = 1, so each tick adds 3.
    assert [ai.action.name for ai in plan.actions] == ["tick", "tick", "finish"]
    assert_plan_is_over_original_problem(problem, plan)


def test_a_lifted_delta_grounds_per_binding():
    """`recharge(r): battery(r) += rate(r)` -- the expression's fluent carries
    the action's parameter, so the grounder makes one expression per binding
    and each rover charges at its own rate."""
    from unified_planning.shortcuts import Object, UserType

    Rover = UserType("rover")
    battery = Fluent("battery", IntType(), r=Rover)
    rate = Fluent("rate", IntType(), r=Rover)

    recharge = InstantaneousAction("recharge", r=Rover)
    r = recharge.parameter("r")
    recharge.add_increase_effect(battery(r), rate(r))

    problem = Problem("rated_recharge")
    problem.add_fluent(battery, default_initial_value=0)
    problem.add_fluent(rate, default_initial_value=1)
    slow, fast = Object("slow", Rover), Object("fast", Rover)
    problem.add_objects([slow, fast])
    problem.set_initial_value(rate(fast), 3)
    problem.add_goal(GE(battery(fast), 6))
    problem.add_action(recharge)

    planner = PLASPPlanner(problem, "seq")
    assert numeric_facts(planner.compiled_task, "numEffectExpr(") == [
        'numEffectExpr(action(("recharge",R)), variable(("battery",R)), '
        'expr(variable(("rate",R)), 0)) :- action(action(("recharge",R))).'
    ]

    plan = planner.plan()
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert [(ai.action.name, str(ai.actual_parameters[0])) for ai in plan.actions] == \
        [("recharge", "fast"), ("recharge", "fast")]
    assert_plan_is_over_original_problem(problem, plan)


# ---------------------------------------------------------------------------
# Assignment of an expression
# ---------------------------------------------------------------------------

def test_an_assignment_reading_a_fluent_is_encoded_and_solved():
    """`x := y + 1`, the FLUENTS_IN_NUMERIC_ASSIGNMENTS shape proper."""
    def sync(tick, x, y):
        tick.add_effect(x(), Plus(y(), 1))

    problem = delta_problem(sync, threshold=6, helper_value=5)
    planner = PLASPPlanner(problem, "seq")

    assert numeric_facts(planner.compiled_task, "numAssignExpr(") == [
        'numAssignExpr(action(("tick")), variable(("x")), '
        'expr(variable(("y")), 1)) :- action(action(("tick"))).'
    ]
    assert numeric_facts(planner.compiled_task, "numAssign(") == []

    plan = planner.plan()
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert [ai.action.name for ai in plan.actions] == ["tick", "finish"]
    assert_plan_is_over_original_problem(problem, plan)


def test_a_constant_assignment_still_uses_the_constant_fact():
    """tank's `bucket := 0` and `bucket := 3` must stay the numAssign integers
    they always were -- the expression path is for expressions only."""
    planner = PLASPPlanner(tank_problem(), "seq")
    assert numeric_facts(planner.compiled_task, "numAssign(") == [
        'numAssign(action(("pour")), variable(("bucket")), 0) '
        ':- action(action(("pour"))).',
        'numAssign(action(("refill")), variable(("bucket")), 3) '
        ':- action(action(("refill"))).',
    ]
    assert numeric_facts(planner.compiled_task, "numAssignExpr(") == []


# ---------------------------------------------------------------------------
# The engine declares the kind
# ---------------------------------------------------------------------------

def test_the_engine_declares_and_solves_the_kind():
    problem = tank_problem()
    assert "FLUENTS_IN_NUMERIC_ASSIGNMENTS" in problem.kind.features
    assert "GENERAL_NUMERIC_PLANNING" in problem.kind.features

    with OneshotPlanner(name="PLASPPlanner") as planner:
        assert planner.supports(problem.kind)
        result = planner.solve(problem)
    assert result.status == Status.SOLVED_SATISFICING, \
        [str(m) for m in result.log_messages or []]
    assert_plan_is_over_original_problem(problem, result.plan)


def test_a_static_fluent_read_is_supported_too():
    """UP counts a *static* fluent's value as good as a constant: the kind
    stays SIMPLE_NUMERIC_PLANNING with no assignment feature at all. So this
    shape passed the kind gate even before the expression facts existed -- and
    then crashed in the fact builder, which demanded a constant. It must plan."""
    def creep(tick, x, y):
        tick.add_increase_effect(x(), y())

    problem = delta_problem(creep)
    assert "FLUENTS_IN_NUMERIC_ASSIGNMENTS" not in problem.kind.features

    with OneshotPlanner(name="PLASPPlanner") as planner:
        assert planner.supports(problem.kind)
        result = planner.solve(problem)
    assert result.status == Status.SOLVED_SATISFICING, \
        [str(m) for m in result.log_messages or []]
    assert_plan_is_over_original_problem(problem, result.plan)


# ---------------------------------------------------------------------------
# Rescaling: constants inside the expression scale, coefficients do not
# ---------------------------------------------------------------------------

def test_a_fractional_constant_beside_a_fluent_term_is_rescaled():
    """`x += y + 1/2` is put on the integer grid by storing *x* doubled: the
    delta becomes `2*y + 1`, and `y` -- which the effect only reads -- keeps its
    own units, so nothing but the fluent that needed the finer grid gets one."""
    def creep(tick, x, y):
        tick.add_increase_effect(x(), Plus(y(), Fraction(1, 2)))

    problem = delta_problem(creep, threshold=3, fluent_type=RealType())
    planner = PLASPPlanner(problem, "seq")
    assert planner.compiled_task.numeric_scale == 2

    assert numeric_facts(planner.compiled_task, "numEffectExpr(") == [
        'numEffectExpr(action(("tick")), variable(("x")), '
        'expr(sum((variable(("y")),2)), 1)) :- action(action(("tick"))).'
    ]
    initial = numeric_facts(planner.compiled_task, "initialState(")
    assert any('value(variable(("y")), 1))' in line for line in initial), initial

    # y = 1, so each tick adds 3/2; the plan is checked in the task's own units.
    plan = planner.plan()
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert [ai.action.name for ai in plan.actions] == ["tick", "tick", "finish"]
    assert_plan_is_over_original_problem(problem, plan)


def test_a_fractional_coefficient_on_a_static_fluent_is_folded_in():
    """`x += y/2` where `y` is static: the coefficient never reaches the
    encoding as a coefficient, because the grounder can read y's value out of
    the initial state. `x`'s own factor of 2 is what makes the arithmetic whole,
    so the folded delta is the looked-up value itself and no division is left --
    which is what keeps the effect a constant `numEffect` rather than an
    expression whose value has to be reified per step."""
    def half(tick, x, y):
        tick.add_increase_effect(x(), Div(y(), 2))

    problem = delta_problem(half, fluent_type=RealType())
    planner = PLASPPlanner(problem, "seq")
    assert planner.compiled_task.numeric_scale == 2

    assert numeric_facts(planner.compiled_task, "numEffect(") == [
        'numEffect(action(("tick")), variable(("x")), RAWSTAT0) :- '
        'action(action(("tick"))), '
        'initialState(variable(("y")), value(variable(("y")), RAWSTAT0)).'
    ]

    # y = 1, so each tick adds 1/2 and the threshold of 3 needs six of them.
    plan = planner.plan()
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert [ai.action.name for ai in plan.actions] == ["tick"] * 6 + ["finish"]
    assert_plan_is_over_original_problem(problem, plan)


def test_a_fractional_coefficient_on_a_written_fluent_is_cleared_by_the_target():
    """The same effect where `y` is written too, so its value cannot be looked
    up. Nothing is folded and nothing needs to be: storing `x` twice as fine
    makes the 1/2 the whole coefficient 1, and only `x` moves."""
    def half(tick, x, y):
        tick.add_increase_effect(x(), Div(y(), 2))
        tick.add_increase_effect(y(), 1)      # y is no longer static

    problem = delta_problem(half, threshold=3, fluent_type=RealType())
    planner = PLASPPlanner(problem, "seq")
    assert planner.compiled_task.numeric_scale == 2
    assert numeric_facts(planner.compiled_task, "numEffectExpr(") == [
        'numEffectExpr(action(("tick")), variable(("x")), '
        'expr(variable(("y")), 0)) :- action(action(("tick"))).'
    ]

    # y starts at 1 and grows by 1 a tick, so x gains 1/2, 1, 3/2, ... and
    # passes 3 on the third tick.
    plan = planner.plan()
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert [ai.action.name for ai in plan.actions] == ["tick"] * 3 + ["finish"]
    assert_plan_is_over_original_problem(problem, plan)


def test_a_self_referential_fractional_coefficient_is_still_rejected():
    """`x += x/2` is the one shape no scaling reaches: whatever `x` is stored
    multiplied by, the coefficient it needs is halved by the same factor. This
    is sailing-wind's `v := 0.9 v + ...`, and it is a real boundary of an
    integer encoding rather than a gap in the scaling."""
    def compound(tick, x, y):
        tick.add_increase_effect(x(), Div(x(), 2))

    problem = delta_problem(compound, start=1, fluent_type=RealType())
    with pytest.raises(NotImplementedError, match="fractional coefficient"):
        PLASPPlanner(problem, "seq")


# ---------------------------------------------------------------------------
# The boundary: past linear is still out, now at encoding time
# ---------------------------------------------------------------------------

def test_a_product_of_two_written_fluents_is_still_rejected():
    def multiply(tick, x, y):
        tick.add_increase_effect(x(), Times(x(), y()))
        tick.add_increase_effect(y(), 1)      # y is written, so nothing to fold

    problem = delta_problem(multiply)
    with pytest.raises(NotImplementedError, match="not linear"):
        PLASPPlanner(problem, "seq")


def test_a_non_linear_condition_over_two_written_fluents_is_still_rejected():
    def creep(tick, x, y):
        tick.add_increase_effect(x(), y())
        tick.add_increase_effect(y(), 1)

    problem = delta_problem(creep)
    x, y = problem.fluent("x"), problem.fluent("y")
    problem.add_goal(GE(Times(x(), y()), 1))
    with pytest.raises(NotImplementedError, match="not linear"):
        PLASPPlanner(problem, "seq")


def test_a_product_with_a_static_factor_is_linear_after_all():
    """zenotravel's `(* (distance ?c1 ?c2) (slow-burn ?a))`, in miniature: `y` is
    a table rather than state, so the grounder reads its value out of the
    initial state and the product is an ordinary coefficient by the time the
    rule grounds."""
    def scaled(tick, x, y):
        tick.add_increase_effect(x(), Times(x(), y()))

    problem = delta_problem(scaled, start=1, threshold=4, helper_value=1)
    planner = PLASPPlanner(problem, "seq")

    effects = numeric_facts(planner.compiled_task, "numEffectExpr(")
    assert len(effects) == 1, effects
    assert 'initialState(variable(("y")), value(variable(("y")), RAWSTAT0))' in effects[0]
    assert 'sum((variable(("x")),RAWSTAT0))' in effects[0]

    # y is 1, so x doubles every tick: 1, 2, 4 -- two ticks to reach 4.
    plan = planner.plan()
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert [ai.action.name for ai in plan.actions] == ["tick", "tick", "finish"]
    assert_plan_is_over_original_problem(problem, plan)


def test_a_static_product_in_a_condition_is_looked_up_at_grounding_time():
    """The same fold on the condition side, where the comparison is multiplied
    through rather than scaled -- so it works even when the looked-up value
    could not have been divided back out exactly."""
    def creep(tick, x, y):
        tick.add_increase_effect(x(), 1)

    problem = delta_problem(creep, threshold=1, helper_value=2)
    x, y = problem.fluent("x"), problem.fluent("y")
    problem.add_goal(GE(x(), Times(y(), y())))

    planner = PLASPPlanner(problem, "seq")
    goals = numeric_facts(planner.compiled_task, "numGoal(")
    assert len(goals) == 1 and "RAWSTAT0" in goals[0], goals

    # y is 2, so the goal is x >= 4: four ticks, plus the `finish` that sets
    # `done` (which the search is free to place as soon as x >= 1).
    plan = planner.plan()
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    names = [ai.action.name for ai in plan.actions]
    assert names.count("tick") == 4 and names.count("finish") == 1, names
    assert_plan_is_over_original_problem(problem, plan)


# ---------------------------------------------------------------------------
# Conditional numeric effects, stated by the numeric layer itself
# ---------------------------------------------------------------------------

def test_a_conditional_numeric_delta_is_encoded_natively_and_solved():
    """`when (y = 1) then x += 1`: caused/3 carries a variable's *new* value, so
    a conditional increase cannot ride on it -- the numeric layer collects the
    deltas that fire and applies their sum instead. Petrobras' `sail` in
    miniature, and no compiler is asked for."""
    def gated(tick, x, y):
        tick.add_increase_effect(x(), 1, condition=Equals(y(), 1))

    problem = delta_problem(gated)
    planner = PLASPPlanner(problem, "seq")
    assert planner.compilationlist == []
    assert numeric_facts(planner.compiled_task, "numCondEffect(") == [
        'numCondEffect(action(("tick")), effect((cond,"tick",0)), variable(("x")), 1) '
        ':- action(action(("tick"))).'
    ]

    # y is 1, so the guard always holds and each tick adds 1.
    plan = planner.plan()
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert [ai.action.name for ai in plan.actions] == ["tick"] * 3 + ["finish"]
    assert_plan_is_over_original_problem(problem, plan)


def test_two_conditional_deltas_on_one_fluent_add_up():
    """PDDL's own reading, and the reason a delta cannot ride on caused/3: two
    effects that both fire contribute both deltas. Stated as two `caused` atoms
    they would be two different new values -- not a state at all."""
    def gated(tick, x, y):
        tick.add_increase_effect(x(), 1, condition=GE(y(), 1))
        tick.add_increase_effect(x(), 2, condition=GE(y(), 1))

    problem = delta_problem(gated, threshold=6)
    planner = PLASPPlanner(problem, "seq")

    # y is 1, so both fire and each tick adds 3: two ticks reach 6.
    plan = planner.plan()
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert [ai.action.name for ai in plan.actions] == ["tick", "tick", "finish"]
    assert_plan_is_over_original_problem(problem, plan)


def test_a_conditional_delta_adds_to_an_unconditional_one():
    """The same sum, with one of the two effects unguarded. The unconditional
    delta joins the sum rather than staying on `numEffect`, or the two would
    again be two `caused` atoms."""
    def gated(tick, x, y):
        tick.add_increase_effect(x(), 1)
        tick.add_increase_effect(x(), 2, condition=GE(y(), 1))

    problem = delta_problem(gated, threshold=6)
    planner = PLASPPlanner(problem, "seq")
    assert numeric_facts(planner.compiled_task, "numEffect(") == []

    plan = planner.plan()
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert [ai.action.name for ai in plan.actions] == ["tick", "tick", "finish"]
    assert_plan_is_over_original_problem(problem, plan)


def test_a_conditional_delta_whose_guard_fails_does_not_fire():
    """The half a wrongly-unconditional encoding would get away with above."""
    def gated(tick, x, y):
        tick.add_increase_effect(x(), 1)
        tick.add_increase_effect(x(), 2, condition=GE(y(), 5))

    problem = delta_problem(gated, threshold=3, helper_value=1)
    planner = PLASPPlanner(problem, "seq")

    # y is 1, so only the unconditional +1 fires: three ticks, not one.
    plan = planner.plan()
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert [ai.action.name for ai in plan.actions] == ["tick"] * 3 + ["finish"]
    assert_plan_is_over_original_problem(problem, plan)


def test_a_conditional_assignment_reading_the_state_is_encoded_natively():
    """`when (y >= 1) then x := y + 5`: the assigned value is evaluated at the
    step before, like any expression effect, and gated like any conditional
    one."""
    def gated(tick, x, y):
        tick.add_effect(x(), Plus(y(), 5), condition=GE(y(), 1))

    problem = delta_problem(gated, threshold=6)
    planner = PLASPPlanner(problem, "seq")
    assert planner.compilationlist == []
    assert numeric_facts(planner.compiled_task, "numCondAssignExpr(")

    # y is 1, so one tick sets x to 6.
    plan = planner.plan()
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert [ai.action.name for ai in plan.actions] == ["tick", "finish"]
    assert_plan_is_over_original_problem(problem, plan)


def test_a_numeric_condition_on_a_conditional_effect_is_encoded_natively():
    """The other half of the same refusal: caused/3 reads an effect's condition
    off precondition/3, which is boolean. A numeric one becomes a one-disjunct
    group of the effect term instead, judged by the disjunctive layer's
    difference #sum -- which is why that layer is loaded for it."""
    def gated(tick, x, y):
        tick.add_effect(x(), 9, condition=GE(y(), 1))

    problem = delta_problem(gated)
    planner = PLASPPlanner(problem, "seq")
    assert planner.compilationlist == []
    assert "disjunctive" in planner.layers
    assert any(line.startswith("orDisjunctNum(effect((cond,")
               for line in planner.compiled_task.fact_lines)

    plan = planner.plan()
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert [ai.action.name for ai in plan.actions] == ["tick", "finish"]
    assert_plan_is_over_original_problem(problem, plan)


def test_a_numeric_condition_that_fails_keeps_the_effect_from_firing():
    """The same shape with a guard the state does not meet: the effect must not
    fire, which a group that is merely *declared* rather than *checked* would
    let it do."""
    def gated(tick, x, y):
        tick.add_effect(x(), 9, condition=GE(y(), 5))

    problem = delta_problem(gated, helper_value=1)
    planner = PLASPPlanner(problem, "seq")
    planner.plan(max_horizon=6)
    assert planner.status != Status.SOLVED_SATISFICING, (
        "the effect fired although its numeric condition is false")


def test_a_conditional_constant_assignment_still_takes_the_native_path():
    """The shape the postcondition path can state on its own is left alone --
    the compiler is asked for only what would otherwise be refused, so the ADL
    families that encode natively today keep doing so."""
    problem = delta_problem(lambda tick, x, y: None)
    x, done = problem.fluent("x"), problem.fluent("done")
    problem.action("tick").add_effect(x(), 9, condition=done())

    planner = PLASPPlanner(problem, "seq")
    assert planner.compilationlist == []
    assert 'postcondition(action(("tick")), effect((cond,"tick",0)), variable(("x")), ' \
           'value(variable(("x")), 9)) :- action(action(("tick"))).' \
           in numeric_facts(planner.compiled_task, "postcondition(")


def test_a_conditional_delta_over_a_static_fluent_combines_both_paths():
    """Petrobras' `sail`, in miniature: the condition is numeric *and* the delta
    reads a static fluent with a fractional coefficient. The condition becomes a
    group of the effect term and the fold turns the coefficient into
    grounding-time arithmetic; neither alone is enough, and neither needs a
    compiler."""
    def gated(tick, x, y):
        tick.add_increase_effect(x(), Div(y(), 2), condition=Equals(y(), 1))

    problem = delta_problem(gated, fluent_type=RealType())
    planner = PLASPPlanner(problem, "seq")
    assert planner.compilationlist == []
    assert planner.compiled_task.numeric_scale == 2

    plan = planner.plan()
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert [ai.action.name for ai in plan.actions] == ["tick"] * 6 + ["finish"]
    assert_plan_is_over_original_problem(problem, plan)
