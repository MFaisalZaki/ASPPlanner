"""End-to-end tests for the PLASPPlanner UP engine and its direct API.

Every solved plan must reference the *original* problem's actions and
objects (not the internal grounded/renamed vocabulary) and validate with
UP's sequential plan validator against the original problem.
"""

import pytest

import aspplanners  # noqa: F401 -- registers the engine
from unified_planning.engines import PlanGenerationResultStatus as Status
from unified_planning.shortcuts import (
    BoolType,
    InstantaneousAction,
    IntType,
    Not,
    Object,
    OneshotPlanner,
    PlanValidator,
    Problem,
    UserType,
)

from aspplanners.plasp.planner import PLASPPlanner


# ---------------------------------------------------------------------------
# Problem builders
# ---------------------------------------------------------------------------

def robot_line_problem(n_locations=4, connected_line=True, sep="_"):
    """Robot on a chain of locations; goal is reaching the last one.

    ``sep='-'`` puts hyphens in object names to exercise the ASP name
    sanitization and its map-back.
    Needs n_locations - 1 moves when ``connected_line``; unsolvable otherwise.
    """
    from unified_planning.shortcuts import Fluent

    Location = UserType("Location")
    robot_at = Fluent("robot_at", BoolType(), l=Location)
    connected = Fluent("connected", BoolType(), a=Location, b=Location)

    move = InstantaneousAction("move", a=Location, b=Location)
    a, b = move.parameter("a"), move.parameter("b")
    move.add_precondition(connected(a, b))
    move.add_precondition(robot_at(a))
    move.add_effect(robot_at(a), False)
    move.add_effect(robot_at(b), True)

    problem = Problem("robot_line")
    problem.add_fluent(robot_at, default_initial_value=False)
    problem.add_fluent(connected, default_initial_value=False)
    problem.add_action(move)
    locs = [Object(f"loc{sep}{i}", Location) for i in range(n_locations)]
    problem.add_objects(locs)
    if connected_line:
        for i in range(n_locations - 1):
            problem.set_initial_value(connected(locs[i], locs[i + 1]), True)
    problem.set_initial_value(robot_at(locs[0]), True)
    problem.add_goal(robot_at(locs[-1]))
    return problem


def numeric_counter_problem(threshold=3):
    """`tick` increments a counter; `finish` needs counter >= threshold.

    The shortest plan is `tick` x threshold followed by `finish`.
    """
    from unified_planning.shortcuts import Fluent, GE

    counter = Fluent("counter", IntType())
    done = Fluent("done", BoolType())

    tick = InstantaneousAction("tick")
    tick.add_precondition(Not(done()))
    tick.add_increase_effect(counter(), 1)

    finish = InstantaneousAction("finish")
    finish.add_precondition(GE(counter(), threshold))
    finish.add_effect(done(), True)

    problem = Problem("counter")
    problem.add_fluent(counter, default_initial_value=0)
    problem.add_fluent(done, default_initial_value=False)
    problem.add_action(tick)
    problem.add_action(finish)
    problem.add_goal(done())
    return problem


def rover_recharge_problem(target=2, with_bool_goal=False):
    """A rover whose `battery` (int fluent) starts at 0; `recharge` adds 1.

    The goal is a NUMERIC formula ``battery(rover1) >= target`` rather than a
    boolean fluent -- the case the seq encoding checks with numGoal against
    numval at the goal step. Shortest plan is ``recharge`` x target.
    When ``with_bool_goal``, also require a boolean ``deployed`` goal so the
    conjunctive numeric+boolean goal path is exercised together.
    """
    from unified_planning.shortcuts import Fluent, GE

    Rover = UserType("rover")
    battery = Fluent("battery", IntType(), r=Rover)
    recharge = InstantaneousAction("recharge", r=Rover)
    r = recharge.parameter("r")
    recharge.add_increase_effect(battery(r), 1)

    problem = Problem("rover_recharge")
    problem.add_fluent(battery, default_initial_value=0)
    problem.add_action(recharge)
    rover = Object("rover1", Rover)
    problem.add_object(rover)
    problem.add_goal(GE(battery(rover), target))

    if with_bool_goal:
        deployed = Fluent("deployed", BoolType(), r=Rover)
        deploy = InstantaneousAction("deploy", r=Rover)
        deploy.add_effect(deployed(deploy.parameter("r")), True)
        problem.add_fluent(deployed, default_initial_value=False)
        problem.add_action(deploy)
        problem.add_goal(deployed(rover))
    return problem


def assert_plan_is_over_original_problem(problem, plan):
    """The lifted plan must be stated in terms of the task passed to the
    planner: every step references one of ITS action objects (identity, not
    just an equal name), with matching arity, and every actual parameter is
    an object of that task. The plan must also validate against it."""
    original_actions = {id(a) for a in problem.actions}
    original_objects = {o.name: o for o in problem.all_objects}
    for ai in plan.actions:
        assert id(ai.action) in original_actions, (
            f"plan action {ai.action.name!r} is not (identically) an action "
            f"of the original problem -- map-back is broken"
        )
        assert len(ai.actual_parameters) == len(ai.action.parameters), (
            f"{ai} has {len(ai.actual_parameters)} arguments but "
            f"{ai.action.name} takes {len(ai.action.parameters)}"
        )
        for actual in ai.actual_parameters:
            obj = actual.object()
            assert original_objects.get(obj.name) == obj, (
                f"plan argument {obj.name!r} of {ai} is not an object of the "
                f"original problem"
            )
    with PlanValidator(name="sequential_plan_validator") as validator:
        result = validator.validate(problem, plan)
    assert str(result.status) == "ValidationResultStatus.VALID", (
        f"plan does not validate against the original problem: {result}"
    )


# ---------------------------------------------------------------------------
# UP engine interface
# ---------------------------------------------------------------------------

def test_solves_and_maps_back_to_original_problem():
    problem = robot_line_problem()
    with OneshotPlanner(name="PLASPPlanner") as planner:
        result = planner.solve(problem)
    assert result.status == Status.SOLVED_SATISFICING
    assert len(result.plan.actions) == 3
    assert_plan_is_over_original_problem(problem, result.plan)


def test_hyphenated_names_map_back():
    problem = robot_line_problem(sep="-")
    with OneshotPlanner(name="PLASPPlanner") as planner:
        result = planner.solve(problem)
    assert result.status == Status.SOLVED_SATISFICING
    assert_plan_is_over_original_problem(problem, result.plan)
    # the original vocabulary is hyphenated; the plan must be too
    plan_str = str(result.plan)
    assert "loc-0" in plan_str and "loc_0" not in plan_str


def test_goal_already_satisfied_is_solved_with_empty_plan():
    problem = robot_line_problem()
    problem.clear_goals()
    goal_fluent = problem.fluent("robot_at")
    problem.add_goal(goal_fluent(problem.object("loc_0")))
    with OneshotPlanner(name="PLASPPlanner") as planner:
        result = planner.solve(problem)
    assert result.status == Status.SOLVED_SATISFICING
    assert len(result.plan.actions) == 0


def test_unsolvable_reports_unsolvable():
    problem = robot_line_problem(connected_line=False)
    with OneshotPlanner(name="PLASPPlanner", params={"max_horizon": 5}) as planner:
        result = planner.solve(problem)
    assert result.status == Status.UNSOLVABLE_INCOMPLETELY
    assert result.plan is None


def test_multiple_goals():
    problem = robot_line_problem()
    robot_at = problem.fluent("robot_at")
    connected = problem.fluent("connected")
    problem.clear_goals()
    problem.add_goal(robot_at(problem.object("loc_3")))
    problem.add_goal(connected(problem.object("loc_0"), problem.object("loc_1")))
    with OneshotPlanner(name="PLASPPlanner") as planner:
        result = planner.solve(problem)
    assert result.status == Status.SOLVED_SATISFICING
    assert_plan_is_over_original_problem(problem, result.plan)


def test_negative_precondition():
    problem = robot_line_problem()
    blocked = problem.fluent("robot_at")  # reuse: forbid moving into loc_2 if at loc_1? keep simple
    from unified_planning.shortcuts import Fluent

    # a `charge` action possible only while NOT at the last location
    charged = Fluent("charged", BoolType())
    problem.add_fluent(charged, default_initial_value=False)
    charge = InstantaneousAction("charge")
    charge.add_precondition(Not(blocked(problem.object("loc_3"))))
    charge.add_effect(charged(), True)
    problem.add_action(charge)
    problem.add_goal(charged())
    with OneshotPlanner(name="PLASPPlanner") as planner:
        result = planner.solve(problem)
    assert result.status == Status.SOLVED_SATISFICING
    assert_plan_is_over_original_problem(problem, result.plan)
    # charge must come before the robot reaches loc_3
    names = [ai.action.name for ai in result.plan.actions]
    assert "charge" in names


def test_numeric_counter():
    problem = numeric_counter_problem(threshold=3)
    with OneshotPlanner(name="PLASPPlanner") as planner:
        result = planner.solve(problem)
    assert result.status == Status.SOLVED_SATISFICING
    names = [ai.action.name for ai in result.plan.actions]
    assert names.count("tick") == 3 and names[-1] == "finish"
    assert_plan_is_over_original_problem(problem, result.plan)


def test_numeric_goal_formula():
    """A goal stated as `battery(rover1) >= 2` (numeric comparison, not a
    boolean fluent) must plan and validate against the original problem."""
    problem = rover_recharge_problem(target=2)
    with OneshotPlanner(name="PLASPPlanner") as planner:
        result = planner.solve(problem)
    assert result.status == Status.SOLVED_SATISFICING
    names = [ai.action.name for ai in result.plan.actions]
    assert names == ["recharge", "recharge"]
    assert_plan_is_over_original_problem(problem, result.plan)


def test_numeric_goal_greater_than_zero():
    """The user's `recharge > 0` case: a single strict-inequality numeric goal
    from a zero initial value needs exactly one increment."""
    from unified_planning.shortcuts import Fluent, GT

    battery = Fluent("battery", IntType())
    recharge = InstantaneousAction("recharge")
    recharge.add_increase_effect(battery(), 1)
    problem = Problem("recharge_gt0")
    problem.add_fluent(battery, default_initial_value=0)
    problem.add_action(recharge)
    problem.add_goal(GT(battery(), 0))

    with OneshotPlanner(name="PLASPPlanner") as planner:
        result = planner.solve(problem)
    assert result.status == Status.SOLVED_SATISFICING
    assert [ai.action.name for ai in result.plan.actions] == ["recharge"]
    assert_plan_is_over_original_problem(problem, result.plan)


def test_numeric_goal_unreachable_is_unsolvable():
    """A numeric goal that no reachable state satisfies stays unsolved within
    the horizon -- the constraint must actually gate the goal, not be vacuous.
    `battery` only ever increases, so `battery <= -1` is never satisfiable."""
    from unified_planning.shortcuts import Fluent, LE

    battery = Fluent("battery", IntType())
    recharge = InstantaneousAction("recharge")
    recharge.add_increase_effect(battery(), 1)
    problem = Problem("recharge_unsat")
    problem.add_fluent(battery, default_initial_value=0)
    problem.add_action(recharge)
    problem.add_goal(LE(battery(), -1))

    with OneshotPlanner(name="PLASPPlanner", params={"max_horizon": 4}) as planner:
        result = planner.solve(problem)
    assert result.status == Status.UNSOLVABLE_INCOMPLETELY
    assert result.plan is None


def test_conjunctive_numeric_and_boolean_goal():
    """A goal mixing a numeric comparison and a boolean fluent must satisfy
    both -- the AND split routes each conjunct to its own goal path."""
    problem = rover_recharge_problem(target=2, with_bool_goal=True)
    with OneshotPlanner(name="PLASPPlanner") as planner:
        result = planner.solve(problem)
    assert result.status == Status.SOLVED_SATISFICING
    names = [ai.action.name for ai in result.plan.actions]
    assert names.count("recharge") == 2 and names.count("deploy") == 1
    assert_plan_is_over_original_problem(problem, result.plan)


def test_numeric_with_hyphenated_names():
    """Numeric tasks skip grounding, so hyphenated action/fluent/object and
    PARAMETER names all flow into the lifted encoding (parameters become ASP
    variables, where hyphens are illegal without sanitization)."""
    from collections import OrderedDict
    from unified_planning.shortcuts import Fluent, GE

    Agent = UserType("agent-kind")
    power = Fluent("power-level", IntType(), a=Agent)
    ready = Fluent("is-ready", BoolType(), a=Agent)

    boost = InstantaneousAction("boost-up", OrderedDict([("the-agent", Agent)]))
    a = boost.parameter("the-agent")
    boost.add_precondition(Not(ready(a)))
    boost.add_increase_effect(power(a), 1)

    arm = InstantaneousAction("arm-agent", OrderedDict([("the-agent", Agent)]))
    a = arm.parameter("the-agent")
    arm.add_precondition(GE(power(a), 2))
    arm.add_effect(ready(a), True)

    problem = Problem("numeric-hyphen")
    problem.add_fluent(power, default_initial_value=0)
    problem.add_fluent(ready, default_initial_value=False)
    problem.add_action(boost)
    problem.add_action(arm)
    unit = Object("unit-1", Agent)
    problem.add_object(unit)
    problem.add_goal(ready(unit))

    with OneshotPlanner(name="PLASPPlanner") as planner:
        result = planner.solve(problem)
    assert result.status == Status.SOLVED_SATISFICING
    names = [ai.action.name for ai in result.plan.actions]
    assert names == ["boost-up", "boost-up", "arm-agent"]
    assert_plan_is_over_original_problem(problem, result.plan)


def test_colliding_names_are_rejected():
    problem = robot_line_problem(sep="-")
    problem.add_object(Object("loc_0", problem.user_types[0]))  # collides with loc-0
    import pytest
    with pytest.raises(ValueError, match="collide"):
        PLASPPlanner(problem, "seq")


def test_timeout_reports_timeout():
    problem = robot_line_problem(n_locations=8)
    with OneshotPlanner(name="PLASPPlanner") as planner:
        result = planner.solve(problem, timeout=1e-9)
    assert result.status == Status.TIMEOUT


# ---------------------------------------------------------------------------
# Direct API compatibility (used by external projects)
# ---------------------------------------------------------------------------

def test_fixed_horizon_returns_empty_plan_when_unsat():
    problem = robot_line_problem()  # needs 3 steps
    blocked = PLASPPlanner(problem, "seq").plan(horizon=1)
    assert not blocked.actions

    solved = PLASPPlanner(problem, "seq").plan(horizon=3)
    assert len(solved.actions) == 3


def test_max_horizon_search_finds_shortest_plan():
    problem = robot_line_problem()
    planner = PLASPPlanner(problem, "seq")
    plan = planner.plan(max_horizon=6)
    assert len(plan.actions) == 3
    assert isinstance(planner.logs, list)


def test_lp_program_is_dumpable():
    planner = PLASPPlanner(robot_line_problem(), "seq")
    program = planner.lp_program()
    assert "occurs" in program        # encoding rules present
    assert "initialState(" in program  # task facts present
    assert all(line.endswith(".") or line.startswith("%") or not line.strip()
               or line.startswith("#") for line in planner.task_facts.splitlines())


# ---------------------------------------------------------------------------
# Grounder selection
# ---------------------------------------------------------------------------

def test_the_grounder_is_the_one_that_supports_the_task():
    """Which grounder runs is asked of the installed engines rather than
    hard-coded, so a task moves to the right one as its kind changes."""
    from unified_planning.shortcuts import Compiler

    from aspplanners.common.compilation import GROUNDERS, select_grounder

    classical = robot_line_problem()
    numeric = numeric_counter_problem()
    for problem in (classical, numeric):
        chosen = select_grounder(problem.kind)
        assert chosen in GROUNDERS
        with Compiler(name=chosen) as compiler:
            assert compiler.supports(problem.kind)
        # And it is the most preferred one that does.
        for rejected in GROUNDERS[:GROUNDERS.index(chosen)]:
            with Compiler(name=rejected) as compiler:
                assert not compiler.supports(problem.kind)

    assert select_grounder(classical.kind, candidates=()) is None


def test_the_task_reaches_clingo_uncompiled():
    """The default pipeline is empty: nothing rewrites the task, and it is not
    pre-ground either -- gringo grounds it anyway, and the lifted encoding gives
    it the same static relations to prune with."""
    for problem in (robot_line_problem(), robot_line_problem(connected_line=False),
                    numeric_counter_problem()):
        assert PLASPPlanner.__new__(PLASPPlanner)._check_compilationlist(problem, None) == []


def test_a_reachability_grounder_can_still_be_asked_for():
    """Reachability analysis is the one thing the lifted encoding cannot do --
    it prunes on *dynamic* preconditions, which static folding cannot see. That
    is what select_grounder stays for; passing it back has to keep working."""
    from unified_planning.shortcuts import CompilationKind

    from aspplanners.common.compilation import REACHABILITY_GROUNDERS, select_grounder

    problem = robot_line_problem()
    grounder = select_grounder(problem.kind, REACHABILITY_GROUNDERS)
    assert grounder == "fast-downward-reachability-grounder"

    planner = PLASPPlanner(problem, compilationlist=[[grounder, CompilationKind.GROUNDING]])
    plan = planner.plan(max_horizon=6)
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert len(plan.actions) == 3
    assert_plan_is_over_original_problem(problem, plan)

    # A numeric task has no reachability grounder that takes it.
    assert select_grounder(numeric_counter_problem().kind, REACHABILITY_GROUNDERS) is None


def negative_precondition_problem():
    """`advance` needs NOT blocked, which already holds. `ready` is there so the
    initial state has a true fluent -- without one the encoder's degenerate
    "no true fluents" fallback emits everything and hides the question."""
    from unified_planning.shortcuts import Fluent

    blocked, ready, go = Fluent("blocked"), Fluent("ready"), Fluent("go")
    advance = InstantaneousAction("advance")
    advance.add_precondition(Not(blocked))
    advance.add_precondition(ready)
    advance.add_effect(go, True)

    problem = Problem("negative_precondition")
    problem.add_fluent(blocked, default_initial_value=False)
    problem.add_fluent(ready, default_initial_value=True)
    problem.add_fluent(go, default_initial_value=False)
    problem.add_action(advance)
    problem.add_goal(go)
    return problem


def test_negative_conditions_are_encoded_not_compiled_away():
    """The encoding is multi-valued, so `value(V, false)` is a value like any
    other and a mirror fluent per negatively-read one would be pure overhead."""
    problem = negative_precondition_problem()
    pipeline = PLASPPlanner.__new__(PLASPPlanner)._check_compilationlist(problem, None)
    assert not any("negative" in name for name, _kind in pipeline)

    planner = PLASPPlanner(problem)
    plan = planner.plan(max_horizon=4)
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert [ai.action.name for ai in plan.actions] == ["advance"]
    assert_plan_is_over_original_problem(problem, plan)


def test_only_negatively_read_fluents_get_a_false_initial_value():
    """A false initial value is what makes `not holds(V, value(V,false), t-1)`
    satisfiable at step 0 -- but every such chain then propagates through every
    step, so only the fluents that need one get one."""
    planner = PLASPPlanner(negative_precondition_problem())
    initial = planner.compiled_task.facts["_initial_state"]
    assert any("blocked" in line and "false" in line for line in initial), initial
    assert any("ready" in line and "true" in line for line in initial), initial
    # `go` is false too, but is only ever read positively (in the goal).
    assert not any("go" in line for line in initial), initial


def disjunction_problem(disjuncts=None):
    """`finish` needs `Or(*disjuncts)`; `set_b` is the only thing that makes a
    literal true. Defaults to `a or b`, solvable in two steps."""
    from unified_planning.shortcuts import Fluent, Or

    a, b, done = Fluent("a"), Fluent("b"), Fluent("done")
    set_b = InstantaneousAction("set_b")
    set_b.add_effect(b, True)
    finish = InstantaneousAction("finish")
    finish.add_precondition(Or(a, b) if disjuncts is None else Or(*disjuncts(a, b)))
    finish.add_effect(done, True)

    problem = Problem("disjunction")
    for fluent in (a, b, done):
        problem.add_fluent(fluent, default_initial_value=False)
    problem.add_action(set_b)
    problem.add_action(finish)
    problem.add_goal(done)
    return problem


def test_a_disjunction_is_encoded_not_compiled_away():
    """precondition facts are conjunctive, so a disjunction gets its own
    orGroup/orDisjunct vocabulary instead of splitting the action in two."""
    problem = disjunction_problem()
    pipeline = PLASPPlanner.__new__(PLASPPlanner)._check_compilationlist(problem, None)
    assert not any("disjunctive" in name for name, _kind in pipeline)

    planner = PLASPPlanner(problem)
    plan = planner.plan(max_horizon=4)
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert [ai.action.name for ai in plan.actions] == ["set_b", "finish"]
    assert_plan_is_over_original_problem(problem, plan)

    facts = planner.compiled_task.fact_lines
    assert any(line.startswith("orGroup(") for line in facts), sorted(facts)
    # One group, two disjuncts -- not two copies of `finish`.
    assert len({line for line in facts if line.startswith("orGroup(")}) == 1
    assert len({line for line in facts if line.startswith("action(action(")}) == 2


def test_a_disjunct_is_still_read_conjunctively():
    """`or(and(a, b), c)`: the first disjunct needs BOTH a and b. Getting only
    `a` must not satisfy it -- that is the failure mode a flat list of literals,
    with the conjunction lost, would have."""
    from unified_planning.shortcuts import And, Fluent, Or

    def build(with_set_b):
        a, b, c, done = Fluent("a"), Fluent("b"), Fluent("c"), Fluent("done")
        set_a = InstantaneousAction("set_a")
        set_a.add_effect(a, True)
        finish = InstantaneousAction("finish")
        finish.add_precondition(Or(And(a, b), c))
        finish.add_effect(done, True)

        problem = Problem("conjunctive_disjunct")
        for fluent in (a, b, c, done):
            problem.add_fluent(fluent, default_initial_value=False)
        problem.add_action(set_a)
        if with_set_b:
            set_b = InstantaneousAction("set_b")
            set_b.add_effect(b, True)
            problem.add_action(set_b)
        problem.add_action(finish)
        problem.add_goal(done)
        return problem

    # `c` is unreachable and only `a` can be established, so the and-disjunct
    # stays half-satisfied and there is no plan.
    planner = PLASPPlanner(build(with_set_b=False))
    planner.plan(max_horizon=5)
    assert planner.status == Status.UNSOLVABLE_INCOMPLETELY, planner.logs

    # Add the missing half and the same disjunct becomes satisfiable.
    problem = build(with_set_b=True)
    planner = PLASPPlanner(problem)
    plan = planner.plan(max_horizon=5)
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert len(plan.actions) == 3
    assert_plan_is_over_original_problem(problem, plan)


def test_a_disjunctive_goal_is_encoded_too():
    from unified_planning.shortcuts import Fluent, Or

    a, b = Fluent("a"), Fluent("b")
    set_b = InstantaneousAction("set_b")
    set_b.add_effect(b, True)
    problem = Problem("disjunctive_goal")
    for fluent in (a, b):
        problem.add_fluent(fluent, default_initial_value=False)
    problem.add_action(set_b)
    problem.add_goal(Or(a, b))

    planner = PLASPPlanner(problem)
    plan = planner.plan(max_horizon=4)
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert [ai.action.name for ai in plan.actions] == ["set_b"]
    assert any(line.startswith("goalOrGroup(")
               for line in planner.compiled_task.fact_lines)


def counter_problem(precondition=None, goal=None, start=0):
    """`tick` increments `counter`; `finish` sets `done` under `precondition`.

    A kit for the numeric-negation cases: the caller states the condition as a
    function of the `counter` fluent expression.
    """
    from unified_planning.shortcuts import Fluent

    counter, done = Fluent("counter", IntType()), Fluent("done", BoolType())
    tick = InstantaneousAction("tick")
    tick.add_increase_effect(counter(), 1)
    finish = InstantaneousAction("finish")
    if precondition is not None:
        finish.add_precondition(precondition(counter()))
    finish.add_effect(done(), True)

    problem = Problem("counter_condition")
    problem.add_fluent(counter, default_initial_value=start)
    problem.add_fluent(done, default_initial_value=False)
    problem.add_action(tick)
    problem.add_action(finish)
    problem.add_goal(goal(counter(), done()) if goal is not None else done())
    return problem


def test_negated_numeric_equality_is_encoded_as_a_neq_comparison():
    """`not (counter = 0)` has no `holds` chain to negate -- it is the numeric
    comparison's complement, checked against numval like any other one."""
    from unified_planning.shortcuts import Equals

    problem = counter_problem(precondition=lambda c: Not(Equals(c, 0)))
    planner = PLASPPlanner(problem)
    plan = planner.plan(max_horizon=4)
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    # `finish` is blocked at counter == 0, so one tick has to come first.
    assert [ai.action.name for ai in plan.actions] == ["tick", "finish"]
    assert_plan_is_over_original_problem(problem, plan)

    assert any("numPrecondition(" in line and ", neq," in line
               for line in planner.compiled_task.fact_lines), \
        sorted(planner.compiled_task.fact_lines)

    # And the complement really is enforced: `not (counter != 0)` pins the
    # counter to 0, which `tick` only ever moves away from.
    planner = PLASPPlanner(counter_problem(
        precondition=lambda c: Not(Not(Equals(c, 0))), start=1))
    planner.plan(max_horizon=4)
    assert planner.status == Status.UNSOLVABLE_INCOMPLETELY, planner.logs


def test_a_numeric_literal_in_a_disjunct_is_checked_against_numval():
    """A disjunct's numeric literal cannot be an orDisjunct atom (it is read off
    numval, not holds), so it gets its own orDisjunctNum vocabulary -- and the
    disjunct is still enumerated when that is all it contains."""
    from unified_planning.shortcuts import Equals, Or

    # `done or counter != 2`, starting at counter == 2: neither disjunct holds
    # initially, and only `tick` can make the numeric one true.
    problem = counter_problem(goal=lambda c, done: Or(done, Not(Equals(c, 2))),
                              start=2)
    planner = PLASPPlanner(problem)
    plan = planner.plan(max_horizon=3)
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert len(plan.actions) == 1, [ai.action.name for ai in plan.actions]
    assert_plan_is_over_original_problem(problem, plan)

    facts = planner.compiled_task.fact_lines
    assert any(line.startswith("goalOrDisjunctNum(") and ", neq," in line
               for line in facts), sorted(facts)
    # The numeric-only disjunct is declared, or the group could never hold
    # through it (a disjunct with no orDisjunct atom of its own).
    assert len({line for line in facts
                if line.startswith("goalOrDisjunctId(")}) == 2, sorted(facts)


def test_a_numeric_disjunct_that_can_never_hold_is_not_vacuously_true():
    """The group holds through a disjunct only when the disjunct does; a
    numeric literal that no state satisfies must not let it through."""
    from unified_planning.shortcuts import Equals, LT, Or

    # `counter < 0 or counter = 100` -- the counter starts at 0 and ticks by
    # one, so within the horizon neither disjunct is ever satisfied.
    problem = counter_problem(
        precondition=lambda c: Or(LT(c, 0), Equals(c, 100)))
    planner = PLASPPlanner(problem)
    planner.plan(max_horizon=4)
    assert planner.status == Status.UNSOLVABLE_INCOMPLETELY, planner.logs


def test_a_double_negation_cancels_instead_of_staying_negative():
    """De Morgan on `not (a and not X)` hands the inner `not X` a negative
    polarity; the second `not` has to flip it back rather than re-assert it."""
    from unified_planning.shortcuts import And, Equals, Fluent

    counter, a = Fluent("counter", IntType()), Fluent("a", BoolType())
    tick = InstantaneousAction("tick")
    tick.add_increase_effect(counter(), 1)

    problem = Problem("double_negation")
    problem.add_fluent(counter, default_initial_value=0)
    problem.add_fluent(a, default_initial_value=True)
    problem.add_action(tick)
    # `not (a and counter != 1)` == `not a or counter = 1`. `a` is true and
    # nothing falsifies it, so the plan has to establish `counter = 1`.
    problem.add_goal(Not(And(a(), Not(Equals(counter(), 1)))))

    planner = PLASPPlanner(problem)
    plan = planner.plan(max_horizon=4)
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert [ai.action.name for ai in plan.actions] == ["tick"]
    assert_plan_is_over_original_problem(problem, plan)
    # The inner negation cancelled: the disjunct is an equality, not its
    # complement.
    assert any(line.startswith("goalOrDisjunctNum(") and ", eq," in line
               for line in planner.compiled_task.fact_lines)


# ---------------------------------------------------------------------------
# Quantifiers
# ---------------------------------------------------------------------------

def forall_problem(n_objects=3, universal_effect=False):
    """`finish` needs every thing marked. With `universal_effect`, one `sweep`
    marks them all at once and the goal is the forall instead."""
    from unified_planning.shortcuts import BoolType, Fluent, Forall, Object, UserType, Variable

    Thing = UserType("Thing")
    marked = Fluent("marked", BoolType(), t=Thing)
    done = Fluent("done")
    x = Variable("x", Thing)

    problem = Problem("forall")
    problem.add_fluent(marked, default_initial_value=False)
    problem.add_fluent(done, default_initial_value=False)
    problem.add_objects([Object(f"o{i}", Thing) for i in range(1, n_objects + 1)])

    if universal_effect:
        sweep = InstantaneousAction("sweep")
        sweep.add_effect(marked(x), True, forall=[x])
        problem.add_action(sweep)
        problem.add_goal(Forall(marked(x), x))
        return problem, 1

    mark = InstantaneousAction("mark", t=Thing)
    mark.add_effect(marked(mark.parameter("t")), True)
    finish = InstantaneousAction("finish")
    finish.add_precondition(Forall(marked(x), x))
    finish.add_effect(done, True)
    problem.add_action(mark)
    problem.add_action(finish)
    problem.add_goal(done)
    return problem, n_objects + 1


@pytest.mark.parametrize("universal_effect", [False, True], ids=["condition", "effect"])
def test_forall_is_expanded_by_the_grounder_not_by_a_compiler(universal_effect):
    """A `forall` is a conjunction over the universe, and precondition/goal
    facts are already conjunctive: emit one with its variable free and let
    gringo range it. So the quantifier remover does not have to run."""
    problem, expected_length = forall_problem(universal_effect=universal_effect)
    pipeline = PLASPPlanner.__new__(PLASPPlanner)._check_compilationlist(problem, None)
    assert not any("quantifiers" in name for name, _kind in pipeline)

    planner = PLASPPlanner(problem)
    plan = planner.plan(max_horizon=8)
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert len(plan.actions) == expected_length
    assert_plan_is_over_original_problem(problem, plan)

    # One rule, ranged by has/2 -- not one fact per object.
    quantified = [line for line in planner.compiled_task.fact_lines if "Q_X" in line]
    assert quantified, sorted(planner.compiled_task.fact_lines)
    for line in quantified:
        assert 'has(Q_X, type("Thing"))' in line, line


def test_forall_stays_one_rule_however_big_the_universe_is():
    sizes = {}
    for n in (3, 12):
        planner = PLASPPlanner(forall_problem(n_objects=n)[0])
        sizes[n] = sum(1 for line in planner.compiled_task.facts["_actions"]
                       if line.startswith("precondition("))
    assert sizes[3] == sizes[12] == 1, sizes


def test_exists_is_encoded_as_disjuncts_indexed_by_binding():
    """An `exists` is a disjunction over the universe, so it reuses the same
    orGroup/orDisjunct vocabulary -- one rule whose disjunct id is the quantified
    variable, enumerated by the grounder instead of copied per object by a
    compiler."""
    from unified_planning.shortcuts import BoolType, Exists, Fluent, Object, UserType, Variable

    def build(n_objects):
        Thing = UserType("Thing")
        seen, done = Fluent("seen", BoolType(), t=Thing), Fluent("done")
        x = Variable("x", Thing)
        look = InstantaneousAction("look", t=Thing)
        look.add_effect(seen(look.parameter("t")), True)
        finish = InstantaneousAction("finish")
        finish.add_precondition(Exists(seen(x), x))
        finish.add_effect(done, True)

        problem = Problem("exists")
        problem.add_fluent(seen, default_initial_value=False)
        problem.add_fluent(done, default_initial_value=False)
        problem.add_objects([Object(f"o{i}", Thing) for i in range(1, n_objects + 1)])
        problem.add_action(look)
        problem.add_action(finish)
        problem.add_goal(done)
        return problem

    problem = build(3)
    pipeline = PLASPPlanner.__new__(PLASPPlanner)._check_compilationlist(problem, None)
    assert not any("quantifiers" in name for name, _kind in pipeline)

    planner = PLASPPlanner(problem)
    plan = planner.plan(max_horizon=6)
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert [ai.action.name for ai in plan.actions] == ["look", "finish"]
    assert_plan_is_over_original_problem(problem, plan)

    disjuncts = [line for line in planner.compiled_task.fact_lines
                 if line.startswith("orDisjunct(")]
    assert len(disjuncts) == 1, disjuncts
    assert "Q_X" in disjuncts[0] and 'has(Q_X, type("Thing"))' in disjuncts[0]

    # And it stays one rule however many objects there are.
    for n in (3, 15):
        facts = PLASPPlanner(build(n)).compiled_task.fact_lines
        assert len([l for l in facts if l.startswith("orDisjunct(")]) == 1


def test_an_exists_that_can_never_hold_is_unsolvable():
    """The or-group has a disjunct per binding and none of them can hold, so
    nothing satisfies the group -- the encoding must not treat an empty-ish
    disjunction as vacuously true."""
    from unified_planning.shortcuts import BoolType, Exists, Fluent, Object, UserType, Variable

    Thing = UserType("Thing")
    seen, done = Fluent("seen", BoolType(), t=Thing), Fluent("done")
    x = Variable("x", Thing)
    finish = InstantaneousAction("finish")
    finish.add_precondition(Exists(seen(x), x))
    finish.add_effect(done, True)

    problem = Problem("exists_unreachable")
    problem.add_fluent(seen, default_initial_value=False)
    problem.add_fluent(done, default_initial_value=False)
    problem.add_objects([Object(f"o{i}", Thing) for i in range(1, 4)])
    problem.add_action(finish)
    problem.add_goal(done)

    planner = PLASPPlanner(problem)
    planner.plan(max_horizon=4)
    assert planner.status == Status.UNSOLVABLE_INCOMPLETELY, planner.logs


def test_the_default_pipeline_compiles_nothing_away():
    """Everything the encoding needs, it states itself; all that is left of the
    pipeline is the grounding decision."""
    for problem in (robot_line_problem(), numeric_counter_problem(),
                    disjunction_problem(), forall_problem()[0]):
        pipeline = PLASPPlanner.__new__(PLASPPlanner)._check_compilationlist(problem, None)
        assert all("grounder" in name for name, _kind in pipeline), pipeline


def test_a_disjunction_nested_inside_a_disjunct_is_rejected():
    """`or(and(a, or(b, c)), d)` needs distribution into DNF before it is a flat
    set of conjunctive disjuncts, which is the one shape the encoding does not
    take -- so it says so instead of getting it wrong."""
    from unified_planning.shortcuts import And, Fluent, Or

    a, b, c, d, done = (Fluent(name) for name in ("a", "b", "c", "d", "done"))
    finish = InstantaneousAction("finish")
    finish.add_precondition(Or(And(a, Or(b, c)), d))
    finish.add_effect(done, True)

    problem = Problem("nested_disjunction")
    for fluent in (a, b, c, d):
        problem.add_fluent(fluent, default_initial_value=True)
    problem.add_fluent(done, default_initial_value=False)
    problem.add_action(finish)
    problem.add_goal(done)

    with pytest.raises(NotImplementedError, match="itself disjunctive"):
        PLASPPlanner(problem)

    # ...and putting the remover back is exactly what the message asks for.
    from unified_planning.shortcuts import CompilationKind

    planner = PLASPPlanner(problem, compilationlist=[
        ["up_disjunctive_conditions_remover", CompilationKind.DISJUNCTIVE_CONDITIONS_REMOVING],
    ])
    assert len(planner.plan(max_horizon=3).actions) == 1
