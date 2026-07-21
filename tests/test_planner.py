"""End-to-end tests for the ASPPlanner UP engine and its direct API.

Every solved plan must reference the *original* problem's actions and
objects (not the internal grounded/renamed vocabulary) and validate with
UP's sequential plan validator against the original problem.
"""

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
    with OneshotPlanner(name="ASPPlanner") as planner:
        result = planner.solve(problem)
    assert result.status == Status.SOLVED_SATISFICING
    assert len(result.plan.actions) == 3
    assert_plan_is_over_original_problem(problem, result.plan)


def test_hyphenated_names_map_back():
    problem = robot_line_problem(sep="-")
    with OneshotPlanner(name="ASPPlanner") as planner:
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
    with OneshotPlanner(name="ASPPlanner") as planner:
        result = planner.solve(problem)
    assert result.status == Status.SOLVED_SATISFICING
    assert len(result.plan.actions) == 0


def test_unsolvable_reports_unsolvable():
    problem = robot_line_problem(connected_line=False)
    with OneshotPlanner(name="ASPPlanner", params={"max_horizon": 5}) as planner:
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
    with OneshotPlanner(name="ASPPlanner") as planner:
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
    with OneshotPlanner(name="ASPPlanner") as planner:
        result = planner.solve(problem)
    assert result.status == Status.SOLVED_SATISFICING
    assert_plan_is_over_original_problem(problem, result.plan)
    # charge must come before the robot reaches loc_3
    names = [ai.action.name for ai in result.plan.actions]
    assert "charge" in names


def test_numeric_counter():
    problem = numeric_counter_problem(threshold=3)
    with OneshotPlanner(name="ASPPlanner") as planner:
        result = planner.solve(problem)
    assert result.status == Status.SOLVED_SATISFICING
    names = [ai.action.name for ai in result.plan.actions]
    assert names.count("tick") == 3 and names[-1] == "finish"
    assert_plan_is_over_original_problem(problem, result.plan)


def test_numeric_goal_formula():
    """A goal stated as `battery(rover1) >= 2` (numeric comparison, not a
    boolean fluent) must plan and validate against the original problem."""
    problem = rover_recharge_problem(target=2)
    with OneshotPlanner(name="ASPPlanner") as planner:
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

    with OneshotPlanner(name="ASPPlanner") as planner:
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

    with OneshotPlanner(name="ASPPlanner", params={"max_horizon": 4}) as planner:
        result = planner.solve(problem)
    assert result.status == Status.UNSOLVABLE_INCOMPLETELY
    assert result.plan is None


def test_conjunctive_numeric_and_boolean_goal():
    """A goal mixing a numeric comparison and a boolean fluent must satisfy
    both -- the AND split routes each conjunct to its own goal path."""
    problem = rover_recharge_problem(target=2, with_bool_goal=True)
    with OneshotPlanner(name="ASPPlanner") as planner:
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

    with OneshotPlanner(name="ASPPlanner") as planner:
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
    with OneshotPlanner(name="ASPPlanner") as planner:
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
