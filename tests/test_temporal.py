"""End-to-end tests for the temporal (PDDL 2.1 durative-action) encoding.

Both backends encode durative actions as SMTPlan-style happenings, so the
behavioural tests run against both: the plan has to be a schedule (a UP
``TimeTriggeredPlan``), stated over the *original* problem's actions and
objects, and it has to validate against that problem with UP's time-triggered
validator -- which is what independently re-checks the durations and the
over-all conditions the encodings are responsible for.
"""

import os
from fractions import Fraction

import pytest

import aspplanners  # noqa: F401 -- registers both engines
from unified_planning.engines import PlanGenerationResultStatus as Status
from unified_planning.io import PDDLReader
from unified_planning.plans import SequentialPlan, TimeTriggeredPlan
from unified_planning.shortcuts import (
    DurativeAction,
    EndTiming,
    Fluent,
    GE,
    InstantaneousAction,
    IntType,
    OneshotPlanner,
    OpenTimeInterval,
    PlanValidator,
    Problem,
    StartTiming,
    get_environment,
)

from aspplanners.common.validation import as_validated
from aspplanners.plasp.planner import PLASPPlanner

PDDL_DIR = os.path.join(os.path.dirname(__file__), "pddl")

def _aba_planner(problem, **kwargs):
    pytest.importorskip("aspforaba", reason="ABA backend requires the optional `aba` extra")
    from aspplanners.abaplan.planner import ABAPlan
    return ABAPlan(problem, **kwargs)


def _plasp_planner(problem, **kwargs):
    return PLASPPlanner(problem, **kwargs)


BACKENDS = [("plasp", _plasp_planner), ("aba", _aba_planner)]
backends = pytest.mark.parametrize("backend,make_planner", BACKENDS,
                                   ids=[b[0] for b in BACKENDS])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse(domain, problem_file="problem.pddl"):
    return PDDLReader().parse_problem(
        os.path.join(PDDL_DIR, domain, "domain.pddl"),
        os.path.join(PDDL_DIR, domain, problem_file),
    )


def intervals(plan):
    """``{action name: [(start, end), ...]}`` over a TimeTriggeredPlan.

    An instantaneous step is the degenerate interval ``(t, t)``.
    """
    out = {}
    for start, action_instance, duration in plan.timed_actions:
        end = start if duration is None else start + duration
        out.setdefault(action_instance.action.name, []).append((start, end))
    return out


def assert_schedule_is_over_original_problem(problem, plan):
    """The lifted schedule must be stated in terms of the task passed to the
    planner -- every step references one of ITS action objects (identity, not
    just an equal name) with matching arity over its objects -- and validate
    against it with the time-triggered validator."""
    assert isinstance(plan, TimeTriggeredPlan), f"expected a schedule, got {type(plan).__name__}"
    original_actions = {id(a) for a in problem.actions}
    original_objects = {o.name: o for o in problem.all_objects}
    for start, action_instance, duration in plan.timed_actions:
        assert id(action_instance.action) in original_actions, (
            f"plan action {action_instance.action.name!r} is not (identically) an action "
            "of the original problem -- map-back is broken"
        )
        assert len(action_instance.actual_parameters) == len(action_instance.action.parameters)
        for actual in action_instance.actual_parameters:
            obj = actual.object()
            assert original_objects.get(obj.name) == obj, (
                f"plan argument {obj.name!r} of {action_instance} is not an object of the "
                "original problem"
            )
        assert isinstance(action_instance.action, DurativeAction) == (duration is not None), (
            f"{action_instance} is scheduled with duration {duration}, which does not match "
            "whether it is a durative action"
        )
    with PlanValidator(name="up_time_triggered_validator") as validator:
        result = validator.validate(as_validated(problem), plan)
    assert str(result.status) == "ValidationResultStatus.VALID", (
        f"schedule does not validate against the original problem: {result}"
    )


# ---------------------------------------------------------------------------
# Required concurrency: match-cellar
# ---------------------------------------------------------------------------

@backends
def test_matchcellar_nests_the_fuse_inside_the_match(backend, make_planner):
    problem = parse("matchcellar")
    planner = make_planner(problem)
    plan = planner.plan(max_horizon=8)

    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert_schedule_is_over_original_problem(problem, plan)

    by_action = intervals(plan)
    (mend_start, mend_end), = by_action["mend_fuse"]
    (light_start, light_end), = by_action["light_match"]
    # The point of the domain: no schedule that only sequences the two exists.
    assert light_start < mend_start and mend_end < light_end, (
        f"mend_fuse [{mend_start}, {mend_end}] is not strictly inside "
        f"light_match [{light_start}, {light_end}]"
    )
    assert mend_end - mend_start == 5 and light_end - light_start == 6


@backends
def test_durations_come_from_the_task(backend, make_planner):
    """Static-fluent durations are read off the initial state, and a duration
    inequality is honoured within its bounds."""
    problem = parse("tempdrive")
    planner = make_planner(problem)
    plan = planner.plan(max_horizon=10)

    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert_schedule_is_over_original_problem(problem, plan)

    travel_time = {(str(k.args[0]), str(k.args[1])): int(v.constant_value())
                   for k, v in problem.initial_values.items()
                   if k.fluent().name == "travel-time"}
    drives = []
    for start, action_instance, duration in plan.timed_actions:
        if action_instance.action.name != "drive":
            continue
        a, b = (str(p) for p in action_instance.actual_parameters)
        assert duration == travel_time[(a, b)], (
            f"drive({a}, {b}) is scheduled for {duration}, not its travel-time "
            f"{travel_time[(a, b)]}"
        )
        drives.append((start, start + duration))

    (recharge_start, recharge_end), = intervals(plan)["recharge"]
    assert 2 <= recharge_end - recharge_start <= 5, "duration inequality bounds ignored"
    # recharge cuts the charge a drive needs over-all, so the two may not overlap.
    for start, end in drives:
        assert recharge_end <= start or end <= recharge_start, (
            f"recharge [{recharge_start}, {recharge_end}] overlaps drive [{start}, {end}] "
            "even though it breaks the drive's over-all condition"
        )


# ---------------------------------------------------------------------------
# Over-all conditions and open intervals
# ---------------------------------------------------------------------------

def interleaving_problem(with_invariant):
    """`flip` is only applicable while `work` is running, and it destroys `on`.

    With the over-all condition, the only plan that reaches the goal is one the
    invariant forbids, so the task is unsolvable; without it, the same plan is
    the answer. That isolates the over-all check from everything else.
    """
    on, busy, done, flipped = Fluent("on"), Fluent("busy"), Fluent("done"), Fluent("flipped")

    work = DurativeAction("work")
    work.set_fixed_duration(4)
    work.add_effect(StartTiming(), busy, True)
    work.add_effect(EndTiming(), busy, False)
    work.add_effect(EndTiming(), done, True)
    if with_invariant:
        work.add_condition(OpenTimeInterval(StartTiming(), EndTiming()), on)

    flip = InstantaneousAction("flip")
    flip.add_precondition(busy)
    flip.add_effect(on, False)
    flip.add_effect(flipped, True)

    problem = Problem("interleaving")
    for fluent, value in ((on, True), (busy, False), (done, False), (flipped, False)):
        problem.add_fluent(fluent, default_initial_value=value)
    problem.add_action(work)
    problem.add_action(flip)
    problem.add_goal(done)
    problem.add_goal(flipped)
    return problem


@backends
def test_over_all_condition_forbids_the_only_interleaving(backend, make_planner):
    planner = make_planner(interleaving_problem(with_invariant=True))
    planner.plan(max_horizon=6)
    assert planner.status == Status.UNSOLVABLE_INCOMPLETELY, (
        "an action that breaks an over-all condition was scheduled inside the interval"
    )


@backends
def test_the_same_interleaving_is_a_plan_without_the_invariant(backend, make_planner):
    problem = interleaving_problem(with_invariant=False)
    planner = make_planner(problem)
    plan = planner.plan(max_horizon=6)
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert_schedule_is_over_original_problem(problem, plan)
    (work_start, work_end), = intervals(plan)["work"]
    (flip_at, _), = intervals(plan)["flip"]
    assert work_start < flip_at < work_end


@backends
def test_a_durative_action_left_open_is_not_a_plan(backend, make_planner):
    """The goal of `a` is reached by its at-start effect alone, but its end snap
    can never fire; SMTPlan puts !run(a, H-1) in its goal for exactly this."""
    reached, unreachable = Fluent("reached"), Fluent("unreachable")
    action = DurativeAction("a")
    action.set_fixed_duration(2)
    action.add_effect(StartTiming(), reached, True)
    action.add_condition(EndTiming(), unreachable)

    problem = Problem("dangling")
    problem.add_fluent(reached, default_initial_value=False)
    problem.add_fluent(unreachable, default_initial_value=False)
    problem.add_action(action)
    problem.add_goal(reached)

    planner = make_planner(problem)
    planner.plan(max_horizon=6)
    assert planner.status == Status.UNSOLVABLE_INCOMPLETELY, (
        "a durative action was left running at the horizon"
    )


# ---------------------------------------------------------------------------
# Time resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("time_scale,solvable", [(1, False), (2, True), (10, True)])
def test_time_scale_sets_the_happening_resolution(time_scale, solvable):
    """Happenings sit on an integer grid `time_scale` times finer than the
    durations' gcd. Match-cellar's 6 and 5 share a gcd of 1, so on that grid
    itself there is no room to open and close the fuse strictly inside the
    match; one refinement is already enough."""
    planner = PLASPPlanner(parse("matchcellar"), time_scale=time_scale)
    planner.plan(max_horizon=8)
    assert (planner.status == Status.SOLVED_SATISFICING) == solvable, planner.logs


# ---------------------------------------------------------------------------
# The lifted temporal path
# ---------------------------------------------------------------------------

def _ground_actions(planner, horizon):
    """The action/1 atoms gringo instantiates, and the total ground program."""
    import clingo

    control = clingo.Control(["-n", "1"])
    for path in planner.encoding_paths:
        control.load(path)
    control.add("base", [], planner.task_facts)
    control.ground([("base", [])])
    actions = {str(a.symbol) for a in control.symbolic_atoms.by_signature("action", 1)}
    control.ground([("check", [clingo.Number(horizon)])]
                   + [("step", [clingo.Number(t)]) for t in range(1, horizon + 1)])
    return actions, sum(1 for _ in control.symbolic_atoms)


def test_the_lifted_temporal_encoding_grounds_no_wider_than_a_pre_grounder():
    """A temporal task is handed to clingo whole, because no reachability
    grounder takes it and a plain enumerating one would only do gringo's work
    twice. This pins that the lifted program really is no bigger: `drive` is
    declared for 2 of the 9 type-consistent pairs, exactly as up_grounder leaves
    it -- the start snap through its static `link` precondition, and the end snap
    because it is declared under its start."""
    from unified_planning.shortcuts import CompilationKind

    problem = parse("tempdrive")
    assert not any("grounder" in name for name, _kind
                   in PLASPPlanner.__new__(PLASPPlanner)._check_compilationlist(problem, None))

    pre_ground = [
        ["up_quantifiers_remover", CompilationKind.QUANTIFIERS_REMOVING],
        ["up_negative_conditions_remover", CompilationKind.NEGATIVE_CONDITIONS_REMOVING],
        ["up_disjunctive_conditions_remover", CompilationKind.DISJUNCTIVE_CONDITIONS_REMOVING],
        ["up_grounder", CompilationKind.GROUNDING],
    ]
    lifted_actions, lifted_atoms = _ground_actions(PLASPPlanner(problem), 6)
    _ground_names, ground_atoms = _ground_actions(
        PLASPPlanner(problem, compilationlist=pre_ground), 6)

    assert len(lifted_actions) == 6, sorted(lifted_actions)   # 2 drives + recharge, x2 snaps
    assert lifted_atoms <= ground_atoms
    drive_ends = {a for a in lifted_actions if "drive_end" in a}
    assert len(drive_ends) == 2, (
        f"the end snap ground to {len(drive_ends)} bindings, so it is not being declared "
        f"under its start snap: {sorted(drive_ends)}")


def test_a_parameterised_duration_is_looked_up_in_the_encoding():
    """`(= ?duration (travel-time ?a ?b))` has no value until its parameters are
    bound, so on a lifted task the durationValue fact reads the initial state
    rather than carrying a number."""
    planner = PLASPPlanner(parse("tempdrive"))
    durations = [line for line in planner.compiled_task.facts["_durative"]
                 if line.startswith("durationValue")]
    drive, = [line for line in durations if "drive" in line]
    assert "initialState(" in drive and "travel_time" in drive, drive
    # recharge's bounds are numbers, so they stay numbers (2..5 at unit 1/10).
    recharge, = [line for line in durations if "recharge" in line]
    assert "20..50" in recharge and "initialState(" not in recharge, recharge


def test_durations_are_scaled_exactly_not_rounded():
    problem = parse("matchcellar")
    planner = PLASPPlanner(problem, time_scale=10)
    assert planner.compiled_task.time_unit == Fraction(1, 10)
    durations = {line for line in planner.compiled_task.facts["_durative"]
                 if line.startswith("durationValue")}
    assert any(", 60)" in line for line in durations), durations
    assert any(", 50)" in line for line in durations), durations


# ---------------------------------------------------------------------------
# The temporal layer stays out of the way of everything else
# ---------------------------------------------------------------------------

def test_a_task_without_durative_actions_still_yields_a_sequential_plan():
    from test_planner import robot_line_problem

    problem = robot_line_problem(n_locations=4)
    planner = PLASPPlanner(problem)
    assert not planner.compiled_task.is_temporal
    assert planner.compiled_task.time_unit == 1

    plan = planner.plan(max_horizon=6)
    assert planner.status == Status.SOLVED_SATISFICING
    assert isinstance(plan, SequentialPlan)
    assert [ai.action.name for ai in plan.actions] == ["move"] * 3


# ---------------------------------------------------------------------------
# Unsupported fragments fail loudly
# ---------------------------------------------------------------------------

def test_a_zero_duration_action_is_rejected():
    fluent = Fluent("p")
    action = DurativeAction("a")
    action.set_fixed_duration(0)
    action.add_effect(EndTiming(), fluent, True)
    problem = Problem("zero")
    problem.add_fluent(fluent, default_initial_value=False)
    problem.add_action(action)
    problem.add_goal(fluent)

    with pytest.raises(NotImplementedError, match="positive amount of time"):
        PLASPPlanner(problem)


def test_an_intermediate_effect_is_rejected():
    fluent = Fluent("p")
    action = DurativeAction("a")
    action.set_fixed_duration(4)
    action.add_effect(StartTiming(1), fluent, True)
    problem = Problem("intermediate")
    problem.add_fluent(fluent, default_initial_value=False)
    problem.add_action(action)
    problem.add_goal(fluent)

    with pytest.raises(NotImplementedError, match="non-zero delay"):
        PLASPPlanner(problem)


def test_a_snap_action_name_the_task_already_uses_is_rejected():
    fluent = Fluent("p")
    durative = DurativeAction("foo")
    durative.set_fixed_duration(2)
    durative.add_effect(EndTiming(), fluent, True)
    clash = InstantaneousAction("foo_start")
    clash.add_effect(fluent, True)

    problem = Problem("clash")
    problem.add_fluent(fluent, default_initial_value=False)
    problem.add_action(durative)
    problem.add_action(clash)
    problem.add_goal(fluent)

    with pytest.raises(ValueError, match="already uses for another action"):
        PLASPPlanner(problem)


@backends
def test_an_action_that_breaks_its_own_invariant_never_runs(backend, make_planner):
    """A start snap deleting one of its own over-all conditions makes the
    interval illegal from the first state it covers. That action drops out, but
    the rest of the task still solves."""
    on, done = Fluent("on"), Fluent("done")
    bad = DurativeAction("bad")
    bad.set_fixed_duration(2)
    bad.add_condition(OpenTimeInterval(StartTiming(), EndTiming()), on)
    bad.add_effect(StartTiming(), on, False)
    bad.add_effect(EndTiming(), done, True)

    problem = Problem("self_breaking")
    problem.add_fluent(on, default_initial_value=True)
    problem.add_fluent(done, default_initial_value=False)
    problem.add_action(bad)
    problem.add_goal(done)

    planner = make_planner(problem)
    planner.plan(max_horizon=5)
    assert planner.status == Status.UNSOLVABLE_INCOMPLETELY, planner.logs

    good = InstantaneousAction("good")
    good.add_effect(done, True)
    problem.add_action(good)
    planner = make_planner(problem)
    plan = planner.plan(max_horizon=5)
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert [ai.action.name for _s, ai, _d in plan.timed_actions] == ["good"]


def test_aba_rejects_a_numeric_over_all_condition():
    """The ABA reduction turns an over-all condition into a static test on the
    actions that could break it, which only exists for boolean ones."""
    pytest.importorskip("aspforaba", reason="ABA backend requires the optional `aba` extra")
    from aspplanners.abaplan.planner import ABAPlan

    level, done = Fluent("level", IntType(0, 10)), Fluent("done")
    action = DurativeAction("a")
    action.set_fixed_duration(2)
    action.add_condition(OpenTimeInterval(StartTiming(), EndTiming()), GE(level, 1))
    action.add_effect(EndTiming(), done, True)

    problem = Problem("numeric_over_all")
    problem.add_fluent(level, default_initial_value=5)
    problem.add_fluent(done, default_initial_value=False)
    problem.add_action(action)
    problem.add_goal(done)

    with pytest.raises(NotImplementedError, match="boolean over-all conditions only"):
        ABAPlan(problem)
    # The PLASP backend evaluates it against numval and solves the same task.
    planner = PLASPPlanner(problem)
    planner.plan(max_horizon=4)
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs


# ---------------------------------------------------------------------------
# UP engine interface
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("engine", ["PLASPPlanner", "ABAPlanner"])
def test_engine_accepts_and_solves_a_temporal_task(engine):
    if engine == "ABAPlanner":
        pytest.importorskip("aspforaba", reason="ABA backend requires the optional `aba` extra")
    problem = parse("matchcellar")
    with OneshotPlanner(name=engine) as planner:
        assert planner.supports(problem.kind)
        result = planner.solve(problem)
    assert result.status == Status.SOLVED_SATISFICING, [str(m) for m in result.log_messages or []]
    assert_schedule_is_over_original_problem(problem, result.plan)


def test_plasp_accepts_a_makespan_metric_and_an_undefined_initial_value():
    """The two features every IPC temporal instance carries.

    `(:metric minimize (total-time))` is MAKESPAN and a `(:functions ...)` cross
    product that `(:init ...)` only partly fills is UNDEFINED_INITIAL_NUMERIC;
    neither says anything about whether a plan exists, and leaving them out of
    supported_kind() rejected the whole temporal track at `supports()`.
    """
    problem = parse("tempdrive", "problem-ipc-shaped.pddl")
    assert {"MAKESPAN", "UNDEFINED_INITIAL_NUMERIC"} <= problem.kind.features

    with OneshotPlanner(name="PLASPPlanner") as planner:
        assert planner.supports(problem.kind)
        result = planner.solve(problem)
    assert result.status == Status.SOLVED_SATISFICING, [str(m) for m in result.log_messages or []]
    assert_schedule_is_over_original_problem(problem, result.plan)
    # The undefined `travel-time` entries default to 0, but the durations were
    # read off the initial state before that, so the two drives keep their own.
    assert sorted(end - start for start, end in intervals(result.plan)["drive"]) == [2, 3]


def test_aba_takes_the_metric_but_still_declines_an_undefined_initial_value():
    """The ABA reduction is over ground STRIPS and no installed UP grounder
    accepts an undefined initial value, so it has to keep saying no to that --
    at `supports()`, rather than raising from inside the encoder."""
    with_metric = parse("tempdrive", "problem-metric-only.pddl")
    undefined = parse("tempdrive", "problem-ipc-shaped.pddl")
    assert "MAKESPAN" in with_metric.kind.features
    assert "UNDEFINED_INITIAL_NUMERIC" not in with_metric.kind.features

    with OneshotPlanner(name="ABAPlanner") as planner:
        assert planner.supports(with_metric.kind)
        assert not planner.supports(undefined.kind)


@pytest.mark.parametrize("engine", ["PLASPPlanner", "ABAPlanner"])
def test_engines_are_satisficing_not_optimal(engine):
    """Metrics are accepted, never optimised: the search returns the first plan
    it finds, so a caller asking for an optimal engine must not be given one."""
    from unified_planning.engines import OptimalityGuarantee

    factory = get_environment().factory
    engine_class = factory.engine(engine)
    assert engine_class.satisfies(OptimalityGuarantee.SATISFICING)
    assert not engine_class.satisfies(OptimalityGuarantee.SOLVED_OPTIMALLY)


# ---------------------------------------------------------------------------
# Rescaling numeric values leaves the time grid alone
# ---------------------------------------------------------------------------

def _fractional_alongside(problem, duration_fluent_in_condition=False):
    """`problem` plus an unrelated fluent whose values force a rescaling.

    The point is that rescaling has to touch the numeric fluents and *not* the
    durations, which live on their own integer grid and are turned back into
    absolute times by multiplying by the time unit.
    """
    from fractions import Fraction
    from unified_planning.shortcuts import LE, RealType

    task = problem.clone()
    fuel = Fluent("fuel", RealType())
    task.add_fluent(fuel, default_initial_value=Fraction(0))
    burn = InstantaneousAction("burn")
    burn.add_increase_effect(fuel(), Fraction(1, 2))
    if duration_fluent_in_condition:
        # `travel-time` is what `drive` reads its duration off, so asking for it
        # in scaled units too is contradictory -- but only once it needs a
        # factor of its own, which a fractional value is what forces.
        travel = task.fluent("travel-time")
        locations = list(task.all_objects)
        task.set_initial_value(travel(locations[0], locations[1]), Fraction(3, 2))
        burn.add_precondition(LE(travel(locations[0], locations[1]), 1))
    task.add_action(burn)
    return task


def test_rescaling_numeric_values_leaves_the_durations_alone():
    """A fractional numeric fluent forces a factor of 2; the schedule's own
    numbers -- the time unit and every durationValue -- must not move with it."""
    plain = PLASPPlanner(parse("tempdrive"))
    scaled = PLASPPlanner(_fractional_alongside(parse("tempdrive")))

    assert plain.compiled_task.numeric_scale == 1
    assert scaled.compiled_task.numeric_scale == 2
    assert scaled.compiled_task.time_unit == plain.compiled_task.time_unit

    def durations(planner):
        return {line for line in planner.compiled_task.facts["_durative"]
                if line.startswith("durationValue")}

    assert durations(scaled) == durations(plain)


def test_a_fractional_duration_fluent_read_in_a_numeric_condition_is_rejected():
    """`travel-time` is looked up in the initial state at grounding time, and a
    fractional value puts it on a grid of its own; a condition reading it would
    then see the scaled stand-in, and it cannot be both."""
    task = _fractional_alongside(parse("tempdrive"), duration_fluent_in_condition=True)
    with pytest.raises(NotImplementedError, match="read as a durative action's duration"):
        PLASPPlanner(task)


def test_an_integral_duration_fluent_may_still_be_compared():
    """The conflict is about the *grid*, not about the two readings. A duration
    fluent whose values are already whole needs no factor, so comparing it
    alongside a task that rescales something else is fine -- which a task-wide
    factor could not have allowed, since it would have moved this fluent too."""
    from unified_planning.shortcuts import LE as _LE

    task = _fractional_alongside(parse("tempdrive"))
    travel = task.fluent("travel-time")
    locations = list(task.all_objects)
    burn = next(a for a in task.actions if a.name == "burn")
    burn.add_precondition(_LE(travel(locations[0], locations[1]), 10))

    planner = PLASPPlanner(task)
    assert planner.compiled_task.numeric_scale == 2
    assert any(line.startswith("numPrecondition(")
               for line in planner.compiled_task.fact_lines)


def test_a_temporal_task_with_fractional_values_still_schedules():
    """End to end: the rescaled numeric layer and the untouched time grid have to
    coexist in one plan, validated against the original units."""
    from fractions import Fraction
    from unified_planning.shortcuts import GE as _GE, RealType

    task = _fractional_alongside(parse("tempdrive"))
    fuel = task.fluent("fuel")
    task.add_goal(_GE(fuel(), Fraction(1, 2)))

    planner = PLASPPlanner(task)
    plan = planner.plan(max_horizon=12)
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert any(ai.action.name == "burn" for _s, ai, _d in plan.timed_actions)
    assert_schedule_is_over_original_problem(task, plan)


# ---------------------------------------------------------------------------
# State-independent over-all conditions
# ---------------------------------------------------------------------------

def parameter_guard_problem(goal_pair):
    """`link(?a, ?b)` carries `(over all (not (= ?a ?b)))`, satellite's guard.

    Nothing an action does can change whether two parameters are equal, so the
    condition is settled by the binding rather than being an invariant to watch.
    Asking for `linked(x, x)` therefore has no plan, and `linked(x, y)` has one.
    """
    from unified_planning.shortcuts import BoolType, Not, Equals, Object, UserType

    Node = UserType("Node")
    linked = Fluent("linked", BoolType(), a=Node, b=Node)

    link = DurativeAction("link", a=Node, b=Node)
    link.set_fixed_duration(2)
    link.add_condition(OpenTimeInterval(StartTiming(), EndTiming()),
                       Not(Equals(link.parameter("a"), link.parameter("b"))))
    link.add_effect(EndTiming(), linked(link.parameter("a"), link.parameter("b")), True)

    problem = Problem("parameter_guard")
    problem.add_fluent(linked, default_initial_value=False)
    problem.add_objects([Object("x", Node), Object("y", Node)])
    problem.add_action(link)
    problem.add_goal(linked(problem.object(goal_pair[0]), problem.object(goal_pair[1])))
    return problem


@backends
def test_a_parameter_only_over_all_condition_rules_out_the_binding(backend, make_planner):
    """`(over all (not (= ?a ?b)))` must still forbid the equal binding."""
    planner = make_planner(parameter_guard_problem(("x", "x")))
    planner.plan(max_horizon=4)
    assert planner.status == Status.UNSOLVABLE_INCOMPLETELY, (
        "link(x, x) was scheduled even though its over-all condition forbids it"
    )


@backends
def test_a_parameter_only_over_all_condition_still_allows_the_others(backend, make_planner):
    problem = parameter_guard_problem(("x", "y"))
    planner = make_planner(problem)
    plan = planner.plan(max_horizon=4)
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert_schedule_is_over_original_problem(problem, plan)
    assert [ai.action.name for _s, ai, _d in plan.timed_actions] == ["link"]


def test_a_parameter_only_over_all_condition_is_folded_into_the_signature():
    """It should not reach the encoding as an `overall/3` fact at all: folding it
    into the action signature is what keeps the ruled-out bindings out of the
    grounding, which an invariant fact would not do."""
    planner = PLASPPlanner(parameter_guard_problem(("x", "y")))
    lines = planner.compiled_task.fact_lines
    assert not any(line.startswith("overall(") for line in lines), (
        f"expected no overall/3 fact, got {[l for l in lines if l.startswith('overall(')]}"
    )
    assert any("!=" in line and line.startswith("action(") for line in lines), (
        "the parameter test was dropped instead of being folded into the signature"
    )


# ---------------------------------------------------------------------------
# Durations that are arithmetic over static fluents
# ---------------------------------------------------------------------------

@backends
def test_an_arithmetic_duration_is_evaluated_exactly(backend, make_planner):
    """`(/ (distance ?a ?b) (speed ?v))` with distance 5 and speed 2 lasts 2.5.

    Truncating the division anywhere -- and clingo's `/` does truncate -- turns
    that leg into 2, which still schedules and still validates against a
    validator asked the same wrong question, so the durations are checked here
    against the arithmetic itself.
    """
    problem = parse("arithdrive")
    planner = make_planner(problem)
    plan = planner.plan(max_horizon=8)

    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert_schedule_is_over_original_problem(problem, plan)
    legs = {tuple(str(p) for p in ai.actual_parameters): duration
            for _s, ai, duration in plan.timed_actions}
    assert legs[("car", "l1", "l2")] == Fraction(5, 2), legs
    assert legs[("car", "l2", "l3")] == Fraction(3, 2), legs


@backends
def test_a_zero_length_leg_is_not_an_action(backend, make_planner):
    """A binding whose arithmetic duration comes out 0 has no duration at all,
    so it cannot be scheduled even where every other precondition holds."""
    planner = make_planner(parse("arithdrive", "problem-zero-leg.pddl"))
    planner.plan(max_horizon=8)
    assert planner.status == Status.UNSOLVABLE_INCOMPLETELY, (
        "a zero-duration leg was scheduled; the happening encoding needs every "
        "action to span a positive amount of time"
    )


def test_a_zero_length_leg_does_not_sink_the_whole_task():
    """The zero only takes its own binding out. Other actions keep their
    durations and the task still plans -- a task-level rejection would have made
    depots (which states `(distance ?y ?y)` as 0) unencodable."""
    problem = parse("arithdrive")
    problem.set_initial_value(
        problem.fluent("distance")(problem.object("l1"), problem.object("l1")), 0)
    planner = PLASPPlanner(problem)
    plan = planner.plan(max_horizon=8)
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    legs = {tuple(str(p) for p in ai.actual_parameters): d
            for _s, ai, d in plan.timed_actions}
    assert legs[("car", "l1", "l2")] == Fraction(5, 2), legs


def test_an_arithmetic_duration_survives_numeric_rescaling():
    """A fractional value elsewhere in the task rescales the numbers, and the
    fluents inside a duration have to be left out of that.

    `dist(l1) + 1` is 6. Scaling `dist` to 10 alongside everything else would
    make it 11 -- not even a multiple of the right answer, because the constant
    does not scale with it. A bare-fluent duration was already exempt; one
    nested in an expression is not a fluent expression at all, so it had to be
    looked for at depth.
    """
    from unified_planning.shortcuts import Object, Plus, RealType, UserType

    Loc = UserType("Loc")
    dist = Fluent("dist", RealType(), l=Loc)      # static: never an effect
    charge = Fluent("charge", RealType())
    arrived = Fluent("arrived", l=Loc)

    go = DurativeAction("go", l=Loc)
    go.set_fixed_duration(Plus(dist(go.parameter("l")), 1))
    go.add_effect(EndTiming(), arrived(go.parameter("l")), True)
    go.add_increase_effect(EndTiming(), charge, Fraction(1, 2))   # forces rescaling

    problem = Problem("rescaled_arithmetic_duration")
    problem.add_fluent(dist, default_initial_value=0)
    problem.add_fluent(charge, default_initial_value=0)
    problem.add_fluent(arrived, default_initial_value=False)
    problem.add_object(Object("l1", Loc))
    problem.set_initial_value(dist(problem.object("l1")), 5)
    problem.add_action(go)
    problem.add_goal(arrived(problem.object("l1")))

    planner = PLASPPlanner(problem)
    assert planner.compiled_task.numeric_scale > 1, (
        "the fractional increase should have rescaled the task; without that this "
        "test would pass whether or not durations are exempt"
    )
    plan = planner.plan(max_horizon=4)
    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    (_start, _action, duration), = plan.timed_actions
    assert duration == 6, f"dist(l1) + 1 should last 6, got {duration}"


@backends
def test_a_fractional_duration_keeps_its_value(backend, make_planner):
    """`(/ 5.9 2)` is 2.95, and the schedule has to say so.

    The fluents read as durations sit out the task-wide numeric rescaling (it
    would move every reported plan time), so they get an integer grid of their
    own that the duration arithmetic divides back out. Getting that factor wrong
    scales the leg rather than breaking it, which is why the value is asserted
    and not just the status -- and why the schedule is validated against the
    original task, where the duration is still 5.9.
    """
    problem = parse("arithdrive", "problem-fractional.pddl")
    planner = make_planner(problem)
    plan = planner.plan(max_horizon=8)

    assert planner.status == Status.SOLVED_SATISFICING, planner.logs
    assert_schedule_is_over_original_problem(problem, plan)
    legs = {tuple(str(p) for p in ai.actual_parameters): duration
            for _s, ai, duration in plan.timed_actions}
    assert legs[("car", "l1", "l2")] == Fraction(59, 20), legs
    assert legs[("car", "l2", "l3")] == Fraction(3, 2), legs
