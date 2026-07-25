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
)

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
        result = validator.validate(problem, plan)
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
    control.load(planner.encoding_path)
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

@pytest.mark.parametrize("engine", ["ASPPlanner", "ABAPlanner"])
def test_engine_accepts_and_solves_a_temporal_task(engine):
    if engine == "ABAPlanner":
        pytest.importorskip("aspforaba", reason="ABA backend requires the optional `aba` extra")
    problem = parse("matchcellar")
    with OneshotPlanner(name=engine) as planner:
        assert planner.supports(problem.kind)
        result = planner.solve(problem)
    assert result.status == Status.SOLVED_SATISFICING, [str(m) for m in result.log_messages or []]
    assert_schedule_is_over_original_problem(problem, result.plan)
