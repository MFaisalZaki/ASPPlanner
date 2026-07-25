"""Durative actions -> snap actions, shared by both backends.

PDDL 2.1's durative actions are encoded the way SMTPlan's happening encoder
encodes them: each one is split into the pair of instantaneous *snap* actions
that carry its at-start and at-end conditions and effects, leaving only the
over-all conditions and the duration to be re-attached by whatever couples the
two halves back together (the temporal layer of
``aspplanners/plasp/encodings/sequential-horizon.lp`` for the PLASP backend, the
``run``/``rem`` atoms of :mod:`aspplanners.abaplan.encoder` for the ABA one).

Both backends reason over integers, so durations are put on an integer grid
here as well: see :func:`time_unit`.
"""

from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from itertools import chain
from math import gcd
from typing import List, Tuple

from unified_planning.model import (
    Action,
    DurativeAction,
    InstantaneousAction,
    Problem,
)
from unified_planning.shortcuts import EffectKind


# How many times finer than the greatest common divisor of the task's durations
# the integer time grid is, and so how small the minimum separation between two
# happenings (SMTPlan's epsilon, the real constant 1/10) can be. Domains that
# need required concurrency need it above 1 -- match-cellar's mend_fuse has to
# fit strictly inside light_match, and both durations sit on the same grid, so
# with a grid no finer than that there is nowhere to put the two extra
# happenings. Raising it costs grounding: the encodings' remaining-duration
# recursion is quadratic in the largest scaled duration.
DEFAULT_TIME_SCALE = 10


def effect_groups(action: Action):
    """``(timing, effects)`` pairs of an action; an instantaneous action has a
    single group under the timing ``None``, a durative one has one per timing."""
    if isinstance(action, DurativeAction):
        return list(action.effects.items())
    return [(None, list(action.effects))]


def all_effects(action: Action):
    """Every effect of an action, whatever its timing."""
    return list(chain.from_iterable(effects for _, effects in effect_groups(action)))


def add_effect(action: Action, effect, timing=None) -> None:
    """Re-add an effect to an action, at `timing` when the action is durative."""
    where = () if timing is None else (timing,)
    if effect.kind == EffectKind.ASSIGN:
        action.add_effect(*where, effect.fluent, effect.value, effect.condition, forall=effect.forall)
    elif effect.kind == EffectKind.DECREASE:
        action.add_decrease_effect(*where, effect.fluent, effect.value, effect.condition, forall=effect.forall)
    elif effect.kind == EffectKind.INCREASE:
        action.add_increase_effect(*where, effect.fluent, effect.value, effect.condition, forall=effect.forall)


@dataclass(frozen=True)
class FluentDuration:
    """A duration the encoding reads off a static fluent instead of a number.

    `(= ?duration (travel-time ?a ?b))` is only a number once the parameters are
    bound, and the task is not necessarily ground by the time it reaches the
    encoding. So the lookup is deferred: the fluent's value in the initial state,
    times `multiplier` over `divisor`, is the duration in whole time units --
    the same scaling :func:`time_unit` applies to a constant, just carried into
    the encoding rather than performed here.
    """
    fluent: object
    multiplier: int
    divisor: int


@dataclass
class SnapDecomposition:
    """One durative action split into its two snap actions.

    `start` and `end` are ordinary instantaneous actions holding the at-start
    and at-end conditions and effects; `overall` are the conditions that belong
    to neither, which the interval has to keep true throughout; `duration` is
    the admissible duration interval in whole time units (see :func:`time_unit`),
    or a :class:`FluentDuration` when the task states it as a static fluent.
    """
    action: DurativeAction
    start: InstantaneousAction
    end: InstantaneousAction
    overall: List
    duration: object

    @property
    def snap_names(self):
        return {self.start.name: 'start', self.end.name: 'end'}


def durative_actions(problem: Problem) -> List[DurativeAction]:
    return [a for a in problem.actions if isinstance(a, DurativeAction)]


def split_durative_actions(problem: Problem, time_scale: int = DEFAULT_TIME_SCALE):
    """``(time_unit, decompositions)`` for every durative action of `problem`.

    On a task without durative actions this is ``(Fraction(1), [])``, so callers
    can run it unconditionally.
    """
    actions = durative_actions(problem)
    if not actions:
        return Fraction(1), []
    unit = time_unit(problem, actions, time_scale)
    decompositions = [_decompose(problem, da, unit) for da in actions]
    # The snap actions join the task's own actions, so their names have to be
    # free: a hand-written `foo_start` alongside a durative `foo` would silently
    # merge the two in every encoding downstream.
    taken = {a.name for a in problem.actions}
    for snap in decompositions:
        for name in (snap.start.name, snap.end.name):
            if name in taken:
                raise ValueError(
                    f"Splitting durative action {snap.action.name!r} needs the name "
                    f"{name!r}, which the task already uses for another action.")
    return unit, decompositions


def time_unit(problem: Problem, actions: List[DurativeAction],
              time_scale: int = DEFAULT_TIME_SCALE) -> Fraction:
    """The real duration one integer time unit of an encoding stands for.

    Durations are put on the coarsest grid that represents all of them exactly
    -- their greatest common divisor -- and that grid is then refined
    `time_scale` times, so the minimum separation between two happenings is
    1/time_scale of the greatest common divisor. Normalising by the gcd first is
    what keeps the integers small: durations of 100 and 150 become 20 and 30,
    not 1000 and 1500.
    """
    if time_scale < 1:
        raise ValueError(f"time_scale must be a positive integer, got {time_scale!r}")
    bounds = [value
              for da in actions
              for side in (da.duration.lower, da.duration.upper)
              for value in duration_bound(problem, da, side)]
    if any(bound <= 0 for bound in bounds):
        raise NotImplementedError(
            "A durative action has a non-positive duration bound; the happening "
            "encoding needs every action to span a positive amount of time.")
    denominator = 1
    for bound in bounds:
        denominator = denominator * bound.denominator // gcd(denominator, bound.denominator)
    numerator = 0
    for bound in bounds:
        numerator = gcd(numerator, int(bound * denominator))
    return Fraction(numerator, denominator) / time_scale


def duration_bound(problem: Problem, da: DurativeAction, expression) -> List[Fraction]:
    """Every value one side of a duration constraint can take, as exact rationals.

    A constant contributes itself, and so does a *ground* static fluent (one no
    action ever writes), read off the initial state. A static fluent still
    carrying parameters contributes every value it takes there -- one per
    binding, and the encoding does not know which bindings survive, so all of
    them go into the time grid. A duration that depends on the *state* is
    rejected: an action's duration is fixed when it starts.
    """
    values = _duration_values(problem, expression)
    if values:
        return values
    raise NotImplementedError(
        f"Duration bound {expression} of durative action {da.name!r} is neither a "
        "constant nor a static fluent with known initial values.")


def _duration_values(problem: Problem, expression) -> List[Fraction]:
    """The values `expression` can take, or [] if it is not a usable duration."""
    if expression.is_int_constant() or expression.is_real_constant():
        return [Fraction(expression.constant_value())]
    if not expression.is_fluent_exp() or _is_written(problem, expression.fluent().name):
        return []
    if _is_ground(expression):
        value = problem.initial_values.get(expression)
        if value is not None and (value.is_int_constant() or value.is_real_constant()):
            return [Fraction(value.constant_value())]
        return []
    return [Fraction(value.constant_value())
            for key, value in problem.initial_values.items()
            if key.fluent().name == expression.fluent().name
            and (value.is_int_constant() or value.is_real_constant())]


def _is_ground(expression) -> bool:
    return not any(arg.is_parameter_exp() for arg in expression.args)


def _is_written(problem: Problem, fluent_name: str) -> bool:
    return any(eff.fluent.fluent().name == fluent_name
               for action in problem.actions for eff in all_effects(action))


def _decompose(problem: Problem, da: DurativeAction, unit: Fraction) -> SnapDecomposition:
    """Split one durative action into its snap actions and duration interval.

    The at-start condition/effect pair becomes one instantaneous action and the
    at-end pair another; the over-all conditions belong to neither. A condition
    on a *closed* end of the invariant interval additionally becomes a
    precondition of the snap at that end, which is where PDDL 2.1 puts it.
    """
    if da.simulated_effects:
        raise NotImplementedError(
            f"Simulated effects in durative action {da.name!r} are not supported.")
    if getattr(da, 'continuous_effects', None):
        raise NotImplementedError(
            f"Continuous effects in durative action {da.name!r} are not supported; the "
            "temporal encoding covers PDDL 2.1 durative actions, not PDDL+ processes.")

    params = OrderedDict((p.name, p.type) for p in da.parameters)
    snaps = {
        'start': InstantaneousAction(f"{da.name}_start", params, da.environment),
        'end':   InstantaneousAction(f"{da.name}_end", params, da.environment),
    }

    overall = []
    for interval, conditions in da.conditions.items():
        lower, upper = _timepoint(da, interval.lower), _timepoint(da, interval.upper)
        if lower == upper:
            for condition in conditions:
                snaps[lower].add_precondition(condition)
            continue
        if (lower, upper) != ('start', 'end'):
            raise NotImplementedError(
                f"Condition interval {interval} of durative action {da.name!r} is not an "
                "at-start, at-end or over-all condition.")
        overall += list(conditions)
        # An over-all interval closed at an end also constrains the state at
        # that end, which the open-interval invariant check does not see.
        if not interval.is_left_open():
            for condition in conditions:
                snaps['start'].add_precondition(condition)
        if not interval.is_right_open():
            for condition in conditions:
                snaps['end'].add_precondition(condition)

    for timing, effects in da.effects.items():
        snap = snaps[_timepoint(da, timing)]
        for effect in effects:
            add_effect(snap, effect)

    return SnapDecomposition(action=da, start=snaps['start'], end=snaps['end'],
                             overall=overall,
                             duration=_scaled_duration(problem, da, unit))


def _timepoint(da: DurativeAction, timing) -> str:
    """'start' or 'end' for a timing at an endpoint of a durative action."""
    if timing.delay != 0:
        raise NotImplementedError(
            f"Durative action {da.name!r} has a condition or effect at a non-zero delay "
            f"from its endpoints ({timing}); the happening encoding places them at the "
            "snap actions only.")
    return 'start' if timing.is_from_start() else 'end'


def _scaled_duration(problem: Problem, da: DurativeAction, unit: Fraction):
    """The durative action's duration interval in whole time units.

    ``(int, int)`` for numeric bounds; a :class:`FluentDuration` when the task
    states the duration as a static fluent, whose value the encoding looks up
    per parameter binding. An open bound is nudged inwards by one unit: on an
    integer grid, ``?duration > lo`` is ``?duration >= lo + 1``.
    """
    lower_expression, upper_expression = da.duration.lower, da.duration.upper
    # A fluent that still carries parameters has no value yet, so its lookup is
    # deferred to the encoding; a ground one (the task was pre-ground) resolves
    # here like any constant.
    deferred = [e for e in (lower_expression, upper_expression)
                if e.is_fluent_exp() and not _is_ground(e)]
    if deferred:
        if (lower_expression != upper_expression
                or da.duration.is_left_open() or da.duration.is_right_open()):
            raise NotImplementedError(
                f"Durative action {da.name!r} states a duration interval "
                f"[{lower_expression}, {upper_expression}] over a fluent whose parameters "
                "are not bound; a parameterised duration is only supported as a fixed "
                "`(= ?duration (f ...))`.")
        # unit = numerator/denominator, so `raw * denominator / numerator` is the
        # duration in whole time units -- exact, because the unit divides every
        # value the fluent takes (that is what time_unit computed it from).
        return FluentDuration(fluent=lower_expression,
                              multiplier=unit.denominator, divisor=unit.numerator)

    lower = duration_bound(problem, da, lower_expression)[0] / unit
    upper = duration_bound(problem, da, upper_expression)[0] / unit
    assert lower.denominator == 1 and upper.denominator == 1, (
        f"Duration of {da.name!r} is not a whole number of time units.")
    lower = int(lower) + (1 if da.duration.is_left_open() else 0)
    upper = int(upper) - (1 if da.duration.is_right_open() else 0)
    if lower > upper:
        raise ValueError(
            f"Durative action {da.name!r} has an empty duration interval "
            f"[{lower}, {upper}] in scaled time units.")
    if lower < 1:
        raise NotImplementedError(
            f"Durative action {da.name!r} can have zero duration; the happening encoding "
            "needs every action to span at least one time unit.")
    return lower, upper
