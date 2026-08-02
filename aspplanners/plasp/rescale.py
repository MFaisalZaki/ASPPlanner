"""Put a task's numeric *values* on an integer grid, in place.

clingo terms are integers, so a task whose fluents take fractional values cannot
be encoded as it stands. Every numeric rule in the encodings is linear and
homogeneous in those values -- ``S1 OP S2`` over ``K*V + C`` sides, ``X + D`` for
an effect, ``K`` for an assignment -- so multiplying *every* fluent value and
*every* additive constant by one positive integer preserves each rule's truth
value exactly. That factor is the least common multiple of the denominators the
task actually uses; ``(increase (count_log) 0.5)`` needs 2, and the plan that
comes back is a plan for the original task, because a plan is a sequence of
actions and carries no units.

Two things are deliberately *not* scaled:

* **Coefficients.** ``(* 2 (x ?f))`` already commutes with the scaling, and
  scaling the 2 as well would square it. A fractional *coefficient* is made
  whole by multiplying its own comparison through instead, which is local and
  needs no help from here (see :class:`~aspplanners.plasp.facts.ASPNumComparison`).
* **Durations.** They live on their own integer grid, computed by
  :func:`~aspplanners.common.temporal.time_unit`, and the planner turns a
  happening back into an absolute time by multiplying by that unit. Scaling them
  by the task-wide factor would make every reported plan time wrong by that
  factor, so a fluent *read* as a duration sits out this pass -- the encoding
  looks its value up in the initial state at grounding time.

  Sitting out the task-wide factor is not the same as staying fractional, and
  satellite states `(= (slew_time ?a ?b) 5.9)`. So each duration fluent gets a
  factor of its own instead (:func:`duration_fluent_scales`), which the
  ``durationValue`` arithmetic divides straight back out -- it is scaled only
  where it is read as a duration, so no plan time moves.

This pass belongs to the PLASP backend rather than to the shared front-end: the
shapes it walks have to track those of :mod:`aspplanners.plasp.facts` exactly,
and it is unsound for the ABA backend, whose arithmetic admits a product of two
fluents (which scales by the *square* of the factor).
"""

from fractions import Fraction
from math import gcd

from aspplanners.common.temporal import all_effects, durative_actions
from aspplanners.plasp.facts import _is_numeric_fnode, _linear_form

# Denominators past this are not a task stated in awkward units, they are a task
# stated in the wrong ones; scaling by such a factor would make the integers
# meaningless long before it made them wrong.
MAX_NUMERIC_SCALE = 10 ** 6


def _fluent_names(expression):
    """Every fluent name `expression` reads, at any depth."""
    if expression.is_fluent_exp():
        return {expression.fluent().name}
    return set().union(set(), *(_fluent_names(arg) for arg in expression.args))


def _duration_fluent_names(task):
    """Every fluent name read as a durative action's duration, at any depth."""
    return {name
            for da in durative_actions(task)
            for bound in (da.duration.lower, da.duration.upper)
            for name in _fluent_names(bound)}


def duration_fluent_scales(task):
    """``{fluent name: factor}`` putting each duration fluent's values on an
    integer grid -- computed, not applied (see :func:`apply_duration_scales`).

    Duration fluents sit out :func:`scale_numeric_constants` because scaling
    them would make every reported plan time wrong by that factor. But their
    values still have to reach clingo as integer terms, and satellite states
    `(= (slew_time ?a ?b) 5.9)`. So each one gets its *own* factor, and the
    duration arithmetic divides that factor straight back out -- the fluent is
    scaled only where it is read as a duration, so nothing else can see it.

    Computed before the durative split and applied after, because the time grid
    (:func:`~aspplanners.common.temporal.time_unit`) has to be built from the
    task's real durations, not from these scaled stand-ins.
    """
    duration_fluents = _duration_fluent_names(task)
    if not duration_fluents:
        return {}

    denominators = {}
    for fluent_exp, value in task.explicit_initial_values.items():
        name = fluent_exp.fluent().name
        if name in duration_fluents and _is_numeric_value(value):
            denominators.setdefault(name, set()).add(
                Fraction(value.constant_value()).denominator)
    for fluent, value in task.fluents_defaults.items():
        if fluent.name in duration_fluents and _is_numeric_value(value):
            denominators.setdefault(fluent.name, set()).add(
                Fraction(value.constant_value()).denominator)

    scales = {}
    for name in duration_fluents:
        scale = 1
        for denominator in denominators.get(name, ()):
            scale = scale * denominator // gcd(scale, denominator)
        scales[name] = scale

    fractional = {name for name, scale in scales.items() if scale > 1}
    if not fractional:
        return scales
    for name in sorted(fractional):
        _check_scale(scales[name], denominators.get(name, {1}))
    # A duration fluent that is also compared or assigned somewhere would need
    # one value for the duration and another for that comparison; the global
    # pass refuses the same overlap for the same reason.
    _, in_numeric_slots = _walk(task, None, set())
    conflicting = fractional & in_numeric_slots
    if conflicting:
        raise NotImplementedError(
            f"Fluent(s) {sorted(conflicting)} are read as a durative action's duration "
            f"and also appear in a numeric condition or effect, and their values are "
            f"fractional; the duration reading has to be put on an integer grid, which "
            f"the other reading would then see. Use a separate fluent for the duration.")
    return scales


def apply_duration_scales(task, scales) -> None:
    """Multiply each duration fluent's initial values by its factor, in place.

    Run after the initial state is complete and after the durative split, so the
    only thing left to read these values is the fact builder.
    """
    fractional = {name for name, scale in (scales or {}).items() if scale > 1}
    if not fractional:
        return
    em = task.environment.expression_manager
    for fluent_exp, value in list(task.explicit_initial_values.items()):
        name = fluent_exp.fluent().name
        if name in fractional and _is_numeric_value(value):
            task.set_initial_value(
                fluent_exp, em.Real(Fraction(value.constant_value()) * scales[name]))
    for fluent, value in list(task.fluents_defaults.items()):
        if fluent.name in fractional and _is_numeric_value(value):
            task.fluents_defaults[fluent] = em.Real(
                Fraction(value.constant_value()) * scales[fluent.name])


def _static_fluent_names(task):
    """Fluent names no action ever writes -- their value is the initial one."""
    written = {effect.fluent.fluent().name
               for action in task.actions
               for effect in all_effects(action)}
    return {fluent.name for fluent in task.fluents if fluent.name not in written}


def static_numeric_folds(task):
    """``{fluent name: divisor}`` for the static fluents an effect reads with a
    fractional coefficient.

    ``(decrease (current_fuel ?sh) (/ (distance ?from ?to) 5))`` is not a
    coefficient the encoding has to state, because ``distance`` is static: the
    grounder can read its value out of the initial state and do the division
    itself, once per parameter binding
    (:func:`~aspplanners.plasp.facts._fold_static_effect_terms`). What it needs
    in return is for that division to be exact, which is this function's other
    half -- :func:`scale_numeric_constants` multiplies the divisors in, on top of
    the factor the task's own values need, so every looked-up value is a multiple
    of its divisor by construction.

    A duration fluent is left out: it sits out the task-wide factor and rides its
    own grid instead (:func:`duration_fluent_scales`), so nothing here can
    promise what its stored value is a multiple of.
    """
    static = _static_fluent_names(task) - _duration_fluent_names(task)
    if not static:
        return {}

    folds = {}
    for action in task.actions:
        for effect in all_effects(action):
            if not (effect.fluent.type.is_int_type() or effect.fluent.type.is_real_type()):
                continue
            try:
                terms, _constant = _linear_form(effect.value)
            except (NotImplementedError, ZeroDivisionError):
                # Not a shape the encoding states at all; the fact builder says
                # so, with a message about the shape rather than about scaling.
                continue
            for expr, coefficient in terms.values():
                denominator = coefficient.denominator
                if denominator == 1 or coefficient == 0:
                    continue
                name = expr.up_expr.fluent().name
                if name in static:
                    folds[name] = _lcm(folds.get(name, 1), denominator)
    return folds


def _lcm(a, b):
    return a * b // gcd(a, b)


def _check_folds_divide(task, folds, scale):
    """Every folded fluent's scaled values are a multiple of its divisor.

    Guaranteed by construction -- `scale` is a multiple of every divisor times
    the factor clearing the values' own denominators -- so this is a check on the
    reasoning, not on the task. It runs anyway because the failure it guards
    against is silent: clingo's ``/`` truncates, so an inexact fold would return
    a wrong plan rather than raise.
    """
    def check(name, value, divisor):
        if divisor is None or not _is_numeric_value(value):
            return
        scaled = Fraction(value.constant_value())
        if (scaled / divisor).denominator != 1:
            raise NotImplementedError(
                f"Fluent {name!r} is read by a numeric effect with a coefficient of "
                f"denominator {divisor}, and its value {scaled} (after scaling by "
                f"{scale}) is not a multiple of it, so folding that value in at "
                f"grounding time would divide inexactly. State the task in units "
                f"that make the coefficient integral.")

    # The initial state comes from the same three places `_walk` scales, and all
    # three can reach a folded fluent: the values a task states, a fluent's own
    # default, and the per-type default `initialize_fluent_defaults` falls back
    # to. The last is keyed by type rather than by fluent, so it is checked
    # against every divisor rather than against its own.
    for fluent_exp, value in task.explicit_initial_values.items():
        check(fluent_exp.fluent().name, value, folds.get(fluent_exp.fluent().name))
    for fluent, value in task.fluents_defaults.items():
        check(fluent.name, value, folds.get(fluent.name))
    for _user_type, value in task.initial_defaults.items():
        for name, divisor in folds.items():
            check(name, value, divisor)


def scale_numeric_constants(task, folds=None) -> int:
    """Multiply `task`'s numeric values by the least factor making them whole.

    Returns that factor, and mutates `task` only when it is greater than 1 -- a
    task whose numbers are already integers is left byte-for-byte alone, which
    is what keeps this pass invisible to everything that worked before it.

    "Whole" covers one thing beyond the values themselves: a fractional
    coefficient on a *static* fluent, whose value the fact builder folds into the
    effect's arithmetic. Those divisors multiply the factor rather than joining
    its least common multiple -- the values have to be a multiple of the divisor
    *after* their own denominators are cleared, not merely integral
    (see :func:`static_numeric_folds`).
    """
    # A fluent read as a durative action's duration keeps the task's own units:
    # the encoding reads its value straight out of the initialState fact and
    # scales it by the time unit itself (see common.temporal.FluentDuration).
    # Nested occurrences count too -- depots' `(/ (distance ?y ?z) (speed ?x))`
    # is not itself a fluent expression, but scaling either fluent inside it
    # would scale the duration the encoding then reads.
    duration_fluents = _duration_fluent_names(task)

    denominators, scaled_fluents = _walk(task, None, duration_fluents)
    scale = 1
    for denominator in denominators:
        scale = scale * denominator // gcd(scale, denominator)

    # The caller passes the plan it will hand the fact builder, so the divisors
    # this factor covers and the ones that get folded cannot drift apart.
    folds = static_numeric_folds(task) if folds is None else folds
    fold_factor = 1
    for divisor in folds.values():
        fold_factor = _lcm(fold_factor, divisor)
    scale *= fold_factor

    if scale == 1:
        return 1

    _check_scale(scale, denominators, folds.values())
    _check_bounded_types(task, scale)
    _check_duration_fluents(scaled_fluents & duration_fluents, scale)
    _walk(task, scale, duration_fluents)
    _check_folds_divide(task, folds, scale)
    return scale


# ---------------------------------------------------------------------------
# Walking the task: one enumeration of the slots, run twice
# ---------------------------------------------------------------------------

def _walk(task, scale, duration_fluents):
    """Collect denominators when `scale` is None, rewrite in place when it is not.

    Both passes visit exactly the same slots, so the factor computed by the first
    cannot turn out to cover less than the second rewrites.
    """
    em = task.environment.expression_manager
    state = _State(em, scale)

    for action in task.actions:
        # Rewritten conditions are collected before anything is cleared: UP is
        # free to normalize what it is given back, so replacing them one at a
        # time would have the list shifting under the loop.
        preconditions = [state.condition(p) for p in getattr(action, 'preconditions', ())]
        if scale is not None and hasattr(action, 'preconditions'):
            action.clear_preconditions()
            for precondition in preconditions:
                action.add_precondition(precondition)

        conditions = [(interval, [state.condition(c) for c in cs])
                      for interval, cs in getattr(action, 'conditions', {}).items()]
        if scale is not None and conditions:
            action.clear_conditions()
            for interval, cs in conditions:
                for condition in cs:
                    action.add_condition(interval, condition)

        for effect in all_effects(action):
            # Only a *numeric* effect's value is a value of the kind being
            # scaled; a boolean or object assignment carries no units.
            if effect.fluent.type.is_int_type() or effect.fluent.type.is_real_type():
                value = state.numeric(effect.value)
                if scale is not None:
                    effect.set_value(value)
            condition = state.condition(effect.condition)
            if scale is not None and condition is not effect.condition:
                effect.set_condition(condition)

    goals = [state.condition(goal) for goal in list(task.goals)]
    if scale is not None:
        task.clear_goals()
        for goal in goals:
            task.add_goal(goal)

    # The initial state comes from three places and all three have to agree: the
    # values a task states, a fluent's own declared default, and the per-type
    # default. The last two matter because `initialize_fluent_defaults` reads
    # them *after* this pass, when it fills in the fluents a task left out --
    # leaving a fractional default unscaled would lay a fractional value down
    # there, past the point anything else could catch it.
    for fluent_exp, value in list(task.explicit_initial_values.items()):
        if fluent_exp.fluent().name in duration_fluents:
            continue
        rewritten = state.numeric(value) if _is_numeric_value(value) else value
        if scale is not None and rewritten is not value:
            task.set_initial_value(fluent_exp, rewritten)

    for fluent, value in list(task.fluents_defaults.items()):
        if fluent.name in duration_fluents or not _is_numeric_value(value):
            continue
        rewritten = state.numeric(value)
        if scale is not None:
            task.fluents_defaults[fluent] = rewritten

    for user_type, value in list(task.initial_defaults.items()):
        if not _is_numeric_value(value):
            continue
        rewritten = state.numeric(value)
        if scale is not None:
            task.initial_defaults[user_type] = rewritten

    return state.denominators, state.fluents


def _is_numeric_value(value):
    return value.is_int_constant() or value.is_real_constant()


# ---------------------------------------------------------------------------
# The expression rewriter
# ---------------------------------------------------------------------------

class _State:
    """Collects denominators (`scale is None`) or rebuilds expressions.

    Running one class in both modes is what keeps the two passes over the task in
    step: there is a single description of which sub-expression is a value.
    """

    def __init__(self, em, scale):
        self.em = em
        self.scale = scale
        self.denominators = set()
        self.fluents = set()          # numeric fluents reached in a scaled slot

    def numeric(self, f):
        """One numeric term: its constants are values and scale, its fluents do not.

        Handles exactly the shapes `facts._linear_form` handles, minus the ones
        whose constants are *coefficients*: a TIMES or a DIV is returned as it
        stands, because ``k * v`` already scales with ``v`` and scaling ``k``
        too would apply the factor twice. Anything else is returned untouched as
        well -- the fact builders raise on those shapes with a message of their
        own, and being stricter here would only replace it with a worse one.
        """
        if f.is_int_constant():
            return f if self.scale is None else self.em.Int(f.constant_value() * self.scale)
        if f.is_real_constant():
            fraction = Fraction(f.constant_value())
            if self.scale is None:
                self.denominators.add(fraction.denominator)
                return f
            return self.em.Real(fraction * self.scale)
        if f.is_fluent_exp():
            self.fluents.add(f.fluent().name)
            return f
        if f.is_plus():
            args = [self.numeric(a) for a in f.args]
            return f if self.scale is None else self.em.Plus(args)
        if f.is_minus():
            args = [self.numeric(a) for a in f.args]
            return f if self.scale is None else self.em.Minus(args[0], args[1])
        return f

    def condition(self, f):
        """One boolean condition: descend to the numeric comparisons inside it."""
        if f.is_and() or f.is_or() or f.is_not() or f.is_implies():
            args = [self.condition(a) for a in f.args]
            if self.scale is None:
                return f
            if f.is_and():
                return self.em.And(args)
            if f.is_or():
                return self.em.Or(args)
            if f.is_not():
                return self.em.Not(args[0])
            return self.em.Implies(args[0], args[1])
        if f.is_exists() or f.is_forall():
            body = self.condition(f.arg(0))
            if self.scale is None:
                return f
            build = self.em.Exists if f.is_exists() else self.em.Forall
            return build(body, *f.variables())
        # `=` is only a numeric comparison when a side is numeric; object and
        # parameter equality take the boolean path and have nothing to scale.
        # This is facts.is_numeric_comparison's test, kept identical on purpose.
        numeric_equality = f.is_equals() and (_is_numeric_fnode(f.arg(0))
                                              or _is_numeric_fnode(f.arg(1)))
        if f.is_lt() or f.is_le() or numeric_equality:
            lhs, rhs = self.numeric(f.arg(0)), self.numeric(f.arg(1))
            if self.scale is None:
                return f
            if f.is_lt():
                return self.em.LT(lhs, rhs)
            if f.is_le():
                return self.em.LE(lhs, rhs)
            return self.em.Equals(lhs, rhs)
        return f


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def _check_scale(scale, denominators, fold_divisors=()):
    """Cap the factor itself. How big the scaled *values* get is checked where
    they are emitted, by `facts._in_clingo_range`, which sees the actual numbers
    rather than a bound on them."""
    if scale <= MAX_NUMERIC_SCALE:
        return
    because = f"the least common multiple of the denominators {sorted(denominators)}"
    if fold_divisors:
        because += (f", times {sorted(set(fold_divisors))} for the fractional "
                    f"coefficients folded into effect arithmetic")
    raise NotImplementedError(
        f"Making this task's numeric values integral needs a scale factor of "
        f"{scale}, {because}; the ASP encoding caps it at {MAX_NUMERIC_SCALE}. "
        f"State the task in units that make its numbers whole.")


def _check_bounded_types(task, scale):
    """A declared bound cannot be scaled with the values it bounds.

    `integer[0, 10]` is a distinct type object, it is the key of the fluent's
    entry in the task, and a constant's own type is the singleton range around
    its value -- so UP rejects `set_initial_value(f, 12)` outright, and there is
    no way to widen the bound without rebuilding the fluent and everything keyed
    by it. Saying so is better than scaling into an unrelated UPTypeError.
    """
    for fluent in task.fluents:
        fluent_type = fluent.type
        if not (fluent_type.is_int_type() or fluent_type.is_real_type()):
            continue
        if fluent_type.lower_bound is None and fluent_type.upper_bound is None:
            continue
        raise NotImplementedError(
            f"Fluent {fluent.name!r} has the bounded numeric type {fluent_type}, and "
            f"this task's numeric values have to be multiplied by {scale} to be "
            f"integral for the ASP encoding; the declared bound cannot be scaled "
            f"with them. Widen the fluent's type, or state the task in units that "
            f"make its numbers whole.")


def _check_duration_fluents(conflicting, scale):
    if not conflicting:
        return
    raise NotImplementedError(
        f"Fluent(s) {sorted(conflicting)} are read as a durative action's duration "
        f"and also appear in a numeric condition or effect that has to be multiplied "
        f"by {scale} to make this task's values integral; their initial values "
        f"cannot be both scaled and left alone. Use a separate fluent for the "
        f"duration, or state the task in units that make its numbers whole.")


__all__ = ['scale_numeric_constants', 'duration_fluent_scales',
           'apply_duration_scales', 'static_numeric_folds', 'MAX_NUMERIC_SCALE']
