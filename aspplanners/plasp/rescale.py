"""Put a task's numeric *values* on an integer grid.

clingo terms are integers, so a task whose fluents take fractional values cannot
be encoded as it stands. The fix is to multiply those values by a factor and
divide it back out wherever the encoding reads them -- and the factor is *per
fluent*, because the two things that force one are per fluent as well:

* a fluent whose stated values are fractional (`(= (x b) 3.5)`) needs a factor
  clearing their denominators;
* a fluent an effect moves by a fractional multiple of another (`(increase (x ?b)
  (* 1.5 (v ?b)))`) needs one clearing *that*: with ``x`` stored doubled the
  delta is ``3 * v``, a whole coefficient, where no uniform factor could have
  made it one -- scaling every value alike moves ``v`` too and the 3/2 survives.

The two are one constraint system, since a fluent's factor appears in the
coefficient of every effect that *reads* it as well as in every effect on it,
and :func:`numeric_value_scales` solves it by raising factors until nothing is
fractional (or the cap says the task is stated in the wrong units).

What the factors mean downstream, in one sentence each:

* **Values** are stored multiplied: ``initialState`` carries ``S_f * v``, and so
  does every ``numval`` the encoding derives from it.
* **Coefficients** are divided: a linear form reads ``V_f`` as ``V'_f / S_f``, so
  the form is in the task's own units whatever the storage.
* **Effects** are multiplied back: an effect on ``t`` states ``S_t`` times its
  form, which is the delta in the units ``t`` is stored in.
* **Comparisons** need neither, because both sides are multiplied through by the
  least common denominator of their coefficients -- a positive factor, so the
  comparison is preserved exactly. That is also why a *comparison's* constants
  put no constraint on any factor.

Nothing else in the task is rewritten: the expressions a task states are read as
they stand, and only its initial values are mutated (by
:func:`apply_value_scales`, once the initial state is complete). A task whose
numbers are already whole gets a factor of 1 everywhere and is left byte-for-byte
alone.

**Durations** sit out entirely. They live on their own integer grid, computed by
:func:`~aspplanners.common.temporal.time_unit`, and the planner turns a happening
back into an absolute time by multiplying by that unit -- so scaling a duration
fluent with the rest would make every reported plan time wrong by that factor.
Sitting out is not the same as staying fractional, and satellite states
`(= (slew_time ?a ?b) 5.9)`, so each duration fluent gets a factor of its own
(:func:`duration_fluent_scales`) which the ``durationValue`` arithmetic divides
straight back out.

This pass belongs to the PLASP backend rather than to the shared front-end: the
shapes it walks have to track those of :mod:`aspplanners.plasp.facts` exactly.
"""

from fractions import Fraction
from math import gcd

from aspplanners.common.temporal import all_effects, durative_actions
from aspplanners.plasp.facts import _is_numeric_fnode, _linear_form
from aspplanners.plasp.numeric_terms import NumericContext

# Denominators past this are not a task stated in awkward units, they are a task
# stated in the wrong ones; scaling by such a factor would make the integers
# meaningless long before it made them wrong.
MAX_NUMERIC_SCALE = 10 ** 6

# How many times the constraint system below is swept before it is called
# divergent. Each sweep can only raise a factor, and a factor raised by an
# effect that reads a fluent it also writes (`v += v/2 + w/3`) can raise it
# again, so a task can need several -- but a cycle of effects that each demand a
# finer grid than the last (`x += y/2`, `y += x/2`) diverges, and is stopped by
# the cap rather than by this.
_MAX_SWEEPS = 64


def _lcm(a, b):
    return a * b // gcd(a, b)


def _fraction_gcd(values):
    """The largest rational every value in `values` is an integer multiple of.

    ``None`` when there is nothing to divide -- an empty set, or one holding only
    zero, where every rational qualifies and no divisor can be promised.
    """
    numerator, denominator = 0, 1
    for value in values:
        value = Fraction(value)
        numerator = gcd(numerator, value.numerator)
        denominator = _lcm(denominator, value.denominator)
    if numerator == 0:
        return None
    return Fraction(numerator, denominator)


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


def _is_numeric_value(value):
    return value.is_int_constant() or value.is_real_constant()


def _numeric_fluent_names(task):
    return {fluent.name for fluent in task.fluents
            if fluent.type.is_int_type() or fluent.type.is_real_type()}


def _static_fluent_names(task):
    """Fluent names no action ever writes -- their value is the initial one."""
    written = {effect.fluent.fluent().name
               for action in task.actions
               for effect in all_effects(action)}
    return {fluent.name for fluent in task.fluents if fluent.name not in written}


def _stated_values(task):
    """``{fluent name: [Fraction, ...]}`` -- every value the task lays down.

    The three places an initial value can come from, which is the same set
    :func:`apply_value_scales` multiplies and
    :func:`~aspplanners.common.compilation.initialize_fluent_defaults` fills the
    rest from: the values a task states, a fluent's own declared default, and
    the per-type default. The last is keyed by type rather than by fluent, so it
    counts for every numeric fluent of that type.
    """
    values = {}
    for fluent_exp, value in task.explicit_initial_values.items():
        if _is_numeric_value(value):
            values.setdefault(fluent_exp.fluent().name, []).append(
                Fraction(value.constant_value()))
    for fluent, value in task.fluents_defaults.items():
        if _is_numeric_value(value):
            values.setdefault(fluent.name, []).append(Fraction(value.constant_value()))
    for user_type, value in task.initial_defaults.items():
        if not _is_numeric_value(value):
            continue
        for fluent in task.fluents:
            if fluent.type == user_type:
                values.setdefault(fluent.name, []).append(Fraction(value.constant_value()))
    return values


# ---------------------------------------------------------------------------
# The constraint system
# ---------------------------------------------------------------------------

class _EffectShape:
    """One numeric effect, reduced to what the factors have to satisfy.

    `terms` maps each fluent the effect *reads* to the monomials of its
    coefficient, and `constant` holds the monomials of the additive rest. A
    monomial is ``(rational, [static fluent names])``: the plain case has no
    names, and the rest are the products the grounder resolves out of the
    initial state (see :func:`~aspplanners.plasp.facts._fold_statics`) -- their
    value is only known per binding, so what the factors can be held to is that
    the *greatest common divisor* of the values each one can take comes out
    whole.

    Every rational here is in the task's own units. The shapes are read with a
    context whose factors are all 1 for exactly that reason; the factors being
    solved for are applied to them here rather than baked into them.
    """

    __slots__ = ('target', 'terms', 'constant')

    def __init__(self, target, terms, constant):
        self.target = target
        self.terms = terms
        self.constant = constant


def _monomials(coefficient, context):
    """A :class:`~aspplanners.plasp.numeric_terms.Coeff` as ``(rational, names)`` pairs."""
    return [(value, [context.fluent_of(v) for v in variables])
            for variables, value in coefficient.monomials.items()]


def _effect_shapes(task, statics):
    """Every numeric effect of `task` as an :class:`_EffectShape`.

    An effect whose value is a shape the encoding cannot state at all is skipped
    rather than raised on: the fact builder refuses it with a message about that
    shape, which is more use than one about scaling.
    """
    shapes = []
    for action in task.actions:
        for effect in all_effects(action):
            if not (effect.fluent.type.is_int_type() or effect.fluent.type.is_real_type()):
                continue
            context = NumericContext(static_fluents=statics)
            try:
                terms, constant = _linear_form(effect.value, context)
            except (NotImplementedError, ZeroDivisionError):
                continue
            read = {}
            for expr, coefficient in terms.values():
                if coefficient.is_zero():
                    continue
                name = expr.up_expr.fluent().name
                read.setdefault(name, []).extend(_monomials(coefficient, context))
            shapes.append(_EffectShape(effect.fluent.fluent().name, read,
                                       _monomials(constant, context)))
    return shapes


def numeric_value_scales(task):
    """``{fluent name: factor}`` putting every numeric value on an integer grid.

    The factor is 1 for every fluent of a task whose numbers are already whole,
    which is what keeps this pass invisible to the tasks that do not need it.
    Computed, not applied -- see :func:`apply_value_scales`.
    """
    numeric = _numeric_fluent_names(task)
    if not numeric:
        return {}
    duration_fluents = _duration_fluent_names(task)
    values = _stated_values(task)
    statics = _static_fluent_names(task)

    # A fluent's own values are the floor: whatever else happens, `S_f * v` has
    # to be an integer for every value the task lays down.
    scales = {name: 1 for name in numeric}
    for name in numeric:
        if name in duration_fluents:
            continue          # its own grid, see duration_fluent_scales
        for value in values.get(name, ()):
            scales[name] = _lcm(scales[name], Fraction(value).denominator)

    gcds = {name: _fraction_gcd(values.get(name, ())) for name in numeric}
    shapes = _effect_shapes(task, statics)

    def monomial_denominator(target, monomial, divide_by=1):
        """What ``target * monomial / divide_by`` is still missing to be whole.

        The looked-up values of a folded product are only known to be multiples
        of their own greatest common divisor, so that is what the monomial can
        promise; a fluent every one of whose values is zero promises nothing,
        and its effect is left to the fact builder to refuse (0 divides by
        everything, so picking a factor off it would be picking one at random).
        """
        coefficient, names = monomial
        product = Fraction(target) * coefficient / divide_by
        for name in names:
            divisor = gcds.get(name)
            if divisor is None:
                return None
            product *= divisor
        return product.denominator

    def needed(shape):
        """The factor `shape`'s target is still missing, or 1 when it is not."""
        target = scales.get(shape.target, 1)
        need = 1
        for name, monomials in shape.terms.items():
            for monomial in monomials:
                denominator = monomial_denominator(target, monomial, scales.get(name, 1))
                if denominator is None:
                    return None
                need = _lcm(need, denominator)
        for monomial in shape.constant:
            denominator = monomial_denominator(target, monomial)
            if denominator is None:
                return None
            need = _lcm(need, denominator)
        return need

    for _sweep in range(_MAX_SWEEPS):
        changed = False
        for shape in shapes:
            if shape.target not in scales:
                continue
            need = needed(shape)
            if need is None or need == 1:
                continue
            scales[shape.target] *= need
            # Raising the target's factor multiplies every one of this shape's
            # coefficients by the same amount, so one pass settles it -- unless
            # the factor cancels, which is what an effect reading its own target
            # does. Left to the sweep, that doubles forever and surfaces as a
            # cap complaint about a factor of 2^64; caught here it says what the
            # task actually asked for.
            if needed(shape) not in (None, 1):
                _refuse_self_reference(shape)
            changed = True
        if not changed:
            break

    scales = {name: scale for name, scale in scales.items() if scale > 1}
    if scales:
        _check_scale(scales, values)
        _check_bounded_types(task, scales)
        _check_duration_fluents(set(scales) & duration_fluents, scales)
    return scales


def apply_value_scales(task, scales) -> None:
    """Multiply each fluent's initial values by its factor, in place.

    Run after the initial state is complete, so that the values
    :func:`~aspplanners.common.compilation.initialize_fluent_defaults` laid down
    are covered too and the only thing left to read any of them is the fact
    builder.
    """
    if not scales:
        return
    em = task.environment.expression_manager
    for fluent_exp, value in list(task.explicit_initial_values.items()):
        scale = scales.get(fluent_exp.fluent().name)
        if scale and scale > 1 and _is_numeric_value(value):
            task.set_initial_value(
                fluent_exp, em.Real(Fraction(value.constant_value()) * scale))
    for fluent, value in list(task.fluents_defaults.items()):
        scale = scales.get(fluent.name)
        if scale and scale > 1 and _is_numeric_value(value):
            task.fluents_defaults[fluent] = em.Real(Fraction(value.constant_value()) * scale)


def stored_value_gcds(task, scales):
    """``{fluent name: divisor}`` -- what every stored value of a fluent divides by.

    The promise the fact builder needs to state a looked-up value as a division
    rather than refuse it: a static fluent read as a coefficient carries its own
    storage factor, and dividing that back out is only exact because every value
    it can take is a multiple of this (see
    :meth:`~aspplanners.plasp.numeric_terms.Coeff.render`). Fluents whose values
    are all zero are left out: everything divides zero, and promising an
    arbitrary divisor on that basis would be a promise about a task that does
    not read the fluent anyway.
    """
    divisors = {}
    for name, values in _stated_values(task).items():
        scale = scales.get(name, 1)
        divisor = _fraction_gcd(Fraction(value) * scale for value in values)
        if divisor is not None and divisor.denominator == 1 and divisor.numerator > 1:
            divisors[name] = divisor.numerator
    return divisors


def scale_numeric_constants(task, folds=None) -> int:
    """The single factor that describes this task's scaling, for reporting.

    The least common multiple of :func:`numeric_value_scales` -- 1 exactly when
    nothing had to be scaled. Kept because it is what
    ``ASPEncodingResult.numeric_scale`` reports and what callers written against
    the earlier, task-wide factor ask for; it computes and checks, and mutates
    nothing (`folds` is accepted and ignored, for the same reason).
    """
    scale = 1
    for factor in numeric_value_scales(task).values():
        scale = _lcm(scale, factor)
    return scale


# ---------------------------------------------------------------------------
# Durations: their own grid
# ---------------------------------------------------------------------------

def _numeric_slot_fluents(task):
    """Fluent names read by a numeric comparison or a numeric effect.

    The slots a scaled value would be visible in, which is what makes a fluent
    read as a duration *and* compared somewhere a conflict: the duration grid
    and the value grid cannot both have it.
    """
    found = set()

    def walk_numeric(f):
        found.update(_fluent_names(f))

    def walk_condition(f):
        if f.is_and() or f.is_or() or f.is_not() or f.is_implies():
            for arg in f.args:
                walk_condition(arg)
            return
        if f.is_exists() or f.is_forall():
            walk_condition(f.arg(0))
            return
        numeric_equality = f.is_equals() and (_is_numeric_fnode(f.arg(0))
                                              or _is_numeric_fnode(f.arg(1)))
        if f.is_lt() or f.is_le() or numeric_equality:
            walk_numeric(f.arg(0))
            walk_numeric(f.arg(1))

    for action in task.actions:
        for precondition in getattr(action, 'preconditions', ()):
            walk_condition(precondition)
        for _interval, conditions in getattr(action, 'conditions', {}).items():
            for condition in conditions:
                walk_condition(condition)
        for effect in all_effects(action):
            if effect.fluent.type.is_int_type() or effect.fluent.type.is_real_type():
                found.add(effect.fluent.fluent().name)
                walk_numeric(effect.value)
            walk_condition(effect.condition)
    for goal in task.goals:
        walk_condition(goal)
    return found


def duration_fluent_scales(task):
    """``{fluent name: factor}`` putting each duration fluent's values on an
    integer grid -- computed, not applied (see :func:`apply_duration_scales`).

    Duration fluents sit out :func:`numeric_value_scales` because scaling them
    would make every reported plan time wrong by that factor. But their values
    still have to reach clingo as integer terms, and satellite states
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
        _check_one_scale(name, scales[name], denominators.get(name, {1}))
    # A duration fluent that is also compared or assigned somewhere would need
    # one value for the duration and another for that comparison; the value pass
    # refuses the same overlap for the same reason.
    conflicting = fractional & _numeric_slot_fluents(task)
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


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def _refuse_self_reference(shape):
    """An effect on `t` whose factor cancels: the wall a finer grid never reaches.

    The coefficient the encoding would state for a term reading the target
    itself is ``S_t * c / S_t`` -- the target's own factor cancels, so no
    resolution makes it whole. Sailing-wind's ``(assign (v ?b) (+ ... (* (r ?b)
    (v ?b))))`` with ``r = 0.9`` is the shape, and it is a real boundary of an
    integer encoding rather than a gap in the scaling: each application would
    need a tenth of the resolution of the last, without bound.
    """
    def term(value, names):
        product = ' * '.join(sorted(names))
        if not names:
            return str(value)
        return product if value == 1 else f"{value} * {product}"

    coefficients = ' + '.join(term(value, names)
                              for value, names in shape.terms.get(shape.target, ()))
    raise NotImplementedError(
        f"The numeric effect on {shape.target!r} reads {shape.target!r} itself with the "
        f"fractional coefficient {coefficients or 'it states'}, which no scaling of the "
        f"task can make whole: scaling {shape.target!r}'s values scales the "
        f"coefficient's own input by the same factor, so it cancels. The ASP encoding "
        f"cannot state it; state the task in units that make the coefficient integral.")


def _check_scale(scales, values):
    """Cap the factors themselves. How big the scaled *values* get is checked
    where they are emitted, by `facts._in_clingo_range`, which sees the actual
    numbers rather than a bound on them."""
    for name in sorted(scales):
        if scales[name] > MAX_NUMERIC_SCALE:
            _check_one_scale(name, scales[name],
                             sorted({Fraction(v).denominator
                                     for v in values.get(name, ())} - {1}))


def _check_one_scale(name, scale, denominators):
    if scale <= MAX_NUMERIC_SCALE:
        return
    because = (f"its values have the denominators {sorted(denominators)}"
               if denominators else
               "the effects on it read other fluents with coefficients that fine")
    raise NotImplementedError(
        f"Storing the values of {name!r} finely enough for the ASP encoding needs a "
        f"factor of {scale} ({because}); the encoding caps it at {MAX_NUMERIC_SCALE}, "
        f"past which the integers are meaningless long before they are wrong. State "
        f"the task in units that make its numbers whole.")


def _check_bounded_types(task, scales):
    """A declared bound cannot be scaled with the values it bounds.

    `integer[0, 10]` is a distinct type object, it is the key of the fluent's
    entry in the task, and a constant's own type is the singleton range around
    its value -- so UP rejects `set_initial_value(f, 12)` outright, and there is
    no way to widen the bound without rebuilding the fluent and everything keyed
    by it. Saying so is better than scaling into an unrelated UPTypeError.

    Only the fluents actually being scaled are refused: a bounded fluent whose
    own values (and every effect on it) are already whole is left alone.
    """
    for fluent in task.fluents:
        fluent_type = fluent.type
        if not (fluent_type.is_int_type() or fluent_type.is_real_type()):
            continue
        if fluent_type.lower_bound is None and fluent_type.upper_bound is None:
            continue
        scale = scales.get(fluent.name)
        if not scale or scale == 1:
            continue
        raise NotImplementedError(
            f"Fluent {fluent.name!r} has the bounded numeric type {fluent_type}, and "
            f"this task's numeric values have to be multiplied by {scale} to be "
            f"integral for the ASP encoding; the declared bound cannot be scaled "
            f"with them. Widen the fluent's type, or state the task in units that "
            f"make its numbers whole.")


def _check_duration_fluents(conflicting, scales):
    if not conflicting:
        return
    raise NotImplementedError(
        f"Fluent(s) {sorted(conflicting)} are read as a durative action's duration "
        f"and also appear in a numeric condition or effect that has to be multiplied "
        f"by {sorted(scales[name] for name in conflicting)} to make this task's values "
        f"integral; their initial values cannot be both scaled and left alone. Use a "
        f"separate fluent for the duration, or state the task in units that make its "
        f"numbers whole.")


__all__ = ['numeric_value_scales', 'apply_value_scales', 'stored_value_gcds',
           'scale_numeric_constants', 'duration_fluent_scales', 'apply_duration_scales',
           'MAX_NUMERIC_SCALE']
