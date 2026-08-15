"""Coefficients a static fluent's value only settles at grounding time.

A linear form's coefficients are exact rationals -- until the task multiplies
two fluents. ``zenotravel``'s ``(* (distance ?c1 ?c2) (slow-burn ?a))`` is not
linear as an expression, but ``distance`` and ``slow-burn`` are *static*: no
action of the task writes them, so their values are whatever the initial state
says, and the grounder can read them out of it once per parameter binding. The
product is then a number by the time the rule grounds, and the expression around
it is linear after all.

:class:`Coeff` is what makes that statable. It is a polynomial

    ``r_1 * V_11 * V_12 * ... + r_2 * V_21 * ... + ...``

whose variables are the ASP variables an ``initialState`` body atom binds to a
static fluent's stored value, and whose ``r_i`` are exact rationals. A task that
states an ordinary coefficient produces the one-monomial, no-variable case, so
the arithmetic below is the rational arithmetic it replaces.

Two rationals are folded into the ``r_i`` where they arise, which is what keeps
the polynomial's *value* in the task's own units:

  * the factor a looked-up fluent's stored values carry (see
    :func:`~aspplanners.plasp.rescale.extra_fluent_scales`), so the monomial is
    ``variable / scale`` rather than the stored number;
  * whatever rational the expression multiplied the fluent by.

What a :class:`Coeff` can be rendered as depends on where it sits. A *comparison*
is multiplied through by the least common denominator of both its sides, so its
coefficients always render as whole numbers times the lookups. An *effect* has
no such move -- multiplying it through would change the fluent's value -- so a
leftover denominator has to divide the lookups exactly, which
:meth:`Coeff.render` states as a division and
:func:`~aspplanners.plasp.rescale.extra_fluent_scales` is what arranges.
"""

from fractions import Fraction
from math import gcd
from typing import Dict, Iterable, Optional, Tuple

__all__ = ['Coeff', 'NumericContext', 'InexactCoefficient']


def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


class InexactCoefficient(Exception):
    """A :class:`Coeff` cannot be rendered as an exact clingo term.

    Raised by :meth:`Coeff.render` alone, and turned into a task-level refusal
    by the caller, which is the one that knows *which* effect could not be
    stated. Deliberately not a ``NotImplementedError``: the fact builder catches
    it to re-raise with the effect's own message.
    """


class Coeff:
    """A linear form's coefficient or additive constant.

    Immutable. ``monomials`` maps a sorted tuple of ASP variable names to the
    rational they multiply; the empty tuple is the plain-rational part, so
    ``Coeff.rational(x).monomials == {(): x}`` and the arithmetic below
    degenerates to ``Fraction`` arithmetic on a task that multiplies no two
    fluents. A zero rational is dropped, so ``is_zero`` is a dictionary test.
    """

    __slots__ = ('monomials',)

    def __init__(self, monomials: Optional[Dict[Tuple[str, ...], Fraction]] = None):
        self.monomials = {variables: value
                          for variables, value in (monomials or {}).items()
                          if value != 0}

    # -- construction ------------------------------------------------------

    @classmethod
    def rational(cls, value) -> 'Coeff':
        return cls({(): Fraction(value)})

    @classmethod
    def lookup(cls, variable: str, scale=1) -> 'Coeff':
        """The value the grounder binds `variable` to, in the task's own units.

        `scale` is the factor that stored value carries, and is divided straight
        back out here so that everything downstream can treat the monomial as
        the fluent's actual value.
        """
        return cls({(variable,): Fraction(1, 1) / Fraction(scale)})

    ZERO: 'Coeff'
    ONE: 'Coeff'

    # -- arithmetic --------------------------------------------------------

    def __add__(self, other: 'Coeff') -> 'Coeff':
        monomials = dict(self.monomials)
        for variables, value in other.monomials.items():
            monomials[variables] = monomials.get(variables, Fraction(0)) + value
        return Coeff(monomials)

    def __mul__(self, other: 'Coeff') -> 'Coeff':
        monomials: Dict[Tuple[str, ...], Fraction] = {}
        for left_vars, left in self.monomials.items():
            for right_vars, right in other.monomials.items():
                key = tuple(sorted(left_vars + right_vars))
                monomials[key] = monomials.get(key, Fraction(0)) + left * right
        return Coeff(monomials)

    def __neg__(self) -> 'Coeff':
        return Coeff({variables: -value for variables, value in self.monomials.items()})

    def scaled(self, factor) -> 'Coeff':
        """This coefficient times the rational `factor`."""
        factor = Fraction(factor)
        if factor == 0:
            return Coeff.ZERO
        return Coeff({variables: value * factor for variables, value in self.monomials.items()})

    # -- inspection --------------------------------------------------------

    def is_zero(self) -> bool:
        return not self.monomials

    @property
    def is_rational(self) -> bool:
        """True when no lookup is involved, so this is an ordinary number."""
        return set(self.monomials) <= {()}

    @property
    def value(self) -> Fraction:
        """The rational this stands for; only defined when :attr:`is_rational`."""
        assert self.is_rational, f"{self!r} is not a plain rational"
        return self.monomials.get((), Fraction(0))

    @property
    def denominator(self) -> int:
        """The least multiplier making every monomial's rational whole."""
        denominator = 1
        for value in self.monomials.values():
            denominator = _lcm(denominator, value.denominator)
        return denominator

    @property
    def variables(self) -> Tuple[str, ...]:
        """Every lookup variable this coefficient reads, in a stable order."""
        return tuple(sorted({v for variables in self.monomials for v in variables}))

    def __eq__(self, other):
        if isinstance(other, Coeff):
            return self.monomials == other.monomials
        return NotImplemented

    def __hash__(self):
        return hash(tuple(sorted(self.monomials.items())))

    def __repr__(self):
        return f"Coeff({self.monomials!r})"

    # -- rendering ---------------------------------------------------------

    def render(self, multiplier=1, exact_divisors=None) -> str:
        """This coefficient times `multiplier`, as one clingo term.

        The whole-number case is the term itself, so a task that folds nothing
        emits exactly the integers it emitted before. A monomial that still
        carries a denominator after `multiplier` is applied is rendered as a
        division, which clingo *truncates* -- so it is only allowed where the
        caller can promise the division is exact, by naming the variables whose
        bound values are known to be multiples of their divisor in
        `exact_divisors` (``{variable: divisor}``). Anything else raises
        :class:`InexactCoefficient`, because a truncated coefficient would
        return a wrong plan rather than fail.
        """
        multiplier = Fraction(multiplier)
        exact_divisors = exact_divisors or {}
        parts = []
        for variables, value in sorted(self.monomials.items()):
            parts.append(self._render_monomial(variables, value * multiplier, exact_divisors))
        if not parts:
            return '0'
        if len(parts) == 1:
            return parts[0]
        term = parts[0]
        for part in parts[1:]:
            term += f"-{part[1:]}" if part.startswith('-') else f"+{part}"
        return f"({term})"

    @staticmethod
    def _render_monomial(variables, value: Fraction, exact_divisors) -> str:
        # Divide by whatever of the denominator the caller vouched for, one
        # variable at a time: `(A/5)*(B/2)` rather than `(A*B)/10`, so each
        # division is the one that was checked and no intermediate product is
        # built that the checked values would not bound.
        factors, remaining = [], value.denominator
        for variable in variables:
            divisor = exact_divisors.get(variable, 1)
            divisor = gcd(divisor, remaining)
            if divisor > 1:
                remaining //= divisor
                factors.append(f"({variable}/{divisor})")
            else:
                factors.append(variable)
        if remaining != 1:
            raise InexactCoefficient(
                f"{value} * {' * '.join(variables) or '1'} is not a whole number and "
                f"nothing promises the division by {remaining} is exact")
        numerator = value.numerator
        if not factors:
            return str(numerator)
        if numerator == 1:
            return '*'.join(factors)
        if numerator == -1:
            return '-' + '*'.join(factors)
        return f"{numerator}*" + '*'.join(factors)


Coeff.ZERO = Coeff()
Coeff.ONE = Coeff.rational(1)


class NumericContext:
    """The task-wide numeric facts every expression builder has to agree on.

    Threaded through :func:`~aspplanners.plasp.facts.parseexpr` and the effect
    builders rather than read off a module global, because it is per *owner*:
    the lookups it hands out have to end up in the body of the rules that name
    them, and one owner (an action, a goal, a durative action) is exactly the
    scope over which a lookup variable's name has to be unique.

    Three per-fluent numbers live here, and the difference between the first two
    is the whole reason this class exists rather than a plain dictionary:

      * ``value_scales`` -- the *extra* factor a fluent's values carry beyond
        the task-wide rescale. 1 for almost every fluent, and more only where an
        effect on it would otherwise have a fractional coefficient (see
        :func:`~aspplanners.plasp.rescale.extra_fluent_scales`). A linear form
        is stated in the units the task-wide rescale left it in, so a
        coefficient read off such a fluent is divided by this and an effect's
        delta multiplied by its target's.
      * ``stored_scales`` -- the *whole* factor, task-wide rescale included. A
        folded lookup binds the number sitting in the initial state, and a
        product of two of those would carry the task-wide factor twice over, so
        a fold divides by this one and comes back in the task's own units.
      * ``value_gcds`` -- what every one of a fluent's stored values is a
        multiple of. That is the promise :meth:`Coeff.render` needs to state a
        leftover denominator as a division; without it an effect reading a
        static fluent with a fractional coefficient is refused rather than
        rounded.

    ``static_fluents`` are the numeric fluents no action writes. Their values
    are whatever the initial state says, so an expression reading only those is
    a number the grounder can work out -- which is what :meth:`fold` turns into
    a :class:`Coeff`.
    """

    #: Prefix of the ASP variable a folded static lookup binds its value to.
    PREFIX = 'RAWSTAT'

    def __init__(self, value_scales=None, stored_scales=None, static_fluents=(),
                 value_gcds=None):
        self.value_scales = dict(value_scales or {})
        self.stored_scales = dict(stored_scales or {})
        self.static_fluents = frozenset(static_fluents)
        self.value_gcds = dict(value_gcds or {})
        self._lookups: Dict[str, Tuple[str, Tuple[str, ...]]] = {}
        self._by_key: Dict[str, str] = {}
        self._fluent_of: Dict[str, str] = {}
        self.exact_divisors: Dict[str, int] = {}

    # -- scales ------------------------------------------------------------

    def scale(self, fluent_name: str) -> Fraction:
        """The extra factor this fluent's stored values carry."""
        return Fraction(self.value_scales.get(fluent_name, 1))

    def stored_scale(self, fluent_name: str) -> Fraction:
        """The whole factor between this fluent's stored and original values."""
        return Fraction(self.stored_scales.get(fluent_name,
                                               self.value_scales.get(fluent_name, 1)))

    def is_static(self, fluent_name: str) -> bool:
        return fluent_name in self.static_fluents

    # -- static lookups ----------------------------------------------------

    def fold(self, key: str, term: str, fluent_name: str, bindings=()) -> Coeff:
        """A static fluent occurrence as a :class:`Coeff` over its lookup.

        `key` identifies the occurrence -- two occurrences of the same fluent
        expression share one variable, so a rule that reads it twice carries one
        ``initialState`` atom. `term` is the ASP term of the fluent
        (``variable(("distance",C1,C2))``) and `bindings` the ``has(...)`` atoms
        its quantified arguments need.
        """
        variable = self._by_key.get(key)
        if variable is None:
            variable = f"{self.PREFIX}{len(self._lookups)}"
            self._by_key[key] = variable
            self._fluent_of[variable] = fluent_name
            self._lookups[variable] = (
                f"initialState({term}, value({term}, {variable}))", tuple(bindings))
            divisor = int(self.value_gcds.get(fluent_name, 1) or 1)
            if divisor > 1:
                self.exact_divisors[variable] = divisor
        return Coeff.lookup(variable, self.stored_scale(fluent_name))

    def fluent_of(self, variable: str) -> str:
        """The fluent whose value `variable` was handed out for."""
        return self._fluent_of[variable]

    def body(self, *coefficients: Coeff) -> Tuple[str, ...]:
        """The body atoms the rules naming these coefficients have to carry.

        The ``initialState`` lookup of every variable they read, and the
        ``has(...)`` atoms that lookup's own arguments need -- a static fluent
        under a ``forall`` is looked up once per binding, so its quantified
        arguments have to be ranged here just as an ordinary condition's are.
        """
        atoms = []
        for variable in sorted({v for c in coefficients for v in c.variables}):
            lookup, bindings = self._lookups[variable]
            atoms.extend(bindings)
            atoms.append(lookup)
        seen, unique = set(), []
        for atom in atoms:
            if atom not in seen:
                seen.add(atom)
                unique.append(atom)
        return tuple(unique)

    def divisors_for(self, coefficients: Iterable[Coeff]) -> Dict[str, int]:
        """The ``{variable: divisor}`` map :meth:`Coeff.render` needs for these."""
        wanted = {v for c in coefficients for v in c.variables}
        return {variable: divisor for variable, divisor in self.exact_divisors.items()
                if variable in wanted}


#: The context a builder gets when nobody supplies one: no extra scales and no
#: static fluent foldable, which is the behaviour every task had before folding
#: existed -- a product of two fluents is refused rather than resolved.
NO_FOLDING = NumericContext()
