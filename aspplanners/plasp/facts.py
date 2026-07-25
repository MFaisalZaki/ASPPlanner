"""PLASP fact builders: render a compiled UP problem into the ASP fact
vocabulary (``variable``/``action``/``precondition``/``numGoal``/...) consumed
by the encodings in ``aspplanners/plasp/encodings``.

This is a recreation of the PLASP tool's translation. Every builder is an
:class:`~aspplanners.lp_io.ASPTerm`, so its identity is its rendered ASP text
and sets of facts deduplicate accordingly.
"""

from unified_planning.shortcuts import FNode, EffectKind

from aspplanners.lp_io import ASPTerm


def asp_name(name):
    """ASP-side rendering of a UP name: PDDL hyphens become underscores.

    All names are emitted inside quoted clingo strings (where hyphens would
    be legal), but the underscore vocabulary is the contract downstream .lp
    encodings match on, and action parameters become ASP *variables* where
    hyphens are not legal. The planner maps model atoms back through this
    same function; the encoder rejects tasks where the mapping collides.
    """
    return name.replace('-', '_')

def _is_numeric_fnode(f):
    """Is this expression integer/real-valued (fluent, constant, or arithmetic)?"""
    return (f.is_int_constant() or f.is_real_constant() or f.is_plus() or f.is_minus()
            or (f.is_fluent_exp() and not f.type.is_bool_type()))

def flatten_atoms(parsed):
    """`parseexpr`'s result as a flat list; nested lists are conjunctions."""
    atoms = parsed if isinstance(parsed, list) else [parsed]
    while any(isinstance(a, list) for a in atoms):
        atoms = [e for a in atoms for e in (a if isinstance(a, list) else [a])]
    return atoms


class ASPDisjunction:
    """A disjunctive condition: at least one of its disjuncts has to hold.

    Not an :class:`~aspplanners.lp_io.ASPTerm`, because it has no rendering of
    its own -- it is emitted as ``orGroup``/``orDisjunct`` facts by whoever owns
    the condition, which is the only place the action or goal it belongs to is
    known (see :class:`ASPAction`).

    Each disjunct is ``(id_term, atoms, bindings)``: `atoms` all have to hold for
    that disjunct to, and `bindings` are the ``has(_, type(...))`` atoms its rule
    body needs. An ``or`` numbers its disjuncts 0, 1, ...; an ``exists`` is the
    same structure with the disjuncts indexed by the quantified variable's
    binding, so the grounder enumerates them instead of a compiler.
    """

    def __init__(self, disjuncts):
        self.disjuncts = disjuncts

    @property
    def negated_fluent_names(self):
        return {atom.up_expr._content.payload.name
                for _id, atoms, _bindings in self.disjuncts for atom in atoms
                if atom.value == 'false'}


def _quantified_bindings(f):
    """``(disjunct id term, has(...) bindings)`` for a quantifier's variables."""
    variables = list(f.variables())
    names = [QUANTIFIED_PREFIX + asp_name(v.name).upper() for v in variables]
    bindings = [f'has({name}, {str(ASPType(v.type))})' for name, v in zip(names, variables)]
    return (names[0] if len(names) == 1 else f"({','.join(names)})"), bindings


def _as_disjuncts(f, t, index):
    """The disjuncts one argument of a disjunction contributes.

    A conjunctive argument is one disjunct numbered `index`; an ``exists`` is a
    family of them indexed by its variables' bindings, which is what lets an
    `or` and an `exists` share a vocabulary.
    """
    quantified = f.is_exists() if t != 'false' else f.is_forall()
    if quantified:
        disjunct_id, bindings = _quantified_bindings(f)
        atoms = flatten_atoms(parseexpr(f.arg(0), t))
        _reject_nested(f, atoms)
        return [(f"({index},{disjunct_id})", atoms, bindings)]
    atoms = flatten_atoms(parseexpr(f, t))
    _reject_nested(f, atoms)
    return [(str(index), atoms, sorted({b for a in atoms for b in a.bindings}))]


def _reject_nested(f, atoms):
    if any(isinstance(a, ASPDisjunction) for a in atoms):
        raise NotImplementedError(
            f"Disjunct {f} is itself disjunctive; the encoding states a disjunction as a "
            "flat set of conjunctive disjuncts, so nesting one inside another needs "
            "up_disjunctive_conditions_remover in the compilation pipeline.")


def parseexpr(f, t=None):
    """!
    given a FNode from UP representing a fluent and a possible timestep return a
    string representation of it.
    """
    assert isinstance(f, FNode), f"Expected a FNode, got {type(f)}"
    if  f.is_fluent_exp(): # for fluents
        return ASPExpr(f, 'true' if t is None else t)
    if f.is_bool_constant():
        return ASPExpr(f, str(f).lower())
    if f.is_not():
        return parseexpr(f.args[0], 'false')
    negated = t == 'false'
    # A conjunction is what precondition/goal facts *are*, so it just flattens --
    # and so does a `forall`, which is a conjunction over the universe, emitted
    # with its variable left free for the grounder to range (see _arg_term).
    # De Morgan swaps which is which under negation.
    if f.is_and() if not negated else f.is_or():
        return [parseexpr(arg, t) for arg in f.args]
    if f.is_forall() if not negated else f.is_exists():
        return parseexpr(f.arg(0), t)
    # A disjunction is not conjunctive, so it becomes an ASPDisjunction that the
    # owning action or goal emits as its own orGroup/orDisjunct facts. An
    # `exists` is one too, with the disjuncts indexed by its variables.
    if f.is_or() or f.is_and():
        return ASPDisjunction([d for i, arg in enumerate(f.args)
                               for d in _as_disjuncts(arg, t, i)])
    if f.is_exists() or f.is_forall():
        disjunct_id, bindings = _quantified_bindings(f)
        atoms = flatten_atoms(parseexpr(f.arg(0), t))
        _reject_nested(f, atoms)
        return ASPDisjunction([(disjunct_id, atoms, bindings)])
    if f.is_lt() or f.is_le():
        cmp = ASPNumComparison(f, 'lt' if f.is_lt() else 'le')
        return cmp.negated() if t == 'false' else cmp
    if f.is_equals() and (_is_numeric_fnode(f.args[0]) or _is_numeric_fnode(f.args[1])):
        if t == 'false':
            raise NotImplementedError(
                "Negated numeric equality is not supported in the ASP encoding."
            )
        return ASPNumComparison(f, 'eq')
    if f.is_equals():
        lhs, rhs = f.args[0], f.args[1]
        # Object-fluent equality: encode as a normal precondition where the
        # fluent variable's value is the constant/parameter on the other side.
        if lhs.is_fluent_exp() or rhs.is_fluent_exp():
            if t == 'false':
                raise NotImplementedError(
                    "Negated object-fluent equality is not supported in the plasp encoding."
                )
            fluent_side, value_side = (lhs, rhs) if lhs.is_fluent_exp() else (rhs, lhs)
            if value_side.is_fluent_exp():
                raise NotImplementedError(
                    "Equality between two object-fluent expressions is not supported."
                )
            return ASPExpr(fluent_side, _equality_value_term(value_side))
        # Parameter/object equality: lifted-plasp folds this into the action
        # signature body so only matching bindings instantiate the action.
        return ASPEquality(f, 'true' if t is None else t)
    if f.is_implies():
        assert False, "Implies should have been removed before."
    else:
        raise TypeError(f"Unsupported thing: {f} of type {type(f)}")
    return []

def is_numeric_comparison(f):
    """Is this FNode a numeric comparison (or the negation of one)?

    ``lt``/``le`` and numeric ``=`` are the linear comparisons the ASP
    encoding evaluates against ``numval``; UP folds ``GE``/``GT`` and their
    negations into these shapes. Used to route a *goal* onto the numeric
    path instead of the boolean/object state-goal path."""
    g = f.args[0] if f.is_not() else f
    if g.is_lt() or g.is_le():
        return True
    return g.is_equals() and (_is_numeric_fnode(g.args[0]) or _is_numeric_fnode(g.args[1]))


def _signature(parameters):
    """(ASP variable, ASPType) for each action/fluent parameter."""
    return [(asp_name(p.name).upper(), ASPType(p.type)) for p in parameters]


def _head_term(name, signature):
    """The ``("name", P1, ...)`` tuple term an action-like head is built from.

    A zero-parameter head is ``("name")``, which clingo reads as the plain
    string (a one-element parenthesis is not a tuple); the planner's plan
    extraction relies on that.
    """
    inner = f'"{asp_name(name)}"'
    if signature:
        inner += ',' + ','.join(variable for variable, _ in signature)
    return f'({inner})'


def action_declaration_atom(name, parameters) -> str:
    """The atom that declares an action, with each parameter as the ASP variable
    :class:`ASPAction` binds it to.

    Doubly wrapped -- ``action(action(("name", P1, ...)))`` -- because plasp tags
    a term with its declaring predicate, so the *term* is ``action((...))`` and
    the *fact* declaring it wraps that again. Useful as an :class:`ASPAction`
    ``signature_guard``, which is how one action's parameter bindings get
    restricted to another's.
    """
    return f'action(action({_head_term(name, _signature(parameters))}))'


def _equality_value_term(arg):
    if arg.is_object_exp():
        return f'constant("{asp_name(arg.object().name)}")'
    if arg.is_parameter_exp():
        return asp_name(arg.parameter().name).upper()
    raise TypeError(f"Unsupported equality term: {arg} of type {type(arg)}")

# Prefix for the ASP variable a *quantified* variable renders as, keeping it
# clear of an action parameter that happens to share its name.
QUANTIFIED_PREFIX = 'Q_'


def _arg_term(arg):
    # Returns (term_str, asp_type, kind). Objects become `constant("name")`
    # ground terms. Parameters become uppercase ASP variables, already bound by
    # the action's signature rule. A quantified variable becomes one too, but
    # nothing binds it yet, so whoever emits the rule has to add the
    # `has(_, type(...))` atom that ranges it over the universe -- which is how
    # a `forall` gets expanded, by the grounder rather than by a compiler.
    if arg.is_object_exp():
        return (f'constant("{asp_name(arg.object().name)}")', ASPType(arg.type), 'object')
    if arg.is_parameter_exp():
        return (asp_name(arg.parameter().name).upper(), ASPType(arg.type), 'parameter')
    if arg.is_variable_exp():
        variable = arg.variable()
        return (QUANTIFIED_PREFIX + asp_name(variable.name).upper(),
                ASPType(variable.type), 'quantified')
    raise TypeError(f"Unsupported fluent argument: {arg} of type {type(arg)}")

def _int_value(f):
    """The FNode's constant as a Python int; PDDL functions are real-typed in
    UP, so integral reals are accepted (clingo terms are integers)."""
    if f.is_int_constant():
        return f.constant_value()
    if f.is_real_constant():
        frac = f.constant_value()
        if frac.denominator != 1:
            raise NotImplementedError(
                f"Non-integral numeric constant is not supported by the ASP encoding: {f}")
        return int(frac)
    raise NotImplementedError(f"Expected a numeric constant, got: {f}")

def _num_side(f):
    """Normalize one side of a numeric comparison to (fluent_term|None, int_const).

    Supports the linear shapes that survive UP's compilation of PDDL numeric
    conditions: an int constant, a numeric fluent, and sums/differences of
    one fluent with constants (e.g. ``(+ (economy ?t) 1)``). Anything richer
    (two fluents on one side, multiplication) raises -- the PLASP encoding
    keeps one variable per side for readability; use the ABA backend for
    arithmetic that couples several fluents.
    """
    if f.is_int_constant() or f.is_real_constant():
        return None, _int_value(f)
    if f.is_fluent_exp():
        return ASPExpr(f, None), 0
    if f.is_plus():
        var, const = None, 0
        for arg in f.args:
            v, c = _num_side(arg)
            const += c
            if v is not None:
                if var is not None:
                    raise NotImplementedError(
                        f"Numeric side with two fluents is not supported: {f}")
                var = v
        return var, const
    if f.is_minus():
        lv, lc = _num_side(f.args[0])
        rv, rc = _num_side(f.args[1])
        if rv is not None:
            raise NotImplementedError(
                f"Subtracting a fluent is not supported: {f}")
        return lv, lc - rc
    raise NotImplementedError(f"Unsupported numeric term: {f} of type {f.node_type}")


class ASPBooleanType(ASPTerm):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"boolean({str(self.value).lower()})"


class ASPType(ASPTerm):
    def __init__(self, t):
        self.up_type = t

    def __str__(self):
        return f"type(\"{asp_name(self.up_type.name)}\")"


class ASPConstant(ASPTerm):
    def __init__(self, c):
        self.up_constant = c

    def __str__(self):
        return f"constant(\"{asp_name(self.up_constant.name)}\")"


class ASPHasConstant(ASPTerm):
    def __init__(self, c):
        self.up_constant  = c
        self.asp_type     = ASPType(c.type)
        self.asp_constant = ASPConstant(c)
        # Walk the type-hierarchy ancestor chain so an object typed as a
        # leaf (e.g. depot1) also satisfies has(_, type("place")) when the
        # encoding's action / variable rules ask for the supertype. Without
        # this, supertype-typed parameters never ground and entire actions
        # disappear from the program.
        self.asp_ancestor_types = []
        t = c.type
        while t is not None and t.is_user_type():
            self.asp_ancestor_types.append(ASPType(t))
            t = t.father

    def __str__(self):
        return '\n'.join(
            f"has({str(self.asp_constant)}, {str(at)})."
            for at in self.asp_ancestor_types
        )


class ASPFluent(ASPTerm):
    def __init__(self, f):
        self.up_fluent = f
        name = asp_name(f.name)
        arity_types = [(asp_name(a.name).upper(), ASPType(a.type)) for a in f.signature]
        self._head = f"\"{name}\"," + ','.join(a[0] for a in arity_types) if len(arity_types) > 0 else f"\"{name}\""
        self._head = f"variable(({self._head}))"
        self._body = ', '.join(f'has({a}, {str(t)})' for a, t in arity_types)

    def __str__(self):
        return f"variable({self._head})." if len(self._body) == 0 else f"variable({self._head}) :- {self._body}."


class ASPNumFluent(ASPFluent):
    """Declares an int-valued fluent as numVariable(...) so the encoding's
    numval/3 projection (and only it) picks its holds/3 atoms up as numbers."""
    def __str__(self):
        head = f"numVariable({self._head})"
        return f"{head}." if len(self._body) == 0 else f"{head} :- {self._body}."


class ASPExpr(ASPTerm):
    def __init__(self, f, value):
        self.up_expr = f
        terms = [_arg_term(a) for a in f.args]
        name = asp_name(f._content.payload.name)
        head_args = ','.join(t for t, _, _ in terms)
        self._head = f"\"{name}\",{head_args}" if head_args else f"\"{name}\""
        self.value = value
        # `has(_, type(...))` atoms for the quantified variables in this
        # expression, which the rule emitting it has to carry in its body:
        # ranging them over the universe there is what expands the `forall`.
        self.bindings = [f'has({term}, {str(asp_type)})'
                         for term, asp_type, kind in terms if kind == 'quantified']

    def __str__(self):
        return f"variable(({self._head}))"


class ASPEquality(ASPTerm):
    """Lifted-plasp parameter equality between two non-fluent terms.

    The encoder hoists this into the action's signature rule body so the
    action only instantiates for parameter bindings that satisfy the
    constraint. ``value == 'true'`` means ``=`` and ``'false'`` means ``!=``.
    """
    def __init__(self, f, value):
        self.up_expr = f
        self.value = value
        self.lhs = _equality_value_term(f.args[0])
        self.rhs = _equality_value_term(f.args[1])

    def __str__(self):
        op = '=' if self.value == 'true' else '!='
        return f'{self.lhs} {op} {self.rhs}'


class ASPNumComparison(ASPTerm):
    """Arithmetic comparison ``lhs OP rhs`` with each side value(V)+C.

    Rendered as ``op, expr(V1,C1), expr(V2,C2)`` (wrapped by the caller into a
    ``numPrecondition``/``numGoal`` fact); the encoding evaluates the sides
    against ``numval/3``. A constant-only side uses the pseudo-variable
    ``none`` (numval fixes it to 0, so the constant carries the value).
    Each side is a single fluent plus a constant -- expressions coupling two
    fluents (e.g. ``f + g``) raise; use the ABA backend for those.
    """
    def __init__(self, f, op):
        self.up_expr = f
        self.op = op
        self.lhs = _num_side(f.args[0])
        self.rhs = _num_side(f.args[1])

    def negated(self):
        # not(a < b) == b <= a ; not(a <= b) == b < a
        neg = ASPNumComparison.__new__(ASPNumComparison)
        neg.up_expr = self.up_expr
        neg.op = {'lt': 'le', 'le': 'lt'}[self.op]
        neg.lhs, neg.rhs = self.rhs, self.lhs
        return neg

    @staticmethod
    def _side_str(side):
        var, const = side
        return f'expr({str(var) if var is not None else "none"}, {const})'

    def __str__(self):
        return f'{self.op}, {self._side_str(self.lhs)}, {self._side_str(self.rhs)}'


class ASPGroundedFluent(ASPTerm):
    def __init__(self, f):
        self.up_fluent = f
        self._arity = list(map(lambda e: ASPConstant(e._content.payload), f.args))
        self._head = f"\"{asp_name(f._content.payload.name)}\""

    def __str__(self):
        _ret_str = f"{self._head}," + ','.join(str(a) for a in self._arity) if len(self._arity) > 0 else f"{self._head}"
        return f'variable(({_ret_str}))'


class ASPAction(ASPTerm):
    def __init__(self, a, static_fluents=None, signature_guard=None):
        # static_fluents: set of fluent names whose value is fixed by the
        # initial state (no action has them in its effects). Positive
        # preconditions on these are folded into the action signature body
        # so the grounder pre-filters parameter bindings by the static
        # relation, instead of cross-producing all type-typed bindings and
        # pruning at solve time. Without this, actions whose only
        # parameter-narrowing comes from a static precondition (e.g. TPP
        # `next/2`, Mystery `attacks/2`/`eats/2`) ground to |obj|^arity and
        # blow past clasp's 28-bit ID space.
        # signature_guard: an atom that already restricts every parameter, used
        # instead of the per-parameter has(_, type(...)) atoms. An end snap
        # passes its start snap's declaration atom, which is both tighter (the
        # start's folded static preconditions carry over) and sound -- an end
        # snap can only ever fire for a binding whose start exists.
        static_fluents = static_fluents or set()
        self.up_action = a
        # Fluents this action reads as *false*, whose `holds(V, value(V,false))`
        # chain therefore has to exist from step 0 on (see the encoder).
        self.negated_fluents = set()
        self.signature = _signature(a.parameters)
        self._head = f"action({_head_term(a.name, self.signature)})"
        self._sig_body = signature_guard if signature_guard is not None else \
            ', '.join(f'has({p[0]}, {str(p[1])})' for p in self.signature)


        # iterate over the preconditions.
        self._preconditions = []
        equality_atoms = []
        static_atoms = []
        group_index = 0
        for precondition in a.preconditions:
            # And-within-And nesting yields nested lists — flatten them.
            variablelist = flatten_atoms(parseexpr(precondition))
            for variable in variablelist:
                if isinstance(variable, ASPDisjunction):
                    self._preconditions += self._disjunction_facts(variable, group_index)
                    self.negated_fluents |= variable.negated_fluent_names
                    group_index += 1
                    continue
                if isinstance(variable, ASPEquality):
                    equality_atoms.append(str(variable))
                    continue
                if isinstance(variable, ASPNumComparison):
                    head = f'numPrecondition({self._head}, {str(variable)})'
                    self._preconditions.append(f"{head} :- action({self._head}).")
                    continue
                fluent_name = variable.up_expr._content.payload.name if isinstance(variable, ASPExpr) and variable.up_expr.is_fluent_exp() else None
                if fluent_name is not None and variable.value == 'false':
                    self.negated_fluents.add(fluent_name)
                # Only positive (true-valued) static preconditions are safe
                # to fold via initialState/2; closed-world handling for
                # `not initialState(...)` is non-trivial so we leave
                # negative cases on the runtime path.
                bindings = getattr(variable, 'bindings', [])
                if (fluent_name in static_fluents) and variable.value == 'true' and not bindings:
                    static_atoms.append(f'initialState({str(variable)}, value({str(variable)}, true))')
                    continue
                head = f'precondition({self._head}, {str(variable)}, value({str(variable)}, {variable.value}))'
                # action(...) ground-restricts every parameter via the action signature rule;
                # adding has(_, type(...)) per parameter forces the grounder to iterate
                # |objects|^arity body matches per rule with no semantic effect. A
                # *quantified* variable is the exception: nothing else binds it, and
                # ranging it over the universe here is what expands the forall.
                body = ', '.join([f'action({self._head})'] + bindings)
                self._preconditions.append(f"{head} :- {body}.")

        extra_body = equality_atoms + static_atoms
        if extra_body:
            joined = ', '.join(extra_body)
            self._sig_body = f'{self._sig_body}, {joined}' if self._sig_body else joined

        # iterate over the unconditional effects.
        self._postconditions = []
        num_deltas = {}   # fluent term -> summed constant delta
        for eff in a.unconditional_effects:
            if eff.kind in (EffectKind.INCREASE, EffectKind.DECREASE):
                term = str(parseexpr(eff.fluent))
                sign = 1 if eff.kind == EffectKind.INCREASE else -1
                num_deltas[term] = num_deltas.get(term, 0) + sign * _int_value(eff.value)
                continue
            if eff.kind == EffectKind.ASSIGN and not eff.fluent.type.is_bool_type():
                term = str(parseexpr(eff.fluent))
                head = f"numAssign({self._head}, {term}, {_int_value(eff.value)})"
                self._postconditions.append(f"{head} :- action({self._head}).")
                continue
            variable = parseexpr(eff.fluent)
            value    = str(eff.value).lower()
            head = f"postcondition({self._head}, effect(unconditional), {str(variable)}, value({str(variable)}, {value}))"
            # `forall ?x . f(?x) := v` needs ?x ranged the same way a forall
            # condition does; the caused/3 rule then fires once per binding.
            body = ', '.join([f'action({self._head})'] + list(variable.bindings))
            self._postconditions.append(f"{head} :- {body}.")
        for term, delta in num_deltas.items():
            if delta == 0:
                continue
            head = f"numEffect({self._head}, {term}, {delta})"
            self._postconditions.append(f"{head} :- action({self._head}).")

        # Conditional effects (`when C then F := V`). The existential/sequential
        # encoding already supports them via:
        #   caused(Var,Val,T) :- occurs(A,T), postcondition(A,Eff,Var,Val),
        #                        holds(VP,VVP,T-1) : precondition(Eff,VP,VVP).
        # i.e. an effect fires when every precondition(Eff, ...) atom holds at T-1.
        # We assign each conditional effect a unique Effect term parameterised by
        # the action's parameters so distinct ground instances don't share
        # preconditions (which would mix bindings across instances and require
        # every binding's condition to hold simultaneously).
        params_tail = (',' + ','.join(p[0] for p in self.signature)) if self.signature else ''
        for idx, eff in enumerate(a.conditional_effects):
            assert not eff.condition.is_or(), (
                f"Disjunctive condition in conditional effect of action {a.name} is not supported; "
                "apply DisjunctiveConditionsRemover to effect conditions or split the action."
            )
            variable = parseexpr(eff.fluent)
            value    = str(eff.value).lower()
            effect_term = f'effect((cond,"{asp_name(a.name)}",{idx}{params_tail}))'
            head = (
                f"postcondition({self._head}, {effect_term}, "
                f"{str(variable)}, value({str(variable)}, {value}))"
            )
            self._postconditions.append(f"{head} :- action({self._head}).")

            cond_atoms = parseexpr(eff.condition)
            cond_atoms = [cond_atoms] if not isinstance(cond_atoms, list) else cond_atoms
            for ca in cond_atoms:
                if isinstance(ca, ASPEquality):
                    # QuantifiersRemover.simplify() should ground-evaluate
                    # parameter/object equalities away. If one survives here
                    # it must reference an action parameter — fold it into
                    # the effect's precondition by emitting it as a body
                    # constraint on the rule itself.
                    self._postconditions[-1] = self._postconditions[-1][:-1] + f", {str(ca)}."
                    continue
                if isinstance(ca, ASPExpr) and ca.up_expr.is_fluent_exp() and ca.value == 'false':
                    self.negated_fluents.add(ca.up_expr._content.payload.name)
                cond_head = (
                    f"precondition({effect_term}, {str(ca)}, "
                    f"value({str(ca)}, {ca.value}))"
                )
                self._postconditions.append(f"{cond_head} :- action({self._head}).")

    def _disjunction_facts(self, disjunction, index):
        """orGroup/orDisjunct facts for one disjunctive precondition.

        The group is a plain fact of the action; each disjunct contributes one
        orDisjunct per literal, under whatever bindings that disjunct needs --
        which for an `exists` is what ranges it over the universe.
        """
        group = f'or({index})'
        facts = [f"orGroup({self._head}, {group}) :- action({self._head})."]
        for disjunct_id, atoms, bindings in disjunction.disjuncts:
            body = ', '.join([f'action({self._head})'] + list(bindings))
            for atom in atoms:
                if not isinstance(atom, ASPExpr):
                    raise NotImplementedError(
                        f"Unsupported literal {atom} in a disjunctive condition of action "
                        f"{self.up_action.name!r}.")
                head = (f"orDisjunct({self._head}, {group}, {disjunct_id}, "
                        f"{str(atom)}, value({str(atom)}, {atom.value}))")
                facts.append(f"{head} :- {body}.")
        return facts

    def __str__(self):
        _sig = [
            f"action({self._head})." if len(self._sig_body) == 0 else f"action({self._head}) :- {self._sig_body}."
        ]
        _sig += self._preconditions
        _sig += self._postconditions
        return '\n'.join(_sig)


class ASPDurativeAction(ASPTerm):
    """The temporal facts tying a durative action to its two snap actions.

    A durative action is encoded as the pair of instantaneous *snap* actions
    PDDL 2.1 decomposes it into -- exactly what SMTPlan's happening encoder
    does. Their at-start / at-end conditions and effects travel as ordinary
    ``precondition``/``postcondition`` facts of the snap actions themselves
    (see :class:`ASPAction`), so all this builder emits is what the encoding's
    temporal layer needs to couple the two halves back together:

      * ``durativeAction/1``  -- the durative action term, which also switches
        the temporal layer on;
      * ``snap/3``            -- which ordinary action is its start and its end;
      * ``durationValue/2``   -- the admissible durations in scaled time units
        (an interval when the task states a duration *constraint* rather than a
        fixed duration);
      * ``overall/3`` / ``numOverall/4`` -- the over-all conditions, which
        belong to neither snap: the encoding checks them at every happening the
        action spans.

    Each fact is guarded by the snap action it belongs to, so on a lifted task
    they instantiate for exactly the parameter bindings the snap actions do.
    """

    def __init__(self, da, start_name, end_name, duration_bounds, overall_conditions):
        self.up_action = da
        signature = _signature(da.parameters)
        self._head  = f"durative({_head_term(da.name, signature)})"
        self._start = f"action({_head_term(start_name, signature)})"
        self._end   = f"action({_head_term(end_name, signature)})"

        # Numeric bounds go straight into the fact; a duration stated as a
        # static fluent becomes a lookup in the initial state instead, so it
        # resolves per parameter binding at grounding time and the task never
        # has to be ground first (see common.temporal.FluentDuration).
        self._duration_body = None
        if isinstance(duration_bounds, tuple):
            lower, upper = duration_bounds
            self._duration = str(lower) if lower == upper else f"{lower}..{upper}"
        else:
            fluent = parseexpr(duration_bounds.fluent, None)
            self._duration = (f"RAWDURATION * {duration_bounds.multiplier} "
                              f"/ {duration_bounds.divisor}")
            self._duration_body = (
                f"initialState({str(fluent)}, value({str(fluent)}, RAWDURATION))")

        self._overall = []
        for condition in overall_conditions:
            atoms = parseexpr(condition)
            atoms = [atoms] if not isinstance(atoms, list) else atoms
            while any(isinstance(a, list) for a in atoms):
                atoms = [e for a in atoms for e in (a if isinstance(a, list) else [a])]
            for atom in atoms:
                if isinstance(atom, ASPNumComparison):
                    self._overall.append(f"numOverall({self._head}, {str(atom)})")
                elif isinstance(atom, ASPExpr):
                    self._overall.append(
                        f"overall({self._head}, {str(atom)}, value({str(atom)}, {atom.value}))")
                else:
                    raise NotImplementedError(
                        f"Unsupported over-all condition in durative action "
                        f"{da.name!r}: {condition}")

    def __str__(self):
        # `action(<term>)` is the declaration atom of the snap action (plasp
        # tags terms with their declaring predicate); guarding on it is what
        # instantiates these facts for exactly the bindings the snap has.
        started, ended = f"action({self._start})", f"action({self._end})"
        duration_body = started if self._duration_body is None \
            else f"{started}, {self._duration_body}"
        facts = [
            f"durativeAction({self._head}) :- {started}.",
            f"snap({self._head}, start, {self._start}) :- {started}.",
            f"snap({self._head}, end, {self._end}) :- {ended}.",
            f"durationValue({self._head}, {self._duration}) :- {duration_body}.",
        ]
        facts += [f"{atom} :- {started}." for atom in self._overall]
        return '\n'.join(facts)


class ASPStateVarVal(ASPTerm):
    def __init__(self, fluent, value):
        self.fluent = ASPGroundedFluent(fluent)
        self.value  = str(value).lower()

    def __str__(self):
        return f"{str(self.fluent)}, value({str(self.fluent)}, {self.value})"


class ASPInitialState(ASPStateVarVal):
    def __str__(self):
        return f"initialState({super().__str__()})."


class ASPGoalState(ASPTerm):
    """``goal(V, value(V, val))`` for one state goal.

    Built from a parsed :class:`ASPExpr` rather than a ground fluent, so a
    ``forall`` goal works the same way a ``forall`` precondition does: the
    quantified variable stays free in the head and its ``has(_, type(...))``
    binding goes in the body, leaving the expansion to the grounder.
    """

    def __init__(self, expr):
        if not isinstance(expr, ASPExpr):
            raise NotImplementedError(
                f"Unsupported state goal: {expr} of type {type(expr).__name__}.")
        self.expr = expr
        self.value = expr.value
        self.bindings = list(expr.bindings)

    @property
    def fluent_name(self):
        return self.expr.up_expr._content.payload.name

    def __str__(self):
        head = f"goal({str(self.expr)}, value({str(self.expr)}, {self.value}))"
        return f"{head}." if not self.bindings else f"{head} :- {', '.join(self.bindings)}."


class ASPGoalDisjunction(ASPTerm):
    """goalOrGroup/goalOrDisjunct facts for one disjunctive goal.

    The goal-time analogue of the orGroup/orDisjunct facts :class:`ASPAction`
    emits: same structure, checked against the state at the goal step rather
    than the one before an action.
    """

    def __init__(self, disjunction, index):
        self.disjunction = disjunction
        self._group = f'or({index})'

    @property
    def negated_fluent_names(self):
        return self.disjunction.negated_fluent_names

    def __str__(self):
        lines = [f"goalOrGroup({self._group})."]
        for disjunct_id, atoms, bindings in self.disjunction.disjuncts:
            for atom in atoms:
                if not isinstance(atom, ASPExpr):
                    raise NotImplementedError(
                        f"Unsupported literal {atom} in a disjunctive goal.")
                head = (f"goalOrDisjunct({self._group}, {disjunct_id}, "
                        f"{str(atom)}, value({str(atom)}, {atom.value}))")
                lines.append(f"{head}." if not bindings
                             else f"{head} :- {', '.join(bindings)}.")
        return '\n'.join(lines)


class ASPNumGoal(ASPTerm):
    """A numeric goal ``lhs OP rhs`` checked against numval at the goal step.

    The goal-time analogue of ASPNumComparison/numPrecondition: it reuses that
    class's ``op, expr(V,C), expr(V,C)`` rendering (via parseexpr, which folds
    ``GE``/``GT`` and negations into ``lt``/``le``/``eq``) and wraps it as a
    standalone ``numGoal(...)`` fact. The encoding rejects any model whose
    numval at the queried timestep violates it.
    """
    def __init__(self, f):
        self.cmp = parseexpr(f)
        assert isinstance(self.cmp, ASPNumComparison), (
            f"ASPNumGoal expects a numeric comparison, got {type(self.cmp)} for {f}"
        )

    def __str__(self):
        return f"numGoal({str(self.cmp)})."

