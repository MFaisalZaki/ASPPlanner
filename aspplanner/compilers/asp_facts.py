
from typing import IO, Iterable, List, Union

from clingo import ast as clingo_ast

from unified_planning.shortcuts import FNode, EffectKind

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
    if f.is_and() or f.is_or():
        return [parseexpr(arg, t) for arg in f.args]
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


def _equality_value_term(arg):
    if arg.is_object_exp():
        return f'constant("{asp_name(arg.object().name)}")'
    if arg.is_parameter_exp():
        return asp_name(arg.parameter().name).upper()
    raise TypeError(f"Unsupported equality term: {arg} of type {type(arg)}")

def _arg_term(arg):
    # Returns (term_str, asp_type, needs_has_binding). Objects become
    # `constant("name")` ground terms; parameters become uppercase ASP
    # variables that must be bound by `has(_, type(...))` in the rule body.
    if arg.is_object_exp():
        return (f'constant("{asp_name(arg.object().name)}")', ASPType(arg.type), False)
    if arg.is_parameter_exp():
        return (asp_name(arg.parameter().name).upper(), ASPType(arg.type), True)
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
    (two fluents on one side, multiplication) raises.
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


class ASPTerm:
    """Base for every ASP fact/term builder in this module: identity is the
    rendered string, so sets of facts deduplicate on their final ASP text."""

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, value):
        return str(self) == str(value)


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
    """Arithmetic precondition ``lhs OP rhs`` with each side value(V)+C.

    Rendered as ``numPrecondition(Action, op, expr(V1,C1), expr(V2,C2))``
    facts; the encoding evaluates the sides against ``numval/3`` at T-1.
    A constant-only side uses the pseudo-variable ``none`` (numval fixes it
    to 0, so the constant carries the value).
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
    def __init__(self, a, static_fluents=None):
        # static_fluents: set of fluent names whose value is fixed by the
        # initial state (no action has them in its effects). Positive
        # preconditions on these are folded into the action signature body
        # so the grounder pre-filters parameter bindings by the static
        # relation, instead of cross-producing all type-typed bindings and
        # pruning at solve time. Without this, actions whose only
        # parameter-narrowing comes from a static precondition (e.g. TPP
        # `next/2`, Mystery `attacks/2`/`eats/2`) ground to |obj|^arity and
        # blow past clasp's 28-bit ID space.
        static_fluents = static_fluents or set()
        self.up_action = a
        name = asp_name(a.name)
        self.signature = list(map(lambda p: (asp_name(p.name).upper(), ASPType(p.type)), a.parameters))
        self._head = f"\"{name}\"," + ','.join(p[0] for p in self.signature) if len(self.signature) > 0 else f"\"{name}\""
        self._head = f"action(({self._head}))"
        self._sig_body = ', '.join(f'has({p[0]}, {str(p[1])})' for p in self.signature)


        # iterate over the preconditions.
        self._preconditions = []
        equality_atoms = []
        static_atoms = []
        for precondition in a.preconditions:
            variablelist = parseexpr(precondition)
            variablelist = [variablelist] if not isinstance(variablelist, list) else variablelist
            # And-within-And nesting yields nested lists — flatten them.
            while any(isinstance(v, list) for v in variablelist):
                variablelist = [e for v in variablelist for e in (v if isinstance(v, list) else [v])]
            for variable in variablelist:
                if isinstance(variable, ASPEquality):
                    equality_atoms.append(str(variable))
                    continue
                if isinstance(variable, ASPNumComparison):
                    head = f'numPrecondition({self._head}, {str(variable)})'
                    self._preconditions.append(f"{head} :- action({self._head}).")
                    continue
                fluent_name = variable.up_expr._content.payload.name if isinstance(variable, ASPExpr) and variable.up_expr.is_fluent_exp() else None
                # Only positive (true-valued) static preconditions are safe
                # to fold via initialState/2; closed-world handling for
                # `not initialState(...)` is non-trivial so we leave
                # negative cases on the runtime path.
                if (fluent_name in static_fluents) and variable.value == 'true':
                    static_atoms.append(f'initialState({str(variable)}, value({str(variable)}, true))')
                    continue
                head = f'precondition({self._head}, {str(variable)}, value({str(variable)}, {variable.value}))'
                # action(...) ground-restricts every parameter via the action signature rule;
                # adding has(_, type(...)) per parameter forces the grounder to iterate
                # |objects|^arity body matches per rule with no semantic effect.
                self._preconditions.append(f"{head} :- action({self._head}).")

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
            self._postconditions.append(f"{head} :- action({self._head}).")
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
                cond_head = (
                    f"precondition({effect_term}, {str(ca)}, "
                    f"value({str(ca)}, {ca.value}))"
                )
                self._postconditions.append(f"{cond_head} :- action({self._head}).")

    def __str__(self):
        _sig = [
            f"action({self._head})." if len(self._sig_body) == 0 else f"action({self._head}) :- {self._sig_body}."
        ]
        _sig += self._preconditions
        _sig += self._postconditions
        return '\n'.join(_sig)


class ASPStateVarVal(ASPTerm):
    def __init__(self, fluent, value):
        self.fluent = ASPGroundedFluent(fluent)
        self.value  = str(value).lower()

    def __str__(self):
        return f"{str(self.fluent)}, value({str(self.fluent)}, {self.value})"


class ASPInitialState(ASPStateVarVal):
    def __str__(self):
        return f"initialState({super().__str__()})."


class ASPGoalState(ASPStateVarVal):
    def __str__(self):
        return f"goal({super().__str__()})."


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


# ---------------------------------------------------------------------------
# Reading and writing .lp files as ASPTerm objects
# ---------------------------------------------------------------------------

class ASPStatement(ASPTerm):
    """A statement parsed from ASP text, wrapping its clingo AST node.

    ``str()`` renders it back to valid (clingo-normalized) ASP syntax:
    comments are discarded, whitespace is normalized, and body literals are
    separated by ';'. The underlying ``clingo.ast.AST`` stays available as
    ``self.node`` for structural inspection.
    """

    def __init__(self, node):
        self.node = node

    def __str__(self):
        return str(self.node)


class ASPFact(ASPStatement):
    """A body-less rule: ``head.``"""

    @property
    def head(self) -> str:
        return str(self.node.head)


class ASPRule(ASPStatement):
    """``head :- body.``"""

    @property
    def head(self) -> str:
        return str(self.node.head)

    @property
    def body(self) -> List[str]:
        return [str(b) for b in self.node.body]


class ASPConstraint(ASPStatement):
    """An integrity constraint: ``:- body.``

    clingo's AST renders these as ``#false :- body.``; str() keeps the
    conventional head-less spelling.
    """

    @property
    def body(self) -> List[str]:
        return [str(b) for b in self.node.body]

    def __str__(self):
        return f":- {'; '.join(self.body)}."


class ASPWeakConstraint(ASPStatement):
    """``:~ body. [weight@priority, terms]``"""

    @property
    def body(self) -> List[str]:
        return [str(b) for b in self.node.body]


class ASPDirective(ASPStatement):
    """Any non-rule statement: #program, #show, #defined, #external,
    #script, #const, ..."""


def _wrap_statement(node) -> ASPStatement:
    if node.ast_type == clingo_ast.ASTType.Rule:
        head = node.head
        is_constraint = (head.ast_type == clingo_ast.ASTType.Literal
                         and head.atom.ast_type == clingo_ast.ASTType.BooleanConstant
                         and not head.atom.value)
        if is_constraint:
            return ASPConstraint(node)
        if len(node.body) == 0:
            return ASPFact(node)
        return ASPRule(node)
    if node.ast_type == clingo_ast.ASTType.Minimize:
        return ASPWeakConstraint(node)
    return ASPDirective(node)


def _is_base_program_node(node) -> bool:
    return (node.ast_type == clingo_ast.ASTType.Program
            and node.name == 'base' and len(node.parameters) == 0)


def parse_lp(text: str) -> List[ASPStatement]:
    """Parse ASP program text into a list of ASPStatement terms.

    clingo always emits an implicit ``#program base.`` as the first node
    (duplicating an explicit one); it is dropped so statements before any
    #program directive stay in the implicit base part and parse -> dump ->
    parse is a fixpoint.
    """
    nodes = []
    clingo_ast.parse_string(text, nodes.append)
    if nodes and _is_base_program_node(nodes[0]):
        nodes = nodes[1:]
    return [_wrap_statement(n) for n in nodes]


def parse_lp_file(path) -> List[ASPStatement]:
    """Parse an .lp file into a list of ASPStatement terms."""
    with open(path, 'r') as f:
        return parse_lp(f.read())


def dump_lp(terms: Iterable, destination: Union[str, IO[str]]) -> None:
    """Write ASP terms to `destination` (a file path or file-like object),
    one statement per line.

    Accepts anything that renders to ASP text with str(): parsed
    ASPStatement objects, the fact-builder ASPTerm classes above (which may
    render to several lines), or plain strings.
    """
    text = '\n'.join(str(t) for t in terms) + '\n'
    if hasattr(destination, 'write'):
        destination.write(text)
    else:
        with open(destination, 'w') as f:
            f.write(text)
