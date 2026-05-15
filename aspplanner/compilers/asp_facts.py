
from unified_planning.shortcuts import FNode

def parseexpr(f, t=None):
    """!
    given a FNode from UP representing a fluent and a possible timestep return a
    string representation of it.
    """
    assert isinstance(f, FNode), f"Expected a FNode, got {type(f)}"
    if  f.is_fluent_exp(): # for fluents
        return ASPExpr(f, 'true' if t is None else t) #(ASPFluent(f._content.payload), 'true' if t is None else t)
    if f.is_bool_constant():
        return ASPExpr(f, str(f).lower()) #(ASPFluent(f._content.payload), str(f).lower())
    if f.is_not():
        return parseexpr(f.args[0], 'false')
    if f.is_and() or f.is_or():
        return [parseexpr(arg, t) for arg in f.args]
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

def _equality_value_term(arg):
    if arg.is_object_exp():
        return f'constant("{arg.object().name}")'
    if arg.is_parameter_exp():
        return arg.parameter().name.upper()
    raise TypeError(f"Unsupported equality term: {arg} of type {type(arg)}")

def _arg_term(arg):
    # Returns (term_str, asp_type, needs_has_binding). Objects become
    # `constant("name")` ground terms; parameters become uppercase ASP
    # variables that must be bound by `has(_, type(...))` in the rule body.
    if arg.is_object_exp():
        return (f'constant("{arg.object().name}")', ASPType(arg.type), False)
    if arg.is_parameter_exp():
        return (arg.parameter().name.upper(), ASPType(arg.type), True)
    raise TypeError(f"Unsupported fluent argument: {arg} of type {type(arg)}")

class ASPRule:
    def __init__(self, expr):
        self.head = expr.split(":-")[0].strip()
        self.body = expr.split(":-")[1].strip()[:-1]
        
    def __str__(self):
        return f"{self.head} :- {self.body}."
    
    def __hash__(self):
        return hash(str(self))
    
    def __eq__(self, value):
        return str(self) == str(value)

class ASPConstraint:
    def __init__(self, expr):
        self.body = expr.replace(":-", "").strip()[:-1]
        
    def __str__(self):
        return f":- {self.body}."
    
    def __hash__(self):
        return hash(str(self))
    
    def __eq__(self, value):
        return str(self) == str(value)

class ASPFact:
    def __init__(self, fact):
        self.fact = fact[:-1]
        
    def __str__(self):
        return f"{str(self.fact)}."
    
    def __hash__(self):
        return hash(str(self))
    
    def __eq__(self, value):
        return str(self) == str(value)

class ASPCmd:
    def __init__(self, cmd):
        self.cmd = cmd
        
    def __str__(self):
        return f"{str(self.cmd)}"
    
    def __hash__(self):
        return hash(str(self))
        
    def __eq__(self, value):
        return str(self) == str(value)

class ASPBooleanType:
    def __init__(self, value):
        self.value = value
        
    def __str__(self):
        return f"boolean({str(self.value).lower()})"
    
    def __hash__(self):
        return hash(str(self))
    
    def __eq__(self, value):
        return str(self) == str(value)

class ASPType:
    def __init__(self, t):
        self.up_type = t
        
    def __str__(self):
        return f"type(\"{self.up_type.name}\")"
    
    def __hash__(self):
        return hash(str(self))
    
    def __eq__(self, value):
        return str(self) == str(value)

class ASPVariable:
    def __init__(self, v):
        self.up_variable = v
        

    def __str__(self):
        return f"variable(\"{self.up_variable.name}\")"
    
    def __hash__(self):
        return hash(str(self))
    
    def __eq__(self, value):
        return str(self) == str(value)

class ASPConstant:
    def __init__(self, c):
        self.up_constant = c
    
    def __str__(self):
        return f"constant(\"{self.up_constant.name}\")"
    
    def __hash__(self):
        return hash(str(self))
    
    def __eq__(self, value):
        return str(self) == str(value)

class ASPHasConstant:
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
    
    def __hash__(self):
        return hash(str(self))
    
    def __eq__(self, value):
        return str(self) == str(value)

class ASPFluent:
    def __init__(self, f):
        self.up_fluent = f
        self._arity_types = list(map(lambda a: (a.name.upper(), ASPType(a.type)), f.signature))
        self._head = f"\"{f.name}\"," + ','.join(a[0] for a in self._arity_types) if len(self._arity_types) > 0 else f"\"{f.name}\""
        self._head = f"variable(({self._head}))"
        self._body = ', '.join(f'has({a}, {str(t)})' for a, t in self._arity_types)
        
    def __str__(self):
        return f"variable({self._head})." if len(self._body) == 0 else f"variable({self._head}) :- {self._body}."
    
    def __hash__(self):
        return hash(str(self))
    
    def __eq__(self, value):
        return str(self) == str(value)

class ASPExpr:
    def __init__(self, f, value):
        self.up_expr = f
        terms = [_arg_term(a) for a in f.args]
        self._arity_types = [(t, ty) for t, ty, _ in terms]
        name = f._content.payload.name
        head_args = ','.join(t for t, _, _ in terms)
        self._head = f"\"{name}\",{head_args}" if head_args else f"\"{name}\""
        self._body = ', '.join(f'has({t}, {ty})' for t, ty, needs in terms if needs)
        self.value = value
        
    def __str__(self):
        return f"variable(({self._head}))"
    
    def __hash__(self):
        return hash(str(self))
    
    def __eq__(self, value):
        return str(self) == str(value)

class ASPEquality:
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
        self._arity_types = []

    def __str__(self):
        op = '=' if self.value == 'true' else '!='
        return f'{self.lhs} {op} {self.rhs}'

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, value):
        return str(self) == str(value)

class ASPGroundedFluent:
    def __init__(self, f):
        self.up_fluent = f
        self._arity = list(map(lambda e: ASPConstant(e._content.payload), f.args))
        self._head = f"\"{f._content.payload.name}\""
        
    def __str__(self):
        _ret_str = f"{self._head}," + ','.join(str(a) for a in self._arity) if len(self._arity) > 0 else f"{self._head}"
        return f'variable(({_ret_str}))'
    
    def __hash__(self):
        return hash(str(self))
    
    def __eq__(self, value):
        return str(self) == str(value)

class ASPAction:
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
        self.signature = list(map(lambda p: (p.name.upper(), ASPType(p.type)), a.parameters))
        self._head = f"\"{a.name}\"," + ','.join(p[0] for p in self.signature) if len(self.signature) > 0 else f"\"{a.name}\""
        self._head = f"action(({self._head}))"
        self._sig_body = ', '.join(f'has({p[0]}, {str(p[1])})' for p in self.signature)


        # iterate over the preconditions.
        self._preconditions = []
        equality_atoms = []
        static_atoms = []
        for precondition in a.preconditions:
            variablelist = parseexpr(precondition)
            variablelist = [variablelist] if not isinstance(variablelist, list) else variablelist
            for variable in variablelist:
                if isinstance(variable, ASPEquality):
                    equality_atoms.append(str(variable))
                    continue
                fluent_name = variable.up_expr._content.payload.name if isinstance(variable, ASPExpr) else None
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
        for eff in a.unconditional_effects:
            variable = parseexpr(eff.fluent)
            value    = str(eff.value).lower()
            _val = value
            head = f"postcondition({self._head}, effect(unconditional), {str(variable)}, value({str(variable)}, {_val}))"
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
            effect_term = f'effect((cond,"{a.name}",{idx}{params_tail}))'
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
    
    def __hash__(self):
        return hash(str(self))
    
    def __eq__(self, value):
        return str(self) == str(value)

class ASPStateVarVal:
    def __init__(self, fluent, value):
        self.fluent = ASPGroundedFluent(fluent)
        self.value  = str(value).lower()
        
    def __str__(self):
        return f"{str(self.fluent)}, value({str(self.fluent)}, {self.value})"
    
    def __hash__(self):
        return hash(str(self))
    
    def __eq__(self, value):
        return str(self) == str(value)

class ASPInitialState(ASPStateVarVal):
    def __init__(self, fluent, value):
        super().__init__(fluent, value)
    
    def __str__(self):
        return f"initialState({super().__str__()})."

class ASPGoalState(ASPStateVarVal):
    def __init__(self, fluent, value):
        super().__init__(fluent, value)

    def __str__(self):
        return f"goal({super().__str__()})."

class ASPOccursFluent:
    def __init__(self, fact):
        self.fact = fact
        self.timestep = fact.arguments[1].number
        self.fact_str = str(fact)
    
    def __str__(self):
        return f'occurs(action("{self.fact_str}"), {self.timestep})'
    
    def __hash__(self):
        return hash(str(self))
    
    def __eq__(self, value):
        return str(self) == str(value)