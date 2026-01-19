
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
    if f.is_implies():
        assert False, "Implies should have been removed before."
    else:
        raise TypeError(f"Unsupported thing: {f} of type {type(f)}")
    return []

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
        
    def __str__(self):
        return f"has({str(self.asp_constant)}, {str(self.asp_type)})."
    
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
        self._arity_types = list(map(lambda a: (str(a).upper(), ASPType(a.type)), f.args))
        self._head = f"\"{f._content.payload.name}\"," + ','.join(a[0] for a in self._arity_types) if len(self._arity_types) > 0 else f"\"{f._content.payload.name}\""
        self._body = ', '.join(f'has({a}, {str(t)})' for a, t in self._arity_types)
        self.value   = value
        
    def __str__(self):
        return f"variable(({self._head}))"
    
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
    def __init__(self, a):
        self.up_action = a
        self.signature = list(map(lambda p: (p.name.upper(), ASPType(p.type)), a.parameters))
        self._head = f"\"{a.name}\"," + ','.join(p[0] for p in self.signature) if len(self.signature) > 0 else f"\"{a.name}\""
        self._head = f"action(({self._head}))"
        self._sig_body = ', '.join(f'has({p[0]}, {str(p[1])})' for p in self.signature)
        

        # iterate over the preconditions.
        self._preconditions = []
        for precondition in a.preconditions:
            variablelist = parseexpr(precondition)
            variablelist = [variablelist] if not isinstance(variablelist, list) else variablelist
            for variable in variablelist:
                head = f'precondition({self._head}, {str(variable)}, value({str(variable)}, {variable.value}))'
                body = [f"action({self._head})"]
                for argname, argtype in variable._arity_types:
                    body.append(f"has({argname}, {str(argtype)})")
                body = ', '.join(body)
                self._preconditions.append(f"{head} :- {body}.")

        # iterate over the unconditional effects.
        self._postconditions = []
        for eff in a.unconditional_effects:
            variable = parseexpr(eff.fluent)
            value    = str(eff.value).lower()
            _val = value
            head = f"postcondition({self._head}, effect(unconditional), {str(variable)}, value({str(variable)}, {_val}))"
            body = [f"action({self._head})"]
            for argname, argtype in variable._arity_types:
                body.append(f"has({argname}, {str(argtype)})")
            body = ', '.join(body)
            self._postconditions.append(f"{head} :- {body}.")

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