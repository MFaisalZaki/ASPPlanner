"""This module defines the ASP encoder class."""

from dataclasses import dataclass
from itertools import chain

from unified_planning.engines.compilers.utils import replace_action
from unified_planning.shortcuts import OperatorKind
from unified_planning.shortcuts import EffectKind, Compiler, CompilationKind

from unified_planning.model import (
    Problem,
    Action,
)

from typing import Callable, Dict, Optional, Set
from functools import partial

from unified_planning.plans import ActionInstance

from aspplanner.compilers.tim_typer import TIMTypeInferenceCompiler

from aspplanner.compilers.asp_facts import (
    asp_name,
    ASPType,
    ASPBooleanType,
    ASPConstant,
    ASPHasConstant,
    ASPFluent,
    ASPNumFluent,
    ASPAction,
    ASPInitialState,
    ASPGoalState
)


def _check_asp_name_collisions(problem: Problem) -> None:
    """The ASP rendering maps '-' to '_' (see asp_facts.asp_name); two UP
    names that only differ there would become indistinguishable in the model
    atoms, so fail loudly before encoding."""
    groups = (
        ('action', (a.name for a in problem.actions)),
        ('object', (o.name for o in problem.all_objects)),
        ('fluent', (f.name for f in problem.fluents)),
        ('type',   (t.name for t in problem.user_types)),
    )
    for what, names in groups:
        seen = {}
        for name in names:
            rendered = asp_name(name)
            if rendered in seen and seen[rendered] != name:
                raise ValueError(
                    f"ASP renaming '-'->'_' makes {what} names collide: "
                    f"{seen[rendered]!r} and {name!r}")
            seen[rendered] = name

def _compose_map_backs(map_backs):
    """Chain the per-stage plan map-backs of the compilation pipeline.

    The extracted plan is stated on the LAST stage's problem, so the maps
    apply in reverse pipeline order; each lifts the action instance one
    stage closer to the user's original problem."""
    def _map_back(action_instance):
        for map_back in reversed(map_backs):
            action_instance = map_back(action_instance)
            if action_instance is None:
                return None
        return action_instance
    return _map_back


@dataclass
class ASPEncodingResult:
    """What the planner needs from the compilation pipeline.

    `problem` is the fully compiled problem (grounded for classical tasks,
    lifted for numeric ones; renamed either way) whose actions/objects the
    ASP model's atoms refer to. `facts` holds the ASP fact lines grouped by
    kind; `map_back_action_instance` lifts a plan step on `problem` back to
    the user's original problem.
    """
    problem: Problem
    facts: Dict[str, Set[str]]
    map_back_action_instance: Callable[[ActionInstance], Optional[ActionInstance]]

    @property
    def fact_lines(self) -> Set[str]:
        """All fact lines, flattened."""
        return set().union(*self.facts.values()) if self.facts else set()


def _render_facts(asp_objects, wrap: Optional[str] = None) -> Set[str]:
    """Render ASP fact builders to normalized fact lines.

    Builders may emit several newline-separated lines (ASPHasConstant,
    ASPAction); each line is optionally wrapped in a declaration predicate
    (`wrap='type'` turns the term `type("t")` into the fact `type(type("t"))`,
    plasp's convention of tagging terms with their declaring predicate) and
    always ends with a dot.
    """
    lines = set()
    for asp_object in asp_objects:
        for line in str(asp_object).split('\n'):
            line = line.strip()
            if not line:
                continue
            if wrap is not None:
                line = f'{wrap}({line})'
            if not line.endswith('.'):
                line += '.'
            lines.add(line)
    return lines


class ASPEncoder:
    """
    This is a recreation of the PLASP tool: compiles a UP Problem into the
    ASP facts consumed by the encodings in `aspplanner/encodings/`.
    """

    name = "aspencoder"

    def compile(self, problem: Problem) -> ASPEncodingResult:
        assert isinstance(problem, Problem)

        new_problem = problem.clone()
        new_problem.name = f"{self.name}_{problem.name}"

        # Each compilation stage contributes a map-back; composed in reverse
        # they lift a plan on the final problem to the user's original problem.
        map_backs = []

        # Compilation pipeline:
        # step one remove delete then set effects.
        new_problem.clear_actions()
        delete_then_set_map = {}
        for a in problem.actions:
            clean_action = self._remove_delete_then_set(a)
            new_problem.add_action(clean_action)
            delete_then_set_map[clean_action] = a
        map_backs.append(partial(replace_action, map=delete_then_set_map))

        # Numeric tasks skip the grounding step entirely: the Fast Downward
        # reachability grounder rejects them, and pre-grounding with UP's
        # grounder only bloats the program — the lifted encoding already
        # carries everything gringo needs to instantiate (action signature
        # rules bind parameters via has(_, type(...)) and folded static
        # preconditions prune the bindings). PDDL (:functions ...) parse as
        # real-typed fluents; the fact builders accept them as long as every
        # constant is integral (clingo terms are integers) and raise otherwise.
        kind = new_problem.kind
        numeric = kind.has_int_fluents() or kind.has_real_fluents()

        compilationlist  = []
        compilationlist += [["up_quantifiers_remover", CompilationKind.QUANTIFIERS_REMOVING]]
        compilationlist += [["up_negative_conditions_remover", CompilationKind.NEGATIVE_CONDITIONS_REMOVING]]
        compilationlist += [["up_disjunctive_conditions_remover", CompilationKind.DISJUNCTIVE_CONDITIONS_REMOVING]]
        if not numeric:
            compilationlist += [["fast-downward-reachability-grounder", CompilationKind.GROUNDING]]
        compiler_names = [c[0] for c in compilationlist]
        compiler_kinds = [c[1] for c in compilationlist]
        with Compiler(names=compiler_names, compilation_kinds=compiler_kinds) as compiler:
            grounded_result = compiler.compile(new_problem)
        map_backs.append(grounded_result.map_back_action_instance)

        new_problem = grounded_result.problem

        # step two check if we can infer types for untyped problems.
        if len(new_problem.user_types) == 1:
            tim_result = TIMTypeInferenceCompiler().compile(new_problem)
            new_problem = tim_result.problem
            map_backs.append(tim_result.map_back_action_instance)

        # Names are sanitized ('-' -> '_') at fact-rendering time by
        # asp_facts.asp_name and mapped back the same way during plan
        # extraction; make sure that mapping is injective for this task.
        _check_asp_name_collisions(new_problem)

        # A fluent is "static" iff no action ever lists it in its effects.
        # Folding positive preconditions on such fluents into the action
        # signature body lets the grounder pre-filter parameter bindings
        # by the static relation (see ASPAction docstring).
        modified_fluent_names = set()
        for a in new_problem.actions:
            for eff in a.effects:
                modified_fluent_names.add(eff.fluent._content.payload.name)
        static_fluent_names = {f.name for f in new_problem.fluents if f.name not in modified_fluent_names}

        # Fill in default values (bool False, int 0) BEFORE the initial-state
        # facts are emitted: an uninitialized numeric fluent would otherwise
        # have no holds/3 chain, which silently disables every numeric
        # precondition that reads it.
        self._initialize_fluents(new_problem)

        initial_state = set(ASPInitialState(fluent, value) for fluent, value in new_problem.initial_values.items() if not value.is_false())
        # This is a corner case where the initial state has no true fluents. In this case we need to add all the fluents of the problem.
        if len(initial_state) == 0:
            initial_state = set(ASPInitialState(fluent, value) for fluent, value in new_problem.initial_values.items())

        facts = {
            '_types':          _render_facts((ASPType(t) for t in new_problem.user_types), wrap='type'),
            '_default_values': _render_facts(ASPBooleanType(v) for v in [True, False]),
            '_constants':      _render_facts((ASPConstant(obj) for obj in new_problem.all_objects), wrap='constant'),
            '_has':            _render_facts(ASPHasConstant(obj) for obj in new_problem.all_objects),
            '_variables':      _render_facts(ASPFluent(fluent) for fluent in new_problem.fluents),
            '_num_variables':  _render_facts(ASPNumFluent(fluent) for fluent in new_problem.fluents if fluent.type.is_int_type() or fluent.type.is_real_type()),
            '_actions':        _render_facts(ASPAction(action, static_fluent_names) for action in new_problem.actions),
            '_initial_state':  _render_facts(initial_state),
            '_goal_state':     _render_facts(chain.from_iterable(self._generate_asp_goal_state(g) for g in new_problem.goals)),
        }

        return ASPEncodingResult(
            problem=new_problem,
            facts=facts,
            map_back_action_instance=_compose_map_backs(map_backs),
        )
    
    def _remove_delete_then_set(self, dirty_action: Action) -> Action:
        """!
        Removes delete-then-set effects from the list of effects.
        @param effects: list of effects
        @return list of effects without delete-then-set effects
        """

        def has_positive_effect(fluent, action) -> bool:
            """ Does the action has an effect that assigns the fluent to true? """
            for eff in action.effects:
                if eff.kind == EffectKind.ASSIGN and eff.fluent == fluent and eff.value.is_true():
                    return True
            return False

        clean_effects = []
        for eff in dirty_action.effects:
        # we avoid adding the effect if it is a delete effect and the action has also an add effect for the same fluent
            if eff.fluent.type.is_bool_type(): # only check boolean fluents
                if eff.kind == EffectKind.ASSIGN and \
                    eff.value.is_false() and \
                    has_positive_effect(eff.fluent, dirty_action):
                    pass
                else: 
                    clean_effects.append(eff)
            else: 
                clean_effects.append(eff)

        fixed_action = dirty_action.clone() # we copy the old action
        fixed_action.clear_effects()        # and remove all the effects
        for eff in clean_effects:           # now we copy over only the good effects
            if eff.kind == EffectKind.ASSIGN:
                fixed_action.add_effect(eff.fluent, eff.value, eff.condition, forall=eff.forall)
            if eff.kind == EffectKind.DECREASE:
                fixed_action.add_decrease_effect(eff.fluent, eff.value, eff.condition, forall=eff.forall)
            if eff.kind == EffectKind.INCREASE:
                fixed_action.add_increase_effect(eff.fluent, eff.value, eff.condition, forall=eff.forall)
        return fixed_action

    def _generate_asp_goal_state(self, goal_state):
        goal_predicates = [goal_state] if goal_state.node_type != OperatorKind.AND else goal_state.args
        ret_goals = []
        for g in goal_predicates:
            _is_true = g.node_type != OperatorKind.NOT
            value = str(_is_true).lower()
            ret_goals.append(ASPGoalState(g if _is_true else g.args[0], value))
        return ret_goals
    
    def _initialize_fluents(self, task:Problem):
        """
        Initialize the int and real fluents of a given task with a default value of 0.
        Any Boolean fluent is initialized with a default value of False.
        Args:
            task (Problem): The UP task object
        Updates:
            task.initial_defaults: Adds default values for real and integer types.
            task.explicit_initial_values: Sets initial values for uninitialized fluents.
        """
        from unified_planning.shortcuts import Fraction
        from unified_planning.model.fluent import get_all_fluent_exp
        # update the initial defaults to account for real and integer types.
        # Use the task's own environment: with a non-global environment the
        # global one holds different type/expression manager instances.
        _env = task.environment
        _tm = _env.type_manager
        _em = _env.expression_manager
        task.initial_defaults.update({_tm.RealType():_em.Real(Fraction(0))})
        task.initial_defaults.update({_tm.IntType() :_em.Int(0)})
        task.initial_defaults.update({_tm.BoolType() :_em.Bool(False)})

        # list unitialized fluents.
        fluentslist = list(chain.from_iterable([list(get_all_fluent_exp(task, f)) for f in task.fluents]))
        initialized_fluents  = list(task.explicit_initial_values.keys())
        unintialized_fluents = list(filter(lambda x: not x in initialized_fluents, fluentslist))
        
        # update the initial values for the fluents that are not initialized.
        for fe in unintialized_fluents:
            task.set_initial_value(fe, task.initial_defaults[fe.type]) 