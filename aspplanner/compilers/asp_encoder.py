"""This module defines the ASP encoder class."""

import unified_planning as up
import unified_planning.engines as engines

from itertools import chain
from collections import defaultdict, OrderedDict

from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
from unified_planning.engines.results import CompilerResult
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.engines.compilers.utils import replace_action
from unified_planning.shortcuts import OperatorKind, InstantaneousAction, FNode, Fluent, And
from unified_planning.shortcuts import EffectKind, CompilationKind, Compiler
from unified_planning.model.walkers.names_extractor import NamesExtractor
from unified_planning.engines.compilers.quantifiers_remover import QuantifiersRemover
from unified_planning.engines.compilers.disjunctive_conditions_remover import DisjunctiveConditionsRemover
from unified_planning.engines.compilers.conditional_effects_remover import ConditionalEffectsRemover

from unified_planning.model import (
    Problem,
    ProblemKind,
    Action,
)

from typing import Optional, Dict
from functools import partial

from aspplanner.compilers.renamer import Renamer
from aspplanner.compilers.tim_typer import TIMTypeInferenceCompiler

from aspplanner.compilers.asp_facts import (
    ASPType,
    ASPBooleanType,
    ASPConstant,
    ASPHasConstant,
    ASPFluent,
    ASPAction,
    ASPInitialState,
    ASPGoalState
)

class ASPEncoder(engines.engine.Engine, CompilerMixin):
    """
    This is a recreation of the PLASP tool
    """

    def __init__(self):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.GROUNDING)
        self.fluent_map = defaultdict(str)
        self.fluent_map_args = defaultdict(dict)

    @property
    def name(self):
        return "aspencoder"

    @staticmethod
    def supported_kind() -> ProblemKind:
        supported_kind = ProblemKind(version=LATEST_PROBLEM_KIND_VERSION)
        supported_kind.set_problem_class("ACTION_BASED")
        supported_kind.set_typing("FLAT_TYPING")
        supported_kind.set_typing("HIERARCHICAL_TYPING")
        supported_kind.set_numbers("BOUNDED_TYPES")
        supported_kind.set_problem_type("SIMPLE_NUMERIC_PLANNING")
        supported_kind.set_problem_type("GENERAL_NUMERIC_PLANNING")
        supported_kind.set_fluents_type("INT_FLUENTS")
        supported_kind.set_fluents_type("REAL_FLUENTS")
        supported_kind.set_fluents_type("OBJECT_FLUENTS")
        supported_kind.set_conditions_kind("NEGATIVE_CONDITIONS")
        supported_kind.set_conditions_kind("DISJUNCTIVE_CONDITIONS")
        supported_kind.set_conditions_kind("EQUALITIES")
        supported_kind.set_conditions_kind("EXISTENTIAL_CONDITIONS")
        supported_kind.set_conditions_kind("UNIVERSAL_CONDITIONS")
        supported_kind.set_effects_kind("CONDITIONAL_EFFECTS")
        supported_kind.set_effects_kind("INCREASE_EFFECTS")
        supported_kind.set_effects_kind("DECREASE_EFFECTS")
        supported_kind.set_effects_kind("STATIC_FLUENTS_IN_BOOLEAN_ASSIGNMENTS")
        supported_kind.set_effects_kind("STATIC_FLUENTS_IN_NUMERIC_ASSIGNMENTS")
        supported_kind.set_effects_kind("STATIC_FLUENTS_IN_OBJECT_ASSIGNMENTS")
        supported_kind.set_effects_kind("FLUENTS_IN_BOOLEAN_ASSIGNMENTS")
        supported_kind.set_effects_kind("FLUENTS_IN_NUMERIC_ASSIGNMENTS")
        supported_kind.set_effects_kind("FLUENTS_IN_OBJECT_ASSIGNMENTS")
        supported_kind.set_effects_kind("FORALL_EFFECTS")
        supported_kind.set_simulated_entities("SIMULATED_EFFECTS")
        supported_kind.set_constraints_kind("STATE_INVARIANTS")
        supported_kind.set_constraints_kind("TRAJECTORY_CONSTRAINTS")
        supported_kind.set_quality_metrics("ACTIONS_COST")
        supported_kind.set_actions_cost_kind("STATIC_FLUENTS_IN_ACTIONS_COST")
        supported_kind.set_actions_cost_kind("FLUENTS_IN_ACTIONS_COST")
        supported_kind.set_quality_metrics("PLAN_LENGTH")
        supported_kind.set_quality_metrics("OVERSUBSCRIPTION")
        supported_kind.set_quality_metrics("MAKESPAN")
        supported_kind.set_quality_metrics("FINAL_VALUE")
        supported_kind.set_actions_cost_kind("INT_NUMBERS_IN_ACTIONS_COST")
        supported_kind.set_actions_cost_kind("REAL_NUMBERS_IN_ACTIONS_COST")
        supported_kind.set_oversubscription_kind("INT_NUMBERS_IN_OVERSUBSCRIPTION")
        supported_kind.set_oversubscription_kind("REAL_NUMBERS_IN_OVERSUBSCRIPTION")
        return supported_kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= ASPEncoder.supported_kind()

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return True

    @staticmethod
    def resulting_problem_kind(
        problem_kind: ProblemKind,
        compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        return problem_kind.clone() # we do not change the problem kind

    def _compile(
        self,
        problem: "up.model.AbstractProblem",
        compilation_kind: "up.engines.CompilationKind",
    ) -> CompilerResult:
        assert isinstance(problem, Problem)

        env = problem.environment
        em = env.expression_manager

        self.new_problem = problem.clone()
        self.new_problem.name = f"{self.name}_{problem.name}"
        
        # Compilation pipeline:
        # step one remove delete then set effects.
        self.new_problem.clear_actions()
        for a in problem.actions:
            self.new_problem.add_action(self._remove_delete_then_set(a))
        
        compilationlist  = []
        compilationlist += [["up_quantifiers_remover", CompilationKind.QUANTIFIERS_REMOVING]]
        compilationlist += [["up_negative_conditions_remover", CompilationKind.NEGATIVE_CONDITIONS_REMOVING]]
        compilationlist += [["up_disjunctive_conditions_remover", CompilationKind.DISJUNCTIVE_CONDITIONS_REMOVING]]
        compiler_names = [c[0] for c in compilationlist]
        compiler_kinds = [c[1] for c in compilationlist]
        with Compiler(names=compiler_names, compilation_kinds=compiler_kinds) as compiler:
            self.grounded_problem = compiler.compile(self.new_problem)
        
        self.new_problem = self.grounded_problem.problem
        # note that disjunctive conditions remover decouples the conjunction of preconditions
        # and this impacts the renamer.
        _actions = self.new_problem.actions.copy()
        self.new_problem.clear_actions()
        for a in _actions:
            self.new_problem.add_action(self._fix_action_preconditions(a))

        # step two check if we can infer types for untyped problems.
        if len(self.new_problem.user_types) == 1:
            self.new_problem = TIMTypeInferenceCompiler().compile(self.new_problem.clone()).problem
        
        # last step rename all the fluents and actions to avoid name clashes with the ASP encoding.
        self.new_problem = Renamer().compile(self.new_problem).problem

        # to make the ASP solver life's easier, we will create the helper fluent for the goal predicate.
        # this predicate once is set to true it remains true, and this is should be used only by
        # the goal predicate dimension.
        setattr(self.new_problem, 'asp_encoding',     {})
        setattr(self.new_problem, 'asp_encoding_str', {})

        # A fluent is "static" iff no action ever lists it in its effects.
        # Folding positive preconditions on such fluents into the action
        # signature body lets the grounder pre-filter parameter bindings
        # by the static relation (see ASPAction docstring).
        modified_fluent_names = set()
        for a in self.new_problem.actions:
            for eff in a.effects:
                modified_fluent_names.add(eff.fluent._content.payload.name)
        static_fluent_names = {f.name for f in self.new_problem.fluents if f.name not in modified_fluent_names}

        self.new_problem.asp_encoding['_types']          = set(ASPType(t) for t in self.new_problem.user_types)
        self.new_problem.asp_encoding['_default_values'] = set(ASPBooleanType(v) for v in [True, False])
        self.new_problem.asp_encoding['_constants']      = set(ASPConstant(obj) for obj in self.new_problem.all_objects)
        self.new_problem.asp_encoding['_has']            = set(ASPHasConstant(obj) for obj in self.new_problem.all_objects)
        self.new_problem.asp_encoding['_variables']      = set(ASPFluent(fluent) for fluent in self.new_problem.fluents)
        self.new_problem.asp_encoding['_actions']        = set(ASPAction(action, static_fluent_names) for action in self.new_problem.actions)
        self.new_problem.asp_encoding['_initial_state']  = set(ASPInitialState(fluent, value) for fluent, value in self.new_problem.initial_values.items() if not value.is_false())
        self.new_problem.asp_encoding['_goal_state']     = set(chain.from_iterable(self._generate_asp_goal_state(g) for g in self.new_problem.goals))
        
        # This is a corner case where the initial state has no true fluents. In this case we need to add all the fluents of the problem.
        if len(self.new_problem.asp_encoding['_initial_state']) == 0:
            self.new_problem.asp_encoding['_initial_state'] = set(ASPInitialState(fluent, value) for fluent, value in self.new_problem.initial_values.items())

        for k, v in self.new_problem.asp_encoding.items():
            if 'constant' in k: self.new_problem.asp_encoding_str[k] = set(f'constant({e})' for e in chain.from_iterable(str(s).split('\n') for s in v))
            elif 'type' in k: self.new_problem.asp_encoding_str[k] = set(f'type({e})' for e in chain.from_iterable(str(s).split('\n') for s in v))
            elif 'default_values' in k: self.new_problem.asp_encoding_str[k] = set(f'{e}.' for e in chain.from_iterable(str(s).split('\n') for s in v))
            else: self.new_problem.asp_encoding_str[k] = set(chain.from_iterable(str(s).split('\n') for s in v))
        
        # now check if every entry ends with a dot.
        for k, v in self.new_problem.asp_encoding_str.items():
            self.new_problem.asp_encoding_str[k] = set(line if line.strip().endswith('.') else line.strip() + '.' for line in v)

        # Make sure to initialise all the fluents.
        self._initialize_fluents(self.new_problem)
        
        return CompilerResult(
            self.new_problem, partial(replace_action, map={a: a for a in self.new_problem.actions}), self.name
        )
    
    def _fix_action_preconditions(self, action: Action):
            if len(action.preconditions) <= 1: return action
            _new_action = action.clone()
            _new_action.clear_preconditions()
            _new_action.add_precondition(And(action.preconditions))
            return _new_action

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
        from unified_planning.shortcuts import get_environment
        from unified_planning.model.fluent import get_all_fluent_exp
        from itertools import chain
        # update the initial defaults to account for real and integer types.
        _env = get_environment()
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