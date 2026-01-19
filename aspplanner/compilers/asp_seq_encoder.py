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
from unified_planning.model.walkers.names_extractor import NamesExtractor
from unified_planning.engines.compilers.quantifiers_remover import QuantifiersRemover
from unified_planning.engines.compilers.disjunctive_conditions_remover import DisjunctiveConditionsRemover

from unified_planning.model import (
    Problem,
    ProblemKind,
    Action,
)

from typing import Optional, Dict
from functools import partial

from aspplanner.compilers.delete_then_set_remover import DeleteThenSetRemover
from aspplanner.compilers.renamer import Renamer

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


class ASPSeqEncoder(engines.engine.Engine, CompilerMixin):
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
        return "aspseqencoder"

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
        supported_kind.set_quality_metrics("MAKESPAN")
        supported_kind.set_quality_metrics("FINAL_VALUE")
        supported_kind.set_actions_cost_kind("INT_NUMBERS_IN_ACTIONS_COST")
        supported_kind.set_actions_cost_kind("REAL_NUMBERS_IN_ACTIONS_COST")
        supported_kind.set_oversubscription_kind("INT_NUMBERS_IN_OVERSUBSCRIPTION")
        supported_kind.set_oversubscription_kind("REAL_NUMBERS_IN_OVERSUBSCRIPTION")
        return supported_kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= ASPSeqEncoder.supported_kind()

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

        assert len(problem.user_types) > 1, "ASP Encoder cannot deal with problems that are untyped."

        env = problem.environment
        em = env.expression_manager

        self.basic_problem = problem
        # Apply all required compilations before translating into ASP.
        removed_delete_then_task  = DeleteThenSetRemover().compile(problem).problem
        removed_quantifiers_task  = QuantifiersRemover().compile(removed_delete_then_task).problem
        # removed_impiles_task      = ImpliesRewrite().compile(removed_quantifiers_task).problem # this one is fukcing buggy and I won't spend time with it.
        removed_impiles_task = removed_quantifiers_task
        removed_disjunctions_task = DisjunctiveConditionsRemover().compile(removed_impiles_task).problem
        
        # make sure that all actions are anded.
        # this is due to an issue from the DisjunctiveConditionsRemover since it does not an preconditions together.
        # I am not sure if UP considers the default list as an And or not.
        self.__resolve_actions_preconditions__(removed_disjunctions_task)
        
        renamed_problem = Renamer().compile(removed_disjunctions_task).problem
        

        # Use this for translation.
        original_problem = renamed_problem.clone()
        new_problem = original_problem.clone()
        new_problem.name = f"{self.name}_{problem.name}"

        setattr(new_problem, 'asp_encoding',     {})
        setattr(new_problem, 'asp_encoding_str', {})

        new_problem.asp_encoding['_types']          = set(ASPType(t) for t in original_problem.user_types)
        new_problem.asp_encoding['_default_values'] = set(ASPBooleanType(v) for v in [True, False])
        new_problem.asp_encoding['_constants']      = set(ASPConstant(obj) for obj in original_problem.all_objects)
        new_problem.asp_encoding['_has']            = set(ASPHasConstant(obj) for obj in original_problem.all_objects)
        new_problem.asp_encoding['_variables']      = set(ASPFluent(fluent) for fluent in original_problem.fluents)
        new_problem.asp_encoding['_actions']        = set(ASPAction(action) for action in original_problem.actions)
        new_problem.asp_encoding['_initial_state']  = set(ASPInitialState(fluent, value) for fluent, value in original_problem.initial_values.items() if not value.is_false())
        new_problem.asp_encoding['_goal_state']     = set(chain.from_iterable(self.__generate_asp_goal_state__(g) for g in original_problem.goals))
        
        # This is a corner case where the initial state has no true fluents. In this case we need to add all the fluents of the problem.
        if len(new_problem.asp_encoding['_initial_state']) == 0:
            new_problem.asp_encoding['_initial_state'] = set(ASPInitialState(fluent, value) for fluent, value in original_problem.initial_values.items())

        for k, v in new_problem.asp_encoding.items():
            if 'constant' in k: new_problem.asp_encoding_str[k] = set(f'constant({e})' for e in chain.from_iterable(str(s).split('\n') for s in v))
            elif 'type' in k: new_problem.asp_encoding_str[k] = set(f'type({e})' for e in chain.from_iterable(str(s).split('\n') for s in v))
            elif 'default_values' in k: new_problem.asp_encoding_str[k] = set(f'{e}.' for e in chain.from_iterable(str(s).split('\n') for s in v))
            else: new_problem.asp_encoding_str[k] = set(chain.from_iterable(str(s).split('\n') for s in v))
        
        # now check if every entry ends with a dot.
        for k, v in new_problem.asp_encoding_str.items():
            new_problem.asp_encoding_str[k] = set(line if line.strip().endswith('.') else line.strip() + '.' for line in v)

        return CompilerResult(
            new_problem, partial(replace_action, map={a: a for a in original_problem.actions}), self.name
        )
    
    def __resolve_actions_preconditions__(self, problem: Problem):

        for idx, action in enumerate(problem.actions):
            if len(action.preconditions) <= 1: continue
            _expr = And(action.preconditions)
            problem.actions[idx].clear_preconditions()
            problem.actions[idx].add_precondition(_expr)

    def __generate_asp_goal_state__(self, goal_state):
        goal_predicates = [goal_state] if goal_state.node_type != OperatorKind.AND else goal_state.args
        ret_goals = []
        for g in goal_predicates:
            _is_true = g.node_type != OperatorKind.NOT
            value = str(_is_true).lower()
            ret_goals.append(ASPGoalState(g if _is_true else g.args[0], value))
        return ret_goals