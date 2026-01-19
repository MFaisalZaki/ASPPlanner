"""This module defines the delete-then-set remover class."""

import unified_planning as up
import unified_planning.engines as engines
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
from unified_planning.engines.results import CompilerResult
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.engines.compilers.utils import replace_action
from unified_planning.shortcuts import EffectKind, OperatorKind
from unified_planning.model.walkers import Simplifier
from unified_planning.model.walkers.names_extractor import NamesExtractor
from unified_planning.model.walkers.dnf import Nnf

from unified_planning.model import (
    Problem,
    ProblemKind,
    Action,
)

from typing import Optional, Dict
from functools import partial

# TODO: check if this can be better integrated via the 
# add_engine: https://github.com/aiplan4eu/unified-planning/blob/master/unified_planning/engines/factory.py
class ImpliesRewrite(engines.engine.Engine, CompilerMixin):
    """
    This compiler rewrites implications in the problem.
    """

    def __init__(self):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.GROUNDING)

    @property
    def name(self):
        return "dtsrm"

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
        return problem_kind <= ImpliesRewrite.supported_kind()

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return True #compilation_kind == CompilationKind.DELETE_THEN_SET_REMOVING # we do not support anything in particular, just cleaning up the problem

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
        """
        Takes an instance of a :class:`~unified_planning.model.Problem` and the wanted :class:`~unified_planning.engines.CompilationKind`
        and returns a :class:`~unified_planning.engines.results.CompilerResult` where the :meth:`problem<unified_planning.engines.results.CompilerResult.problem>`
        field does not have state innvariants.

        :param problem: The instance of the
        :class:`~unified_planning.model.Problem` that must be returned without
        state innvariants.

        :param compilation_kind: The
        :class:`~unified_planning.engines.CompilationKind` that must be applied
        on the given problem; only
        :class:`~unified_planning.engines.CompilationKind.STATE_INVARIANTS_REMOVING`
        is supported by this compiler
        
        :return: The resulting :class:`~unified_planning.engines.results.CompilerResult` data structure.
        """
        assert isinstance(problem, Problem)
        env = problem.environment
        self.em = env.expression_manager
    
        new_problem = problem.clone()
        new_problem.name = f"{self.name}_{problem.name}"
        new_problem.clear_actions()

        self.simplifier = Simplifier(env, new_problem)
        self.name_extractor = NamesExtractor()
        self.nnf_converter = Nnf(env)

        new_to_old: Dict[Action, Optional[Action]] = {}

        for a in problem.actions:
            new_action = a.clone()
            new_action.clear_preconditions()
            for precondition in [self.__simplify_precondition__(self.__rewrite_implications__(p)) for p in a.preconditions]:
                new_action.add_precondition(precondition)
            new_problem.add_action(new_action)
            new_to_old[new_action] = a
        
        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )
    
    def __simplify_precondition__(self, fluent):
        predicates = []
        for arg in fluent.args:
            if len(self.name_extractor.extract_names(arg)) == 0: continue # skip this
            predicates.append(arg)
        if fluent.is_and():
            return self.em.And(predicates)
        elif fluent.is_or():
            return self.em.Or(predicates)
        elif fluent.node_type == OperatorKind.FLUENT_EXP:
            return fluent
        elif fluent.is_not():
            return self.em.Not(self.__simplify_precondition__(fluent.args[0]))
        else:
            raise NotImplementedError("Unsupported fluent type")
        
    def __rewrite_implications__(self, fluent):
        if fluent.is_implies():
            return self.em.Or(self.em.Not(fluent.args[0]), fluent.args[1])
        elif fluent.is_and():
            return self.em.And([self.__rewrite_implications__(arg) for arg in fluent.args])
        elif fluent.is_or():
            return self.em.Or([self.__rewrite_implications__(arg) for arg in fluent.args])
        elif fluent.is_not():
            return self.em.Not(self.__rewrite_implications__(fluent.args[0]))
        else:
            return fluent

    