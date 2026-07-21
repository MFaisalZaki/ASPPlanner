from typing import Callable, IO, Optional
import unified_planning as up
from unified_planning.engines import PlanGenerationResultStatus, PlanGenerationResult
from unified_planning.engines.results import LogMessage, LogLevel

from aspplanner.plasp.planner import ASPPlanner

# We have to args: linear, upper_bound
class UPASPPlanner(up.engines.Engine, up.engines.mixins.OneshotPlannerMixin):
    def __init__(self, **options):
        # Read known user-options and store them for using in the `solve` method
        up.engines.Engine.__init__(self)
        up.engines.mixins.OneshotPlannerMixin.__init__(self)
        self.conf = options

    @property
    def name(self) -> str:
        return "ASPPlanner"

    @staticmethod
    def supported_kind():
        # Only claim what the encoding actually handles end-to-end: classical
        # planning plus SIMPLE numeric planning (integer-valued fluents,
        # constant-delta increase/decrease/assign, linear comparisons).
        # Quantified/disjunctive/negative conditions are compiled away by the
        # UP compilers in the encoder pipeline. Reals are accepted because
        # PDDL (:functions ...) parse as real-typed; the encoder raises on
        # non-integral constants.
        from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
        supported_kind = up.model.ProblemKind(version=LATEST_PROBLEM_KIND_VERSION)
        supported_kind.set_problem_class('ACTION_BASED')
        supported_kind.set_problem_type('SIMPLE_NUMERIC_PLANNING')
        supported_kind.set_typing('FLAT_TYPING')
        supported_kind.set_typing('HIERARCHICAL_TYPING')
        supported_kind.set_numbers('BOUNDED_TYPES')
        supported_kind.set_fluents_type('INT_FLUENTS')
        supported_kind.set_fluents_type('REAL_FLUENTS')
        supported_kind.set_conditions_kind('NEGATIVE_CONDITIONS')
        supported_kind.set_conditions_kind('DISJUNCTIVE_CONDITIONS')
        supported_kind.set_conditions_kind('EQUALITIES')
        supported_kind.set_conditions_kind('EXISTENTIAL_CONDITIONS')
        supported_kind.set_conditions_kind('UNIVERSAL_CONDITIONS')
        supported_kind.set_effects_kind('CONDITIONAL_EFFECTS')
        supported_kind.set_effects_kind('INCREASE_EFFECTS')
        supported_kind.set_effects_kind('DECREASE_EFFECTS')
        return supported_kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= UPASPPlanner.supported_kind()

    def _solve(self, problem: 'up.model.Problem',
              callback: Optional[Callable[['up.engines.PlanGenerationResult'], None]] = None,
              timeout: Optional[float] = None,
              output_stream: Optional[IO[str]] = None) -> 'up.engines.PlanGenerationResult':

        encoding    = self.conf.get('encoding', 'seq')
        horizon     = self.conf.get('horizon')
        max_horizon = self.conf.get('max_horizon', 1000)

        planner = ASPPlanner(problem, encoding)
        plan = planner.plan(horizon=horizon, max_horizon=max_horizon, timeout=timeout)
        status = planner.status
        solved = status == PlanGenerationResultStatus.SOLVED_SATISFICING
        logs = [LogMessage(LogLevel.INFO, message) for message in planner.logs]
        return PlanGenerationResult(status, plan if solved else None, self.name, log_messages=logs)

    def destroy(self):
        pass