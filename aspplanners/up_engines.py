"""Unified Planning engine adapters for both backends, in one place.

Two ``OneshotPlanner`` engines are registered on ``import aspplanners``:

  * ``ASPPlanner``  -> :class:`UPPLASPPlanner`, the PLASP/clingo backend
    (:class:`aspplanners.plasp.planner.PLASPPlanner`); the default, numeric-capable.
  * ``ABAPlanner``  -> :class:`UPABAPlanner`, the STRIPS-to-ABA/aspforaba backend
    (:class:`aspplanners.abaplan.planner.ABAPlan`); optional, needs the ``aba`` extra.

The ABA backend's ``aspforaba`` dependency is imported lazily inside the planner,
so importing this module (and registering both engines) works without it.
"""

from typing import Callable, IO, Optional

import unified_planning as up
from unified_planning.engines import PlanGenerationResult, PlanGenerationResultStatus
from unified_planning.engines.results import LogMessage, LogLevel

from aspplanners.plasp.planner import PLASPPlanner
from aspplanners.abaplan.planner import ABAPlan


class UPPLASPPlanner(up.engines.Engine, up.engines.mixins.OneshotPlannerMixin):
    """UP engine adapter for the PLASP/clingo backend (engine name ``PLASPPlanner``).

    Recognised options:
      - ``encoding``    (str, default ``"seq"``)
      - ``horizon``     (int, solve at one fixed horizon)
      - ``max_horizon`` (int, default 1000, bound for the deepening search)
    """

    def __init__(self, **options):
        up.engines.Engine.__init__(self)
        up.engines.mixins.OneshotPlannerMixin.__init__(self)
        self.conf = options

    @property
    def name(self) -> str:
        return "PLASPPlanner"

    @staticmethod
    def supported_kind():
        # Classical planning plus SIMPLE numeric planning (integer-valued
        # fluents, constant-delta increase/decrease/assign, linear comparisons).
        # Quantified/disjunctive/negative conditions are compiled away by the UP
        # compilers in the encoder pipeline. Reals are accepted because PDDL
        # (:functions ...) parse as real-typed; the encoder raises on
        # non-integral constants.
        from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
        kind = up.model.ProblemKind(version=LATEST_PROBLEM_KIND_VERSION)
        kind.set_problem_class('ACTION_BASED')
        kind.set_problem_type('SIMPLE_NUMERIC_PLANNING')
        kind.set_typing('FLAT_TYPING')
        kind.set_typing('HIERARCHICAL_TYPING')
        kind.set_numbers('BOUNDED_TYPES')
        kind.set_fluents_type('INT_FLUENTS')
        kind.set_fluents_type('REAL_FLUENTS')
        kind.set_conditions_kind('NEGATIVE_CONDITIONS')
        kind.set_conditions_kind('DISJUNCTIVE_CONDITIONS')
        kind.set_conditions_kind('EQUALITIES')
        kind.set_conditions_kind('EXISTENTIAL_CONDITIONS')
        kind.set_conditions_kind('UNIVERSAL_CONDITIONS')
        kind.set_effects_kind('CONDITIONAL_EFFECTS')
        kind.set_effects_kind('INCREASE_EFFECTS')
        kind.set_effects_kind('DECREASE_EFFECTS')
        return kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= UPPLASPPlanner.supported_kind()

    def _solve(self, problem: 'up.model.Problem',
               callback: Optional[Callable[['up.engines.PlanGenerationResult'], None]] = None,
               timeout: Optional[float] = None,
               output_stream: Optional[IO[str]] = None) -> 'up.engines.PlanGenerationResult':
        encoding = self.conf.get('encoding', 'seq')
        horizon = self.conf.get('horizon')
        max_horizon = self.conf.get('max_horizon', 1000)

        planner = PLASPPlanner(problem, encoding)
        plan = planner.plan(horizon=horizon, max_horizon=max_horizon, timeout=timeout)
        status = planner.status
        solved = status == PlanGenerationResultStatus.SOLVED_SATISFICING
        logs = [LogMessage(LogLevel.INFO, message) for message in planner.logs]
        return PlanGenerationResult(status, plan if solved else None, self.name, log_messages=logs)

    def destroy(self):
        pass


class UPABAPlanner(up.engines.Engine, up.engines.mixins.OneshotPlannerMixin):
    """UP engine adapter for the ABA backend (engine name ``ABAPlanner``).

    Recognised options:
      - ``max_horizon`` (int, default 1000)
      - ``semantics``   (str, default ``"ST"``)
    """

    def __init__(self, **options):
        up.engines.Engine.__init__(self)
        up.engines.mixins.OneshotPlannerMixin.__init__(self)
        self.conf = options

    @property
    def name(self) -> str:
        return "ABAPlanner"

    @staticmethod
    def supported_kind():
        # Classical planning plus simple numeric planning (integer/real fluents
        # with increase/decrease/assign effects and linear comparisons, via
        # finite-domain propositionalisation). Quantified/disjunctive/negative
        # conditions are compiled away by the UP compilers in the pipeline.
        # Conditional effects are NOT supported by the ABA encoding.
        kind = up.model.ProblemKind()
        kind.set_problem_class("ACTION_BASED")
        kind.set_problem_type("SIMPLE_NUMERIC_PLANNING")
        kind.set_typing("FLAT_TYPING")
        kind.set_typing("HIERARCHICAL_TYPING")
        kind.set_numbers("BOUNDED_TYPES")
        kind.set_fluents_type("INT_FLUENTS")
        kind.set_fluents_type("REAL_FLUENTS")
        kind.set_conditions_kind("NEGATIVE_CONDITIONS")
        kind.set_conditions_kind("DISJUNCTIVE_CONDITIONS")
        kind.set_conditions_kind("EQUALITIES")
        kind.set_conditions_kind("EXISTENTIAL_CONDITIONS")
        kind.set_conditions_kind("UNIVERSAL_CONDITIONS")
        kind.set_effects_kind("INCREASE_EFFECTS")
        kind.set_effects_kind("DECREASE_EFFECTS")
        return kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= UPABAPlanner.supported_kind()

    def _solve(self, problem: "up.model.Problem",
               callback: Optional[Callable[["up.engines.PlanGenerationResult"], None]] = None,
               timeout: Optional[float] = None,
               output_stream: Optional[IO[str]] = None) -> "up.engines.PlanGenerationResult":
        max_horizon = int(self.conf.get("max_horizon", 1000))
        semantics = self.conf.get("semantics", "ST")

        planner = ABAPlan(problem)
        plan = planner.plan(max_horizon=max_horizon, semantics=semantics)
        status = planner.status
        solved = status == PlanGenerationResultStatus.SOLVED_SATISFICING
        logs = [LogMessage(LogLevel.INFO, message) for message in planner.logs]
        return PlanGenerationResult(status, plan if solved else None, self.name, log_messages=logs)

    def destroy(self):
        pass
