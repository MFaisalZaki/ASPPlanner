"""unified-planning OneshotPlanner wrapper around :class:`ABAPlan`."""

from typing import Callable, IO, Optional

import unified_planning as up
from unified_planning.engines import PlanGenerationResult, PlanGenerationResultStatus
from unified_planning.engines.results import LogMessage, LogLevel

from aspplanner.abaplan.planner import ABAPlan


class UPABAPlanner(up.engines.Engine, up.engines.mixins.OneshotPlannerMixin):
    """UP engine adapter for the ABA (Assumption-Based Argumentation) backend.

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
        # STRIPS only: no numeric planning. Quantified/disjunctive/negative
        # conditions are compiled away by the UP compilers in the pipeline.
        kind = up.model.ProblemKind()
        kind.set_problem_class("ACTION_BASED")
        kind.set_typing("FLAT_TYPING")
        kind.set_typing("HIERARCHICAL_TYPING")
        kind.set_conditions_kind("NEGATIVE_CONDITIONS")
        kind.set_conditions_kind("DISJUNCTIVE_CONDITIONS")
        kind.set_conditions_kind("EQUALITIES")
        kind.set_conditions_kind("EXISTENTIAL_CONDITIONS")
        kind.set_conditions_kind("UNIVERSAL_CONDITIONS")
        return kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= UPABAPlanner.supported_kind()

    def _solve(
        self,
        problem: "up.model.Problem",
        callback: Optional[Callable[["up.engines.PlanGenerationResult"], None]] = None,
        timeout: Optional[float] = None,
        output_stream: Optional[IO[str]] = None,
    ) -> "up.engines.PlanGenerationResult":
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
