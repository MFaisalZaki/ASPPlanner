"""Unified Planning engine adapters for both backends, in one place.

Two ``OneshotPlanner`` engines are registered on ``import aspplanners``:

  * ``PLASPPlanner`` -> :class:`UPPLASPPlanner`, the PLASP/clingo backend
    (:class:`aspplanners.plasp.planner.PLASPPlanner`); the default, numeric-capable.
  * ``ABAPlanner``   -> :class:`UPABAPlanner`, the STRIPS-to-ABA/aspforaba backend
    (:class:`aspplanners.abaplan.planner.ABAPlan`); optional, needs the ``aba`` extra.

The ABA backend's ``aspforaba`` dependency is imported lazily inside the planner,
so importing this module (and registering both engines) works without it.
"""

from typing import Callable, IO, Optional

import unified_planning as up
from unified_planning.engines import PlanGenerationResult, PlanGenerationResultStatus
from unified_planning.engines.results import LogMessage, LogLevel

from aspplanners.common.temporal import DEFAULT_TIME_SCALE
from aspplanners.plasp.planner import PLASPPlanner
from aspplanners.abaplan.planner import ABAPlan


class UPPLASPPlanner(up.engines.Engine, up.engines.mixins.OneshotPlannerMixin):
    """UP engine adapter for the PLASP/clingo backend (engine name ``PLASPPlanner``).

    Recognised options:
      - ``encoding``    (str, default ``"seq"``)
      - ``horizon``     (int, solve at one fixed horizon)
      - ``max_horizon`` (int, default 1000, bound for the deepening search)
      - ``time_scale``  (int, default 10, resolution of the temporal encoding's
        integer happening times, as a multiple of the durations' gcd)
      - ``compilationlist`` (list of ``[engine_name, CompilationKind]`` pairs;
        takes over the automatic selection, see
        :meth:`PLASPPlanner._check_compilationlist`)
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
        # Negative, quantified and disjunctive conditions need no compiling --
        # the encoding states all of them (see PLASPPlanner._check_compilationlist).
        # Reals are accepted because PDDL (:functions ...) parse as real-typed;
        # the encoder raises on non-integral constants.
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
        # Temporal planning over the PDDL 2.1 durative-action fragment, encoded
        # as SMTPlan's happenings (see encodings/sequential-horizon.lp). Left
        # out on purpose: SELF_OVERLAPPING (a durative action may not overlap
        # itself), INTERMEDIATE_CONDITIONS_AND_EFFECTS (conditions and effects
        # sit at the two snap actions only), TIMED_EFFECTS/TIMED_GOALS, and
        # PROCESSES/EVENTS (PDDL+ continuous change).
        kind.set_time('CONTINUOUS_TIME')
        kind.set_time('DURATION_INEQUALITIES')
        kind.set_expression_duration('INT_TYPE_DURATIONS')
        kind.set_expression_duration('REAL_TYPE_DURATIONS')
        kind.set_expression_duration('STATIC_FLUENTS_IN_DURATIONS')
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
        time_scale = int(self.conf.get('time_scale', DEFAULT_TIME_SCALE))
        compilationlist = self.conf.get('compilationlist')

        planner = PLASPPlanner(problem, encoding, compilationlist=compilationlist,
                               time_scale=time_scale)
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
      - ``time_scale``  (int, default 10, resolution of the temporal encoding's
        integer happening times, as a multiple of the durations' gcd)
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
        # Conditional effects are NOT supported by the ABA encoding, and neither
        # are numeric over-all conditions (the reduction turns an over-all
        # condition into a static test on the actions that could break it, which
        # only exists for boolean ones) -- ProblemKind has no feature for that
        # distinction, so it is raised at encoding time instead.
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
        # Temporal planning over the same PDDL 2.1 durative-action fragment the
        # PLASP backend covers; see UPPLASPPlanner.supported_kind.
        kind.set_time("CONTINUOUS_TIME")
        kind.set_time("DURATION_INEQUALITIES")
        kind.set_expression_duration("INT_TYPE_DURATIONS")
        kind.set_expression_duration("REAL_TYPE_DURATIONS")
        kind.set_expression_duration("STATIC_FLUENTS_IN_DURATIONS")
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
        time_scale = int(self.conf.get("time_scale", DEFAULT_TIME_SCALE))

        planner = ABAPlan(problem, time_scale=time_scale)
        plan = planner.plan(max_horizon=max_horizon, semantics=semantics)
        status = planner.status
        solved = status == PlanGenerationResultStatus.SOLVED_SATISFICING
        logs = [LogMessage(LogLevel.INFO, message) for message in planner.logs]
        return PlanGenerationResult(status, plan if solved else None, self.name, log_messages=logs)

    def destroy(self):
        pass
