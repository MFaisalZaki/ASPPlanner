"""Plan validation against a UP problem, shared by every backend planner."""

from typing import Optional, Tuple

from unified_planning.shortcuts import PlanValidator
from unified_planning.engines import ValidationResultStatus
from unified_planning.plans import TimeTriggeredPlan

from aspplanners.common.compilation import initialize_fluent_defaults


def as_validated(problem):
    """`problem` under the closed-world reading the encoders give it.

    A task that leaves fluents out of its initial state is encoded as though the
    missing ones were false / zero (see
    :func:`~aspplanners.common.compilation.initialize_fluent_defaults`). The
    validator has to be handed the same reading, or it is asked to evaluate a
    condition over a value that does not exist and calls the plan inapplicable.

    Returns `problem` itself when nothing is missing, so the common case does not
    pay for a clone. The clone is never handed back to the caller, so the plan's
    action instances still belong to the problem the user passed in -- which the
    UP validators accept, as they read the action off the instance.
    """
    if not (problem.kind.has_undefined_initial_numeric()
            or problem.kind.has_undefined_initial_symbolic()):
        return problem
    completed = problem.clone()
    initialize_fluent_defaults(completed)
    return completed


def validate_plan(problem, plan) -> Tuple[bool, Optional[str]]:
    """Validate `plan` against `problem` with the UP validator that fits it.

    Sequential plans go to ``sequential_plan_validator``; the schedules the
    temporal encoding produces go to ``up_time_triggered_validator``, which is
    the one that re-checks durations and over-all conditions.

    Returns ``(is_valid, reason)``; `reason` is None/empty when valid. A
    missing plan or problem is reported as invalid rather than raising.
    """
    if plan is None or problem is None:
        return False, "No plan or problem provided."
    name = ('up_time_triggered_validator' if isinstance(plan, TimeTriggeredPlan)
            else 'sequential_plan_validator')
    with PlanValidator(name=name) as validator:
        result = validator.validate(as_validated(problem), plan)
    return result.status == ValidationResultStatus.VALID, result.reason
