"""Plan validation against a UP problem, shared by every backend planner."""

from typing import Optional, Tuple

from unified_planning.shortcuts import PlanValidator
from unified_planning.engines import ValidationResultStatus
from unified_planning.plans import TimeTriggeredPlan


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
        result = validator.validate(problem, plan)
    return result.status == ValidationResultStatus.VALID, result.reason
