"""Plan validation against a UP problem, shared by every backend planner."""

from typing import Optional, Tuple

from unified_planning.shortcuts import PlanValidator
from unified_planning.engines import ValidationResultStatus


def validate_plan(problem, plan) -> Tuple[bool, Optional[str]]:
    """Validate `plan` against `problem` with UP's sequential plan validator.

    Returns ``(is_valid, reason)``; `reason` is None/empty when valid. A
    missing plan or problem is reported as invalid rather than raising.
    """
    if plan is None or problem is None:
        return False, "No plan or problem provided."
    with PlanValidator(name='sequential_plan_validator') as validator:
        result = validator.validate(problem, plan)
    return result.status == ValidationResultStatus.VALID, result.reason
