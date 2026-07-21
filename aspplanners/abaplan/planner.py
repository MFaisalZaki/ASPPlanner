"""ABAPlan: solve a planning task via its Assumption-Based Argumentation
encoding, deepening the horizon until a plan is found.

This is the sibling of :class:`aspplanners.plasp.planner.PLASPPlanner`. The
STRIPS-to-ABA reduction and framework construction live in
:class:`aspplanners.abaplan.encoder.ABAEncoder`; this class drives the search:
extend the framework one step per horizon, ask aspforaba for a stable extension
deriving the ``goal`` atom, then read the plan off that extension and map it
back onto the user's problem.

``aspforaba`` (the ABA solver) is an optional dependency: install it via the
``aba`` extra (``pip install -e ".[aba]"``). It is imported lazily so that
``import aspplanners`` works without it.
"""

from typing import List, Optional

import clingo

from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.plans.sequential_plan import SequentialPlan
from unified_planning.plans import ActionInstance

from aspplanners.common.validation import validate_plan
from aspplanners.abaplan.encoder import ABAEncoder


class ABAPlan:
    """STRIPS-to-ABA planner over an incrementally deepened horizon.

    The solve status of the last :meth:`plan` call is kept in ``self.status``
    (a ``PlanGenerationResultStatus``) and human-readable notes in ``self.logs``,
    mirroring :class:`~aspplanners.plasp.planner.PLASPPlanner`.
    """

    def __init__(self, problem):
        self.problem = problem
        self.encoder = ABAEncoder(problem)
        self.status: Optional[PlanGenerationResultStatus] = None
        self.logs: List[str] = []

    # ------------------------------------------------------------------ #
    # Solving.
    # ------------------------------------------------------------------ #
    def _solve_horizon(self, semantics):
        """Ask aspforaba whether ``goal`` is supported by a stable extension of
        the current framework; return that extension, or ``None`` if there is
        none.

        This reaches into aspforaba's lower-level API: build the framework,
        ground it, then query for a model in which the ``goal`` atom is
        supported under the chosen `semantics` (default stable, ``"ST"``).
        """
        from aspforaba import ABASolver

        formula = self.encoder.formula
        solver = ABASolver(assumptions=formula["assumptions"],
                           rules=formula["rules"],
                           contraries=formula["contraries"])
        # `goal` is absent from the framework until its rule body is derivable
        # at all (e.g. a goal fluent no action can establish); nothing to solve.
        if "goal" not in solver.abaf.atom_to_idx:
            return None

        solver._initialize_clingo(semantics)
        solver.ctl.ground([("base", [])], context=solver)

        goal_idx = solver.abaf.atom_to_idx["goal"]
        query = (clingo.Function("supported", [clingo.Function(f"a{goal_idx}")]), True)
        result = solver.ctl.solve(assumptions=[query], on_model=solver._record_model)
        if not result.satisfiable:
            return None
        return solver._interpret_extension(solver._last_model)

    def _extract_plan(self, extension) -> SequentialPlan:
        """Read the ordered action instances off a stable extension.

        The extension's assumptions are the action atoms plus the auxiliary
        assumptions; keep only those registered as actions, order them by step,
        and lift onto the user's original problem.
        """
        action_atoms = self.encoder.action_atoms
        steps = sorted(
            (action_atoms[a] for a in extension.assumptions if a in action_atoms),
            key=lambda step_action: step_action[1],
        )
        plan = SequentialPlan([ActionInstance(action) for action, _k in steps])
        return plan.replace_action_instances(self.encoder.map_back)

    def plan(self, max_horizon=1000, semantics="ST") -> SequentialPlan:
        """Search increasing horizons until a plan is found.

        Extends the ABA framework one step per horizon, and at each horizon
        asks the solver for a stable extension deriving the goal. Returns the
        plan mapped back onto the original problem (empty when none is found up
        to `max_horizon`); the outcome is also recorded in ``self.status``.
        """
        self.logs = []
        for k in range(max_horizon + 1):
            self.encoder.encode(k)
            extension = self._solve_horizon(semantics)
            if extension is None:
                continue

            plan = self._extract_plan(extension)
            is_valid, reason = validate_plan(self.problem, plan)
            if not is_valid:
                self.status = PlanGenerationResultStatus.INTERNAL_ERROR
                self.logs.append(f"Plan validation failed at horizon {k}: {reason}")
                return SequentialPlan([])
            self.status = PlanGenerationResultStatus.SOLVED_SATISFICING
            return plan

        self.status = PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY
        self.logs.append(
            f"No plan found up to horizon {max_horizon} "
            "(the task may be solvable with a longer horizon).")
        return SequentialPlan([])
