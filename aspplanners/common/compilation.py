"""Shared UP compilation-pipeline helpers.

Both planners (the PLASP/ASP encoder and the ABA-framework encoder) put a UP
problem through a list of UP compilers and then have to lift an extracted plan
back onto the user's original problem. These two helpers capture exactly that
shared front-end; the encoding-specific translation lives in each backend.
"""

from typing import Callable, List, Optional, Tuple

from unified_planning.shortcuts import Compiler
from unified_planning.model import Problem
from unified_planning.plans import ActionInstance


def compose_map_backs(map_backs: List[Callable]) -> Callable[[ActionInstance], Optional[ActionInstance]]:
    """Chain the per-stage plan map-backs of a compilation pipeline.

    The extracted plan is stated on the LAST stage's problem, so the maps
    apply in reverse pipeline order; each lifts the action instance one stage
    closer to the user's original problem. A map returning ``None`` (the step
    has no counterpart upstream) short-circuits to ``None``.
    """
    def _map_back(action_instance: ActionInstance) -> Optional[ActionInstance]:
        for map_back in reversed(map_backs):
            action_instance = map_back(action_instance)
            if action_instance is None:
                return None
        return action_instance
    return _map_back


def run_compilers(problem: Problem, compilationlist: List[list]) -> Tuple[Problem, Callable]:
    """Run a UP compilation pipeline described as ``[[name, kind], ...]``.

    Returns the compiled problem and its single composed map-back (the one UP
    builds across the whole `Compiler` chain). Callers that also do their own
    pre/post stages collect this alongside those and feed everything to
    :func:`compose_map_backs`.
    """
    compiler_names = [c[0] for c in compilationlist]
    compiler_kinds = [c[1] for c in compilationlist]
    with Compiler(names=compiler_names, compilation_kinds=compiler_kinds) as compiler:
        result = compiler.compile(problem)
    return result.problem, result.map_back_action_instance
