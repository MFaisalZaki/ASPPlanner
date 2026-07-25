"""Shared UP compilation-pipeline helpers.

Both planners (the PLASP/ASP encoder and the ABA-framework encoder) put a UP
problem through a list of UP compilers and then have to lift an extracted plan
back onto the user's original problem. These two helpers capture exactly that
shared front-end; the encoding-specific translation lives in each backend.
"""

from typing import Callable, List, Optional, Sequence, Tuple

from unified_planning.shortcuts import Compiler, get_environment
from unified_planning.model import Problem
from unified_planning.plans import ActionInstance


# Grounders that prune by reachability analysis, in the order we prefer them.
# These are the only ones worth running ahead of an ASP encoding: gringo grounds
# the task anyway, so a grounder that merely enumerates type-consistent bindings
# does the same work twice and hands gringo a bigger program than the lifted
# encoding would have produced -- the action signature rules bind parameters via
# has(_, type(...)) and fold static preconditions into the body, which prunes
# what a non-reachability grounder leaves in.
REACHABILITY_GROUNDERS = (
    "fast-downward-reachability-grounder",
)

# Every installed grounder, most-pruning first. For a backend that has no lifted
# path at all, a grounder that only enumerates still beats not grounding.
GROUNDERS = REACHABILITY_GROUNDERS + (
    "fast-downward-grounder",
    "up_grounder",
)


def select_grounder(problem_kind, candidates: Sequence[str] = GROUNDERS) -> Optional[str]:
    """The first grounder in `candidates` that supports `problem_kind`, or None.

    Which grounder copes with numeric fluents or with durative actions is a fact
    about the installed engines, so it is asked of them rather than restated
    here -- a task that gains a feature (or an installation that gains an
    engine) then moves to the right grounder on its own. Whether a task *should*
    be grounded at all is a separate, deliberate choice each backend makes; pass
    `REACHABILITY_GROUNDERS` to ask only for grounders that pay their way ahead
    of an ASP encoding.
    """
    installed = set(get_environment().factory.engines)
    for name in candidates:
        if name not in installed:
            continue
        with Compiler(name=name) as compiler:
            if compiler.supports(problem_kind):
                return name
    return None


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
