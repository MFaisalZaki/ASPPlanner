"""Shared UP compilation-pipeline helpers.

Both planners (the PLASP/ASP encoder and the ABA-framework encoder) put a UP
problem through a list of UP compilers and then have to lift an extracted plan
back onto the user's original problem. These two helpers capture exactly that
shared front-end; the encoding-specific translation lives in each backend.
"""

from fractions import Fraction
from itertools import chain
from typing import Callable, List, Optional, Sequence, Tuple

from unified_planning.shortcuts import Compiler, get_environment
from unified_planning.model import Problem
from unified_planning.model.fluent import get_all_fluent_exp
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


def initialize_fluent_defaults(task: Problem) -> None:
    """Give every fluent instance of `task` an initial value, in place.

    PDDL leaves most of the initial state implicit -- an unstated predicate is
    false, and the IPC benchmarks routinely declare a `(:functions ...)` entry
    for a whole cross product while `(:init ...)` only lists the tuples that are
    actually reachable (`travel-slow` in elevators-time, say). UP flags that as
    UNDEFINED_INITIAL_SYMBOLIC / UNDEFINED_INITIAL_NUMERIC; the value laid down
    here is PDDL's own closed-world reading of it, bool -> false and numeric -> 0.

    Both encoders need this before they read the initial state, and neither would
    fail loudly without it: `Problem.initial_values` silently *omits* the fluents
    with no value, so an uninitialized numeric fluent ends up with no holds/3
    chain (PLASP) and no value domain at step 0 (ABA), which quietly makes every
    numeric condition reading it unsatisfiable rather than raising.
    """
    _env = task.environment
    _tm, _em = _env.type_manager, _env.expression_manager
    # Use the task's own environment: with a non-global environment the global
    # one holds different type/expression manager instances.
    task.initial_defaults.update({_tm.RealType(): _em.Real(Fraction(0))})
    task.initial_defaults.update({_tm.IntType(): _em.Int(0)})
    task.initial_defaults.update({_tm.BoolType(): _em.Bool(False)})

    def _default(fluent_exp):
        """The value to lay down for a fluent instance with none of its own.

        The fluent's own declared default comes first; then the per-type
        defaults set above -- which a *bounded* type (`integer[0, 10]`) is not a
        key of, since it is a distinct type object from `IntType()`, so those
        fall through to a value derived from the type itself.
        """
        declared = task.fluents_defaults.get(fluent_exp.fluent())
        if declared is not None:
            return declared
        default = task.initial_defaults.get(fluent_exp.type)
        if default is not None:
            return default
        if fluent_exp.type.is_bool_type():
            return _em.Bool(False)
        if fluent_exp.type.is_int_type():
            return _em.Int(max(0, fluent_exp.type.lower_bound or 0))
        if fluent_exp.type.is_real_type():
            return _em.Real(Fraction(max(0, fluent_exp.type.lower_bound or 0)))
        raise NotImplementedError(
            f"Fluent {fluent_exp} has no initial value and no default for its type "
            f"{fluent_exp.type}; give it one in the problem.")

    initialized = set(task.explicit_initial_values.keys())
    for fluent_exp in chain.from_iterable(get_all_fluent_exp(task, f) for f in task.fluents):
        if fluent_exp not in initialized:
            task.set_initial_value(fluent_exp, _default(fluent_exp))


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
