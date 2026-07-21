"""
tim_typer.py
============

A Unified Planning ``Compiler`` that performs TIM-style type inference:
turns an untyped (or coarsely-typed) ``Problem`` into a ``Problem`` whose
fluents, action parameters, and objects carry inferred types.

Algorithm
---------

Property-space partitioning over Fast Downward reachability: the problem
is grounded with FD's reachability grounder, objects are clustered by the
*equal set* of (fluent, argument-position) pairs they inhabit across
reachable atoms (true initial atoms, grounded action pre/effects, goals),
and each cluster becomes one ``UserType``. Positions and action parameters
that a single cluster inhabits get that cluster's type; ambiguous ones get
the root type, so the typing only ever excludes bindings that reachability
already proved impossible.

When reachability grounding is unavailable (e.g. numeric tasks), the
problem is returned unchanged: without reachability information a
co-occurrence analysis cannot soundly exclude any binding.

The plan back-mapping is the identity on action and object names — the
type system changes, the action structure does not.

Limitations
-----------

  * ``InstantaneousAction`` only (no durative actions, no quantifiers).
  * No subtype lattice beyond one root + leaf clusters.

Usage
-----

    from unified_planning.io import PDDLReader, PDDLWriter
    from tim_typer import TIMTypeInferenceCompiler

    problem = PDDLReader().parse_problem("domain.pddl", "problem.pddl")
    result  = TIMTypeInferenceCompiler().compile(problem)
    typed   = result.problem

    PDDLWriter(typed).write_domain("domain-typed.pddl")
    PDDLWriter(typed).write_problem("problem-typed.pddl")

    # Plan back-conversion for any plan found on `typed`:
    untyped_plan = plan.replace_action_instances(result.map_back_action_instance)
"""

from __future__ import annotations

import logging
import sys
from collections import OrderedDict, defaultdict
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import unified_planning as up
from unified_planning.engines.engine import Engine
from unified_planning.engines.results import CompilerResult
from unified_planning.exceptions import UPTypeError
from unified_planning.io import PDDLReader, PDDLWriter
from unified_planning.model import (
    Fluent,
    InstantaneousAction,
    Object,
    Parameter,
    Problem,
    ProblemKind,
)
from unified_planning.plans import ActionInstance
from unified_planning.shortcuts import UserType, Compiler, CompilationKind

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers: collect & rebuild expressions
# ---------------------------------------------------------------------------

def _collect(expr,
             param_pos: Dict[str, Set[Tuple[str, int]]],
             object_pos: Dict[str, Set[Tuple[str, int]]],
             eq_groups: List[List[Tuple[str, str]]]) -> None:
    if expr is None:
        return

    if expr.is_fluent_exp():
        fname = expr.fluent().name
        for i, arg in enumerate(expr.args):
            key = (fname, i)
            if arg.is_parameter_exp():
                param_pos[arg.parameter().name].add(key)
            elif arg.is_object_exp():
                object_pos[arg.object().name].add(key)

    if expr.is_equals():
        sides: List[Tuple[str, str]] = []
        for arg in expr.args:
            if arg.is_parameter_exp():
                sides.append(('param', arg.parameter().name))
            elif arg.is_object_exp():
                sides.append(('object', arg.object().name))
        if len(sides) >= 2:
            eq_groups.append(sides)

    for child in expr.args:
        _collect(child, param_pos, object_pos, eq_groups)


def _rebuild(expr, em, param_map: Dict[str, Parameter],
             new_fluents: Dict[str, Fluent],
             new_objects: Dict[str, Object]):
    if expr.is_fluent_exp():
        nf = new_fluents[expr.fluent().name]
        new_args = [_rebuild(a, em, param_map, new_fluents, new_objects)
                    for a in expr.args]
        return em.FluentExp(nf, tuple(new_args))
    if expr.is_parameter_exp():
        return em.ParameterExp(param_map[expr.parameter().name])
    if expr.is_object_exp():
        return em.ObjectExp(new_objects[expr.object().name])
    if expr.is_bool_constant():
        return em.Bool(expr.bool_constant_value())
    if expr.is_int_constant():
        return em.Int(expr.int_constant_value())
    if expr.is_real_constant():
        return em.Real(expr.real_constant_value())

    new_args = [_rebuild(a, em, param_map, new_fluents, new_objects)
                for a in expr.args]

    if expr.is_and():     return em.And(*new_args)
    if expr.is_or():      return em.Or(*new_args)
    if expr.is_not():     return em.Not(new_args[0])
    if expr.is_implies(): return em.Implies(new_args[0], new_args[1])
    if expr.is_iff():     return em.Iff(new_args[0], new_args[1])
    if expr.is_equals():  return em.Equals(new_args[0], new_args[1])
    if expr.is_lt():      return em.LT(new_args[0], new_args[1])
    if expr.is_le():      return em.LE(new_args[0], new_args[1])
    if expr.is_plus():    return em.Plus(*new_args)
    if expr.is_minus():   return em.Minus(*new_args)
    if expr.is_times():   return em.Times(*new_args)
    if expr.is_div():     return em.Div(*new_args)

    raise NotImplementedError(f"Unhandled FNode kind: {expr.node_type}")


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------

class TIMTypeInferenceCompiler(Engine):
    """A flat-TIM type inference compiler for UP problems.

    No standard ``CompilationKind`` corresponds to type inference, so this
    compiler is invoked directly via :meth:`compile` rather than through the
    ``OneshotPlanner`` / ``Compiler`` factory machinery. The returned
    :class:`CompilerResult` is fully compatible with the rest of UP, including
    plan back-conversion.
    """

    def __init__(self) -> None:
        Engine.__init__(self)

    @property
    def name(self) -> str:
        return "tim_type_inferer"

    @staticmethod
    def supported_kind() -> ProblemKind:
        kind = ProblemKind()
        kind.set_problem_class("ACTION_BASED")
        kind.set_typing("FLAT_TYPING")
        kind.set_conditions_kind("NEGATIVE_CONDITIONS")
        kind.set_conditions_kind("DISJUNCTIVE_CONDITIONS")
        kind.set_conditions_kind("EQUALITIES")
        kind.set_effects_kind("CONDITIONAL_EFFECTS")
        kind.set_effects_kind("INCREASE_EFFECTS")
        kind.set_effects_kind("DECREASE_EFFECTS")
        return kind

    @staticmethod
    def supports(problem_kind: ProblemKind) -> bool:
        return problem_kind <= TIMTypeInferenceCompiler.supported_kind()

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def compile(self, problem: Problem) -> CompilerResult:
        if not isinstance(problem, Problem):
            raise NotImplementedError(
                "TIMTypeInferenceCompiler only supports classical Problem "
                "instances."
            )

        # Property-space partitioning via FD reachability: groups objects by
        # the *equal set* of (fluent, position) pairs they actually inhabit
        # across reachable atoms.
        grounded_problem = None
        try:
            grounded_problem = self._ground_problem(problem).problem
        except Exception as e:
            _LOGGER.info(
                "Reachability grounding unavailable (%s: %s); skipping type "
                "inference.", type(e).__name__, e)

        if grounded_problem is None:
            # Without reachability information, inference is either useless
            # (walking materialized closed-world defaults puts every object
            # in every position, collapsing to one type) or unsound (a
            # co-occurrence analysis cannot prove an object is never needed
            # in a position). Leave the problem untyped.
            return CompilerResult(problem, lambda ai: ai, self.name)

        new_problem = self._infer_and_build_from_reachability(
            problem, grounded_problem)

        original_actions = {a.name: a for a in problem.actions}
        original_objects = {o.name: o for o in problem.all_objects}
        em = problem.environment.expression_manager

        def map_back(ai: ActionInstance) -> Optional[ActionInstance]:
            orig_action = original_actions.get(ai.action.name)
            if orig_action is None:
                return None
            orig_params = tuple(
                em.ObjectExp(original_objects[p.object().name])
                for p in ai.actual_parameters
            )
            return ActionInstance(orig_action, orig_params)

        return CompilerResult(new_problem, map_back, self.name)

    def _ground_problem(self, problem):
        compilationlist = []
        compilationlist += [["up_quantifiers_remover", CompilationKind.QUANTIFIERS_REMOVING]]
        compilationlist += [["up_disjunctive_conditions_remover", CompilationKind.DISJUNCTIVE_CONDITIONS_REMOVING]]
        compilationlist += [["fast-downward-reachability-grounder", CompilationKind.GROUNDING]]
        compiler_names = [c[0] for c in compilationlist]
        compiler_kinds = [c[1] for c in compilationlist]
        with Compiler(names=compiler_names, compilation_kinds=compiler_kinds) as compiler:
            grounded_problem = compiler.compile(problem)
        return grounded_problem

    # -----------------------------------------------------------------------
    # Reachability-based property-space partitioning
    # -----------------------------------------------------------------------

    @staticmethod
    def _collect_reachable_atoms(grounded: Problem) -> Set[Tuple[str, Tuple[str, ...]]]:
        """Walk a grounded UP Problem and return every (fluent_name, (obj_names))
        ground atom mentioned in the initial state, action pre/effects, or goals."""
        reachable: Set[Tuple[str, Tuple[str, ...]]] = set()

        def walk(expr) -> None:
            if expr is None:
                return
            if expr.is_fluent_exp():
                names: List[str] = []
                ground = True
                for a in expr.args:
                    if a.is_object_exp():
                        names.append(a.object().name)
                    else:
                        ground = False
                        break
                if ground and len(names) == len(expr.args):
                    reachable.add((expr.fluent().name, tuple(names)))
            for child in expr.args:
                walk(child)

        # Walk only atoms that are actually TRUE initially: Problem.initial_values
        # materializes closed-world defaults for every object x fluent
        # combination, which would put every object in every position and
        # collapse the partition to a single cluster.
        for fnode, value in grounded.explicit_initial_values.items():
            if not value.is_false():
                walk(fnode)
        for action in grounded.actions:
            for pre in action.preconditions:
                walk(pre)
            for eff in action.effects:
                walk(eff.fluent)
                walk(eff.value)
                if eff.condition is not None:
                    walk(eff.condition)
        for goal in grounded.goals:
            walk(goal)

        return reachable

    def _infer_and_build_from_reachability(self, problem: Problem,
                                           grounded: Problem) -> Problem:
        reachable = self._collect_reachable_atoms(grounded)

        # Per-object property space: positions where this object appears in
        # any reachable atom.
        raw_props: Dict[str, Set[Tuple[str, int]]] = defaultdict(set)
        for fname, args in reachable:
            for i, oname in enumerate(args):
                raw_props[oname].add((fname, i))

        # Cluster objects by EQUAL property space — Mystery splits cleanly here
        # because pleasure-people never reach (pain, 0) and vice versa, even
        # though both share (craves, 0).
        cluster_for_props: Dict[FrozenSet[Tuple[str, int]], str] = {}
        obj_cluster: Dict[str, str] = {}
        for obj in problem.all_objects:
            props = frozenset(raw_props.get(obj.name, set()))
            if props not in cluster_for_props:
                hint = (sorted(props)[0][0].lower().replace('-', '_')
                        if props else "isolated")
                cluster_for_props[props] = f"t{len(cluster_for_props) + 1}_{hint}"
            obj_cluster[obj.name] = cluster_for_props[props]

        # Per-(fluent, position) cluster set: which clusters' members appear here?
        pos_clusters: Dict[Tuple[str, int], Set[str]] = defaultdict(set)
        for fname, args in reachable:
            for i, oname in enumerate(args):
                pos_clusters[(fname, i)].add(obj_cluster[oname])

        # UserType lattice: a single root + one leaf per object cluster as
        # children of the root. UP supports only one parent per UserType; we
        # don't attempt finer supertypes here — multi-cluster positions get
        # the root, single-cluster positions get the leaf.
        ROOT = "t_default"
        type_objs: Dict[str, "up.model.types.Type"] = {ROOT: UserType(ROOT)}
        for cname in set(cluster_for_props.values()):
            type_objs[cname] = UserType(cname, father=type_objs[ROOT])

        def position_type(pos: Tuple[str, int]) -> str:
            cs = pos_clusters.get(pos)
            if not cs:
                return ROOT
            return next(iter(cs)) if len(cs) == 1 else ROOT

        # Action parameter types: intersection of position cluster sets across
        # every position the parameter occurs in. If the intersection narrows
        # to a single cluster, that's the tight type; otherwise fall back to
        # the root.
        action_param_pos: Dict[str, Dict[str, Set[Tuple[str, int]]]] = {}
        for action in problem.actions:
            if not isinstance(action, InstantaneousAction):
                raise NotImplementedError(
                    f"Only InstantaneousAction supported; got "
                    f"{type(action).__name__}."
                )
            param_pos: Dict[str, Set[Tuple[str, int]]] = defaultdict(set)
            sink_obj_pos: Dict[str, Set[Tuple[str, int]]] = defaultdict(set)
            eq_groups: List[List[Tuple[str, str]]] = []
            for pre in action.preconditions:
                _collect(pre, param_pos, sink_obj_pos, eq_groups)
            for eff in action.effects:
                _collect(eff.fluent, param_pos, sink_obj_pos, eq_groups)
                _collect(eff.value, param_pos, sink_obj_pos, eq_groups)
                if eff.condition is not None:
                    _collect(eff.condition, param_pos, sink_obj_pos, eq_groups)
            action_param_pos[action.name] = dict(param_pos)

        def param_type(action_name: str, param_name: str) -> str:
            positions = action_param_pos.get(action_name, {}).get(param_name, set())
            if not positions:
                return ROOT
            inter: Optional[Set[str]] = None
            for pos in positions:
                cs = pos_clusters.get(pos, set())
                inter = set(cs) if inter is None else (inter & cs)
                if not inter:
                    break
            if inter and len(inter) == 1:
                return next(iter(inter))
            return ROOT

        # ---- rebuild new typed problem ----
        new = Problem(problem.name)
        em = new.environment.expression_manager

        new_fluents: Dict[str, Fluent] = {}
        for fluent in problem.fluents:
            sig: "OrderedDict[str, up.model.types.Type]" = OrderedDict()
            for i, p in enumerate(fluent.signature):
                sig[p.name] = type_objs[position_type((fluent.name, i))]
            nf = Fluent(fluent.name, fluent.type, sig)
            new_fluents[fluent.name] = nf
            default = problem.fluents_defaults.get(fluent)
            if default is not None:
                new.add_fluent(nf, default_initial_value=default)
            else:
                new.add_fluent(nf)

        new_objects: Dict[str, Object] = {}
        for obj in problem.all_objects:
            nobj = Object(obj.name, type_objs[obj_cluster[obj.name]])
            new_objects[obj.name] = nobj
            new.add_object(nobj)

        for action in problem.actions:
            params: "OrderedDict[str, up.model.types.Type]" = OrderedDict()
            for p in action.parameters:
                params[p.name] = type_objs[param_type(action.name, p.name)]
            new_action = InstantaneousAction(action.name, params)
            param_map = {p.name: new_action.parameter(p.name)
                         for p in action.parameters}

            for pre in action.preconditions:
                new_action.add_precondition(
                    _rebuild(pre, em, param_map, new_fluents, new_objects))

            for eff in action.effects:
                f = _rebuild(eff.fluent, em, param_map, new_fluents, new_objects)
                v = _rebuild(eff.value, em, param_map, new_fluents, new_objects)
                cond = (em.TRUE() if eff.condition is None
                        else _rebuild(eff.condition, em, param_map,
                                      new_fluents, new_objects))
                if eff.is_increase():
                    new_action.add_increase_effect(f, v, cond)
                elif eff.is_decrease():
                    new_action.add_decrease_effect(f, v, cond)
                else:
                    new_action.add_effect(f, v, cond)

            new.add_action(new_action)

        empty: Dict[str, Parameter] = {}
        for fnode, value in problem.explicit_initial_values.items():
            try:
                new.set_initial_value(
                    _rebuild(fnode, em, empty, new_fluents, new_objects),
                    _rebuild(value, em, empty, new_fluents, new_objects),
                )
            except UPTypeError:
                # The inferred position types exclude this ground instance
                # (its objects never inhabit the fluent's positions in any
                # reachable atom). That can only be true of a default-valued
                # entry -- a TRUE atom would have been walked as reachable
                # and widened the position type -- so dropping it preserves
                # the closed-world semantics.
                if not value.is_false():
                    raise

        for goal in problem.goals:
            new.add_goal(_rebuild(goal, em, empty, new_fluents, new_objects))

        return new

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> int:
    if len(sys.argv) != 5:
        print("Usage: python tim_typer.py "
              "<in_domain.pddl> <in_problem.pddl> "
              "<out_domain.pddl> <out_problem.pddl>", file=sys.stderr)
        return 2

    in_dom, in_prob, out_dom, out_prob = sys.argv[1:5]

    problem = PDDLReader().parse_problem(in_dom, in_prob)
    result  = TIMTypeInferenceCompiler().compile(problem)

    writer = PDDLWriter(result.problem)
    writer.write_domain(out_dom)
    writer.write_problem(out_prob)

    n_types = len(result.problem.user_types)
    print(f"[{result.engine_name}] inferred {n_types} type(s); "
          f"wrote {out_dom} and {out_prob}.")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
