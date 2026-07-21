"""ABAPlan: encode a classical planning task as an Assumption-Based
Argumentation (ABA) framework and solve it with aspforaba.

This is the sibling of :class:`aspplanner.plasp.planner.ASPPlanner`. It follows
Fan's STRIPS-to-ABA reduction: actions become assumptions, frame axioms are
defaults defeated by effects, and a plan exists at horizon ``K`` iff there is a
stable extension deriving the ``goal`` atom. It shares the UP front-end with the
PLASP backend (:mod:`aspplanner.common`) -- compilation pipeline, plan map-back,
validation -- but has its own encoding, solver, and plan extraction.

``aspforaba`` (the ABA solver) is an optional dependency: install it via the
``aba`` extra (``pip install -e ".[aba]"``). It is imported lazily so that
``import aspplanner`` works without it.
"""

from collections import defaultdict
from typing import List, Optional

import clingo

from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.shortcuts import CompilationKind
from unified_planning.plans.sequential_plan import SequentialPlan
from unified_planning.plans import ActionInstance

from aspplanner.common.compilation import run_compilers
from aspplanner.common.validation import validate_plan


# The UP compilation pipeline ABAPlan grounds every task through. Unlike the
# PLASP backend there is no numeric path: the STRIPS-to-ABA reduction always
# grounds via Fast Downward's reachability grounder.
_COMPILATION_LIST = [
    ["up_quantifiers_remover", CompilationKind.QUANTIFIERS_REMOVING],
    ["up_negative_conditions_remover", CompilationKind.NEGATIVE_CONDITIONS_REMOVING],
    ["up_disjunctive_conditions_remover", CompilationKind.DISJUNCTIVE_CONDITIONS_REMOVING],
    ["fast-downward-reachability-grounder", CompilationKind.GROUNDING],
]


class ABAPlan:
    """STRIPS-to-ABA planner over an incrementally deepened horizon.

    The solve status of the last :meth:`plan` call is kept in ``self.status``
    (a ``PlanGenerationResultStatus``) and human-readable notes in ``self.logs``,
    mirroring :class:`~aspplanner.plasp.planner.ASPPlanner`.
    """

    def __init__(self, problem):
        self.problem = problem
        # Shared front-end: run the UP compilation pipeline and keep the
        # composed map-back that lifts a plan onto the user's original problem.
        self.grounded_problem, self._map_back = run_compilers(problem, _COMPILATION_LIST)

        # The ABA framework, built incrementally across horizons.
        self._formula = defaultdict(list)   # {"rules"|"assumptions"|"contraries": [...]}
        self._assumptions = set()           # dedup guard for self._formula["assumptions"]
        # Maps each action-assumption atom to the (grounded action, step) it
        # stands for, so plan extraction is a dict lookup rather than string
        # surgery on atom names.
        self._action_atoms = {}

        self.status: Optional[PlanGenerationResultStatus] = None
        self.logs: List[str] = []

        self._extract_task()

    # ------------------------------------------------------------------ #
    # Task extraction: classify fluents, collect actions, init, goals.
    # ------------------------------------------------------------------ #
    def _extract_task(self):
        gp = self.grounded_problem

        # A fluent is "flexible" iff some action effect mentions it; the rest
        # are rigid (timeless) facts.
        self._flexible_names = {
            e.fluent.fluent().name for a in gp.actions for e in a.effects
        }

        # Initial state: flexible atoms are time-indexed (laid down at step 0),
        # rigid atoms are timeless. Also collect every flexible instance ever
        # mentioned so a frame axiom can be emitted for it at each step.
        self._init_true_flexible = []
        self._rigid_facts = []
        self._all_flexible_instances = set()
        for fnode, value in gp.initial_values.items():
            atom = (fnode.fluent().name, tuple(str(a) for a in fnode.args))
            is_true = str(value) == "true"
            if atom[0] in self._flexible_names:
                self._all_flexible_instances.add(atom)
                if is_true:
                    self._init_true_flexible.append(atom)
            elif is_true:
                self._rigid_facts.append(atom)

        # Actions (already ground, so action.parameters is empty).
        self._actions = []
        for action in gp.actions:
            preconds = []
            for p in action.preconditions:
                if not p.is_fluent_exp():
                    raise NotImplementedError(
                        f"Non-fluent precondition not supported in STRIPS encoding: {p}"
                    )
                atom = (p.fluent().name, tuple(str(a) for a in p.args))
                preconds.append(atom)
                if atom[0] in self._flexible_names:
                    self._all_flexible_instances.add(atom)
            pos_eff, neg_eff = [], []
            for e in action.effects:
                atom = (e.fluent.fluent().name, tuple(str(a) for a in e.fluent.args))
                self._all_flexible_instances.add(atom)
                (pos_eff if str(e.value) == "true" else neg_eff).append(atom)
            self._actions.append({
                "name": action.name,
                "up_action": action,
                "preconditions": preconds,
                "pos_effects": pos_eff,
                "neg_effects": neg_eff,
            })

        # Goal: a (possibly trivial) conjunction of fluent atoms.
        self._goal_literals = []
        for g in gp.goals:
            self._collect_goal(g, self._goal_literals)
        for atom in self._goal_literals:
            if atom[0] in self._flexible_names:
                self._all_flexible_instances.add(atom)

    def _collect_goal(self, expr, out):
        if expr.is_and():
            for arg in expr.args:
                self._collect_goal(arg, out)
        elif expr.is_fluent_exp():
            out.append((expr.fluent().name, tuple(str(a) for a in expr.args)))
        else:
            raise NotImplementedError(f"Unsupported goal expr: {expr}")

    # ------------------------------------------------------------------ #
    # Atom names. Atoms in aspforaba are opaque strings; they only need to be
    # unique and stable. A trailing `_k` tags the timestep.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _f(fname, args, k=None):
        s = "_".join((fname,) + tuple(args))
        return s if k is None else f"{s}_{k}"

    @classmethod
    def _neg(cls, atom, k):
        return "neg_" + cls._f(*atom, k)

    @classmethod
    def _not(cls, atom, k):
        return "not_" + cls._f(*atom, k)

    @classmethod
    def _df(cls, atom, k):
        return "df_" + cls._f(*atom, k)

    @staticmethod
    def _act(name, k):
        return f"{name}_{k}"

    @staticmethod
    def _ae(name, k):
        return f"ae_{name}_{k}"

    @staticmethod
    def _ha(k):
        return f"ha_{k}"

    # ------------------------------------------------------------------ #
    # Framework mutators.
    # ------------------------------------------------------------------ #
    def _add_rule(self, head, body):
        self._formula["rules"].append((head, list(body)))

    def _add_assumption(self, atom):
        if atom not in self._assumptions:
            self._assumptions.add(atom)
            self._formula["assumptions"].append(atom)

    def _add_contrary(self, assumption, contrary):
        self._formula["contraries"].append((assumption, contrary))

    # ------------------------------------------------------------------ #
    # Encoding stages.
    # ------------------------------------------------------------------ #
    def _encode_base(self):
        """Time-independent layer: rigid facts + initial flexible state at k=0."""
        for atom in self._rigid_facts:
            self._add_rule(self._f(*atom), [])
        for atom in self._init_true_flexible:
            self._add_rule(self._f(*atom, 0), [])

    def _encode_transition(self, k):
        """Add the rules/assumptions/contraries carrying state at step k to
        step k+1 (frame axioms included) via the actions firing at step k."""
        action_atoms = [self._act(a["name"], k) for a in self._actions]

        for action in self._actions:
            a_atom = self._act(action["name"], k)
            ae_atom = self._ae(action["name"], k + 1)
            ha_atom = self._ha(k)

            # Firing the action -> its effects are applied (ae) and "some action
            # happened at step k" (ha, which gates the frame axioms below).
            self._add_rule(ae_atom, [a_atom])
            self._add_rule(ha_atom, [a_atom])
            for atom in action["pos_effects"]:
                self._add_rule(self._f(*atom, k + 1), [ae_atom])
            for atom in action["neg_effects"]:
                self._add_rule(self._neg(atom, k + 1), [ae_atom])

            # The action atom is an assumption; remember what it denotes so the
            # plan can be read straight off the stable extension.
            self._add_assumption(a_atom)
            self._action_atoms[a_atom] = (action["up_action"], k)

            # not_p_k: "precondition p does not hold at step k" (assumption),
            # contrary is p holding. Shared across actions sharing precondition p.
            for atom in action["preconditions"]:
                not_atom = self._not(atom, k)
                if atom[0] in self._flexible_names:
                    contrary = self._f(*atom, k)
                else:
                    # Rigid precondition: its contrary is the timeless fact.
                    # (The reachability grounder usually strips these, but we
                    # still handle them for robustness.)
                    contrary = self._f(*atom)
                if not_atom not in self._assumptions:
                    self._add_assumption(not_atom)
                    self._add_contrary(not_atom, contrary)

            # bar(a_k) = {other actions at step k} U {not_p_k for each pre p}:
            # only one action fires per step, and every precondition must hold.
            for other in action_atoms:
                if other != a_atom:
                    self._add_contrary(a_atom, other)
            for atom in action["preconditions"]:
                self._add_contrary(a_atom, self._not(atom, k))

        # Frame axioms: a flexible instance holds at step k+1 if it held at
        # step k, some action happened, and it was not defeated by a negative
        # effect (the default assumption df, contrary neg_*).
        for atom in self._all_flexible_instances:
            df_atom = self._df(atom, k + 1)
            self._add_rule(self._f(*atom, k + 1),
                           [self._f(*atom, k), self._ha(k), df_atom])
            self._add_assumption(df_atom)
            self._add_contrary(df_atom, self._neg(atom, k + 1))

    def _set_goal(self, k):
        """(Re)write the single `goal <- ...` rule to point at horizon k."""
        self._formula["rules"] = [(h, b) for (h, b) in self._formula["rules"] if h != "goal"]
        body = [self._f(*atom, k) if atom[0] in self._flexible_names else self._f(*atom)
                for atom in self._goal_literals]
        self._add_rule("goal", body)

    def _encode(self, k):
        """Extend the framework in place so it covers horizon ``k``."""
        if k == 0:
            if not self._formula["rules"]:
                self._encode_base()
        else:
            self._encode_transition(k - 1)
        self._set_goal(k)

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

        solver = ABASolver(assumptions=self._formula["assumptions"],
                            rules=self._formula["rules"],
                            contraries=self._formula["contraries"])
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
        ``not_*``/``df_*`` assumptions; keep only those registered as actions,
        order them by step, and lift onto the user's original problem.
        """
        steps = sorted(
            (self._action_atoms[a] for a in extension.assumptions if a in self._action_atoms),
            key=lambda step_action: step_action[1],
        )
        plan = SequentialPlan([ActionInstance(action) for action, _k in steps])
        return plan.replace_action_instances(self._map_back)

    def plan(self, max_horizon=1000, semantics="ST") -> SequentialPlan:
        """Search increasing horizons until a plan is found.

        Extends the ABA framework one step per horizon, and at each horizon
        asks the solver for a stable extension deriving the goal. Returns the
        plan mapped back onto the original problem (empty when none is found up
        to `max_horizon`); the outcome is also recorded in ``self.status``.
        """
        self.logs = []
        for k in range(max_horizon + 1):
            self._encode(k)
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
