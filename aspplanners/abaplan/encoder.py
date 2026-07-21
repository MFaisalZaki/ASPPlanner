"""The ABA encoder: turn a compiled UP task into an Assumption-Based
Argumentation framework, built incrementally per horizon.

This is the sibling of :class:`aspplanners.plasp.encoder.PLASPEncoder`. It runs
the shared UP compilation pipeline (:mod:`aspplanners.common`), extracts the
STRIPS task, and -- via :meth:`ABAEncoder.encode` -- extends the framework
(assumptions / rules / contraries) one step at a time following Fan's
STRIPS-to-ABA reduction. The search loop that drives it and reads plans off the
solver lives in :class:`aspplanners.abaplan.planner.ABAPlan`.

Simple numeric planning is supported by *finite-domain propositionalisation*:
aspforaba reasons over propositional atoms only, so each numeric fluent's value
at each step is tracked by ``val_...`` atoms over a per-step bounded domain, and
arithmetic effects/comparisons become rules over those atoms. Only *single-fluent*
numeric expressions are supported (a fluent compared to a constant, or modified
by a constant): coupling several fluents (``f + g``, ``increase(f, g)``) would
enumerate the product of their domains, which propositionalises far too
expensively, so those are rejected. The PLASP backend handles such expressions
with native ``#sum`` arithmetic instead.
"""

from collections import defaultdict
from itertools import product

from unified_planning.shortcuts import CompilationKind, EffectKind

from aspplanners.common.compilation import run_compilers


# UP compilers run before the reduction. The grounder differs by task: the FD
# reachability grounder prunes classical tasks well but rejects numeric ones,
# so numeric tasks use UP's generic grounder (which preserves int fluents and
# increase/decrease/assign effects).
_PRE_COMPILERS = [
    ["up_quantifiers_remover", CompilationKind.QUANTIFIERS_REMOVING],
    ["up_negative_conditions_remover", CompilationKind.NEGATIVE_CONDITIONS_REMOVING],
    ["up_disjunctive_conditions_remover", CompilationKind.DISJUNCTIVE_CONDITIONS_REMOVING],
]


def _compilation_list(problem):
    kind = problem.kind
    numeric = kind.has_int_fluents() or kind.has_real_fluents()
    grounder = "up_grounder" if numeric else "fast-downward-reachability-grounder"
    return _PRE_COMPILERS + [[grounder, CompilationKind.GROUNDING]]


class ABAEncoder:
    """Builds the ABA framework for a task, extended one horizon at a time.

    After construction, :meth:`encode` grows the framework in place; the driver
    reads :attr:`formula` (assumptions/rules/contraries), :attr:`action_atoms`
    (atom -> (grounded action, step)) and :attr:`map_back` off it.
    """

    def __init__(self, problem):
        self.problem = problem
        # Shared front-end: run the UP compilation pipeline and keep the
        # composed map-back that lifts a plan onto the user's original problem.
        self.grounded_problem, self.map_back = run_compilers(problem, _compilation_list(problem))

        # The ABA framework, built incrementally across horizons.
        self.formula = defaultdict(list)    # {"rules"|"assumptions"|"contraries": [...]}
        self._assumptions = set()           # dedup guard for formula["assumptions"]
        # Maps each action-assumption atom to the (grounded action, step) it
        # stands for, so plan extraction is a dict lookup rather than string
        # surgery on atom names.
        self.action_atoms = {}
        # Numeric fluent value domain per step: {k: {fkey: set(int)}}.
        self._domain = {}
        # str(comparison FNode) -> short stable id, so comparison atoms are safe
        # aspforaba identifiers.
        self._cmp_ids = {}

        self._extract_task()

    # ------------------------------------------------------------------ #
    # Task extraction: classify fluents, collect actions, init, goals.
    # ------------------------------------------------------------------ #
    def _extract_task(self):
        gp = self.grounded_problem

        self._numeric_names = {
            f.name for f in gp.fluents if f.type.is_int_type() or f.type.is_real_type()
        }
        # A boolean fluent is "flexible" iff some action effect mentions it; the
        # rest are rigid (timeless) facts. Numeric fluents are handled separately.
        self._flexible_names = {
            e.fluent.fluent().name for a in gp.actions for e in a.effects
            if e.fluent.fluent().name not in self._numeric_names
        }

        # Initial state: flexible boolean atoms are laid down at step 0, rigid
        # atoms are timeless, numeric fluents seed their value domain at step 0.
        self._init_true_flexible = []
        self._rigid_facts = []
        self._all_flexible_instances = set()
        self._num_init = {}
        for fnode, value in gp.initial_values.items():
            name = fnode.fluent().name
            args = tuple(str(a) for a in fnode.args)
            if name in self._numeric_names:
                self._num_init[(name, args)] = self._as_int(value)
                continue
            atom = (name, args)
            if name in self._flexible_names:
                self._all_flexible_instances.add(atom)
                if str(value) == "true":
                    self._init_true_flexible.append(atom)
            elif str(value) == "true":
                self._rigid_facts.append(atom)

        # Actions (already ground, so action.parameters is empty). Preconditions
        # split into boolean-fluent atoms and numeric comparisons; effects into
        # boolean add/delete and numeric increase/decrease/assign.
        self._actions = []
        for action in gp.actions:
            bool_pre, num_pre = [], []
            for p in action.preconditions:
                if p.is_fluent_exp() and p.fluent().type.is_bool_type():
                    atom = (p.fluent().name, tuple(str(a) for a in p.args))
                    bool_pre.append(atom)
                    if atom[0] in self._flexible_names:
                        self._all_flexible_instances.add(atom)
                elif self._is_num_comparison(p):
                    self._require_single_fluent(p, "precondition")
                    num_pre.append(p)
                else:
                    raise NotImplementedError(
                        f"Unsupported precondition in STRIPS/ABA encoding: {p}")

            pos_eff, neg_eff, num_eff = [], [], []
            for e in action.effects:
                if e.condition is not None and not e.condition.is_true():
                    raise NotImplementedError(
                        f"Conditional effects are not supported by the ABA encoding: {e}")
                target = (e.fluent.fluent().name, tuple(str(a) for a in e.fluent.args))
                if e.fluent.fluent().name in self._numeric_names:
                    if self._num_fluents_in(e.value):
                        raise NotImplementedError(
                            f"Numeric effect with a fluent-valued change "
                            f"({e.fluent} <- {e.value}) is not supported by the ABA "
                            "encoding (single-fluent numeric only); use the PLASP backend.")
                    num_eff.append((e.kind, target, e.value))
                    continue
                self._all_flexible_instances.add(target)
                (pos_eff if str(e.value) == "true" else neg_eff).append(target)
            self._actions.append({
                "name": action.name,
                "up_action": action,
                "bool_pre": bool_pre,
                "num_pre": num_pre,
                "pos_effects": pos_eff,
                "neg_effects": neg_eff,
                "num_effects": num_eff,
            })

        # Goal: a conjunction of boolean fluent atoms and/or numeric comparisons.
        self._goal_literals = []      # boolean (name, args)
        self._goal_comparisons = []   # numeric comparison FNodes
        for g in gp.goals:
            self._collect_goal(g)
        for atom in self._goal_literals:
            if atom[0] in self._flexible_names:
                self._all_flexible_instances.add(atom)

    def _collect_goal(self, expr):
        if expr.is_and():
            for arg in expr.args:
                self._collect_goal(arg)
        elif expr.is_fluent_exp() and expr.fluent().type.is_bool_type():
            self._goal_literals.append((expr.fluent().name, tuple(str(a) for a in expr.args)))
        elif self._is_num_comparison(expr):
            self._require_single_fluent(expr, "goal")
            self._goal_comparisons.append(expr)
        else:
            raise NotImplementedError(f"Unsupported goal expr: {expr}")

    # ------------------------------------------------------------------ #
    # Numeric helpers: comparison detection, integer coercion, evaluation.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_num_comparison(f):
        g = f.args[0] if f.is_not() else f
        return g.is_lt() or g.is_le() or g.is_equals()

    def _require_single_fluent(self, fnode, what):
        """The ABA encoding only handles numeric comparisons over one fluent
        (compared to a constant). Coupling several fluents (``f + g``, ``f >= g``)
        would enumerate the product of their value domains -- too expensive to
        propositionalise -- so reject it up front."""
        involved = self._num_fluents_in(fnode)
        if len(involved) > 1:
            raise NotImplementedError(
                f"Numeric {what} couples several fluents "
                f"({', '.join(self._nkey(fk) for fk in involved)}) in {fnode}; the ABA "
                "encoding supports only single-fluent numeric expressions -- "
                "use the PLASP backend.")

    @staticmethod
    def _as_int(fnode):
        if fnode.is_int_constant():
            return int(fnode.constant_value())
        if fnode.is_real_constant():
            frac = fnode.constant_value()
            if frac.denominator != 1:
                raise NotImplementedError(f"Non-integral numeric constant: {fnode}")
            return int(frac)
        raise NotImplementedError(f"Expected a numeric constant, got: {fnode}")

    def _num_fluents_in(self, fnode, out=None):
        """Ordered, de-duplicated numeric fluent instances appearing in `fnode`."""
        if out is None:
            out = []
        if fnode.is_fluent_exp() and fnode.fluent().name in self._numeric_names:
            key = (fnode.fluent().name, tuple(str(a) for a in fnode.args))
            if key not in out:
                out.append(key)
        else:
            for a in fnode.args:
                self._num_fluents_in(a, out)
        return out

    def _eval_num(self, fnode, assign):
        """Evaluate an arithmetic FNode to an int given `assign`: fkey -> value."""
        if fnode.is_int_constant() or fnode.is_real_constant():
            return self._as_int(fnode)
        if fnode.is_fluent_exp():
            return assign[(fnode.fluent().name, tuple(str(a) for a in fnode.args))]
        if fnode.is_plus():
            return sum(self._eval_num(a, assign) for a in fnode.args)
        if fnode.is_minus():
            return self._eval_num(fnode.args[0], assign) - self._eval_num(fnode.args[1], assign)
        if fnode.is_times():
            acc = 1
            for a in fnode.args:
                acc *= self._eval_num(a, assign)
            return acc
        raise NotImplementedError(f"Unsupported arithmetic term: {fnode}")

    def _eval_cmp(self, fnode, assign):
        """Evaluate a comparison FNode to a bool given `assign`."""
        if fnode.is_not():
            return not self._eval_cmp(fnode.args[0], assign)
        lhs, rhs = self._eval_num(fnode.args[0], assign), self._eval_num(fnode.args[1], assign)
        if fnode.is_lt():
            return lhs < rhs
        if fnode.is_le():
            return lhs <= rhs
        if fnode.is_equals():
            return lhs == rhs
        raise NotImplementedError(f"Unsupported comparison: {fnode}")

    # ------------------------------------------------------------------ #
    # Atom names. Atoms in aspforaba are opaque keys; they only need to be
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

    @staticmethod
    def _vstr(v):
        return f"m{-v}" if v < 0 else str(v)

    @classmethod
    def _nkey(cls, fkey):
        return "_".join((fkey[0],) + fkey[1])

    @classmethod
    def _val(cls, fkey, v, k):
        return f"val_{cls._nkey(fkey)}_{cls._vstr(v)}_{k}"

    @classmethod
    def _mod(cls, fkey, k):
        return f"mod_{cls._nkey(fkey)}_{k}"

    @classmethod
    def _dfn(cls, fkey, v, k):
        return f"dfn_{cls._nkey(fkey)}_{cls._vstr(v)}_{k}"

    def _cmp_atom(self, cmp_fnode, k):
        cmp_id = self._cmp_ids.setdefault(str(cmp_fnode), f"cmp{len(self._cmp_ids)}")
        return f"{cmp_id}_{k}"

    # ------------------------------------------------------------------ #
    # Framework mutators.
    # ------------------------------------------------------------------ #
    def _add_rule(self, head, body):
        self.formula["rules"].append((head, list(body)))

    def _add_assumption(self, atom):
        if atom not in self._assumptions:
            self._assumptions.add(atom)
            self.formula["assumptions"].append(atom)

    def _add_contrary(self, assumption, contrary):
        self.formula["contraries"].append((assumption, contrary))

    # ------------------------------------------------------------------ #
    # Numeric value domains (per step), computed incrementally.
    # ------------------------------------------------------------------ #
    def _seed_domain(self):
        self._domain[0] = {fkey: {v} for fkey, v in self._num_init.items()}

    def _advance_domain(self, k):
        """domain[k+1] = every value that can persist from step k, plus every
        value an action's numeric effect can produce from a step-k state."""
        prev = self._domain[k]
        nxt = {fkey: set(vals) for fkey, vals in prev.items()}   # persistence
        for action in self._actions:
            for kind, target, value in action["num_effects"]:
                involved = list(self._num_fluents_in(value))
                if kind in (EffectKind.INCREASE, EffectKind.DECREASE) and target not in involved:
                    involved.append(target)
                for combo in self._domain_product(prev, involved):
                    nxt.setdefault(target, set()).add(self._apply_effect(kind, target, value, combo))
        self._domain[k + 1] = nxt

    def _domain_product(self, domain, fkeys):
        """Yield every assignment (dict) over `fkeys` from their step domains.

        Single-fluent numeric is enforced at extraction, so `fkeys` holds at
        most one fluent here; the general form still handles the empty case
        (a constant expression -> one empty assignment)."""
        if not fkeys:
            yield {}
            return
        value_lists = [sorted(domain.get(fk, {0})) for fk in fkeys]
        for combo in product(*value_lists):
            yield dict(zip(fkeys, combo))

    def _apply_effect(self, kind, target, value, assign):
        delta = self._eval_num(value, assign)
        if kind == EffectKind.INCREASE:
            return assign[target] + delta
        if kind == EffectKind.DECREASE:
            return assign[target] - delta
        if kind == EffectKind.ASSIGN:
            return delta
        raise NotImplementedError(f"Unsupported numeric effect kind: {kind}")

    # ------------------------------------------------------------------ #
    # Encoding stages.
    # ------------------------------------------------------------------ #
    def encode(self, k):
        """Extend the framework in place so it covers horizon ``k``."""
        if k == 0:
            if not self.formula["rules"]:
                self._encode_base()
        else:
            self._encode_transition(k - 1)
        self._set_goal(k)

    def _encode_base(self):
        """Time-independent layer: rigid facts + initial state at k=0."""
        for atom in self._rigid_facts:
            self._add_rule(self._f(*atom), [])
        for atom in self._init_true_flexible:
            self._add_rule(self._f(*atom, 0), [])
        self._seed_domain()
        for fkey, v in self._num_init.items():
            self._add_rule(self._val(fkey, v, 0), [])

    def _encode_transition(self, k):
        """Add the rules/assumptions/contraries carrying state at step k to
        step k+1 (frame axioms included) via the actions firing at step k."""
        self._advance_domain(k)
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
            self._encode_numeric_effects(action, ae_atom, k)

            # The action atom is an assumption; remember what it denotes so the
            # plan can be read straight off the stable extension.
            self._add_assumption(a_atom)
            self.action_atoms[a_atom] = (action["up_action"], k)

            # bar(a_k): only one action fires per step, and every precondition
            # (boolean or numeric) must hold at step k for a_k to stand.
            for other in action_atoms:
                if other != a_atom:
                    self._add_contrary(a_atom, other)
            for atom in action["bool_pre"]:
                self._add_contrary(a_atom, self._require_bool(atom, k))
            for cmp_fnode in action["num_pre"]:
                self._add_contrary(a_atom, self._require_cmp(cmp_fnode, k))

        self._encode_bool_frame(k)
        self._encode_num_frame(k)

    def _require_bool(self, atom, k):
        """Assumption `not_p_k` ("p fails at step k") with contrary "p holds at
        step k"; returns the assumption so it can be added to an action's bar."""
        not_atom = self._not(atom, k)
        if not_atom not in self._assumptions:
            contrary = self._f(*atom, k) if atom[0] in self._flexible_names else self._f(*atom)
            self._add_assumption(not_atom)
            self._add_contrary(not_atom, contrary)
        return not_atom

    def _require_cmp(self, cmp_fnode, k):
        """Assumption "comparison fails at step k" with contrary the derived
        "comparison holds at step k" atom; returns the assumption."""
        cmp_atom = self._emit_comparison(cmp_fnode, k)
        not_atom = "not_" + cmp_atom
        if not_atom not in self._assumptions:
            self._add_assumption(not_atom)
            self._add_contrary(not_atom, cmp_atom)
        return not_atom

    def _emit_comparison(self, cmp_fnode, k):
        """Derive "comparison holds at step k" from the step-k value atoms: one
        rule per assignment of the involved fluents that satisfies it. Returns
        the comparison atom (idempotent per (comparison, step))."""
        cmp_atom = self._cmp_atom(cmp_fnode, k)
        if getattr(self, "_emitted_cmps", None) is None:
            self._emitted_cmps = set()
        if cmp_atom in self._emitted_cmps:
            return cmp_atom
        self._emitted_cmps.add(cmp_atom)

        fkeys = self._num_fluents_in(cmp_fnode)
        for assign in self._domain_product(self._domain[k], fkeys):
            if self._eval_cmp(cmp_fnode, assign):
                self._add_rule(cmp_atom, [self._val(fk, assign[fk], k) for fk in fkeys])
        return cmp_atom

    def _encode_numeric_effects(self, action, ae_atom, k):
        """Value-producing rules for the action's numeric effects at step k+1,
        plus the `mod` atom that defeats value persistence for each target."""
        for kind, target, value in action["num_effects"]:
            self._add_rule(self._mod(target, k + 1), [ae_atom])
            involved = list(self._num_fluents_in(value))
            if kind in (EffectKind.INCREASE, EffectKind.DECREASE) and target not in involved:
                involved.append(target)
            for assign in self._domain_product(self._domain[k], involved):
                new_v = self._apply_effect(kind, target, value, assign)
                body = [ae_atom] + [self._val(fk, assign[fk], k) for fk in involved]
                self._add_rule(self._val(target, new_v, k + 1), body)

    def _encode_bool_frame(self, k):
        """A flexible boolean instance holds at k+1 if it held at k, some action
        happened, and it was not defeated by a negative effect (default df,
        contrary neg_*)."""
        for atom in self._all_flexible_instances:
            df_atom = self._df(atom, k + 1)
            self._add_rule(self._f(*atom, k + 1),
                           [self._f(*atom, k), self._ha(k), df_atom])
            self._add_assumption(df_atom)
            self._add_contrary(df_atom, self._neg(atom, k + 1))

    def _encode_num_frame(self, k):
        """A numeric value persists to k+1 if it held at k, some action happened,
        and its fluent was not modified this step (default dfn, contrary mod)."""
        for fkey, values in self._domain[k].items():
            mod_atom = self._mod(fkey, k + 1)
            for v in values:
                dfn_atom = self._dfn(fkey, v, k + 1)
                self._add_rule(self._val(fkey, v, k + 1),
                               [self._val(fkey, v, k), self._ha(k), dfn_atom])
                self._add_assumption(dfn_atom)
                self._add_contrary(dfn_atom, mod_atom)

    def _set_goal(self, k):
        """(Re)write the single `goal <- ...` rule to point at horizon k."""
        self.formula["rules"] = [(h, b) for (h, b) in self.formula["rules"] if h != "goal"]
        body = [self._f(*atom, k) if atom[0] in self._flexible_names else self._f(*atom)
                for atom in self._goal_literals]
        body += [self._emit_comparison(cmp_fnode, k) for cmp_fnode in self._goal_comparisons]
        self._add_rule("goal", body)
