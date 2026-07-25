import os
import time
from fractions import Fraction
from typing import List, Optional, Tuple

import clingo

from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.plans import SequentialPlan, TimeTriggeredPlan, ActionInstance

from aspplanners.common.temporal import DEFAULT_TIME_SCALE
from aspplanners.plasp.encoder import PLASPEncoder
from aspplanners.plasp.facts import asp_name
from aspplanners.lp_io import ASPStatement, parse_lp_file, dump_lp
from aspplanners.common.validation import validate_plan

_ENCODINGS_DIR = os.path.join(os.path.dirname(__file__), 'encodings')

# encoder type -> (encoder class, clingo encoding file)
ENCODERS = {
    'seq': (PLASPEncoder, os.path.join(_ENCODINGS_DIR, 'sequential-horizon.lp')),
}


class PLASPPlanner:
    """Horizon-based planner: compiles a UP problem to ASP facts and searches
    for a plan with clingo, deepening the horizon until a model is found.

    Registered with Unified Planning as the ``ASPPlanner`` engine.

    The solve status of the last `plan()` call is kept in `self.status`
    (a `PlanGenerationResultStatus`) and human-readable notes in `self.logs`.
    """

    def __init__(self, problem, encoder_type='seq', compilationlist: Optional[List[List[str]]] = None,
                 time_scale: int = DEFAULT_TIME_SCALE):
        if encoder_type not in ENCODERS:
            raise ValueError(
                f"Unsupported encoder type: {encoder_type!r}; available: {sorted(ENCODERS)}")
        encoder_cls, self.encoding_path = ENCODERS[encoder_type]
        self.problem       = problem
        self.compiled_task = encoder_cls(time_scale=time_scale).compile(problem, self._check_compilationlist(problem, compilationlist))
        self.task          = self.compiled_task.problem
        # The task facts never change across horizons: build the string once.
        self.task_facts    = '\n'.join(sorted(self.compiled_task.fact_lines))
        # Model atoms carry ASP-rendered names ('-' -> '_'); map them back to
        # the compiled task's vocabulary (the encoder guarantees injectivity).
        self._actions_by_asp_name = {asp_name(a.name): a for a in self.task.actions}
        self._objects_by_asp_name = {asp_name(o.name): o for o in self.task.all_objects}
        self.logs: List[str] = []
        self.status: Optional[PlanGenerationResultStatus] = None

    def _check_compilationlist(self, problem, compilationlist: Optional[List[List[str]]]) -> List[List[str]]:
        """The UP compilers to run before encoding: by default, none."""
        if compilationlist is not None:
            return compilationlist

        # Empty: the task goes to clingo as it stands.
        #
        # Nothing needs compiling away, because the encoding states all of it --
        # negative conditions (it is multi-valued, so `value(V, false)` is a
        # value like any other), `forall` (a conjunction over the universe,
        # emitted with its variable free for the grounder to range), and `or` /
        # `exists` (an orGroup/orDisjunct pair, the latter with its disjuncts
        # indexed by the quantified variable's binding).
        #
        # Nor is the task pre-ground. gringo grounds it either way, and the
        # lifted encoding gives it as much to prune with: action signature rules
        # bind parameters via has(_, type(...)) and fold static preconditions
        # into the rule body. On the shipped classical domains that reaches the
        # same program a reachability grounder does, byte for byte, for a
        # quarter of the build time.
        #
        # The exception is an action narrowed *only* by a dynamic precondition
        # -- `use(?x, ?m)` gated by `loaded(?x, ?m)`, where `loaded` is only ever
        # established for some pairs. Static folding cannot see that; reachability
        # analysis can, and the lifted program is quadratically bigger. For a
        # domain shaped like that, pass a compilationlist ending in
        #   [select_grounder(problem.kind, REACHABILITY_GROUNDERS), CompilationKind.GROUNDING]
        # (which is what common.compilation.select_grounder is there for).
        #
        # PDDL (:functions ...) parse as real-typed fluents; the fact builders
        # accept them as long as every constant is integral (clingo terms are
        # integers) and raise otherwise.
        return []

    def validate(self, plan) -> Tuple[bool, Optional[str]]:
        """Validate a plan against the original problem with UP's sequential
        plan validator. Returns (is_valid, reason); reason is None/empty when
        the plan is valid."""
        return validate_plan(self.problem, plan)

    def lp_program(self) -> str:
        """The complete logic program: the compiled task's facts followed by
        the encoding rules. Useful for dumping to a file or feeding a clingo
        Control of your own.

        The encoding is multi-shot (`#program base/step(t)/check(t)`): ground
        it with parts [('base', []), ('step', [1..h]), ('check', [h])] and
        assign the external `query(h)` to true for a fixed-horizon solve.
        The facts come first so they belong to the implicit base part.
        """
        with open(self.encoding_path, 'r') as f:
            encoding = f.read()
        return f"%% Task facts\n{self.task_facts}\n\n{encoding}"

    def encoding_terms(self) -> List[ASPStatement]:
        """The loaded encoding parsed into ASPTerm statements (facts, rules,
        constraints, directives) for programmatic inspection or rewriting;
        write a modified list back with `aspplanners.lp_io.dump_lp`.
        Unlike `lp_program()`, the rendering is clingo-normalized (comments
        dropped, whitespace normalized)."""
        return parse_lp_file(self.encoding_path)

    def dump_lp_program(self, destination) -> None:
        """Write the complete logic program (task facts + encoding, verbatim
        with comments) to a file path or file-like object."""
        program = self.lp_program()
        if hasattr(destination, 'write'):
            destination.write(program)
        else:
            with open(destination, 'w') as f:
                f.write(program)

    def _empty_plan(self):
        """The "no plan" value, of whichever plan class this task produces."""
        return TimeTriggeredPlan([]) if self.compiled_task.is_temporal else SequentialPlan([])

    def plan(self, horizon=None, max_horizon=1000, timeout=None):
        """Iterative-deepening search over horizons 0..max_horizon, or a single
        solve at `horizon` when given.

        Deepening is multi-shot: one clingo Control instance grounds only the
        new step(t)/check(t) parts per horizon instead of regrounding the whole
        program each iteration.

        Returns the plan mapped back onto the original problem -- a
        `SequentialPlan`, or a `TimeTriggeredPlan` when the task has durative
        actions. The empty plan means "no plan found" (check `self.status` to
        distinguish an unsatisfiable/timed-out search from a goal that is
        trivially reached), except when the goal already holds in the initial
        state, in which case the empty plan IS the solution and `self.status`
        is SOLVED_SATISFICING.
        """
        deadline = time.monotonic() + timeout if timeout is not None else None

        ctl = clingo.Control(arguments=['-n', '1'])
        ctl.load(self.encoding_path)
        ctl.add('base', [], self.task_facts)

        if horizon is not None:
            parts = [('base', []), ('check', [clingo.Number(horizon)])]
            parts += [('step', [clingo.Number(t)]) for t in range(1, horizon + 1)]
            ctl.ground(parts)
            ctl.assign_external(clingo.Function('query', [clingo.Number(horizon)]), True)
            outcome, symbols = self._solve(ctl, deadline)
            if outcome == 'unsat':
                self.status = PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY
                self.logs.append(f'No plan exists at the fixed horizon {horizon}.')
                return self._empty_plan()
            return self._conclude(outcome, symbols, horizon)

        ctl.ground([('base', []), ('check', [clingo.Number(0)])])
        ctl.assign_external(clingo.Function('query', [clingo.Number(0)]), True)
        for t in range(0, max_horizon + 1):
            outcome, symbols = self._solve(ctl, deadline)
            if outcome != 'unsat':
                return self._conclude(outcome, symbols, t)
            if t == max_horizon:
                break
            # Retire the horizon-t goal test and extend the program by one step.
            ctl.release_external(clingo.Function('query', [clingo.Number(t)]))
            ctl.ground([('step', [clingo.Number(t + 1)]), ('check', [clingo.Number(t + 1)])])
            ctl.assign_external(clingo.Function('query', [clingo.Number(t + 1)]), True)

        self.status = PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY
        self.logs.append(
            f'No plan found up to horizon {max_horizon} (the task may be solvable with a longer horizon).')
        return self._empty_plan()

    # ------------------------------------------------------------------
    # Solving
    # ------------------------------------------------------------------

    def _solve(self, ctl, deadline) -> Tuple[str, Optional[list]]:
        """Run one solve call against the current grounding.

        Returns ('sat', symbols) | ('unsat', None) | ('timeout', None).
        """
        time_left = None
        if deadline is not None:
            time_left = deadline - time.monotonic()
            if time_left <= 0:
                return 'timeout', None

        models = []
        with ctl.solve(on_model=lambda m: models.append(m.symbols(shown=True)),
                       async_=True) as handle:
            if not handle.wait(time_left):
                handle.cancel()
                return 'timeout', None
            result = handle.get()

        if result.satisfiable:
            return 'sat', models[-1]
        return 'unsat', None

    def _conclude(self, outcome, symbols, horizon):
        """Turn a non-unsat solve outcome into a plan + status."""
        if outcome == 'timeout':
            self.status = PlanGenerationResultStatus.TIMEOUT
            self.logs.append(f'Timed out while solving horizon {horizon}.')
            return self._empty_plan()

        _plan = self._extract_plan(symbols)
        is_valid, reason = self.validate(_plan)
        if not is_valid:
            self.status = PlanGenerationResultStatus.INTERNAL_ERROR
            self.logs.append(f'Plan validation failed: {reason}')
            return self._empty_plan()
        self.status = PlanGenerationResultStatus.SOLVED_SATISFICING
        return _plan

    # ------------------------------------------------------------------
    # Plan extraction
    # ------------------------------------------------------------------

    def _extract_plan(self, symbols):
        """Build a plan from the model's occurs/2 atoms and lift it back onto
        the original problem via the compiler pipeline's composed map-back.

        A temporal task yields a `TimeTriggeredPlan`, everything else a
        `SequentialPlan`.
        """
        if self.compiled_task.is_temporal:
            return self._extract_time_triggered_plan(symbols)
        occurs = sorted((s for s in symbols if s.match('occurs', 2)),
                        key=lambda s: s.arguments[1].number)
        steps = [self._construct_action(self._action_tuple(s)) for s in occurs]
        _plan = SequentialPlan(steps)
        return _plan.replace_action_instances(self.compiled_task.map_back_action_instance)

    def _extract_time_triggered_plan(self, symbols) -> TimeTriggeredPlan:
        """Read a schedule off the model's occurs/2 and delta/2 atoms.

        Happening t sits at scaled time delta(1) + ... + delta(t) -- the
        encoding leaves the absolute times implicit and reports only the gaps.
        One snap action fires per happening, and a durative action may not
        overlap itself, so its start is closed by the next end of the same
        action; the pair becomes one entry with the elapsed time as duration.
        Scaled times are multiplied by the encoding's time unit to land back on
        the task's own time scale.
        """
        unit = self.compiled_task.time_unit
        snaps = self.compiled_task.snap_actions
        gaps = {s.arguments[0].number: s.arguments[1].number
                for s in symbols if s.match('delta', 2)}
        clock, happening_time = 0, {}
        for step in sorted(gaps):
            clock += gaps[step]
            happening_time[step] = clock

        steps, open_starts = [], {}
        for symbol in sorted((s for s in symbols if s.match('occurs', 2)),
                             key=lambda s: s.arguments[1].number):
            action_tuple = self._action_tuple(symbol)
            at = Fraction(happening_time.get(symbol.arguments[1].number, 0)) * unit
            snap = snaps.get(action_tuple[0])
            if snap is None:
                steps.append([at, self._construct_action(action_tuple), None])
                continue
            durative_name, half = snap
            key = (durative_name, action_tuple[1:])
            if half == 'start':
                open_starts[key] = len(steps)
                steps.append([at, self._construct_action(action_tuple), None])
            else:
                start = steps[open_starts.pop(key)]
                start[2] = at - start[0]
        if open_starts:
            raise ValueError(
                f"The ASP model leaves {sorted(k[0] for k in open_starts)} running at the "
                "horizon; the encoding's goal test should have ruled that out.")

        _plan = TimeTriggeredPlan([tuple(step) for step in steps])
        return _plan.replace_action_instances(self.compiled_task.map_back_action_instance)

    @staticmethod
    def _action_tuple(occurs_symbol) -> tuple:
        """occurs(action(("name", constant("o1"), ...)), T) -> (name, o1, ...).

        Zero-parameter actions appear as action(("name")), which clingo
        represents as the plain string (a one-element parenthesis is not
        a tuple term).
        """
        term = occurs_symbol.arguments[0].arguments[0]
        if term.type == clingo.SymbolType.String:
            return (term.string,)
        name = term.arguments[0].string
        args = []
        for t in term.arguments[1:]:
            if t.type == clingo.SymbolType.Function and t.name == 'constant':
                args.append(t.arguments[0].string)
            elif t.type == clingo.SymbolType.String:
                args.append(t.string)
            else:
                args.append(str(t))
        return (name, *args)

    def _construct_action(self, action_tuple) -> ActionInstance:
        name, *arg_names = action_tuple
        # A snap action is not an action of the compiled task: it is half of the
        # durative action the plan step really is.
        snap = self.compiled_task.snap_actions.get(name)
        if snap is not None:
            name = asp_name(snap[0])
        up_action = self._actions_by_asp_name.get(name)
        if up_action is None:
            raise ValueError(f"Action {name!r} from the ASP model not found in the compiled task.")
        up_args = []
        for arg in arg_names:
            up_object = self._objects_by_asp_name.get(arg)
            if up_object is None:
                raise ValueError(
                    f"Object {arg!r} (argument of action {name!r}) not found in the compiled task.")
            up_args.append(up_object)
        return ActionInstance(up_action, up_args)
