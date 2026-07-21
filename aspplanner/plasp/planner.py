import os
import time
from typing import List, Optional, Tuple

import clingo

from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.plans import SequentialPlan, ActionInstance
from unified_planning.shortcuts import CompilationKind

from aspplanner.plasp.encoder import PLASPEncoder
from aspplanner.plasp.facts import asp_name
from aspplanner.lp_io import ASPStatement, parse_lp_file, dump_lp
from aspplanner.common.validation import validate_plan

_ENCODINGS_DIR = os.path.join(os.path.dirname(__file__), 'encodings')

# encoder type -> (encoder class, clingo encoding file)
ENCODERS = {
    'seq': (PLASPEncoder, os.path.join(_ENCODINGS_DIR, 'sequential-horizon.lp')),
}


class ASPPlanner:
    """Horizon-based planner: compiles a UP problem to ASP facts and searches
    for a plan with clingo, deepening the horizon until a model is found.

    The solve status of the last `plan()` call is kept in `self.status`
    (a `PlanGenerationResultStatus`) and human-readable notes in `self.logs`.
    """

    def __init__(self, problem, encoder_type='seq', compilationlist: Optional[List[List[str]]] = None):
        if encoder_type not in ENCODERS:
            raise ValueError(
                f"Unsupported encoder type: {encoder_type!r}; available: {sorted(ENCODERS)}")
        encoder_cls, self.encoding_path = ENCODERS[encoder_type]
        self.problem       = problem
        self.compiled_task = encoder_cls().compile(problem, self._check_compilationlist(problem, compilationlist))
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
        # Numeric tasks skip the grounding step entirely: the Fast Downward
        # reachability grounder rejects them, and pre-grounding with UP's
        # grounder only bloats the program — the lifted encoding already
        # carries everything gringo needs to instantiate (action signature
        # rules bind parameters via has(_, type(...)) and folded static
        # preconditions prune the bindings). PDDL (:functions ...) parse as
        # real-typed fluents; the fact builders accept them as long as every
        # constant is integral (clingo terms are integers) and raise otherwise.
        
        if compilationlist is not None:
            return compilationlist
        
        kind = problem.kind
        numeric = kind.has_int_fluents() or kind.has_real_fluents()
        retlsit = []
        if compilationlist is None:
            retlsit += [["up_quantifiers_remover", CompilationKind.QUANTIFIERS_REMOVING]]
            retlsit += [["up_negative_conditions_remover", CompilationKind.NEGATIVE_CONDITIONS_REMOVING]]
            retlsit += [["up_disjunctive_conditions_remover", CompilationKind.DISJUNCTIVE_CONDITIONS_REMOVING]]

        if not numeric:
            retlsit += [["fast-downward-reachability-grounder", CompilationKind.GROUNDING]]
        
        return retlsit

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
        write a modified list back with `aspplanner.lp_io.dump_lp`.
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

    def plan(self, horizon=None, max_horizon=1000, timeout=None) -> SequentialPlan:
        """Iterative-deepening search over horizons 0..max_horizon, or a single
        solve at `horizon` when given.

        Deepening is multi-shot: one clingo Control instance grounds only the
        new step(t)/check(t) parts per horizon instead of regrounding the whole
        program each iteration.

        Returns the plan mapped back onto the original problem; the empty plan
        means "no plan found" (check `self.status` to distinguish an
        unsatisfiable/timed-out search from a goal that is trivially reached),
        except when the goal already holds in the initial state, in which case
        the empty plan IS the solution and `self.status` is SOLVED_SATISFICING.
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
                return SequentialPlan([])
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
        return SequentialPlan([])

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

    def _conclude(self, outcome, symbols, horizon) -> SequentialPlan:
        """Turn a non-unsat solve outcome into a plan + status."""
        if outcome == 'timeout':
            self.status = PlanGenerationResultStatus.TIMEOUT
            self.logs.append(f'Timed out while solving horizon {horizon}.')
            return SequentialPlan([])

        _plan = self._extract_plan(symbols)
        is_valid, reason = self.validate(_plan)
        if not is_valid:
            self.status = PlanGenerationResultStatus.INTERNAL_ERROR
            self.logs.append(f'Plan validation failed: {reason}')
            return SequentialPlan([])
        self.status = PlanGenerationResultStatus.SOLVED_SATISFICING
        return _plan

    # ------------------------------------------------------------------
    # Plan extraction
    # ------------------------------------------------------------------

    def _extract_plan(self, symbols) -> SequentialPlan:
        """Build a plan from the model's occurs/2 atoms and lift it back onto
        the original problem via the compiler pipeline's composed map-back."""
        occurs = sorted((s for s in symbols if s.match('occurs', 2)),
                        key=lambda s: s.arguments[1].number)
        steps = [self._construct_action(self._action_tuple(s)) for s in occurs]
        _plan = SequentialPlan(steps)
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
