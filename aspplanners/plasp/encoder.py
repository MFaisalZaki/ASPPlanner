"""The PLASP encoder: compiles a UP Problem into ASP facts."""

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import chain

from unified_planning.engines.compilers.utils import replace_action
from unified_planning.shortcuts import OperatorKind
from unified_planning.shortcuts import EffectKind

from unified_planning.model import (
    Problem,
    Action,
    DurativeAction,
)

from typing import Callable, Dict, List, Optional, Set, Tuple
from functools import partial

from unified_planning.plans import ActionInstance

from aspplanners.common.tim_typer import TIMTypeInferenceCompiler
from aspplanners.common.compilation import compose_map_backs, run_compilers
from aspplanners.common.temporal import (
    DEFAULT_TIME_SCALE,
    add_effect as _add_effect,
    all_effects as _all_effects,
    effect_groups as _effect_groups,
    split_durative_actions,
)

from aspplanners.plasp.facts import (
    asp_name,
    ASPType,
    ASPBooleanType,
    ASPConstant,
    ASPHasConstant,
    ASPFluent,
    ASPNumFluent,
    ASPAction,
    ASPDurativeAction,
    action_declaration_atom,
    ASPInitialState,
    ASPGoalState,
    ASPNumGoal,
    is_numeric_comparison,
)


def _check_asp_name_collisions(problem: Problem, action_names) -> None:
    """The ASP rendering maps '-' to '_' (see asp_facts.asp_name); two UP
    names that only differ there would become indistinguishable in the model
    atoms, so fail loudly before encoding.

    `action_names` is passed in rather than read off the problem because a
    temporal task encodes the *snap* actions the durative ones were split into,
    whose names have to be checked instead.
    """
    groups = (
        ('action', action_names),
        ('object', (o.name for o in problem.all_objects)),
        ('fluent', (f.name for f in problem.fluents)),
        ('type',   (t.name for t in problem.user_types)),
    )
    for what, names in groups:
        seen = {}
        for name in names:
            rendered = asp_name(name)
            if rendered in seen and seen[rendered] != name:
                raise ValueError(
                    f"ASP renaming '-'->'_' makes {what} names collide: "
                    f"{seen[rendered]!r} and {name!r}")
            seen[rendered] = name

@dataclass
class ASPEncodingResult:
    """What the planner needs from the compilation pipeline.

    `problem` is the fully compiled problem (grounded for classical tasks,
    lifted for numeric ones; renamed either way) whose actions/objects the
    ASP model's atoms refer to. `facts` holds the ASP fact lines grouped by
    kind; `map_back_action_instance` lifts a plan step on `problem` back to
    the user's original problem.

    On a temporal task the model's actions are the *snap* actions the durative
    ones were split into, which are not actions of `problem`: `snap_actions`
    maps each one back to the durative action it is half of, and `time_unit` is
    the real duration of one of the encoding's integer time units, so a
    happening's absolute time is `time_unit` times its scaled time.
    """
    problem: Problem
    facts: Dict[str, Set[str]]
    map_back_action_instance: Callable[[ActionInstance], Optional[ActionInstance]]
    snap_actions: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    time_unit: Fraction = Fraction(1)
    # Whether plans for this task are schedules. Not simply `snap_actions != {}`:
    # a task can be temporal and still have every durative action pruned as
    # unreachable, and its plans are still validated as schedules.
    is_temporal: bool = False

    @property
    def fact_lines(self) -> Set[str]:
        """All fact lines, flattened."""
        return set().union(*self.facts.values()) if self.facts else set()


def _render_facts(asp_objects, wrap: Optional[str] = None) -> Set[str]:
    """Render ASP fact builders to normalized fact lines.

    Builders may emit several newline-separated lines (ASPHasConstant,
    ASPAction); each line is optionally wrapped in a declaration predicate
    (`wrap='type'` turns the term `type("t")` into the fact `type(type("t"))`,
    plasp's convention of tagging terms with their declaring predicate) and
    always ends with a dot.
    """
    lines = set()
    for asp_object in asp_objects:
        for line in str(asp_object).split('\n'):
            line = line.strip()
            if not line:
                continue
            if wrap is not None:
                line = f'{wrap}({line})'
            if not line.endswith('.'):
                line += '.'
            lines.add(line)
    return lines


class PLASPEncoder:
    """
    This is a recreation of the PLASP tool: compiles a UP Problem into the
    ASP facts consumed by the encodings in `aspplanners/plasp/encodings/`.
    """

    name = "plaspencoder"

    def __init__(self, time_scale: int = DEFAULT_TIME_SCALE):
        if time_scale < 1:
            raise ValueError(f"time_scale must be a positive integer, got {time_scale!r}")
        self.time_scale = time_scale

    def compile(self, problem: Problem, compilationlist: List[List[str]]) -> ASPEncodingResult:
        assert isinstance(problem, Problem)

        new_problem = problem.clone()
        new_problem.name = f"{self.name}_{problem.name}"

        # Each compilation stage contributes a map-back; composed in reverse
        # they lift a plan on the final problem to the user's original problem.
        map_backs = []

        # Compilation pipeline:
        # step one remove delete then set effects.
        new_problem.clear_actions()
        delete_then_set_map = {}
        for a in problem.actions:
            clean_action = self._remove_delete_then_set(a)
            new_problem.add_action(clean_action)
            delete_then_set_map[clean_action] = a
        map_backs.append(partial(replace_action, map=delete_then_set_map))

        new_problem, grounded_map_back = run_compilers(new_problem, compilationlist)
        map_backs.append(grounded_map_back)

        # step two check if we can infer types for untyped problems. TIM works
        # on instantaneous actions only, and a temporal task has been grounded
        # by this point anyway, so there are no parameter types left to infer.
        temporal = problem.kind.has_continuous_time()
        if len(new_problem.user_types) == 1 and not temporal:
            tim_result = TIMTypeInferenceCompiler().compile(new_problem)
            new_problem = tim_result.problem
            map_backs.append(tim_result.map_back_action_instance)

        # step three split every durative action into its two snap actions, the
        # PDDL 2.1 decomposition SMTPlan's happening encoder is built on. The
        # snaps carry the at-start/at-end halves and are encoded as ordinary
        # actions; `durative_facts` re-couples them (see ASPDurativeAction).
        time_unit, durative_facts, snap_actions, asp_actions = \
            self._split_durative_actions(new_problem)

        # Names are sanitized ('-' -> '_') at fact-rendering time by
        # asp_facts.asp_name and mapped back the same way during plan
        # extraction; make sure that mapping is injective for this task.
        _check_asp_name_collisions(new_problem, [a.name for a, _guard in asp_actions])

        # A fluent is "static" iff no action ever lists it in its effects.
        # Folding positive preconditions on such fluents into the action
        # signature body lets the grounder pre-filter parameter bindings
        # by the static relation (see ASPAction docstring).
        modified_fluent_names = set()
        for a in new_problem.actions:
            for eff in _all_effects(a):
                modified_fluent_names.add(eff.fluent._content.payload.name)
        static_fluent_names = {f.name for f in new_problem.fluents if f.name not in modified_fluent_names}

        # Fill in default values (bool False, int 0) BEFORE the initial-state
        # facts are emitted: an uninitialized numeric fluent would otherwise
        # have no holds/3 chain, which silently disables every numeric
        # precondition that reads it.
        self._initialize_fluents(new_problem)

        # False initial values are dropped, because a variable that is only ever
        # read as true needs no holds/3 chain for its false side and every such
        # chain propagates through every step. A variable read as *false*
        # somewhere does need one, though: `not holds(V, value(V,false), t-1)`
        # is how the step rule checks a negative condition, and without the
        # step-0 fact it can never be satisfied. That is what makes
        # up_negative_conditions_remover unnecessary -- the encoding tracks the
        # false value directly instead of a task needing a mirror fluent.
        asp_actions = [(ASPAction(action, static_fluent_names, guard), guard)
                       for action, guard in asp_actions]
        goal_terms = list(chain.from_iterable(
            self._generate_asp_goal_state(g) for g in new_problem.goals))
        negated_fluents = set().union(*(a.negated_fluents for a, _guard in asp_actions)) \
            if asp_actions else set()
        negated_fluents |= {g.fluent_name for g in goal_terms
                            if isinstance(g, ASPGoalState) and g.value == 'false'}

        initial_state = set(
            ASPInitialState(fluent, value) for fluent, value in new_problem.initial_values.items()
            if not value.is_false() or fluent.fluent().name in negated_fluents)
        # This is a corner case where the initial state has no true fluents. In this case we need to add all the fluents of the problem.
        if len(initial_state) == 0:
            initial_state = set(ASPInitialState(fluent, value) for fluent, value in new_problem.initial_values.items())

        facts = {
            '_types':          _render_facts((ASPType(t) for t in new_problem.user_types), wrap='type'),
            '_default_values': _render_facts(ASPBooleanType(v) for v in [True, False]),
            '_constants':      _render_facts((ASPConstant(obj) for obj in new_problem.all_objects), wrap='constant'),
            '_has':            _render_facts(ASPHasConstant(obj) for obj in new_problem.all_objects),
            '_variables':      _render_facts(ASPFluent(fluent) for fluent in new_problem.fluents),
            '_num_variables':  _render_facts(ASPNumFluent(fluent) for fluent in new_problem.fluents if fluent.type.is_int_type() or fluent.type.is_real_type()),
            '_actions':        _render_facts(action for action, _guard in asp_actions),
            '_initial_state':  _render_facts(initial_state),
            '_goal_state':     _render_facts(goal_terms),
            '_durative':       _render_facts(durative_facts),
            '_time':           {'minSeparation(1).'} if durative_facts else set(),
        }

        return ASPEncodingResult(
            problem=new_problem,
            facts=facts,
            map_back_action_instance=compose_map_backs(map_backs),
            snap_actions=snap_actions,
            time_unit=time_unit,
            is_temporal=temporal or bool(snap_actions),
        )

    # ------------------------------------------------------------------
    # Temporal: durative actions -> snap actions + temporal facts
    # ------------------------------------------------------------------

    def _split_durative_actions(self, problem: Problem):
        """Decompose every durative action into its two snap actions.

        Returns ``(time_unit, durative_facts, snap_actions, asp_actions)``: the
        real duration of one integer time unit, the ASPDurativeAction builders
        coupling each pair of snaps, a map from snap action name to
        ``(durative action name, 'start'|'end')`` for plan extraction, and the
        actions to render as ASPAction facts, each with the signature guard it
        should be declared under (None for the ordinary per-parameter one).
        """
        unit, decompositions = split_durative_actions(problem, self.time_scale)
        asp_actions = [(a, None) for a in problem.actions if not isinstance(a, DurativeAction)]
        durative_facts, snap_actions = [], {}
        for snap in decompositions:
            # An end snap usually has no condition of its own, so on a lifted
            # task nothing would narrow its parameters and it would instantiate
            # for every type-consistent binding -- far more than the durative
            # action itself has. Declaring it under its start snap fixes that:
            # an end can only ever fire for a binding whose start exists, and
            # the start's folded static preconditions carry over.
            start_atom = action_declaration_atom(snap.start.name, snap.action.parameters)
            asp_actions += [(snap.start, None), (snap.end, start_atom)]
            snap_actions[asp_name(snap.start.name)] = (snap.action.name, 'start')
            snap_actions[asp_name(snap.end.name)] = (snap.action.name, 'end')
            durative_facts.append(ASPDurativeAction(
                snap.action, snap.start.name, snap.end.name, snap.duration, snap.overall))
        return unit, durative_facts, snap_actions, asp_actions

    def _remove_delete_then_set(self, dirty_action: Action) -> Action:
        """!
        Removes delete-then-set effects from the list of effects.
        @param effects: list of effects
        @return list of effects without delete-then-set effects

        A durative action is cleaned per timing: an add and a delete of the same
        fluent only race when they happen at the same endpoint.
        """

        def has_positive_effect(fluent, effects) -> bool:
            """ Does the group have an effect that assigns the fluent to true? """
            for eff in effects:
                if eff.kind == EffectKind.ASSIGN and eff.fluent == fluent and eff.value.is_true():
                    return True
            return False

        def clean(effects):
            kept = []
            for eff in effects:
            # we avoid adding the effect if it is a delete effect and the group has also an add effect for the same fluent
                if eff.fluent.type.is_bool_type(): # only check boolean fluents
                    if eff.kind == EffectKind.ASSIGN and \
                        eff.value.is_false() and \
                        has_positive_effect(eff.fluent, effects):
                        pass
                    else:
                        kept.append(eff)
                else:
                    kept.append(eff)
            return kept

        fixed_action = dirty_action.clone() # we copy the old action
        fixed_action.clear_effects()        # and remove all the effects
        for timing, effects in _effect_groups(dirty_action):
            for eff in clean(effects):      # now we copy over only the good effects
                _add_effect(fixed_action, eff, timing)
        return fixed_action

    def _generate_asp_goal_state(self, goal_state):
        goal_predicates = [goal_state] if goal_state.node_type != OperatorKind.AND else goal_state.args
        ret_goals = []
        for g in goal_predicates:
            # Numeric comparison goals (e.g. `battery >= 1`) are checked against
            # numval at the goal step, like numeric preconditions; everything
            # else is a boolean/object state goal on a grounded fluent.
            if is_numeric_comparison(g):
                ret_goals.append(ASPNumGoal(g))
                continue
            _is_true = g.node_type != OperatorKind.NOT
            value = str(_is_true).lower()
            ret_goals.append(ASPGoalState(g if _is_true else g.args[0], value))
        return ret_goals
    
    def _initialize_fluents(self, task:Problem):
        """
        Initialize the int and real fluents of a given task with a default value of 0.
        Any Boolean fluent is initialized with a default value of False.
        Args:
            task (Problem): The UP task object
        Updates:
            task.initial_defaults: Adds default values for real and integer types.
            task.explicit_initial_values: Sets initial values for uninitialized fluents.
        """
        from unified_planning.shortcuts import Fraction
        from unified_planning.model.fluent import get_all_fluent_exp
        # update the initial defaults to account for real and integer types.
        # Use the task's own environment: with a non-global environment the
        # global one holds different type/expression manager instances.
        _env = task.environment
        _tm = _env.type_manager
        _em = _env.expression_manager
        task.initial_defaults.update({_tm.RealType():_em.Real(Fraction(0))})
        task.initial_defaults.update({_tm.IntType() :_em.Int(0)})
        task.initial_defaults.update({_tm.BoolType() :_em.Bool(False)})

        # list unitialized fluents.
        fluentslist = list(chain.from_iterable([list(get_all_fluent_exp(task, f)) for f in task.fluents]))
        initialized_fluents  = list(task.explicit_initial_values.keys())
        unintialized_fluents = list(filter(lambda x: not x in initialized_fluents, fluentslist))

        def _default(fluent_exp):
            """The value to lay down for a fluent instance with none of its own.

            The fluent's own declared default comes first; then the per-type
            defaults set above -- which a *bounded* type (`integer[0, 10]`) is
            not a key of, since it is a distinct type object from `IntType()`,
            so those fall through to a value derived from the type itself.
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

        # update the initial values for the fluents that are not initialized.
        for fe in unintialized_fluents:
            task.set_initial_value(fe, _default(fe))