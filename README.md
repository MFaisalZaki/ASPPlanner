# ASPPlanners

A lightweight planner that solves automated planning problems by compiling them to Answer Set Programming (ASP) and delegating search to [clingo](https://potassco.org/clingo/). ASPPlanners plugs into the [Unified Planning](https://github.com/aiplan4eu/unified-planning) (UP) framework as a `OneshotPlanner` engine, so any UP problem can be solved through a uniform interface.

Two backends share the same UP front-end (compilation pipeline, map-back, validation) and register as UP engines on import:

| engine | encoding | solver |
|---|---|---|
| `PLASPPlanner` (default) | PLASP-style multi-valued ASP, multi-shot over the horizon | [clingo](https://potassco.org/clingo/) |
| `ABAPlanner` (optional) | STRIPS-to-ABA (assumption-based argumentation) reduction | [aspforaba](https://bitbucket.org/coreo-group/aspforaba) |

Both are **satisficing** planners: iterative deepening stops at the first horizon that admits a plan, and reports `SOLVED_SATISFICING`. Because the horizon grows one step at a time, the plan returned is shortest *in number of steps*, but no quality metric (`total-cost`, `total-time`) is optimised — metrics are accepted and ignored.

---

# 1. Installation and quick usage

## Requirements

Python 3.10+. Runtime dependencies are installed automatically: `clingo>=5.6.0`, `unified-planning>=1.1.0`, `up_fast_downward>=0.5.2`.

## Install

```bash
git clone https://github.com/MFaisalZaki/ASPPlanners.git
cd ASPPlanners
pip install -e .
```

The `ABAPlanner` backend additionally needs `aspforaba`; install it with the optional `aba` extra:

```bash
pip install -e ".[aba]"
```

It is imported lazily, so `import aspplanners` and the default `PLASPPlanner` engine work without it.

## Quick usage

### Through the Unified Planning framework

Importing `aspplanners` registers the engine, so the standard UP entry points work out of the box:

```python
import aspplanners  # registers the PLASPPlanner and ABAPlanner engines
from unified_planning.shortcuts import OneshotPlanner

# `problem` is any unified_planning.model.Problem you constructed or parsed.
with OneshotPlanner(name="PLASPPlanner") as planner:
    result = planner.solve(problem, timeout=60)
    print(result.status)   # SOLVED_SATISFICING
    print(result.plan)     # move(l1, l2)
```

From PDDL files:

```python
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner
import aspplanners

problem = PDDLReader().parse_problem("gripper/domain.pddl", "gripper/prob01.pddl")
with OneshotPlanner(name="PLASPPlanner") as planner:
    print(planner.solve(problem, timeout=60).plan)
```

### Direct API

You can also drive the planner directly if you don't need the UP result wrapper:

```python
from aspplanners.plasp.planner import PLASPPlanner

planner = PLASPPlanner(problem, encoder_type="seq")
plan = planner.plan(max_horizon=100, timeout=60)   # or plan(horizon=10)
print(planner.status)   # PlanGenerationResultStatus of the last call
print(planner.logs)     # human-readable notes
print(planner.layers)   # ('core', 'numeric') — the encoding layers this task needs
```

`plan()` returns a `SequentialPlan` — or a `TimeTriggeredPlan` when the task has durative actions. It is empty when no plan was found (check `planner.status`) — or when the goal already holds in the initial state, in which case `status` is `SOLVED_SATISFICING`.

### Engine options

| option | engine | default | meaning |
|---|---|---|---|
| `max_horizon` | both | 1000 | upper bound on the iterative-deepening search |
| `horizon` | both | — | solve at one fixed horizon instead of deepening |
| `time_scale` | both | 10 | resolution of the temporal encoding, see [`time_scale`](#time_scale) |
| `encoder_type` | PLASP | `"seq"` | encoding family |
| `semantics` | ABA | `"ST"` | aspforaba extension semantics |
| `timeout` | PLASP only | — | honored by `PLASPPlanner`; **`ABAPlanner` ignores it**, so bound its search with `max_horizon` |

```python
with OneshotPlanner(name="PLASPPlanner", params={"max_horizon": 50}) as planner:
    ...
```

### Guarantees

- **Plans in your vocabulary.** Returned plans reference the actions and objects of the problem you passed in — every internal compilation stage (grounding, type inference, renaming, numeric rescaling, durative split) is mapped back before the plan is handed over.
- **Built-in validation.** Every returned plan is checked against the *original* problem before it is returned — with UP's `sequential_plan_validator`, or `up_time_triggered_validator` for temporal plans. Nothing that fails validation is reported as solved.

### Inspecting the generated logic program

```python
text = planner.lp_program()          # task facts + multi-shot encoding, verbatim
planner.dump_lp_program("task.lp")   # same, written to a file

terms = planner.encoding_terms()     # encoding parsed into ASPTerm statements
print(planner.encoding_paths)        # the .lp files loaded, in load order
```

`encoding_terms()` returns typed statements (`ASPFact`, `ASPRule`, `ASPConstraint`, `ASPWeakConstraint`, `ASPDirective`) wrapping clingo AST nodes — filter or rewrite them programmatically, then write them back out:

```python
from aspplanners.lp_io import parse_lp_file, dump_lp, ASPRule

terms = parse_lp_file("my-encoding.lp")
rules = [t for t in terms if isinstance(t, ASPRule)]
dump_lp(terms, "normalized.lp")   # also accepts fact-builder terms and plain strings
```

The encoding is split into `#program base / step(t) / check(t)` parts; ground `base` + `step(1..h)` + `check(h)` and set the external `query(h)` to true to solve at horizon `h`.

---

# 2. Supported PDDL features

## At a glance

| PDDL requirement | `PLASPPlanner` | `ABAPlanner` | notes |
|---|---|---|---|
| `:strips` | ✅ | ✅ | |
| `:typing` | ✅ | ✅ | flat and hierarchical |
| `:equality` | ✅ | ✅ | |
| `:negative-preconditions` | ✅ | ✅ | native in PLASP (multi-valued); compiled away for ABA |
| `:disjunctive-preconditions` | ✅ | ✅ | native in PLASP; compiled away for ABA |
| `:existential-preconditions` | ✅ | ✅ | native in PLASP; compiled away for ABA |
| `:universal-preconditions` | ✅ | ✅ | native in PLASP; compiled away for ABA |
| `:conditional-effects` | ✅ | ❌ | the ABA reduction has no encoding for them |
| `forall` **effects** | ❌ | ❌ | not declared by either engine — the single largest gap, see [§3](#3-benchmark-results) |
| `:fluents` / `:numeric-fluents` | ✅ linear | ✅ simple numeric | see [Numeric planning](#numeric-planning) |
| `:action-costs` | ⚠️ accepted | ⚠️ accepted | parsed and ignored; **not optimised** |
| `:durative-actions` | ✅ | ✅ | PDDL 2.1, see [Temporal planning](#temporal-planning) |
| `:duration-inequalities` | ✅ | ✅ | |
| `:timed-initial-literals` | ❌ | ❌ | `TIMED_EFFECTS` / `TIMED_GOALS` |
| `:continuous-effects`, PDDL+ `:processes` / `:events` | ❌ | ❌ | |
| `:derived-predicates` (axioms) | ❌ | ❌ | |
| `:preferences` / `:constraints` | ❌ | ❌ | rejected by the UP PDDL reader before the encoder sees them |

The authoritative list is `supported_kind()` in [aspplanners/up_engines.py](aspplanners/up_engines.py), in UP `ProblemKind` terms:

| `ProblemKind` feature | PLASP | ABA |
|---|---|---|
| `ACTION_BASED`, `FLAT_TYPING`, `HIERARCHICAL_TYPING`, `BOUNDED_TYPES` | ✅ | ✅ |
| `NEGATIVE_CONDITIONS`, `DISJUNCTIVE_CONDITIONS`, `EQUALITIES`, `EXISTENTIAL_CONDITIONS`, `UNIVERSAL_CONDITIONS` | ✅ | ✅ |
| `INT_FLUENTS`, `REAL_FLUENTS`, `INCREASE_EFFECTS`, `DECREASE_EFFECTS`, `SIMPLE_NUMERIC_PLANNING` | ✅ | ✅ |
| `GENERAL_NUMERIC_PLANNING`, `STATIC_FLUENTS_IN_NUMERIC_ASSIGNMENTS`, `FLUENTS_IN_NUMERIC_ASSIGNMENTS` | ✅ | ❌ |
| `CONDITIONAL_EFFECTS` | ✅ | ❌ |
| `UNDEFINED_INITIAL_NUMERIC`, `UNDEFINED_INITIAL_SYMBOLIC` | ✅ | ❌ |
| `CONTINUOUS_TIME`, `DURATION_INEQUALITIES`, `INT_TYPE_DURATIONS`, `REAL_TYPE_DURATIONS`, `STATIC_FLUENTS_IN_DURATIONS` | ✅ | ✅ |
| `MAKESPAN`, `ACTIONS_COST`, `PLAN_LENGTH`, `FINAL_VALUE` and the four `*_IN_ACTIONS_COST` kinds | ⚠️ accepted, never optimised | ⚠️ same |
| `FORALL_EFFECTS`, `TIMED_EFFECTS`, `TIMED_GOALS`, `PROCESSES`, `EVENTS`, `SELF_OVERLAPPING`, `INTERMEDIATE_CONDITIONS_AND_EFFECTS`, `OVERSUBSCRIPTION` | ❌ | ❌ |

A task outside the declared kind is refused up front with `UNSUPPORTED_PROBLEM` rather than silently mis-encoded. `ProblemKind` cannot express every distinction, so a few shapes are raised at *encoding* time instead, as `NotImplementedError`:

- a product or quotient of two numeric fluents (not linear),
- a fractional coefficient in a numeric effect,
- a bounded numeric type on a task that needs rescaling,
- a numeric over-all condition in a durative action (**ABA backend only**).

`GENERAL_NUMERIC_PLANNING` is declared because an effect that merely *reads* a fluent already pushes a task's kind there, and the feature has no linear/non-linear split.

## Nothing is compiled away

Before a problem reaches the ASP encoder it is put through a list of UP compilers — by default, **none of them**. Every condition shape is stated in the encoding itself:

| shape | how it is encoded |
|---|---|
| negative conditions | the encoding is multi-valued, so `value(V, false)` is a value like any other. A mirror fluent per negatively-read one would be pure overhead; the encoder just emits the false initial value for the fluents actually read as false |
| `forall` | a conjunction over the universe — and `precondition`/`goal` facts are already conjunctive. One rule with the variable left free and `has(_, type(...))` in the body; gringo does the expanding |
| `or`, `exists` | disjunctions, which conjunctive facts cannot state, so they get their own `orGroup`/`orDisjunct` vocabulary: at least one disjunct has to hold, and a disjunct holds when all of its literals do. An `exists` is the same shape with its disjuncts indexed by the quantified variable's binding |
| numeric comparisons | `<`, `<=`, `=` and their negations against `numval`, wherever a condition can appear: `numPrecondition`, `numGoal`, `numOverall`, and `orDisjunctNum` inside a disjunct. A negation is the comparison's complement (`not (x = y)` is `neq`), not a `not` over a `holds` chain the numeric side does not have |

Staying lifted is the point. An action with *k* disjunctions of 4 literals each — the UP disjunction remover writes out 4<sup>k</sup> copies of it, the encoding writes one group:

| `or` groups | ground actions, native | with the remover | compile, native | with the remover |
|---|---|---|---|---|
| 2 | 1 | 16 | 0.01s | 0.01s |
| 4 | 1 | 256 | 0.02s | 0.20s |
| 5 | **1** | **1024** | **0.04s** | **7.07s** |

The same holds for `exists` (over 40 objects: 41 ground actions instead of 80) and for `forall` (an action with `(forall (?x) (marked ?x))` and `(forall (?x ?y) (near ?x ?y))` stays at 2 precondition facts whether there are 3 objects or 30, against 3 and 30 for the remover).

The one shape the encoding does not take is a disjunction nested inside a disjunct (`or(and(a, or(b, c)), d)`), which would need distribution into DNF; that raises and asks for `up_disjunctive_conditions_remover` back. Everything else — including the De Morgan cases, where `not (a or b)` is a conjunction and `not (forall x φ)` is an `exists` — is handled directly.

## Numeric planning

Integer- and real-valued fluents with `increase` / `decrease` / `assign` effects whose value is a **linear** expression over the state — a constant, or `k₁·V₁ + … + C` evaluated against the previous step — and linear comparison preconditions and goals.

Reals are accepted because PDDL `(:functions ...)` parse as real-typed. Clingo terms are integers, so a task stating fractional values is rescaled to whole ones before it is encoded ([aspplanners/plasp/rescale.py](aspplanners/plasp/rescale.py)). The plan is unaffected — it is a sequence of actions. A fluent with no entry in the initial state gets a default laid down first (bool → false, numeric → 0, PDDL's own closed-world reading).

Not supported: a product or quotient of two fluents, a fractional coefficient in an effect, and a bounded numeric type on a task that needs rescaling (its bound would not move with the values it bounds). All three raise rather than silently approximate.

## Temporal planning

Tasks with durative actions are encoded the way [SMTPlan](https://github.com/KCL-Planning/SMTPlan)'s happening encoder encodes them, and both backends support the same fragment:

```python
from unified_planning.io import PDDLReader
from aspplanners.plasp.planner import PLASPPlanner

problem = PDDLReader().parse_problem("matchcellar/domain.pddl", "matchcellar/problem.pddl")
plan = PLASPPlanner(problem).plan(max_horizon=8)
for start, action, duration in plan.timed_actions:
    print(f"{start}: {action} [{duration}]")
# 1/10: light_match(m1) [6]
# 1: mend_fuse(f1) [5]
```

A timestep is a *happening*, and the model reports the gap between consecutive ones; a durative action becomes the pair of instantaneous **snap actions** carrying its at-start and at-end conditions and effects, plus a remaining-duration counter that must reach exactly zero where the end snap fires. Durative actions overlap freely, so domains with required concurrency (like the match-cellar above, where the fuse has to be mended strictly inside a burning match) are solved rather than rejected.

**Covered:** at-start / at-end / over-all conditions, at-start and at-end effects, fixed durations, duration inequalities (`(and (>= ?duration 2) (<= ?duration 5))`), and durations read off a static function (`(= ?duration (travel-time ?a ?b))`).

**Not covered:** PDDL+ processes and events, timed initial literals and timed goals, conditions or effects at an intermediate time, a durative action overlapping *itself*, and snap actions that have to be genuinely simultaneous (a happening carries one snap action, so they are sequentialised ε apart). The ABA backend additionally rejects numeric over-all conditions; use the PLASP backend for those.

Temporal tasks are solved on the *lifted* encoding — no grounder does reachability analysis on them, so pre-grounding would only duplicate gringo's work. A duration stated as a static function has no value until its parameters are bound, so the encoder defers that lookup into the ASP: `durationValue(...)` reads it off `initialState` per binding. The two snaps of a durative action ground to the same bindings the action itself does, because the end snap is declared under its start.

### `time_scale`

Clingo terms are integers, so happenings live on an integer time grid. `time_scale` (default 10) says how many times finer that grid is than the greatest common divisor of the task's durations, which makes ε — the minimum separation between two happenings — 1/10 of that gcd, matching SMTPlan's ε. Durations are normalised by their gcd first, so 100 and 150 become 20 and 30 rather than 1000 and 1500, and the discretisation is exact for rational durations rather than an approximation.

Lower it (`PLASPPlanner(problem, time_scale=1)`) when a domain has large, coprime durations and no required concurrency: **the encoding's remaining-duration recursion is quadratic in the largest scaled duration**, and this is the dominant cost on the temporal track — see [§3.5](#35-temporal).

## Grounding

The encoding is **not** pre-ground. gringo grounds the task either way, and the lifted encoding gives it as much to prune with — action signature rules bind parameters via `has(_, type(...))` and fold static preconditions into the rule body. Scaled up, that reaches the same program the Fast Downward reachability grounder does, byte for byte:

| task | ground actions / atoms, lifted | with the grounder |
|---|---|---|
| gripper, 20 balls | 164 / 36447 | 164 / 36447 |
| transport, 16 locations | 62 / 5641 | 62 / 5641 |
| blocksworld, 9 blocks | 180 / 25018 | 180 / 25018 |

**The one thing it cannot do is reachability analysis**, which prunes on *dynamic* preconditions where static folding is blind. If an action is narrowed only by a fluent some other action establishes — `use(?x, ?m)` gated by `loaded(?x, ?m)`, where `loaded` only ever holds for compatible pairs — the lifted program is quadratically bigger:

| items × machines | ground actions, lifted | with the grounder |
|---|---|---|
| 4 × 4 | 20 | 8 |
| 14 × 14 | **210** | **28** |

For a domain shaped like that, ask for the grounder back — that is what `aspplanners.common.compilation.select_grounder` is for:

```python
from unified_planning.shortcuts import CompilationKind
from aspplanners.common.compilation import REACHABILITY_GROUNDERS, select_grounder

grounder = select_grounder(problem.kind, REACHABILITY_GROUNDERS)   # None if none applies
planner = PLASPPlanner(problem, compilationlist=[[grounder, CompilationKind.GROUNDING]])
```

## Customizing the compilation pipeline

Pass `compilationlist` to take over the choice entirely. Each entry is a `[engine_name, CompilationKind]` pair applied in order, and the list is used verbatim — the automatic grounder selection is bypassed, so include or omit the grounder yourself:

```python
from unified_planning.shortcuts import CompilationKind
from aspplanners.plasp.planner import PLASPPlanner

# Solve on the lifted encoding: run the removers but skip the grounder.
pipeline = [
    ["up_quantifiers_remover",            CompilationKind.QUANTIFIERS_REMOVING],
    ["up_negative_conditions_remover",    CompilationKind.NEGATIVE_CONDITIONS_REMOVING],
    ["up_disjunctive_conditions_remover", CompilationKind.DISJUNCTIVE_CONDITIONS_REMOVING],
]
planner = PLASPPlanner(problem, encoder_type="seq", compilationlist=pipeline)
plan = planner.plan()
```

## Encoding layers

The encoding is one `.lp` file per feature, under [aspplanners/plasp/encodings/seq/](aspplanners/plasp/encodings/seq/):

| Layer | File | Covers |
|---|---|---|
| `core` | [core.lp](aspplanners/plasp/encodings/seq/core.lp) | multi-valued STRIPS over the horizon: the action choice, preconditions, (conditional) effects, inertia, the goal test |
| `numeric` | [numeric.lp](aspplanners/plasp/encodings/seq/numeric.lp) | linear numeric fluents, comparisons, effects and goals |
| `disjunctive` | [disjunctive.lp](aspplanners/plasp/encodings/seq/disjunctive.lp) | disjunctive, existential and nested conditions and goals |
| `temporal` | [temporal.lp](aspplanners/plasp/encodings/seq/temporal.lp) | PDDL 2.1 durative actions as SMTPlan-style happenings, annotated with the SMTPlan constraint each rule mirrors |

clingo merges `#program` parts of the same name across the files it is given, so the program for a task is the concatenation of the layers it needs, loaded in dependency order (`core` first — it declares the program parts and the `query(t)` external). Layers reference each other's predicates only through `#defined`, which is what lets one be left out without breaking the rest.

**Layers are chosen per task, and you do not normally pick them.** `encoder_type="seq"` infers them from the facts the encoder emitted, widened by what the compiled problem's `ProblemKind` reports:

```python
planner = PLASPPlanner(problem, encoder_type="seq")
planner.layers          # ('core', 'numeric', 'temporal')
planner.encoding_paths  # the .lp files that will be loaded, in load order
```

The emitted facts are the authority — they describe the task *as encoded*, after the compilation pipeline, TIM inference, numeric rescaling and the durative split. `ProblemKind` is a second opinion only: it reports declared features rather than the vocabulary the encoding consumes, and the two need not agree (a durative action whose duration reads a fluent produces numeric facts on a problem whose kind has no numeric feature at all). Where they differ, the union wins — an unneeded layer grounds to nothing, while a missing one would be silently ignored.

---

# 3. Benchmark results

## 3.1 Setup

Everything below is one sweep, run 2026-07-27 → 2026-07-28 through the [aspbench](benchmarks/) harness on a slurm cluster (two hosts, one CPU per task).

| | |
|---|---|
| **Benchmark set** | 9,328 unique tasks over 332 domains |
| **Time limit** | 1800 s wall per task (slurm job capped at 35 min) |
| **Memory limit** | 8192 MB per task (slurm job capped at 9 GB) |
| **`PLASPPlanner-seq`** | `encoding=seq`, `max_horizon=1000`, `time_scale=10`, all three tracks |
| **`ABAPlanner-ST`** | `semantics=ST`, `max_horizon=100`, `time_scale=2`, **classical track only** |

| track | tasks | domains | source |
|---|---|---|---|
| classical | 3,294 | 76 | [AI-Planning/classical-domains](https://github.com/AI-Planning/classical-domains) |
| numeric | 3,354 | 162 | [pyPMT/numeric-domains](https://github.com/pyPMT/numeric-domains) (1,817) + classical-domains instances carrying a `(:functions …)` block (1,537) |
| temporal | 2,680 | 94 | [potassco/pddl-instances](https://github.com/potassco/pddl-instances), the `*-time*` / `*-temporal*` IPC domains |

A task's track is decided by **reading its domain file**, not by which repository it came from — which is why 1,537 IPC "classical" instances land on the numeric track: they carry a `(total-cost)` function.

The run artifacts (`results.csv`, per-task JSONs, error logs) are not checked in — `sandbox*` and `benchmark-run/` are gitignored. See [Reproducing](#reproducing).

### Reading the status column

| status | meaning |
|---|---|
| `SOLVED` | a plan was found **and validated against the original problem** |
| `UNSUPPORTED` | refused up front — the task's `ProblemKind` is outside `supported_kind()` |
| `ERROR` | an exception; overwhelmingly the **UP PDDL reader**, not the planner (see below) |
| `TIMEOUT` | hit the 1800 s task limit |
| `MEMOUT` | hit the 8 GB task limit |
| `KILLED` | the scheduler killed the job before the harness could write a result |
| `EXHAUSTED` | `max_horizon` reached with no plan — the task may still be solvable at a deeper horizon |

## 3.2 Headline coverage

`PLASPPlanner-seq`, all three tracks:

| track | solved | of all tasks | of *encodable* tasks |
|---|---|---|---|
| classical | **440** | 440 / 3,294 (13%) | 440 / 1,872 (**24%**) |
| numeric | **273** | 273 / 3,354 (8%) | 273 / 2,041 (**13%**) |
| temporal | **24** | 24 / 2,680 (1%) | 24 / 1,236 (**2%**) |
| **total** | **737** | 737 / 9,328 (8%) | 737 / 5,149 (**14%**) |

*Encodable* = attempted, minus `UNSUPPORTED`, minus the tasks the **UP PDDL reader** could not parse. Both columns matter and they say different things: the left one is raw IPC coverage, the right one is how the search does on the tasks that actually reach clingo.

Full status breakdown:

| planner | SOLVED | UNSUPPORTED | TIMEOUT | ERROR | MEMOUT | KILLED | EXHAUSTED | total |
|---|---|---|---|---|---|---|---|---|
| `PLASPPlanner-seq` | 737 | 2,813 | 2,993 | 1,444 | 1,059 | 280 | 2 | 9,328 |
| `ABAPlanner-ST` (classical only) | 316 | 836 | 138 | 651 | 966 | 387 | 0 | 3,294 |

Every one of the 1,053 `SOLVED` results across both planners passed plan validation — no unvalidated plan was counted.

### The three failure modes, separated

**`ERROR` is almost entirely the front end, not the planner.** Of 1,444 PLASP errors, **1,366 (95%)** are the UP PDDL reader failing on the benchmark files themselves:

| exception | count | typical cause |
|---|---|---|
| `SyntaxError` | 645 | `total-time`, `duration`, `preference` expressions the reader will not take |
| `ParseException` | 230 | malformed / unsupported PDDL syntax |
| `ParseSyntaxException` | 220 | same |
| `UPProblemDefinitionError` | 215 | `Name p1 already defined` — an object and a predicate sharing a name |
| `KeyError` | 46 | a type name used before it is declared |
| `UPExpressionDefinitionError` | 10 | wrong fluent arity in the instance file |

These are IPC files the reader does not accept; the encoder never sees them. Only **72** errors are genuine encoder limitations, and they are honest refusals rather than wrong answers:

| encoder refusal | count |
|---|---|
| `Unsupported over-all condition in durative action` (a quantified implication) | 30 |
| `Expected a numeric constant, got: battery-level-full` | 20 |
| `A product of two numeric fluents is not linear` | 20 |
| `Making this task's numeric values integral needs a scale factor of …` | 2 |

The remaining 6 are 3 `RecursionError`s and 3 `std::bad_alloc`s inside clingo.

**`UNSUPPORTED` is one feature, plus a stale run.** Re-checking all 2,813 refused tasks against the *current* `supported_kind()`:

| | tasks |
|---|---|
| still refused at HEAD | 1,350 |
| — carrying `FORALL_EFFECTS` | 1,115 |
| — carrying `TIMED_EFFECTS` | 284 |
| **would now be accepted** | **1,463** |

The sweep predates the commits that declared `GENERAL_NUMERIC_PLANNING`, `FLUENTS_IN_NUMERIC_ASSIGNMENTS` and `CONDITIONAL_EFFECTS` (`573585b`, `af5e10c`), so 1,463 tasks — `15-puzzle`, `plotting`, `petrobras`, `fo-farmland`, `pancake`, `tpp`, the four `umts-*` temporal domains — were refused by a version that is no longer HEAD. **Coverage on the numeric and temporal tracks is therefore a lower bound**; a re-run is the cheapest way to improve these numbers.

Everything else is `FORALL_EFFECTS`: a `(forall (?x) (when …))` effect. Neither backend declares it, and at 1,115 tasks it is by a wide margin the largest remaining feature gap.

### Runtime on solved instances

| planner / track | n | median | mean | max | median peak MB |
|---|---|---|---|---|---|
| PLASP, classical | 440 | 4.9 s | 96.6 s | 1747.7 s | 140 |
| PLASP, numeric | 273 | 17.3 s | 190.2 s | 1772.9 s | 176 |
| PLASP, temporal | 24 | 68.1 s | 366.6 s | 1619.3 s | 1145 |
| PLASP, all | 737 | 8.5 s | 140.0 s | 1772.9 s | 152 |
| ABA, classical | 316 | 18.2 s | 173.2 s | 1792.8 s | 287 |

The distribution is sharply bimodal: **53% of solved tasks finish in under 10 s and 73% in under 60 s**, and the rest run into the wall (q90 = 442 s, q95 = 1005 s). Iterative deepening either reaches the right horizon quickly or spends the whole budget grinding through horizons that have no plan.

### The real ceiling is plan length, not instance size

| plan length | solved instances |
|---|---|
| 0–5 | 111 (15%) |
| 6–10 | 279 (38%) |
| 11–15 | 148 (20%) |
| 16–20 | 109 (15%) |
| 21–30 | 60 (8%) |
| 31+ | 30 (4%) |

Median 10, maximum **53**. Because the horizon is deepened one step at a time and each horizon is a fresh satisfiability question, cost grows with the *optimal step count*, and almost nothing past ~30 steps is reachable in 30 minutes. This is the single best predictor of whether an instance is solved — better than object count, better than domain. It also explains the shape of the per-domain results below: domains with short plans are solved nearly completely, domains with long plans are solved not at all.

## 3.3 Classical

**440 / 3,294 (13%); 440 / 1,872 encodable (24%).**

| status | count | share of encodable |
|---|---|---|
| TIMEOUT | 1,263 | 67% |
| SOLVED | 440 | 24% |
| MEMOUT | 138 | 7% |
| KILLED | 29 | 2% |
| EXHAUSTED | 2 | <1% |

The classical track is **search-bound**: two thirds of everything the encoder accepts runs out of time, and only 7% runs out of memory. Grounding is not the problem here — finding the horizon is.

39 of 76 domains yield at least one solved instance:

| domain | solved | rest |
|---|---|---|
| psr-small | 49 / 50 | TIMEOUT 1 |
| no-mprime | 32 / 35 | TIMEOUT 2, KILLED 1 |
| mprime | 31 / 35 | TIMEOUT 4 |
| movie | **30 / 30** | — |
| elevators-00-strips | 30 / 150 | TIMEOUT 120 |
| miconic | 30 / 150 | TIMEOUT 120 |
| blocks | 20 / 136 | TIMEOUT 53, ERROR 56, KILLED 5, MEMOUT 2 |
| mystery | 19 / 30 | TIMEOUT 10, EXHAUSTED 1 |
| no-mystery | 19 / 30 | TIMEOUT 10, EXHAUSTED 1 |
| blocks-3op | 14 / 30 | TIMEOUT 16 |
| pipesworld-notankage | 13 / 50 | TIMEOUT 37 |
| tsp | 13 / 30 | TIMEOUT 17 |

The pattern is exactly the plan-length ceiling: `psr-small`, `movie` and `mprime` have short plans and are solved essentially completely; `elevators-00-strips` and `miconic` each solve a clean prefix — every instance of sizes `s1`–`s6`, nothing from `s7` on — and time out on the remaining 120. The two `EXHAUSTED` results (`mystery:prob07`, `no-mystery:prob07`) reached `max_horizon=1000` without a plan — the only tasks in the whole sweep where the search bound, rather than a resource limit, ended it.

806 classical tasks were refused as `UNSUPPORTED` — the ADL variants (`miconic-fulladl`, `miconic-simpleadl`, `elevators-00-adl`, `schedule`), all on `forall` effects. 616 more failed in the PDDL reader.

## 3.4 Numeric

**273 / 3,354 (8%); 273 / 2,041 encodable (13%).**

| status | count | share of encodable |
|---|---|---|
| TIMEOUT | 1,461 | 72% |
| SOLVED | 273 | 13% |
| KILLED | 216 | 11% |
| MEMOUT | 46 | 2% |
| ERROR | 45 | 2% |

Split by source, since the track mixes two very different populations:

| source | solved | tasks | domains |
|---|---|---|---|
| IPC classical instances with `(total-cost)` | 177 | 1,537 | 69 |
| genuinely numeric domains (`numeric-domains`) | 96 | 1,817 | 93 |

The 177 are effectively classical coverage under a different label — the metric is accepted and ignored, so these behave like §3.3. The 96 are the real numeric result.

| domain | solved | rest |
|---|---|---|
| mprime | 28 / 30 | TIMEOUT 2 |
| ged-opt14-strips | **20 / 20** | — |
| minecraft-sword-advanced | **20 / 20** | — |
| minecraft-pogo-advanced | 17 / 20 | TIMEOUT 1, ERROR 2 |
| data-network-opt18 | 15 / 20 | TIMEOUT 5 |
| organic-synthesis-split-opt18 | 12 / 20 | TIMEOUT 4, MEMOUT 4 |
| pegsol-08-strips | 12 / 30 | TIMEOUT 18 |
| counters | 10 / 55 | TIMEOUT 44, MEMOUT 1 |
| block-grouping | 9 / 192 | KILLED 167, TIMEOUT 15, MEMOUT 1 |
| organic-synthesis-split-sat18 | 9 / 20 | MEMOUT 9, TIMEOUT 2 |

40 of 162 domains yield at least one solved instance. `minecraft-*` and `ged` — short plans, small numeric state — go 20/20 and 17/20. `counters` and `block-grouping` are the opposite: small tasks whose plans are long, and they fail almost completely (`block-grouping`'s 167 `KILLED` are slurm reaping jobs at the 35-minute wall before the harness's own 30-minute alarm could record a `TIMEOUT`; read them as timeouts).

`UNSUPPORTED` is heaviest here: 1,291 tasks, 1,091 of them from `numeric-domains`. **This is the number most distorted by the stale run** — a large share of the refusals are `GENERAL_NUMERIC_PLANNING` / `FLUENTS_IN_NUMERIC_ASSIGNMENTS`, both declared at HEAD. Only 22 numeric tasks failed in the PDDL reader, so this track is the cleanest input of the three and the one where a re-run should move the most.

## 3.5 Temporal

**24 / 2,680 (1%); 24 / 1,236 encodable (2%).** This is the weak track, and the reason is specific and measurable.

| status | count | share of encodable |
|---|---|---|
| **MEMOUT** | **875** | **71%** |
| TIMEOUT | 269 | 22% |
| KILLED | 35 | 3% |
| ERROR | 33 | 3% |
| SOLVED | 24 | 2% |

**The temporal encoding does not run out of time — it runs out of memory, early.** The 875 memouts reach a median peak of **7.25 GB after a median of 146 seconds**: they exhaust an 8 GB budget in under three minutes, before search has meaningfully started. That is grounding, not solving. The direct cause is documented in [§2](#time_scale): the remaining-duration recursion is **quadratic in the largest scaled duration**, and the sweep ran at `time_scale=10`, which multiplies every duration by ten before that recursion is ground.

Even the solved instances show it — median peak memory 1,145 MB against 140 MB on classical, an 8× gap.

| domain | solved | rest |
|---|---|---|
| peg-solitaire-temporal-satisficing-strips | 8 / 30 | TIMEOUT 22 |
| peg-solitaire-temporal-satisficing | 4 / 20 | TIMEOUT 16 |
| airport-temporal-strips | 3 / 50 | MEMOUT 32, TIMEOUT 15 |
| pipesworld-no-tankage-temporal-strips | 2 / 50 | MEMOUT 20, TIMEOUT 28 |
| rovers-time-simple-automatic | 2 / 20 | MEMOUT 10, TIMEOUT 8 |
| depots-time-simple-automatic | 1 / 22 | MEMOUT 5, TIMEOUT 16 |
| driverlog-time-simple-automatic | 1 / 20 | MEMOUT 10, TIMEOUT 9 |
| pipesworld-metric-time | 1 / 50 | MEMOUT 25, TIMEOUT 24 |
| pipesworld-tankage-temporal-strips | 1 / 50 | MEMOUT 25, TIMEOUT 24 |
| satellite-time-simple-automatic | 1 / 20 | MEMOUT 12, TIMEOUT 6, KILLED 1 |

10 of 94 domains yield a solved instance. The two `peg-solitaire` domains are the only ones that scale at all, and tellingly they are the ones with unit durations — nothing for `time_scale` to blow up. Every domain with heterogeneous real durations (`depots`, `driverlog`, `rovers`, `satellite`, `pipesworld`) solves exactly its first instance or two and then memouts.

The solved plans are correct time-triggered plans with genuine concurrency — `airport-temporal-strips:instance-4` is 20 snap actions at makespan 129.0 in 12.7 s; `driverlog-time-simple-automatic:instance-1` is makespan 92.7 in 12.3 s. The encoding is right; it is the discretisation constant that is too expensive.

716 temporal tasks were refused as `UNSUPPORTED` (the `airport-temporal-*-adl` family on `forall` effects, plus 284 on `TIMED_EFFECTS` — timed initial literals, which the whole `*-time-windows-*` family uses). Another 728 failed in the PDDL reader — the highest of any track, mostly `SyntaxError` on `total-time` and `duration` expressions the UP reader rejects.

**The actionable conclusion:** lower `time_scale`. The default of 10 buys ε-separation precision that domains without required concurrency do not need, and costs quadratically. Re-running the temporal track at `time_scale=1` or `2` is the obvious next experiment.

## 3.6 `PLASPPlanner` vs `ABAPlanner`

`ABAPlanner-ST` was configured for the **classical track only** — its planner config carries `"tracks": ["classical"]`, because the ABA framework grows with the square of the largest scaled duration. Its 0/3,354 numeric and 0/2,800 temporal in the raw summary are **tasks that were never run**, not failures. The comparison below is the 3,294 classical tasks both planners actually attempted.

| | `PLASPPlanner-seq` | `ABAPlanner-ST` |
|---|---|---|
| solved | **440** | 316 |
| encodable | 1,872 | 1,811 |
| coverage of encodable | **24%** | 17% |
| median runtime (solved) | **4.9 s** | 18.2 s |
| median peak memory (solved) | **140 MB** | 287 MB |
| MEMOUT | 138 (7% of encodable) | **966 (53%)** |
| TIMEOUT + KILLED | 1,292 | 525 |
| UNSUPPORTED | 806 | 836 |

Head to head:

| | tasks |
|---|---|
| both solved | 314 |
| **PLASP only** | **126** |
| **ABA only** | **2** |
| neither | 2,852 |

PLASP strictly dominates: it solves everything ABA solves bar two instances, and on those two it timed out rather than failing. **PLASP is the default for a reason** — the ABA reduction is 4× slower at the median, 2× heavier, and its failure mode is memory: 53% of the tasks it accepts exhaust 8 GB, against 7% for PLASP, because the STRIPS-to-ABA reduction materialises an argumentation framework over the ground task. Its 387 `KILLED` are a consequence of `ABAPlanner` not honoring `timeout` — the scheduler reaps the job instead of the harness recording a clean `TIMEOUT`.

`ABAPlanner` also refuses slightly more: no `CONDITIONAL_EFFECTS` (836 tasks, all of them), and no `UNDEFINED_INITIAL_*`.

## 3.7 Summary, and what to fix next

1. **Plan length is the binding constraint.** Median solved plan is 10 steps, maximum 53, 88% at 20 or under. Iterative deepening pays for every horizon it rules out; nothing else in the data predicts coverage as well. Any large gain has to come from a better horizon strategy (a lower bound from a relaxed plan, or a planning-graph-style bound), not from a faster encoding.
2. **`FORALL_EFFECTS` is the biggest feature gap** — 1,115 tasks refused up front, including whole ADL families.
3. **The temporal track is grounding-bound, at a constant you control.** 71% of encodable temporal tasks exhaust 8 GB in under three minutes at `time_scale=10`. Lowering it is the cheapest available improvement to the weakest track.
4. **These numbers are a lower bound.** 1,463 tasks refused by the benchmarked version are accepted at HEAD. The numeric track in particular should be re-run before its coverage is quoted anywhere.
5. **The benchmark input is dirtier than the planner.** 1,366 tasks — 15% of the sweep, and 95% of all `ERROR`s — never reached the encoder because the UP PDDL reader could not parse the IPC files. That is a ceiling on what *any* UP-based planner can score on these suites, not an ASPPlanners result.

## Reproducing

The harness lives in [benchmarks/](benchmarks/) and is documented in [benchmarks/README.md](benchmarks/README.md).

```bash
cd benchmarks
./setup_benchmark.sh          # asks for the time/memory limit, then does everything
```

It creates a virtualenv, installs ASPPlanners and `aspbench` into it, clones the three benchmark repositories, writes an experiment with the limits you gave, and generates the slurm job arrays. Scripted:

```bash
./setup_benchmark.sh --time-limit 30m --memory-limit 8GB \
                     --tracks "classical numeric temporal" --partition compute --yes
```

The five stages are separable, and only `solve` imports `unified_planning` — so a sweep can be generated on a laptop and run on the cluster:

```
aspbench init      → an experiment directory (limits + planner configurations)
aspbench discover  → what tasks a benchmark repository holds
aspbench generate  → one run command per (planner, task), plus slurm arrays
aspbench solve     → run ONE pair under its limits, dump a JSON result   (slurm calls this)
aspbench analyze   → results.csv + the coverage report reproduced above
aspbench report    → paper-ready tables (text + LaTeX) and figures
```

Two caveats when reading a sweep of your own:

- `task_id` is `suite:domain:instance` and does **not** include the IPC year, so a domain appearing in two IPC editions collides. In this run 120 temporal tasks collided that way (`floor-tile-temporal-satisficing` and friends, IPC-2011 vs IPC-2014), which is why 9,448 generated commands produce 9,328 unique results.
- `MISSING` in the raw summary means "no result file for a (planner, task) pair", which conflates *never scheduled* (`ABAPlanner-ST` on numeric and temporal) with *lost*. Check the planner config's `tracks` key before reading a zero as a failure.
