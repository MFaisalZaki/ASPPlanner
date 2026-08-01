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
| `forall` **effects** | ✅ | ✅ unconditional only | native in PLASP (the quantifier is expanded by the grounder, like a `forall` condition); compiled away for ABA, so `(forall … (when …))` is still refused there as a conditional effect |
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
| `FORALL_EFFECTS` | ✅ | ✅ (expanded by the pipeline's quantifier remover) |
| `UNDEFINED_INITIAL_NUMERIC`, `UNDEFINED_INITIAL_SYMBOLIC` | ✅ | ❌ |
| `CONTINUOUS_TIME`, `DURATION_INEQUALITIES`, `INT_TYPE_DURATIONS`, `REAL_TYPE_DURATIONS`, `STATIC_FLUENTS_IN_DURATIONS` | ✅ | ✅ |
| `MAKESPAN`, `ACTIONS_COST`, `PLAN_LENGTH`, `FINAL_VALUE` and the four `*_IN_ACTIONS_COST` kinds | ⚠️ accepted, never optimised | ⚠️ same |
| `TIMED_EFFECTS`, `TIMED_GOALS`, `PROCESSES`, `EVENTS`, `SELF_OVERLAPPING`, `INTERMEDIATE_CONDITIONS_AND_EFFECTS`, `OVERSUBSCRIPTION` | ❌ | ❌ |

A task outside the declared kind is refused up front with `UNSUPPORTED_PROBLEM` rather than silently mis-encoded. `ProblemKind` cannot express every distinction, so a few shapes are raised at *encoding* time instead, as `NotImplementedError`:

- a product or quotient of two numeric fluents (not linear),
- a fractional coefficient in a numeric effect,
- a bounded numeric type on a task that needs rescaling,
- a **conditional** numeric effect other than the assignment of a constant — an `increase` or a value read off the state is applied by `numEffect`/`numAssignExpr`, which hang off `occurs/2` with no room for the effect's condition,
- a numeric over-all condition in a durative action (**ABA backend only**).

`GENERAL_NUMERIC_PLANNING` is declared because an effect that merely *reads* a fluent already pushes a task's kind there, and the feature has no linear/non-linear split.

## Nothing is compiled away

Before a problem reaches the ASP encoder it is put through a list of UP compilers — by default, **none of them**. Every condition shape is stated in the encoding itself:

| shape | how it is encoded |
|---|---|
| negative conditions | the encoding is multi-valued, so `value(V, false)` is a value like any other. A mirror fluent per negatively-read one would be pure overhead; the encoder just emits the false initial value for the fluents actually read as false |
| `forall` | a conjunction over the universe — and `precondition`/`goal` facts are already conjunctive. One rule with the variable left free and `has(_, type(...))` in the body; gringo does the expanding |
| `forall` **effects** | the same free variable, on the effect side. `(forall (?o) (not (in ?o)))` is one `postcondition` rule ranged by `has(Q_O, type(...))`. A conditional one — `(forall (?o) (when (in ?o) (at ?o ?to)))` — additionally *indexes its effect term* by the binding: `effect((cond,"move",0,FROM,TO,Q_O))`. That is the whole semantics. `caused/3` fires an effect only when **every** `precondition` of its effect term holds, so one term shared across the bindings would say "move the objects iff *all* of them are inside"; a term per binding says "move each object that is" |
| `or`, `exists` | disjunctions, which conjunctive facts cannot state, so they get their own `orGroup`/`orDisjunct` vocabulary: at least one disjunct has to hold, and a disjunct holds when all of its literals do. An `exists` is the same shape with its disjuncts indexed by the quantified variable's binding |
| an action that sets **and** clears one fluent | PDDL's own rule, arbitrated at the step rather than by deleting one of the effects: the add wins. Only an add that *always* fires shadows a delete up front; a conditional or quantified one leaves both in, and `core.lp` settles the bindings where they actually collide. This is what the ADL `forall`/`when` pairs need — miconic's `stop` clears `boarded` for the arrivals and sets it for the boarders, and dropping the clear strands every passenger aboard |
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
| `core` | [core.lp](aspplanners/plasp/encodings/seq/core.lp) | multi-valued STRIPS over the horizon: the action choice, preconditions, (conditional) effects, add-wins arbitration, inertia, the goal test |
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

Every number below comes from the checked-in tables in [results/](results/) — per-instance and per-domain, one directory per track — distilled from the sweep's sandbox by [`results/generate_results.py`](results/generate_results.py).

## 3.1 Setup

One sweep, run 2026-07-29 → 2026-07-31 through the [aspbench](benchmarks/) harness on a slurm cluster (one CPU per task).

| | |
|---|---|
| **Benchmark set** | 6,738 unique tasks over 247 domains; 10,032 (planner, task) pairs |
| **Time limit** | 1800 s wall per task (slurm job capped at 35 min) |
| **Memory limit** | 8192 MB per task (slurm job capped at 9 GB) |
| **`PLASPPlanner-seq`** | `encoding=seq`, `max_horizon=1000`, `time_scale=10`, all three tracks |
| **`ABAPlanner-ST`** | `semantics=ST`, `max_horizon=100`, `time_scale=2`, **classical track only** |

| track | tasks | domains | source |
|---|---|---|---|
| classical | 3,294 | 76 | [AI-Planning/classical-domains](https://github.com/AI-Planning/classical-domains) |
| numeric | 3,354 | 162 | [pyPMT/numeric-domains](https://github.com/pyPMT/numeric-domains) (1,817) + classical-domains instances carrying a `(:functions …)` block (1,537) |
| temporal | 90 | 9 | [nergmada/ipc2018-temporal-track](https://github.com/nergmada/ipc2018-temporal-track) — the IPC-2018 temporal track as competed, 9 domains × 10 instances |

A task's track is decided by **reading its domain file**, not by which repository it came from — which is why 1,537 IPC "classical" instances land on the numeric track: they carry a `(total-cost)` function.

**What changed since the previous sweep.** This run is on a version that declares `FORALL_EFFECTS`, `GENERAL_NUMERIC_PLANNING`, `FLUENTS_IN_NUMERIC_ASSIGNMENTS` and `CONDITIONAL_EFFECTS` (`573585b`, `af5e10c`, `82e4817`, `b88917f`) — the features the previous sweep's `UNSUPPORTED` column was almost entirely made of — and on the IPC-2018 temporal set rather than the 2,680-task multi-track archive. Classical and numeric are directly comparable; **temporal is not**, it is a different and much smaller benchmark.

| track | previous sweep | this sweep |
|---|---|---|
| classical | 440 / 3,294 | **612 / 3,294** |
| numeric | 273 / 3,354 | **387 / 3,354** |
| temporal | 24 / 2,680 | 5 / 90 *(different set)* |

The raw run artifacts (per-task JSONs, tracebacks, slurm logs — ~20 GB) are not checked in; `sandbox*` and `benchmark-run/` are gitignored. The distilled tables are, under [results/](results/). See [Reproducing](#reproducing).

### Reading the status column

| status | meaning |
|---|---|
| `SOLVED` | a plan was found **and validated against the original problem** |
| `UNSUPPORTED` | refused up front — the task's `ProblemKind` is outside `supported_kind()`, or the encoder declined a numeric expression it cannot linearise |
| `ERROR` | an exception; 99% of them the **UP PDDL reader**, not the planner (see below) |
| `TIMEOUT` | hit the 1800 s task limit |
| `MEMOUT` | hit the 8 GB task limit |
| `KILLED` | the scheduler reaped the job before the harness could write a result — read these as timeouts |
| `EXHAUSTED` | `max_horizon` reached with no plan — the task may still be solvable at a deeper horizon |

## 3.2 Headline coverage

`PLASPPlanner-seq`, all three tracks:

| track | solved | of all tasks | of *encodable* tasks |
|---|---|---|---|
| classical | **612** | 612 / 3,294 (19%) | 612 / 2,678 (**23%**) |
| numeric | **387** | 387 / 3,354 (12%) | 387 / 2,832 (**14%**) |
| temporal | **5** | 5 / 90 (6%) | 5 / 90 (**6%**) |
| **total** | **1,004** | 1,004 / 6,738 (15%) | 1,004 / 5,600 (**18%**) |

*Encodable* = attempted, minus `UNSUPPORTED`, minus the tasks the **UP PDDL reader** could not parse. Both columns matter and they say different things: the left one is raw IPC coverage, the right one is how the search does on the tasks that actually reach clingo.

Full status breakdown:

| planner | SOLVED | UNSUPPORTED | TIMEOUT | ERROR | MEMOUT | KILLED | EXHAUSTED | total |
|---|---|---|---|---|---|---|---|---|
| `PLASPPlanner-seq` | 1,004 | 497 | 3,475 | 641 | 624 | 471 | 26 | 6,738 |
| `ABAPlanner-ST` (classical only) | 318 | 836 | 171 | 647 | 948 | 374 | 0 | 3,294 |

Every one of the 1,322 `SOLVED` results across both planners passed plan validation — no unvalidated plan was counted.

### The failure modes, separated

**`UNSUPPORTED` is now a numeric-encoding question only.** `PLASPPlanner` refuses **nothing** on the classical and temporal tracks: the previous sweep's 2,813 refusals — the whole ADL family on `forall` effects, the numeric domains on `GENERAL_NUMERIC_PLANNING` — are gone, and the 497 that remain are all on the numeric track and all *encoding-time* refusals, where the encoder has read the task and declined a specific expression:

| encoder refusal | tasks | domains |
|---|---|---|
| `A product of two numeric fluents is not linear` | 287 | the whole `nlnp-*` family, `zenotravel`, `sailing-wind-*`, `factory-robot`, `gear-car`, … |
| `The numeric effect on X reads Y with a fractional coefficient` | 100 | `fo-farmland` (50), `hydropower` (30), `fo-sailing` (20) |
| `The conditional numeric effect … is not supported` | 70 | `petrobras` |
| `Making this task's numeric values integral needs a scale factor of 5e12` | 40 | `worksworld` |

These are honest refusals rather than wrong answers, and they mark exactly where the linear-integer ASP encoding stops: products of two fluents, coefficients that do not scale to integers, and numeric effects under a condition.

`ABAPlanner`'s 836 refusals are one feature — `CONDITIONAL_EFFECTS`, which is what every `(forall … (when …))` effect in this corpus expands into. **`PLASPPlanner` solves 174 of those 836** (`schedule` 37, `miconic-fulladl` 35, `elevators-00-adl` 34, `miconic-simpleadl` 34, `airport-adl` 11, `briefcaseworld` 7, `nurikabe-opt18` 7, `maintenance-opt14-adl` 5, `caldera-opt18` 4) — the concrete payoff of encoding `forall`/`when` rather than refusing it, and most of the classical track's gain over the previous sweep.

**`ERROR` is the front end, not the planner.** Of 641 PLASP errors, **638 (99.5%)** are the UP PDDL reader failing on the benchmark files themselves:

| exception | count | typical cause |
|---|---|---|
| `SyntaxError` | 224 | expressions the reader will not take (`total-time`, `duration`, `preference`) |
| `ParseException` | 199 | malformed / unsupported PDDL syntax |
| `UPProblemDefinitionError` | 129 | `Name p1 already defined` — an object and a predicate sharing a name |
| `KeyError` | 46 | a type name used before it is declared |
| `ParseSyntaxException` | 30 | same as `ParseException` |
| `UPExpressionDefinitionError` | 10 | wrong fluent arity in the instance file |

Seven domains account for 555 of the 641: `logistics00` (174), `elevators-00-full` (129), `blocks` (56), `psr-middle` and `psr-large` (50 each), `optical-telegraphs` and `philosophers` (48 each). The encoder never sees these tasks. The remaining 3 errors are `RecursionError`s in `plotting`.

**`KILLED` is the 35-minute slurm wall firing before the harness's own 30-minute alarm.** 471 tasks, and they concentrate in a handful of numeric domains that ground slowly — `block-grouping` (167), `15-puzzle` (64), `pancake` (29), `2048` (16). Read them as timeouts; none of them was close to a plan.

### Runtime on solved instances

| planner / track | n | median | mean | q90 | max | median peak MB |
|---|---|---|---|---|---|---|
| PLASP, classical | 612 | 5.3 s | 108.4 s | 366.5 s | 1767.6 s | 136 |
| PLASP, numeric | 387 | 41.4 s | 377.1 s | 1763.2 s | 2077.3 s | 184 |
| PLASP, temporal | 5 | 15.0 s | 15.8 s | — | 40.8 s | 245 |
| PLASP, all | 1,004 | 10.5 s | 211.5 s | 744.0 s | 2077.3 s | 148 |
| ABA, classical | 318 | 19.3 s | 176.4 s | 554.5 s | 1772.4 s | 292 |

The distribution is sharply bimodal: **50% of solved tasks finish in under 10 s and 69% in under 60 s**, and the rest run into the wall (q90 = 744 s, q95 = 1482 s). Iterative deepening either reaches the right horizon quickly or spends the whole budget grinding through horizons that have no plan.

The memouts say the same thing from the other side: a PLASP `MEMOUT` reaches a median peak of **7.3 GB** after a median of **184 s** (classical) or **287 s** (numeric). Where memory is the binding limit it binds early, during grounding — not after a long search.

### The real ceiling is plan length, not instance size

| plan length | solved instances |
|---|---|
| 0–5 | 186 (19%) |
| 6–10 | 358 (36%) |
| 11–15 | 206 (21%) |
| 16–20 | 144 (14%) |
| 21–30 | 71 (7%) |
| 31+ | 39 (4%) |

Median 10, maximum **53** — unchanged from the previous sweep despite 36% more solved tasks. Because the horizon is deepened one step at a time and each horizon is a fresh satisfiability question, cost grows with the *optimal step count*, and almost nothing past ~30 steps is reachable in 30 minutes. This is the single best predictor of whether an instance is solved — better than object count, better than domain. It also explains the shape of the per-domain results below: domains with short plans are solved nearly completely, domains with long plans are solved not at all.

## 3.3 Classical

**612 / 3,294 (19%); 612 / 2,678 encodable (23%).** Full tables: [results/classical/](results/classical/).

| status | count | share of encodable |
|---|---|---|
| TIMEOUT | 1,763 | 66% |
| SOLVED | 612 | 23% |
| MEMOUT | 206 | 8% |
| KILLED | 74 | 3% |
| EXHAUSTED | 23 | 1% |
| UNSUPPORTED | **0** | — |

The classical track is **search-bound**: two thirds of everything the encoder accepts runs out of time, and only 8% runs out of memory. Grounding is not the problem here — finding the horizon is. Nothing on this track is refused any more, so `encodable` now differs from `attempted` only by the 616 tasks the PDDL reader rejected.

48 of 76 domains yield at least one solved instance:

| domain | solved | rest |
|---|---|---|
| psr-small | 49 / 50 | TIMEOUT 1 |
| schedule | 37 / 150 | TIMEOUT 113 |
| miconic-fulladl | 35 / 150 | TIMEOUT 111, MEMOUT 4 |
| miconic-simpleadl | 34 / 150 | TIMEOUT 116 |
| elevators-00-adl | 34 / 151 | TIMEOUT 116, ERROR 1 |
| mprime | 32 / 35 | TIMEOUT 2, KILLED 1 |
| no-mprime | 31 / 35 | TIMEOUT 2, KILLED 2 |
| movie | **30 / 30** | — |
| elevators-00-strips | 30 / 150 | TIMEOUT 120 |
| miconic | 30 / 150 | TIMEOUT 120 |
| blocks | 20 / 136 | ERROR 56, TIMEOUT 54, KILLED 6 |
| mystery | 18 / 30 | TIMEOUT 10, EXHAUSTED 1, KILLED 1 |
| no-mystery | 18 / 30 | TIMEOUT 10, EXHAUSTED 1, KILLED 1 |
| blocks-3op | 14 / 30 | TIMEOUT 16 |
| tsp | 13 / 30 | TIMEOUT 17 |
| ferry | 12 / 30 | TIMEOUT 18 |

The four ADL families in that list — `schedule`, `miconic-fulladl`, `miconic-simpleadl`, `elevators-00-adl` — contributed **nothing at all** to the previous sweep; all 601 of their tasks were refused on `forall` effects. They now behave exactly like their STRIPS siblings: a clean prefix solved, the rest timing out. That is where 140 of the 172 extra classical solutions come from.

The pattern otherwise is the plan-length ceiling: `psr-small`, `movie` and `mprime` have short plans and are solved essentially completely; `elevators-00-strips` and `miconic` each solve every instance of sizes `s1`–`s6` and nothing from `s7` on. The 23 `EXHAUSTED` results (21 in `elevators-00-full`, one each in `mystery` and `no-mystery`) reached `max_horizon=1000` without a plan — the only classical tasks where the search bound, rather than a resource limit, ended the run.

## 3.4 Numeric

**387 / 3,354 (12%); 387 / 2,832 encodable (14%).** Full tables: [results/numeric/](results/numeric/).

| status | count | share of encodable |
|---|---|---|
| TIMEOUT | 1,683 | 59% |
| KILLED | 392 | 14% |
| SOLVED | 387 | 14% |
| MEMOUT | 367 | 13% |
| EXHAUSTED | 3 | <1% |

Split by source, since the track mixes two very different populations:

| source | tasks | domains | solved | UNSUPPORTED | ERROR |
|---|---|---|---|---|---|
| IPC classical instances with `(total-cost)` | 1,537 | 69 | 206 | 0 | 2 |
| genuinely numeric domains (`numeric-domains`) | 1,817 | 93 | 181 | 497 | 23 |

The 206 are effectively classical coverage under a different label — the metric is accepted and ignored, so these behave like §3.3. The 181 are the real numeric result, and they are up from 96 in the previous sweep: declaring `GENERAL_NUMERIC_PLANNING` and `FLUENTS_IN_NUMERIC_ASSIGNMENTS` turned whole domains from a refusal into a run.

71 of 162 domains yield at least one solved instance:

| domain | solved | rest |
|---|---|---|
| plotting | **76 / 87** | KILLED 5, EXHAUSTED 3, ERROR 3 |
| mprime | 24 / 30 | TIMEOUT 6 |
| ged-opt14-strips | **20 / 20** | — |
| minecraft-sword-advanced | **20 / 20** | — |
| data-network-opt18 | 15 / 20 | TIMEOUT 5 |
| organic-synthesis-split-opt18 | 12 / 20 | MEMOUT 5, TIMEOUT 3 |
| pegsol-08-strips | 12 / 30 | TIMEOUT 18 |
| counters | 10 / 55 | TIMEOUT 44, KILLED 1 |
| minecraft-pogo-advanced | 9 / 20 | TIMEOUT 11 |
| organic-synthesis-split-sat18 | 9 / 20 | MEMOUT 8, TIMEOUT 3 |
| woodworking-opt08-strips | 9 / 30 | TIMEOUT 21 |
| transport-opt08-strips | 9 / 30 | TIMEOUT 21 |
| caldera-split-opt18 | 8 / 20 | TIMEOUT 12 |
| cybersec | 8 / 30 | TIMEOUT 21, KILLED 1 |

`plotting` is the single largest gain of the sweep: 87 tasks refused outright before, 76 solved now. `minecraft-*` and `ged` — short plans, small numeric state — go 20/20. `counters` and `block-grouping` (7/192) are the opposite: small tasks whose plans are long, and they fail almost completely.

The refusals that remain are concentrated and diagnostic: the 13 `nlnp-*` and `sailing-wind-*` domains built on non-linear arithmetic (287 tasks), `fo-farmland`/`hydropower`/`fo-sailing` on fractional coefficients (100), `petrobras` on conditional numeric effects (70), and `worksworld` on a scale factor of 5·10¹² (40). Non-linear arithmetic is the single feature standing between `PLASPPlanner` and the rest of this track.

## 3.5 Temporal

**5 / 90 (6%).** This is the weak track, and the reason is specific and measurable. Full tables: [results/temporal/](results/temporal/).

Note that this is the IPC-2018 temporal track — 9 domains × 10 instances, the hardest temporal benchmark in circulation — not the older multi-track archive the previous sweep used. The two numbers are not comparable.

| status | count | share |
|---|---|---|
| **MEMOUT** | **51** | **57%** |
| TIMEOUT | 29 | 32% |
| SOLVED | 5 | 6% |
| KILLED | 5 | 6% |
| UNSUPPORTED | **0** | — |

**The temporal encoding does not run out of time — it runs out of memory, early.** The 51 memouts reach a median peak of **7.3 GB after a median of 117 seconds**: they exhaust an 8 GB budget in under two minutes, before search has meaningfully started. That is grounding, not solving. The direct cause is documented in [§2](#time_scale): the remaining-duration recursion is **quadratic in the largest scaled duration**, and the sweep ran at `time_scale=10`, which multiplies every duration by ten before that recursion is ground.

| domain | solved | rest |
|---|---|---|
| airport-temporal-strips | 2 / 10 | MEMOUT 8 |
| quantum_circuit | 2 / 10 | MEMOUT 5, TIMEOUT 3 |
| Cushing | 1 / 10 | TIMEOUT 9 |
| Floortile | 0 / 10 | TIMEOUT 10 |
| Parking | 0 / 10 | TIMEOUT 7, MEMOUT 3 |
| Mapanalyser | 0 / 10 | MEMOUT 8, KILLED 2 |
| road-traffic-accident | 0 / 10 | MEMOUT 8, KILLED 2 |
| sokoban | 0 / 10 | MEMOUT 9, KILLED 1 |
| trucks-time-strips | 0 / 10 | MEMOUT 10 |

The split is clean: the three domains that solve anything (`Cushing`, `quantum_circuit`, `airport-temporal-strips`) are the ones whose durations are small or uniform, and they time out rather than memout. Every domain with large heterogeneous real durations — `trucks-time-strips`, `sokoban`, `Mapanalyser`, `road-traffic-accident` — memouts wholesale.

The solved plans are correct time-triggered plans with genuine concurrency:

| task | plan | makespan | time | peak MB |
|---|---|---|---|---|
| `airport-temporal-strips:4` | 20 snap actions | 129.0 | 15.0 s | 281 |
| `airport-temporal-strips:10` | 18 | 127.8 | 17.2 s | 278 |
| `quantum_circuit:3` | 5 | 17.0 | 40.8 s | 245 |
| `quantum_circuit:1` | 4 | 12.5 | 4.1 s | 192 |
| `Cushing:pfile1` | 6 | 8.0 | 1.8 s | 139 |

The encoding is right; it is the discretisation constant that is too expensive. **The actionable conclusion is unchanged: lower `time_scale`.** The default of 10 buys ε-separation precision that domains without required concurrency do not need, and costs quadratically. Re-running the temporal track at `time_scale=1` or `2` is still the obvious next experiment — and it is now cheap, at 90 tasks.

## 3.6 `PLASPPlanner` vs `ABAPlanner`

`ABAPlanner-ST` was configured for the **classical track only** — its planner config carries `"tracks": ["classical"]`, because the ABA framework grows with the square of the largest scaled duration. Its numeric and temporal tasks were never run, and are absent from [results/numeric/](results/numeric/) and [results/temporal/](results/temporal/) rather than recorded as 0/n. The comparison below is the 3,294 classical tasks both planners attempted.

| | `PLASPPlanner-seq` | `ABAPlanner-ST` |
|---|---|---|
| solved | **612** | 318 |
| encodable | 2,678 | 1,811 |
| coverage of encodable | **23%** | 18% |
| median runtime (solved) | **5.3 s** | 19.3 s |
| median peak memory (solved) | **136 MB** | 292 MB |
| MEMOUT | 206 (8% of encodable) | **948 (52%)** |
| TIMEOUT + KILLED | 1,837 | 545 |
| UNSUPPORTED | **0** | 836 |

Head to head:

| | tasks |
|---|---|
| both solved | 314 |
| **PLASP only** | **298** |
| **ABA only** | **4** |
| neither | 2,678 |

PLASP strictly dominates: it solves everything ABA solves bar four instances (`blocks:probBLOCKS-10-1`, `driverlog:pfile2`, `hiking-opt14-strips:ptesting-1-2-5`, `zenotravel:pfile7`), and on those four it timed out rather than failing. **PLASP is the default for a reason** — the ABA reduction is 4× slower at the median, 2× heavier, and its failure mode is memory: 52% of the tasks it accepts exhaust 8 GB, against 8% for PLASP, because the STRIPS-to-ABA reduction materialises an argumentation framework over the ground task. Its 374 `KILLED` are a consequence of `ABAPlanner` not honoring `timeout` — the scheduler reaps the job instead of the harness recording a clean `TIMEOUT`.

The gap has widened since the previous sweep (126 PLASP-only tasks then, 298 now) almost entirely because of `CONDITIONAL_EFFECTS`: ABA still refuses all 836 ADL tasks, and PLASP now solves 174 of them.

## 3.7 Summary, and what to fix next

1. **Plan length is the binding constraint.** Median solved plan is 10 steps, maximum 53, 89% at 20 or under — unchanged even though 36% more tasks are solved. Iterative deepening pays for every horizon it rules out; nothing else in the data predicts coverage as well. Any large gain has to come from a better horizon strategy (a lower bound from a relaxed plan, or a planning-graph-style bound), not from a faster encoding.
2. **The feature gap that dominated the previous sweep is closed.** `PLASPPlanner` refuses nothing on classical and nothing on temporal. The 2,813 `UNSUPPORTED` verdicts of the previous run are down to 497, all numeric, all encoding-time — and the tasks that unblocked are worth 172 extra classical and 114 extra numeric solutions.
3. **What remains refused is non-linear arithmetic.** 287 of the 497 refusals are a product of two fluents; another 100 are coefficients that will not scale to integers, 70 are conditional numeric effects, 40 an unrepresentable scale factor. That is a real boundary of a linear-integer ASP encoding, not an oversight — but conditional numeric effects (`petrobras`, 70 tasks) are the one item on that list that looks reachable.
4. **The temporal track is grounding-bound, at a constant you control.** 57% of temporal tasks exhaust 8 GB in under two minutes at `time_scale=10`. Lowering it is the cheapest available improvement to the weakest track, and at 90 tasks the experiment costs a few hours.
5. **The benchmark input is dirtier than the planner.** 641 tasks — 10% of the sweep, and 99.5% of all `ERROR`s — never reached the encoder because the UP PDDL reader could not parse the IPC files, six domains accounting for most of it. That is a ceiling on what *any* UP-based planner can score on these suites, not an ASPPlanners result.

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

The checked-in tables under [results/](results/) are distilled from the finished sandbox with:

```bash
python results/generate_results.py --sandbox-dir sandbox-results/sandbox
```

Two caveats when reading a sweep of your own:

- `task_id` is `suite:domain:instance` and does **not** include the IPC year, so a domain appearing in two IPC editions collides — the previous sweep lost 120 temporal tasks that way. The IPC-2018 temporal set has no such duplicates, so this run's 10,032 generated commands produce exactly 10,032 result rows.
- `MISSING` in the raw summary means "no result file for a (planner, task) pair", which conflates *never scheduled* (`ABAPlanner-ST` on numeric and temporal) with *lost*. Check the planner config's `tracks` key before reading a zero as a failure; `results/generate_results.py` drops the never-scheduled pairs rather than recording them as 0/n.
