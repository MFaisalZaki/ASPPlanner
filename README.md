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
| classical | 4,831 | 145 | [AI-Planning/classical-domains](https://github.com/AI-Planning/classical-domains) |
| numeric | 1,817 | 93 | [pyPMT/numeric-domains](https://github.com/pyPMT/numeric-domains) |
| temporal | 90 | 9 | [nergmada/ipc2018-temporal-track](https://github.com/nergmada/ipc2018-temporal-track) — the IPC-2018 temporal track as competed, 9 domains × 10 instances |

A task's track is decided by **reading its domain file**, not by which repository it came from. Declaring a function is not enough to make a domain numeric: IPC's standard STRIPS encoding declares `(total-cost)` purely so plans can be scored, so a fluent has to be *used as state* — compared in a precondition, or assigned somewhere other than a cost accumulator. Filing on the declaration alone put 1,537 cost-annotated IPC instances (69 domains: `sokoban-*`, `woodworking-*`, `transport-*`, …) on the numeric track; they are classical, and are counted as such here.

| track | previous sweep | this sweep |
|---|---|---|
| classical | 440 / 3,294 | **818 / 4,831** |
| numeric | 273 / 3,354 | **181 / 1,817** |
| temporal | 24 / 2,680 | 5 / 90 *(different set)* |

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

## 3.7 Summary, and what to fix next

1. **Plan length is the binding constraint.** Median solved plan is 10 steps, maximum 53, 89% at 20 or under — unchanged even though 36% more tasks are solved. Iterative deepening pays for every horizon it rules out; nothing else in the data predicts coverage as well. Any large gain has to come from a better horizon strategy (a lower bound from a relaxed plan, or a planning-graph-style bound), not from a faster encoding.
2. **The feature gap that dominated the previous sweep is closed.** `PLASPPlanner` refuses nothing on classical and nothing on temporal. The 2,813 `UNSUPPORTED` verdicts of the previous run are down to 497, all numeric, all encoding-time — and the tasks that unblocked are worth **818 classical and 181 numeric solutions**, against 440 and 273 in the previous run over the same 6,648 tasks.
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