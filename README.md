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

Lower it (`PLASPPlanner(problem, time_scale=1)`) when a domain has large, coprime durations and no required concurrency: **the encoding's remaining-duration recursion is quadratic in the largest scaled duration**, which is the dominant grounding cost on the temporal track — at `time_scale=10` a duration of 30 becomes 300 grid points and the counter's transitive closure is 300² rules. The temporal track was not run in the sweep reported in [§3](#3-benchmark-results).


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

Every number below comes from the checked-in tables in [results/](results/) — [`results.csv`](results/results.csv) per instance and [`domains.csv`](results/domains.csv) per domain, plus the same two split into one directory per track — distilled from the sweep's sandbox by [`results/generate_results.py`](results/generate_results.py).

## 3.1 Setup

One sweep, run 2026-08-11 → 2026-08-13 through the [aspbench](benchmarks/) harness on a slurm cluster (one CPU per task).

| | |
|---|---|
| **Benchmark set** | 6,648 unique tasks over 238 domains |
| **Time limit** | 1800 s wall per task (slurm job capped at 35 min) |
| **Memory limit** | 8192 MB per task (slurm job capped at 9 GB) |
| **`PLASPPlanner-seq`** | `encoding=seq`, `max_horizon=1000`, `time_scale=10`, classical + numeric |

| track | tasks | domains | source |
|---|---|---|---|
| classical | 4,831 | 145 | [AI-Planning/classical-domains](https://github.com/AI-Planning/classical-domains) |
| numeric | 1,817 | 93 | [pyPMT/numeric-domains](https://github.com/pyPMT/numeric-domains) |

A task's track is decided by **reading its domain file**, not by which repository it came from. Declaring a function is not enough to make a domain numeric: IPC's standard STRIPS encoding declares `(total-cost)` purely so plans can be scored, so a fluent has to be *used as state* — compared in a precondition, or assigned somewhere other than a cost accumulator.

**What changed since the previous sweep.** This one runs `PLASPPlanner-seq` alone: `ABAPlanner-ST` was not run, and neither was the temporal track, so the head-to-head comparison and the temporal numbers reported previously are absent rather than restated. The configuration and the benchmark set are unchanged, which makes the movement the encoding's own — the sequential numeric work in `7d21dc2`, comparing by difference under a single `#sum` and reifying values only where an expression effect needs them.

| | previous sweep | this sweep |
|---|---|---|
| classical | 815 / 4,831 | **819 / 4,831** |
| numeric | 180 / 1,817 | **228 / 1,817** |
| total | 995 / 6,648 | **1,047 / 6,648** |
| `UNSUPPORTED` | 397 | 397 |
| numeric median runtime on solved | 65.7 s | **21.7 s** |
| numeric median peak memory on solved | 244 MB | **201 MB** |

The numeric track carries the gain — **+48, a 27% improvement** — and eight of its domains go from solving nothing to solving something: `sugar` (0 → 6), `rover` (0 → 4), `rover-linear` (0 → 4), `depots` (0 → 2), and one instance each in `coins`, `pathwaysmetric`, `petrobras` and `plant-watering`. The largest single gains are `minecraft-pogo-advanced` (9 → 17), `sugar` (0 → 6) and `mprime` (23 → 28). 24 domains improved and one regressed by a single instance — `woodworking-sat08-strips`, at the timeout boundary. Classical moved +4, which is scheduling noise.

The gain is not only in coverage. Numeric `MEMOUT` fell 339 → 240 and `KILLED` 387 → 307 while `TIMEOUT` rose 488 → 611: tasks that used to die on memory now survive to spend their whole budget.

The raw run artifacts (per-task JSONs, tracebacks, slurm logs — ~20 GB) are not checked in; `sandbox*` and `benchmark-run/` are gitignored. The distilled tables are, under [results/](results/) — [`results.csv`](results/results.csv) per instance, [`domains.csv`](results/domains.csv) per domain, and both split per track. See [Reproducing](#reproducing).

### Reading the status column

| status | meaning |
|---|---|
| `SOLVED` | a plan was found **and validated against the original problem** |
| `UNSUPPORTED` | refused up front — the task's `ProblemKind` is outside `supported_kind()`, or the encoder declined a numeric expression it cannot linearise |
| `ERROR` | an exception; 98.6% of them the **UP PDDL reader**, not the planner (see §3.2) |
| `TIMEOUT` | hit the 1800 s task limit |
| `MEMOUT` | hit the 8 GB task limit |
| `KILLED` | the scheduler reaped the job before the harness could write a result — read these as timeouts |
| `EXHAUSTED` | `max_horizon` reached with no plan — the task may still be solvable at a deeper horizon |

## 3.2 Headline coverage

| track | solved | of all tasks | of *encodable* tasks |
|---|---|---|---|
| classical | **819** | 819 / 4,831 (17%) | 819 / 4,212 (**19%**) |
| numeric | **228** | 228 / 1,817 (13%) | 228 / 1,389 (**16%**) |
| **total** | **1,047** | 1,047 / 6,648 (16%) | 1,047 / 5,601 (**19%**) |

*Encodable* = attempted, minus `UNSUPPORTED`, minus the tasks the **UP PDDL reader** could not parse. Both columns matter and they say different things: the left one is raw IPC coverage, the right one is how the search does on the tasks that actually reach clingo.

Full status breakdown:

| planner | SOLVED | UNSUPPORTED | TIMEOUT | ERROR | MEMOUT | KILLED | EXHAUSTED | total |
|---|---|---|---|---|---|---|---|---|
| `PLASPPlanner-seq` | 1,047 | 397 | 3,598 | 650 | 547 | 383 | 26 | 6,648 |

All 1,047 `SOLVED` results passed plan validation — no unvalidated plan was counted.

### The failure modes, separated

**`UNSUPPORTED` is a numeric-encoding question only.** `PLASPPlanner` refuses **nothing** on the classical track. The 397 that remain are all on the numeric track and all *encoding-time* refusals, where the encoder has read the task and declined a specific expression:

| encoder refusal | tasks | domains |
|---|---|---|
| `A product of two numeric fluents is not linear` | 287 | 13 — the whole `nlnp-*` family (145), `zenotravel` (23), `sailing-wind-*` (40), `factory-robot`, `gear-car`, `line-exchange-snp` (20 each) |
| `… reads … with the fractional coefficient 2/5` | 50 | 1 — `fo-farmland` |
| `Making this task's numeric values integral needs a scale factor of 5e13` | 40 | 1 — `worksworld` |
| `… reads … with the fractional coefficient 3/2` | 20 | 1 — `fo-sailing` |

Unchanged from the previous sweep, and that is the point: this is not a moving frontier but a real boundary of a linear-integer ASP encoding — products of two fluents, and coefficients that no rescaling makes whole. The 100 tasks unblocked last sweep (`petrobras`, `hydropower`) stayed encodable, and `petrobras` has now returned its first solution.

**`ERROR` is the front end, not the planner.** 641 of the 650 are UP PDDL reader failures on the benchmark files themselves — 98.6%:

| exception | count | typical cause |
|---|---|---|
| `SyntaxError` | 224 | expressions the reader will not take (`total-time`, `duration`, `preference`) |
| `ParseException` | 199 | malformed / unsupported PDDL syntax |
| `UPProblemDefinitionError` | 129 | `Name p1 already defined` — an object and a predicate sharing a name |
| `KeyError` | 46 | a type name used before it is declared |
| `ParseSyntaxException` | 30 | same as `ParseException` |
| `UPExpressionDefinitionError` | 10 | wrong fluent arity in the instance file |
| `RecursionError` | 3 | `plotting` |

Seven domains account for 555 of the 641: `logistics00` (174), `elevators-00-full` (129), `blocks` (56), `psr-middle` and `psr-large` (50 each), `optical-telegraphs` and `philosophers` (48 each). The encoder never sees these tasks — a ceiling on what *any* UP-based planner scores on these suites. Five of them are a **single unparseable domain file** withholding every instance beneath it (`psr-large`, `psr-middle`, `optical-telegraphs`, `philosophers`, `storage`): 226 instances, 3.4% of the benchmark, behind five fixes.

**Nine errors are the planner's**, and they are new this sweep:

| exception | count | where |
|---|---|---|
| `RuntimeError: … simplifySum … Integer overflow!` | 7 | `numeric/satellite`, raised by clingo inside `ctl.ground` — the difference-`#sum` comparison builds weights past clingo's limit on this domain's constants |
| `RuntimeError: std::bad_alloc` | 2 | `thoughtful-sat14-strips`, `minecraft-pogo-advanced` — grounding a program too large to allocate |

No coverage was lost to them: all seven `satellite` instances were `MEMOUT`/`KILLED` before and none is solved now. But the overflow is a genuine defect rather than a resource limit, and it is the one item in this sweep that the last change introduced.

### Runtime on solved instances

| track | n | median | mean | q90 | max | median peak MB |
|---|---|---|---|---|---|---|
| classical | 819 | 6.3 s | 126.0 s | 418.4 s | 1749.1 s | 144 |
| numeric | 228 | 21.7 s | 422.0 s | 1803.7 s | 2097.5 s | 201 |
| **all** | 1,047 | 8.2 s | 190.4 s | 641.9 s | 2097.5 s | 149 |

The distribution is sharply bimodal: **53% of solved tasks finish in under 10 s and 71% in under 60 s**, and the rest run into the wall (q90 = 642 s, q95 = 1362 s). Iterative deepening either reaches the right horizon quickly or spends the whole budget grinding through horizons that have no plan.

The memouts say the same thing from the other side: a `MEMOUT` reaches a median peak of **7.3 GB** after a median of 198 s (classical) or **7.0 GB** after 153 s (numeric). Where memory is the binding limit it binds early, during grounding — not after a long search.

A caveat for anyone quoting these: 36 solved tasks exceeded the 1800 s limit, the worst at 2,076 s of solve time. The deadline is checked around `ctl.solve` but not around `ctl.ground`, so a long grounding step between two solve calls overshoots it.

### The real ceiling is plan length, not instance size

| plan length | solved |
|---|---|
| 0–5 | 186 (18%) |
| 6–10 | 377 (36%) |
| 11–15 | 223 (21%) |
| 16–20 | 147 (14%) |
| 21–30 | 73 (7%) |
| 31+ | 41 (4%) |

Median 10, 89% at 20 steps or under, maximum **53** — the same ceiling in four consecutive sweeps, and unmoved by a numeric encoding that shifted coverage by 5%. Because the horizon is deepened one step at a time and each horizon is a fresh satisfiability question, cost grows with the *optimal step count*, and almost nothing past ~30 steps is reachable in 30 minutes. It also explains the shape of the per-domain results below: domains with short plans are solved nearly completely, domains with long plans are solved not at all.

## 3.3 Classical

**819 / 4,831 (17%); 819 / 4,212 encodable (19%).** Full tables: [results/classical/](results/classical/).

| status | count | share of encodable |
|---|---|---|
| TIMEOUT | 2,987 | 71% |
| SOLVED | 819 | 19% |
| MEMOUT | 307 | 7% |
| KILLED | 76 | 2% |
| EXHAUSTED | 23 | 1% |
| UNSUPPORTED | **0** | — |

The classical track is **search-bound**: 71% of everything the encoder accepts runs out of time, and only 7% runs out of memory. Grounding is not the problem here — finding the horizon is. Nothing on this track is refused, so `encodable` differs from `attempted` only by the 619 tasks the PDDL reader rejected.

85 of 145 domains yield at least one solved instance:

| domain | solved | rest |
|---|---|---|
| psr-small | 49 / 50 | TIMEOUT 1 |
| schedule | 37 / 150 | TIMEOUT 113 |
| miconic-fulladl | 35 / 150 | TIMEOUT 111, MEMOUT 4 |
| elevators-00-adl | 34 / 151 | TIMEOUT 116, ERROR 1 |
| miconic-simpleadl | 34 / 150 | TIMEOUT 116 |
| mprime | 32 / 35 | TIMEOUT 2, KILLED 1 |
| no-mprime | 31 / 35 | TIMEOUT 3, KILLED 1 |
| elevators-00-strips | 30 / 150 | TIMEOUT 120 |
| miconic | 30 / 150 | TIMEOUT 120 |
| movie | **30 / 30** | — |
| blocks | 20 / 136 | ERROR 56, TIMEOUT 53, KILLED 5, MEMOUT 2 |
| ged-opt14-strips | **20 / 20** | — |
| mystery | 18 / 30 | TIMEOUT 11, EXHAUSTED 1 |
| no-mystery | 18 / 30 | TIMEOUT 11, EXHAUSTED 1 |
| data-network-opt18 | 15 / 20 | TIMEOUT 5 |
| blocks-3op | 14 / 30 | TIMEOUT 16 |
| tsp | 13 / 30 | TIMEOUT 17 |
| ferry | 12 / 30 | TIMEOUT 18 |

The four ADL families in that list — `schedule`, `miconic-fulladl`, `miconic-simpleadl`, `elevators-00-adl` — behave exactly like their STRIPS siblings, a clean prefix solved and the rest timing out, for 140 solved between them: `forall`/`when` costs nothing here because the encoding states conditional effects directly rather than compiling them away. The pattern otherwise is the plan-length ceiling: `psr-small`, `movie` and `mprime` have short plans and are solved essentially completely; `elevators-00-strips` and `miconic` each solve every instance of sizes `s1`–`s6` and nothing from `s7` on. The 23 `EXHAUSTED` results (21 in `elevators-00-full`, one each in `mystery` and `no-mystery`) reached `max_horizon=1000` without a plan — the only classical tasks where the search bound, rather than a resource limit, ended the run.

## 3.4 Numeric

**228 / 1,817 (13%); 228 / 1,389 encodable (16%).** Full tables: [results/numeric/](results/numeric/).

| status | count | share of encodable |
|---|---|---|
| TIMEOUT | 611 | 44% |
| KILLED | 307 | 22% |
| MEMOUT | 240 | 17% |
| SOLVED | 228 | 16% |
| EXHAUSTED | 3 | <1% |

This is the track that moved, and it moved in the right direction on both axes: coverage 13% of encodable against 16% now, median runtime on solved 65.7 s → 21.7 s. Unlike classical it is still **not purely search-bound** — `KILLED` and `MEMOUT` together are 39% of what the encoder accepts, against 52% last sweep. What remains there are the domains that ground slowly or hugely and fail before the search gets a chance: `block-grouping` (167 KILLED of 192 tasks), `15-puzzle` (65 KILLED, 33 MEMOUT), `pancake` (34 KILLED), `petrobras` (69 MEMOUT), `tpp` (29 MEMOUT, 11 KILLED).

43 of 93 domains yield at least one solved instance, up from 35:

| domain | solved | rest |
|---|---|---|
| plotting | 77 / 87 | KILLED 4, EXHAUSTED 3, ERROR 3 |
| mprime | 28 / 30 | TIMEOUT 2 |
| minecraft-sword-advanced | **20 / 20** | — |
| minecraft-pogo-advanced | 17 / 20 | MEMOUT 2, ERROR 1 |
| counters | 10 / 55 | TIMEOUT 45 |
| block-grouping | 9 / 192 | KILLED 167, TIMEOUT 16 |
| sugar | 6 / 20 | TIMEOUT 14 |
| forestfire | 5 / 20 | TIMEOUT 15 |
| pancake | 5 / 50 | KILLED 34, MEMOUT 10, TIMEOUT 1 |
| elevators | 4 / 30 | TIMEOUT 26 |
| rover | 4 / 20 | TIMEOUT 16 |
| rover-linear | 4 / 10 | TIMEOUT 6 |
| the 20 `sec_clear_*-linear` singletons | **20 / 20** | — |

`plotting` alone is 34% of the numeric track's coverage — down from 42%, because the rest of the track grew around it rather than because it lost anything. It is the shape the encoding is best at: many objects, short plans, conditional effects the multi-valued encoding states directly.

## 3.5 Summary, and what to fix next

1. **The numeric encoding got materially better, and the classical one did not move.** Comparing by difference under a single `#sum` is worth +48 solved, +8 domains, and a 3× cut in median solve time on the numeric track, with no change to the refusal set. That is the whole of this sweep's movement; classical's +4 is noise.
2. **Fix the `satellite` integer overflow.** Seven tasks now die inside clingo's grounder because the difference-`#sum` builds weights past its limit. No coverage was lost — they were memouts before — but it is the one regression the last change introduced, and the failure is a crash rather than a refusal, which is the wrong way round for a limit the encoder can detect.
3. **Plan length is still the binding constraint, and it is the strategy's, not the encoding's.** Median solved plan is 10 steps, 89% at 20 or under, maximum 53 — unchanged across four sweeps and unmoved by an encoding change that shifted numeric coverage by a quarter. Any large gain has to come from a better horizon strategy — a lower bound from a relaxed plan, or a planning-graph-style bound — not from a faster encoding.
4. **Numeric grounding is the next resource ceiling.** 547 numeric tasks (39% of encodable) are `KILLED` or `MEMOUT` rather than out of time, concentrated in five domains: `block-grouping`, `15-puzzle`, `pancake`, `petrobras`, `tpp`. These fail before search begins, so they are an encoding-size problem and independent of the horizon strategy above.
5. **What is refused is the honest boundary.** 287 products of two fluents, 70 fractional coefficients, 40 unrepresentable scale factors — unchanged this sweep, and unchangeable without leaving linear-integer ASP.
6. **The benchmark input is dirtier than the planner.** 641 tasks — 10% of the sweep, and 98.6% of all `ERROR`s — never reached the encoder because the UP PDDL reader could not parse the IPC files. Five of those are one unparseable domain file each, withholding 226 instances between them: the cheapest coverage in this table is not in the planner at all.

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