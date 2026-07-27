# ASPPlanners

A lightweight planner that solves automated planning problems by compiling them to Answer Set Programming (ASP) and delegating search to [clingo](https://potassco.org/clingo/). ASPPlanners plugs into the [Unified Planning](https://github.com/aiplan4eu/unified-planning) (UP) framework as a `OneshotPlanner` engine, so any UP problem can be solved through a uniform interface.

## Features

- **UP integration**: registers itself as the `PLASPPlanner` engine on import, usable through `OneshotPlanner` like any other UP planner. Honors the `timeout` argument and reports proper statuses (`SOLVED_SATISFICING`, `UNSOLVABLE_INCOMPLETELY`, `TIMEOUT`).
- **Plans in your vocabulary**: returned plans reference the actions and objects of the problem you passed in — every internal compilation stage (grounding, type inference, renaming) is mapped back before the plan is handed over.
- **Multi-shot ASP search**: iterative-deepening over the horizon using clingo's incremental (iclingo-style) interface — each new horizon grounds only one additional step instead of regrounding the whole program.
- **Numeric planning**: integer-valued fluents with constant-delta `increase`/`decrease`/`assign` effects and linear comparison preconditions (simple numeric planning).
- **No pre-grounding**: gringo grounds the task anyway, and the lifted encoding gives it the same static relations to prune with, so the whole job goes to clingo. On the classical domains here that reaches the same program a reachability grounder does, byte for byte.
- **Temporal planning**: PDDL 2.1 durative actions, encoded as the *happenings* of [SMTPlan](https://github.com/KCL-Planning/SMTPlan) — each durative action splits into its two snap actions, the timesteps carry the happening times, and the remaining-duration and over-all constraints tie the halves back together. Required concurrency works (match-cellar solves), and the result is a UP `TimeTriggeredPlan` checked with `up_time_triggered_validator`.
- **No condition compilation**: negative conditions, `forall`, `or` and `exists` are all encoded directly rather than compiled away, so nothing rewrites the task before it reaches clingo — an action with 5 disjunctive preconditions stays one action instead of becoming 1024. Supply `compilationlist` if you want a pipeline of your own anyway.
- **Built-in validation**: every returned plan is checked against the *original* problem with UP's `sequential_plan_validator` before being handed back.
- **Two backends**: the default `PLASPPlanner` (PLASP-style ASP encoding, solved with clingo) and an optional `ABAPlanner` (STRIPS-to-ABA reduction, solved with [aspforaba](https://bitbucket.org/coreo-group/aspforaba)). Both share the UP front-end (compilation pipeline, map-back, validation) and register as UP engines on import.

## Installation

ASPPlanners targets Python 3.10+.

```bash
git clone https://github.com/MFaisalZaki/ASPPlanners.git
cd ASPPlanners
pip install -e .
```

Runtime dependencies (installed automatically): `clingo>=5.6.0`, `unified-planning>=1.1.0`, `up_fast_downward>=0.5.2`.

The `ABAPlanner` backend additionally needs `aspforaba`; install it with the optional `aba` extra (`pip install -e ".[aba]"`). It is imported lazily, so `import aspplanners` and the default `PLASPPlanner` engine work without it.

## Usage

### Through the Unified Planning framework

Importing `aspplanners` registers the engine, so the standard UP entry points work out of the box:

```python
import aspplanners  # registers the PLASPPlanner engine
from unified_planning.shortcuts import OneshotPlanner

# `problem` is any unified_planning.model.Problem you constructed or parsed.
with OneshotPlanner(name="PLASPPlanner") as planner:
    result = planner.solve(problem, timeout=60)
    print(result.status)
    print(result.plan)
```

Engine options: `params={"max_horizon": 50}` bounds the deepening search (default 1000), `params={"horizon": 10}` solves at one fixed horizon instead, and `params={"time_scale": 2}` sets the resolution of the temporal encoding (see [Temporal planning](#temporal-planning)).

The optional `ABAPlanner` engine is selected the same way — `OneshotPlanner(name="ABAPlanner")`. Its options are `max_horizon` (default 1000), `semantics` (default `"ST"`, aspforaba's extension semantics) and `time_scale`. Unlike `PLASPPlanner`, it does not honor `timeout`, so bound the search with `max_horizon`. It needs the `aba` extra.

### Direct API

You can also drive the planner directly if you don't need the UP result wrapper:

```python
from aspplanners.plasp.planner import PLASPPlanner

planner = PLASPPlanner(problem, encoder_type="seq")
plan = planner.plan(max_horizon=100, timeout=60)   # or plan(horizon=10)
print(planner.status)   # PlanGenerationResultStatus of the last call
print(planner.logs)     # human-readable notes
```

`plan()` returns a `SequentialPlan` — or a `TimeTriggeredPlan` when the task has durative actions. It is empty when no plan was found (check `planner.status`) — or when the goal already holds in the initial state, in which case `status` is `SOLVED_SATISFICING`.

#### Temporal planning

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

Temporal tasks are solved on the *lifted* encoding — no grounder does reachability analysis on them, so pre-grounding would only duplicate gringo's work. A duration stated as a static function has no value until its parameters are bound, so the encoder defers that lookup into the ASP: `durationValue(...)` reads it off `initialState` per binding. The two snaps of a durative action ground to the same bindings the action itself does, because the end snap is declared under its start (it usually has no condition of its own, so nothing else would narrow it).

What is covered: at-start / at-end / over-all conditions, at-start and at-end effects, fixed durations, duration inequalities (`(and (>= ?duration 2) (<= ?duration 5))`), and durations read off a static function (`(= ?duration (travel-time ?a ?b))`). What is not: PDDL+ processes and events, timed initial literals and timed goals, conditions or effects at an intermediate time, a durative action overlapping *itself*, and snap actions that have to be genuinely simultaneous (a happening carries one snap action, so they are sequentialised ε apart). The ABA backend additionally rejects numeric over-all conditions; use the PLASP backend for those.

Clingo terms are integers, so happenings live on an integer time grid. `time_scale` (default 10) says how many times finer that grid is than the greatest common divisor of the task's durations, which makes ε — the minimum separation between two happenings — 1/10 of that gcd, matching SMTPlan's ε. Durations are normalised by their gcd first, so 100 and 150 become 20 and 30 rather than 1000 and 1500, and the discretisation is exact for rational durations rather than an approximation. Lower it (`PLASPPlanner(problem, time_scale=1)`) when a domain has large, coprime durations and no required concurrency: the encoding's remaining-duration recursion is quadratic in the largest scaled duration.

#### Customizing the compilation pipeline

Before a problem reaches the ASP encoder it is put through a list of UP compilers — by default, none of them.

**Nothing is compiled away.** Every condition shape is stated in the encoding itself, so all that is left of the pipeline is the grounding decision below:

| shape | how it is encoded |
|---|---|
| negative conditions | the encoding is multi-valued, so `value(V, false)` is a value like any other. A mirror fluent per negatively-read one would be pure overhead; the encoder just emits the false initial value for the fluents actually read as false |
| `forall` | a conjunction over the universe — and `precondition`/`goal` facts are already conjunctive. One rule with the variable left free and `has(_, type(...))` in the body; gringo does the expanding |
| `or`, `exists` | disjunctions, which conjunctive facts cannot state, so they get their own `orGroup`/`orDisjunct` vocabulary: at least one disjunct has to hold, and a disjunct holds when all of its literals do. An `exists` is the same shape with its disjuncts indexed by the quantified variable's binding |
| numeric comparisons | `<`, `<=`, `=` and their negations against `numval`, wherever a condition can appear: `numPrecondition`, `numGoal`, `numOverall`, and `orDisjunctNum` inside a disjunct. A negation is the comparison's complement (`not (x = y)` is `neq`), not a `not` over a `holds` chain the numeric side does not have |

Staying lifted is the point. An action with *k* disjunctions of 4 literals each — the remover writes out 4<sup>k</sup> copies of it, the encoding writes one group:

| `or` groups | ground actions, native | with the remover | compile, native | with the remover |
|---|---|---|---|---|
| 2 | 1 | 16 | 0.01s | 0.01s |
| 4 | 1 | 256 | 0.02s | 0.20s |
| 5 | **1** | **1024** | **0.04s** | **7.07s** |

The same holds for `exists` (over 40 objects: 41 ground actions instead of 80) and for `forall` (an action with `(forall (?x) (marked ?x))` and `(forall (?x ?y) (near ?x ?y))` stays at 2 precondition facts whether there are 3 objects or 30, against 3 and 30 for the remover).

What the encoding does *not* take is a disjunction nested inside a disjunct (`or(and(a, or(b, c)), d)`), which would need distribution into DNF; that raises and asks for `up_disjunctive_conditions_remover` back. Everything else — including the De Morgan cases, where `not (a or b)` is a conjunction and `not (forall x φ)` is an `exists` — is handled directly.

And it is not pre-ground. gringo grounds the task either way, and the lifted encoding gives it as much to prune with — action signature rules bind parameters via `has(_, type(...))` and fold static preconditions into the rule body. Scaled up, that reaches the same program the FD reachability grounder does, byte for byte:

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

For a domain shaped like that, ask for it back — that is what `aspplanners.common.compilation.select_grounder` is for:

```python
from unified_planning.shortcuts import CompilationKind
from aspplanners.common.compilation import REACHABILITY_GROUNDERS, select_grounder

grounder = select_grounder(problem.kind, REACHABILITY_GROUNDERS)   # None if none applies
planner = PLASPPlanner(problem, compilationlist=[[grounder, CompilationKind.GROUNDING]])
```

Pass `compilationlist` to take over that choice. Each entry is a `[engine_name, CompilationKind]` pair applied in order, and the list is used verbatim — the automatic grounder selection is bypassed, so include or omit the grounder yourself:

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

To inspect or reuse the generated logic program (e.g. with your own clingo `Control`):

```python
text = planner.lp_program()          # task facts + multi-shot encoding, verbatim
planner.dump_lp_program("task.lp")   # same, written to a file

terms = planner.encoding_terms()     # encoding parsed into ASPTerm statements
```

`encoding_terms()` returns typed statements (`ASPFact`, `ASPRule`, `ASPConstraint`, `ASPWeakConstraint`, `ASPDirective`) wrapping clingo AST nodes — filter or rewrite them programmatically, then write them back out:

```python
from aspplanners.lp_io import parse_lp_file, dump_lp, ASPRule

terms = parse_lp_file("my-encoding.lp")
rules = [t for t in terms if isinstance(t, ASPRule)]
dump_lp(terms, "normalized.lp")   # also accepts fact-builder terms and plain strings
```

The encoding is split into `#program base / step(t) / check(t)` parts; ground `base` + `step(1..h)` + `check(h)` and set the external `query(h)` to true to solve at horizon `h`.

#### Encoding layers

The encoding is one `.lp` file per feature, under [encodings/seq/](aspplanners/plasp/encodings/seq/):

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

You can name the layers instead, which is useful for benchmarking a layer's grounding cost or for pinning an encoding across a run:

```python
planner = PLASPPlanner(problem, encoder_type="seq+numeric+temporal")
```

An explicit set is still closed under each layer's requirements and still checked for coverage: naming too few layers raises at construction rather than producing a program that quietly ignores half the task. The same string works as the UP engine's `encoding` parameter (`params={"encoding": "seq+numeric"}`).

Adding a feature — PDDL 3 trajectory constraints, say — means a new `.lp` in `encodings/seq/` plus one `Layer(...)` entry in [layers.py](aspplanners/plasp/layers.py) naming the fact predicates only it reads.

#### The ABA backend directly

`ABAPlan` mirrors `PLASPPlanner`'s driver interface for the STRIPS-to-ABA backend (requires the `aba` extra):

```python
from aspplanners.abaplan.planner import ABAPlan

planner = ABAPlan(problem)
plan = planner.plan(max_horizon=100, semantics="ST")   # "ST" = stable extension semantics
print(planner.status)   # PlanGenerationResultStatus of the last call
print(planner.logs)
```

It runs the same shared front-end (compilation pipeline, map-back, validation) but always grounds the problem — the reduction is over ground STRIPS, so unlike `PLASPPlanner` there is no lifted path to hand the task to, and it takes whichever grounder supports the kind rather than only a reachability one — and builds the ABA framework itself, so it takes no `compilationlist` and no `timeout`; bound the deepening search with `max_horizon`. Temporal tasks work here too — `ABAPlan(problem).plan()` solves match-cellar: `run` and the remaining duration become atoms of the framework, "the interval is still open" and "the duration has elapsed" become assumptions attacked by their contraries, and each step's gap is picked by a set of mutually contrary assumptions the same way its action is. Everything is propositional, so the framework grows with the square of the largest scaled duration and the ABA backend is markedly slower than the PLASP one on temporal tasks; lower `time_scale` (or use `PLASPPlanner`) if that bites.

## Project layout

- [aspplanners/plasp/](aspplanners/plasp/) — the default PLASP backend: `planner.py` (`PLASPPlanner` — core solver loop: compile → incremental ground/solve → extract → map back → validate), `encoder.py` (UP → ASP facts), `facts.py` (fact builders), and `encodings/` (clingo encodings per encoder type).
- [aspplanners/abaplan/](aspplanners/abaplan/) — the optional ABA backend: `encoder.py` (`ABAEncoder` — STRIPS-to-ABA framework construction) and `planner.py` (`ABAPlan` — deepening search over aspforaba).
- [aspplanners/common/](aspplanners/common/) — backend-agnostic front-end shared by both backends: compilation pipeline, plan validation, TIM typing, and `temporal.py` (durative actions → snap actions, and the integer time grid).
- [aspplanners/lp_io.py](aspplanners/lp_io.py) — generic ASP program I/O (`parse_lp`/`dump_lp` and the `ASPStatement` term family).
- [aspplanners/up_engines.py](aspplanners/up_engines.py) — both UP engine adapters (`UPPLASPPlanner` and `UPABAPlanner`, registered as the `PLASPPlanner` and `ABAPlanner` engines) and their supported `ProblemKind`s.
- [tests/](tests/) — end-to-end tests (`pip install -e ".[dev]" && pytest`).
- [benchmarks/](benchmarks/) — the `aspbench` benchmark harness: `setup_benchmark.sh` builds a venv, fetches the classical/numeric/temporal benchmark sets and generates one slurm job per (planner, instance) pair; `aspbench analyze` turns the results into a coverage table. See [benchmarks/README.md](benchmarks/README.md).

## Benchmarking

```bash
cd benchmarks
./setup_benchmark.sh          # asks for the per-task time and memory limits, then does the rest
```
<!-- ./setup_benchmark.sh --partition compute --qos long --time-limit 30m --memory-limit 8GB --yes -->

It creates a virtualenv, installs ASPPlanners and the harness into it, clones [classical-domains](https://github.com/AI-Planning/classical-domains), [numeric-domains](https://github.com/pyPMT/numeric-domains) and the temporal IPC tracks from [pddl-instances](https://github.com/potassco/pddl-instances), writes the experiment configuration and generates the slurm job arrays; then `bash sandbox/slurm/submit_all.sh` runs the sweep and `aspbench analyze --sandbox-dir sandbox` reports coverage per planner and track. Every prompt has a matching flag, so `./setup_benchmark.sh --time-limit 30m --memory-limit 8GB --tracks temporal --yes` is the same run unattended.

## License

MIT — see [pyproject.toml](pyproject.toml) for author and metadata.
