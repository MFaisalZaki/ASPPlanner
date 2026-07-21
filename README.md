# ASPPlanner

A lightweight planner that solves automated planning problems by compiling them to Answer Set Programming (ASP) and delegating search to [clingo](https://potassco.org/clingo/). ASPPlanner plugs into the [Unified Planning](https://github.com/aiplan4eu/unified-planning) (UP) framework as a `OneshotPlanner` engine, so any UP problem can be solved through a uniform interface.

## Features

- **UP integration**: registers itself as the `ASPPlanner` engine on import, usable through `OneshotPlanner` like any other UP planner. Honors the `timeout` argument and reports proper statuses (`SOLVED_SATISFICING`, `UNSOLVABLE_INCOMPLETELY`, `TIMEOUT`).
- **Plans in your vocabulary**: returned plans reference the actions and objects of the problem you passed in — every internal compilation stage (grounding, type inference, renaming) is mapped back before the plan is handed over.
- **Multi-shot ASP search**: iterative-deepening over the horizon using clingo's incremental (iclingo-style) interface — each new horizon grounds only one additional step instead of regrounding the whole program.
- **Numeric planning**: integer-valued fluents with constant-delta `increase`/`decrease`/`assign` effects and linear comparison preconditions (simple numeric planning). Numeric tasks solve on the lifted encoding; classical tasks are pre-grounded with Fast Downward's reachability grounder.
- **Configurable compilation pipeline**: the UP compilers that run before ASP encoding are selected automatically per problem, or supplied explicitly via the `compilationlist` argument when you want full control over the preprocessing.
- **Built-in validation**: every returned plan is checked against the *original* problem with UP's `sequential_plan_validator` before being handed back.
- **Two backends**: the default `ASPPlanner` (PLASP-style ASP encoding, solved with clingo) and an optional `ABAPlanner` (STRIPS-to-ABA reduction, solved with [aspforaba](https://bitbucket.org/coreo-group/aspforaba)). Both share the UP front-end (compilation pipeline, map-back, validation) and register as UP engines on import.

## Installation

ASPPlanner targets Python 3.10+.

```bash
git clone https://github.com/MFaisalZaki/ASPPlanner.git
cd ASPPlanner
pip install -e .
```

Runtime dependencies (installed automatically): `clingo>=5.6.0`, `unified-planning>=1.1.0`, `up_fast_downward>=0.5.2`.

The `ABAPlanner` backend additionally needs `aspforaba`; install it with the optional `aba` extra (`pip install -e ".[aba]"`). It is imported lazily, so `import aspplanner` and the default `ASPPlanner` engine work without it.

## Usage

### Through the Unified Planning framework

Importing `aspplanner` registers the engine, so the standard UP entry points work out of the box:

```python
import aspplanner  # registers the ASPPlanner engine
from unified_planning.shortcuts import OneshotPlanner

# `problem` is any unified_planning.model.Problem you constructed or parsed.
with OneshotPlanner(name="ASPPlanner") as planner:
    result = planner.solve(problem, timeout=60)
    print(result.status)
    print(result.plan)
```

Engine options: `params={"max_horizon": 50}` bounds the deepening search, and `params={"horizon": 10}` solves at one fixed horizon instead.

### Direct API

You can also drive the planner directly if you don't need the UP result wrapper:

```python
from aspplanner.plasp.planner import ASPPlanner

planner = ASPPlanner(problem, encoder_type="seq")
plan = planner.plan(max_horizon=100, timeout=60)   # or plan(horizon=10)
print(planner.status)   # PlanGenerationResultStatus of the last call
print(planner.logs)     # human-readable notes
```

`plan()` always returns a `SequentialPlan`; it is empty when no plan was found (check `planner.status`) — or when the goal already holds in the initial state, in which case `status` is `SOLVED_SATISFICING`.

#### Customizing the compilation pipeline

Before a problem reaches the ASP encoder it is put through a list of UP compilers. By default `ASPPlanner` derives this list from the problem: the quantifier, negative-condition and disjunctive-condition removers always run, and classical (non-numeric) tasks additionally get Fast Downward's reachability grounder — numeric tasks skip grounding and solve on the lifted encoding.

Pass `compilationlist` to take over that choice. Each entry is a `[engine_name, CompilationKind]` pair applied in order, and the list is used verbatim — the automatic numeric-vs-classical grounder selection is bypassed, so include or omit the grounder yourself:

```python
from unified_planning.shortcuts import CompilationKind
from aspplanner.plasp.planner import ASPPlanner

# Solve on the lifted encoding: run the removers but skip the grounder.
pipeline = [
    ["up_quantifiers_remover",            CompilationKind.QUANTIFIERS_REMOVING],
    ["up_negative_conditions_remover",    CompilationKind.NEGATIVE_CONDITIONS_REMOVING],
    ["up_disjunctive_conditions_remover", CompilationKind.DISJUNCTIVE_CONDITIONS_REMOVING],
]
planner = ASPPlanner(problem, encoder_type="seq", compilationlist=pipeline)
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
from aspplanner.lp_io import parse_lp_file, dump_lp

terms = parse_lp_file("my-encoding.lp")
rules = [t for t in terms if isinstance(t, ASPRule)]
dump_lp(terms, "normalized.lp")   # also accepts fact-builder terms and plain strings
```

The encoding is split into `#program base / step(t) / check(t)` parts; ground `base` + `step(1..h)` + `check(h)` and set the external `query(h)` to true to solve at horizon `h`.

## Project layout

- [aspplanner/plasp/](aspplanner/plasp/) — the default PLASP backend: `planner.py` (core solver loop: compile → incremental ground/solve → extract → map back → validate), `encoder.py` (UP → ASP facts), `facts.py` (fact builders), and `encodings/` (clingo encodings per encoder type).
- [aspplanner/abaplan/](aspplanner/abaplan/) — the optional ABA backend: `planner.py` (STRIPS-to-ABA reduction + aspforaba search) and `up_engine.py` (its UP adapter, registered as `ABAPlanner`).
- [aspplanner/common/](aspplanner/common/) — backend-agnostic front-end shared by both backends: compilation pipeline, plan validation, TIM typing.
- [aspplanner/lp_io.py](aspplanner/lp_io.py) — generic ASP program I/O (`parse_lp`/`dump_lp` and the `ASPStatement` term family).
- [aspplanner/up_asp_planner.py](aspplanner/up_asp_planner.py) — UP engine adapter and supported `ProblemKind`.
- [tests/](tests/) — end-to-end tests (`pip install -e ".[dev]" && pytest`).

## License

MIT — see [pyproject.toml](pyproject.toml) for author and metadata.
