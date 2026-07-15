# ASPPlanner

A lightweight planner that solves automated planning problems by compiling them to Answer Set Programming (ASP) and delegating search to [clingo](https://potassco.org/clingo/). ASPPlanner plugs into the [Unified Planning](https://github.com/aiplan4eu/unified-planning) (UP) framework as a `OneshotPlanner` engine, so any UP problem can be solved through a uniform interface.

## Features

- **UP integration**: registers itself as the `ASPPlanner` engine on import, usable through `OneshotPlanner` like any other UP planner. Honors the `timeout` argument and reports proper statuses (`SOLVED_SATISFICING`, `UNSOLVABLE_INCOMPLETELY`, `TIMEOUT`).
- **Plans in your vocabulary**: returned plans reference the actions and objects of the problem you passed in — every internal compilation stage (grounding, type inference, renaming) is mapped back before the plan is handed over.
- **Multi-shot ASP search**: iterative-deepening over the horizon using clingo's incremental (iclingo-style) interface — each new horizon grounds only one additional step instead of regrounding the whole program.
- **Numeric planning**: integer-valued fluents with constant-delta `increase`/`decrease`/`assign` effects and linear comparison preconditions (simple numeric planning). Numeric tasks solve on the lifted encoding; classical tasks are pre-grounded with Fast Downward's reachability grounder.
- **Built-in validation**: every returned plan is checked against the *original* problem with UP's `sequential_plan_validator` before being handed back.

## Installation

ASPPlanner targets Python 3.10+.

```bash
git clone https://github.com/MFaisalZaki/ASPPlanner.git
cd ASPPlanner
pip install -e .
```

Runtime dependencies (installed automatically): `clingo>=5.6.0`, `unified-planning>=1.1.0`, `up_fast_downward>=0.5.2`.

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
from aspplanner.asp_planner import ASPPlanner

planner = ASPPlanner(problem, encoder_type="seq")
plan = planner.plan(max_horizon=100, timeout=60)   # or plan(horizon=10)
print(planner.status)   # PlanGenerationResultStatus of the last call
print(planner.logs)     # human-readable notes
```

`plan()` always returns a `SequentialPlan`; it is empty when no plan was found (check `planner.status`) — or when the goal already holds in the initial state, in which case `status` is `SOLVED_SATISFICING`.

To inspect or reuse the generated logic program (e.g. with your own clingo `Control`):

```python
text = planner.lp_program()          # task facts + multi-shot encoding, verbatim
planner.dump_lp_program("task.lp")   # same, written to a file

terms = planner.encoding_terms()     # encoding parsed into ASPTerm statements
```

`encoding_terms()` returns typed statements (`ASPFact`, `ASPRule`, `ASPConstraint`, `ASPWeakConstraint`, `ASPDirective`) wrapping clingo AST nodes — filter or rewrite them programmatically, then write them back out:

```python
from aspplanner.compilers.asp_facts import parse_lp_file, dump_lp

terms = parse_lp_file("my-encoding.lp")
rules = [t for t in terms if isinstance(t, ASPRule)]
dump_lp(terms, "normalized.lp")   # also accepts fact-builder terms and plain strings
```

The encoding is split into `#program base / step(t) / check(t)` parts; ground `base` + `step(1..h)` + `check(h)` and set the external `query(h)` to true to solve at horizon `h`.

## Project layout

- [aspplanner/asp_planner.py](aspplanner/asp_planner.py) — core solver loop (compile → incremental ground/solve → extract → map back → validate).
- [aspplanner/up_asp_planner.py](aspplanner/up_asp_planner.py) — UP engine adapter and supported `ProblemKind`.
- [aspplanner/compilers/](aspplanner/compilers/) — UP-to-ASP compilation pipeline (encoder, fact builders, TIM typing).
- [aspplanner/encodings/](aspplanner/encodings/) — clingo encodings used by each encoder type.
- [tests/](tests/) — end-to-end tests (`pip install -e ".[dev]" && pytest`).

## License

MIT — see [pyproject.toml](pyproject.toml) for author and metadata.
