# ASPPlanner

A lightweight planner that solves automated planning problems by compiling them to Answer Set Programming (ASP) and delegating search to [clingo](https://potassco.org/clingo/). ASPPlanner plugs into the [Unified Planning](https://github.com/aiplan4eu/unified-planning) (UP) framework as a `OneshotPlanner` engine, so any UP problem can be solved through a uniform interface.

## Features

- **UP integration**: registers itself as the `ASPPlanner` engine on import, usable through `OneshotPlanner` like any other UP planner.
- **ASP-based search**: encodes the planning task into a logic program and lets clingo handle grounding and solving.
- **Iterative-deepening horizon**: starts at horizon `0` and increases until a plan is found.
- **Built-in validation**: every returned plan is checked with UP's `sequential_plan_validator` before being handed back.
- **Broad problem support**: action-based, numeric, hierarchical typing, conditional effects, disjunctive/universal/existential conditions, increase/decrease effects, and more (see `UPASPPlanner.supported_kind`).

## Installation

ASPPlanner targets Python 3.10+.

```bash
git clone https://github.com/MFaisalZaki/ASPPlanner.git
cd ASPPlanner
pip install -e .
```

Runtime dependencies (installed automatically): `clingo>=5.6.0`, `unified-planning>=1.1.0`, `lark>=1.1.0`.

## Usage

### Through the Unified Planning framework

Importing `aspplanner` registers the engine, so the standard UP entry points work out of the box:

```python
import aspplanner  # registers the ASPPlanner engine
from unified_planning.shortcuts import OneshotPlanner

# `problem` is any unified_planning.model.Problem you constructed or parsed.
with OneshotPlanner(name="ASPPlanner") as planner:
    result = planner.solve(problem)
    print(result.status)
    print(result.plan)
```

### Direct API

You can also drive the planner directly if you don't need the UP result wrapper:

```python
from aspplanner.asp_planner import ASPPlanner

planner = ASPPlanner(problem, encoder_type="seq")
plan = planner.plan()
```

The `seq` encoder uses the bundled sequential horizon-based ASP encoding at [aspplanner/encodings/sequential-horizon.lp](aspplanner/encodings/sequential-horizon.lp).

## Project layout

- [aspplanner/asp_planner.py](aspplanner/asp_planner.py) — core solver loop (compile → ground → solve → extract → validate).
- [aspplanner/up_asp_planner.py](aspplanner/up_asp_planner.py) — UP engine adapter and supported `ProblemKind`.
- [aspplanner/compilers/](aspplanner/compilers/) — UP-to-ASP compilation pipeline (encoder, fact builders, renamer, TIM typing).
- [aspplanner/encodings/](aspplanner/encodings/) — clingo encodings used by each encoder type.
- [aspplanner/grammars/](aspplanner/grammars/) — Lark grammar for parsing ASP plan facts back into UP action instances.
- [aspplanner/utilities.py](aspplanner/utilities.py) — plan parsing and validation helpers.

## License

MIT — see [pyproject.toml](pyproject.toml) for author and metadata.
