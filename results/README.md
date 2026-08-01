# Benchmark results

The distilled tables of the sweep reported in [§3 of the top-level README](../README.md#3-benchmark-results):
10,032 (planner, task) pairs over 6,738 tasks and 247 domains, run 2026-07-29 → 2026-07-31 through the
[aspbench](../benchmarks/) harness at 1800 s / 8192 MB per task.

| | `PLASPPlanner-seq` | `ABAPlanner-ST` |
|---|---|---|
| configuration | `encoding=seq`, `max_horizon=1000`, `time_scale=10` | `semantics=ST`, `max_horizon=100`, `time_scale=2` |
| tracks | classical, numeric, temporal | classical, numeric (temporal excluded) |
| tracks *as run in this sweep* | classical, numeric, temporal | **classical only, and only 76 of its 145 domains** |
| solved & validated | **1,004 / 6,738 (15%)** | 318 / 3,294 (10%) |
| of *encodable* tasks | **1,004 / 5,600 (18%)** | 318 / 1,811 (18%) |
| median runtime on solved | **10.5 s** | 19.3 s |
| median peak memory on solved | **148 MB** | 292 MB |

Every `SOLVED` row here passed plan validation against the original problem; nothing else is counted as solved.

## Layout

| path | contents |
|---|---|
| [`results.csv`](results.csv) | every (planner, task) pair — one row, all three tracks |
| [`summary.json`](summary.json) | the headline counts, machine-readable |
| [`classical/`](classical/) | [README](classical/README.md) · [domains.csv](classical/domains.csv) · [instances.csv](classical/instances.csv) |
| [`numeric/`](numeric/) | [README](numeric/README.md) · [domains.csv](numeric/domains.csv) · [instances.csv](numeric/instances.csv) |
| [`temporal/`](temporal/) | [README](temporal/README.md) · [domains.csv](temporal/domains.csv) · [instances.csv](temporal/instances.csv) |

`instances.csv` is one row per (planner, instance): status, whether the plan validated, plan length,
makespan, parse/solve/total seconds, peak memory, and a one-line `reason` for everything that did not
solve. `domains.csv` aggregates those to one row per (planner, domain): coverage, the full outcome
breakdown, and medians over the solved instances. Each track's `README.md` renders the per-domain
table for reading in the browser.

`ABAPlanner-ST` appears only under `classical/`: its planner configuration carries
`"tracks": ["classical"]`, so its numeric and temporal tasks were never run. They are absent rather
than recorded as 0/n, which would read as failure.

## Reading the status column

| status | meaning |
|---|---|
| `SOLVED` | a plan was found **and validated against the original problem** |
| `UNSUPPORTED` | refused up front — the task's `ProblemKind` is outside `supported_kind()`, or the encoder declined a numeric expression it cannot linearise |
| `ERROR` | an exception; 99% of them the **UP PDDL reader**, not the planner |
| `TIMEOUT` | hit the 1800 s task limit |
| `MEMOUT` | hit the 8 GB task limit |
| `KILLED` | the scheduler reaped the job before the harness could write a result — read these as timeouts |
| `EXHAUSTED` | `max_horizon` reached with no plan; the task may still be solvable deeper |

*Encodable* = attempted, minus `UNSUPPORTED`, minus `ERROR`. Both denominators matter and they say
different things: raw coverage is what the planner scores on IPC suites as they are, coverage of
encodable is how the search does on the tasks that actually reach clingo.

## Regenerating

The sweep's sandbox (per-task JSONs, tracebacks, slurm logs — ~20 GB) is gitignored. These tables are
rebuilt from it with:

```bash
python results/generate_results.py --sandbox-dir sandbox-results/sandbox
```

See [Reproducing](../README.md#reproducing) for how to produce the sandbox in the first place.
