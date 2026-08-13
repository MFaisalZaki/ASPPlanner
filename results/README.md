# Benchmark results

The distilled tables of the sweep reported in [§3 of the top-level README](../README.md#3-benchmark-results):
**6,648 (planner, task) pairs** over 6,648 tasks and 238 domains, run 2026-08-11 → 2026-08-13 through
the in-tree harness now replaced by [pyPMTEvalToolkit](https://github.com/pyPMT/pyPMTEvalToolkit), at 1800 s / 8192 MB per task. This sweep ran
`PLASPPlanner-seq` alone, on classical and numeric; `ABAPlanner-ST` and the temporal track were not
run, so neither appears here.

| | `PLASPPlanner-seq` |
|---|---|
| configuration | `encoding=seq`, `max_horizon=1000`, `time_scale=10` |
| tracks run | classical, numeric |
| tasks attempted | 6,648 |
| solved & validated | **1,047 / 6,648 (15.8%)** |
| of *encodable* tasks | **1,047 / 5,601 (18.7%)** |
| median runtime on solved | 8.2 s |
| median peak memory on solved | 149 MB |
| median plan length | 10 |

Every `SOLVED` row here passed plan validation against the original problem; nothing else is counted
as solved.

Against the previous sweep, on the same 6,648 tasks: **995 → 1,047**, the whole of it on the numeric
track (180 → 228) after the sequential numeric work in `7d21dc2`. 24 domains improved, one regressed
by a single instance.

## Layout

| path | contents |
|---|---|
| [`results.csv`](results.csv) | **per instance** — every (planner, task) pair, both tracks, 6,648 rows |
| [`domains.csv`](domains.csv) | **per domain** — every (planner, domain) pair, both tracks, 238 rows |
| [`summary.json`](summary.json) | the headline counts, machine-readable |
| [`classical/`](classical/) | [README](classical/README.md) · [domains.csv](classical/domains.csv) · [instances.csv](classical/instances.csv) |
| [`numeric/`](numeric/) | [README](numeric/README.md) · [domains.csv](numeric/domains.csv) · [instances.csv](numeric/instances.csv) |

The two top-level tables are the ones to hand to somebody else: `results.csv` is the sweep per
instance, `domains.csv` the same sweep per domain. The per-track directories are those two split by
track, plus the per-domain table rendered as markdown for reading in the browser.

`results.csv` / `instances.csv` carry one row per (planner, instance): status, whether the plan
validated, plan length, makespan, parse/solve/total seconds, peak memory, and a one-line `reason` for
everything that did not solve. `domains.csv` aggregates those to one row per (planner, domain): task
count, coverage, a column per status, and medians over the solved instances.

A planner or track that was not run is absent from these files rather than recorded as 0/n, which
would read as failure. The previous sweep's `ABAPlanner-ST` and temporal tables are in the history of
this directory, at `dee3e45`.

## Reading the status column

| status | meaning |
|---|---|
| `SOLVED` | a plan was found **and validated against the original problem** |
| `UNSUPPORTED` | refused up front — the task's `ProblemKind` is outside `supported_kind()`, or the encoder declined a numeric expression it cannot linearise |
| `ERROR` | an exception; 98.6% of them the **UP PDDL reader**, only 9 the planner |
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
