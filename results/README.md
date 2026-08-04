# Benchmark results

The distilled tables of the sweep reported in [§3 of the top-level README](../README.md#3-benchmark-results):
**13,386 (planner, task) pairs** over 6,738 tasks and 247 domains, run 2026-08-02 → 2026-08-04 through
the [aspbench](../benchmarks/) harness at 1800 s / 8192 MB per task. Both planners were run over the
**whole** classical and numeric benchmark this time; the previous sweep ran `ABAPlanner-ST` on a
3,294-task subset of classical only.

| | `PLASPPlanner-seq` | `ABAPlanner-ST` |
|---|---|---|
| configuration | `encoding=seq`, `max_horizon=1000`, `time_scale=10` | `semantics=ST`, `max_horizon=100`, `time_scale=2` |
| tracks run | classical, numeric, temporal | classical, numeric (temporal not configured) |
| tasks attempted | 6,738 | 6,648 |
| solved & validated | **1,000 / 6,738 (14.8%)** | 334 / 6,648 (5.0%) |
| of *encodable* tasks | **1,000 / 5,700 (17.5%)** | 334 / 2,323 (14.4%) |
| median runtime on solved | **10.2 s** | 19.8 s |
| median peak memory on solved | **148 MB** | 322 MB |
| median plan length | 10 | 10 |

Every `SOLVED` row here passed plan validation against the original problem; nothing else is counted
as solved. On the 6,648 tasks both planners attempted the union is 1,002 — `ABAPlanner-ST` solves
**7** tasks `PLASPPlanner-seq` does not.

## Layout

| path | contents |
|---|---|
| [`results.csv`](results.csv) | **per instance** — every (planner, task) pair, all three tracks, 13,386 rows |
| [`domains.csv`](domains.csv) | **per domain** — every (planner, domain) pair, all three tracks, 485 rows |
| [`summary.json`](summary.json) | the headline counts, machine-readable |
| [`classical/`](classical/) | [README](classical/README.md) · [domains.csv](classical/domains.csv) · [instances.csv](classical/instances.csv) |
| [`numeric/`](numeric/) | [README](numeric/README.md) · [domains.csv](numeric/domains.csv) · [instances.csv](numeric/instances.csv) |
| [`temporal/`](temporal/) | [README](temporal/README.md) · [domains.csv](temporal/domains.csv) · [instances.csv](temporal/instances.csv) |

The two top-level tables are the ones to hand to somebody else: `results.csv` is the sweep per
instance, `domains.csv` the same sweep per domain. The per-track directories are those two split by
track, plus the per-domain table rendered as markdown for reading in the browser.

`results.csv` / `instances.csv` carry one row per (planner, instance): status, whether the plan
validated, plan length, makespan, parse/solve/total seconds, peak memory, and a one-line `reason` for
everything that did not solve. `domains.csv` aggregates those to one row per (planner, domain): task
count, coverage, a column per status, and medians over the solved instances.

`temporal/` holds `PLASPPlanner-seq` only: `ABAPlanner-ST`'s planner configuration carries
`"tracks": ["classical", "numeric"]`, so its temporal tasks were never run. They are absent rather
than recorded as 0/90, which would read as failure.

## Reading the status column

| status | meaning |
|---|---|
| `SOLVED` | a plan was found **and validated against the original problem** |
| `UNSUPPORTED` | refused up front — the task's `ProblemKind` is outside `supported_kind()`, or the encoder declined a numeric expression it cannot linearise |
| `ERROR` | an exception; for `PLASPPlanner-seq` 99.5% of them the **UP PDDL reader**, for `ABAPlanner-ST` the reader plus the Fast Downward grounder it needs |
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
