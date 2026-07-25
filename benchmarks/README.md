# aspbench — benchmark harness for ASPPlanners

Runs the planners of this repository over the classical, numeric and temporal
benchmark sets, one slurm job per (planner, instance) pair, and turns the
results into a coverage table.

The whole thing is one script away:

```bash
cd benchmarks
./setup_benchmark.sh          # asks for the time/memory limit, then does everything
```

That creates a virtualenv, installs ASPPlanners and this harness into it,
clones the benchmark repositories, writes an experiment with the limits you
gave, and generates the slurm job arrays. It ends by printing the `sbatch`
command that starts the sweep.

Everything it asks for also has a flag, so a scripted run is the same script:

```bash
./setup_benchmark.sh --time-limit 30m --memory-limit 8GB --tracks "numeric temporal" \
                     --max-instances 20 --partition compute --yes
```

## Where the benchmarks come from

| track | repository | what is taken |
|---|---|---|
| classical | [AI-Planning/classical-domains](https://github.com/AI-Planning/classical-domains) | every domain with an `api.py` |
| numeric | [pyPMT/numeric-domains](https://github.com/pyPMT/numeric-domains) | every domain with an `api.py` |
| temporal | [potassco/pddl-instances](https://github.com/potassco/pddl-instances) | the `*-time*` / `*-temporal*` IPC domains |

There is no dedicated temporal-domains repository the way there is for the
other two tracks; the IPC archive above is the closest thing, and it carries
every temporal track from IPC-2002 onwards in one uniform layout. Only its
temporal domains are taken (`--suite-tracks pddl-instances=temporal`), so its
classical and numeric domains do not duplicate the two repos above.

A task's track is decided by **reading its domain file**, not by which
repository it came from: `:durative-action` (or a PDDL+ `:process`/`:event`)
makes it temporal, a `(:functions ...)` block makes it numeric, anything else
is classical. That keeps the labels honest whatever you point the tool at —
including your own task directory, which needs no `api.py`:

```bash
aspbench discover --tasks-dir mine=/path/to/my-domains
```

Four repository layouts are recognised: an `api.py` domain directory, an IPC
`instances/` directory (with a shared domain file or one `domains/domain-N.pddl`
per instance), one sub-directory per instance under `instances/`
(`instances/korf1/korf1_{domain,problem}.pddl`), and a plain domain file with
its problems as siblings.

Discovery is defensive about the state these repositories are actually in. An
`api.py` is authoritative *for its own directory only* — a few are copy-pasted
from a sibling domain and resolve to that domain's files, which would file its
instances under the wrong name and hide the ones actually present; those are
ignored and the directory is read directly instead. The domain file does not
have to be called `domain.pddl` (`domain_snp.pddl` and `korf1_domain.pddl` are
found too). And a directory that looks like a domain but yields nothing is
reported on stderr rather than dropped:

```
note: no tasks from .../numeric-domains/foo (api.py lists no existing (domain, problem) pair)
```

so a benchmark set never contributes fewer instances than it holds without
saying so.

## The four stages

```
aspbench init      → an experiment directory (limits + planner configurations)
aspbench discover  → what tasks a benchmark repository holds
aspbench generate  → one run command per (planner, task), plus slurm arrays
aspbench solve     → run ONE pair under its limits, dump a JSON result   (slurm calls this)
aspbench analyze   → results.csv + a coverage report
aspbench report    → paper-ready tables (text + LaTeX) and figures
```

Everything except `solve` is stdlib-only — none of it imports
`unified_planning`, so you can generate a sweep on a laptop and let only the
compute nodes carry the planner. (`report`'s *figures* want matplotlib; its
tables do not — see below.)

## Experiment configuration

An experiment is a directory, the same shape
[pyPMTEvalToolkit](https://github.com/pyPMT/pyPMTEvalToolkit) uses:

```
experiment/
├── exp-details.json        limits + task selection
└── planners/
    ├── plasp-seq.json
    └── aba-st.json
```

```jsonc
// exp-details.json
{
    "cfgs": {
        "timelimit": "00:30:00",        // also accepts "30m" or 1800
        "memorylimit": "8GB",           // also accepts 8192 (MB)
        "slurm-time-headroom": "00:05:00",
        "slurm-memory-headroom": "1GB",
        "slurm": {
            "cpus-per-task": 1,
            "partition": null,
            "account": null,
            "max-parallel-jobs": 50,    // --array=...%50
            "max-array-size": 1000,     // split arrays larger than this
            "extra-directives": []      // verbatim #SBATCH lines
        }
    },
    "tasks": {
        "tracks": ["classical", "numeric", "temporal"],
        "max-instances-per-domain": 10,
        "selection": "even",            // "even" spreads across sizes, "first" takes the smallest
        "include-domains": [],          // glob patterns
        "exclude-domains": [],
        "ipc-years": []
    }
}
```

```jsonc
// planners/plasp-seq.json
{
    "planner-tag": "PLASPPlanner-seq",     // names the result directory and the CSV column
    "up-planner-name": "PLASPPlanner",     // the registered UP engine
    "planner-params": {"encoding": "seq", "max_horizon": 1000, "time_scale": 10},
    "tracks": ["classical", "numeric"]   // optional: restrict this planner to some tracks
}
```

**Every `.json` in `planners/` is benchmarked.** Nothing filters the set — to
leave a planner out, delete its file. The setup script copies in any template
configuration the experiment does not have yet (existing files are never
overwritten, so local edits survive a re-run), and installs the `aba` extra
automatically when a configuration asks for the `ABAPlanner` engine.

`planner-params` goes to `OneshotPlanner(params=...)` verbatim, so anything the
engine accepts works: `encoding`, `horizon`, `max_horizon`, `time_scale` and
`compilationlist` for `PLASPPlanner`; `max_horizon`, `semantics` and `time_scale`
for `ABAPlanner`. A `compilationlist` names its `CompilationKind` as a string
and the runner resolves it, so a configuration can take over the compilation
pipeline — listing the removers without a grounder, for instance, benchmarks
the lifted encoding against the pre-ground one:

```json
"compilationlist": [
    ["up_quantifiers_remover", "QUANTIFIERS_REMOVING"],
    ["up_disjunctive_conditions_remover", "DISJUNCTIVE_CONDITIONS_REMOVING"]
]
```

## Sandbox layout

```
sandbox/
├── tasks.json                  the resolved task list (what "attempted" means)
├── cmds/<planner>.txt          one aspbench-solve command per line
├── slurm/aspbench-<planner>.sbatch    job array, one index per line of that file
├── slurm/submit_all.sh
├── run_local.sh                the same commands through xargs -P, no scheduler
├── results/<planner>/<task>.json
├── errors/                     tracebacks of crashed tasks
└── analysis/                   results.csv, summary.txt, summary.json
```

## What a run records

Each result JSON carries the task and planner identity, the limits it ran
under, `parse`/`solve`/`total` seconds, peak memory, the plan and its
length (makespan for temporal plans), the `ProblemKind` features of the task,
the planner's log messages, and a status:

| status | meaning |
|---|---|
| `SOLVED` | a plan came back and it validated |
| `UNSOLVABLE` | proven unsolvable |
| `EXHAUSTED` | the deepening search hit `max_horizon` |
| `TIMEOUT` / `MEMOUT` | the task's own limit fired |
| `UNSUPPORTED` | the engine does not support this `ProblemKind` |
| `ERROR` | it crashed; the traceback is in `errors/` |
| `KILLED` | the scheduler killed the job before it could report |
| `MISSING` | the pair never produced a result at all |

Some deliberate choices behind those:

* **Limits are enforced twice.** The runner arms its own alarm and address-space
  limit, and slurm gets those limits *plus* headroom. A task that runs out of
  time comes back as a `TIMEOUT` row rather than as a missing file and a line
  in the accounting log. Slurm stays the backstop for what the process cannot
  catch itself — chiefly a solve that dies inside clingo's grounder, where no
  Python handler gets to run.
* **`UNSUPPORTED` is not a failure.** The engine's `ProblemKind` is checked
  before solving, so a linear-numeric domain the encoding cannot express is
  recorded as out of scope instead of counted as a miss (or a crash).
* **Coverage counts validated plans only.** A returned plan that fails the
  validator is a bug, and `analyze` lists those separately rather than adding
  them to a coverage number.
* **`MISSING` is counted.** `tasks.json` says what was attempted, so a coverage
  percentage is never computed over a quietly smaller denominator.
* **Every run gets its own working directory.** The runner `chdir`s into
  `runs/<planner>__<task>.<pid>` before solving and removes it afterwards.
  Fast Downward's translator writes `output.sas` into the *working* directory,
  and it is not the only tool that does; on a cluster that directory would
  otherwise be wherever `sbatch` was called from, shared by every task in the
  array. `--keep-run-dir` leaves it behind when you want to inspect what a run
  produced, and the path is recorded in the result JSON as `run.work-dir`.

## Running the sweep

```bash
bash sandbox/slurm/submit_all.sh              # one job array per planner
squeue -u $USER -n aspbench                   # watch it
bash sandbox/run_local.sh 8                   # or run it locally, 8 at a time
```

One job array per planner rather than one `sbatch` file per task: a full sweep
is tens of thousands of pairs, which is slow to submit one at a time and
unfriendly to schedulers with a submission-rate limit. `--per-task-scripts`
still emits the one-file-per-task form for sites that need it.

Arrays longer than `max-array-size` are split automatically, each chunk
reading its own slice of the command file.

## Collecting the results

```bash
aspbench analyze --sandbox-dir sandbox --per-domain
```

prints (and writes to `analysis/`) coverage per planner and track, a status
breakdown, runtime statistics over solved instances, and — when more than one
planner ran — a head-to-head restricted to the tasks *every* planner attempted,
with the count of instances only that planner solved.

## Paper-ready tables and figures

```bash
pip install -e "benchmarks[plots]"          # matplotlib, for the figures only
aspbench report --sandbox-dir sandbox
```

Writes to `sandbox/report/`:

| file | what |
|---|---|
| `results.txt` | coverage, outcomes, IPC scores and per-domain coverage as text |
| `coverage.tex`, `per-domain-coverage.tex`, `outcomes.tex` | the same tables as booktabs LaTeX, ready to `\input` |
| `report.json` | every number above, for scripting |
| `plots/survival.*` | **survival (cactus) plot** — instances solved within a time budget, log x |
| `plots/survival-per-track.*` | the same, faceted by track |
| `plots/memory-survival.*` | instances solved within a memory budget |
| `plots/coverage-per-track.*` | grouped coverage bars, labelled with percentages |
| `plots/outcomes.*` | stacked outcome bars (solved / timeout / memout / error / …) |
| `plots/runtime-A-vs-B.*` | log-log runtime scatter per planner pair, colored by track |
| `plots/cost-A-vs-B.*` | the same for plan cost (length, or makespan for schedules) |

Both PDF (vector, `pdf.fonttype 42` so venues can edit the text) and PNG, via
`--formats`. `--no-plots` produces the tables alone, with no matplotlib needed.

**IPC scores.** `results.txt` reports the two standard ones, summed over
instances so the maximum is the instance count: *quality* is the satisficing
rule `best_cost / cost` (0 when unsolved), *time* is the agile rule
`1 / (1 + log10(t / t_best))`. Cost is the plan length, or the makespan for a
temporal plan.

**About the figures.** They are meant to go into a paper as they are: colors
come from a colorblind-validated categorical set, every series also carries its
own marker and dash pattern so nothing depends on color alone (and greyscale
printing survives), outcome colors are reserved for outcomes and never reused
as a series color, and there are no dual axes. In the runtime scatters, an
instance one planner failed is pinned to the cutoff line and drawn **hollow**,
so a point on the border reads as "did not finish" rather than "took exactly
the limit". Series are capped — six in the line plots, three in the scatters,
where every pair of colors is compared at once — and anything past the cap is
reported rather than silently dropped.

## Resuming and iterating

`generate --skip-existing` drops the pairs that already have a result, so after
a partial sweep (or after adding a planner configuration) you regenerate and
submit again without redoing finished work:

```bash
aspbench generate --exp-dir experiment --sandbox-dir sandbox \
    --tasks-dir numeric-domains=benchmark-tasks/numeric-domains \
    --venv-dir venv --skip-existing
bash sandbox/slurm/submit_all.sh
```

## Smoke test

The repository's own PDDL fixtures make a sweep that finishes in seconds and
exercises all three tracks:

```bash
aspbench generate --exp-dir experiment --sandbox-dir /tmp/smoke \
    --tasks-dir tests=../tests/pddl --venv-dir venv
bash /tmp/smoke/run_local.sh 4
aspbench analyze --sandbox-dir /tmp/smoke --per-domain
```
