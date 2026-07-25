#!/usr/bin/env bash
#
# One-shot setup for an ASPPlanners benchmark sweep:
#
#   1. ask for the per-task time and memory limits (and a few other knobs),
#   2. create a virtualenv and install ASPPlanners + this harness into it,
#   3. fetch the benchmark repositories for the tracks you picked,
#   4. write the experiment configuration with those limits,
#   5. generate one run command per (planner, task) and the slurm job arrays.
#
# Everything is prompted with a default, and every prompt has a matching flag,
# so the same script drives an interactive setup and a scripted one (--yes).
# Re-running it is safe: an existing venv, clone or experiment is reused.
#
# Usage:
#   ./setup_benchmark.sh                      # interactive
#   ./setup_benchmark.sh --yes                # all defaults, no prompts
#   ./setup_benchmark.sh --time-limit 30m --memory-limit 8GB --tracks temporal --yes
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------- defaults --
WORK_DIR="${REPO_DIR}/benchmark-run"
VENV_DIR=""
TASKS_DIR=""
SANDBOX_DIR=""
EXP_DIR=""
TIME_LIMIT="00:30:00"
MEMORY_LIMIT="8GB"
MAX_INSTANCES="0"
TRACKS="classical numeric temporal"
PARTITION=""
ACCOUNT=""
QOS=""
MAX_PARALLEL="50"
WITH_ABA="no"
SKIP_FETCH="no"
SKIP_INSTALL="no"
PER_TASK_SCRIPTS="no"
ASSUME_YES="no"
PYTHON_BIN="${PYTHON_BIN:-python3}"

CLASSICAL_REPO="https://github.com/AI-Planning/classical-domains.git"
NUMERIC_REPO="https://github.com/pyPMT/numeric-domains.git"
# The IPC archive is where the temporal benchmarks live: every `*-time*` and
# `*-temporal*` domain from IPC-2002 onwards, in one uniform layout. There is no
# equivalent of classical-domains/numeric-domains for the temporal track.
TEMPORAL_REPO="https://github.com/potassco/pddl-instances.git"

usage() {
    cat <<'EOF'
One-shot setup for an ASPPlanners benchmark sweep:

  1. ask for the per-task time and memory limits (and a few other knobs),
  2. create a virtualenv and install ASPPlanners + this harness into it,
  3. fetch the benchmark repositories for the tracks you picked,
  4. write the experiment configuration with those limits,
  5. generate one run command per (planner, task) and the slurm job arrays.

Re-running it is safe: an existing venv, clone or experiment is reused.

Usage:
  ./setup_benchmark.sh                      # interactive
  ./setup_benchmark.sh --yes                # all defaults, no prompts
  ./setup_benchmark.sh --time-limit 30m --memory-limit 8GB --tracks temporal --yes

Options:
  --work-dir DIR          root for venv/tasks/sandbox   (default: <repo>/benchmark-run)
  --venv-dir DIR          virtualenv location           (default: <work-dir>/venv)
  --tasks-dir DIR         where benchmarks are cloned   (default: <work-dir>/benchmark-tasks)
  --sandbox-dir DIR       commands, results, logs       (default: <work-dir>/sandbox)
  --exp-dir DIR           experiment configuration      (default: <work-dir>/experiment)
  --time-limit VALUE      per task, e.g. 00:30:00 or 30m
  --memory-limit VALUE    per task, e.g. 8GB
  --max-instances N       instances per domain, 0 for all
  --tracks "a b c"        subset of: classical numeric temporal
  --partition NAME        slurm partition
  --account NAME          slurm account
  --qos NAME              slurm QOS
  --max-parallel N        cap on concurrently running array jobs
  --with-aba              install the aspforaba extra even if no planner needs it
                          (it is installed automatically when one does)
  --per-task-scripts      also emit one .sbatch per task (default: job arrays)
  --skip-fetch            do not clone/update the benchmark repositories
  --skip-install          do not create the venv or install anything
  -y, --yes               accept every default, never prompt
  -h, --help              this message
EOF
}

# ------------------------------------------------------------------- args ---
while [ $# -gt 0 ]; do
    case "$1" in
        --work-dir)         WORK_DIR="$2"; shift 2 ;;
        --venv-dir)         VENV_DIR="$2"; shift 2 ;;
        --tasks-dir)        TASKS_DIR="$2"; shift 2 ;;
        --sandbox-dir)      SANDBOX_DIR="$2"; shift 2 ;;
        --exp-dir)          EXP_DIR="$2"; shift 2 ;;
        --time-limit)       TIME_LIMIT="$2"; shift 2 ;;
        --memory-limit)     MEMORY_LIMIT="$2"; shift 2 ;;
        --max-instances)    MAX_INSTANCES="$2"; shift 2 ;;
        --tracks)           TRACKS="$2"; shift 2 ;;
        --partition)        PARTITION="$2"; shift 2 ;;
        --account)          ACCOUNT="$2"; shift 2 ;;
        --qos)              QOS="$2"; shift 2 ;;
        --max-parallel)     MAX_PARALLEL="$2"; shift 2 ;;
        --with-aba)         WITH_ABA="yes"; shift ;;
        --per-task-scripts) PER_TASK_SCRIPTS="yes"; shift ;;
        --skip-fetch)       SKIP_FETCH="yes"; shift ;;
        --skip-install)     SKIP_INSTALL="yes"; shift ;;
        -y|--yes)           ASSUME_YES="yes"; shift ;;
        -h|--help)          usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

VENV_DIR="${VENV_DIR:-${WORK_DIR}/venv}"
TASKS_DIR="${TASKS_DIR:-${WORK_DIR}/benchmark-tasks}"
SANDBOX_DIR="${SANDBOX_DIR:-${WORK_DIR}/sandbox}"
EXP_DIR="${EXP_DIR:-${WORK_DIR}/experiment}"

# ---------------------------------------------------------------- helpers ---
say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33m warning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m error:\033[0m %s\n' "$*" >&2; exit 1; }

# ask <prompt> <default> <variable-name>
ask() {
    local prompt="$1" default="$2" varname="$3" answer=""
    if [ "$ASSUME_YES" = "yes" ] || [ ! -t 0 ]; then
        answer="$default"
    else
        read -r -p "$(printf '%s [%s]: ' "$prompt" "$default")" answer || answer=""
        answer="${answer:-$default}"
    fi
    printf -v "$varname" '%s' "$answer"
}

ask_yes_no() {
    local prompt="$1" default="$2" varname="$3" answer=""
    ask "$prompt (yes/no)" "$default" answer
    # tr rather than ${answer,,}: macOS still ships bash 3.2.
    case "$(printf '%s' "$answer" | tr '[:upper:]' '[:lower:]')" in
        y|yes|true|1) printf -v "$varname" '%s' 'yes' ;;
        *)            printf -v "$varname" '%s' 'no' ;;
    esac
}

has_track() {
    case " $TRACKS " in *" $1 "*) return 0 ;; *) return 1 ;; esac
}

clone_or_update() {
    local url="$1" dest="$2"
    if [ -d "${dest}/.git" ]; then
        say "updating $(basename "$dest")"
        git -C "$dest" fetch --depth 1 origin HEAD --quiet || warn "could not update ${dest}"
        git -C "$dest" reset --hard FETCH_HEAD --quiet || warn "could not fast-forward ${dest}"
    elif [ -d "$dest" ]; then
        say "reusing $(basename "$dest") (not a git checkout)"
    else
        say "cloning $(basename "$dest")"
        git clone --depth 1 --quiet "$url" "$dest"
    fi
}

# --------------------------------------------------------------- prompting --
if [ "$ASSUME_YES" != "yes" ] && [ -t 0 ]; then
    echo
    echo "ASPPlanners benchmark setup -- press enter to accept a default."
    echo
fi

ask "Per-task time limit (HH:MM:SS or 30m)"      "$TIME_LIMIT"    TIME_LIMIT
ask "Per-task memory limit (e.g. 8GB)"           "$MEMORY_LIMIT"  MEMORY_LIMIT
ask "Tracks to benchmark"                        "$TRACKS"        TRACKS
ask "Instances per domain (0 = all)"             "$MAX_INSTANCES" MAX_INSTANCES
ask "Slurm partition (blank = site default)"     "$PARTITION"     PARTITION
ask "Slurm account (blank = site default)"       "$ACCOUNT"       ACCOUNT
ask "Slurm QOS (blank = site default)"           "$QOS"           QOS
ask "Max array jobs running at once"             "$MAX_PARALLEL"  MAX_PARALLEL

for track in $TRACKS; do
    case "$track" in
        classical|numeric|temporal) ;;
        *) die "unknown track '${track}' (expected: classical, numeric, temporal)" ;;
    esac
done

echo
say "work dir     ${WORK_DIR}"
say "venv         ${VENV_DIR}"
say "tasks        ${TASKS_DIR}"
say "sandbox      ${SANDBOX_DIR}"
say "experiment   ${EXP_DIR}"
say "limits       ${TIME_LIMIT} / ${MEMORY_LIMIT} per task"
say "tracks       ${TRACKS}"
echo

mkdir -p "$WORK_DIR" "$TASKS_DIR"

# ------------------------------------------------------------- experiment --
# Set up before installing: the planner configurations decide what has to be
# installed (the ABA backend needs an extra), and every one of them is used.
if [ ! -f "${EXP_DIR}/exp-details.json" ]; then
    say "creating experiment at ${EXP_DIR}"
    mkdir -p "${EXP_DIR}"
    cp "${SCRIPT_DIR}/exp-configurations/default/exp-details.json" "${EXP_DIR}/"
else
    say "reusing experiment at ${EXP_DIR}"
fi

# Copy in any template configuration the experiment does not have yet, without
# touching the ones it does: a config added to exp-configurations/ after the
# experiment was first created would otherwise never reach it, while local edits
# to an existing config survive a re-run.
mkdir -p "${EXP_DIR}/planners"
for template in "${SCRIPT_DIR}"/exp-configurations/default/planners/*.json; do
    name="$(basename "$template")"
    if [ ! -f "${EXP_DIR}/planners/${name}" ]; then
        cp "$template" "${EXP_DIR}/planners/"
        say "added planner configuration ${name}"
    fi
done

# Every configuration in the directory is benchmarked. To leave one out, delete
# its file -- nothing here filters the set.
PLANNER_COUNT=$(find "${EXP_DIR}/planners" -maxdepth 1 -name '*.json' | wc -l | tr -d ' ')
if [ "$PLANNER_COUNT" -eq 0 ]; then
    die "no planner configuration in ${EXP_DIR}/planners"
fi
say "planners (${PLANNER_COUNT}): $(ls "${EXP_DIR}/planners" | tr '\n' ' ')"

# The ABA backend lives behind an optional extra; install it when a planner
# configuration actually asks for that engine, so a sweep never fails on a
# missing dependency it could have installed.
if grep -lq '"ABAPlanner"' "${EXP_DIR}/planners/"*.json 2>/dev/null; then
    if [ "$WITH_ABA" != "yes" ]; then
        say "a planner configuration uses ABAPlanner; installing the aba extra"
    fi
    WITH_ABA="yes"
fi

# ------------------------------------------------------------- install -----
if [ "$SKIP_INSTALL" = "yes" ]; then
    say "skipping installation (--skip-install)"
    [ -x "${VENV_DIR}/bin/aspbench" ] || warn "no aspbench in ${VENV_DIR}; generation will fail"
else
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "python interpreter not found: ${PYTHON_BIN}"
    if [ ! -d "$VENV_DIR" ]; then
        say "creating virtualenv at ${VENV_DIR}"
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    else
        say "reusing virtualenv at ${VENV_DIR}"
    fi
    # shellcheck disable=SC1091
    . "${VENV_DIR}/bin/activate"
    say "installing ASPPlanners and the benchmark harness"
    python -m pip install --quiet --upgrade pip setuptools wheel
    if [ "$WITH_ABA" = "yes" ]; then
        python -m pip install --quiet -e "${REPO_DIR}[aba]"
    else
        python -m pip install --quiet -e "${REPO_DIR}"
    fi
    python -m pip install --quiet -e "${SCRIPT_DIR}"
    deactivate
fi

ASPBENCH="${VENV_DIR}/bin/aspbench"
VENV_PYTHON="${VENV_DIR}/bin/python"
[ -x "$ASPBENCH" ] || die "aspbench was not installed into ${VENV_DIR}"

# ------------------------------------------------------------- benchmarks --
TASKS_ARGS=()
SUITE_TRACK_ARGS=()
if [ "$SKIP_FETCH" != "yes" ]; then
    if has_track classical; then clone_or_update "$CLASSICAL_REPO" "${TASKS_DIR}/classical-domains"; fi
    if has_track numeric;   then clone_or_update "$NUMERIC_REPO"   "${TASKS_DIR}/numeric-domains"; fi
    if has_track temporal;  then clone_or_update "$TEMPORAL_REPO"  "${TASKS_DIR}/pddl-instances"; fi
fi

if has_track classical && [ -d "${TASKS_DIR}/classical-domains" ]; then
    TASKS_ARGS+=(--tasks-dir "classical-domains=${TASKS_DIR}/classical-domains")
fi
if has_track numeric && [ -d "${TASKS_DIR}/numeric-domains" ]; then
    TASKS_ARGS+=(--tasks-dir "numeric-domains=${TASKS_DIR}/numeric-domains")
fi
if has_track temporal && [ -d "${TASKS_DIR}/pddl-instances" ]; then
    TASKS_ARGS+=(--tasks-dir "pddl-instances=${TASKS_DIR}/pddl-instances")
    # The IPC archive holds all three tracks; take only its temporal domains so
    # its classical and numeric ones do not duplicate the two repos above.
    SUITE_TRACK_ARGS+=(--suite-tracks "pddl-instances=temporal")
fi
if [ ${#TASKS_ARGS[@]} -eq 0 ]; then
    die "no benchmark repository available for tracks: ${TRACKS}"
fi

# ------------------------------------------------------- experiment limits --
say "writing the limits into ${EXP_DIR}/exp-details.json"
ASPBENCH_TIME="$TIME_LIMIT" ASPBENCH_MEM="$MEMORY_LIMIT" ASPBENCH_TRACKS="$TRACKS" \
ASPBENCH_MAXINST="$MAX_INSTANCES" ASPBENCH_PARTITION="$PARTITION" \
ASPBENCH_ACCOUNT="$ACCOUNT" ASPBENCH_QOS="$QOS" ASPBENCH_PARALLEL="$MAX_PARALLEL" \
"$VENV_PYTHON" - "$EXP_DIR" <<'PY'
import json, os, sys

exp_dir = sys.argv[1]
path = os.path.join(exp_dir, 'exp-details.json')
with open(path) as handle:
    details = json.load(handle)

cfgs = details.setdefault('cfgs', {})
cfgs['timelimit'] = os.environ['ASPBENCH_TIME']
cfgs['memorylimit'] = os.environ['ASPBENCH_MEM']
slurm = cfgs.setdefault('slurm', {})
slurm['partition'] = os.environ['ASPBENCH_PARTITION'] or None
slurm['account'] = os.environ['ASPBENCH_ACCOUNT'] or None
slurm['qos'] = os.environ['ASPBENCH_QOS'] or None
slurm['max-parallel-jobs'] = int(os.environ['ASPBENCH_PARALLEL'] or 0)

tasks = details.setdefault('tasks', {})
tasks['tracks'] = os.environ['ASPBENCH_TRACKS'].split()
tasks['max-instances-per-domain'] = int(os.environ['ASPBENCH_MAXINST'] or 0)
details['name'] = os.path.basename(os.path.abspath(exp_dir))

with open(path, 'w') as handle:
    json.dump(details, handle, indent=4)
    handle.write('\n')
print(f'  time={cfgs["timelimit"]} memory={cfgs["memorylimit"]} '
      f'tracks={tasks["tracks"]} instances/domain={tasks["max-instances-per-domain"]}')
PY

# --------------------------------------------------------------- generate --
GENERATE_ARGS=(
    generate
    --exp-dir "$EXP_DIR"
    --sandbox-dir "$SANDBOX_DIR"
    --venv-dir "$VENV_DIR"
    "${TASKS_ARGS[@]}"
)
if [ ${#SUITE_TRACK_ARGS[@]} -gt 0 ]; then
    GENERATE_ARGS+=("${SUITE_TRACK_ARGS[@]}")
fi
if [ "$PER_TASK_SCRIPTS" = "yes" ]; then
    GENERATE_ARGS+=(--per-task-scripts)
fi

echo
say "generating run commands"
"$ASPBENCH" "${GENERATE_ARGS[@]}"

cat <<EOF

Next steps
  submit the sweep      bash ${SANDBOX_DIR}/slurm/submit_all.sh
  or run it locally     bash ${SANDBOX_DIR}/run_local.sh 8
  watch it              squeue -u \$USER -n aspbench
  collect the results   ${ASPBENCH} analyze --sandbox-dir ${SANDBOX_DIR} --per-domain

  re-generate after changing the experiment (skips finished tasks):
    ${ASPBENCH} generate --exp-dir ${EXP_DIR} --sandbox-dir ${SANDBOX_DIR} \\
        --venv-dir ${VENV_DIR} ${TASKS_ARGS[*]} --skip-existing
EOF
