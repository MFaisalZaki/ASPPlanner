"""Experiment configuration: resource limits, task selection, planner set.

An experiment directory holds

    <exp-dir>/exp-details.json      limits + task selection
    <exp-dir>/planners/*.json       one planner configuration per file

which is the layout of `pyPMTEvalToolkit
<https://github.com/pyPMT/pyPMTEvalToolkit>`_, so an experiment written for one
is readable by the other.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_EXP_DETAILS: Dict[str, Any] = {
    'name': 'default',
    'cfgs': {
        'timelimit': '00:30:00',
        'memorylimit': '8GB',
        # Head-room the scheduler gets on top of the task's own limits, so the
        # runner hits its limit first and records TIMEOUT/MEMOUT rather than
        # the job vanishing into a slurm cancellation with no result file.
        'slurm-time-headroom': '00:05:00',
        'slurm-memory-headroom': '1GB',
        'slurm': {
            'cpus-per-task': 1,
            'partition': None,
            'account': None,
            'qos': None,
            'max-parallel-jobs': 50,
            'max-array-size': 1000,
            'extra-directives': [],
        },
    },
    'tasks': {
        'tracks': ['classical', 'numeric', 'temporal'],
        # 0 means every instance of every domain. Set a cap for a smoke run.
        'max-instances-per-domain': 0,
        'selection': 'even',
        'include-domains': [],
        'exclude-domains': [],
        'ipc-years': [],
    },
}


@dataclass
class PlannerConfig:
    """One planner configuration: a UP engine name plus its parameters."""

    tag: str
    engine: str
    params: Dict[str, Any] = field(default_factory=dict)
    tracks: Optional[List[str]] = None      # restrict this planner to some tracks
    path: Optional[str] = None

    @classmethod
    def from_file(cls, path: str) -> 'PlannerConfig':
        with open(path, 'r') as handle:
            raw = json.load(handle)
        tag = raw.get('planner-tag') or os.path.splitext(os.path.basename(path))[0]
        engine = raw.get('up-planner-name') or raw.get('engine')
        if not engine:
            raise ValueError(f'{path}: "up-planner-name" is required')
        return cls(
            tag=tag,
            engine=engine,
            params=raw.get('planner-params') or {},
            tracks=raw.get('tracks'),
            path=os.path.abspath(path),
        )

    def runs_track(self, track: str) -> bool:
        return not self.tracks or track in self.tracks


@dataclass
class Experiment:
    """Parsed ``exp-details.json`` plus the planner configurations beside it."""

    name: str
    details: Dict[str, Any]
    planners: List[PlannerConfig]
    path: str

    # -- resource limits, normalised ------------------------------------
    @property
    def time_limit_seconds(self) -> int:
        return parse_time(self._cfg('timelimit', '00:30:00'))

    @property
    def memory_limit_mb(self) -> int:
        return parse_memory(self._cfg('memorylimit', '8GB'))

    @property
    def slurm_time(self) -> str:
        head = parse_time(self._cfg('slurm-time-headroom', '00:05:00'))
        return format_time(self.time_limit_seconds + head)

    @property
    def slurm_memory(self) -> str:
        head = parse_memory(self._cfg('slurm-memory-headroom', '1GB'))
        return f'{self.memory_limit_mb + head}M'

    @property
    def slurm(self) -> Dict[str, Any]:
        merged = dict(DEFAULT_EXP_DETAILS['cfgs']['slurm'])
        merged.update(self.details.get('cfgs', {}).get('slurm') or {})
        return merged

    @property
    def task_selection(self) -> Dict[str, Any]:
        merged = dict(DEFAULT_EXP_DETAILS['tasks'])
        merged.update(self.details.get('tasks') or {})
        return merged

    def _cfg(self, key: str, fallback):
        return (self.details.get('cfgs') or {}).get(key, fallback)

    @classmethod
    def load(cls, exp_dir: str) -> 'Experiment':
        exp_dir = os.path.abspath(os.path.expanduser(exp_dir))
        details_file = exp_dir if os.path.isfile(exp_dir) else os.path.join(exp_dir, 'exp-details.json')
        if os.path.isfile(exp_dir):
            exp_dir = os.path.dirname(exp_dir)
        if not os.path.isfile(details_file):
            raise FileNotFoundError(f'experiment details not found: {details_file}')
        with open(details_file, 'r') as handle:
            details = json.load(handle)

        planners_dir = os.path.join(exp_dir, 'planners')
        if not os.path.isdir(planners_dir):
            raise FileNotFoundError(f'planner configurations not found: {planners_dir}')
        planners = [PlannerConfig.from_file(os.path.join(planners_dir, name))
                    for name in sorted(os.listdir(planners_dir)) if name.endswith('.json')]
        if not planners:
            raise ValueError(f'no planner configuration (*.json) in {planners_dir}')

        tags = [p.tag for p in planners]
        duplicates = {t for t in tags if tags.count(t) > 1}
        if duplicates:
            raise ValueError(f'duplicate planner tags would overwrite each other: {sorted(duplicates)}')

        return cls(name=details.get('name', os.path.basename(exp_dir)),
                   details=details, planners=planners, path=exp_dir)


# ----------------------------------------------------------------------
# Limit parsing
# ----------------------------------------------------------------------

_MEMORY_RE = re.compile(r'^\s*([0-9]*\.?[0-9]+)\s*([kmgt]?b?)\s*$', re.IGNORECASE)
_MEMORY_UNITS = {'': 1, 'b': 1 / (1024 ** 2), 'k': 1 / 1024, 'kb': 1 / 1024,
                 'm': 1, 'mb': 1, 'g': 1024, 'gb': 1024, 't': 1024 ** 2, 'tb': 1024 ** 2}


def parse_memory(value) -> int:
    """``"8GB"`` / ``"8192"`` / ``8192`` -> mebibytes. A bare number is MiB."""
    if isinstance(value, (int, float)):
        return int(value)
    match = _MEMORY_RE.match(str(value))
    if not match:
        raise ValueError(f'cannot parse memory limit: {value!r} (try "8GB" or "8192MB")')
    amount, unit = float(match.group(1)), match.group(2).lower()
    return max(1, int(round(amount * _MEMORY_UNITS[unit])))


def parse_time(value) -> int:
    """``"00:30:00"`` / ``"30m"`` / ``"1800"`` / ``1800`` -> seconds."""
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().lower()
    if ':' in text:
        parts = [float(p) for p in text.split(':')]
        if len(parts) == 2:            # MM:SS
            return int(parts[0] * 60 + parts[1])
        if len(parts) == 3:            # HH:MM:SS
            return int(parts[0] * 3600 + parts[1] * 60 + parts[2])
        if len(parts) == 4:            # D-HH:MM:SS written with colons
            return int(parts[0] * 86400 + parts[1] * 3600 + parts[2] * 60 + parts[3])
        raise ValueError(f'cannot parse time limit: {value!r}')
    match = re.match(r'^([0-9]*\.?[0-9]+)\s*([smhd]?)$', text)
    if not match:
        raise ValueError(f'cannot parse time limit: {value!r} (try "00:30:00" or "30m")')
    amount = float(match.group(1))
    return int(amount * {'': 1, 's': 1, 'm': 60, 'h': 3600, 'd': 86400}[match.group(2)])


def format_time(seconds: int) -> str:
    """Seconds -> the ``HH:MM:SS`` (or ``D-HH:MM:SS``) slurm wants."""
    days, rest = divmod(int(seconds), 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    if days:
        return f'{days}-{hours:02d}:{minutes:02d}:{secs:02d}'
    return f'{hours:02d}:{minutes:02d}:{secs:02d}'


def write_default_experiment(exp_dir: str, time_limit: Optional[str] = None,
                             memory_limit: Optional[str] = None) -> str:
    """Write a starter experiment directory; used by ``aspbench init``."""
    os.makedirs(os.path.join(exp_dir, 'planners'), exist_ok=True)
    details = json.loads(json.dumps(DEFAULT_EXP_DETAILS))
    if time_limit:
        details['cfgs']['timelimit'] = time_limit
    if memory_limit:
        details['cfgs']['memorylimit'] = memory_limit
    details['name'] = os.path.basename(os.path.abspath(exp_dir))
    details_file = os.path.join(exp_dir, 'exp-details.json')
    with open(details_file, 'w') as handle:
        json.dump(details, handle, indent=4)
        handle.write('\n')

    starters = {
        'plasp-seq.json': {
            'planner-tag': 'PLASPPlanner-seq',
            'up-planner-name': 'PLASPPlanner',
            'planner-params': {'encoding': 'seq', 'max_horizon': 1000, 'time_scale': 10},
        },
        'aba-st.json': {
            'planner-tag': 'ABAPlanner-ST',
            'up-planner-name': 'ABAPlanner',
            'planner-params': {'max_horizon': 100, 'semantics': 'ST', 'time_scale': 2},
            'tracks': ['classical', 'numeric'],      # temporal only is off; see the config's comment
        },
    }
    for name, body in starters.items():
        path = os.path.join(exp_dir, 'planners', name)
        if os.path.exists(path):
            continue
        with open(path, 'w') as handle:
            json.dump(body, handle, indent=4)
            handle.write('\n')
    return details_file
