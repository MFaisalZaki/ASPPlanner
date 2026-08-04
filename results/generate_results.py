#!/usr/bin/env python3
"""Rebuild ``results/`` from a finished sweep's sandbox directory.

The sandbox itself (per-task JSONs, tracebacks, slurm logs) is gitignored and
weighs ~20 GB; this script distils it into the checked-in tables:

    results/results.csv                 every (planner, task) pair, one row
    results/domains.csv                 every (planner, domain) pair, all tracks
    results/summary.json                the headline counts
    results/<track>/domains.csv         per-domain coverage and outcome counts
    results/<track>/instances.csv       per-instance outcome, runtime, plan
    results/<track>/README.md           the per-domain table, rendered

The two top-level tables are the ones to hand to somebody else: ``results.csv``
is the sweep per instance, ``domains.csv`` the same sweep per domain. The
per-track directories are those two split by track, plus a rendered table.

Usage:

    python results/generate_results.py --sandbox-dir sandbox-results/sandbox

A ``.running`` marker with no result file beside it is a job the scheduler
reaped before the harness could write anything: it is reported as ``KILLED``,
not silently dropped, so no coverage number is computed over a denominator
smaller than the task list. A planner that was never configured for a track is
left out of that track's files entirely rather than recorded as 0/n.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

TRACKS = ['classical', 'numeric', 'temporal']
STATUSES = ['SOLVED', 'TIMEOUT', 'MEMOUT', 'KILLED', 'ERROR', 'UNSUPPORTED', 'EXHAUSTED', 'UNSOLVABLE']

INSTANCE_COLUMNS = [
    'planner', 'suite', 'domain', 'instance', 'ipc', 'status', 'validated',
    'plan_length', 'makespan', 'parse_seconds', 'solve_seconds', 'total_seconds',
    'peak_memory_mb', 'reason',
]
DOMAIN_COLUMNS = (
    ['planner', 'track', 'suite', 'domain', 'tasks', 'solved', 'coverage_pct']
    + STATUSES
    + ['median_solved_seconds', 'median_solved_mb', 'median_plan_length']
)
RESULT_COLUMNS = ['track'] + INSTANCE_COLUMNS + ['time_limit', 'memory_limit', 'task_id']


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def load_rows(sandbox: str) -> List[Dict[str, Any]]:
    results_dir = os.path.join(sandbox, 'results')
    rows = []
    for tag in sorted(os.listdir(results_dir)):
        tag_dir = os.path.join(results_dir, tag)
        if not os.path.isdir(tag_dir):
            continue
        names = os.listdir(tag_dir)
        done = {n[:-len('.json')] for n in names if n.endswith('.json')}
        for name in sorted(names):
            if name.endswith('.json'):
                payload = _read(os.path.join(tag_dir, name))
            elif name.endswith('.running') and name[:-len('.running')] not in done:
                payload = _read(os.path.join(tag_dir, name))
                if payload is not None:
                    payload['status'] = 'KILLED'      # reaped before it could report
            else:
                continue
            if payload and payload.get('status'):
                rows.append(_row(payload))
    return rows


def _read(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, 'r') as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _row(payload: Dict[str, Any]) -> Dict[str, Any]:
    task = payload.get('task') or {}
    planner = payload.get('planner') or {}
    metrics = payload.get('metrics') or {}
    timings = payload.get('timings') or {}
    limits = payload.get('limits') or {}
    return {
        'planner': planner.get('tag'),
        'track': task.get('track'),
        'suite': task.get('suite'),
        'domain': task.get('domain'),
        'instance': task.get('instance'),
        'ipc': task.get('ipc') or '',
        'status': payload.get('status'),
        'validated': metrics.get('validated'),
        'plan_length': metrics.get('plan-length'),
        'makespan': metrics.get('makespan'),
        'parse_seconds': _round(timings.get('parse-seconds')),
        'solve_seconds': _round(timings.get('solve-seconds')),
        'total_seconds': _round(timings.get('total-seconds')),
        'peak_memory_mb': metrics.get('peak-memory-mb'),
        'reason': _reason(payload),
        'time_limit': limits.get('time-seconds'),
        'memory_limit': limits.get('memory-mb'),
        'task_id': task.get('task-id'),
    }


def _reason(payload: Dict[str, Any]) -> str:
    """Why a non-SOLVED task ended that way, in one line."""
    error = payload.get('error') or {}
    if error:
        return f'{error.get("type")}: {_squash(error.get("message"))}'
    logs = payload.get('logs') or []
    return _squash(logs[0]) if logs else ''


def _squash(text: Optional[str]) -> str:
    if not text:
        return ''
    line = ' '.join(str(text).split())
    return line if len(line) <= 220 else line[:217] + '...'


def _round(value) -> Optional[float]:
    return None if value is None else round(float(value), 3)


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------

def planners_of(rows) -> List[str]:
    """Planner tags, the default engine first — it heads the rendered tables."""
    tags = {row['planner'] for row in rows}
    return sorted(tags, key=lambda tag: (not tag.startswith('PLASP'), tag))


def domain_rows(rows) -> List[Dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row['planner'], row['track'], row['suite'], row['domain'])].append(row)
    out = []
    for (planner, track, suite, domain), group in sorted(groups.items()):
        counts = Counter(row['status'] for row in group)
        solved = [row for row in group if row['status'] == 'SOLVED']
        out.append({
            'planner': planner, 'track': track, 'suite': suite, 'domain': domain,
            'tasks': len(group), 'solved': len(solved),
            'coverage_pct': round(100.0 * len(solved) / len(group), 1),
            **{status: counts.get(status, 0) for status in STATUSES},
            'median_solved_seconds': _median(r['total_seconds'] for r in solved),
            'median_solved_mb': _median(r['peak_memory_mb'] for r in solved),
            'median_plan_length': _median(r['plan_length'] for r in solved),
        })
    return out


def _median(values) -> Optional[float]:
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 1) if values else None


def summarize(rows, sandbox: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {'planners': {}}
    for planner in planners_of(rows):
        mine = [row for row in rows if row['planner'] == planner]
        entry: Dict[str, Any] = {'tracks': {}}
        for track in TRACKS:
            group = [row for row in mine if row['track'] == track]
            if not group:
                continue
            counts = Counter(row['status'] for row in group)
            solved = [row for row in group if row['status'] == 'SOLVED']
            encodable = len(group) - counts.get('UNSUPPORTED', 0) - counts.get('ERROR', 0)
            entry['tracks'][track] = {
                'tasks': len(group),
                'domains': len({row['domain'] for row in group}),
                'solved': len(solved),
                'encodable': encodable,
                'coverage_pct': round(100.0 * len(solved) / len(group), 1),
                'coverage_of_encodable_pct': (
                    round(100.0 * len(solved) / encodable, 1) if encodable else None),
                'status': {s: counts.get(s, 0) for s in STATUSES if counts.get(s)},
                'median_solved_seconds': _median(r['total_seconds'] for r in solved),
                'median_solved_mb': _median(r['peak_memory_mb'] for r in solved),
            }
        solved = [row for row in mine if row['status'] == 'SOLVED']
        entry['total'] = {
            'tasks': len(mine),
            'solved': len(solved),
            'validated': sum(1 for row in solved if row['validated']),
            'median_solved_seconds': _median(r['total_seconds'] for r in solved),
            'median_plan_length': _median(r['plan_length'] for r in solved),
        }
        summary['planners'][planner] = entry
    summary['sandbox'] = os.path.basename(os.path.abspath(sandbox))
    return summary


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------

def write_csv(path: str, columns: List[str], rows) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sort_key(row) -> tuple:
    return (row['planner'], row['track'], row['suite'], row['domain'], row['instance'] or '')


def write_track_readme(path: str, track: str, rows, domains, summary) -> None:
    planners = planners_of(rows)
    lines = [f'# {track.capitalize()} track', '']
    for planner in planners:
        entry = summary['planners'][planner]['tracks'][track]
        lines.append(
            f'**`{planner}`** — {entry["solved"]}/{entry["tasks"]} solved and validated '
            f'({entry["coverage_pct"]}%), {entry["solved"]}/{entry["encodable"]} of encodable '
            f'({entry["coverage_of_encodable_pct"]}%), across {entry["domains"]} domains. '
            f'Median runtime on solved tasks {entry["median_solved_seconds"]} s, '
            f'median peak memory {entry["median_solved_mb"]} MB.')
        lines.append('')
        lines.append('| status | tasks |')
        lines.append('|---|---|')
        for status, count in sorted(entry['status'].items(), key=lambda kv: -kv[1]):
            lines.append(f'| {status} | {count} |')
        lines.append('')
    lines += [
        '*Encodable* = attempted, minus `UNSUPPORTED` (refused on `ProblemKind`) and minus',
        '`ERROR` (overwhelmingly the UP PDDL reader failing on the benchmark file itself).',
        '',
        '## Per-domain coverage', '',
    ]
    header = '| domain | tasks | ' + ' | '.join(f'{p} solved' for p in planners) + ' | outcomes (' + planners[0] + ') |'
    lines.append(header)
    lines.append('|---' * (3 + len(planners)) + '|')
    by_domain = defaultdict(dict)
    for entry in domains:
        by_domain[entry['domain']][entry['planner']] = entry
    for domain in sorted(by_domain):
        entries = by_domain[domain]
        first = entries[planners[0]]
        cells = []
        for planner in planners:
            entry = entries.get(planner)
            cells.append('—' if entry is None else f'{entry["solved"]} ({entry["coverage_pct"]}%)')
        rest = ', '.join(f'{s} {first[s]}' for s in STATUSES if s != 'SOLVED' and first[s])
        lines.append(f'| {domain} | {first["tasks"]} | ' + ' | '.join(cells) + f' | {rest or "—"} |')
    lines += ['', 'Full detail: [`domains.csv`](domains.csv), [`instances.csv`](instances.csv).', '']
    with open(path, 'w') as handle:
        handle.write('\n'.join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--sandbox-dir', required=True,
                        help='a finished sweep: <sandbox>/results/<planner-tag>/*.json')
    parser.add_argument('--output-dir', default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()

    rows = load_rows(args.sandbox_dir)
    if not rows:
        print(f'No results under {args.sandbox_dir}/results')
        return 1
    rows.sort(key=sort_key)

    out = os.path.abspath(args.output_dir)
    os.makedirs(out, exist_ok=True)
    write_csv(os.path.join(out, 'results.csv'), RESULT_COLUMNS, rows)
    write_csv(os.path.join(out, 'domains.csv'), DOMAIN_COLUMNS, domain_rows(rows))

    summary = summarize(rows, args.sandbox_dir)
    with open(os.path.join(out, 'summary.json'), 'w') as handle:
        json.dump(summary, handle, indent=2)
        handle.write('\n')

    for track in TRACKS:
        track_rows = [row for row in rows if row['track'] == track]
        if not track_rows:
            continue
        track_dir = os.path.join(out, track)
        write_csv(os.path.join(track_dir, 'instances.csv'), INSTANCE_COLUMNS, track_rows)
        domains = domain_rows(track_rows)
        write_csv(os.path.join(track_dir, 'domains.csv'), DOMAIN_COLUMNS, domains)
        write_track_readme(os.path.join(track_dir, 'README.md'), track, track_rows, domains, summary)
        print(f'{track:10s} {len(track_rows):6d} rows, {len(domains)} (planner, domain) pairs')

    print(f'Wrote {len(rows)} rows to {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
