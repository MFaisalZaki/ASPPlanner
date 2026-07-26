#!/usr/bin/env python3
"""Cluster the crash logs a benchmark sweep leaves in ``sandbox/errors``.

A sweep writes one ``.log`` per crashed task, each holding the instance id, the
exception headline, and the traceback.  With a couple of thousand of those the
question is never "what does this one say" but "how many *distinct* bugs are in
here, and which of them are mine".

So the script does three things a ``grep | sort | uniq -c`` cannot:

* it separates failures raised while *parsing* the PDDL (a reader limitation,
  or an instance this codebase never has to support) from failures raised
  while *planning* -- only the latter are necessarily bugs in aspplanners;
* it normalises the headline (object names, line/col numbers, arities) into a
  signature, so ``Name p7 already defined`` and ``Name cart already defined``
  land in one cluster instead of two hundred;
* it reports, per cluster, the deepest aspplanners frame, which is the line to
  go fix.

Usage::

    python benchmarks/analyze_errors.py sandbox/errors
    python benchmarks/analyze_errors.py sandbox/errors --mine        # our bugs only
    python benchmarks/analyze_errors.py sandbox/errors --cluster 3   # drill into one
    python benchmarks/analyze_errors.py sandbox/errors --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# A traceback frame line: '  File "/path/to/x.py", line 12, in func'
_FRAME_RE = re.compile(r'^\s*File "(?P<path>[^"]+)", line (?P<line>\d+), in (?P<func>.*)$')
# 'SomeError: message' / 'some.module.SomeError: message'
_HEADLINE_RE = re.compile(r'^(?P<exc>[A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning))'
                          r'(?::\s*(?P<msg>.*))?$')

_OUR_PACKAGE = 'aspplanners'
_READER_MARKERS = ('unified_planning/io/pddl_reader.py', 'unified_planning/io/pddl/',
                   'pyparsing')

# Substitutions applied, in order, to turn a message into a cluster signature.
# Each strips the part that varies per instance while keeping what varies per
# *bug*, so distinct root causes never collapse into one another.
_NORMALISERS: Sequence[Tuple[re.Pattern, str]] = (
    (re.compile(r'\(at char \d+\)'), '(at char N)'),
    (re.compile(r'\(line:\s*\d+,\s*col:\s*\d+\)'), '(line:N, col:N)'),
    (re.compile(r'[Ll]ine:?\s*\d+,\s*col:?\s*\d+'), 'line N, col N'),
    (re.compile(r'\bline \d+\b'), 'line N'),
    (re.compile(r'\barity \d+\b'), 'arity N'),
    (re.compile(r'\b\d+ parameters\b'), 'N parameters'),
    (re.compile(r"'[^']*'"), "'X'"),
    (re.compile(r'"[^"]*"'), '"X"'),
    (re.compile(r'\((?:[^()]|\([^()]*\))*\)'), '(...)'),
    (re.compile(r'\b\d+(?:\.\d+)?\b'), 'N'),
    (re.compile(r'\s+'), ' '),
)

# Messages whose *whole* interesting part is a name we just stripped need a
# hand-written signature, otherwise they normalise down to something useless.
# A ``\1``-style backreference in the replacement is expanded against the match.
_SPECIAL_SIGNATURES: Sequence[Tuple[re.Pattern, str]] = (
    # pyparsing's expected/found pair is the whole diagnosis and is *not*
    # instance-specific: "Expected 'problem', found 'domain'" is a mispaired
    # file, "Expected ')', found '('" is PDDL the reader cannot chew.  Keeping
    # the tokens is the difference between one useless cluster and two useful.
    (re.compile(r"^Expected (?P<want>\S+(?: \S+)?), found (?P<got>\S+?)\s*(?:\(at char|$)"),
     r'Expected \g<want>, found \g<got>'),
    (re.compile(r'^Name .* already defined'), 'Name <x> already defined (error_used_name)'),
    (re.compile(r'^Undefined name found'), 'Undefined name found: <x>'),
    (re.compile(r'^Not able to handle'), 'Not able to handle: <expr>'),
    (re.compile(r'^Found invalid expression'), 'Found invalid expression: <expr>'),
    (re.compile(r'^Unsupported numeric term'), 'Unsupported numeric term: <expr>'),
)

PARSE, SOLVE, OTHER = 'parse', 'solve', 'other'

# Which *file* the reader choked on, read off the source lines the traceback
# quotes.  A domain that will not parse takes every instance of that domain
# down with it and is one fix; a problem that will not parse is per-instance.
_TARGETS: Sequence[Tuple[str, str]] = (
    ('self._pp_domain', 'domain file'),
    ('self._pp_problem', 'problem file'),
)


@dataclass
class Failure:
    """One crash log, reduced to the fields worth grouping on."""

    path: str
    suite: str
    domain: str
    instance: str
    exc: str
    message: str
    signature: str
    stage: str
    target: str = ''
    frames: List[Tuple[str, int, str]] = field(default_factory=list)
    our_frame: Optional[Tuple[str, int, str]] = None

    @property
    def task(self) -> str:
        return f'{self.suite}:{self.domain}:{self.instance}'

    @property
    def our_frame_str(self) -> str:
        if not self.our_frame:
            return ''
        path, line, func = self.our_frame
        return f'{_short_path(path)}:{line} in {func}()'


def _short_path(path: str) -> str:
    """Trim an absolute path back to its package-relative part."""
    for marker in (f'/{_OUR_PACKAGE}/', '/site-packages/', '/benchmarks/'):
        index = path.rfind(marker)
        if index != -1:
            keep = path[index + 1:] if marker != '/site-packages/' else path[index + len(marker):]
            return keep
    return os.path.basename(path)


def _normalise(message: str) -> str:
    for pattern, signature in _SPECIAL_SIGNATURES:
        match = pattern.search(message)
        if match:
            return match.expand(signature) if '\\' in signature else signature
    text = message
    for pattern, replacement in _NORMALISERS:
        text = pattern.sub(replacement, text)
    return text.strip()


def _split_task(header: str, filename: str) -> Tuple[str, str, str]:
    parts = header.split(':')
    if len(parts) >= 3:
        return parts[0], parts[1], ':'.join(parts[2:])
    # Fall back on the filename: '<planner>__<suite>_<domain>_<instance>.log'.
    stem = os.path.splitext(filename)[0]
    _, _, rest = stem.partition('__')
    bits = rest.split('_')
    if len(bits) >= 3:
        return bits[0], bits[1], '_'.join(bits[2:])
    return 'unknown', 'unknown', stem


def _classify_stage(frames: Sequence[Tuple[str, int, str]], has_our_frame: bool) -> str:
    """parse = died in the PDDL reader, solve = died in the planner."""
    paths = [path for path, _, _ in frames]
    if has_our_frame:
        return SOLVE
    if any(marker in path for path in paths for marker in _READER_MARKERS):
        return PARSE
    if any('runner.py' in path and func == 'solve' for path, _, func in frames):
        # The runner's own frame tells us which call was in flight.
        for path, _, func in frames:
            if 'engines' in path or 'oneshot_planner' in path:
                return SOLVE
        return PARSE if any('parse' in func for _, _, func in frames) else OTHER
    return OTHER


def parse_log(path: str) -> Optional[Failure]:
    try:
        with open(path, 'r', errors='replace') as handle:
            text = handle.read()
    except OSError as error:
        print(f'warning: cannot read {path}: {error}', file=sys.stderr)
        return None
    lines = text.splitlines()
    if not lines:
        return None

    header = lines[0].strip()
    # The headline runs from line 1 to the first blank line; UP likes to append
    # 'Error from line: .. col: ..' on a second line.
    headline_lines: List[str] = []
    for line in lines[1:]:
        if not line.strip():
            break
        headline_lines.append(line.strip())
    headline = ' '.join(headline_lines)

    match = _HEADLINE_RE.match(headline)
    if match:
        exc, message = match.group('exc'), (match.group('msg') or '')
    else:
        exc, message = 'Unknown', headline

    frames = [(m.group('path'), int(m.group('line')), m.group('func').strip())
              for m in (_FRAME_RE.match(line) for line in lines) if m]
    our_frames = [f for f in frames if f'/{_OUR_PACKAGE}/' in f[0]]
    our_frame = our_frames[-1] if our_frames else None

    target = next((label for marker, label in _TARGETS if marker in text), '')
    suite, domain, instance = _split_task(header, os.path.basename(path))
    return Failure(
        path=path,
        suite=suite,
        domain=domain,
        instance=instance,
        exc=exc.rsplit('.', 1)[-1],
        message=message,
        signature=f'{exc.rsplit(".", 1)[-1]}: {_normalise(message)}' if message else exc,
        stage=_classify_stage(frames, our_frame is not None),
        target=target,
        frames=frames,
        our_frame=our_frame,
    )


def load(directory: str) -> Tuple[List[Failure], List[str]]:
    """Return the parsed failures and the logs that held nothing to parse.

    An empty log is its own signal -- the task died before it could write a
    traceback, i.e. it was killed (OOM, walltime, scheduler) rather than
    raising -- so it is reported, never silently dropped from the denominator.
    """
    names = sorted(n for n in os.listdir(directory) if n.endswith('.log'))
    failures, unparsed = [], []
    for name in names:
        failure = parse_log(os.path.join(directory, name))
        if failure is None:
            unparsed.append(name)
        else:
            failures.append(failure)
    return failures, unparsed


def _bar(count: int, total: int, width: int = 24) -> str:
    filled = round(width * count / total) if total else 0
    return '#' * filled + '.' * (width - filled)


def _print_table(title: str, counter: Counter, total: int, limit: int = 20) -> None:
    print(f'\n{title}')
    print('-' * len(title))
    for key, count in counter.most_common(limit):
        print(f'  {count:5d}  {100 * count / total:5.1f}%  {_bar(count, total)}  {key}')
    if len(counter) > limit:
        print(f'  ... and {len(counter) - limit} more')


def summarise(failures: Sequence[Failure], unparsed: Sequence[str], top: int,
              mine_only: bool) -> List[Tuple[str, List[Failure]]]:
    total = len(failures)
    print(f'{total + len(unparsed)} crash log(s); {total} with a traceback')
    if unparsed:
        print(f'\n{len(unparsed)} empty/unreadable log(s) -- killed before writing '
              f'a traceback (OOM or walltime), not an exception:')
        for name in list(unparsed)[:10]:
            print(f'  {name}')
        if len(unparsed) > 10:
            print(f'  ... and {len(unparsed) - 10} more')

    stages = Counter(f.stage for f in failures)
    print('\nWhere it died')
    print('-------------')
    labels = {PARSE: 'PDDL parsing (unified_planning reader)',
              SOLVE: f'planning ({_OUR_PACKAGE} on the stack)',
              OTHER: 'elsewhere / indeterminate'}
    for stage in (PARSE, SOLVE, OTHER):
        count = stages.get(stage, 0)
        if count:
            print(f'  {count:5d}  {100 * count / total:5.1f}%  {_bar(count, total)}  {labels[stage]}')

    parse_failures = [f for f in failures if f.stage == PARSE]
    if parse_failures:
        print('\nWhich file the reader choked on')
        print('-------------------------------')
        targets = Counter(f.target or 'after both parsed (semantic)' for f in parse_failures)
        for label, count in targets.most_common():
            print(f'  {count:5d}  {100 * count / len(parse_failures):5.1f}%  '
                  f'{_bar(count, len(parse_failures))}  {label}')
        broken = Counter(f'{f.suite}:{f.domain}' for f in parse_failures
                         if f.target == 'domain file')
        if broken:
            print(f'\n  {len(broken)} domain file(s) never parse, taking '
                  f'{sum(broken.values())} instance(s) with them -- one fix each, not many:')
            for name, count in broken.most_common():
                example = next(f for f in parse_failures
                               if f.target == 'domain file' and f'{f.suite}:{f.domain}' == name)
                print(f'    {count:5d}  {name:62s} {example.signature}')

    _print_table('Exception types', Counter(f.exc for f in failures), total)
    _print_table('Domains', Counter(f'{f.suite}:{f.domain}' for f in failures), total, limit=15)

    clusters: Dict[str, List[Failure]] = defaultdict(list)
    for failure in failures:
        if mine_only and failure.stage != SOLVE:
            continue
        clusters[failure.signature].append(failure)

    ordered = sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)
    shown = ordered[:top]
    scope = f'{_OUR_PACKAGE} failures' if mine_only else 'all failures'
    covered = sum(len(v) for v in clusters.values())
    print(f'\nDistinct failure signatures ({len(ordered)} over {covered} {scope}); top {len(shown)}')
    print('=' * 72)
    for index, (signature, group) in enumerate(shown, start=1):
        first = group[0]
        domains = Counter(f'{f.suite}:{f.domain}' for f in group)
        our_frames = Counter(f.our_frame_str for f in group if f.our_frame)
        targets = Counter(f.target for f in group if f.target)
        where = f'{first.stage}'
        if targets:
            where += ' (' + ', '.join(f'{t} x{c}' for t, c in targets.most_common()) + ')'
        print(f'\n[{index}] {len(group):4d}x  {signature}')
        print(f'      stage : {where}')
        print(f'      spread: {len(domains)} domain(s) -- '
              + ', '.join(f'{d} x{c}' for d, c in domains.most_common(4))
              + (' ...' if len(domains) > 4 else ''))
        print(f'      sample: {first.task}')
        print(f'         msg: {first.message[:160] or "(no message)"}')
        if our_frames:
            for frame, count in our_frames.most_common(3):
                print(f'    {_OUR_PACKAGE}: {frame}  (x{count})')
    return ordered


def drill(ordered: Sequence[Tuple[str, List[Failure]]], index: int, limit: int) -> None:
    if not 1 <= index <= len(ordered):
        print(f'\nNo cluster {index} (have 1..{len(ordered)})', file=sys.stderr)
        return
    signature, group = ordered[index - 1]
    print(f'\n\nCluster [{index}] {signature} -- {len(group)} failure(s)')
    print('=' * 72)
    for failure in group[:limit]:
        print(f'  {failure.task}')
        print(f'      {failure.message[:200]}')
        print(f'      log: {failure.path}')
    if len(group) > limit:
        print(f'  ... and {len(group) - limit} more')
    print('\nFull traceback of the first one:')
    print('-' * 32)
    with open(group[0].path, 'r', errors='replace') as handle:
        print(handle.read().rstrip())


def write_csv(failures: Sequence[Failure], path: str) -> None:
    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['suite', 'domain', 'instance', 'stage', 'target', 'exception',
                         'signature', 'message', 'aspplanners_frame', 'log'])
        for f in failures:
            writer.writerow([f.suite, f.domain, f.instance, f.stage, f.target, f.exc,
                             f.signature, f.message, f.our_frame_str, f.path])
    print(f'\nWrote {len(failures)} row(s) to {path}')


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('errors_dir', nargs='?', default='sandbox/errors',
                        help='directory of per-task .log files (default: sandbox/errors)')
    parser.add_argument('--top', type=int, default=20, help='clusters to show (default: 20)')
    parser.add_argument('--mine', action='store_true',
                        help=f'cluster only failures with {_OUR_PACKAGE} on the stack')
    parser.add_argument('--cluster', type=int, default=None,
                        help='print every instance in cluster N plus a full traceback')
    parser.add_argument('--limit', type=int, default=25,
                        help='instances to list when drilling into a cluster')
    parser.add_argument('--csv', default=None, help='also write one row per failure here')
    args = parser.parse_args(argv)

    directory = os.path.abspath(os.path.expanduser(args.errors_dir))
    if not os.path.isdir(directory):
        print(f'No such directory: {directory}', file=sys.stderr)
        return 1

    failures, unparsed = load(directory)
    if not failures and not unparsed:
        print(f'No .log files in {directory}')
        return 0

    ordered = summarise(failures, unparsed, args.top, args.mine)
    if args.cluster is not None:
        drill(ordered, args.cluster, args.limit)
    if args.csv:
        write_csv(failures, args.csv)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
