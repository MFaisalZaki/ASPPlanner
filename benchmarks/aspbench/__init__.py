"""Benchmark harness for the planners in :mod:`aspplanners`.

Four stages, one per CLI subcommand:

``discover``  walk a benchmark repository and classify every (domain, instance)
              pair into a track (classical / numeric / temporal);
``generate``  cross the discovered tasks with the planner configurations of an
              experiment and emit one run command per pair, plus the slurm job
              arrays that run them;
``solve``     run *one* (planner, task) pair under its own time/memory limits
              and dump a JSON result;
``analyze``   aggregate those JSONs into a CSV and a coverage table.

``discover``/``generate``/``analyze`` are import-light on purpose: they never
touch ``unified_planning``, so commands can be generated on a laptop and only
the compute nodes need the planner installed.
"""

__version__ = "0.1.0"
