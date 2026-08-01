# Temporal track

**`PLASPPlanner-seq`** — 5/90 solved and validated (5.6%), 5/90 of encodable (5.6%), across 9 domains. Median runtime on solved tasks 15.0 s, median peak memory 245.1 MB.

| status | tasks |
|---|---|
| MEMOUT | 51 |
| TIMEOUT | 29 |
| SOLVED | 5 |
| KILLED | 5 |

*Encodable* = attempted, minus `UNSUPPORTED` (refused on `ProblemKind`) and minus
`ERROR` (overwhelmingly the UP PDDL reader failing on the benchmark file itself).

## Per-domain coverage

| domain | tasks | PLASPPlanner-seq solved | outcomes (PLASPPlanner-seq) |
|---|---|---|---|
| Cushing | 10 | 1 (10.0%) | TIMEOUT 9 |
| Floortile | 10 | 0 (0.0%) | TIMEOUT 10 |
| Mapanalyser | 10 | 0 (0.0%) | MEMOUT 8, KILLED 2 |
| Parking | 10 | 0 (0.0%) | TIMEOUT 7, MEMOUT 3 |
| airport-temporal-strips | 10 | 2 (20.0%) | MEMOUT 8 |
| quantum_circuit | 10 | 2 (20.0%) | TIMEOUT 3, MEMOUT 5 |
| road-traffic-accident | 10 | 0 (0.0%) | MEMOUT 8, KILLED 2 |
| sokoban | 10 | 0 (0.0%) | MEMOUT 9, KILLED 1 |
| trucks-time-strips | 10 | 0 (0.0%) | MEMOUT 10 |

Full detail: [`domains.csv`](domains.csv), [`instances.csv`](instances.csv).
