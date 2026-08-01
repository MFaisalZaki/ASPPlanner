# Classical track

**`PLASPPlanner-seq`** — 612/3294 solved and validated (18.6%), 612/2678 of encodable (22.9%), across 76 domains. Median runtime on solved tasks 5.3 s, median peak memory 136.0 MB.

| status | tasks |
|---|---|
| TIMEOUT | 1763 |
| ERROR | 616 |
| SOLVED | 612 |
| MEMOUT | 206 |
| KILLED | 74 |
| EXHAUSTED | 23 |

**`ABAPlanner-ST`** — 318/3294 solved and validated (9.7%), 318/1811 of encodable (17.6%), across 76 domains. Median runtime on solved tasks 19.3 s, median peak memory 291.8 MB.

| status | tasks |
|---|---|
| MEMOUT | 948 |
| UNSUPPORTED | 836 |
| ERROR | 647 |
| KILLED | 374 |
| SOLVED | 318 |
| TIMEOUT | 171 |

*Encodable* = attempted, minus `UNSUPPORTED` (refused on `ProblemKind`) and minus
`ERROR` (overwhelmingly the UP PDDL reader failing on the benchmark file itself).

## Per-domain coverage

| domain | tasks | PLASPPlanner-seq solved | ABAPlanner-ST solved | outcomes (PLASPPlanner-seq) |
|---|---|---|---|---|
| airport | 50 | 11 (22.0%) | 11 (22.0%) | TIMEOUT 39 |
| airport-adl | 50 | 11 (22.0%) | 0 (0.0%) | TIMEOUT 9, MEMOUT 22, KILLED 8 |
| assembly | 30 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 8, MEMOUT 19, KILLED 3 |
| barman-opt14-strips | 14 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 14 |
| barman-sat14-strips | 20 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 20 |
| blocks | 136 | 20 (14.7%) | 21 (15.4%) | TIMEOUT 54, KILLED 6, ERROR 56 |
| blocks-3op | 30 | 14 (46.7%) | 9 (30.0%) | TIMEOUT 16 |
| briefcaseworld | 30 | 7 (23.3%) | 0 (0.0%) | TIMEOUT 23 |
| caldera-opt18 | 20 | 4 (20.0%) | 0 (0.0%) | MEMOUT 13, KILLED 3 |
| caldera-sat18 | 20 | 0 (0.0%) | 0 (0.0%) | MEMOUT 18, KILLED 2 |
| childsnack-opt14-strips | 20 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 20 |
| childsnack-sat14-strips | 20 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 20 |
| depot | 22 | 2 (9.1%) | 2 (9.1%) | TIMEOUT 20 |
| driverlog | 20 | 6 (30.0%) | 7 (35.0%) | TIMEOUT 14 |
| elevators-00-adl | 151 | 34 (22.5%) | 0 (0.0%) | TIMEOUT 116, ERROR 1 |
| elevators-00-full | 150 | 0 (0.0%) | 0 (0.0%) | ERROR 129, EXHAUSTED 21 |
| elevators-00-strips | 150 | 30 (20.0%) | 26 (17.3%) | TIMEOUT 120 |
| ferry | 30 | 12 (40.0%) | 11 (36.7%) | TIMEOUT 18 |
| freecell | 80 | 8 (10.0%) | 1 (1.2%) | TIMEOUT 72 |
| fridge | 30 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 30 |
| grid | 5 | 1 (20.0%) | 0 (0.0%) | TIMEOUT 4 |
| gripper | 20 | 2 (10.0%) | 2 (10.0%) | TIMEOUT 18 |
| hanoi | 30 | 4 (13.3%) | 4 (13.3%) | TIMEOUT 26 |
| hiking-opt14-strips | 20 | 6 (30.0%) | 4 (20.0%) | TIMEOUT 14 |
| hiking-sat14-strips | 20 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 20 |
| logistics00 | 174 | 0 (0.0%) | 0 (0.0%) | ERROR 174 |
| logistics98 | 35 | 2 (5.7%) | 2 (5.7%) | TIMEOUT 30, MEMOUT 1, KILLED 2 |
| maintenance-opt14-adl | 5 | 5 (100.0%) | 0 (0.0%) | — |
| maintenance-sat14-adl | 20 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 20 |
| miconic | 150 | 30 (20.0%) | 27 (18.0%) | TIMEOUT 120 |
| miconic-fulladl | 150 | 35 (23.3%) | 0 (0.0%) | TIMEOUT 111, MEMOUT 4 |
| miconic-simpleadl | 150 | 34 (22.7%) | 0 (0.0%) | TIMEOUT 116 |
| movie | 30 | 30 (100.0%) | 30 (100.0%) | — |
| mprime | 35 | 32 (91.4%) | 7 (20.0%) | TIMEOUT 2, KILLED 1 |
| mystery | 30 | 18 (60.0%) | 8 (26.7%) | TIMEOUT 10, KILLED 1, EXHAUSTED 1 |
| no-mprime | 35 | 31 (88.6%) | 5 (14.3%) | TIMEOUT 2, KILLED 2 |
| no-mystery | 30 | 18 (60.0%) | 8 (26.7%) | TIMEOUT 10, KILLED 1, EXHAUSTED 1 |
| nurikabe-opt18 | 20 | 7 (35.0%) | 0 (0.0%) | TIMEOUT 1, MEMOUT 11, KILLED 1 |
| nurikabe-sat18 | 20 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 2, MEMOUT 14, KILLED 4 |
| openstacks | 30 | 5 (16.7%) | 0 (0.0%) | TIMEOUT 21, MEMOUT 3, KILLED 1 |
| openstacks-strips | 30 | 5 (16.7%) | 5 (16.7%) | TIMEOUT 24, KILLED 1 |
| optical-telegraphs | 48 | 0 (0.0%) | 0 (0.0%) | ERROR 48 |
| organic-synthesis-opt18 | 20 | 0 (0.0%) | 0 (0.0%) | MEMOUT 20 |
| organic-synthesis-sat18 | 20 | 0 (0.0%) | 0 (0.0%) | MEMOUT 18, KILLED 2 |
| pathways | 30 | 4 (13.3%) | 4 (13.3%) | TIMEOUT 4, MEMOUT 18, KILLED 4 |
| pathways-noneg | 30 | 4 (13.3%) | 4 (13.3%) | TIMEOUT 3, MEMOUT 18, KILLED 5 |
| philosophers | 48 | 0 (0.0%) | 0 (0.0%) | ERROR 48 |
| pipesworld-06 | 50 | 10 (20.0%) | 6 (12.0%) | TIMEOUT 40 |
| pipesworld-notankage | 50 | 12 (24.0%) | 10 (20.0%) | TIMEOUT 38 |
| pipesworld-tankage | 50 | 10 (20.0%) | 6 (12.0%) | TIMEOUT 40 |
| psr-large | 50 | 0 (0.0%) | 0 (0.0%) | ERROR 50 |
| psr-middle | 50 | 0 (0.0%) | 0 (0.0%) | ERROR 50 |
| psr-small | 50 | 49 (98.0%) | 48 (96.0%) | TIMEOUT 1 |
| rovers | 40 | 4 (10.0%) | 4 (10.0%) | TIMEOUT 36 |
| rovers-02 | 20 | 4 (20.0%) | 4 (20.0%) | TIMEOUT 16 |
| satellite | 36 | 5 (13.9%) | 4 (11.1%) | TIMEOUT 18, KILLED 13 |
| schedule | 150 | 37 (24.7%) | 0 (0.0%) | TIMEOUT 113 |
| snake-opt18 | 20 | 2 (10.0%) | 0 (0.0%) | TIMEOUT 18 |
| snake-sat18 | 20 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 13, MEMOUT 1, KILLED 6 |
| storage | 30 | 0 (0.0%) | 0 (0.0%) | ERROR 30 |
| termes-opt18 | 20 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 20 |
| termes-sat18 | 20 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 20 |
| thoughtful-sat14-strips | 20 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 8, MEMOUT 11, KILLED 1 |
| tidybot-opt11-strips | 20 | 3 (15.0%) | 1 (5.0%) | TIMEOUT 17 |
| tidybot-opt14-strips | 20 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 20 |
| tidybot-sat11-strips | 20 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 20 |
| tpp | 30 | 5 (16.7%) | 5 (16.7%) | TIMEOUT 25 |
| trucks | 30 | 3 (10.0%) | 2 (6.7%) | TIMEOUT 27 |
| trucks-strips | 30 | 4 (13.3%) | 2 (6.7%) | TIMEOUT 25, KILLED 1 |
| tsp | 30 | 13 (43.3%) | 10 (33.3%) | TIMEOUT 17 |
| tyreworld | 30 | 0 (0.0%) | 0 (0.0%) | ERROR 30 |
| visitall-opt11-strips | 20 | 9 (45.0%) | 8 (40.0%) | TIMEOUT 11 |
| visitall-opt14-strips | 20 | 3 (15.0%) | 2 (10.0%) | TIMEOUT 17 |
| visitall-sat11-strips | 20 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 15, KILLED 5 |
| visitall-sat14-strips | 20 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 4, MEMOUT 15, KILLED 1 |
| zenotravel | 20 | 7 (35.0%) | 8 (40.0%) | TIMEOUT 13 |

Full detail: [`domains.csv`](domains.csv), [`instances.csv`](instances.csv).
