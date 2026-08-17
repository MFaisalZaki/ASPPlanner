# Classical track

**`PLASPPlanner-seq`** — 819/4831 solved and validated (17.0%), 819/4212 of encodable (19.4%), across 145 domains. Median runtime on solved tasks 6.3 s, median peak memory 144.0 MB.

| status | tasks |
|---|---|
| TIMEOUT | 2987 |
| SOLVED | 819 |
| ERROR | 619 |
| MEMOUT | 307 |
| KILLED | 76 |
| EXHAUSTED | 23 |

*Encodable* = attempted, minus `UNSUPPORTED` (refused on `ProblemKind`) and minus
`ERROR` (overwhelmingly the UP PDDL reader failing on the benchmark file itself).

## Per-domain coverage

| domain | tasks | PLASPPlanner-seq solved | outcomes (PLASPPlanner-seq) |
|---|---|---|---|
| agricola-opt18 | 20 | 0 (0.0%) | TIMEOUT 7, KILLED 13 |
| agricola-sat18 | 20 | 0 (0.0%) | TIMEOUT 11, KILLED 9 |
| airport | 50 | 11 (22.0%) | TIMEOUT 39 |
| airport-adl | 50 | 11 (22.0%) | TIMEOUT 9, MEMOUT 24, KILLED 6 |
| assembly | 30 | 0 (0.0%) | TIMEOUT 8, MEMOUT 19, KILLED 3 |
| barman-opt11-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| barman-opt14-strips | 14 | 0 (0.0%) | TIMEOUT 14 |
| barman-sat11-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| barman-sat14-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| blocks | 136 | 20 (14.7%) | TIMEOUT 53, MEMOUT 2, KILLED 5, ERROR 56 |
| blocks-3op | 30 | 14 (46.7%) | TIMEOUT 16 |
| briefcaseworld | 30 | 7 (23.3%) | TIMEOUT 23 |
| caldera-opt18 | 20 | 4 (20.0%) | MEMOUT 13, KILLED 3 |
| caldera-sat18 | 20 | 0 (0.0%) | MEMOUT 18, KILLED 2 |
| caldera-split-opt18 | 20 | 8 (40.0%) | TIMEOUT 12 |
| caldera-split-sat18 | 20 | 2 (10.0%) | TIMEOUT 9, MEMOUT 9 |
| cavediving | 20 | 7 (35.0%) | TIMEOUT 13 |
| childsnack-opt14-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| childsnack-sat14-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| citycar-opt14-adl | 20 | 7 (35.0%) | TIMEOUT 13 |
| citycar-sat14-adl | 20 | 0 (0.0%) | TIMEOUT 20 |
| cybersec | 30 | 8 (26.7%) | TIMEOUT 21, KILLED 1 |
| data-network-opt18 | 20 | 15 (75.0%) | TIMEOUT 5 |
| data-network-sat18 | 20 | 0 (0.0%) | TIMEOUT 16, MEMOUT 2, KILLED 2 |
| depot | 22 | 2 (9.1%) | TIMEOUT 20 |
| driverlog | 20 | 6 (30.0%) | TIMEOUT 14 |
| elevators-00-adl | 151 | 34 (22.5%) | TIMEOUT 116, ERROR 1 |
| elevators-00-full | 150 | 0 (0.0%) | ERROR 129, EXHAUSTED 21 |
| elevators-00-strips | 150 | 30 (20.0%) | TIMEOUT 120 |
| elevators-opt08-strips | 30 | 7 (23.3%) | TIMEOUT 23 |
| elevators-opt11-strips | 20 | 5 (25.0%) | TIMEOUT 15 |
| elevators-sat08-strips | 30 | 1 (3.3%) | TIMEOUT 29 |
| elevators-sat11-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| ferry | 30 | 12 (40.0%) | TIMEOUT 18 |
| flashfill-sat18 | 20 | 5 (25.0%) | TIMEOUT 1, MEMOUT 14 |
| floortile-opt11-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| floortile-opt14-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| floortile-sat11-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| floortile-sat14-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| freecell | 80 | 8 (10.0%) | TIMEOUT 72 |
| fridge | 30 | 0 (0.0%) | TIMEOUT 30 |
| ged-opt14-strips | 20 | 20 (100.0%) | — |
| ged-sat14-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| grid | 5 | 1 (20.0%) | TIMEOUT 4 |
| gripper | 20 | 2 (10.0%) | TIMEOUT 18 |
| hanoi | 30 | 4 (13.3%) | TIMEOUT 26 |
| hiking-opt14-strips | 20 | 6 (30.0%) | TIMEOUT 14 |
| hiking-sat14-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| logistics00 | 174 | 0 (0.0%) | ERROR 174 |
| logistics98 | 35 | 2 (5.7%) | TIMEOUT 30, MEMOUT 1, KILLED 2 |
| maintenance-opt14-adl | 5 | 5 (100.0%) | — |
| maintenance-sat14-adl | 20 | 0 (0.0%) | TIMEOUT 20 |
| miconic | 150 | 30 (20.0%) | TIMEOUT 120 |
| miconic-fulladl | 150 | 35 (23.3%) | TIMEOUT 111, MEMOUT 4 |
| miconic-simpleadl | 150 | 34 (22.7%) | TIMEOUT 116 |
| movie | 30 | 30 (100.0%) | — |
| mprime | 35 | 32 (91.4%) | TIMEOUT 2, KILLED 1 |
| mystery | 30 | 18 (60.0%) | TIMEOUT 11, EXHAUSTED 1 |
| no-mprime | 35 | 31 (88.6%) | TIMEOUT 3, KILLED 1 |
| no-mystery | 30 | 18 (60.0%) | TIMEOUT 11, EXHAUSTED 1 |
| nomystery-opt11-strips | 20 | 7 (35.0%) | TIMEOUT 10, MEMOUT 3 |
| nomystery-sat11-strips | 20 | 1 (5.0%) | TIMEOUT 12, MEMOUT 7 |
| nurikabe-opt18 | 20 | 7 (35.0%) | TIMEOUT 1, MEMOUT 11, KILLED 1 |
| nurikabe-sat18 | 20 | 0 (0.0%) | TIMEOUT 2, MEMOUT 14, KILLED 4 |
| openstacks | 30 | 5 (16.7%) | TIMEOUT 21, MEMOUT 4 |
| openstacks-opt08-adl | 30 | 2 (6.7%) | TIMEOUT 28 |
| openstacks-opt08-strips | 30 | 2 (6.7%) | TIMEOUT 28 |
| openstacks-opt11-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| openstacks-opt14-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| openstacks-sat08-adl | 30 | 3 (10.0%) | TIMEOUT 21, MEMOUT 3, KILLED 3 |
| openstacks-sat08-strips | 30 | 3 (10.0%) | TIMEOUT 27 |
| openstacks-sat11-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| openstacks-sat14-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| openstacks-strips | 30 | 5 (16.7%) | TIMEOUT 24, KILLED 1 |
| optical-telegraphs | 48 | 0 (0.0%) | ERROR 48 |
| organic-synthesis-opt18 | 20 | 0 (0.0%) | MEMOUT 20 |
| organic-synthesis-sat18 | 20 | 0 (0.0%) | MEMOUT 20 |
| organic-synthesis-split-opt18 | 20 | 12 (60.0%) | TIMEOUT 3, MEMOUT 5 |
| organic-synthesis-split-sat18 | 20 | 9 (45.0%) | TIMEOUT 3, MEMOUT 8 |
| parcprinter-08-strips | 30 | 8 (26.7%) | TIMEOUT 22 |
| parcprinter-opt11-strips | 20 | 4 (20.0%) | TIMEOUT 16 |
| parcprinter-sat11-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| parking-opt11-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| parking-opt14-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| parking-sat11-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| parking-sat14-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| pathways | 30 | 4 (13.3%) | TIMEOUT 7, MEMOUT 18, KILLED 1 |
| pathways-noneg | 30 | 4 (13.3%) | TIMEOUT 6, MEMOUT 18, KILLED 2 |
| pegsol-08-strips | 30 | 12 (40.0%) | TIMEOUT 18 |
| pegsol-opt11-strips | 20 | 2 (10.0%) | TIMEOUT 18 |
| pegsol-sat11-strips | 20 | 1 (5.0%) | TIMEOUT 19 |
| petri-net-alignment-opt18 | 20 | 0 (0.0%) | TIMEOUT 20 |
| philosophers | 48 | 0 (0.0%) | ERROR 48 |
| pipesworld-06 | 50 | 10 (20.0%) | TIMEOUT 40 |
| pipesworld-notankage | 50 | 12 (24.0%) | TIMEOUT 38 |
| pipesworld-tankage | 50 | 10 (20.0%) | TIMEOUT 40 |
| psr-large | 50 | 0 (0.0%) | ERROR 50 |
| psr-middle | 50 | 0 (0.0%) | ERROR 50 |
| psr-small | 50 | 49 (98.0%) | TIMEOUT 1 |
| rovers | 40 | 4 (10.0%) | TIMEOUT 36 |
| rovers-02 | 20 | 4 (20.0%) | TIMEOUT 16 |
| satellite | 36 | 5 (13.9%) | TIMEOUT 22, KILLED 9 |
| scanalyzer-08-strips | 30 | 7 (23.3%) | TIMEOUT 23 |
| scanalyzer-opt11-strips | 20 | 4 (20.0%) | TIMEOUT 16 |
| scanalyzer-sat11-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| schedule | 150 | 37 (24.7%) | TIMEOUT 113 |
| settlers-opt18 | 20 | 4 (20.0%) | TIMEOUT 16 |
| settlers-sat18 | 20 | 0 (0.0%) | TIMEOUT 2, MEMOUT 16, KILLED 2 |
| snake-opt18 | 20 | 2 (10.0%) | TIMEOUT 18 |
| snake-sat18 | 20 | 0 (0.0%) | TIMEOUT 9, MEMOUT 11 |
| sokoban-opt08-strips | 30 | 3 (10.0%) | TIMEOUT 27 |
| sokoban-opt11-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| sokoban-sat08-strips | 30 | 1 (3.3%) | TIMEOUT 29 |
| sokoban-sat11-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| spider-opt18 | 20 | 0 (0.0%) | TIMEOUT 20 |
| spider-sat18 | 20 | 0 (0.0%) | TIMEOUT 12, MEMOUT 8 |
| storage | 30 | 0 (0.0%) | ERROR 30 |
| termes-opt18 | 20 | 0 (0.0%) | TIMEOUT 20 |
| termes-sat18 | 20 | 0 (0.0%) | TIMEOUT 20 |
| tetris-opt14-strips | 17 | 2 (11.8%) | TIMEOUT 14, KILLED 1 |
| tetris-sat14-strips | 20 | 0 (0.0%) | TIMEOUT 10, MEMOUT 9, KILLED 1 |
| thoughtful-sat14-strips | 20 | 0 (0.0%) | TIMEOUT 8, MEMOUT 11, ERROR 1 |
| tidybot-opt11-strips | 20 | 3 (15.0%) | TIMEOUT 17 |
| tidybot-opt14-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| tidybot-sat11-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| tpp | 30 | 5 (16.7%) | TIMEOUT 25 |
| transport-opt08-strips | 30 | 9 (30.0%) | TIMEOUT 21 |
| transport-opt11-strips | 20 | 4 (20.0%) | TIMEOUT 16 |
| transport-opt14-strips | 20 | 2 (10.0%) | TIMEOUT 18 |
| transport-sat08-strips | 30 | 4 (13.3%) | TIMEOUT 26 |
| transport-sat11-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| transport-sat14-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| trucks | 30 | 3 (10.0%) | TIMEOUT 27 |
| trucks-strips | 30 | 3 (10.0%) | TIMEOUT 27 |
| tsp | 30 | 13 (43.3%) | TIMEOUT 17 |
| tyreworld | 30 | 0 (0.0%) | ERROR 30 |
| visitall-opt11-strips | 20 | 9 (45.0%) | TIMEOUT 11 |
| visitall-opt14-strips | 20 | 3 (15.0%) | TIMEOUT 17 |
| visitall-sat11-strips | 20 | 0 (0.0%) | TIMEOUT 18, KILLED 2 |
| visitall-sat14-strips | 20 | 0 (0.0%) | TIMEOUT 4, MEMOUT 15, KILLED 1 |
| woodworking-opt08-strips | 30 | 9 (30.0%) | TIMEOUT 21 |
| woodworking-opt11-strips | 20 | 4 (20.0%) | TIMEOUT 16 |
| woodworking-sat08-strips | 30 | 4 (13.3%) | TIMEOUT 25, ERROR 1 |
| woodworking-sat11-strips | 20 | 0 (0.0%) | TIMEOUT 19, ERROR 1 |
| zenotravel | 20 | 7 (35.0%) | TIMEOUT 13 |

Full detail: [`domains.csv`](domains.csv), [`instances.csv`](instances.csv).
