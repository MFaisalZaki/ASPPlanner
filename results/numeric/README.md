# Numeric track

**`PLASPPlanner-seq`** — 387/3354 solved and validated (11.5%), 387/2832 of encodable (13.7%), across 162 domains. Median runtime on solved tasks 41.4 s, median peak memory 184.4 MB.

| status | tasks |
|---|---|
| TIMEOUT | 1683 |
| UNSUPPORTED | 497 |
| KILLED | 392 |
| SOLVED | 387 |
| MEMOUT | 367 |
| ERROR | 25 |
| EXHAUSTED | 3 |

*Encodable* = attempted, minus `UNSUPPORTED` (refused on `ProblemKind`) and minus
`ERROR` (overwhelmingly the UP PDDL reader failing on the benchmark file itself).

## Per-domain coverage

| domain | tasks | PLASPPlanner-seq solved | outcomes (PLASPPlanner-seq) |
|---|---|---|---|
| 15-puzzle | 100 | 0 (0.0%) | TIMEOUT 3, MEMOUT 33, KILLED 64 |
| 2048 | 20 | 0 (0.0%) | TIMEOUT 1, MEMOUT 3, KILLED 16 |
| agricola-opt18 | 20 | 0 (0.0%) | TIMEOUT 7, KILLED 13 |
| agricola-sat18 | 20 | 0 (0.0%) | TIMEOUT 9, KILLED 11 |
| barman-opt11-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| barman-sat11-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| block-grouping | 192 | 7 (3.6%) | TIMEOUT 17, MEMOUT 1, KILLED 167 |
| caldera-split-opt18 | 20 | 8 (40.0%) | TIMEOUT 12 |
| caldera-split-sat18 | 20 | 2 (10.0%) | TIMEOUT 9, MEMOUT 9 |
| cavediving | 20 | 7 (35.0%) | TIMEOUT 13 |
| citycar-opt14-adl | 20 | 7 (35.0%) | TIMEOUT 13 |
| citycar-sat14-adl | 20 | 0 (0.0%) | TIMEOUT 20 |
| coins | 20 | 0 (0.0%) | MEMOUT 18, KILLED 2 |
| counters | 55 | 10 (18.2%) | TIMEOUT 44, KILLED 1 |
| cybersec | 30 | 8 (26.7%) | TIMEOUT 21, KILLED 1 |
| data-network-opt18 | 20 | 15 (75.0%) | TIMEOUT 5 |
| data-network-sat18 | 20 | 0 (0.0%) | TIMEOUT 12, MEMOUT 2, KILLED 6 |
| delivery | 20 | 1 (5.0%) | TIMEOUT 9, MEMOUT 10 |
| depots | 20 | 0 (0.0%) | MEMOUT 16, KILLED 4 |
| driverlog | 20 | 0 (0.0%) | ERROR 20 |
| drone | 20 | 2 (10.0%) | TIMEOUT 18 |
| elevators | 30 | 1 (3.3%) | TIMEOUT 29 |
| elevators-opt08-strips | 30 | 6 (20.0%) | TIMEOUT 24 |
| elevators-opt11-strips | 20 | 4 (20.0%) | TIMEOUT 16 |
| elevators-sat08-strips | 30 | 0 (0.0%) | TIMEOUT 30 |
| elevators-sat11-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| expedition | 20 | 0 (0.0%) | TIMEOUT 19, MEMOUT 1 |
| ext-plant-watering | 20 | 0 (0.0%) | TIMEOUT 14, MEMOUT 6 |
| factory-robot | 20 | 0 (0.0%) | UNSUPPORTED 20 |
| farmland | 50 | 0 (0.0%) | TIMEOUT 50 |
| flashfill-sat18 | 20 | 5 (25.0%) | TIMEOUT 1, MEMOUT 14 |
| floortile-opt11-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| floortile-opt14-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| floortile-sat11-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| floortile-sat14-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| fo-counters | 20 | 1 (5.0%) | TIMEOUT 19 |
| fo-farmland | 50 | 0 (0.0%) | UNSUPPORTED 50 |
| fo-sailing | 20 | 0 (0.0%) | UNSUPPORTED 20 |
| forestfire | 20 | 2 (10.0%) | TIMEOUT 18 |
| gear-car | 20 | 0 (0.0%) | UNSUPPORTED 20 |
| ged-opt14-strips | 20 | 20 (100.0%) | — |
| ged-sat14-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| hydropower | 30 | 0 (0.0%) | UNSUPPORTED 30 |
| line-exchange-snp | 20 | 0 (0.0%) | UNSUPPORTED 20 |
| markettrader | 20 | 0 (0.0%) | MEMOUT 11, KILLED 9 |
| minecraft-pogo-advanced | 20 | 9 (45.0%) | TIMEOUT 11 |
| minecraft-sword-advanced | 20 | 20 (100.0%) | — |
| mprime | 30 | 24 (80.0%) | TIMEOUT 6 |
| nlnp-fo-farmland | 50 | 0 (0.0%) | UNSUPPORTED 50 |
| nlnp-fo-sailing | 20 | 0 (0.0%) | UNSUPPORTED 20 |
| nlnp-hydropower | 30 | 0 (0.0%) | UNSUPPORTED 30 |
| nlnp-rover | 10 | 0 (0.0%) | UNSUPPORTED 10 |
| nlnp-settlers | 25 | 0 (0.0%) | UNSUPPORTED 25 |
| nlnp-sugar | 20 | 0 (0.0%) | UNSUPPORTED 20 |
| nomystery-opt11-strips | 20 | 7 (35.0%) | TIMEOUT 4, MEMOUT 3, KILLED 6 |
| nomystery-sat11-strips | 20 | 1 (5.0%) | TIMEOUT 7, MEMOUT 7, KILLED 5 |
| onlycraft-opt | 20 | 1 (5.0%) | TIMEOUT 17, MEMOUT 1, KILLED 1 |
| onlycraft-sat | 20 | 1 (5.0%) | TIMEOUT 15, MEMOUT 4 |
| openstacks-opt08-adl | 30 | 2 (6.7%) | TIMEOUT 28 |
| openstacks-opt08-strips | 30 | 2 (6.7%) | TIMEOUT 28 |
| openstacks-opt11-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| openstacks-opt14-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| openstacks-sat08-adl | 30 | 3 (10.0%) | TIMEOUT 21, KILLED 6 |
| openstacks-sat08-strips | 30 | 3 (10.0%) | TIMEOUT 27 |
| openstacks-sat11-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| openstacks-sat14-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| organic-synthesis-split-opt18 | 20 | 12 (60.0%) | TIMEOUT 3, MEMOUT 5 |
| organic-synthesis-split-sat18 | 20 | 9 (45.0%) | TIMEOUT 3, MEMOUT 8 |
| pancake | 50 | 5 (10.0%) | TIMEOUT 9, MEMOUT 7, KILLED 29 |
| parcprinter-08-strips | 30 | 8 (26.7%) | TIMEOUT 22 |
| parcprinter-opt11-strips | 20 | 4 (20.0%) | TIMEOUT 16 |
| parcprinter-sat11-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| parking-opt11-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| parking-opt14-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| parking-sat11-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| parking-sat14-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| pathwaysmetric | 30 | 0 (0.0%) | TIMEOUT 3, MEMOUT 27 |
| pegsol-08-strips | 30 | 12 (40.0%) | TIMEOUT 18 |
| pegsol-opt11-strips | 20 | 2 (10.0%) | TIMEOUT 18 |
| pegsol-sat11-strips | 20 | 1 (5.0%) | TIMEOUT 19 |
| petri-net | 20 | 0 (0.0%) | TIMEOUT 20 |
| petri-net-alignment-opt18 | 20 | 0 (0.0%) | TIMEOUT 20 |
| petrobras | 70 | 0 (0.0%) | UNSUPPORTED 70 |
| planes | 13 | 1 (7.7%) | TIMEOUT 1, KILLED 11 |
| plant-watering | 51 | 0 (0.0%) | TIMEOUT 42, MEMOUT 9 |
| plotting | 87 | 76 (87.4%) | KILLED 5, ERROR 3, EXHAUSTED 3 |
| rainbowttles-opt | 20 | 0 (0.0%) | TIMEOUT 1, MEMOUT 17, KILLED 2 |
| rainbowttles-sat | 20 | 0 (0.0%) | MEMOUT 18, KILLED 2 |
| rover | 20 | 0 (0.0%) | MEMOUT 19, KILLED 1 |
| rover-linear | 10 | 0 (0.0%) | MEMOUT 10 |
| sailing | 40 | 0 (0.0%) | TIMEOUT 38, MEMOUT 2 |
| sailing-wind-opt | 20 | 0 (0.0%) | UNSUPPORTED 20 |
| sailing-wind-sat | 20 | 0 (0.0%) | UNSUPPORTED 20 |
| satellite | 20 | 0 (0.0%) | MEMOUT 17, KILLED 3 |
| scanalyzer-08-strips | 30 | 7 (23.3%) | TIMEOUT 23 |
| scanalyzer-opt11-strips | 20 | 4 (20.0%) | TIMEOUT 16 |
| scanalyzer-sat11-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| sec_clear_10_2-linear | 1 | 0 (0.0%) | TIMEOUT 1 |
| sec_clear_10_3-linear | 1 | 0 (0.0%) | TIMEOUT 1 |
| sec_clear_10_4-linear | 1 | 0 (0.0%) | TIMEOUT 1 |
| sec_clear_10_5-linear | 1 | 0 (0.0%) | TIMEOUT 1 |
| sec_clear_2_2-linear | 1 | 1 (100.0%) | — |
| sec_clear_2_3-linear | 1 | 1 (100.0%) | — |
| sec_clear_2_4-linear | 1 | 1 (100.0%) | — |
| sec_clear_2_5-linear | 1 | 1 (100.0%) | — |
| sec_clear_3_2-linear | 1 | 1 (100.0%) | — |
| sec_clear_3_3-linear | 1 | 1 (100.0%) | — |
| sec_clear_3_4-linear | 1 | 1 (100.0%) | — |
| sec_clear_3_5-linear | 1 | 1 (100.0%) | — |
| sec_clear_4_2-linear | 1 | 1 (100.0%) | — |
| sec_clear_4_3-linear | 1 | 1 (100.0%) | — |
| sec_clear_4_4-linear | 1 | 1 (100.0%) | — |
| sec_clear_4_5-linear | 1 | 1 (100.0%) | — |
| sec_clear_5_2-linear | 1 | 1 (100.0%) | — |
| sec_clear_5_3-linear | 1 | 1 (100.0%) | — |
| sec_clear_5_4-linear | 1 | 1 (100.0%) | — |
| sec_clear_5_5-linear | 1 | 1 (100.0%) | — |
| sec_clear_6_2-linear | 1 | 1 (100.0%) | — |
| sec_clear_6_3-linear | 1 | 1 (100.0%) | — |
| sec_clear_6_4-linear | 1 | 1 (100.0%) | — |
| sec_clear_6_5-linear | 1 | 1 (100.0%) | — |
| sec_clear_7_2-linear | 1 | 0 (0.0%) | TIMEOUT 1 |
| sec_clear_7_3-linear | 1 | 0 (0.0%) | TIMEOUT 1 |
| sec_clear_7_4-linear | 1 | 0 (0.0%) | TIMEOUT 1 |
| sec_clear_7_5-linear | 1 | 0 (0.0%) | TIMEOUT 1 |
| sec_clear_8_2-linear | 1 | 0 (0.0%) | TIMEOUT 1 |
| sec_clear_8_3-linear | 1 | 0 (0.0%) | TIMEOUT 1 |
| sec_clear_8_4-linear | 1 | 0 (0.0%) | TIMEOUT 1 |
| sec_clear_8_5-linear | 1 | 0 (0.0%) | TIMEOUT 1 |
| sec_clear_9_2-linear | 1 | 0 (0.0%) | TIMEOUT 1 |
| sec_clear_9_3-linear | 1 | 0 (0.0%) | TIMEOUT 1 |
| sec_clear_9_4-linear | 1 | 0 (0.0%) | TIMEOUT 1 |
| sec_clear_9_5-linear | 1 | 0 (0.0%) | TIMEOUT 1 |
| settlers-opt18 | 20 | 4 (20.0%) | TIMEOUT 16 |
| settlers-sat18 | 20 | 0 (0.0%) | TIMEOUT 2, MEMOUT 16, KILLED 2 |
| settlers-snp | 20 | 0 (0.0%) | TIMEOUT 7, MEMOUT 9, KILLED 4 |
| settlersnumeric | 25 | 0 (0.0%) | TIMEOUT 17, MEMOUT 7, KILLED 1 |
| settlersnumericnoassign | 20 | 0 (0.0%) | TIMEOUT 7, MEMOUT 9, KILLED 4 |
| sokoban-opt08-strips | 30 | 3 (10.0%) | TIMEOUT 27 |
| sokoban-opt11-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| sokoban-sat08-strips | 30 | 1 (3.3%) | TIMEOUT 29 |
| sokoban-sat11-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| spider-opt18 | 20 | 0 (0.0%) | TIMEOUT 20 |
| spider-sat18 | 20 | 0 (0.0%) | TIMEOUT 12, MEMOUT 8 |
| sugar | 20 | 0 (0.0%) | TIMEOUT 20 |
| tetris-opt14-strips | 17 | 2 (11.8%) | TIMEOUT 9, KILLED 6 |
| tetris-sat14-strips | 20 | 0 (0.0%) | TIMEOUT 10, MEMOUT 9, KILLED 1 |
| tpp | 40 | 0 (0.0%) | MEMOUT 31, KILLED 9 |
| tpp-metric | 10 | 0 (0.0%) | TIMEOUT 1, UNSUPPORTED 9 |
| transport-opt08-strips | 30 | 9 (30.0%) | TIMEOUT 21 |
| transport-opt11-strips | 20 | 4 (20.0%) | TIMEOUT 16 |
| transport-opt14-strips | 20 | 2 (10.0%) | TIMEOUT 18 |
| transport-sat08-strips | 30 | 4 (13.3%) | TIMEOUT 26 |
| transport-sat11-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| transport-sat14-strips | 20 | 0 (0.0%) | TIMEOUT 20 |
| woodworking-opt08-strips | 30 | 9 (30.0%) | TIMEOUT 21 |
| woodworking-opt11-strips | 20 | 4 (20.0%) | TIMEOUT 16 |
| woodworking-sat08-strips | 30 | 5 (16.7%) | TIMEOUT 24, ERROR 1 |
| woodworking-sat11-strips | 20 | 0 (0.0%) | TIMEOUT 19, ERROR 1 |
| worksworld | 40 | 0 (0.0%) | UNSUPPORTED 40 |
| zenotravel | 23 | 0 (0.0%) | UNSUPPORTED 23 |
| ztalloc-sum | 20 | 0 (0.0%) | TIMEOUT 20 |

Full detail: [`domains.csv`](domains.csv), [`instances.csv`](instances.csv).
