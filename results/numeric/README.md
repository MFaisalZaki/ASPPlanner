# Numeric track

**`PLASPPlanner-seq`** — 180/1817 solved and validated (9.9%), 180/1397 of encodable (12.9%), across 93 domains. Median runtime on solved tasks 65.7 s, median peak memory 243.9 MB.

| status | tasks |
|---|---|
| TIMEOUT | 488 |
| UNSUPPORTED | 397 |
| KILLED | 387 |
| MEMOUT | 339 |
| SOLVED | 180 |
| ERROR | 23 |
| EXHAUSTED | 3 |

**`ABAPlanner-ST`** — 7/1817 solved and validated (0.4%), 7/317 of encodable (2.2%), across 93 domains. Median runtime on solved tasks 49.2 s, median peak memory 707.5 MB.

| status | tasks |
|---|---|
| UNSUPPORTED | 1469 |
| TIMEOUT | 217 |
| MEMOUT | 50 |
| KILLED | 43 |
| ERROR | 31 |
| SOLVED | 7 |

*Encodable* = attempted, minus `UNSUPPORTED` (refused on `ProblemKind`) and minus
`ERROR` (overwhelmingly the UP PDDL reader failing on the benchmark file itself).

## Per-domain coverage

| domain | tasks | PLASPPlanner-seq solved | ABAPlanner-ST solved | outcomes (PLASPPlanner-seq) |
|---|---|---|---|---|
| 15-puzzle | 100 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 3, MEMOUT 34, KILLED 63 |
| 2048 | 20 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 1, MEMOUT 3, KILLED 16 |
| block-grouping | 192 | 7 (3.6%) | 0 (0.0%) | TIMEOUT 17, MEMOUT 1, KILLED 167 |
| coins | 20 | 0 (0.0%) | 0 (0.0%) | MEMOUT 17, KILLED 3 |
| counters | 55 | 10 (18.2%) | 0 (0.0%) | TIMEOUT 44, KILLED 1 |
| delivery | 20 | 1 (5.0%) | 0 (0.0%) | TIMEOUT 8, MEMOUT 11 |
| depots | 20 | 0 (0.0%) | 0 (0.0%) | MEMOUT 18, KILLED 2 |
| driverlog | 20 | 0 (0.0%) | 0 (0.0%) | ERROR 20 |
| drone | 20 | 2 (10.0%) | 0 (0.0%) | TIMEOUT 18 |
| elevators | 30 | 1 (3.3%) | 0 (0.0%) | TIMEOUT 29 |
| expedition | 20 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 19, MEMOUT 1 |
| ext-plant-watering | 20 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 14, MEMOUT 6 |
| factory-robot | 20 | 0 (0.0%) | 0 (0.0%) | UNSUPPORTED 20 |
| farmland | 50 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 50 |
| fo-counters | 20 | 1 (5.0%) | 0 (0.0%) | TIMEOUT 19 |
| fo-farmland | 50 | 0 (0.0%) | 0 (0.0%) | UNSUPPORTED 50 |
| fo-sailing | 20 | 0 (0.0%) | 0 (0.0%) | UNSUPPORTED 20 |
| forestfire | 20 | 2 (10.0%) | 3 (15.0%) | TIMEOUT 18 |
| gear-car | 20 | 0 (0.0%) | 0 (0.0%) | UNSUPPORTED 20 |
| hydropower | 30 | 0 (0.0%) | 0 (0.0%) | KILLED 30 |
| line-exchange-snp | 20 | 0 (0.0%) | 0 (0.0%) | UNSUPPORTED 20 |
| markettrader | 20 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 1, MEMOUT 13, KILLED 6 |
| minecraft-pogo-advanced | 20 | 9 (45.0%) | 0 (0.0%) | TIMEOUT 11 |
| minecraft-sword-advanced | 20 | 20 (100.0%) | 0 (0.0%) | — |
| mprime | 30 | 23 (76.7%) | 0 (0.0%) | TIMEOUT 7 |
| nlnp-fo-farmland | 50 | 0 (0.0%) | 0 (0.0%) | UNSUPPORTED 50 |
| nlnp-fo-sailing | 20 | 0 (0.0%) | 0 (0.0%) | UNSUPPORTED 20 |
| nlnp-hydropower | 30 | 0 (0.0%) | 0 (0.0%) | UNSUPPORTED 30 |
| nlnp-rover | 10 | 0 (0.0%) | 0 (0.0%) | UNSUPPORTED 10 |
| nlnp-settlers | 25 | 0 (0.0%) | 0 (0.0%) | UNSUPPORTED 25 |
| nlnp-sugar | 20 | 0 (0.0%) | 0 (0.0%) | UNSUPPORTED 20 |
| onlycraft-opt | 20 | 1 (5.0%) | 0 (0.0%) | TIMEOUT 17, MEMOUT 1, KILLED 1 |
| onlycraft-sat | 20 | 1 (5.0%) | 0 (0.0%) | TIMEOUT 15, MEMOUT 4 |
| pancake | 50 | 5 (10.0%) | 0 (0.0%) | TIMEOUT 4, MEMOUT 6, KILLED 35 |
| pathwaysmetric | 30 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 3, MEMOUT 27 |
| petri-net | 20 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 20 |
| petrobras | 70 | 0 (0.0%) | 0 (0.0%) | MEMOUT 44, KILLED 26 |
| planes | 13 | 1 (7.7%) | 0 (0.0%) | KILLED 12 |
| plant-watering | 51 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 43, MEMOUT 8 |
| plotting | 87 | 76 (87.4%) | 0 (0.0%) | KILLED 5, ERROR 3, EXHAUSTED 3 |
| rainbowttles-opt | 20 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 1, MEMOUT 17, KILLED 2 |
| rainbowttles-sat | 20 | 0 (0.0%) | 0 (0.0%) | MEMOUT 17, KILLED 3 |
| rover | 20 | 0 (0.0%) | 4 (20.0%) | MEMOUT 19, KILLED 1 |
| rover-linear | 10 | 0 (0.0%) | 0 (0.0%) | MEMOUT 10 |
| sailing | 40 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 38, MEMOUT 2 |
| sailing-wind-opt | 20 | 0 (0.0%) | 0 (0.0%) | UNSUPPORTED 20 |
| sailing-wind-sat | 20 | 0 (0.0%) | 0 (0.0%) | UNSUPPORTED 20 |
| satellite | 20 | 0 (0.0%) | 0 (0.0%) | MEMOUT 17, KILLED 3 |
| sec_clear_10_2-linear | 1 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 1 |
| sec_clear_10_3-linear | 1 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 1 |
| sec_clear_10_4-linear | 1 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 1 |
| sec_clear_10_5-linear | 1 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 1 |
| sec_clear_2_2-linear | 1 | 1 (100.0%) | 0 (0.0%) | — |
| sec_clear_2_3-linear | 1 | 1 (100.0%) | 0 (0.0%) | — |
| sec_clear_2_4-linear | 1 | 1 (100.0%) | 0 (0.0%) | — |
| sec_clear_2_5-linear | 1 | 1 (100.0%) | 0 (0.0%) | — |
| sec_clear_3_2-linear | 1 | 1 (100.0%) | 0 (0.0%) | — |
| sec_clear_3_3-linear | 1 | 1 (100.0%) | 0 (0.0%) | — |
| sec_clear_3_4-linear | 1 | 1 (100.0%) | 0 (0.0%) | — |
| sec_clear_3_5-linear | 1 | 1 (100.0%) | 0 (0.0%) | — |
| sec_clear_4_2-linear | 1 | 1 (100.0%) | 0 (0.0%) | — |
| sec_clear_4_3-linear | 1 | 1 (100.0%) | 0 (0.0%) | — |
| sec_clear_4_4-linear | 1 | 1 (100.0%) | 0 (0.0%) | — |
| sec_clear_4_5-linear | 1 | 1 (100.0%) | 0 (0.0%) | — |
| sec_clear_5_2-linear | 1 | 1 (100.0%) | 0 (0.0%) | — |
| sec_clear_5_3-linear | 1 | 1 (100.0%) | 0 (0.0%) | — |
| sec_clear_5_4-linear | 1 | 1 (100.0%) | 0 (0.0%) | — |
| sec_clear_5_5-linear | 1 | 1 (100.0%) | 0 (0.0%) | — |
| sec_clear_6_2-linear | 1 | 1 (100.0%) | 0 (0.0%) | — |
| sec_clear_6_3-linear | 1 | 1 (100.0%) | 0 (0.0%) | — |
| sec_clear_6_4-linear | 1 | 1 (100.0%) | 0 (0.0%) | — |
| sec_clear_6_5-linear | 1 | 1 (100.0%) | 0 (0.0%) | — |
| sec_clear_7_2-linear | 1 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 1 |
| sec_clear_7_3-linear | 1 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 1 |
| sec_clear_7_4-linear | 1 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 1 |
| sec_clear_7_5-linear | 1 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 1 |
| sec_clear_8_2-linear | 1 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 1 |
| sec_clear_8_3-linear | 1 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 1 |
| sec_clear_8_4-linear | 1 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 1 |
| sec_clear_8_5-linear | 1 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 1 |
| sec_clear_9_2-linear | 1 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 1 |
| sec_clear_9_3-linear | 1 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 1 |
| sec_clear_9_4-linear | 1 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 1 |
| sec_clear_9_5-linear | 1 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 1 |
| settlers-snp | 20 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 7, MEMOUT 10, KILLED 3 |
| settlersnumeric | 25 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 17, MEMOUT 7, KILLED 1 |
| settlersnumericnoassign | 20 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 7, MEMOUT 11, KILLED 2 |
| sugar | 20 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 20 |
| tpp | 40 | 0 (0.0%) | 0 (0.0%) | MEMOUT 35, KILLED 5 |
| tpp-metric | 10 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 1, UNSUPPORTED 9 |
| worksworld | 40 | 0 (0.0%) | 0 (0.0%) | UNSUPPORTED 40 |
| zenotravel | 23 | 0 (0.0%) | 0 (0.0%) | UNSUPPORTED 23 |
| ztalloc-sum | 20 | 0 (0.0%) | 0 (0.0%) | TIMEOUT 20 |

Full detail: [`domains.csv`](domains.csv), [`instances.csv`](instances.csv).
