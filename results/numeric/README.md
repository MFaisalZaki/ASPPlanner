# Numeric track

**`PLASPPlanner-seq`** — 181/1817 solved and validated (10.0%), 181/1297 of encodable (14.0%), across 93 domains. Median runtime on solved tasks 79.8 s, median peak memory 243.7 MB.

| status | tasks |
|---|---|
| UNSUPPORTED | 497 |
| TIMEOUT | 492 |
| KILLED | 335 |
| MEMOUT | 286 |
| SOLVED | 181 |
| ERROR | 23 |
| EXHAUSTED | 3 |

*Encodable* = attempted, minus `UNSUPPORTED` (refused on `ProblemKind`) and minus
`ERROR` (overwhelmingly the UP PDDL reader failing on the benchmark file itself).

## Per-domain coverage

| domain | tasks | PLASPPlanner-seq solved | outcomes (PLASPPlanner-seq) |
|---|---|---|---|
| 15-puzzle | 100 | 0 (0.0%) | TIMEOUT 3, MEMOUT 33, KILLED 64 |
| 2048 | 20 | 0 (0.0%) | TIMEOUT 1, MEMOUT 3, KILLED 16 |
| block-grouping | 192 | 7 (3.6%) | TIMEOUT 17, MEMOUT 1, KILLED 167 |
| coins | 20 | 0 (0.0%) | MEMOUT 18, KILLED 2 |
| counters | 55 | 10 (18.2%) | TIMEOUT 44, KILLED 1 |
| delivery | 20 | 1 (5.0%) | TIMEOUT 9, MEMOUT 10 |
| depots | 20 | 0 (0.0%) | MEMOUT 16, KILLED 4 |
| driverlog | 20 | 0 (0.0%) | ERROR 20 |
| drone | 20 | 2 (10.0%) | TIMEOUT 18 |
| elevators | 30 | 1 (3.3%) | TIMEOUT 29 |
| expedition | 20 | 0 (0.0%) | TIMEOUT 19, MEMOUT 1 |
| ext-plant-watering | 20 | 0 (0.0%) | TIMEOUT 14, MEMOUT 6 |
| factory-robot | 20 | 0 (0.0%) | UNSUPPORTED 20 |
| farmland | 50 | 0 (0.0%) | TIMEOUT 50 |
| fo-counters | 20 | 1 (5.0%) | TIMEOUT 19 |
| fo-farmland | 50 | 0 (0.0%) | UNSUPPORTED 50 |
| fo-sailing | 20 | 0 (0.0%) | UNSUPPORTED 20 |
| forestfire | 20 | 2 (10.0%) | TIMEOUT 18 |
| gear-car | 20 | 0 (0.0%) | UNSUPPORTED 20 |
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
| onlycraft-opt | 20 | 1 (5.0%) | TIMEOUT 17, MEMOUT 1, KILLED 1 |
| onlycraft-sat | 20 | 1 (5.0%) | TIMEOUT 15, MEMOUT 4 |
| pancake | 50 | 5 (10.0%) | TIMEOUT 9, MEMOUT 7, KILLED 29 |
| pathwaysmetric | 30 | 0 (0.0%) | TIMEOUT 3, MEMOUT 27 |
| petri-net | 20 | 0 (0.0%) | TIMEOUT 20 |
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
| settlers-snp | 20 | 0 (0.0%) | TIMEOUT 7, MEMOUT 9, KILLED 4 |
| settlersnumeric | 25 | 0 (0.0%) | TIMEOUT 17, MEMOUT 7, KILLED 1 |
| settlersnumericnoassign | 20 | 0 (0.0%) | TIMEOUT 7, MEMOUT 9, KILLED 4 |
| sugar | 20 | 0 (0.0%) | TIMEOUT 20 |
| tpp | 40 | 0 (0.0%) | MEMOUT 31, KILLED 9 |
| tpp-metric | 10 | 0 (0.0%) | TIMEOUT 1, UNSUPPORTED 9 |
| worksworld | 40 | 0 (0.0%) | UNSUPPORTED 40 |
| zenotravel | 23 | 0 (0.0%) | UNSUPPORTED 23 |
| ztalloc-sum | 20 | 0 (0.0%) | TIMEOUT 20 |

Full detail: [`domains.csv`](domains.csv), [`instances.csv`](instances.csv).
