# Numeric track

**`PLASPPlanner-seq`** — 228/1817 solved and validated (12.5%), 228/1389 of encodable (16.4%), across 93 domains. Median runtime on solved tasks 21.7 s, median peak memory 200.6 MB.

| status | tasks |
|---|---|
| TIMEOUT | 611 |
| UNSUPPORTED | 397 |
| KILLED | 307 |
| MEMOUT | 240 |
| SOLVED | 228 |
| ERROR | 31 |
| EXHAUSTED | 3 |

*Encodable* = attempted, minus `UNSUPPORTED` (refused on `ProblemKind`) and minus
`ERROR` (overwhelmingly the UP PDDL reader failing on the benchmark file itself).

## Per-domain coverage

| domain | tasks | PLASPPlanner-seq solved | outcomes (PLASPPlanner-seq) |
|---|---|---|---|
| 15-puzzle | 100 | 0 (0.0%) | TIMEOUT 2, MEMOUT 33, KILLED 65 |
| 2048 | 20 | 0 (0.0%) | TIMEOUT 1, MEMOUT 1, KILLED 18 |
| block-grouping | 192 | 9 (4.7%) | TIMEOUT 16, KILLED 167 |
| coins | 20 | 1 (5.0%) | TIMEOUT 17, MEMOUT 2 |
| counters | 55 | 10 (18.2%) | TIMEOUT 45 |
| delivery | 20 | 1 (5.0%) | TIMEOUT 19 |
| depots | 20 | 2 (10.0%) | TIMEOUT 16, MEMOUT 2 |
| driverlog | 20 | 0 (0.0%) | ERROR 20 |
| drone | 20 | 3 (15.0%) | TIMEOUT 17 |
| elevators | 30 | 4 (13.3%) | TIMEOUT 26 |
| expedition | 20 | 0 (0.0%) | TIMEOUT 20 |
| ext-plant-watering | 20 | 0 (0.0%) | TIMEOUT 20 |
| factory-robot | 20 | 0 (0.0%) | UNSUPPORTED 20 |
| farmland | 50 | 0 (0.0%) | TIMEOUT 50 |
| fo-counters | 20 | 3 (15.0%) | TIMEOUT 17 |
| fo-farmland | 50 | 0 (0.0%) | UNSUPPORTED 50 |
| fo-sailing | 20 | 0 (0.0%) | UNSUPPORTED 20 |
| forestfire | 20 | 5 (25.0%) | TIMEOUT 15 |
| gear-car | 20 | 0 (0.0%) | UNSUPPORTED 20 |
| hydropower | 30 | 0 (0.0%) | TIMEOUT 30 |
| line-exchange-snp | 20 | 0 (0.0%) | UNSUPPORTED 20 |
| markettrader | 20 | 0 (0.0%) | TIMEOUT 20 |
| minecraft-pogo-advanced | 20 | 17 (85.0%) | MEMOUT 2, ERROR 1 |
| minecraft-sword-advanced | 20 | 20 (100.0%) | — |
| mprime | 30 | 28 (93.3%) | TIMEOUT 2 |
| nlnp-fo-farmland | 50 | 0 (0.0%) | UNSUPPORTED 50 |
| nlnp-fo-sailing | 20 | 0 (0.0%) | UNSUPPORTED 20 |
| nlnp-hydropower | 30 | 0 (0.0%) | UNSUPPORTED 30 |
| nlnp-rover | 10 | 0 (0.0%) | UNSUPPORTED 10 |
| nlnp-settlers | 25 | 0 (0.0%) | UNSUPPORTED 25 |
| nlnp-sugar | 20 | 0 (0.0%) | UNSUPPORTED 20 |
| onlycraft-opt | 20 | 2 (10.0%) | TIMEOUT 18 |
| onlycraft-sat | 20 | 2 (10.0%) | TIMEOUT 18 |
| pancake | 50 | 5 (10.0%) | TIMEOUT 1, MEMOUT 10, KILLED 34 |
| pathwaysmetric | 30 | 1 (3.3%) | TIMEOUT 1, MEMOUT 27, KILLED 1 |
| petri-net | 20 | 0 (0.0%) | TIMEOUT 20 |
| petrobras | 70 | 1 (1.4%) | MEMOUT 69 |
| planes | 13 | 2 (15.4%) | TIMEOUT 11 |
| plant-watering | 51 | 1 (2.0%) | TIMEOUT 50 |
| plotting | 87 | 77 (88.5%) | KILLED 4, ERROR 3, EXHAUSTED 3 |
| rainbowttles-opt | 20 | 0 (0.0%) | TIMEOUT 1, MEMOUT 14, KILLED 5 |
| rainbowttles-sat | 20 | 0 (0.0%) | TIMEOUT 1, MEMOUT 17, KILLED 2 |
| rover | 20 | 4 (20.0%) | TIMEOUT 16 |
| rover-linear | 10 | 4 (40.0%) | TIMEOUT 6 |
| sailing | 40 | 0 (0.0%) | TIMEOUT 40 |
| sailing-wind-opt | 20 | 0 (0.0%) | UNSUPPORTED 20 |
| sailing-wind-sat | 20 | 0 (0.0%) | UNSUPPORTED 20 |
| satellite | 20 | 0 (0.0%) | MEMOUT 13, ERROR 7 |
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
| settlers-snp | 20 | 0 (0.0%) | TIMEOUT 20 |
| settlersnumeric | 25 | 0 (0.0%) | TIMEOUT 24, MEMOUT 1 |
| settlersnumericnoassign | 20 | 0 (0.0%) | TIMEOUT 19, MEMOUT 1 |
| sugar | 20 | 6 (30.0%) | TIMEOUT 14 |
| tpp | 40 | 0 (0.0%) | MEMOUT 29, KILLED 11 |
| tpp-metric | 10 | 0 (0.0%) | MEMOUT 1, UNSUPPORTED 9 |
| worksworld | 40 | 0 (0.0%) | UNSUPPORTED 40 |
| zenotravel | 23 | 0 (0.0%) | UNSUPPORTED 23 |
| ztalloc-sum | 20 | 0 (0.0%) | TIMEOUT 2, MEMOUT 18 |

Full detail: [`domains.csv`](domains.csv), [`instances.csv`](instances.csv).
