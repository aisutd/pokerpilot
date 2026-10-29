# Training run `adaptive_noanchor`

Updated 2026-10-26 18:36. Status: **in progress or stopped early**.

## Settings

| setting | value |
|---|---|
| stage | online |
| init | runs\offline_online\offline_best.pt |
| curriculum | tables |
| select | worst |
| dataset_mix | 0.0 |
| anchor_alpha | 0.5 |
| hands | 150000 |
| lr | 0.0001 |
| eps_start | 0.2 |
| eps_end | 0.05 |
| offline_epochs | 40 |
| cql_alpha | 1.0 |
| seed | 0 |
| hidden | [256, 256] |
| reward | net |

## Online training (simulator)

Recent bb/100 includes exploration, so it is noisier than validation.

| hands | epsilon | loss | mean Q (bb) | recent bb/100 | fold | call | raise | all-in | time |
|---|---|---|---|---|---|---|---|---|---|
| 5000 | 0.188 | 1.338 | 4.30 | 502.9 | 39% | 10% | 24% | 6% | 1 min |
| 10000 | 0.175 | 2.025 | 8.91 | 1356.4 | 13% | 25% | 26% | 10% | 3 min |
| 15000 | 0.163 | 1.769 | 8.19 | 471.2 | 19% | 29% | 28% | 9% | 6 min |
| 20000 | 0.150 | 1.506 | 7.34 | 252.8 | 15% | 29% | 32% | 10% | 10 min |
| 25000 | 0.138 | 1.497 | 6.75 | 440.6 | 16% | 28% | 30% | 9% | 12 min |
| 30000 | 0.125 | 1.474 | 6.08 | 501.4 | 19% | 30% | 26% | 9% | 17 min |
| 35000 | 0.113 | 1.516 | 7.22 | 695.9 | 14% | 27% | 27% | 10% | 19 min |
| 40000 | 0.100 | 1.436 | 6.55 | 78.3 | 13% | 32% | 25% | 10% | 21 min |
| 45000 | 0.087 | 1.532 | 7.95 | 1025.8 | 13% | 25% | 27% | 9% | 23 min |
| 50000 | 0.075 | 1.566 | 7.78 | 271.6 | 16% | 28% | 26% | 8% | 26 min |
| 55000 | 0.062 | 1.497 | 7.75 | 233.5 | 18% | 30% | 25% | 9% | 31 min |
| 60000 | 0.050 | 1.446 | 7.57 | 197.1 | 18% | 32% | 24% | 9% | 33 min |
| 65000 | 0.050 | 1.470 | 7.76 | 517.6 | 17% | 29% | 24% | 8% | 173 min |
| 70000 | 0.050 | 1.439 | 8.16 | 473.8 | 16% | 33% | 25% | 9% | 178 min |
| 75000 | 0.050 | 1.470 | 8.22 | 791.9 | 12% | 30% | 25% | 8% | 182 min |
| 80000 | 0.050 | 1.431 | 7.60 | 467.3 | 17% | 33% | 23% | 8% | 191 min |
| 85000 | 0.050 | 1.367 | 7.09 | 100.7 | 14% | 32% | 25% | 8% | 196 min |
| 90000 | 0.050 | 1.325 | 7.27 | 1098.0 | 17% | 29% | 23% | 8% | 200 min |

## Validation checkpoints

bb/100 on validation deals (2000 hands per table), ± 95% CI. `online_best.pt` = the checkpoint at **50000** hands (best worst score).

| hands | random | calling_station | rule_based | dataset_styles | mean | worst |
|---|---|---|---|---|---|---|
| 25000 | +480.2 ± 290.6 | +4170.0 ± 661.4 | +88.6 ± 121.3 | +298.7 ± 145.0 | +1259.4 | +88.6 |
| 50000 | +334.0 ± 291.0 | +4480.3 ± 649.7 | +208.5 ± 142.7 | +305.3 ± 161.9 | +1332.0 | +208.5 |
| 75000 | +554.0 ± 312.7 | +4341.7 ± 664.7 | +13.0 ± 159.0 | +285.5 ± 182.0 | +1298.6 | +13.0 |

Hands played (VPIP) / raised preflop (PFR) at each table:

| hands | random | calling_station | rule_based | dataset_styles |
|---|---|---|---|---|
| 25000 | 48% / 16% | 84% / 51% | 89% / 41% | 89% / 45% |
| 50000 | 46% / 26% | 86% / 53% | 93% / 30% | 93% / 28% |
| 75000 | 48% / 30% | 90% / 62% | 93% / 54% | 93% / 60% |
