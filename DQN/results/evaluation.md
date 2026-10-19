# DQN evaluation

Simulator: 1000 hands per hero per lineup, 6-max, test seeds 50000+, stacks sampled from the dataset. bb/100 with 95% confidence interval.

| hero | random | calling_station | rule_based | dataset_styles |
|---|---|---|---|---|
| random | -137.5 ± 693.6 | -731.7 ± 992.7 | -589.0 ± 242.3 | -271.0 ± 251.6 |
| calling_station | +692.2 ± 962.9 | -11.0 ± 12.9 | -1611.0 ± 311.5 | -924.4 ± 260.9 |
| rule_based | +355.1 ± 351.8 | +1761.6 ± 629.9 | +146.5 ± 126.8 | +26.5 ± 62.9 |
| dqn:scratch/online_best | +715.4 ± 557.6 | +3665.3 ± 986.4 | -270.9 ± 215.3 | +634.8 ± 254.0 |
| dqn:offline_online/online_best | +368.4 ± 403.2 | +3839.5 ± 991.5 | -42.5 ± 187.6 | +459.8 ± 241.4 |
| dqn:offline_online/offline_best | +69.3 ± 128.3 | +684.9 ± 341.4 | +34.4 ± 57.8 | +4.4 ± 29.7 |

Paired difference (DQN minus baseline, same deals), bb/100 ± 95% CI:

| DQN | vs baseline | random | calling_station | rule_based | dataset_styles |
|---|---|---|---|---|---|
| dqn:scratch/online_best | random | +853.0 ± 756.0 | +4397.1 ± 713.1 | +318.1 ± 266.3 | +905.8 ± 315.6 |
| dqn:scratch/online_best | calling_station | +23.2 ± 885.4 | +3676.3 ± 975.3 | +1340.1 ± 304.5 | +1559.2 ± 310.1 |
| dqn:scratch/online_best | rule_based | +360.3 ± 582.5 | +1903.8 ± 733.2 | -417.3 ± 234.5 | +608.3 ± 254.7 |
| dqn:offline_online/online_best | random | +505.9 ± 701.9 | +4571.3 ± 669.9 | +546.5 ± 260.5 | +730.8 ± 317.6 |
| dqn:offline_online/online_best | calling_station | -323.8 ± 954.2 | +3850.5 ± 980.2 | +1568.5 ± 292.0 | +1384.2 ± 308.6 |
| dqn:offline_online/online_best | rule_based | +13.2 ± 460.0 | +2078.0 ± 762.9 | -189.0 ± 205.9 | +433.3 ± 236.3 |
| dqn:offline_online/offline_best | random | +206.8 ± 698.7 | +1416.6 ± 917.6 | +623.4 ± 245.4 | +275.4 ± 251.8 |
| dqn:offline_online/offline_best | calling_station | -622.9 ± 957.2 | +695.9 ± 336.5 | +1645.4 ± 306.1 | +928.8 ± 259.8 |
| dqn:offline_online/offline_best | rule_based | -285.8 ± 322.2 | -1076.6 ± 486.7 | -112.1 ± 123.8 | -22.2 ± 64.3 |

Hero action distribution and VPIP / PFR (dataset_styles lineup):

| hero | VPIP | PFR | fold | check | call | raise_half | raise_pot | all_in |
|---|---|---|---|---|---|---|---|---|
| random | 0.76 | 0.63 | 0.18 | 0.04 | 0.17 | 0.20 | 0.17 | 0.24 |
| calling_station | 0.94 | 0.00 | 0.00 | 0.43 | 0.57 | 0.00 | 0.00 | 0.00 |
| rule_based | 0.30 | 0.12 | 0.58 | 0.13 | 0.16 | 0.03 | 0.11 | 0.00 |
| dqn:scratch/online_best | 0.93 | 0.58 | 0.07 | 0.17 | 0.30 | 0.19 | 0.13 | 0.13 |
| dqn:offline_online/online_best | 0.94 | 0.62 | 0.06 | 0.12 | 0.26 | 0.23 | 0.24 | 0.08 |
| dqn:offline_online/offline_best | 0.14 | 0.09 | 0.74 | 0.09 | 0.08 | 0.02 | 0.09 | 0.00 |

Dataset test split (Q-function diagnostics, not winnings):

| checkpoint | TD loss | agree w/ logged | corr(Q, return) | mean Q logged | mean return |
|---|---|---|---|---|---|
| dqn:scratch/online_best | 0.2241 | 0.156 | 0.204 | +3.01 | -0.32 |
| dqn:offline_online/online_best | 0.2151 | 0.171 | 0.177 | +2.67 | -0.32 |
| dqn:offline_online/offline_best | 0.2458 | 0.779 | 0.103 | +0.59 | -0.32 |
