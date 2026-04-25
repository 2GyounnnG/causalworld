# Weight Sweep Instability

Instability is flagged when `H16/H1 > 20` or `H16 > 1.0`.

The current weight sweep is complete when aggregate rows contain n=3 for every prior/weight group at H=16.

Completeness: COMPLETE.

## Aggregate H16/H1

| prior | prior_weight | h1 | h16 | h16_over_h1 | unstable | reasons |
| --- | --- | --- | --- | --- | --- | --- |
| euclidean | 0.001 | 0.07933458684 | 0.1809946463 | 2.281409074 | false |  |
| euclidean | 0.01 | 0.1045917055 | 0.1725369454 | 1.649623597 | false |  |
| euclidean | 0.1 | 0.1592057673 | 0.2382917923 | 1.496753518 | false |  |
| euclidean | 1.0 | 0.2104365346 | 0.3215721937 | 1.528119603 | false |  |
| none | 0.001 | 0.1324307558 | 0.4171957141 | 3.150293235 | false |  |
| none | 0.01 | 0.1324307558 | 0.4171957141 | 3.150293235 | false |  |
| none | 0.1 | 0.1324307558 | 0.4171957141 | 3.150293235 | false |  |
| none | 1.0 | 0.1324307558 | 0.4171957141 | 3.150293235 | false |  |
| spectral | 0.001 | 0.06662473029 | 0.2599633544 | 3.901904792 | false |  |
| spectral | 0.01 | 0.04199693666 | 1.908429248 | 45.44210601 | true | H16/H1 > 20; H16 > 1 |
| spectral | 0.1 | 0.03484405912 | 0.1300585177 | 3.732588022 | false |  |
| spectral | 1.0 | 0.03482212888 | 0.1443177964 | 4.144427726 | false |  |

## Best H=16 Settings

| prior | best_weight | h16_mean | h16_std | n |
| --- | --- | --- | --- | --- |
| none | 0.001 | 0.4171957141 | 0.2430955445 | 3 |
| euclidean | 0.01 | 0.1725369454 | 0.05998918753 | 3 |
| spectral | 0.1 | 0.1300585177 | 0.09771104769 | 3 |

Best spectral improves over best Euclidean by 24.61990248%.
Euclidean regularization improves over no-prior baseline when tuned.
Spectral w=0.01 is a long-horizon instability regime, not a uniformly bad spectral result.
best_spectral_vs_best_euclidean_pct: 24.61990248
best_spectral_vs_none_pct: 68.8255384

## Seed-Level Instability Flags

| prior | prior_weight | seed | h1 | h16 | h16_over_h1 | reasons |
| --- | --- | --- | --- | --- | --- | --- |
| spectral | 0.01 | 1 | 0.04773703327 | 3.274033636 | 68.58477396 | H16/H1 > 20; H16 > 1 |
| spectral | 0.01 | 2 | 0.04517014669 | 2.338572881 | 51.77253236 | H16/H1 > 20; H16 > 1 |

## Spectral w=0.01 Long-Horizon Instability

Spectral `w=0.01` seed `1` and seed `2` are long-horizon unstable in the current sweep.

| prior | prior_weight | seed | h1 | h16 | h16_over_h1 | unstable | reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| spectral | 0.01 | 1 | 0.04773703327 | 3.274033636 | 68.58477396 | true | H16/H1 > 20; H16 > 1 |
| spectral | 0.01 | 2 | 0.04517014669 | 2.338572881 | 51.77253236 | true | H16/H1 > 20; H16 > 1 |

## Notes

- This script reads existing analysis artifacts only.
- Training code and raw result JSONs are not modified.
