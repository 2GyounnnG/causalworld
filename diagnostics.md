# Phase-3 Pilot Diagnostics

## Causal Graph Size

| step_in_episode | mean_nodes | mean_edges | mean_H1_error_of_spectral_prior |
| --- | --- | --- | --- |
| 1 | 1.0000 | 0.0000 | 0.0264 |
| 2 | 2.0000 | 0.1250 | 0.0257 |
| 4 | 4.0000 | 0.6000 | 0.0302 |
| 8 | 8.0000 | 2.3850 | 0.0216 |
| 16 | 16.0000 | 8.3450 | 0.0273 |

Mean causal graph nodes at step 16 is at least 10, so this audit does not by itself indicate an identity-like Laplacian.

## Weight Sensitivity

| prior | weight | H16_mean | H16_std | seed_values |
| --- | --- | --- | --- | --- |
| euclidean | 0.01 | 0.3557 | 0.0548 | 0.3512, 0.2909, 0.4248 |
| euclidean | 0.03 | 0.4576 | 0.0056 | 0.4497, 0.4626, 0.4605 |
| euclidean | 0.1 | 0.5744 | 0.1766 | 0.5804, 0.7876, 0.3551 |
| euclidean | 0.3 | 0.6947 | 0.1685 | 0.4755, 0.8852, 0.7235 |
| euclidean | 1.0 | 0.8306 | 0.1838 | 0.5820, 0.8889, 1.0209 |
| spectral | 0.01 | 0.3376 | 0.0892 | 0.2599, 0.2903, 0.4626 |
| spectral | 0.03 | 0.4227 | 0.2324 | 0.6821, 0.1183, 0.4677 |
| spectral | 0.1 | 0.2811 | 0.0228 | 0.2515, 0.2849, 0.3069 |
| spectral | 0.3 | 0.3036 | 0.0658 | 0.2165, 0.3754, 0.3190 |
| spectral | 1.0 | 0.2922 | 0.1294 | 0.4563, 0.2805, 0.1399 |

Spectral beats euclidean at 5/5 tested weights. Plot: `weight_sweep.png`.

## Encoder Capacity

| hidden_dim | H16_mean | H16_std | seed_values |
| --- | --- | --- | --- |
| 32 | 0.9192 | 0.3518 | 0.7794, 0.5755, 1.4026 |
| 64 | 1.4624 | 0.5363 | 0.8095, 2.1231, 1.4546 |
| 128 | 3.0431 | 1.6891 | 1.4449, 2.3047, 5.3797 |

H=16 error does not drop monotonically with hidden_dim, so this diagnostic does not support a simple capacity-limited explanation.

## Seed Stability

| metric | value |
| --- | --- |
| H16_mean | 0.3281 |
| H16_std | 0.1142 |
| H16_min | 0.2060 |
| H16_max | 0.5710 |
| std_over_mean | 0.3479 |

Seed H=16 values: 0.2599, 0.2903, 0.4626, 0.2437, 0.4170, 0.2190, 0.5710, 0.3521, 0.2593, 0.2060

**Flag:** std/mean is above 0.3, so 3-seed averaging is insufficient for the main grid. Rerun the main grid with 10 seeds before treating effect sizes as stable.

## Can I Trust The Phase-3 Pilot Result?

The Phase-3 pilot remains suggestive but should not yet be treated as settled evidence because 10-seed variability is too high for 3-seed claims.
