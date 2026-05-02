# Graph Prior Preflight Report

Created: `2026-05-01T22:21:53.074149+00:00`
Schema version: `graph_prior_preflight_v2`

Scope: lightweight dataset-adapter preflight using the existing GNN trainer, graph prior loss, and rollout evaluator.
No ISO17, rMD17 top-up, full sweeps, or large experiments were run.

## Configuration

| field | value |
| --- | --- |
| dataset | nbody_distance |
| topology | distance_knn |
| prior_weight | 0.1 |
| seeds | 0,1,2,3,4 |
| epochs | 20 |
| train transitions | 96 |
| eval transitions | 32 |
| horizons | 16,32 |
| raw normalization | lambda_max |
| N permuted graphs | 8 |
| M random graphs | 0 |
| include temporal prior | True |
| calibrate prior strength | True |
| calibration reference prior | graph |
| calibration target ratio | 1.0 |

## Stage 0: Raw Graph-Dynamics Diagnostic

| graph type | D_dX_norm | R_low K=2 | R_low K=4 | R_low K=8 | D gap control-true |
| --- | --- | --- | --- | --- | --- |
| true_graph | 0.656891 | 0.0280 | 0.0446 | 0.1137 | 0.000000 |
| permuted_graph | 0.519343 | 0.0369 | 0.0959 | 0.2154 | -0.137548 |

Positive `D gap control-true` means raw temporal changes are smoother on the true graph than on that control.

## Stage 1: Mini Training

| prior | H=16 | H=32 | nominal lambda | effective lambda | initial prior loss | prior loss mean | effective prior contribution | final train loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 0.0554 +/- 0.0117 (n=5) | 0.1443 +/- 0.0121 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.000000 +/- 0.000000 (n=5) | 0.0000 +/- 0.0000 (n=5) | 0.000000 +/- 0.000000 (n=5) | 0.000020 +/- 0.000012 (n=5) |
| graph | 0.0245 +/- 0.0070 (n=5) | 0.0772 +/- 0.0311 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.992640 +/- 0.141206 (n=5) | 0.1341 +/- 0.0158 (n=5) | 0.013413 +/- 0.001577 (n=5) | 0.001380 +/- 0.000260 (n=5) |
| permuted_graph | 0.0257 +/- 0.0062 (n=5) | 0.0898 +/- 0.0417 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.051245 +/- 0.009564 (n=5) | 1.958140 +/- 0.207220 (n=5) | 0.2344 +/- 0.0371 (n=5) | 0.011837 +/- 0.001840 (n=5) | 0.001109 +/- 0.000174 (n=5) |
| temporal_smooth | 0.0213 +/- 0.0074 (n=5) | 0.0596 +/- 0.0265 (n=5) | 0.100000 +/- 0.000000 (n=5) | 16887.224934 +/- 2783.665993 (n=5) | 0.000006 +/- 0.000001 (n=5) | 0.0000 +/- 0.0000 (n=5) | 0.011520 +/- 0.001754 (n=5) | 0.001683 +/- 0.000379 (n=5) |

Graph gain vs none at H=32: `+46.5%`
True-vs-permuted gain at H=32: `+14.0%`

## Stage 2: Latent Audit

| prior | D_true(Delta_H) norm | R_low true K=2 | R_low true K=4 | R_low true K=8 |
| --- | --- | --- | --- | --- |
| graph | 4.7669 +/- 0.0953 (n=5) | 0.1644 +/- 0.0128 (n=5) | 0.2275 +/- 0.0152 (n=5) | 0.3304 +/- 0.0160 (n=5) |
| permuted_graph | 5.0459 +/- 0.3035 (n=5) | 0.1411 +/- 0.0247 (n=5) | 0.1928 +/- 0.0260 (n=5) | 0.2892 +/- 0.0294 (n=5) |

Paired `D_true_norm(permuted - graph)`: `0.2790 +/- 0.3593 (n=5)`
Graph lower latent-energy count: `4/5`

## Latent-Energy Correlation

Pearson r between `D_true_norm(Delta_H)` and H=32 rollout over audited runs: `-0.2683`

## Strict Protocol Interpretation

| field | value |
| --- | --- |
| Legacy automatic classification | topology_aligned_latent_smoothing |
| Strict manuscript label | temporal_sufficient |
| Diagnostic failure mode | temporal_smoothing_explains_gain_for_this_k_and_budget |
| Recommended next experiment | run_multi_seed_validation_and_soft_radial_or_inverse_distance_graph_sweep |
| Claim boundary | Applies only to this distance-kNN construction, seed, budget, and model condition. |

The strict label follows the staged manuscript hierarchy and should be preferred over the legacy automatic classification when they disagree.

## Final Classification

`topology_aligned_latent_smoothing`

## Temporal Prior Interpretation

H=32 temporal_smooth rollout: `0.0596`; temporal gain vs none: `+58.7%`; graph gain vs temporal_smooth: `-29.5%`.

Interpretation: graph gain may be temporal-smoothing-like rather than topology-specific.

Decision rules:
- `no_graph_gain`: true graph mini run does not beat GNN none at H=32.
- `generic_smoothing`: true graph beats none but does not beat the permuted graph at H=32.
- `candidate_topology_specific`: true graph beats both none and permuted, but latent alignment is absent or not audited.
- `topology_aligned_latent_smoothing`: true graph beats both rollout controls and its learned `Delta_H` is smoother/more low-frequency in the true graph basis.
- `temporal_smooth`: graph-free temporal baseline; it does not change the graph classification by itself.

## Files

- `analysis_out/preflight_runs/nbody_robustness_5seed/distance_k_12/standard_ep20/summary.csv`
- latent artifacts under `analysis_out/preflight_runs/nbody_robustness_5seed/distance_k_12/standard_ep20/artifacts`
