# Graph Prior Preflight Report

Created: `2026-05-01T21:34:01.906403+00:00`
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
| true_graph | 0.584279 | 0.0280 | 0.0456 | 0.1178 | 0.000000 |
| permuted_graph | 0.462816 | 0.0369 | 0.0959 | 0.2153 | -0.121463 |

Positive `D gap control-true` means raw temporal changes are smoother on the true graph than on that control.

## Stage 1: Mini Training

| prior | H=16 | H=32 | nominal lambda | effective lambda | initial prior loss | prior loss mean | effective prior contribution | final train loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 0.0567 +/- 0.0104 (n=5) | 0.1522 +/- 0.0115 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.000000 +/- 0.000000 (n=5) | 0.0000 +/- 0.0000 (n=5) | 0.000000 +/- 0.000000 (n=5) | 0.000021 +/- 0.000011 (n=5) |
| graph | 0.0256 +/- 0.0076 (n=5) | 0.0814 +/- 0.0332 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.674187 +/- 0.087288 (n=5) | 0.0963 +/- 0.0136 (n=5) | 0.009632 +/- 0.001363 (n=5) | 0.000957 +/- 0.000220 (n=5) |
| permuted_graph | 0.0246 +/- 0.0068 (n=5) | 0.0759 +/- 0.0276 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.044348 +/- 0.009334 (n=5) | 1.549142 +/- 0.211866 (n=5) | 0.1862 +/- 0.0318 (n=5) | 0.008143 +/- 0.001618 (n=5) | 0.000751 +/- 0.000106 (n=5) |
| temporal_smooth | 0.0242 +/- 0.0074 (n=5) | 0.0695 +/- 0.0240 (n=5) | 0.100000 +/- 0.000000 (n=5) | 10466.660402 +/- 1626.318049 (n=5) | 0.000007 +/- 0.000001 (n=5) | 0.0000 +/- 0.0000 (n=5) | 0.007964 +/- 0.001391 (n=5) | 0.001137 +/- 0.000321 (n=5) |

Graph gain vs none at H=32: `+46.5%`
True-vs-permuted gain at H=32: `-7.3%`

## Stage 2: Latent Audit

Skipped because the mini true-graph run did not beat the permuted control at H=32.

## Strict Protocol Interpretation

| field | value |
| --- | --- |
| Legacy automatic classification | generic_smoothing |
| Strict manuscript label | generic_smoothing |
| Diagnostic failure mode | distance_knn_topology_not_isolated_from_spectral_control |
| Recommended next experiment | run_multi_seed_validation_and_soft_radial_or_inverse_distance_graph_sweep |
| Claim boundary | Applies only to this distance-kNN construction, seed, budget, and model condition. |

The strict label follows the staged manuscript hierarchy and should be preferred over the legacy automatic classification when they disagree.

## Final Classification

`generic_smoothing`

## Temporal Prior Interpretation

H=32 temporal_smooth rollout: `0.0695`; temporal gain vs none: `+54.4%`; graph gain vs temporal_smooth: `-17.3%`.

Interpretation: graph gain may be temporal-smoothing-like rather than topology-specific.

Decision rules:
- `no_graph_gain`: true graph mini run does not beat GNN none at H=32.
- `generic_smoothing`: true graph beats none but does not beat the permuted graph at H=32.
- `candidate_topology_specific`: true graph beats both none and permuted, but latent alignment is absent or not audited.
- `topology_aligned_latent_smoothing`: true graph beats both rollout controls and its learned `Delta_H` is smoother/more low-frequency in the true graph basis.
- `temporal_smooth`: graph-free temporal baseline; it does not change the graph classification by itself.

## Files

- `analysis_out/preflight_runs/nbody_robustness_5seed/distance_k_08/standard_ep20/summary.csv`
