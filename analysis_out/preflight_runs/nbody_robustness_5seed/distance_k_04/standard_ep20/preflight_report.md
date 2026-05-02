# Graph Prior Preflight Report

Created: `2026-05-01T20:59:26.481155+00:00`
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
| true_graph | 0.467707 | 0.0023 | 0.0556 | 0.1251 | 0.000000 |
| permuted_graph | 0.370268 | 0.0280 | 0.0949 | 0.2162 | -0.097439 |

Positive `D gap control-true` means raw temporal changes are smoother on the true graph than on that control.

## Stage 1: Mini Training

| prior | H=16 | H=32 | nominal lambda | effective lambda | initial prior loss | prior loss mean | effective prior contribution | final train loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 0.0547 +/- 0.0085 (n=5) | 0.1507 +/- 0.0129 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.000000 +/- 0.000000 (n=5) | 0.0000 +/- 0.0000 (n=5) | 0.000000 +/- 0.000000 (n=5) | 0.000019 +/- 0.000010 (n=5) |
| graph | 0.0218 +/- 0.0103 (n=5) | 0.0641 +/- 0.0322 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.291952 +/- 0.044293 (n=5) | 0.0495 +/- 0.0081 (n=5) | 0.004954 +/- 0.000806 (n=5) | 0.000537 +/- 0.000141 (n=5) |
| permuted_graph | 0.0242 +/- 0.0069 (n=5) | 0.0718 +/- 0.0245 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.028981 +/- 0.008324 (n=5) | 1.048783 +/- 0.204297 (n=5) | 0.1222 +/- 0.0185 (n=5) | 0.003507 +/- 0.001013 (n=5) | 0.000325 +/- 0.000104 (n=5) |
| temporal_smooth | 0.0217 +/- 0.0063 (n=5) | 0.0661 +/- 0.0147 (n=5) | 0.100000 +/- 0.000000 (n=5) | 3643.829643 +/- 559.198020 (n=5) | 0.000008 +/- 0.000002 (n=5) | 0.0000 +/- 0.0000 (n=5) | 0.003625 +/- 0.000697 (n=5) | 0.000527 +/- 0.000145 (n=5) |

Graph gain vs none at H=32: `+57.4%`
True-vs-permuted gain at H=32: `+10.6%`

## Stage 2: Latent Audit

| prior | D_true(Delta_H) norm | R_low true K=2 | R_low true K=4 | R_low true K=8 |
| --- | --- | --- | --- | --- |
| graph | 1.4222 +/- 0.0878 (n=5) | 0.1084 +/- 0.0086 (n=5) | 0.1692 +/- 0.0065 (n=5) | 0.3458 +/- 0.0202 (n=5) |
| permuted_graph | 1.6909 +/- 0.0476 (n=5) | 0.0839 +/- 0.0058 (n=5) | 0.1431 +/- 0.0083 (n=5) | 0.2833 +/- 0.0169 (n=5) |

Paired `D_true_norm(permuted - graph)`: `0.2688 +/- 0.0860 (n=5)`
Graph lower latent-energy count: `5/5`

## Latent-Energy Correlation

Pearson r between `D_true_norm(Delta_H)` and H=32 rollout over audited runs: `0.3533`

## Strict Protocol Interpretation

| field | value |
| --- | --- |
| Legacy automatic classification | topology_aligned_latent_smoothing |
| Strict manuscript label | candidate_topology_specific_candidate |
| Diagnostic failure mode | candidate_graph_favorable_for_this_k_and_budget_only |
| Recommended next experiment | run_multi_seed_validation_and_soft_radial_or_inverse_distance_graph_sweep |
| Claim boundary | Applies only to this distance-kNN construction, seed, budget, and model condition. |

The strict label follows the staged manuscript hierarchy and should be preferred over the legacy automatic classification when they disagree.

## Final Classification

`topology_aligned_latent_smoothing`

## Temporal Prior Interpretation

H=32 temporal_smooth rollout: `0.0661`; temporal gain vs none: `+56.1%`; graph gain vs temporal_smooth: `+3.0%`.

Interpretation: graph beats temporal_smooth and permuted_graph, which is stronger evidence for topology-specific graph usefulness.

Decision rules:
- `no_graph_gain`: true graph mini run does not beat GNN none at H=32.
- `generic_smoothing`: true graph beats none but does not beat the permuted graph at H=32.
- `candidate_topology_specific`: true graph beats both none and permuted, but latent alignment is absent or not audited.
- `topology_aligned_latent_smoothing`: true graph beats both rollout controls and its learned `Delta_H` is smoother/more low-frequency in the true graph basis.
- `temporal_smooth`: graph-free temporal baseline; it does not change the graph classification by itself.

## Files

- `analysis_out/preflight_runs/nbody_robustness_5seed/distance_k_04/standard_ep20/summary.csv`
- latent artifacts under `analysis_out/preflight_runs/nbody_robustness_5seed/distance_k_04/standard_ep20/artifacts`
