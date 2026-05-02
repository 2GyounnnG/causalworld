# Graph Prior Preflight Report

Created: `2026-05-01T21:44:33.843212+00:00`
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
| epochs | 5 |
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
| none | 0.3611 +/- 0.1433 (n=5) | 1.1671 +/- 0.7792 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.000000 +/- 0.000000 (n=5) | 0.0000 +/- 0.0000 (n=5) | 0.000000 +/- 0.000000 (n=5) | 0.000459 +/- 0.000241 (n=5) |
| graph | 0.2520 +/- 0.1206 (n=5) | 0.9365 +/- 0.6715 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.992640 +/- 0.141206 (n=5) | 0.4319 +/- 0.0545 (n=5) | 0.043187 +/- 0.005449 (n=5) | 0.012404 +/- 0.001586 (n=5) |
| permuted_graph | 0.2493 +/- 0.1156 (n=5) | 0.9489 +/- 0.7035 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.051245 +/- 0.009564 (n=5) | 1.958140 +/- 0.207220 (n=5) | 0.7653 +/- 0.1360 (n=5) | 0.038589 +/- 0.006248 (n=5) | 0.010874 +/- 0.001327 (n=5) |
| temporal_smooth | 0.2574 +/- 0.1298 (n=5) | 1.0144 +/- 0.7576 (n=5) | 0.100000 +/- 0.000000 (n=5) | 16887.223505 +/- 2783.669251 (n=5) | 0.000006 +/- 0.000001 (n=5) | 0.0000 +/- 0.0000 (n=5) | 0.033843 +/- 0.004599 (n=5) | 0.012635 +/- 0.002536 (n=5) |

Graph gain vs none at H=32: `+19.8%`
True-vs-permuted gain at H=32: `+1.3%`

## Stage 2: Latent Audit

| prior | D_true(Delta_H) norm | R_low true K=2 | R_low true K=4 | R_low true K=8 |
| --- | --- | --- | --- | --- |
| graph | 5.1146 +/- 0.1772 (n=5) | 0.1332 +/- 0.0209 (n=5) | 0.1886 +/- 0.0197 (n=5) | 0.2885 +/- 0.0222 (n=5) |
| permuted_graph | 5.2659 +/- 0.2629 (n=5) | 0.1267 +/- 0.0268 (n=5) | 0.1763 +/- 0.0298 (n=5) | 0.2708 +/- 0.0315 (n=5) |

Paired `D_true_norm(permuted - graph)`: `0.1513 +/- 0.2248 (n=5)`
Graph lower latent-energy count: `4/5`

## Latent-Energy Correlation

Pearson r between `D_true_norm(Delta_H)` and H=32 rollout over audited runs: `0.6944`

## Strict Protocol Interpretation

| field | value |
| --- | --- |
| Legacy automatic classification | topology_aligned_latent_smoothing |
| Strict manuscript label | quick_topology_signal |
| Diagnostic failure mode | quick_signal_requires_standard_and_graph_construction_validation |
| Recommended next experiment | run_multi_seed_validation_and_soft_radial_or_inverse_distance_graph_sweep |
| Claim boundary | Applies only to this distance-kNN construction, seed, budget, and model condition. |

The strict label follows the staged manuscript hierarchy and should be preferred over the legacy automatic classification when they disagree.

## Final Classification

`topology_aligned_latent_smoothing`

## Temporal Prior Interpretation

H=32 temporal_smooth rollout: `1.0144`; temporal gain vs none: `+13.1%`; graph gain vs temporal_smooth: `+7.7%`.

Interpretation: graph beats temporal_smooth and permuted_graph, which is stronger evidence for topology-specific graph usefulness.

Decision rules:
- `no_graph_gain`: true graph mini run does not beat GNN none at H=32.
- `generic_smoothing`: true graph beats none but does not beat the permuted graph at H=32.
- `candidate_topology_specific`: true graph beats both none and permuted, but latent alignment is absent or not audited.
- `topology_aligned_latent_smoothing`: true graph beats both rollout controls and its learned `Delta_H` is smoother/more low-frequency in the true graph basis.
- `temporal_smooth`: graph-free temporal baseline; it does not change the graph classification by itself.

## Files

- `analysis_out/preflight_runs/nbody_robustness_5seed/distance_k_12/quick_ep5/summary.csv`
- latent artifacts under `analysis_out/preflight_runs/nbody_robustness_5seed/distance_k_12/quick_ep5/artifacts`
