# Graph Prior Preflight Report

Created: `2026-05-01T23:02:30.102005+00:00`
Schema version: `graph_prior_preflight_v2`

Scope: lightweight dataset-adapter preflight using the existing GNN trainer, graph prior loss, and rollout evaluator.
No ISO17, rMD17 top-up, full sweeps, or large experiments were run.

## Configuration

| field | value |
| --- | --- |
| dataset | metr_la |
| topology |  |
| prior_weight | 0.1 |
| seeds | 0,1,2 |
| epochs | 5 |
| train transitions | 160 |
| eval transitions | 64 |
| horizons | 16,32 |
| raw normalization | lambda_max |
| N permuted graphs | 8 |
| M random graphs | 8 |
| include temporal prior | True |
| calibrate prior strength | True |
| calibration reference prior | graph |
| calibration target ratio | 1.0 |

## Stage 0: Raw Graph-Dynamics Diagnostic

| graph type | D_dX_norm | R_low K=2 | R_low K=4 | R_low K=8 | D gap control-true |
| --- | --- | --- | --- | --- | --- |
| true_graph | 0.237967 | 0.0126 | 0.0292 | 0.0557 | 0.000000 |
| permuted_graph | 0.304427 | 0.0048 | 0.0144 | 0.0333 | 0.066460 |
| random_graph | 0.473690 | 0.0041 | 0.0144 | 0.0350 | 0.235723 |

Positive `D gap control-true` means raw temporal changes are smoother on the true graph than on that control.

## Stage 1: Mini Training

| prior | H=16 | H=32 | nominal lambda | effective lambda | initial prior loss | prior loss mean | effective prior contribution | final train loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 0.1699 +/- 0.0543 (n=3) | 0.4034 +/- 0.0609 (n=3) | 0.100000 +/- 0.000000 (n=3) | 0.100000 +/- 0.000000 (n=3) | 0.000000 +/- 0.000000 (n=3) | 0.0000 +/- 0.0000 (n=3) | 0.000000 +/- 0.000000 (n=3) | 0.000183 +/- 0.000081 (n=3) |
| graph | 0.1554 +/- 0.0551 (n=3) | 0.5243 +/- 0.2845 (n=3) | 0.100000 +/- 0.000000 (n=3) | 0.100000 +/- 0.000000 (n=3) | 3.291113 +/- 1.432440 (n=3) | 0.3021 +/- 0.1333 (n=3) | 0.030214 +/- 0.013328 (n=3) | 0.005177 +/- 0.002023 (n=3) |
| permuted_graph | 0.1619 +/- 0.0534 (n=3) | 0.5026 +/- 0.2222 (n=3) | 0.100000 +/- 0.000000 (n=3) | 0.016487 +/- 0.000273 (n=3) | 19.998894 +/- 8.807434 (n=3) | 1.4870 +/- 0.6861 (n=3) | 0.024479 +/- 0.011219 (n=3) | 0.003274 +/- 0.001053 (n=3) |
| temporal_smooth | 0.1602 +/- 0.0589 (n=3) | 0.5097 +/- 0.2517 (n=3) | 0.100000 +/- 0.000000 (n=3) | 6784.887291 +/- 1235.294654 (n=3) | 0.000048 +/- 0.000020 (n=3) | 0.0000 +/- 0.0000 (n=3) | 0.103125 +/- 0.130772 (n=3) | 0.015804 +/- 0.018371 (n=3) |

Graph gain vs none at H=32: `-30.0%`
True-vs-permuted gain at H=32: `-4.3%`

## Stage 2: Latent Audit

Skipped because the mini true-graph run did not beat the permuted control at H=32.

## Strict Protocol Interpretation

| field | value |
| --- | --- |
| Legacy automatic classification | no_graph_gain |
| Strict manuscript label | no_graph_gain |
| Diagnostic failure mode | candidate_graph_construction_risk |
| Recommended next experiment | test_official_road_adjacency_road_distance_or_learned_traffic_graph |
| Claim boundary | Applies only to the tested correlation-derived graph, not all traffic graphs. |

The strict label follows the staged manuscript hierarchy and should be preferred over the legacy automatic classification when they disagree.

## Final Classification

`no_graph_gain`

## Temporal Prior Interpretation

H=32 temporal_smooth rollout: `0.5097`; temporal gain vs none: `-26.3%`; graph gain vs temporal_smooth: `-2.9%`.

Interpretation: temporal_smooth is included as a graph-free baseline; use it to separate temporal regularization from graph-specific effects.

Decision rules:
- `no_graph_gain`: true graph mini run does not beat GNN none at H=32.
- `generic_smoothing`: true graph beats none but does not beat the permuted graph at H=32.
- `candidate_topology_specific`: true graph beats both none and permuted, but latent alignment is absent or not audited.
- `topology_aligned_latent_smoothing`: true graph beats both rollout controls and its learned `Delta_H` is smoother/more low-frequency in the true graph basis.
- `temporal_smooth`: graph-free temporal baseline; it does not change the graph classification by itself.

## Files

- `analysis_out/preflight_runs/metr_la_3seed/corr_T2000_train160_ep5/summary.csv`
