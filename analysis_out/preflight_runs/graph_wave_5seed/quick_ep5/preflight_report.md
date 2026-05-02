# Graph Prior Preflight Report

Created: `2026-05-01T23:54:22.170096+00:00`
Schema version: `graph_prior_preflight_v2`

Scope: lightweight dataset-adapter preflight using the existing GNN trainer, graph prior loss, and rollout evaluator.
No ISO17, rMD17 top-up, full sweeps, or large experiments were run.

## Configuration

| field | value |
| --- | --- |
| dataset | graph_wave_lattice |
| topology | lattice |
| prior_weight | 0.1 |
| seeds | 0,1,2,3,4 |
| epochs | 5 |
| train transitions | 96 |
| eval transitions | 32 |
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
| true_graph | 0.577167 | 0.0001 | 0.0018 | 0.0089 | 0.000000 |
| permuted_graph | 0.464986 | 0.0249 | 0.0754 | 0.1955 | -0.112181 |
| random_graph | 0.387498 | 0.0352 | 0.1029 | 0.2056 | -0.189669 |

Positive `D gap control-true` means raw temporal changes are smoother on the true graph than on that control.

## Stage 1: Mini Training

| prior | H=16 | H=32 | nominal lambda | effective lambda | initial prior loss | prior loss mean | effective prior contribution | final train loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 0.3156 +/- 0.1280 (n=5) | 1.1412 +/- 0.6357 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.000000 +/- 0.000000 (n=5) | 0.0000 +/- 0.0000 (n=5) | 0.000000 +/- 0.000000 (n=5) | 0.000367 +/- 0.000208 (n=5) |
| graph | 0.2335 +/- 0.1118 (n=5) | 0.8485 +/- 0.6926 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.274533 +/- 0.118875 (n=5) | 0.0536 +/- 0.0198 (n=5) | 0.005362 +/- 0.001984 (n=5) | 0.002019 +/- 0.000489 (n=5) |
| permuted_graph | 0.2349 +/- 0.1085 (n=5) | 0.8457 +/- 0.6742 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.111768 +/- 0.004648 (n=5) | 0.243326 +/- 0.099071 (n=5) | 0.0479 +/- 0.0150 (n=5) | 0.005402 +/- 0.001858 (n=5) | 0.002073 +/- 0.000497 (n=5) |
| temporal_smooth | 0.2305 +/- 0.1084 (n=5) | 0.8296 +/- 0.6859 (n=5) | 0.100000 +/- 0.000000 (n=5) | 14181.936992 +/- 1653.979164 (n=5) | 0.000002 +/- 0.000001 (n=5) | 0.0000 +/- 0.0000 (n=5) | 0.006353 +/- 0.002300 (n=5) | 0.002777 +/- 0.000843 (n=5) |

Graph gain vs none at H=32: `+25.6%`
True-vs-permuted gain at H=32: `-0.3%`

## Stage 2: Latent Audit

Skipped because the mini true-graph run did not beat the permuted control at H=32.

## Strict Protocol Interpretation

| field | value |
| --- | --- |
| Legacy automatic classification | generic_smoothing |
| Strict manuscript label | generic_smoothing |
| Diagnostic failure mode | topology_not_isolated_from_spectral_smoothing |
| Recommended next experiment | test_simpler_smoothing_controls_or_alternative_graph_construction |
| Claim boundary | Graph-style regularization helps, but topology-specific attribution is not isolated. |

The strict label follows the staged manuscript hierarchy and should be preferred over the legacy automatic classification when they disagree.

## Final Classification

`generic_smoothing`

## Temporal Prior Interpretation

H=32 temporal_smooth rollout: `0.8296`; temporal gain vs none: `+27.3%`; graph gain vs temporal_smooth: `-2.3%`.

Interpretation: graph gain may be temporal-smoothing-like rather than topology-specific.

Decision rules:
- `no_graph_gain`: true graph mini run does not beat GNN none at H=32.
- `generic_smoothing`: true graph beats none but does not beat the permuted graph at H=32.
- `candidate_topology_specific`: true graph beats both none and permuted, but latent alignment is absent or not audited.
- `topology_aligned_latent_smoothing`: true graph beats both rollout controls and its learned `Delta_H` is smoother/more low-frequency in the true graph basis.
- `temporal_smooth`: graph-free temporal baseline; it does not change the graph classification by itself.

## Files

- `analysis_out/preflight_runs/graph_wave_5seed/quick_ep5/summary.csv`
