# Graph Prior Preflight Report

Created: `2026-05-01T23:40:18.546640+00:00`
Schema version: `graph_prior_preflight_v2`

Scope: lightweight dataset-adapter preflight using the existing GNN trainer, graph prior loss, and rollout evaluator.
No ISO17, rMD17 top-up, full sweeps, or large experiments were run.

## Configuration

| field | value |
| --- | --- |
| dataset | spring_mass_lattice |
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
| true_graph | 0.576065 | 0.0001 | 0.0017 | 0.0085 | 0.000000 |
| permuted_graph | 0.466370 | 0.0251 | 0.0757 | 0.1957 | -0.109696 |
| random_graph | 0.385723 | 0.0351 | 0.1029 | 0.2054 | -0.190343 |

Positive `D gap control-true` means raw temporal changes are smoother on the true graph than on that control.

## Stage 1: Mini Training

| prior | H=16 | H=32 | nominal lambda | effective lambda | initial prior loss | prior loss mean | effective prior contribution | final train loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 0.3117 +/- 0.1263 (n=5) | 1.1284 +/- 0.6290 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.000000 +/- 0.000000 (n=5) | 0.0000 +/- 0.0000 (n=5) | 0.000000 +/- 0.000000 (n=5) | 0.000360 +/- 0.000209 (n=5) |
| graph | 0.2347 +/- 0.1099 (n=5) | 0.8536 +/- 0.6610 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.260635 +/- 0.112274 (n=5) | 0.0410 +/- 0.0145 (n=5) | 0.004105 +/- 0.001454 (n=5) | 0.001641 +/- 0.000372 (n=5) |
| permuted_graph | 0.2349 +/- 0.1087 (n=5) | 0.8490 +/- 0.6498 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.111906 +/- 0.004744 (n=5) | 0.230789 +/- 0.093711 (n=5) | 0.0366 +/- 0.0112 (n=5) | 0.004129 +/- 0.001391 (n=5) | 0.001690 +/- 0.000377 (n=5) |
| temporal_smooth | 0.2343 +/- 0.1075 (n=5) | 0.8425 +/- 0.6369 (n=5) | 0.100000 +/- 0.000000 (n=5) | 14263.628153 +/- 1670.886777 (n=5) | 0.000002 +/- 0.000001 (n=5) | 0.0000 +/- 0.0000 (n=5) | 0.004761 +/- 0.001699 (n=5) | 0.002227 +/- 0.000553 (n=5) |

Graph gain vs none at H=32: `+24.4%`
True-vs-permuted gain at H=32: `-0.5%`

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

H=32 temporal_smooth rollout: `0.8425`; temporal gain vs none: `+25.3%`; graph gain vs temporal_smooth: `-1.3%`.

Interpretation: graph gain may be temporal-smoothing-like rather than topology-specific.

Decision rules:
- `no_graph_gain`: true graph mini run does not beat GNN none at H=32.
- `generic_smoothing`: true graph beats none but does not beat the permuted graph at H=32.
- `candidate_topology_specific`: true graph beats both none and permuted, but latent alignment is absent or not audited.
- `topology_aligned_latent_smoothing`: true graph beats both rollout controls and its learned `Delta_H` is smoother/more low-frequency in the true graph basis.
- `temporal_smooth`: graph-free temporal baseline; it does not change the graph classification by itself.

## Files

- `analysis_out/preflight_runs/spring_mass_5seed/quick_ep5/summary.csv`
