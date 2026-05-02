# Graph Prior Preflight Report

Created: `2026-05-01T23:51:20.385244+00:00`
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
| epochs | 20 |
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
| none | 0.0322 +/- 0.0058 (n=5) | 0.0815 +/- 0.0136 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.000000 +/- 0.000000 (n=5) | 0.0000 +/- 0.0000 (n=5) | 0.000000 +/- 0.000000 (n=5) | 0.000004 +/- 0.000001 (n=5) |
| graph | 0.0230 +/- 0.0076 (n=5) | 0.0800 +/- 0.0379 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.260635 +/- 0.112274 (n=5) | 0.0133 +/- 0.0041 (n=5) | 0.001327 +/- 0.000408 (n=5) | 0.000150 +/- 0.000028 (n=5) |
| permuted_graph | 0.0224 +/- 0.0086 (n=5) | 0.0832 +/- 0.0452 (n=5) | 0.100000 +/- 0.000000 (n=5) | 0.111906 +/- 0.004744 (n=5) | 0.230789 +/- 0.093711 (n=5) | 0.0118 +/- 0.0032 (n=5) | 0.001329 +/- 0.000404 (n=5) | 0.000145 +/- 0.000038 (n=5) |
| temporal_smooth | 0.0223 +/- 0.0099 (n=5) | 0.0796 +/- 0.0505 (n=5) | 0.100000 +/- 0.000000 (n=5) | 14263.628811 +/- 1670.886241 (n=5) | 0.000002 +/- 0.000001 (n=5) | 0.0000 +/- 0.0000 (n=5) | 0.001676 +/- 0.000556 (n=5) | 0.000262 +/- 0.000077 (n=5) |

Graph gain vs none at H=32: `+1.9%`
True-vs-permuted gain at H=32: `+4.0%`

## Stage 2: Latent Audit

| prior | D_true(Delta_H) norm | R_low true K=2 | R_low true K=4 | R_low true K=8 |
| --- | --- | --- | --- | --- |
| graph | 2.6636 +/- 0.1593 (n=5) | 0.1875 +/- 0.0127 (n=5) | 0.2866 +/- 0.0216 (n=5) | 0.4309 +/- 0.0335 (n=5) |
| permuted_graph | 3.2177 +/- 0.1312 (n=5) | 0.1407 +/- 0.0161 (n=5) | 0.2193 +/- 0.0234 (n=5) | 0.3333 +/- 0.0297 (n=5) |

Paired `D_true_norm(permuted - graph)`: `0.5541 +/- 0.0920 (n=5)`
Graph lower latent-energy count: `5/5`

## Latent-Energy Correlation

Pearson r between `D_true_norm(Delta_H)` and H=32 rollout over audited runs: `-0.2196`

## Strict Protocol Interpretation

| field | value |
| --- | --- |
| Legacy automatic classification | topology_aligned_latent_smoothing |
| Strict manuscript label | temporal_sufficient |
| Diagnostic failure mode | graph_free_temporal_smoothing_explains_gain |
| Recommended next experiment | prefer_calibrated_temporal_smoothing_unless_topology_attribution_is_goal |
| Claim boundary | Graph-free temporal smoothing explains the gain; graph structure is not necessary under this condition. |

The strict label follows the staged manuscript hierarchy and should be preferred over the legacy automatic classification when they disagree.

## Final Classification

`topology_aligned_latent_smoothing`

## Temporal Prior Interpretation

H=32 temporal_smooth rollout: `0.0796`; temporal gain vs none: `+2.3%`; graph gain vs temporal_smooth: `-0.5%`.

Interpretation: graph gain may be temporal-smoothing-like rather than topology-specific.

Decision rules:
- `no_graph_gain`: true graph mini run does not beat GNN none at H=32.
- `generic_smoothing`: true graph beats none but does not beat the permuted graph at H=32.
- `candidate_topology_specific`: true graph beats both none and permuted, but latent alignment is absent or not audited.
- `topology_aligned_latent_smoothing`: true graph beats both rollout controls and its learned `Delta_H` is smoother/more low-frequency in the true graph basis.
- `temporal_smooth`: graph-free temporal baseline; it does not change the graph classification by itself.

## Files

- `analysis_out/preflight_runs/spring_mass_5seed/standard_ep20/summary.csv`
- latent artifacts under `analysis_out/preflight_runs/spring_mass_5seed/standard_ep20/artifacts`
