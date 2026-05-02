# Graph Prior Preflight Report

Created: `2026-05-02T00:07:22.948970+00:00`
Schema version: `graph_prior_preflight_v2`

Scope: lightweight dataset-adapter preflight using the existing GNN trainer, graph prior loss, and rollout evaluator.
No ISO17, rMD17 top-up, full sweeps, or large experiments were run.

## Configuration

| field | value |
| --- | --- |
| dataset | graph_heat_lattice |
| topology | lattice |
| prior_weight | 0.1 |
| seeds | 0,1,2 |
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
| true_graph | 0.614624 | 0.0312 | 0.0877 | 0.1972 | 0.000000 |
| permuted_graph | 0.463774 | 0.0280 | 0.0856 | 0.1977 | -0.150850 |
| random_graph | 0.383011 | 0.0387 | 0.0857 | 0.2022 | -0.231613 |

Positive `D gap control-true` means raw temporal changes are smoother on the true graph than on that control.

## Stage 1: Mini Training

| prior | H=16 | H=32 | nominal lambda | effective lambda | initial prior loss | prior loss mean | effective prior contribution | final train loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 0.2559 +/- 0.1253 (n=3) | 0.8578 +/- 0.5826 (n=3) | 0.100000 +/- 0.000000 (n=3) | 0.100000 +/- 0.000000 (n=3) | 0.000000 +/- 0.000000 (n=3) | 0.0000 +/- 0.0000 (n=3) | 0.000000 +/- 0.000000 (n=3) | 0.000273 +/- 0.000122 (n=3) |
| graph | 0.2303 +/- 0.1024 (n=3) | 0.7948 +/- 0.5109 (n=3) | 0.100000 +/- 0.000000 (n=3) | 0.100000 +/- 0.000000 (n=3) | 0.204779 +/- 0.040072 (n=3) | 0.0158 +/- 0.0018 (n=3) | 0.001582 +/- 0.000182 (n=3) | 0.001055 +/- 0.000076 (n=3) |
| permuted_graph | 0.2415 +/- 0.1151 (n=3) | 0.8359 +/- 0.5625 (n=3) | 0.100000 +/- 0.000000 (n=3) | 0.035372 +/- 0.001062 (n=3) | 0.576973 +/- 0.097637 (n=3) | 0.0368 +/- 0.0062 (n=3) | 0.001306 +/- 0.000256 (n=3) | 0.000899 +/- 0.000029 (n=3) |
| temporal_smooth | 0.2280 +/- 0.0939 (n=3) | 0.7912 +/- 0.4778 (n=3) | 0.100000 +/- 0.000000 (n=3) | 2947.126698 +/- 600.440515 (n=3) | 0.000007 +/- 0.000002 (n=3) | 0.0000 +/- 0.0000 (n=3) | 0.002086 +/- 0.000387 (n=3) | 0.001325 +/- 0.000095 (n=3) |

Graph gain vs none at H=32: `+7.3%`
True-vs-permuted gain at H=32: `+4.9%`

## Graph Heat Oracle Baseline

| metric | value |
| --- | --- |
| one-step MSE | 0.00009918 |
| H=16 rollout MSE | 0.00046113 |
| H=32 rollout MSE | 0.00054215 |

## Stage 2: Latent Audit

| prior | D_true(Delta_H) norm | R_low true K=2 | R_low true K=4 | R_low true K=8 |
| --- | --- | --- | --- | --- |
| graph | 2.6312 +/- 0.4652 (n=3) | 0.1890 +/- 0.0310 (n=3) | 0.3257 +/- 0.0639 (n=3) | 0.4604 +/- 0.0986 (n=3) |
| permuted_graph | 3.1072 +/- 0.4138 (n=3) | 0.1547 +/- 0.0327 (n=3) | 0.2536 +/- 0.0607 (n=3) | 0.3658 +/- 0.0961 (n=3) |

Paired `D_true_norm(permuted - graph)`: `0.4760 +/- 0.1695 (n=3)`
Graph lower latent-energy count: `3/3`

## Latent-Energy Correlation

Pearson r between `D_true_norm(Delta_H)` and H=32 rollout over audited runs: `-0.3488`

## Strict Protocol Interpretation

| field | value |
| --- | --- |
| Legacy automatic classification | topology_aligned_latent_smoothing |
| Strict manuscript label | temporal_sufficient |
| Diagnostic failure mode | graph_free_temporal_smoothing_explains_gain |
| Recommended next experiment | test_operator_aligned_heat_prior_or_first_order_heat_residual |
| Claim boundary | True graph provenance alone is insufficient; the tested prior form may be mismatched to the physical operator. |

The strict label follows the staged manuscript hierarchy and should be preferred over the legacy automatic classification when they disagree.

## Final Classification

`topology_aligned_latent_smoothing`

## Temporal Prior Interpretation

H=32 temporal_smooth rollout: `0.7912`; temporal gain vs none: `+7.8%`; graph gain vs temporal_smooth: `-0.5%`.

Interpretation: graph gain may be temporal-smoothing-like rather than topology-specific.

Decision rules:
- `no_graph_gain`: true graph mini run does not beat GNN none at H=32.
- `generic_smoothing`: true graph beats none but does not beat the permuted graph at H=32.
- `candidate_topology_specific`: true graph beats both none and permuted, but latent alignment is absent or not audited.
- `topology_aligned_latent_smoothing`: true graph beats both rollout controls and its learned `Delta_H` is smoother/more low-frequency in the true graph basis.
- `temporal_smooth`: graph-free temporal baseline; it does not change the graph classification by itself.

## Files

- `analysis_out/preflight_runs/graph_heat_3seed/quick_ep5/summary.csv`
- latent artifacts under `analysis_out/preflight_runs/graph_heat_3seed/quick_ep5/artifacts`
