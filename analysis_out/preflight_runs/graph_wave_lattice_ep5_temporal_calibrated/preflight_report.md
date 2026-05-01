# Graph Prior Preflight Report

Created: `2026-05-01T02:18:31.279458+00:00`
Schema version: `graph_prior_preflight_v2`

Scope: lightweight dataset-adapter preflight using the existing GNN trainer, graph prior loss, and rollout evaluator.
No ISO17, rMD17 top-up, full sweeps, or large experiments were run.

## Configuration

| field | value |
| --- | --- |
| dataset | graph_wave_lattice |
| topology | lattice |
| prior_weight | 0.1 |
| seeds | 0 |
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
| none | 0.2303 +/- 0.0000 (n=1) | 0.6825 +/- 0.0000 (n=1) | 0.100000 +/- 0.000000 (n=1) | 0.100000 +/- 0.000000 (n=1) | 0.000000 +/- 0.000000 (n=1) | 0.0000 +/- 0.0000 (n=1) | 0.000000 +/- 0.000000 (n=1) | 0.000233 +/- 0.000000 (n=1) |
| graph | 0.1933 +/- 0.0000 (n=1) | 0.5712 +/- 0.0000 (n=1) | 0.100000 +/- 0.000000 (n=1) | 0.100000 +/- 0.000000 (n=1) | 0.459671 +/- 0.000000 (n=1) | 0.0826 +/- 0.0000 (n=1) | 0.008264 +/- 0.000000 (n=1) | 0.002564 +/- 0.000000 (n=1) |
| permuted_graph | 0.1938 +/- 0.0000 (n=1) | 0.5711 +/- 0.0000 (n=1) | 0.100000 +/- 0.000000 (n=1) | 0.114813 +/- 0.000000 (n=1) | 0.400365 +/- 0.000000 (n=1) | 0.0693 +/- 0.0000 (n=1) | 0.007953 +/- 0.000000 (n=1) | 0.002515 +/- 0.000000 (n=1) |
| temporal_smooth | 0.1951 +/- 0.0000 (n=1) | 0.5663 +/- 0.0000 (n=1) | 0.100000 +/- 0.000000 (n=1) | 16536.146684 +/- 0.000000 (n=1) | 0.000003 +/- 0.000000 (n=1) | 0.0000 +/- 0.0000 (n=1) | 0.009504 +/- 0.000000 (n=1) | 0.003761 +/- 0.000000 (n=1) |

Graph gain vs none at H=32: `+16.3%`
True-vs-permuted gain at H=32: `-0.0%`

## Stage 2: Latent Audit

Skipped because the mini true-graph run did not beat the permuted control at H=32.

## Final Classification

`generic_smoothing`

## Temporal Prior Interpretation

H=32 temporal_smooth rollout: `0.5663`; temporal gain vs none: `+17.0%`; graph gain vs temporal_smooth: `-0.9%`.

Interpretation: graph gain may be temporal-smoothing-like rather than topology-specific.

Decision rules:
- `no_graph_gain`: true graph mini run does not beat GNN none at H=32.
- `generic_smoothing`: true graph beats none but does not beat the permuted graph at H=32.
- `candidate_topology_specific`: true graph beats both none and permuted, but latent alignment is absent or not audited.
- `topology_aligned_latent_smoothing`: true graph beats both rollout controls and its learned `Delta_H` is smoother/more low-frequency in the true graph basis.
- `temporal_smooth`: graph-free temporal baseline; it does not change the graph classification by itself.

## Files

- `analysis_out/preflight_runs/graph_wave_lattice_ep5_temporal_calibrated/summary.csv`
