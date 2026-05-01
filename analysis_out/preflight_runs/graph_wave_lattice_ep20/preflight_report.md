# Graph Prior Preflight Report

Created: `2026-05-01T01:36:40.911406+00:00`
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
| epochs | 20 |
| train transitions | 96 |
| eval transitions | 32 |
| horizons | 16,32 |
| raw normalization | lambda_max |
| N permuted graphs | 8 |
| M random graphs | 8 |

## Stage 0: Raw Graph-Dynamics Diagnostic

| graph type | D_dX_norm | R_low K=2 | R_low K=4 | R_low K=8 | D gap control-true |
| --- | --- | --- | --- | --- | --- |
| true_graph | 0.577167 | 0.0001 | 0.0018 | 0.0089 | 0.000000 |
| permuted_graph | 0.464986 | 0.0249 | 0.0754 | 0.1955 | -0.112181 |
| random_graph | 0.387498 | 0.0352 | 0.1029 | 0.2056 | -0.189669 |

Positive `D gap control-true` means raw temporal changes are smoother on the true graph than on that control.

## Stage 1: Mini Training

| prior | H=16 | H=32 | final train loss | prior loss mean |
| --- | --- | --- | --- | --- |
| none | 0.0347 +/- 0.0000 (n=1) | 0.0826 +/- 0.0000 (n=1) | 0.000005 +/- 0.000000 (n=1) | 0.0000 +/- 0.0000 (n=1) |
| graph | 0.0213 +/- 0.0000 (n=1) | 0.0953 +/- 0.0000 (n=1) | 0.000246 +/- 0.000000 (n=1) | 0.0255 +/- 0.0000 (n=1) |
| permuted_graph | 0.0226 +/- 0.0000 (n=1) | 0.1021 +/- 0.0000 (n=1) | 0.000234 +/- 0.000000 (n=1) | 0.0216 +/- 0.0000 (n=1) |

Graph gain vs none at H=32: `-15.3%`
True-vs-permuted gain at H=32: `+6.6%`

## Stage 2: Latent Audit

| prior | D_true(Delta_H) norm | R_low true K=2 | R_low true K=4 | R_low true K=8 |
| --- | --- | --- | --- | --- |
| graph | 2.7888 +/- 0.0000 (n=1) | 0.1728 +/- 0.0000 (n=1) | 0.2671 +/- 0.0000 (n=1) | 0.4062 +/- 0.0000 (n=1) |
| permuted_graph | 3.1887 +/- 0.0000 (n=1) | 0.1287 +/- 0.0000 (n=1) | 0.2114 +/- 0.0000 (n=1) | 0.3277 +/- 0.0000 (n=1) |

Paired `D_true_norm(permuted - graph)`: `0.3998 +/- 0.0000 (n=1)`
Graph lower latent-energy count: `1/1`

## Latent-Energy Correlation

Pearson r between `D_true_norm(Delta_H)` and H=32 rollout over audited runs: `NA`

## Final Classification

`no_graph_gain`

Decision rules:
- `no_graph_gain`: true graph mini run does not beat GNN none at H=32.
- `generic_smoothing`: true graph beats none but does not beat the permuted graph at H=32.
- `candidate_topology_specific`: true graph beats both none and permuted, but latent alignment is absent or not audited.
- `topology_aligned_latent_smoothing`: true graph beats both rollout controls and its learned `Delta_H` is smoother/more low-frequency in the true graph basis.

## Files

- `analysis_out/preflight_runs/graph_wave_lattice_ep20/summary.csv`
- latent artifacts under `analysis_out/preflight_runs/graph_wave_lattice_ep20/artifacts`
