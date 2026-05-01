# Graph Prior Preflight Report

Created: `2026-05-01T00:51:05.465802+00:00`
Schema version: `graph_prior_preflight_v2`

Scope: lightweight dataset-adapter preflight using the existing GNN trainer, graph prior loss, and rollout evaluator.
No ISO17, rMD17 top-up, full sweeps, or large experiments were run.

## Configuration

| field | value |
| --- | --- |
| dataset | metr_la |
| topology |  |
| prior_weight | 0.1 |
| seeds | 0 |
| epochs | 5 |
| train transitions | 256 |
| eval transitions | 64 |
| horizons | 16,32 |
| raw normalization | lambda_max |
| N permuted graphs | 8 |
| M random graphs | 8 |

## Stage 0: Raw Graph-Dynamics Diagnostic

| graph type | D_dX_norm | R_low K=2 | R_low K=4 | R_low K=8 | D gap control-true |
| --- | --- | --- | --- | --- | --- |
| true_graph | 0.237967 | 0.0126 | 0.0292 | 0.0557 | 0.000000 |
| permuted_graph | 0.304427 | 0.0048 | 0.0144 | 0.0333 | 0.066460 |
| random_graph | 0.473690 | 0.0041 | 0.0144 | 0.0350 | 0.235723 |

Positive `D gap control-true` means raw temporal changes are smoother on the true graph than on that control.

## Stage 1: Mini Training

| prior | H=16 | H=32 | final train loss | prior loss mean |
| --- | --- | --- | --- | --- |

Graph gain vs none at H=32: `NA`
True-vs-permuted gain at H=32: `NA`

## Stage 2: Latent Audit

Skipped because the mini true-graph run did not beat the permuted control at H=32.

## Final Classification

`no_graph_gain`

Decision rules:
- `no_graph_gain`: true graph mini run does not beat GNN none at H=32.
- `generic_smoothing`: true graph beats none but does not beat the permuted graph at H=32.
- `candidate_topology_specific`: true graph beats both none and permuted, but latent alignment is absent or not audited.
- `topology_aligned_latent_smoothing`: true graph beats both rollout controls and its learned `Delta_H` is smoother/more low-frequency in the true graph basis.

## Files

- `analysis_out/preflight_runs/metr_la_corr_T2000/summary.csv`
