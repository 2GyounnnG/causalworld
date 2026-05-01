# Graph Prior Preflight Report

Created: `2026-04-30T22:49:38.148383+00:00`
Schema version: `graph_prior_preflight_v1`

Scope: lightweight HO preflight using the existing trajectory loader, GNN trainer, graph prior loss, and rollout evaluator.
No ISO17, rMD17 top-up, full sweeps, or large experiments were run.

## Configuration

| field | value |
| --- | --- |
| topology | lattice |
| prior_weight | 0.1 |
| seeds | 0 |
| epochs | 5 |
| train transitions | 48 |
| eval transitions | 12 |
| raw normalization | lambda_max |
| N permuted graphs | 8 |
| M random graphs | 8 |

## Stage 0: Raw Graph-Dynamics Diagnostic

| graph type | D_dX_norm | R_low K=2 | R_low K=4 | R_low K=8 | D gap control-true |
| --- | --- | --- | --- | --- | --- |
| true_graph | 0.562362 | 0.0023 | 0.0215 | 0.0398 | 0.000000 |
| permuted_graph | 0.459020 | 0.0158 | 0.0495 | 0.1120 | -0.103342 |
| random_graph | 0.358071 | 0.0184 | 0.0479 | 0.1156 | -0.204291 |

Positive `D gap control-true` means raw temporal changes are smoother on the true graph than on that control.

## Stage 1: Mini Training

| prior | H=16 | H=32 | final train loss | prior loss mean |
| --- | --- | --- | --- | --- |
| none | 0.3694 +/- 0.0000 (n=1) | 1.0849 +/- 0.0000 (n=1) | 0.000807 +/- 0.000000 (n=1) | 0.0000 +/- 0.0000 (n=1) |
| graph | 0.3946 +/- 0.0000 (n=1) | 1.0139 +/- 0.0000 (n=1) | 0.006062 +/- 0.000000 (n=1) | 0.1557 +/- 0.0000 (n=1) |
| permuted_graph | 0.3983 +/- 0.0000 (n=1) | 1.0678 +/- 0.0000 (n=1) | 0.007755 +/- 0.000000 (n=1) | 0.2070 +/- 0.0000 (n=1) |

Graph gain vs none at H=32: `+6.5%`
True-vs-permuted gain at H=32: `+5.0%`

## Stage 2: Latent Audit

| prior | D_true(Delta_H) norm | R_low true K=2 | R_low true K=4 | R_low true K=8 |
| --- | --- | --- | --- | --- |
| graph | 3.6444 +/- 0.0000 (n=1) | 0.0514 +/- 0.0000 (n=1) | 0.1020 +/- 0.0000 (n=1) | 0.1742 +/- 0.0000 (n=1) |
| permuted_graph | 4.0541 +/- 0.0000 (n=1) | 0.0356 +/- 0.0000 (n=1) | 0.0678 +/- 0.0000 (n=1) | 0.1337 +/- 0.0000 (n=1) |

Paired `D_true_norm(permuted - graph)`: `0.4097 +/- 0.0000 (n=1)`
Graph lower latent-energy count: `1/1`

## Latent-Energy Correlation

Pearson r between `D_true_norm(Delta_H)` and H=32 rollout over audited runs: `NA`

## Final Classification

`topology-aligned latent smoothing`

Decision rules:
- `no graph gain`: true graph mini run does not beat GNN none at H=32.
- `generic smoothing`: true graph beats none but does not beat the permuted graph at H=32.
- `candidate topology-specific`: true graph beats both none and permuted, but latent alignment is absent or not audited.
- `topology-aligned latent smoothing`: true graph beats both rollout controls and its learned `Delta_H` is smoother/more low-frequency in the true graph basis.

## Files

- `/home/user/projects/causalworld/analysis_out/graph_prior_preflight_summary.csv`
- latent artifacts under `experiments/artifacts/graph_prior_preflight`
