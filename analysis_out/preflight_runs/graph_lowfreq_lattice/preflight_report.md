# Graph Prior Preflight Report

Created: `2026-04-30T23:30:44.186123+00:00`
Schema version: `graph_prior_preflight_v2`

Scope: lightweight dataset-adapter preflight using the existing GNN trainer, graph prior loss, and rollout evaluator.
No ISO17, rMD17 top-up, full sweeps, or large experiments were run.

## Configuration

| field | value |
| --- | --- |
| dataset | graph_lowfreq_lattice |
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

## Stage 0: Raw Graph-Dynamics Diagnostic

| graph type | D_dX_norm | R_low K=2 | R_low K=4 | R_low K=8 | D gap control-true |
| --- | --- | --- | --- | --- | --- |
| true_graph | 0.078922 | 0.2528 | 0.7101 | 1.0000 | 0.000000 |
| permuted_graph | 0.459954 | 0.0260 | 0.0784 | 0.2099 | 0.381032 |
| random_graph | 0.376278 | 0.0335 | 0.0756 | 0.1951 | 0.297356 |

Positive `D gap control-true` means raw temporal changes are smoother on the true graph than on that control.

## Stage 1: Mini Training

| prior | H=16 | H=32 | final train loss | prior loss mean |
| --- | --- | --- | --- | --- |
| none | 0.2333 +/- 0.0000 (n=1) | 0.7210 +/- 0.0000 (n=1) | 0.000229 +/- 0.000000 (n=1) | 0.0000 +/- 0.0000 (n=1) |
| graph | 0.2253 +/- 0.0000 (n=1) | 0.6919 +/- 0.0000 (n=1) | 0.000483 +/- 0.000000 (n=1) | 0.0023 +/- 0.0000 (n=1) |
| permuted_graph | 0.2142 +/- 0.0000 (n=1) | 0.6703 +/- 0.0000 (n=1) | 0.000583 +/- 0.000000 (n=1) | 0.0056 +/- 0.0000 (n=1) |

Graph gain vs none at H=32: `+4.0%`
True-vs-permuted gain at H=32: `-3.2%`

## Stage 2: Latent Audit

Skipped because the mini true-graph run did not beat the permuted control at H=32.

## Final Classification

`generic_smoothing`

Decision rules:
- `no_graph_gain`: true graph mini run does not beat GNN none at H=32.
- `generic_smoothing`: true graph beats none but does not beat the permuted graph at H=32.
- `candidate_topology_specific`: true graph beats both none and permuted, but latent alignment is absent or not audited.
- `topology_aligned_latent_smoothing`: true graph beats both rollout controls and its learned `Delta_H` is smoother/more low-frequency in the true graph basis.

## Files

- `analysis_out/preflight_runs/graph_lowfreq_lattice/summary.csv`
