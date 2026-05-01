# Graph Prior Preflight Report

Created: `2026-05-01T01:29:37.430666+00:00`
Schema version: `graph_prior_preflight_v2`

Scope: lightweight dataset-adapter preflight using the existing GNN trainer, graph prior loss, and rollout evaluator.
No ISO17, rMD17 top-up, full sweeps, or large experiments were run.

## Configuration

| field | value |
| --- | --- |
| dataset | spring_mass_lattice |
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
| true_graph | 0.576065 | 0.0001 | 0.0017 | 0.0085 | 0.000000 |
| permuted_graph | 0.466370 | 0.0251 | 0.0757 | 0.1957 | -0.109696 |
| random_graph | 0.385723 | 0.0351 | 0.1029 | 0.2054 | -0.190343 |

Positive `D gap control-true` means raw temporal changes are smoother on the true graph than on that control.

## Stage 1: Mini Training

| prior | H=16 | H=32 | final train loss | prior loss mean |
| --- | --- | --- | --- | --- |
| none | 0.2299 +/- 0.0000 (n=1) | 0.6864 +/- 0.0000 (n=1) | 0.000231 +/- 0.000000 (n=1) | 0.0000 +/- 0.0000 (n=1) |
| graph | 0.1937 +/- 0.0000 (n=1) | 0.5740 +/- 0.0000 (n=1) | 0.001927 +/- 0.000000 (n=1) | 0.0615 +/- 0.0000 (n=1) |
| permuted_graph | 0.1977 +/- 0.0000 (n=1) | 0.5841 +/- 0.0000 (n=1) | 0.001812 +/- 0.000000 (n=1) | 0.0525 +/- 0.0000 (n=1) |

Graph gain vs none at H=32: `+16.4%`
True-vs-permuted gain at H=32: `+1.7%`

## Stage 2: Latent Audit

| prior | D_true(Delta_H) norm | R_low true K=2 | R_low true K=4 | R_low true K=8 |
| --- | --- | --- | --- | --- |
| graph | 3.1906 +/- 0.0000 (n=1) | 0.1318 +/- 0.0000 (n=1) | 0.1992 +/- 0.0000 (n=1) | 0.3088 +/- 0.0000 (n=1) |
| permuted_graph | 3.4512 +/- 0.0000 (n=1) | 0.1038 +/- 0.0000 (n=1) | 0.1599 +/- 0.0000 (n=1) | 0.2590 +/- 0.0000 (n=1) |

Paired `D_true_norm(permuted - graph)`: `0.2606 +/- 0.0000 (n=1)`
Graph lower latent-energy count: `1/1`

## Latent-Energy Correlation

Pearson r between `D_true_norm(Delta_H)` and H=32 rollout over audited runs: `NA`

## Final Classification

`topology_aligned_latent_smoothing`

Decision rules:
- `no_graph_gain`: true graph mini run does not beat GNN none at H=32.
- `generic_smoothing`: true graph beats none but does not beat the permuted graph at H=32.
- `candidate_topology_specific`: true graph beats both none and permuted, but latent alignment is absent or not audited.
- `topology_aligned_latent_smoothing`: true graph beats both rollout controls and its learned `Delta_H` is smoother/more low-frequency in the true graph basis.

## Files

- `analysis_out/preflight_runs/spring_mass_lattice_ep5/summary.csv`
- latent artifacts under `analysis_out/preflight_runs/spring_mass_lattice_ep5/artifacts`
