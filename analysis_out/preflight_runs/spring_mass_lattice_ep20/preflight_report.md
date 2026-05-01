# Graph Prior Preflight Report

Created: `2026-05-01T01:29:56.984756+00:00`
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
| true_graph | 0.576065 | 0.0001 | 0.0017 | 0.0085 | 0.000000 |
| permuted_graph | 0.466370 | 0.0251 | 0.0757 | 0.1957 | -0.109696 |
| random_graph | 0.385723 | 0.0351 | 0.1029 | 0.2054 | -0.190343 |

Positive `D gap control-true` means raw temporal changes are smoother on the true graph than on that control.

## Stage 1: Mini Training

| prior | H=16 | H=32 | final train loss | prior loss mean |
| --- | --- | --- | --- | --- |
| none | 0.0373 +/- 0.0000 (n=1) | 0.0809 +/- 0.0000 (n=1) | 0.000006 +/- 0.000000 (n=1) | 0.0000 +/- 0.0000 (n=1) |
| graph | 0.0227 +/- 0.0000 (n=1) | 0.1048 +/- 0.0000 (n=1) | 0.000190 +/- 0.000000 (n=1) | 0.0190 +/- 0.0000 (n=1) |
| permuted_graph | 0.0263 +/- 0.0000 (n=1) | 0.1238 +/- 0.0000 (n=1) | 0.000175 +/- 0.000000 (n=1) | 0.0164 +/- 0.0000 (n=1) |

Graph gain vs none at H=32: `-29.5%`
True-vs-permuted gain at H=32: `+15.4%`

## Stage 2: Latent Audit

| prior | D_true(Delta_H) norm | R_low true K=2 | R_low true K=4 | R_low true K=8 |
| --- | --- | --- | --- | --- |
| graph | 2.7841 +/- 0.0000 (n=1) | 0.1837 +/- 0.0000 (n=1) | 0.2688 +/- 0.0000 (n=1) | 0.4017 +/- 0.0000 (n=1) |
| permuted_graph | 3.1644 +/- 0.0000 (n=1) | 0.1366 +/- 0.0000 (n=1) | 0.2097 +/- 0.0000 (n=1) | 0.3223 +/- 0.0000 (n=1) |

Paired `D_true_norm(permuted - graph)`: `0.3802 +/- 0.0000 (n=1)`
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

- `analysis_out/preflight_runs/spring_mass_lattice_ep20/summary.csv`
- latent artifacts under `analysis_out/preflight_runs/spring_mass_lattice_ep20/artifacts`
