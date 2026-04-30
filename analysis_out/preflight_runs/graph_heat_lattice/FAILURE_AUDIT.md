# Graph Heat Failure Audit

Created: `2026-04-30T23:24:29.625814+00:00`
Scope: graph_heat_lattice only; no ISO17, rMD17 top-up, or full sweeps were run.

## Stage 0 Raw Diagnostics

| graph type | D_dX_norm | R_low K=2 | R_low K=4 | R_low K=8 |
| --- | --- | --- | --- | --- |
| true_graph | 0.614624 | 0.0312 | 0.0877 | 0.1972 |
| permuted_graph | 0.463774 | 0.0280 | 0.0856 | 0.1977 |
| random_graph | 0.383011 | 0.0387 | 0.0857 | 0.2022 |

Raw dynamics smoother on true graph than both controls: `NO`.

## Oracle Heat Baseline

| metric | MSE |
| --- | --- |
| one-step | 0.00009918 |
| H=16 rollout | 0.00046113 |
| H=32 rollout | 0.00054215 |

Oracle one-step MSE is within `5 * noise^2`: `YES`.

## Existing Preflight Comparison

| prior | lambda | H=16 | H=32 |
| --- | --- | --- | --- |
| none | 0 | 0.2087 | 0.6036 |
| graph | 0.1 | 0.2269 | 0.6902 |
| permuted_graph | 0.1 | 0.2064 | 0.6058 |

## Lambda Sanity Check

| prior | lambda | H=16 | H=32 | prior loss mean |
| --- | --- | --- | --- | --- |
| graph | 0.001 | 0.2103 | 0.6288 | 0.058923 |
| graph | 0.010 | 0.2133 | 0.6638 | 0.029639 |
| graph | 0.100 | 0.2269 | 0.6902 | 0.016180 |

Best graph lambda by H=32: `0.001` with H=32 `0.6288`.
Lower lambda improves over graph lambda=0.1: `YES`.
High lambda / over-smoothing fully explains the original miss: `NO`.

## Answers

- Is raw dynamics smoother on true graph than controls? `NO`.
- Is the oracle heat model accurate? `YES` for one-step prediction; see rollout MSE above.
- Is true graph prior hurt due to high lambda / over-smoothing? `PARTLY`.
- Should graph_heat be treated as positive-control failure or no-graph-gain under this latent model? Treat as a no-graph-gain case under this latent model, despite the oracle generator being available.

## Files

- `analysis_out/preflight_runs/graph_heat_lattice/summary.csv`
- `analysis_out/preflight_runs/graph_heat_lattice/FAILURE_AUDIT.md`
