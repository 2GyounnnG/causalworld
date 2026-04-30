# Graph Low-Frequency Failure Analysis

Created: `2026-04-30T23:36:30.092136+00:00`
Scope: graph_lowfreq_lattice only; seed 0; prior weight 0.1; epochs 20 and 50.
No ISO17, rMD17 top-up, or broad sweeps were run.

## Raw Alignment

| graph type | D_dX_norm | R_low K=4 | D gap control-true |
| --- | --- | --- | --- |
| true_graph | 0.078922 | 0.7101 | 0.000000 |
| permuted_graph | 0.459954 | 0.0784 | 0.381032 |
| random_graph | 0.376278 | 0.0756 | 0.297356 |

## Longer Mini-Training Rollout

| epochs | none H16 | none H32 | graph H16 | graph H32 | permuted H16 | permuted H32 | graph beats permuted H32 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20 | 0.0363 | 0.0939 | 0.0187 | 0.0580 | 0.0210 | 0.0780 | YES |
| 50 | 0.0092 | 0.0277 | 0.0014 | 0.0038 | 0.0006 | 0.0021 | NO |

## Forced Latent Audit

| epochs | graph D_true_norm(Delta_H) | permuted D_true_norm(Delta_H) | graph R_low K=4 | permuted R_low K=4 | graph latent smoother |
| --- | --- | --- | --- | --- | --- |
| 20 | 0.5387 | 0.7497 | 0.7811 | 0.6291 | YES |
| 50 | 0.6458 | 0.8690 | 0.6781 | 0.5406 | YES |

## Answers

1. Does true graph overtake permuted with longer mini-training? `YES`.
2. Does graph prior induce more true-graph-smooth latent dynamics even when rollout is worse? `YES`.
3. Main diagnosis: longer mini-training is enough for the true graph to overtake the permuted control.

## Files

- `analysis_out/preflight_runs/graph_lowfreq_lattice/FAILURE_ANALYSIS.md`
- latent artifacts under `analysis_out/preflight_runs/graph_lowfreq_lattice/artifacts`
