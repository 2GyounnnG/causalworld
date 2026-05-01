# Model-Conditioned Prior Preflight Tool

This is a tool-facing README draft for `scripts/graph_prior_preflight_check.py`. It explains how to use the preflight tool and how to read its results without over-claiming what the benchmark supports.

## What This Tool Is For

The preflight tool asks a model-conditioned question:

> Given node-wise trajectories, a candidate graph, a latent dynamics model, and a training budget, which prior family is useful?

It is meant for representative node-wise physical prediction regimes with candidate graph structure, including coupled oscillators, spring mechanics, graph waves, heat diffusion, low-frequency graph dynamics, long-range particle interactions, molecular dynamics analyses from earlier cycles, and traffic forecasting with a data-derived graph.

The output is a control-oriented recommendation about which prior family to carry forward under the tested condition: graph prior, temporal prior, latent audit, or no prior.

## What This Tool Is Not

The tool is not:

- a dataset-intrinsic topology detector.
- a final SOTA benchmark.
- causal graph discovery.
- evidence that establishes a graph is physically true.
- coverage of all physics.

A graph prior can help because it encodes useful topology, because it acts as generic smoothing, or because it mimics a simpler temporal regularizer. The tool separates these possibilities with controls.

## Inputs

Each dataset adapter should provide:

- `X_t`: node-wise trajectories with shape `[T, N, d]`.
- `L_true`: candidate graph Laplacian.
- metadata: graph source, topology, simulator parameters, and run-relevant dataset details.

Each preflight run also specifies:

- model condition: the tested latent dynamics architecture and fixed training settings.
- training budget: quick or standard epochs and transition counts.
- prior families: no prior, graph prior, permuted graph control, optional random graph, and optional calibrated temporal smoothing.

## Prior Families

The current prior families are:

- `none`: baseline transition model with no auxiliary prior.
- `graph_laplacian`: candidate-graph Laplacian prior on learned node-wise latents.
- `permuted_graph`: spectrum-matched permuted graph control.
- `random_graph`: optional random graph control.
- `temporal_smooth`: graph-free temporal smoothness on node-wise latents.

Use `--include-temporal-prior --calibrate-prior-strength` when comparing graph and temporal priors. Calibration is necessary because uncalibrated temporal prior losses can be orders of magnitude smaller than graph prior losses.

Future prior families may include covariance and energy-conservation baselines.

## Core Comparisons

Read results through these comparisons:

- `graph` vs `none`: is any graph prior useful?
- `graph` vs `permuted_graph`: is the candidate topology better than spectrum-matched smoothing?
- `graph` vs calibrated `temporal_smooth`: is topology needed beyond graph-free temporal smoothing?
- quick vs standard budgets: is the effect low-budget-only?
- latent audit: do learned temporal deltas satisfy graph-aligned smoothness, such as lower `D_L(Delta H)`?

Rollout gain alone is not enough for a topology claim.

## Recommended Modes

Quick mode:

```bash
--epochs 5 --train-transitions 96 --eval-transitions 32 --horizons 16 32
```

Use quick mode to screen low-budget prior usefulness.

Standard mode:

```bash
--epochs 20 --train-transitions 96 --eval-transitions 32 --horizons 16 32
```

Use standard mode to test whether quick-mode prior gains persist.

Audit mode:

Use saved latent traces or audit outputs to compare candidate-graph and control-graph smoothness of learned temporal deltas. The main check is `D_L(Delta H)`, supported by low-frequency energy ratios when available.

## Example Commands

Spring-mass lattice with calibrated temporal comparison:

```bash
conda run -n causalworld python scripts/graph_prior_preflight_check.py \
  --dataset spring_mass_lattice \
  --prior-weight 0.1 \
  --epochs 5 \
  --train-transitions 96 \
  --eval-transitions 32 \
  --horizons 16 32 \
  --include-temporal-prior \
  --calibrate-prior-strength \
  --out-dir analysis_out/preflight_runs/spring_mass_lattice_ep5_temporal_calibrated
```

N-body distance with calibrated temporal comparison:

```bash
conda run -n causalworld python scripts/graph_prior_preflight_check.py \
  --dataset nbody_distance \
  --prior-weight 0.1 \
  --epochs 5 \
  --train-transitions 96 \
  --eval-transitions 32 \
  --horizons 16 32 \
  --include-temporal-prior \
  --calibrate-prior-strength \
  --out-dir analysis_out/preflight_runs/nbody_distance_ep5_temporal_calibrated
```

Custom adapter placeholder:

```bash
conda run -n causalworld python scripts/graph_prior_preflight_check.py \
  --dataset <adapter_name> \
  --topology <optional_topology> \
  --prior-weight 0.1 \
  --epochs 5 \
  --train-transitions <train_transitions> \
  --eval-transitions <eval_transitions> \
  --horizons 16 32 \
  --include-temporal-prior \
  --calibrate-prior-strength \
  --out-dir analysis_out/preflight_runs/<custom_run_name>
```

## Classification Labels

Use these labels for tool-facing summaries:

| label | interpretation |
| --- | --- |
| `no_prior_gain` | The tested priors do not improve over no-prior. |
| `graph_generic_smoothing` | Graph helps, but smoothing controls explain the gain. |
| `temporal_smoothing_sufficient` | Calibrated temporal smoothing matches or beats graph. |
| `candidate_topology_specific` | Graph beats none, permuted graph, and calibrated temporal smoothing. |
| `topology_aligned_latent_smoothing` | Latent audit supports graph-aligned learned temporal smoothing. |
| `low_budget_only` | Gain appears in quick mode but not standard mode. |
| `overconstrained` | Prior harms rollout by constraining too much. |
| `inconclusive` | Controls or audit evidence are insufficient. |

## Current Benchmark Snapshot

| case | result |
| --- | --- |
| `spring_mass_lattice` | Temporal smoothing sufficient after calibration. |
| `graph_wave_lattice` | Temporal smoothing sufficient after calibration. |
| `nbody_distance` | Candidate topology-specific in quick mode. |
| `METR-LA correlation graph` | No graph gain under the correlation graph. |
| `graph_heat_lattice` | No graph gain. |
| `HO lattice/Cycle8` | Topology-aligned latent smoothing positive case. |

The most important emerging pattern is budget sensitivity. Spring-mass, graph-wave, and n-body quick runs showed graph-prior gains, but standard-budget runs showed no-prior or permuted controls catching up or outperforming.

## Practical Decision Guide

Use the preflight result to choose the next full-scale run:

- `no_prior_gain`: skip the prior.
- `temporal_smoothing_sufficient`: use temporal smoothing, or test graph only if a topology claim matters.
- `candidate_topology_specific`: run full audit, additional seeds, and stronger controls.
- `topology_aligned_latent_smoothing`: carry the graph prior forward under the tested model condition.

The cleanest graph-prior attribution statement requires graph beating no-prior, spectrum-matched permuted graph, and calibrated temporal smoothing, plus supportive latent audit evidence.
