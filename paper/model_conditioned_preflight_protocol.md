# Model-Conditioned Prior Preflight Protocol

This is a tool-facing protocol draft for the graph-prior preflight workflow. It is not a full paper. Its purpose is to make the tool's scope, comparisons, labels, and practical decision rules explicit.

## Problem Solved

Given a node-wise physical or scientific dynamics dataset, a candidate graph, and a latent dynamics model, the preflight tool recommends a prior family for the tested model and training-budget condition.

The central question is model-conditioned:

> Under this model class, this prior weight or calibrated prior strength, and this training budget, does a candidate graph prior improve rollout prediction, and is that gain better explained by topology, generic graph smoothing, or graph-free temporal smoothing?

The tool is designed for representative node-wise physical/scientific dynamics with candidate graph structure. It helps decide whether a larger run should spend compute on graph priors, temporal priors, latent audits, or additional controls.

## Non-Claims

The preflight result does not claim:

- It is not a dataset-intrinsic topology detector.
- It is not a final SOTA benchmark.
- It is not causal graph discovery.
- It does not establish that a graph is physically true.
- It does not claim coverage of all physics.

A positive result means the tested prior family helped the tested model under the tested budget. A topology-specific label provides evidence for graph-prior attribution under those conditions, not confirmation that the candidate graph is the physical interaction graph.

## Inputs

The protocol assumes the following inputs:

- Node-wise trajectories `X_t` with shape `[T, N, d]`.
- Candidate graph or Laplacian `L_true`.
- Model condition, including encoder, transition model, latent dimension, optimizer, and other fixed training settings.
- Training budget, such as quick 5-epoch or standard 20-epoch preflight runs.
- Prior families to compare.

Dataset adapters should return the trajectory tensor, candidate Laplacian, and metadata describing graph source, topology, simulator parameters, and whether the graph is known true, approximate, or data-derived.

## Prior Families

The current and planned prior families are:

- `none`: no auxiliary prior.
- `graph_laplacian`: Laplacian smoothness on node-wise learned latents using the candidate graph.
- `permuted_graph`: spectrum-matched permuted version of the candidate Laplacian.
- `random_graph`: optional random-edge graph control.
- `temporal_smooth`: graph-free temporal smoothness on learned node-wise latents, calibrated when needed.
- Future: covariance priors and energy-conservation priors.

The calibrated temporal prior is important because its raw prior loss can be orders of magnitude smaller than graph-Laplacian losses. Calibration compares prior families at matched initial effective regularization strength.

## Core Comparisons

The protocol centers on these comparisons:

- `graph` vs `none`: does the graph prior improve rollout at all?
- `graph` vs `permuted_graph`: does the candidate topology beat a spectrum-matched smoothing control?
- `graph` vs calibrated `temporal_smooth`: is the graph gain better than graph-free temporal smoothing?
- Quick vs standard budgets: does the prior gain persist after more training?
- Latent audit when available: do learned temporal deltas become smoother in the candidate graph basis, for example by checking `D_L(Delta H)` and low-frequency energy ratios?

These comparisons should be read together. A graph beating `none` is not enough for a topology claim if it loses to a permuted or calibrated temporal baseline.

## Classification Labels

Use these protocol-level labels when summarizing outcomes:

| label | meaning |
| --- | --- |
| `no_prior_gain` | No tested prior improves meaningfully over the no-prior model. |
| `graph_generic_smoothing` | Graph prior helps versus none but does not beat spectrum-matched or random graph smoothing controls. |
| `temporal_smoothing_sufficient` | Calibrated temporal smoothing matches or beats the graph prior, so graph topology is not necessary to explain the gain. |
| `candidate_topology_specific` | Candidate graph beats none, spectrum-matched permuted graph, and calibrated temporal smoothing under the tested budget. |
| `topology_aligned_latent_smoothing` | Latent audit shows learned temporal deltas are smoother or more low-frequency under the candidate graph when the graph prior is used. |
| `low_budget_only` | A prior helps in quick mode but loses the advantage under standard or longer training. |
| `overconstrained` | The prior appears to harm rollout, likely by suppressing useful latent variation. |
| `inconclusive` | Controls, audits, or repeated runs are insufficient to support a stronger label. |

Implementation-specific report labels may differ slightly, such as `no_graph_gain` or `generic_smoothing`. The protocol labels are intended for higher-level interpretation.

## Recommended Usage Modes

### Quick Mode

Quick mode uses a small training budget, typically 5 epochs. It screens whether a prior is useful as a low-budget regularizer.

Recommended use:

- screen new dataset adapters.
- identify cheap positive candidates.
- run graph, permuted graph, and calibrated temporal controls.

### Standard Mode

Standard mode uses a larger budget, typically 20 epochs. It tests whether the prior advantage persists after the base model has more opportunity to fit the dynamics.

Recommended use:

- validate quick-mode positives.
- identify low-budget-only effects.
- check whether no-prior or permuted controls catch up.

### Audit Mode

Audit mode saves latent traces and checks whether graph-prior training changes latent temporal geometry. The key audit is whether learned temporal deltas `Delta H` are smoother on the candidate graph than on controls, such as by measuring `D_L(Delta H)`.

Recommended use:

- support topology-aligned latent smoothing claims.
- diagnose graph gains that may be generic smoothing.
- avoid over-interpreting rollout metrics alone.

## Example CLI Commands

Spring-mass lattice with calibrated temporal prior:

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

N-body distance graph with calibrated temporal prior:

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

Custom dataset placeholder:

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

## Benchmark Results Snapshot

The current benchmark summary uses existing outputs only.

| case | current readout |
| --- | --- |
| `spring_mass_lattice` | Temporal smoothing is sufficient after calibration. Calibrated `temporal_smooth` H32 is `0.5689`, graph H32 is `0.5740`. |
| `graph_wave_lattice` | Temporal smoothing is sufficient after calibration. Calibrated `temporal_smooth` H32 is `0.5663`, graph H32 is `0.5712`. |
| `nbody_distance` | Candidate topology-specific in quick mode. Graph H32 is `0.5433`, permuted H32 is `0.5838`, temporal H32 is `0.6453`. |
| `METR-LA correlation graph` | No graph gain under the correlation graph in the main T=2000/train160 run. |
| `graph_heat_lattice` | No graph gain, despite using the true generator graph. |
| `HO lattice/Cycle8` | Topology-aligned latent smoothing positive case. |

The broad pattern is early topology-aligned regularization with possible late catch-up: graph priors can help low-budget rollout, but with more training no-prior or spectrum-matched controls may catch up or outperform.

## Practical Recommendation

Use the preflight result to decide what to run at full scale:

- If `no_prior_gain`: skip the prior for the larger run unless there is another reason to test it.
- If `temporal_smoothing_sufficient`: use the temporal prior, or test the graph prior only if a topology claim matters.
- If `candidate_topology_specific`: run full audits, additional seeds, and stronger controls.
- If `topology_aligned_latent_smoothing`: the graph prior is justified under the tested model condition, especially if it also beats calibrated temporal and permuted controls.

The strongest attribution statement comes from agreement between rollout improvement, spectrum-matched controls, calibrated temporal controls, and latent audit evidence.
