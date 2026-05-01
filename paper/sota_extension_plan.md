# SOTA / Reference-Baseline Extension Plan

This plan is for a focused JCS revision extension. It does not change existing results and should not turn the preflight paper into a leaderboard benchmark. The added baselines are for calibration, scope, and reviewer orientation: they answer whether the current preflight conclusions are plausible relative to stronger known model classes, not whether this paper beats external SOTA.

## Baseline Levels

### 1. Reference-Only Baselines

Use published numbers or qualitative positioning only. These baselines are cited to define the landscape and avoid implying that the preflight model is a state-of-the-art forecaster or molecular force-field model.

Appropriate when:

- the repo does not already contain the model family;
- reproducing the baseline would require a separate training/evaluation stack;
- metric alignment is nontrivial;
- the purpose is scope calibration rather than direct ranking.

### 2. SOTA-Lite Reproducible Baselines

Implement small, local, reproducible versions using existing data adapters and smoke-scale configs. These are deliberately not full SOTA reproductions.

Appropriate when:

- the model can reuse existing synthetic or preflight adapters;
- the metric is the same rollout error already used by the preflight protocol;
- the implementation is small enough to audit;
- the result is framed as a calibration check, not a leaderboard claim.

### 3. Full External SOTA Baselines

Run original or maintained external implementations with their expected preprocessing, splits, and metrics.

Appropriate only if the revision truly needs it, because this changes the paper scope. These should be treated as optional external validation and separated from the main preflight claims.

## A. Traffic / METR-LA

### DCRNN

Recommended level: reference-only by default; full external SOTA only if reviewers demand direct traffic forecasting comparisons.

Local status:

- Not implemented in this repo.
- The repo has a METR-LA adapter for the preflight protocol, including local CSV/adjacency loading and correlation-top-k graph construction.
- Running DCRNN locally would require adding a sequence-to-sequence diffusion convolution recurrent model, scheduled sampling/training loop, and METR-LA preprocessing compatible with the chosen implementation.

Data needed:

- METR-LA speed time series as a time-by-sensor matrix.
- Sensor graph adjacency, usually from the canonical adjacency pickle or a distance/similarity matrix.
- Train/validation/test chronological split.
- Normalization statistics from the training split.
- Windowed input/output tensors, e.g. history length and prediction horizon consistent with the DCRNN setup.

Use in manuscript:

- Cite as a representative traffic forecasting reference model.
- State that preflight uses METR-LA as a graph-prior attribution screen, not as a traffic leaderboard entry.

### Graph WaveNet

Recommended level: reference-only by default; full external SOTA only if the paper adds a traffic appendix.

Local status:

- Not implemented in this repo.
- Can in principle be run externally on METR-LA if data are converted to the implementation's expected tensor format.
- Its adaptive adjacency makes it scientifically useful as a reference for strong learned-graph forecasting, but that also makes it a different question from testing a fixed candidate prior.

Data needed:

- METR-LA time-by-sensor speed array, usually saved as processed train/val/test tensors.
- Sensor adjacency or supports if using graph supports; Graph WaveNet can also learn adaptive adjacency.
- Standard scaler parameters and horizon-specific metrics.

Use in manuscript:

- Cite as a strong learned-graph traffic baseline.
- Do not compare directly unless the same data split, normalization, horizons, and metrics are reproduced.

## B. Molecular / rMD17

### SchNet

Recommended level: reference-only for this manuscript.

Local status:

- Not implemented in this repo.
- The current rMD17 path is a latent rollout/preflight setup, not an energy/force prediction benchmark.
- A faithful SchNet reproduction would require an energy/force training target, neighbor/radius graph construction, force loss, and standard molecular splits.

Data needed:

- rMD17 molecule-specific coordinates, nuclear charges, energies, and forces.
- Standard train/validation/test splits.
- Energy and force loss definitions, unit conventions, and force weighting.
- Radius graph or neighbor-list preprocessing.

Use in manuscript:

- Cite SchNet as a representative continuous-filter molecular GNN / force-field baseline.
- Use it to bound scope: the preflight manuscript is about model-conditioned prior selection, not replacing molecular force-field SOTA.

### EGNN or E(n)-Equivariant GNN

Recommended level: reference-only for rMD17, SOTA-lite only for synthetic N-body if a small local implementation is added.

Local status:

- No full equivariant molecular stack exists in the repo.
- A small EGNN-lite synthetic N-body baseline is feasible because the existing N-body adapter already exposes positions and trajectories.
- Full rMD17 EGNN reproduction would require a proper force/energy target and standard splits, so it should remain reference-only unless the paper scope expands.

Data needed for molecular reproduction:

- rMD17 coordinates, atom types, energies, and forces.
- Radius graph construction per frame.
- Energy/force losses and equivariant message-passing model.
- Standard splits and molecule-specific hyperparameters.

Use in manuscript:

- Cite E(n)-equivariant GNNs as the relevant molecular and N-body model class.
- Optionally add an EGNN-lite synthetic check as a calibration baseline for the N-body preflight regime.

## C. Synthetic Physical Dynamics

### Stronger No-Prior GNN

Recommended level: SOTA-lite reproducible baseline.

Purpose:

- Test whether quick graph-prior gains are mostly capacity/budget artifacts.
- Keep the same data and rollout metric, but strengthen the no-prior model via longer budget, larger hidden dimension, deeper transition MLP, or more sampled transitions.

Local status:

- Partially feasible with existing preflight runners.
- Existing `graph_prior_preflight_check.py` supports no-prior, graph, permuted graph, and temporal smoothing, but does not expose all capacity knobs from the command line.
- The first implementation should be a wrapper that prints reproducible commands and flags missing capacity knobs rather than silently changing the core model.

### EGNN-Lite for N-Body

Recommended level: SOTA-lite if implemented as a small synthetic-only baseline.

Purpose:

- Calibrate whether an equivariant model reduces or removes the apparent need for graph-prior regularization on N-body.
- Keep this as a synthetic calibration check, not a molecular SOTA claim.

Local status:

- Feasible but not currently implemented.
- A minimal version would predict next positions/velocities from current positions, velocities, and masses using all-pairs or radius-limited equivariant message passing.
- It should report the same H=16/H=32 rollout-style metrics or a clearly separated coordinate rollout metric.

### Energy-Drift Prior for Spring-Mass

Recommended level: SOTA-lite if implemented carefully.

Purpose:

- Test whether conserving or stabilizing mechanical energy explains spring-mass graph-prior gains better than graph topology.
- This is a mechanism baseline, not a full Hamiltonian neural network.

Local status:

- Feasible because the synthetic spring-mass adapter already computes per-frame energy.
- Minimal version: add a prior penalizing predicted latent or decoded rollout changes that correlate with large energy drift. Because the current preflight model does not decode coordinates, the first version should either use an auxiliary energy head or stay as a TODO stub until a decoded state target exists.

### Hamiltonian-Style Prior

Recommended level: optional reference/future-work unless a small stable implementation is added.

Purpose:

- Mention as a physically motivated direction.
- Do not implement a full HNN unless the training target and evaluation metric are redesigned around state derivatives or energy conservation.

## Decision Table

| baseline | dataset | purpose | cost | risk | recommended action |
| --- | --- | --- | --- | --- | --- |
| DCRNN | METR-LA | Reference traffic forecasting model with fixed graph diffusion. | High for faithful reproduction. | Metric/split mismatch; paper becomes traffic benchmark. | Reference-only; full external run only in appendix if required. |
| Graph WaveNet | METR-LA | Strong learned/adaptive graph traffic model. | High for faithful reproduction. | Adaptive graph changes the scientific question; metric mismatch. | Reference-only; cite as scope calibration. |
| SchNet | rMD17 | Representative molecular energy/force GNN. | High. | Current repo evaluates latent rollout, not force-field metrics. | Reference-only. |
| EGNN / equivariant molecular GNN | rMD17 | Equivariant molecular baseline. | High. | Requires force/energy stack and standard splits. | Reference-only unless manuscript scope expands. |
| Stronger no-prior GNN | synthetic preflight datasets | Capacity/budget calibration. | Low to medium. | May require exposing model capacity knobs. | SOTA-lite wrapper and smoke commands. |
| EGNN-lite | N-body synthetic | Equivariance calibration for N-body prior claims. | Medium. | New model may be under-tuned; coordinate metric may differ from latent rollout metric. | TODO stub now; implement only as synthetic calibration. |
| Energy-drift prior | spring-mass synthetic | Mechanism check against physics regularization. | Medium. | Current latent model has no decoder; energy prior can be ill-defined. | TODO stub now; implement after defining energy target/head. |
| Hamiltonian-style prior | spring-mass / N-body synthetic | Future physical-prior comparison. | High. | Full HNN changes model family and objective. | Mention as future work, not current baseline. |

## Smoke-Test Commands Only

These commands should only validate argument parsing and command construction:

```bash
python scripts/run_stronger_baseline_preflight.py --dry-run --dataset nbody_distance
python scripts/run_nbody_egnn_lite.py --dry-run
python scripts/run_spring_energy_prior.py --dry-run
```

If running from base Python rather than the project environment, use:

```bash
python scripts/run_stronger_baseline_preflight.py --dry-run --conda-env causalworld
```

## Recommended Revision Framing

The strongest defensible addition is not a leaderboard table. It is a short scope paragraph plus an appendix plan:

- DCRNN, Graph WaveNet, SchNet, and full molecular EGNNs are reference baselines.
- Stronger no-prior GNN, EGNN-lite N-body, and spring energy-drift checks are local calibration baselines.
- The preflight protocol remains model-conditioned: it asks whether a prior helps the chosen model and whether the gain survives controls.
- Any quick candidate topology signal remains triage evidence until standard-budget and audit checks support it.
