# JCS Manuscript Outline

## Proposed Title

Model-Conditioned Preflight for Prior Selection and Attribution in Node-Wise Scientific Latent Dynamics

Alternative titles:

- A Computational Preflight Protocol for Prior Attribution in Scientific Latent Dynamics
- Prior Selection Before Full Training: A Model-Conditioned Preflight Workflow for Node-Wise Dynamics
- Control-Oriented Preflight for Graph and Temporal Priors in Scientific Dynamics Models

## One-Sentence Thesis

Prior selection for scientific latent dynamics should be treated as a model-conditioned computational decision problem, where low-cost preflight runs compare graph, control-graph, temporal, and no-prior conditions before committing full training compute or making topology-oriented claims.

## Section-by-Section Outline

### 1. Introduction

Motivate the practical problem: scientific dynamics models often have plausible priors, but a full training run is expensive and post hoc attribution is fragile. A graph prior may help because the graph is scientifically meaningful, because it supplies generic smoothing, or because any smooth temporal regularizer would have helped. Introduce preflight as a small, structured computational workflow that produces actionable recommendations before full-scale training.

Key framing:

- The task is prior selection and attribution, not invention of a new prior.
- Claims are conditioned on model class, prior strength, and training budget.
- The target use case is node-wise physical or scientific dynamics with candidate graph structure.

### 2. Problem Formulation

Define inputs: trajectories `X_t` with shape `[T, N, d]`, candidate graph or Laplacian `L`, latent node states `H_t`, transition model, training budget, and rollout evaluation horizons.

Define the central question:

> Under this model class, prior strength, and training budget, does a candidate prior improve rollout prediction, and is the gain better attributed to topology, generic graph smoothing, or graph-free temporal smoothing?

State non-goals: not causal discovery, not a universal physics model, not SOTA benchmarking, and not proof that a candidate graph is physically true.

### 3. Preflight Protocol

Describe the workflow:

1. Load dataset adapter outputs: trajectories, candidate Laplacian, and metadata.
2. Run raw diagnostics where available, such as `D_L(Delta X)` and low-frequency ratios.
3. Train short-budget models under `none`, `graph_laplacian`, `permuted_graph`, optional `random_graph`, and optional calibrated `temporal_smooth`.
4. Evaluate rollout error at fixed horizons, especially H=16 and H=32.
5. Apply decision labels and recommend the next computational action.
6. In audit mode, inspect learned latent temporal deltas using `D_L(Delta H)` and low-frequency energy.

Emphasize that spectrum-matched permutations preserve Laplacian scale and spectrum while removing node-label semantics.

### 4. Prior Families and Controls

Describe each prior family:

- `none`: base latent dynamics model.
- `graph_laplacian`: smoothness on node-wise learned latents using the candidate graph.
- `permuted_graph`: spectrum-matched graph smoothing control.
- `random_graph`: broad graph-size and edge-count control.
- `temporal_smooth`: graph-free temporal smoothness on learned node-wise latents.

Explain temporal calibration: initial effective regularization strength should be matched because raw temporal prior loss can be orders of magnitude smaller than graph-Laplacian loss.

### 5. Decision Labels and Recommendations

Introduce the protocol labels and their practical meaning. The section should read like a computational decision guide.

Recommended table columns:

- label.
- required evidence.
- recommended action.
- claim allowed.
- claim disallowed.

Examples:

- `temporal_smoothing_sufficient`: use temporal prior or only continue graph prior if topology attribution matters.
- `candidate_topology_specific`: run additional seeds, audit latent traces, and add stronger controls.
- `low_budget_only`: use the prior only as a training-budget regularizer, not as a persistent structural advantage.

### 6. Experimental Stress-Test Design

Frame the experiments as coverage of representative regimes, not a leaderboard.

Regimes to include:

- HO lattice as a controlled topology-specific positive case.
- Spring-mass lattice for second-order mechanics.
- Graph-wave lattice for PDE-like propagation.
- Graph-heat lattice as a graph-generated cautionary case.
- Graph low-frequency lattice as a spectral alignment stress test.
- N-body distance graph for approximate long-range interactions.
- rMD17 molecular dynamics summaries for graph-style smoothing versus exact molecular topology.
- METR-LA correlation graph as a real data-derived graph case.

Report that all results are from existing completed outputs. Do not imply new experiments.

### 7. Results: What the Preflight Reveals

Organize by attribution question rather than by dataset alone.

Subsections:

- Low-budget graph priors can help rollout.
- Spectrum-matched controls often explain graph gains as generic smoothing.
- Calibrated temporal smoothness changes the attribution story.
- Some graph gains are low-budget-only.
- Raw coordinate alignment is useful but insufficient.
- Latent audit can support topology-aligned smoothing when checkpointed traces are available.

Key examples:

- Spring-mass and graph-wave: graph helps at 5 epochs but calibrated temporal smoothing slightly beats graph, so temporal smoothing is sufficient.
- N-body: quick graph run beats none, permuted graph, and calibrated temporal smoothing, supporting candidate-topology-specific usefulness under that condition.
- HO/Cycle 8: graph improves rollout and learned `Delta H` is smoother and more low-frequency on the true graph.
- METR-LA: raw correlation-graph alignment does not translate into rollout gain.
- Graph heat: true generator graph does not automatically imply graph-prior utility.

### 8. Discussion

Interpret the main computational lesson: prior usefulness is conditional and attribution requires controls. Emphasize that preflight is valuable precisely because it can recommend skipping a prior, using a simpler temporal regularizer, or investing in audits before a larger run.

Discuss implications for computational science workflows:

- cheaper triage before large training campaigns.
- clearer distinction between predictive regularization and topology attribution.
- reusable reporting labels for scientific model development.

### 9. Limitations

Summarize model-conditioning, budget-conditioning, approximate graph sources, limited seeds in some reports, reliance on fixed adapters, and the fact that latent audits require saved traces or checkpoints.

### 10. Conclusion

Close with the practical contribution: a preflight protocol that turns prior choice from an intuition-driven decision into a controlled computational screen with explicit attribution boundaries.

## Appendix Plan

- Full protocol label definitions.
- Example CLI commands.
- Dataset adapter metadata.
- Full benchmark table.
- Full Cycle 1-8 attribution notes.
- Derivation or definition of graph Dirichlet energy and low-frequency ratios.
- Additional rollout curves and control comparisons.
