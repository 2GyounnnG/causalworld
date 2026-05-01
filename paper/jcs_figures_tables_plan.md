# JCS Figures and Tables Plan

## Main Figures

### Figure 1: Preflight Workflow Schematic

Show the computational path from dataset adapter to recommendation:

`trajectories + candidate graph -> prior families -> quick run -> controls -> standard run -> audit -> label -> recommended action`

Include the three attribution branches:

- graph beats none only: possible regularization.
- graph beats permuted graph: candidate topology evidence.
- graph beats calibrated temporal smoothness: topology adds value beyond temporal smoothing.

### Figure 2: Control Logic for Attribution

A schematic comparing candidate graph, spectrum-matched permuted graph, random graph, and temporal smoothing. The figure should visually emphasize that the permuted graph preserves Laplacian spectrum while removing node-label semantics.

Suggested panels:

- candidate graph Laplacian.
- permuted Laplacian with identical eigenvalues.
- temporal smoothness baseline with no graph.
- decision logic: graph gain, topology-specific gain, temporal sufficiency.

### Figure 3: Benchmark Overview Across Regimes

Bar or dot plot of H32 rollout for `none`, `graph`, `permuted_graph`, and `temporal_smooth` where available.

Recommended cases:

- HO lattice.
- spring-mass quick calibrated.
- graph-wave quick calibrated.
- n-body quick calibrated.
- METR-LA main run.
- graph heat lattice.

Purpose: show that the same workflow returns different recommendations across regimes.

### Figure 4: Quick Versus Standard Budget

Paired plot showing H32 outcomes at 5 and 20 epochs for:

- spring-mass lattice.
- graph-wave lattice.
- n-body distance.

Message: low-budget graph gains can disappear or become generic smoothing under standard budget.

### Figure 5: Latent Audit Positive Case

Use Cycle 8 HO lattice:

- H32 rollout for graph, permuted_graph, random_graph.
- normalized `D_true(Delta H)`.
- low-frequency ratio `R_low` for K=2, K=4, K=8.

Message: audit mode can support topology-aligned latent smoothing when rollout and latent metrics agree.

## Main Tables

### Table 1: Protocol Modes

| mode | budget | purpose | required outputs | recommendation produced |
| --- | --- | --- | --- | --- |
| quick | short run, e.g. 5 epochs | screen prior utility | rollout and control comparisons | skip, continue, or add controls |
| standard | longer run, e.g. 20 epochs | test persistence | rollout and control comparisons | persistent gain or low-budget-only |
| audit | checkpointed/latent traces | test latent geometry | `D_L(Delta H)`, low-frequency ratios | support or weaken topology-aligned claim |

### Table 2: Prior Families and Controls

| family | role | attribution question |
| --- | --- | --- |
| `none` | baseline | does any prior help? |
| `graph_laplacian` | candidate graph prior | does graph smoothing help? |
| `permuted_graph` | spectrum-matched topology control | does node-label topology matter beyond smoothing scale? |
| `random_graph` | coarse graph control | is the effect robust to unrelated graph structure? |
| `temporal_smooth` | graph-free temporal baseline | is graph topology needed beyond temporal smoothing? |

### Table 3: Preflight Benchmark Summary

Condense `analysis_out/preflight_benchmark_summary.csv`.

Recommended columns:

- case.
- regime.
- graph source.
- budget.
- H32 none.
- H32 graph.
- H32 permuted.
- H32 temporal where available.
- label.
- recommendation.

### Table 4: Claim Table

| manuscript claim | supported? | evidence | required caveat |
| --- | --- | --- | --- |
| Preflight gives actionable prior recommendations. | yes | benchmark summary and reports | conditioned on model and budget |
| Graph priors can improve quick-budget rollout. | yes | spring-mass, graph-wave, n-body, HO | not automatically topology-specific |
| Spectrum-matched controls are necessary for graph attribution. | yes | master decision rules, Cycle 4 diagnostics | random controls alone are insufficient |
| Calibrated temporal smoothing can explain graph gains. | yes | spring-mass and graph-wave calibrated runs | calibration must be reported |
| N-body quick is candidate-topology-specific. | yes, under tested condition | graph beats none, permuted, temporal | not proof of true physical interaction graph |
| HO lattice audit supports topology-aligned latent smoothing. | yes, controlled case | Cycle 8 checkpointed audit | high-prior-weight and model-conditioned |
| Raw graph alignment predicts prior utility. | no | Cycle 6, METR-LA, graph heat | use only as supporting diagnostic |
| Graph-generated data implies graph prior utility. | no | graph heat lattice | latent model and prior can still overconstrain |

### Table 5: What Not To Claim

| avoid | replacement wording |
| --- | --- |
| We propose a new graph prior. | We propose a preflight protocol for choosing and attributing priors. |
| Graph priors are universally best. | Prior utility is model-, budget-, and graph-source-conditioned. |
| The candidate graph is physically true. | The candidate graph prior helped under the tested controls and budget. |
| The method discovers causal structure. | The workflow compares prior usefulness and attribution controls. |
| Results are SOTA benchmarks. | Results are stress tests of a computational decision protocol. |
| Raw smoothness proves topology use. | Raw smoothness is a diagnostic that must be checked against rollout controls and latent audits. |

## Supplementary Figures and Tables

- Full rollout curves for H=1, 2, 4, 8, 16, 32.
- Full preflight benchmark CSV as supplementary table.
- Cycle 1-5 graph specificity table.
- Cycle 6 raw alignment correlations.
- Cycle 8 seed-wise latent audit metrics.
- Example generated preflight reports.
- CLI/config table for reproducibility.
