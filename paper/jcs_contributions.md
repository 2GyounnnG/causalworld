# JCS Contributions

## Proposed Title

Model-Conditioned Preflight for Prior Selection and Attribution in Node-Wise Scientific Latent Dynamics

## Core Contributions

- A computational preflight protocol for prior selection in node-wise physical and scientific latent dynamics, conditioned on a specified model class, prior strength, optimizer setting, and training budget.
- A control-oriented attribution workflow that separates candidate graph benefit from generic graph smoothing and graph-free temporal smoothing.
- A spectrum-matched permuted graph control that preserves Laplacian scale and spectrum while removing node-label semantics, making it the decisive control for topology-specific attribution.
- A calibrated temporal smoothness baseline for fair graph-versus-temporal prior comparison when raw auxiliary losses differ by orders of magnitude.
- Three operational modes: quick mode for low-budget screening, standard mode for persistence checks, and audit mode for latent-geometry diagnostics.
- A practical label set for recommendations: `no_prior_gain`, `graph_generic_smoothing`, `temporal_smoothing_sufficient`, `candidate_topology_specific`, `topology_aligned_latent_smoothing`, `low_budget_only`, `overconstrained`, and `inconclusive`.
- An empirical stress test over representative prediction regimes: harmonic oscillator lattices, spring-mass dynamics, graph waves, graph heat diffusion, low-frequency graph dynamics, n-body interactions, molecular dynamics summaries, and traffic forecasting with a data-derived graph.
- Actionable decision rules for deciding whether a larger run should use no prior, temporal smoothing, a graph prior with audits, or additional controls.

## Journal of Computational Science Positioning

The manuscript should be framed as a computational methodology paper. The central object is the preflight workflow: inputs, controls, calibration, decision labels, and downstream recommendations. The experiments demonstrate how the workflow behaves across regimes; they are not presented as a leaderboard.

Recommended positioning sentence:

> We study prior selection as a model-conditioned computational decision problem: before committing full training compute, a short preflight run should determine whether a prior helps the tested latent dynamics model, whether the effect survives attribution controls, and what evidence is still needed before making topology-oriented claims.

## Claim Table

| claim | supported wording | evidence source | boundary |
| --- | --- | --- | --- |
| Preflight workflow | The protocol provides a practical workflow for prior selection and attribution before full training. | `model_conditioned_preflight_protocol.md`, tool reports, benchmark summary | It recommends under tested model and budget conditions only. |
| Graph gain screening | Graph priors can improve low-budget rollout in several node-wise dynamics regimes. | Spring-mass, graph-wave, n-body quick runs; master attribution summaries | A graph beating no-prior is not enough for topology attribution. |
| Spectrum-matched control | Permuted graph controls are necessary because they preserve Laplacian spectrum while removing node-label topology. | Cycle 4 diagnostics, master decision rules | Random graph controls are useful but weaker scale controls. |
| Temporal baseline | Calibrated temporal smoothness can explain some graph-prior gains. | Spring-mass and graph-wave calibrated temporal runs | Uncalibrated temporal baselines should not be used for attribution. |
| Candidate topology specificity | N-body quick mode supports candidate-topology-specific usefulness under the tested condition. | `nbody_distance_ep5_temporal_calibrated` | This is not a claim that the distance-kNN graph is the true interaction graph. |
| Latent audit value | Checkpointed HO lattice shows rollout improvement with smoother, more low-frequency latent temporal deltas on the true graph. | Cycle 8 report | This is a controlled positive case, not universal latent-alignment evidence. |
| Budget dependence | Some low-budget graph gains disappear or reverse at standard budget. | Spring-mass ep5/ep20, graph-wave ep5/ep20, n-body ep5/ep20 | The protocol is model- and budget-conditioned. |
| Raw alignment caution | Raw graph-dynamics alignment is not sufficient to predict prior utility. | Cycle 6 diagnostics, METR-LA, graph low-frequency lattice | Raw-coordinate diagnostics are supporting probes, not decision rules. |

## What Not To Claim

- Do not claim the paper introduces a new graph prior.
- Do not claim universal superiority of graph priors, spectral priors, or temporal priors.
- Do not claim state-of-the-art prediction performance.
- Do not claim causal graph discovery.
- Do not claim the candidate graph is physically true because it improves rollout.
- Do not claim that raw graph smoothness is sufficient for graph-prior usefulness.
- Do not claim topology specificity when the candidate graph fails to beat a spectrum-matched permuted graph.
- Do not claim graph topology is necessary when calibrated temporal smoothing matches or beats the graph prior.
- Do not claim broad latent-alignment mechanism evidence beyond the audited settings.
