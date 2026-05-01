# JCS Limitations and Claim Boundaries

## Limitations

- The protocol is model-conditioned. A recommendation applies to the tested encoder, transition model, latent dimension, optimizer, prior weight, and training setup.
- The protocol is budget-conditioned. Quick-mode gains may disappear under standard or longer training, as seen in spring-mass, graph-wave, and n-body comparisons.
- Candidate graphs can be true simulator graphs, approximate physical graphs, or data-derived graphs. These sources should not be interpreted equivalently.
- Spectrum-matched permuted controls are strong topology controls, but they do not exhaust every possible graph confound.
- Random graph controls are useful secondary checks but are weaker than permuted controls because they do not preserve the full Laplacian spectrum.
- Calibrated temporal smoothness is required for fair graph-versus-temporal attribution. Uncalibrated temporal results can understate temporal-prior utility.
- Raw-coordinate diagnostics such as `D_L(Delta X)` are not sufficient predictors of prior usefulness because the prior acts on learned latent states.
- Latent audit requires saved latent traces or checkpoints. Missing checkpoint artifacts blocked earlier latent-alignment analysis until a checkpointed Cycle 8 audit was run.
- The benchmark covers representative physical and scientific regimes, not all physics.
- Some reports use small seed counts or preflight-scale budgets by design, so they should guide full-run decisions rather than replace full validation.
- METR-LA uses a correlation-derived sensor graph in the main reported run, not an official road adjacency graph.

## What Not To Claim

- Do not claim a new graph prior.
- Do not claim a universal physics predictor.
- Do not claim causal graph discovery.
- Do not claim state-of-the-art benchmark performance.
- Do not claim that a candidate graph is physically true because its prior improves rollout.
- Do not claim exact topology specificity without beating the spectrum-matched permuted graph control.
- Do not claim graph topology is necessary when calibrated temporal smoothing matches or beats graph.
- Do not claim lambda-robust or budget-robust topology specificity unless adjacent prior weights and longer budgets support it.
- Do not claim raw graph-dynamics alignment is sufficient evidence for graph-prior utility.
- Do not claim graph-generated dynamics automatically benefit from graph latent priors.

## Safe Claim Language

Use:

> Under the tested model condition and training budget, the preflight protocol recommends...

Use:

> The candidate graph prior improves rollout beyond the no-prior baseline but does not beat the spectrum-matched permuted graph, so the gain is best interpreted as generic smoothing.

Use:

> Calibrated temporal smoothing matches or beats the graph prior, so graph topology is not necessary to explain the observed low-budget gain.

Use:

> The graph prior beats no-prior, spectrum-matched permuted graph, and calibrated temporal smoothing under the quick budget; this supports candidate-topology-specific usefulness, pending broader audits and seeds.

Avoid:

> The model discovered the physical graph.

Avoid:

> The graph prior is universally better.

Avoid:

> Raw graph smoothness proves the mechanism.

## Reviewer-Risk Notes

- The manuscript should make clear that the workflow produces recommendations, not final scientific truth.
- The calibrated temporal baseline should be described prominently because it changes the interpretation of spring-mass and graph-wave.
- The HO Cycle 8 audit should be presented as a positive example of audit mode, while rMD17 and other reports should remain bounded by their control outcomes.
- The graph heat and METR-LA cases are important negative examples because they show the workflow can recommend against graph priors.
- The results should distinguish `candidate_topology_specific` from `topology_aligned_latent_smoothing`: the first is a rollout-control result; the second requires audit evidence.
