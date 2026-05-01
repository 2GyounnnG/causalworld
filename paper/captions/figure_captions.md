# Draft Figure Captions for JCS Manuscript

## Figure 1. Model-conditioned preflight workflow

The preflight protocol converts prior selection into a staged computational decision rather than a full-training gamble. Node-wise trajectories and a candidate prior first pass through raw diagnostics, then quick preflight comparisons, then spectrum-matched graph and calibrated temporal controls. Standard and audit modes are used only when the quick result warrants persistence or mechanism checks. The output is a decision label and recommendation, not a claim that the candidate graph is physically true.

## Figure 2. Decision logic for label assignment and prior recommendation

The decision tree separates practical prior utility from topology-specific attribution. A graph prior must first improve over no-prior training, then beat spectrum-matched permuted graph controls and calibrated temporal smoothing before it supports candidate-topology-specific usefulness; optional latent-space audits provide the stronger topology-aligned latent smoothing label. Positive graph-prior performance alone is therefore not sufficient evidence of topology-specific structure.

## Figure 3. Preflight classification overview across regimes

The same preflight protocol returns different computational recommendations across representative regimes: skip the prior when graph harms rollout, treat graph gains as generic or temporal smoothing when controls explain the effect, carry the graph prior forward when it beats calibrated controls, and use audit evidence only for the controlled positive case. This figure emphasizes that the contribution is decision support under tested model conditions, not a universal ranking of priors.

## Figure 4. Calibrated prior-family comparison

Calibrated temporal smoothing changes the attribution story. In spring-mass and graph-wave quick runs, temporal smoothing slightly outperforms the graph prior, so graph topology is not needed to explain the low-budget gain. In the n-body quick run, the graph prior remains better than no-prior, spectrum-matched permuted graph, and calibrated temporal smoothing, supporting candidate-topology-specific usefulness under that tested condition.

## Figure 5. Training-budget dependence

Quick-mode graph gains do not necessarily persist under standard training. Spring-mass and graph-wave show graph improvements at 5 epochs but no-prior models are best at 20 epochs; n-body shows a quick graph advantage but a standard-budget permuted-control advantage. The preflight protocol therefore treats budget as part of the model condition rather than an incidental detail.

## Figure 6. Latent audit positive case

The checkpointed HO lattice audit shows the strongest form of preflight support: graph-prior training improves rollout while learned temporal latent deltas are smoother and more low-frequency in the true graph basis than control-prior models. This figure demonstrates what audit-supported topology-aligned latent smoothing looks like, while remaining a controlled positive case rather than a universal mechanism claim.
