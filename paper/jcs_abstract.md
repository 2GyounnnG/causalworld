# JCS Abstract Draft

## Proposed Title

Model-Conditioned Preflight for Prior Selection and Attribution in Node-Wise Scientific Latent Dynamics

## Abstract

Scientific latent dynamics models are often trained with auxiliary priors, such as graph Laplacian smoothness, temporal smoothness, or covariance-style regularization, but it is rarely clear before a full training campaign whether a prior will improve rollout prediction or whether an observed gain is attributable to the intended scientific structure. We present a computational preflight protocol for model-conditioned prior selection and attribution in node-wise physical and scientific prediction problems. Given trajectories, a candidate graph, a fixed latent dynamics model, and a training budget, the protocol compares no-prior training, candidate graph Laplacian regularization, spectrum-matched permuted graph controls, optional random graph controls, and calibrated graph-free temporal smoothness baselines.

The protocol is organized into quick, standard, and audit modes. Quick mode screens low-budget prior utility; standard mode tests whether early gains persist after more training; audit mode inspects whether learned latent temporal deltas become smoother in the candidate graph basis. The resulting labels distinguish no prior gain, generic graph smoothing, temporal smoothing sufficiency, candidate-topology-specific utility, topology-aligned latent smoothing, low-budget-only effects, and overconstraint.

Across existing runs spanning coupled oscillators, spring mechanics, graph waves, graph heat diffusion, low-frequency graph dynamics, n-body interactions, molecular dynamics summaries, and traffic forecasting with a correlation graph, the protocol identifies a recurring pattern: graph priors can improve low-budget rollout, but calibrated temporal smoothing or spectrum-matched graph controls often explain the gain, and longer training can allow no-prior or control models to catch up. The n-body quick run remains candidate-topology-specific against both calibrated temporal smoothing and a permuted graph, while spring-mass and graph-wave quick gains are explained by calibrated temporal smoothing. A checkpointed harmonic-oscillator lattice audit provides a controlled positive case in which graph-prior training improves rollout and produces smoother, more low-frequency latent temporal deltas on the true graph.

The contribution is not a new graph prior, universal physics predictor, causal discovery method, or state-of-the-art benchmark. Instead, the work provides a practical computational workflow for deciding which priors deserve full-scale training, which controls are needed for attribution, and when topology-oriented claims should be avoided.
