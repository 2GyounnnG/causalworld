# Master Decision Rules

These rules are for interpreting Cycles 1-5 without launching new experiments.

## Rollout Smoothing Claim

Claim graph-style smoothing when graph-prior H=32 rollout improves over GNN none by at least 20% and graph controls also improve. This supports a regularization/smoothing interpretation, not exact topology specificity.

## Covariance Explanation

Treat covariance conditioning as insufficient when covariance is worse than or close to GNN none while graph-style priors strongly improve H=32 rollout. Cycle 1 aspirin and Cycle 3 HO both meet this condition.

## Exact Graph Specificity

Require all of the following at H=32 before calling a setting graph-specific:

- graph mean error is lower than both permuted_graph and random_graph;
- paired mean deltas, control minus graph, are positive for both controls;
- graph wins at least 80% of paired seeds, rounded up;
- bootstrap 95% CI lower bound is above zero for both controls;
- the effect is not contradicted at nearby lambda values if the claim is about method-level robustness.

## Scale-Matched Controls

Permuted_graph is the decisive scale-matched control. Its Laplacian spectrum matches the true graph, so permuted-control wins or ties are evidence against exact node-label/topology semantics, not evidence of a Laplacian-scale mismatch.

Random_graph is useful but weaker for scale attribution: it matches broad size/edge-count/trace constraints in these experiments, but not degree dispersion, Frobenius norm, lambda_max, or spectral gap.

## Lambda Robustness

Claim lambda robustness only when specificity passes the exact-graph criteria across adjacent tested prior weights. Cycle 5 lattice passes at lambda=0.1 and fails at 0.001, 0.005, 0.01, and 0.05, so it is high-lambda-only rather than robust.

## Paper Claim Boundary

Recommended main claim: graph-style node-wise regularization improves long-horizon rollout through generic graph smoothing.

Allowed qualified claim: controlled HO lattice shows true topology specificity at high prior weight lambda=0.1.

Avoid claiming: exact molecular graph specificity on rMD17, lambda-robust topology specificity, or covariance conditioning as the explanation for graph-style gains.
