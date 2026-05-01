# Figure 1 Draft. Model-Conditioned Preflight Workflow

```text
Node-wise trajectories X[1:T, 1:N, d]
Candidate prior information
  - candidate graph or Laplacian L
  - optional graph-free temporal prior
  - dataset and graph metadata
        |
        v
Raw diagnostic stage
  - D_L(Delta X)
  - low-frequency ratios
  - graph source and scale checks
  Note: diagnostic only, not sufficient evidence
        |
        v
Quick preflight
  - fixed model condition
  - short training budget
  - rollout horizons H
        |
        v
Control comparisons
  - none
  - graph_laplacian(L)
  - permuted_graph(P^T L P)
  - optional random_graph
  - calibrated temporal_smooth
        |
        v
Decision branch
  +----------------------+-------------------------+
  | no gain or harm      | skip prior              |
  | temporal sufficient  | use temporal smoothing  |
  | graph beats controls | run stronger checks     |
  | quick-only gain      | mark low_budget_only    |
  +----------------------+-------------------------+
        |
        v
Standard / audit mode when warranted
  - standard budget persistence check
  - latent audit: D_L(Delta H), low-frequency ratios
        |
        v
Protocol label
  - no_prior_gain
  - graph_generic_smoothing
  - temporal_smoothing_sufficient
  - candidate_topology_specific
  - topology_aligned_latent_smoothing
  - low_budget_only
  - overconstrained
  - inconclusive
        |
        v
Actionable recommendation before full training
  - skip prior
  - use temporal prior
  - continue graph prior with audits/seeds
  - add controls before topology claims
```

Message: the protocol is a staged computational screen that reduces wasted full training and prevents false topology attribution by requiring controls before stronger claims.
