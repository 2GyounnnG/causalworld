# JCS Algorithm Box

## Algorithm 1: Model-Conditioned Prior Preflight

**Input:** Node-wise trajectories `X[1:T, 1:N, 1:d]`; candidate Laplacian `L`; model condition `M` including encoder, transition model, latent dimension, optimizer, and fixed training settings; training budget `B`; prior weight or calibrated prior-strength rule; rollout horizons `H`; requested modes from `{quick, standard, audit}`.

**Output:** Preflight report containing rollout metrics, control comparisons, optional latent audit metrics, classification label, and recommended next action.

```text
1. Load dataset adapter
   a. Read trajectories X and candidate Laplacian L.
   b. Record graph source, topology, simulator/data metadata, and whether L is true, approximate, or data-derived.

2. Build prior family set P
   a. Include no-prior baseline: none.
   b. Include candidate graph prior: graph_laplacian(L).
   c. Construct spectrum-matched permuted graph control: permuted_graph(P L P^T).
   d. Optionally construct random graph control with matched coarse graph size statistics.
   e. If requested, include graph-free temporal_smooth prior.

3. Calibrate prior strengths when comparing unlike prior families
   a. Estimate initial auxiliary loss scale for each prior family.
   b. Set effective temporal prior strength so temporal_smooth has matched initial regularization contribution.
   c. Store calibrated weights and initial prior-loss diagnostics.

4. Run raw graph-dynamics diagnostics when available
   a. Compute one-step temporal changes Delta X_t = X_{t+1} - X_t.
   b. Measure D_L(Delta X) for candidate and control graphs.
   c. Measure low-frequency energy ratios in the candidate graph basis.
   d. Treat these diagnostics as supporting evidence, not sufficient decision criteria.

5. Train preflight models under budget B
   For each prior p in P:
      a. Train the fixed model condition M with prior p.
      b. Evaluate rollout error at horizons H.
      c. Save training loss, prior loss, rollout metrics, and run metadata.
      d. If audit mode is enabled, save latent traces or checkpoints.

6. Compare outcomes
   a. Compare graph_laplacian against none.
   b. Compare graph_laplacian against spectrum-matched permuted_graph.
   c. Compare graph_laplacian against calibrated temporal_smooth when available.
   d. Compare quick and standard budgets when both are available.

7. Audit learned latent geometry when traces are available
   a. Compute Delta H_t = H_{t+1} - H_t.
   b. Measure D_L(Delta H) under candidate and control graph bases.
   c. Measure low-frequency energy ratios for Delta H.
   d. Check whether rollout gains coincide with smoother or more low-frequency candidate-graph latent deltas.

8. Assign protocol label
   a. If no tested prior improves over none, label no_prior_gain.
   b. If graph improves over none but not spectrum-matched controls, label graph_generic_smoothing.
   c. If calibrated temporal_smooth matches or beats graph, label temporal_smoothing_sufficient.
   d. If graph beats none, permuted_graph, and calibrated temporal_smooth, label candidate_topology_specific.
   e. If latent audit also supports candidate-graph smoothing, label topology_aligned_latent_smoothing.
   f. If quick-budget gains disappear at standard budget, add low_budget_only.
   g. If a prior harms rollout, label overconstrained.
   h. If required controls or audits are missing, label inconclusive.

9. Recommend next action
   a. Skip priors when no_prior_gain.
   b. Prefer temporal smoothing when temporal_smoothing_sufficient.
   c. Run more seeds, audits, and stronger controls when candidate_topology_specific.
   d. Carry graph prior forward only under the tested model/budget condition unless persistence is shown.
```

## Notes For Manuscript Text

- Step 2c is the main attribution control: the permuted graph is spectrum-matched, so a candidate graph must beat it before topology semantics are invoked.
- Step 3 is necessary for graph-versus-temporal comparison because raw temporal smoothness losses may be much smaller than graph-Laplacian losses.
- Step 7 should be described as audit evidence, not as an assumption of the method.
- Step 8 labels are report-level recommendations, not universal dataset properties.
