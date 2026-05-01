# Table 1. Protocol Labels and Recommended Actions

| label | evidence pattern | recommended action | claim allowed | claim boundary |
| --- | --- | --- | --- | --- |
| `no_prior_gain` | No tested prior improves meaningfully over the no-prior model. | Skip the prior for larger runs unless there is a separate scientific reason to test it. | The tested prior was not useful under the model condition. | Does not imply the prior is useless for all models or budgets. |
| `graph_generic_smoothing` | Graph prior improves over no-prior but does not beat spectrum-matched or other graph smoothing controls. | Treat the result as smoothing-like; test cheaper or simpler smoothness controls. | Graph-style regularization helped prediction. | Do not claim node-label topology specificity. |
| `temporal_smoothing_sufficient` | Calibrated temporal smoothing matches or beats the graph prior. | Prefer temporal smoothing unless graph attribution is itself the target. | Graph-free temporal smoothness explains the observed gain. | Do not claim graph topology is necessary. |
| `candidate_topology_specific` | Candidate graph beats no-prior, spectrum-matched permuted graph, and calibrated temporal smoothing under the tested budget. | Carry graph prior forward with more seeds, audits, and stronger controls. | The candidate graph was useful beyond tested smoothing controls. | Does not prove the graph is the true physical interaction graph. |
| `topology_aligned_latent_smoothing` | Rollout control wins coincide with smoother or more low-frequency learned latent temporal deltas in the candidate graph basis. | Use as the strongest preflight support for graph-prior continuation. | Audit evidence supports topology-aligned latent smoothing. | Still model-, weight-, and budget-conditioned. |
| `low_budget_only` | A prior helps in quick mode but loses the advantage under standard or longer training. | Treat the prior as a low-budget regularizer, not a persistent structural advantage. | The prior can help under constrained training. | Do not extrapolate to full-budget training. |
| `overconstrained` | Prior harms rollout relative to no-prior and controls. | Reduce strength, revise prior placement, or skip the prior. | The tested prior condition suppressed useful predictive variation. | Does not rule out weaker or differently placed priors. |
| `inconclusive` | Required controls, seeds, or audit artifacts are insufficient. | Run missing controls or avoid stronger claims. | Evidence is insufficient for a stronger label. | Do not over-interpret incomplete comparisons. |

Source: `paper/model_conditioned_preflight_protocol.md`, `paper/jcs_draft_methods.md`.
