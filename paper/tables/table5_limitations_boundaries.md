# Table 5. Limitations and Claim Boundaries

| limitation or boundary | manuscript wording to use | implication |
| --- | --- | --- |
| Model conditioning | Recommendations apply under the tested encoder, transition model, prior placement, prior strength, optimizer, training budget, and rollout horizon. | Changing the model condition requires a new preflight interpretation. |
| Budget conditioning | Quick-mode gains can disappear under standard or longer training. | Quick positives should be validated before full-run claims. |
| Candidate graph ambiguity | Candidate graphs may be true, approximate, constructed, or data-derived. | A useful candidate graph prior does not prove physical truth. |
| Control requirements | Topology claims require beating a spectrum-matched permuted graph and calibrated temporal smoothing. | Graph vs none is insufficient for topology attribution. |
| Temporal calibration | Raw temporal prior losses may be much smaller than graph prior losses. | Temporal baselines should be calibrated before attribution. |
| Raw diagnostics | Raw-coordinate smoothness can disagree with learned latent prior utility. | Raw `D_L(Delta X)` should not be used as a sufficient decision rule. |
| Audit artifacts | Latent audits require saved traces or checkpoints. | Missing artifacts limit mechanism claims. |
| Benchmark scope | Regimes are representative stress tests, not universal physics coverage. | Avoid universal predictor or SOTA benchmark framing. |
| Causal claims | The workflow compares predictive priors and controls. | Do not frame as causal discovery. |
| Prior novelty | The graph Laplacian prior is a tested prior family, not the paper's algorithmic novelty. | The contribution is the preflight protocol and attribution workflow. |

Source: `paper/jcs_limitations.md`, `paper/jcs_draft_methods.md`.
