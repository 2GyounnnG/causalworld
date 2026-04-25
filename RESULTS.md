# Results

## Executive Summary

The final thesis is that latent prior geometry is domain-specific and stability-dependent. Spectral priors strongly improve rMD17 aspirin when the graph prior is aligned with molecular relational dynamics, but spectral priors are not universally better. In the completed Wolfram flat 200-epoch comparison, the Euclidean prior has the best H=16 mean, while the spectral prior exhibits heavy-tailed long-horizon instability.

Across the completed analyses, the strongest positive result is the rMD17 aspirin 10-seed comparison: at H=16, the spectral prior reduces rollout error by 51.5% relative to no prior and by 56.5% relative to the Euclidean prior. The supporting ablations show that fixed Laplacians still outperform the no-prior baseline, which argues against the entire gain being an artifact of per-frame Laplacian leakage. The boundary case is Wolfram flat: spectral is not uniformly bad there, but a small number of seeds explode at long horizon, making the mean unacceptable despite many seeds remaining near baseline scale.

## Experiment Inventory

The finalized analysis includes four completed quantitative comparisons:

| experiment | status | key role |
| --- | --- | --- |
| rMD17 aspirin 10-seed main comparison | complete | Main molecular result comparing none, Euclidean, and spectral priors. |
| rMD17 Laplacian ablation | complete | Tests whether spectral gains persist under fixed Laplacian variants. |
| rMD17 prior-weight sweep | complete | Separates prior family effects from weight tuning and identifies instability regimes. |
| Wolfram flat 10-seed 200-epoch comparison | complete | Synthetic boundary case for cross-domain behavior. |

Audit and hygiene status:

- Prior/Laplacian audit: 19 PASS, 2 WARNING, 0 FAIL, 0 UNKNOWN.
- Remaining audit warnings are reviewer-facing rMD17 train/eval frame-index overlap risks.
- Manifest hygiene is clean after dedupe: `manifest_raw.csv` has 1737 rows, `manifest.csv` has 1442 rows, 295 duplicate rollout rows were recorded in `duplicate_manifest_rows.csv`, and duplicate experimental keys after dedupe are 0.
- Old rMD17 aspirin 10-seed runs did not save checkpoints, so post hoc disjoint-frame re-evaluation is blocked for those exact trained models.

## rMD17 Aspirin Main Result

At H=16, the completed 10-seed rMD17 aspirin comparison is:

| prior | n | H=16 mean |
| --- | ---: | ---: |
| none | 10 | 0.2895836779 |
| euclidean | 10 | 0.3225857971 |
| spectral | 10 | 0.140422495 |

Lower rollout error is better. The spectral prior improves over the no-prior baseline by 51.5% and over the Euclidean prior by 56.5%. This is the central positive result: for aspirin dynamics, the spectral graph-structured prior is a strong inductive bias for long-horizon rollout.

The default Euclidean setting is worse than no prior in this main 10-seed comparison. That should not be overinterpreted in isolation, because the weight sweep shows that Euclidean regularization is not intrinsically harmful when tuned.

## Laplacian Ablation

At H=16, the completed spectral Laplacian ablation is:

| Laplacian mode | n | H=16 mean |
| --- | ---: | ---: |
| per_frame | 5 | 0.1111787001 |
| fixed_mean | 5 | 0.1306024097 |
| fixed_frame0 | 5 | 0.1458356608 |

The per-frame Laplacian is best. However, both fixed Laplacian variants still beat the no-prior rMD17 aspirin H=16 baseline of 0.2895836779. This matters because it weakens the explanation that the spectral gain is solely a per-frame Laplacian leakage artifact. Per-frame information may still be a reviewer concern, but the fixed-mode results show that spectral structure remains useful without recomputing the Laplacian from each current sample.

## Weight Sweep

The completed rMD17 aspirin weight sweep clarifies the role of prior strength. The best H=16 settings are:

| family | best weight | H=16 mean |
| --- | ---: | ---: |
| none | 0.001 | 0.4171957141 |
| euclidean | 0.01 | 0.1725369454 |
| spectral | 0.1 | 0.1300585177 |

Euclidean regularization is not intrinsically harmful: tuned Euclidean substantially improves over the no-prior baseline in this sweep. At the same time, the best spectral setting improves over the best Euclidean setting by 24.6%, supporting a spectral advantage when the prior weight is chosen well.

The sweep also reveals an instability regime. Spectral `w=0.01` is long-horizon unstable:

| prior | weight | seed | H=16 | H16/H1 |
| --- | ---: | ---: | ---: | ---: |
| spectral | 0.01 | 1 | 3.274033636 | 68.58477396 |
| spectral | 0.01 | 2 | 2.338572881 | 51.77253236 |

Thus the conclusion is not simply "spectral beats Euclidean." The more precise conclusion is that spectral is a high-gain prior: it can deliver the best long-horizon performance, but it can also destabilize rollout at some weights.

## Wolfram Flat 200-Epoch Boundary Case

The completed Wolfram flat 10-seed 200-epoch comparison gives a boundary case where the spectral prior does not support a universal advantage:

| prior/statistic | H=16 |
| --- | ---: |
| euclidean mean | 0.06538378596 |
| none mean | 0.07396286912 |
| spectral mean | 18.29436963 |
| spectral median | 0.1304059513 |

Euclidean has the best mean at H=16, slightly improving over no prior. Spectral has heavy-tailed instability: the mean is dominated by a few exploding seeds, while the median remains much closer to baseline scale.

The spectral H=16 failures are:

| seed | H=16 | H16/H1 |
| ---: | ---: | ---: |
| 2 | 143.5640717 | 48936.57456 |
| 8 | 35.24406433 | 8828.532057 |
| 1 | 3.485707521 | 946.2799087 |

Most spectral seeds remain near baseline scale, so the failure is heavy-tailed rather than uniform. This is important for interpretation: Wolfram flat does not show that spectral priors are always bad, but it does show that spectral priors can become unstable when geometry, dynamics, or optimization basin are misaligned.

## Interpretation

The consistent pattern is that spectral priors behave like a high-gain inductive bias. When graph geometry aligns with the latent dynamics, as in the rMD17 aspirin setting, the spectral prior can substantially improve long-horizon rollout. When that geometry or the optimization basin is less well aligned, as in the Wolfram flat boundary case or unstable weight settings, spectral regularization can amplify errors and destabilize rollout.

This also explains why long-horizon evaluation is essential. H=1 can look reasonable even when H=16 explodes. The relevant failure mode is not necessarily short-step prediction error; it is compounding latent rollout instability. Any claim about latent prior geometry should therefore be evaluated across horizon, not only at one-step prediction.

## Limitations

The old completed rMD17 aspirin 10-seed run lacks model checkpoints, so post hoc disjoint-frame evaluation is blocked for those exact trained models. Future checkpointed reruns are instrumented, but they have not been run as part of this result set.

rMD17 train/eval frame overlap remains a reviewer-facing warning. The audit reconstructs overlapping sampled frame indices, so the current rMD17 result should be presented as valid under the current evaluation protocol, with disjoint-frame replication listed as future work.

The Wolfram flat experiment is a synthetic boundary case, not a molecular dynamics benchmark. Its role is to test whether the spectral prior generalizes universally; it does not. More molecules, more graph regimes, and checkpointed disjoint reruns are needed before making broad claims about spectral priors across domains.

## Paper-Ready Claims

Supported claims:

- Latent prior geometry is domain-specific and stability-dependent.
- Spectral priors strongly improve rMD17 aspirin long-horizon rollout under the current evaluation protocol.
- In rMD17 aspirin, fixed Laplacian spectral variants still outperform the no-prior baseline, so the gain is not solely explained by per-frame Laplacian recomputation.
- Euclidean regularization is not intrinsically harmful; it improves when tuned in the weight sweep.
- The best tuned spectral setting in the rMD17 weight sweep outperforms the best tuned Euclidean setting.
- Wolfram flat 200-epoch does not support a universal spectral-prior advantage; Euclidean has the best H=16 mean, and spectral shows heavy-tailed long-horizon instability.
- Long-horizon rollout metrics are necessary because short-horizon behavior can mask compounding instability.

Claims not to make:

- Do not claim spectral universally beats Euclidean.
- Do not claim LeJEPA is refuted.
- Do not claim per-frame leakage is impossible.
- Do not claim Wolfram supports spectral advantage.
- Do not claim the old rMD17 10-seed results support post hoc disjoint-frame evaluation, because the checkpoints were not saved.

