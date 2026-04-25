# Results Summary For Paper

## Claim Status

- rMD17 aspirin 10-seed main comparison: CONFIRMED; spectral has lower H=16 rollout error than both none and euclidean.
- rMD17 weight sweep: CONFIRMED.
- rMD17 Laplacian ablation: CONFIRMED.
- Wolfram flat 200-epoch comparison: COMPLETE; Euclidean has the best H=16 mean; spectral shows long-horizon instability.

## Main Confirmed Claims

- The rMD17 aspirin spectral-prior advantage over both no prior and the Euclidean isotropy prior is supported by the completed H=16 table.
- This is a different regime from LeJEPA: known downstream rollout prediction, structured relational data, and temporally correlated samples.

## Claims Still Pending

- No aggregate-completeness blockers remain for the reported rMD17 weight sweep, Laplacian ablation, or Wolfram flat 200-epoch tables.

## Claims That Should Not Be Made Yet

- Do not claim spectral priors are universally optimal.
- Do not claim the Laplacian ablation rules out per-sample information leakage until all fixed-mode seeds finish.
- Do not say "we refute LeJEPA."

## rMD17 Aspirin 10-Seed H=16



| prior | n | mean | std | ci95_low | ci95_high | pct_change_vs_none | pct_change_vs_euclidean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| euclidean | 10 | 0.323 | 0.139 | 0.244 | 0.407 | 11.4 | 0.0 |
| none | 10 | 0.290 | 0.203 | 0.176 | 0.412 | 0.0 | -10.2 |
| spectral | 10 | 0.140 | 0.065 | 0.103 | 0.178 | -51.5 | -56.5 |

Interpretation: lower rollout error is better. This section is final only when all three priors have n=10 at H=16.

## Weight Sweep H=16



| prior | prior_weight | n | mean | std | ci95_low | ci95_high | pct_change_vs_none |
| --- | --- | --- | --- | --- | --- | --- | --- |
| euclidean | 0.001 | 3 | 0.181 | 0.099 | 0.102 | 0.292 | -56.6 |
| euclidean | 0.01 | 3 | 0.173 | 0.060 | 0.127 | 0.240 | -58.6 |
| euclidean | 0.1 | 3 | 0.238 | 0.029 | 0.207 | 0.264 | -42.9 |
| euclidean | 1.0 | 3 | 0.322 | 0.123 | 0.230 | 0.462 | -22.9 |
| none | 0.001 | 3 | 0.417 | 0.243 | 0.165 | 0.650 | 0.0 |
| none | 0.01 | 3 | 0.417 | 0.243 | 0.165 | 0.650 | 0.0 |
| none | 0.1 | 3 | 0.417 | 0.243 | 0.165 | 0.650 | 0.0 |
| none | 1.0 | 3 | 0.417 | 0.243 | 0.165 | 0.650 | 0.0 |
| spectral | 0.001 | 3 | 0.260 | 0.341 | 0.041 | 0.652 | -37.7 |
| spectral | 0.01 | 3 | 1.908 | 1.624 | 0.113 | 3.274 | 357.4 |
| spectral | 0.1 | 3 | 0.130 | 0.098 | 0.062 | 0.242 | -68.8 |
| spectral | 1.0 | 3 | 0.144 | 0.078 | 0.069 | 0.225 | -65.4 |

Interpretation: this section is final only when every prior-weight-prior group has n=3 at H=16.

## Laplacian Ablation H=16



| laplacian_mode | n | mean | std | ci95_low | ci95_high | pct_change_vs_per_frame |
| --- | --- | --- | --- | --- | --- | --- |
| fixed_frame0 | 5 | 0.146 | 0.089 | 0.098 | 0.227 | 31.2 |
| fixed_mean | 5 | 0.131 | 0.112 | 0.067 | 0.232 | 17.5 |
| per_frame | 5 | 0.111 | 0.075 | 0.069 | 0.178 | 0.0 |

Interpretation: this section is final only when every Laplacian mode has n=5 at H=16.

## Wolfram 200-Epoch H=16



| prior | n | mean | std | ci95_low | ci95_high | pct_change_vs_none | pct_change_vs_euclidean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| euclidean | 10 | 0.065 | 0.041 | 0.043 | 0.091 | -11.6 | 0.0 |
| none | 10 | 0.074 | 0.046 | 0.049 | 0.102 | 0.0 | 13.1 |
| spectral | 10 | 18.3 | 45.4 | 0.111 | 50.2 | 2.5e+04 | 2.8e+04 |

The completed Wolfram flat run does not support a universal spectral-prior advantage. Euclidean slightly improves over no prior at H=16, while spectral has a heavy-tailed failure mode: most seeds remain near baseline scale, but a few seeds explode at long horizon.

Interpretation: this section is final only when all three priors have n=10 at H=16.

## Reviewer-Risk Section

- Per-frame Laplacian leakage risk: active risk until fixed Laplacian modes finish and agree directionally with per-frame.
- Euclidean prior weight risk: active risk until lower-weight Euclidean settings are complete.
- Seed/statistical power risk: rMD17 aspirin main result is stronger at n=10; current ablations are underpowered while incomplete.
- rMD17 split/data leakage risk: rMD17 samples come from one trajectory; train/eval use different random seeds but no explicit disjoint-frame audit is yet enforced.
- Synthetic-to-real generalization risk: Wolfram and rMD17 cover different regimes, so cross-domain generalization should be framed cautiously.
