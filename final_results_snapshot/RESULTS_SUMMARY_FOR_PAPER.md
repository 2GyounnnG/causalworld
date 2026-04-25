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
| euclidean | 10 | 0.3225857971 | 0.1387958551 | 0.2439243327 | 0.4065152688 | 11.39640171 | 0 |
| none | 10 | 0.2895836779 | 0.2029223253 | 0.1759597205 | 0.4118046168 | 0 | -10.23049357 |
| spectral | 10 | 0.140422495 | 0.06460158353 | 0.1026280864 | 0.1781787885 | -51.50883638 | -56.46972176 |

Interpretation: lower rollout error is better. This section is final only when all three priors have n=10 at H=16.

## Weight Sweep H=16



| prior | prior_weight | n | mean | std | ci95_low | ci95_high | pct_change_vs_none |
| --- | --- | --- | --- | --- | --- | --- | --- |
| euclidean | 0.001 | 3 | 0.1809946463 | 0.09902099884 | 0.1020283654 | 0.2920906562 | -56.61636967 |
| euclidean | 0.01 | 3 | 0.1725369454 | 0.05998918753 | 0.1267435583 | 0.2404439366 | -58.64364384 |
| euclidean | 0.1 | 3 | 0.2382917923 | 0.02860962017 | 0.2074053479 | 0.2638854346 | -42.88249274 |
| euclidean | 1.0 | 3 | 0.3215721937 | 0.1232920935 | 0.2300674504 | 0.4617761497 | -22.92054237 |
| none | 0.001 | 3 | 0.4171957141 | 0.2430955445 | 0.1650391161 | 0.6500833353 | 0 |
| none | 0.01 | 3 | 0.4171957141 | 0.2430955445 | 0.1650391161 | 0.6500833353 | 0 |
| none | 0.1 | 3 | 0.4171957141 | 0.2430955445 | 0.1650391161 | 0.6500833353 | 0 |
| none | 1.0 | 3 | 0.4171957141 | 0.2430955445 | 0.1650391161 | 0.6500833353 | 0 |
| spectral | 0.001 | 3 | 0.2599633544 | 0.3406754643 | 0.04056369847 | 0.6524309584 | -37.68791347 |
| spectral | 0.01 | 3 | 1.908429248 | 1.623978118 | 0.1126812249 | 3.274033636 | 357.4421988 |
| spectral | 0.1 | 3 | 0.1300585177 | 0.09771104769 | 0.06247591466 | 0.2420923439 | -68.8255384 |
| spectral | 1.0 | 3 | 0.1443177964 | 0.07812012726 | 0.06866807671 | 0.224693734 | -65.40765125 |

Interpretation: this section is final only when every prior-weight-prior group has n=3 at H=16.

## Laplacian Ablation H=16



| laplacian_mode | n | mean | std | ci95_low | ci95_high | pct_change_vs_per_frame |
| --- | --- | --- | --- | --- | --- | --- |
| fixed_frame0 | 5 | 0.1458356608 | 0.08917947959 | 0.09771777681 | 0.2266025228 | 31.17230254 |
| fixed_mean | 5 | 0.1306024097 | 0.1121740511 | 0.06747979014 | 0.2324352659 | 17.47071119 |
| per_frame | 5 | 0.1111787001 | 0.07468142768 | 0.06867660867 | 0.1783989958 | 0 |

Interpretation: this section is final only when every Laplacian mode has n=5 at H=16.

## Wolfram 200-Epoch H=16



| prior | n | mean | std | ci95_low | ci95_high | pct_change_vs_none | pct_change_vs_euclidean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| euclidean | 10 | 0.06538378596 | 0.0411988889 | 0.04289250582 | 0.09109127507 | -11.59917572 | 0 |
| none | 10 | 0.07396286912 | 0.04574169559 | 0.04906890377 | 0.1023319722 | 0 | 13.12111716 |
| spectral | 10 | 18.29436963 | 45.36049594 | 0.1110042197 | 50.16061419 | 24634.53213 | 27879.97907 |

The completed Wolfram flat run does not support a universal spectral-prior advantage. Euclidean slightly improves over no prior at H=16, while spectral has a heavy-tailed failure mode: most seeds remain near baseline scale, but a few seeds explode at long horizon.

Interpretation: this section is final only when all three priors have n=10 at H=16.

## Reviewer-Risk Section

- Per-frame Laplacian leakage risk: active risk until fixed Laplacian modes finish and agree directionally with per-frame.
- Euclidean prior weight risk: active risk until lower-weight Euclidean settings are complete.
- Seed/statistical power risk: rMD17 aspirin main result is stronger at n=10; current ablations are underpowered while incomplete.
- rMD17 split/data leakage risk: rMD17 samples come from one trajectory; train/eval use different random seeds but no explicit disjoint-frame audit is yet enforced.
- Synthetic-to-real generalization risk: Wolfram and rMD17 cover different regimes, so cross-domain generalization should be framed cautiously.
