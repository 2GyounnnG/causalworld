# N-body Robustness 5-seed Summary

Mean +/- sample std across seeds 0,1,2,3,4. 95% CIs use t=2.776 by default.

## Paper-lift Summary

| k | budget | label | label_confidence | graph H32 (mean +/- std) | strongest_comparator | comparator H32 (mean +/- std) |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | quick_ep5 | quick_topology_signal | marginal | 1.0176 +/- 0.7199 | permuted_graph | 1.0604 +/- 0.8055 |
| 4 | standard_ep20 | candidate_graph_favorable | marginal | 0.0641 +/- 0.0322 | temporal_smooth | 0.0661 +/- 0.0147 |
| 8 | quick_ep5 | quick_topology_signal | marginal | 0.9689 +/- 0.6970 | permuted_graph | 1.0289 +/- 0.7698 |
| 8 | standard_ep20 | generic_smoothing | marginal | 0.0814 +/- 0.0332 | temporal_smooth | 0.0695 +/- 0.0240 |
| 12 | quick_ep5 | quick_topology_signal | marginal | 0.9365 +/- 0.6715 | permuted_graph | 0.9489 +/- 0.7035 |
| 12 | standard_ep20 | temporal_sufficient | marginal | 0.0772 +/- 0.0311 | temporal_smooth | 0.0596 +/- 0.0265 |

## H32 Rollout Error

| k | budget | label | label_confidence | none_h32 | graph_h32 | permuted_graph_h32 | temporal_smooth_h32 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | quick_ep5 | quick_topology_signal | marginal | 1.1553 +/- 0.7845 | 1.0176 +/- 0.7199 | 1.0604 +/- 0.8055 | 1.0624 +/- 0.7224 |
| 4 | standard_ep20 | candidate_graph_favorable | marginal | 0.1507 +/- 0.0129 | 0.0641 +/- 0.0322 | 0.0718 +/- 0.0245 | 0.0661 +/- 0.0147 |
| 8 | quick_ep5 | quick_topology_signal | marginal | 1.1607 +/- 0.7998 | 0.9689 +/- 0.6970 | 1.0289 +/- 0.7698 | 1.0589 +/- 0.7744 |
| 8 | standard_ep20 | generic_smoothing | marginal | 0.1522 +/- 0.0115 | 0.0814 +/- 0.0332 | 0.0759 +/- 0.0276 | 0.0695 +/- 0.0240 |
| 12 | quick_ep5 | quick_topology_signal | marginal | 1.1671 +/- 0.7792 | 0.9365 +/- 0.6715 | 0.9489 +/- 0.7035 | 1.0144 +/- 0.7576 |
| 12 | standard_ep20 | temporal_sufficient | marginal | 0.1443 +/- 0.0121 | 0.0772 +/- 0.0311 | 0.0898 +/- 0.0417 | 0.0596 +/- 0.0265 |

## H32 Gains (%)

| k | budget | label | label_confidence | graph_gain_h32_pct | true_vs_permuted_gain_h32_pct | graph_vs_temporal_gain_h32_pct |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | quick_ep5 | quick_topology_signal | marginal | -12.2313 +/- 112.3675 | 0.4373 +/- 10.9231 | 3.9432 +/- 20.1620 |
| 4 | standard_ep20 | candidate_graph_favorable | marginal | 56.6604 +/- 22.7737 | 14.4368 +/- 23.4284 | 5.6509 +/- 39.8652 |
| 8 | quick_ep5 | quick_topology_signal | marginal | -2.2729 +/- 94.1410 | 3.7999 +/- 5.2167 | 6.9384 +/- 8.9463 |
| 8 | standard_ep20 | generic_smoothing | marginal | 46.8745 +/- 20.4838 | -6.3247 +/- 14.4813 | -21.1295 +/- 46.1301 |
| 12 | quick_ep5 | quick_topology_signal | marginal | 4.3464 +/- 81.9311 | 0.1292 +/- 7.1552 | 5.5335 +/- 4.8500 |
| 12 | standard_ep20 | temporal_sufficient | marginal | 46.9133 +/- 19.9293 | 11.8712 +/- 15.4772 | -44.0412 +/- 67.3032 |

## Seed Labels and Warnings

| k | budget | seed-level label counts | warning_flag | warning_reasons |
| --- | --- | --- | --- | --- |
| 4 | quick_ep5 | quick_topology_signal 3/5, no_graph_gain 1/5, generic_smoothing 1/5 | ok |  |
| 4 | standard_ep20 | candidate_graph_favorable 3/5, generic_smoothing 1/5, temporal_sufficient 1/5 | ok |  |
| 8 | quick_ep5 | quick_topology_signal 2/5, no_graph_gain 1/5, temporal_sufficient 1/5, generic_smoothing 1/5 | ok |  |
| 8 | standard_ep20 | generic_smoothing 4/5, candidate_graph_favorable 1/5 | ok |  |
| 12 | quick_ep5 | quick_topology_signal 2/5, generic_smoothing 2/5, no_graph_gain 1/5 | ok |  |
| 12 | standard_ep20 | candidate_graph_favorable 2/5, generic_smoothing 2/5, temporal_sufficient 1/5 | ok |  |
