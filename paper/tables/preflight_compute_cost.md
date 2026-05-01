# Preflight Compute Cost Accounting

Created: `2026-05-01T08:51:04.749901+00:00`

Cost unit: one training epoch over one sampled transition. Estimated cost is `number of trainings x epochs x train_transitions`.
This is a relative accounting table for reviewer-facing scale claims; it is not a wall-clock benchmark.

| regime | run dirs | trainings | audited runs | transition-epoch units |
| --- | --- | --- | --- | --- |
| quick preflight | 10 | 34 | 12 | 18,336 |
| standard check | 3 | 9 | 4 | 17,280 |

Hypothetical full sweep comparator:

| datasets | priors | lambdas | seeds | epochs | train transitions | transition-epoch units |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 4 | 5 | 5 | 100 | 512 | 20,480,000 |

Observed quick-preflight cost is `0.09%` of the hypothetical full sweep under the assumptions above.
All observed/prepared preflight directories together account for `0.17%` of that full-sweep estimate.

Protocol interpretation:
- quick mode is triage: it screens whether a prior is worth further attention under a deliberately small budget.
- standard mode is a persistence check: it asks whether the quick signal survives a larger training budget.
- audit mode is a mechanism check: it inspects latent graph-frequency behavior rather than treating rollout gain as sufficient evidence.
- quick candidate topology signal is not final topology-specific evidence.

Input discovery:
- scanned `analysis_out/preflight_runs` for `run_config.json` files
- output written to `paper/tables/preflight_compute_cost.md`
