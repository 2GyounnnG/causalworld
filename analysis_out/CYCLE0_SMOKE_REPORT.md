# Cycle 0 Smoke Report

Results file: `experiments/results/cycle0_aspirin_smoke/cycle0_aspirin_smoke_results.json`
Schema version: `cycle0_smoke_v1`
Runs observed: 7 / 7

## Required Questions

- Did all runs finish without NaN? YES
- Does graph prior use node-wise H^T L H? YES
- Are MLP graph priors disabled unless mathematically defined? YES
- Are all diagnostics saved? YES
- Is the JSON schema stable? YES
- Is Cycle 1 ready? YES

## Run Table

| run | encoder | prior | status | final_loss | prior_loss_mean | H=1 | H=32 |
|---|---|---|---|---:|---:|---:|---:|
| gnn_graph | gnn_node | graph | ok | 0.0159044 | 0.600803 | 0.0124026 | 0.516382 |
| gnn_none | gnn_node | none | ok | 0.00108359 | 0 | 0.0310864 | 1.12398 |
| gnn_permuted_graph | gnn_node | permuted_graph | ok | 0.0284563 | 1.14722 | 0.0128693 | 0.503654 |
| gnn_random_graph | gnn_node | random_graph | ok | 0.0285241 | 1.14622 | 0.0128342 | 0.504434 |
| mlp_covariance | mlp_global | covariance | ok | 0.402492 | 3.99244 | 0.0514824 | 0.658303 |
| mlp_none | mlp_global | none | ok | 0.00247201 | 0 | 0.0498774 | 0.964836 |
| mlp_sigreg | mlp_global | sigreg | ok | 0.176956 | 1.85825 | 0.0857571 | 0.511103 |

## Notes

- Missing runs: none
- Failed runs: none
- Non-finite diagnostic runs: none
- Runs missing diagnostics: none
- Graph stationarity is explicitly marked unavailable in this smoke path.
- Cycle 1 should not start from this script; this report only gates readiness.
