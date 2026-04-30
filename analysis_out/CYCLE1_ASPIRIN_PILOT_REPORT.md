# Cycle 1 Aspirin Pilot Report

Results file: `experiments/results/cycle1_aspirin_pilot/cycle1_aspirin_pilot_results.json`
Schema version: `cycle1_aspirin_pilot_v1`
Success/failure count: 55 ok / 0 failed
Missing expected runs: none
Runs with non-finite diagnostics: none
Graph prior metadata node-wise: YES

## Rollout Error Mean +/- Std

| encoder | prior | H=1 | H=2 | H=4 | H=8 | H=16 | H=32 |
|---|---|---:|---:|---:|---:|---:|---:|
| mlp_global | none | 0.0245 +/- 0.0066 (n=5) | 0.0257 +/- 0.0072 (n=5) | 0.0289 +/- 0.0071 (n=5) | 0.0414 +/- 0.0105 (n=5) | 0.0798 +/- 0.0168 (n=5) | 0.2283 +/- 0.0992 (n=5) |
| mlp_global | variance | 0.2684 +/- 0.0912 (n=5) | 0.2613 +/- 0.0894 (n=5) | 0.2892 +/- 0.0975 (n=5) | 0.3134 +/- 0.1204 (n=5) | 0.4276 +/- 0.1788 (n=5) | 0.7630 +/- 0.4957 (n=5) |
| mlp_global | covariance | 0.0339 +/- 0.0102 (n=5) | 0.0358 +/- 0.0121 (n=5) | 0.0400 +/- 0.0126 (n=5) | 0.0534 +/- 0.0165 (n=5) | 0.1016 +/- 0.0351 (n=5) | 0.2595 +/- 0.1092 (n=5) |
| mlp_global | sigreg | 0.0854 +/- 0.0124 (n=5) | 0.0846 +/- 0.0118 (n=5) | 0.0878 +/- 0.0112 (n=5) | 0.1010 +/- 0.0170 (n=5) | 0.1539 +/- 0.0525 (n=5) | 0.3788 +/- 0.2163 (n=5) |
| gnn_node | none | 0.0093 +/- 0.0020 (n=5) | 0.0092 +/- 0.0012 (n=5) | 0.0100 +/- 0.0017 (n=5) | 0.0146 +/- 0.0034 (n=5) | 0.0299 +/- 0.0056 (n=5) | 0.1008 +/- 0.0222 (n=5) |
| gnn_node | variance | 0.0842 +/- 0.0179 (n=5) | 0.0779 +/- 0.0158 (n=5) | 0.0858 +/- 0.0128 (n=5) | 0.1046 +/- 0.0192 (n=5) | 0.1595 +/- 0.0507 (n=5) | 0.3336 +/- 0.1698 (n=5) |
| gnn_node | covariance | 0.0109 +/- 0.0026 (n=5) | 0.0105 +/- 0.0017 (n=5) | 0.0116 +/- 0.0022 (n=5) | 0.0168 +/- 0.0040 (n=5) | 0.0344 +/- 0.0056 (n=5) | 0.1192 +/- 0.0207 (n=5) |
| gnn_node | sigreg | 0.0296 +/- 0.0015 (n=5) | 0.0275 +/- 0.0025 (n=5) | 0.0291 +/- 0.0021 (n=5) | 0.0337 +/- 0.0042 (n=5) | 0.0487 +/- 0.0098 (n=5) | 0.1101 +/- 0.0458 (n=5) |
| gnn_node | graph | 0.0033 +/- 0.0005 (n=5) | 0.0034 +/- 0.0003 (n=5) | 0.0040 +/- 0.0008 (n=5) | 0.0055 +/- 0.0009 (n=5) | 0.0106 +/- 0.0027 (n=5) | 0.0330 +/- 0.0157 (n=5) |
| gnn_node | permuted_graph | 0.0034 +/- 0.0004 (n=5) | 0.0036 +/- 0.0004 (n=5) | 0.0042 +/- 0.0008 (n=5) | 0.0059 +/- 0.0009 (n=5) | 0.0114 +/- 0.0024 (n=5) | 0.0350 +/- 0.0160 (n=5) |
| gnn_node | random_graph | 0.0034 +/- 0.0004 (n=5) | 0.0036 +/- 0.0004 (n=5) | 0.0042 +/- 0.0008 (n=5) | 0.0059 +/- 0.0008 (n=5) | 0.0113 +/- 0.0021 (n=5) | 0.0336 +/- 0.0149 (n=5) |

## Diagnostics Mean +/- Std

| encoder | prior | effective_rank | condition_number | projection_gaussianity | prior_loss_mean |
|---|---|---:|---:|---:|---:|
| mlp_global | none | 5.9479 +/- 1.7383 (n=5) | 623.9470 +/- 346.9708 (n=5) | 2.2239 +/- 0.6376 (n=5) | 0.0000 +/- 0.0000 (n=5) |
| mlp_global | variance | 1.5332 +/- 0.5006 (n=5) | 75287.1726 +/- 59799.4724 (n=5) | 6.1721 +/- 2.9402 (n=5) | 0.9125 +/- 0.0158 (n=5) |
| mlp_global | covariance | 4.7820 +/- 1.3227 (n=5) | 842.2539 +/- 438.6868 (n=5) | 3.7300 +/- 3.4919 (n=5) | 3.9953 +/- 0.0022 (n=5) |
| mlp_global | sigreg | 3.6091 +/- 1.1152 (n=5) | 2660.0332 +/- 744.7310 (n=5) | 1.4593 +/- 0.1675 (n=5) | 1.3290 +/- 0.0584 (n=5) |
| gnn_node | none | 3.3548 +/- 0.4533 (n=5) | 15046.0870 +/- 19175.9747 (n=5) | 2.4336 +/- 1.1735 (n=5) | 0.0000 +/- 0.0000 (n=5) |
| gnn_node | variance | 1.3941 +/- 0.1916 (n=5) | 783344.1986 +/- 466386.7934 (n=5) | 8.3930 +/- 2.9626 (n=5) | 0.9544 +/- 0.0061 (n=5) |
| gnn_node | covariance | 3.2144 +/- 0.3011 (n=5) | 13656.3541 +/- 12510.0938 (n=5) | 2.7867 +/- 1.4904 (n=5) | 3.9994 +/- 0.0002 (n=5) |
| gnn_node | sigreg | 2.1232 +/- 0.4870 (n=5) | 30338.5770 +/- 25992.6401 (n=5) | 1.5085 +/- 0.1713 (n=5) | 1.4582 +/- 0.0862 (n=5) |
| gnn_node | graph | 3.1156 +/- 0.6779 (n=5) | 83764.2817 +/- 156906.2296 (n=5) | 3.7998 +/- 0.2998 (n=5) | 0.0895 +/- 0.0314 (n=5) |
| gnn_node | permuted_graph | 3.0515 +/- 0.4773 (n=5) | 40566.3006 +/- 71363.8580 (n=5) | 3.6312 +/- 0.2776 (n=5) | 0.1531 +/- 0.0685 (n=5) |
| gnn_node | random_graph | 3.0495 +/- 0.5216 (n=5) | 32206.5826 +/- 49600.2612 (n=5) | 3.6777 +/- 0.2933 (n=5) | 0.1530 +/- 0.0683 (n=5) |

## Euclidean Conditioning Gains

- MLP none vs MLP covariance condition number: 623.9470 -> 842.2539 (-35.0% loss)
- MLP none vs MLP SIGReg condition number: 623.9470 -> 2660.0332 (-326.3% loss)
- GNN none vs GNN covariance condition number: 15046.0870 -> 13656.3541 (+9.2% gain)
- GNN none vs GNN SIGReg condition number: 15046.0870 -> 30338.5770 (-101.6% loss)

## Graph Specificity

- GNN graph vs GNN permuted_graph H=32: 0.0330 -> 0.0350 (-6.2% loss)
- GNN graph vs GNN random_graph H=32: 0.0330 -> 0.0336 (-1.9% loss)

## Architecture Gain

- MLP none vs GNN none H=32: 0.2283 -> 0.1008 (+55.9% gain)

## Graph Prior Gain Beyond GNN

- GNN none vs GNN graph H=32: 0.1008 -> 0.0330 (+67.3% gain)

## Recommendation

Cycle 2 multi-molecule expansion is justified, but as a controlled test rather than a graph-specific claim. Expand GNN none, GNN graph, GNN permuted_graph, GNN random_graph, and MLP none. Current evidence points to a strong GNN graph-prior/generic smoothing effect beyond GNN, with weak true graph specificity because permuted/random graph controls are close to the true graph.
