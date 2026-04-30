# Cycle 4 Graph Spectra And Loss Diagnostics

Results file: `experiments/results/cycle4_ho_lambda_robustness/cycle4_ho_lambda_robustness_results.json`
Schema version: `cycle4_ho_lambda_robustness_v1`
Graph-prior success/failure count: 72 ok / 0 failed
Config directory: `experiments/configs/cycle4_ho_lambda_robustness`
Data root: `data/ho_raw`

Analysis only: no training or ISO17 commands are run.

## Extracted Run Fields

| topology | prior | lambda | seed | H=32 rollout | final_train_loss | prior_loss_mean |
|---|---|---:|---:|---:|---:|---:|
| lattice | graph | 0.001 | 0 | 0.0405 | 3.150e-05 | 0.2909 |
| lattice | graph | 0.001 | 1 | 0.0394 | 2.821e-05 | 0.2836 |
| lattice | graph | 0.001 | 2 | 0.0343 | 2.426e-05 | 0.1419 |
| lattice | graph | 0.005 | 0 | 0.0214 | 3.167e-05 | 0.0927 |
| lattice | graph | 0.005 | 1 | 0.0284 | 4.197e-05 | 0.1024 |
| lattice | graph | 0.005 | 2 | 0.0161 | 3.025e-05 | 0.0515 |
| lattice | graph | 0.01 | 0 | 0.0298 | 3.697e-05 | 0.0554 |
| lattice | graph | 0.01 | 1 | 0.0135 | 5.224e-05 | 0.0608 |
| lattice | graph | 0.01 | 2 | 0.0111 | 3.758e-05 | 0.0344 |
| lattice | graph | 0.05 | 0 | 0.0293 | 9.014e-05 | 0.0265 |
| lattice | graph | 0.05 | 1 | 0.0132 | 9.877e-05 | 0.0252 |
| lattice | graph | 0.05 | 2 | 0.0072 | 8.991e-05 | 0.0191 |
| lattice | permuted_graph | 0.001 | 0 | 0.0122 | 2.870e-05 | 0.4357 |
| lattice | permuted_graph | 0.001 | 1 | 0.0298 | 2.930e-05 | 0.4426 |
| lattice | permuted_graph | 0.001 | 2 | 0.0435 | 2.793e-05 | 0.2332 |
| lattice | permuted_graph | 0.005 | 0 | 0.0106 | 3.417e-05 | 0.0957 |
| lattice | permuted_graph | 0.005 | 1 | 0.0177 | 4.220e-05 | 0.1082 |
| lattice | permuted_graph | 0.005 | 2 | 0.0230 | 3.180e-05 | 0.0576 |
| lattice | permuted_graph | 0.01 | 0 | 0.0117 | 3.998e-05 | 0.0550 |
| lattice | permuted_graph | 0.01 | 1 | 0.0106 | 5.207e-05 | 0.0647 |
| lattice | permuted_graph | 0.01 | 2 | 0.0056 | 3.481e-05 | 0.0366 |
| lattice | permuted_graph | 0.05 | 0 | 0.0243 | 1.327e-04 | 0.0348 |
| lattice | permuted_graph | 0.05 | 1 | 0.0173 | 1.300e-04 | 0.0391 |
| lattice | permuted_graph | 0.05 | 2 | 0.0082 | 9.949e-05 | 0.0249 |
| lattice | random_graph | 0.001 | 0 | 0.0126 | 2.965e-05 | 0.4387 |
| lattice | random_graph | 0.001 | 1 | 0.0345 | 3.289e-05 | 0.4729 |
| lattice | random_graph | 0.001 | 2 | 0.0434 | 2.743e-05 | 0.2300 |
| lattice | random_graph | 0.005 | 0 | 0.0103 | 3.444e-05 | 0.0957 |
| lattice | random_graph | 0.005 | 1 | 0.0247 | 4.405e-05 | 0.1112 |
| lattice | random_graph | 0.005 | 2 | 0.0233 | 3.169e-05 | 0.0572 |
| lattice | random_graph | 0.01 | 0 | 0.0148 | 4.382e-05 | 0.0552 |
| lattice | random_graph | 0.01 | 1 | 0.0097 | 5.469e-05 | 0.0651 |
| lattice | random_graph | 0.01 | 2 | 0.0049 | 3.625e-05 | 0.0366 |
| lattice | random_graph | 0.05 | 0 | 0.0241 | 1.390e-04 | 0.0347 |
| lattice | random_graph | 0.05 | 1 | 0.0197 | 1.224e-04 | 0.0395 |
| lattice | random_graph | 0.05 | 2 | 0.0082 | 9.892e-05 | 0.0249 |
| scalefree | graph | 0.001 | 0 | 0.0142 | 2.822e-05 | 0.2684 |
| scalefree | graph | 0.001 | 1 | 0.0268 | 3.614e-05 | 0.2835 |
| scalefree | graph | 0.001 | 2 | 0.0281 | 2.181e-05 | 0.1492 |
| scalefree | graph | 0.005 | 0 | 0.0428 | 3.085e-05 | 0.0878 |
| scalefree | graph | 0.005 | 1 | 0.0201 | 4.211e-05 | 0.1015 |
| scalefree | graph | 0.005 | 2 | 0.0382 | 3.021e-05 | 0.0529 |
| scalefree | graph | 0.01 | 0 | 0.0136 | 3.500e-05 | 0.0554 |
| scalefree | graph | 0.01 | 1 | 0.0154 | 5.115e-05 | 0.0616 |
| scalefree | graph | 0.01 | 2 | 0.0099 | 3.293e-05 | 0.0326 |
| scalefree | graph | 0.05 | 0 | 0.0345 | 8.540e-05 | 0.0280 |
| scalefree | graph | 0.05 | 1 | 0.0208 | 8.676e-05 | 0.0232 |
| scalefree | graph | 0.05 | 2 | 0.0147 | 6.878e-05 | 0.0155 |
| scalefree | permuted_graph | 0.001 | 0 | 0.0262 | 2.934e-05 | 0.3713 |
| scalefree | permuted_graph | 0.001 | 1 | 0.0256 | 3.718e-05 | 0.4155 |
| scalefree | permuted_graph | 0.001 | 2 | 0.0542 | 2.549e-05 | 0.2184 |
| scalefree | permuted_graph | 0.005 | 0 | 0.0160 | 2.741e-05 | 0.1007 |
| scalefree | permuted_graph | 0.005 | 1 | 0.0183 | 4.483e-05 | 0.1229 |
| scalefree | permuted_graph | 0.005 | 2 | 0.0126 | 3.393e-05 | 0.0617 |
| scalefree | permuted_graph | 0.01 | 0 | 0.0136 | 3.274e-05 | 0.0603 |
| scalefree | permuted_graph | 0.01 | 1 | 0.0155 | 5.350e-05 | 0.0687 |
| scalefree | permuted_graph | 0.01 | 2 | 0.0128 | 3.406e-05 | 0.0357 |
| scalefree | permuted_graph | 0.05 | 0 | 0.0214 | 7.779e-05 | 0.0279 |
| scalefree | permuted_graph | 0.05 | 1 | 0.0209 | 1.011e-04 | 0.0277 |
| scalefree | permuted_graph | 0.05 | 2 | 0.0102 | 7.195e-05 | 0.0181 |
| scalefree | random_graph | 0.001 | 0 | 0.0347 | 2.981e-05 | 0.3746 |
| scalefree | random_graph | 0.001 | 1 | 0.0272 | 3.800e-05 | 0.4153 |
| scalefree | random_graph | 0.001 | 2 | 0.0559 | 2.627e-05 | 0.2195 |
| scalefree | random_graph | 0.005 | 0 | 0.0131 | 2.747e-05 | 0.1011 |
| scalefree | random_graph | 0.005 | 1 | 0.0175 | 4.510e-05 | 0.1189 |
| scalefree | random_graph | 0.005 | 2 | 0.0136 | 3.285e-05 | 0.0617 |
| scalefree | random_graph | 0.01 | 0 | 0.0353 | 3.490e-05 | 0.0613 |
| scalefree | random_graph | 0.01 | 1 | 0.0223 | 5.460e-05 | 0.0683 |
| scalefree | random_graph | 0.01 | 2 | 0.0128 | 3.477e-05 | 0.0358 |
| scalefree | random_graph | 0.05 | 0 | 0.0212 | 8.121e-05 | 0.0285 |
| scalefree | random_graph | 0.05 | 1 | 0.0197 | 1.022e-04 | 0.0276 |
| scalefree | random_graph | 0.05 | 2 | 0.0112 | 7.239e-05 | 0.0183 |

## Laplacian Scale Diagnostics

For `graph` and `permuted_graph`, the table is the static topology Laplacian. For `random_graph`, it is averaged over the reconstructed training-frame random graphs used by the prior loss.

| topology | prior | n_nodes | n_edges | degree mean | degree std | trace | fro norm | lambda_max | spectral gap | cond/spread |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lattice | graph | 64 | 112 | 3.5000 | 0.6124 | 224.0000 | 32.1248 | 7.6955 | 0.1522 | 50.5483 |
| lattice | permuted_graph | 64 | 112 | 3.5000 | 0.6124 | 224.0000 | 32.1248 | 7.6955 | 0.1522 | 50.5483 |
| lattice | random_graph | 64 | 112 | 3.5000 | 1.7841 | 224.0000 | 34.8291 | 10.1812 | 0.0441 | 38.8105 |
| scalefree | graph | 64 | 124 | 3.8750 | 2.9817 | 248.0000 | 42.1663 | 16.0916 | 0.6109 | 26.3393 |
| scalefree | permuted_graph | 64 | 124 | 3.8750 | 2.9817 | 248.0000 | 42.1663 | 16.0916 | 0.6109 | 26.3393 |
| scalefree | random_graph | 64 | 124 | 3.8750 | 1.8809 | 248.0000 | 37.9055 | 10.8323 | 0.0998 | 30.6549 |

## Effective Prior Strength

| topology | prior | lambda | H=32 mean +/- std | prior_loss_mean +/- std | lambda*fro | lambda*lambda_max |
|---|---|---:|---:|---:|---:|---:|
| lattice | graph | 0.001 | 0.0381 +/- 0.0033 (n=3) | 0.2388 +/- 0.0840 | 0.0321 | 0.0077 |
| lattice | graph | 0.005 | 0.0219 +/- 0.0062 (n=3) | 0.0822 +/- 0.0270 | 0.1606 | 0.0385 |
| lattice | graph | 0.01 | 0.0182 +/- 0.0102 (n=3) | 0.0502 +/- 0.0140 | 0.3212 | 0.0770 |
| lattice | graph | 0.05 | 0.0166 +/- 0.0114 (n=3) | 0.0236 +/- 0.0040 | 1.6062 | 0.3848 |
| lattice | permuted_graph | 0.001 | 0.0285 +/- 0.0157 (n=3) | 0.3705 +/- 0.1190 | 0.0321 | 0.0077 |
| lattice | permuted_graph | 0.005 | 0.0171 +/- 0.0062 (n=3) | 0.0872 +/- 0.0264 | 0.1606 | 0.0385 |
| lattice | permuted_graph | 0.01 | 0.0093 +/- 0.0032 (n=3) | 0.0521 +/- 0.0143 | 0.3212 | 0.0770 |
| lattice | permuted_graph | 0.05 | 0.0166 +/- 0.0081 (n=3) | 0.0329 +/- 0.0072 | 1.6062 | 0.3848 |
| lattice | random_graph | 0.001 | 0.0301 +/- 0.0159 (n=3) | 0.3806 +/- 0.1315 | 0.0348 | 0.0102 |
| lattice | random_graph | 0.005 | 0.0194 +/- 0.0079 (n=3) | 0.0880 +/- 0.0278 | 0.1741 | 0.0509 |
| lattice | random_graph | 0.01 | 0.0098 +/- 0.0049 (n=3) | 0.0523 +/- 0.0145 | 0.3483 | 0.1018 |
| lattice | random_graph | 0.05 | 0.0173 +/- 0.0082 (n=3) | 0.0330 +/- 0.0075 | 1.7415 | 0.5091 |
| scalefree | graph | 0.001 | 0.0230 +/- 0.0077 (n=3) | 0.2337 +/- 0.0736 | 0.0422 | 0.0161 |
| scalefree | graph | 0.005 | 0.0337 +/- 0.0120 (n=3) | 0.0807 +/- 0.0251 | 0.2108 | 0.0805 |
| scalefree | graph | 0.01 | 0.0130 +/- 0.0028 (n=3) | 0.0499 +/- 0.0153 | 0.4217 | 0.1609 |
| scalefree | graph | 0.05 | 0.0233 +/- 0.0101 (n=3) | 0.0223 +/- 0.0063 | 2.1083 | 0.8046 |
| scalefree | permuted_graph | 0.001 | 0.0354 +/- 0.0163 (n=3) | 0.3351 +/- 0.1034 | 0.0422 | 0.0161 |
| scalefree | permuted_graph | 0.005 | 0.0157 +/- 0.0028 (n=3) | 0.0951 +/- 0.0310 | 0.2108 | 0.0805 |
| scalefree | permuted_graph | 0.01 | 0.0140 +/- 0.0014 (n=3) | 0.0549 +/- 0.0172 | 0.4217 | 0.1609 |
| scalefree | permuted_graph | 0.05 | 0.0175 +/- 0.0063 (n=3) | 0.0245 +/- 0.0056 | 2.1083 | 0.8046 |
| scalefree | random_graph | 0.001 | 0.0393 +/- 0.0149 (n=3) | 0.3365 +/- 0.1033 | 0.0379 | 0.0108 |
| scalefree | random_graph | 0.005 | 0.0147 +/- 0.0024 (n=3) | 0.0939 +/- 0.0293 | 0.1895 | 0.0542 |
| scalefree | random_graph | 0.01 | 0.0235 +/- 0.0113 (n=3) | 0.0551 +/- 0.0171 | 0.3791 | 0.1083 |
| scalefree | random_graph | 0.05 | 0.0174 +/- 0.0054 (n=3) | 0.0248 +/- 0.0057 | 1.8953 | 0.5416 |

## H=32 Control Comparisons

| topology | lambda | control | graph mean | control mean | graph advantage |
|---|---:|---|---:|---:|---:|
| lattice | 0.001 | permuted_graph | 0.0381 | 0.0285 | -33.6% |
| lattice | 0.001 | random_graph | 0.0381 | 0.0301 | -26.3% |
| lattice | 0.005 | permuted_graph | 0.0219 | 0.0171 | -28.2% |
| lattice | 0.005 | random_graph | 0.0219 | 0.0194 | -13.1% |
| lattice | 0.01 | permuted_graph | 0.0182 | 0.0093 | -94.9% |
| lattice | 0.01 | random_graph | 0.0182 | 0.0098 | -85.2% |
| lattice | 0.05 | permuted_graph | 0.0166 | 0.0166 | +0.3% |
| lattice | 0.05 | random_graph | 0.0166 | 0.0173 | +4.3% |
| scalefree | 0.001 | permuted_graph | 0.0230 | 0.0354 | +34.9% |
| scalefree | 0.001 | random_graph | 0.0230 | 0.0393 | +41.4% |
| scalefree | 0.005 | permuted_graph | 0.0337 | 0.0157 | -115.2% |
| scalefree | 0.005 | random_graph | 0.0337 | 0.0147 | -128.5% |
| scalefree | 0.01 | permuted_graph | 0.0130 | 0.0140 | +7.2% |
| scalefree | 0.01 | random_graph | 0.0130 | 0.0235 | +44.7% |
| scalefree | 0.05 | permuted_graph | 0.0233 | 0.0175 | -33.3% |
| scalefree | 0.05 | random_graph | 0.0233 | 0.0174 | -34.2% |

## Correlation With H=32 Rollout

Correlations use successful graph-prior runs. Positive values mean larger predictor values accompany larger H=32 rollout error.

| scope | predictor | n | Pearson r | Spearman rho |
|---|---|---:|---:|---:|
| all | lambda | 72 | -0.2540 | -0.4435 |
| all | lambda_eff_fro | 72 | -0.2393 | -0.4080 |
| all | lambda_eff_lambda_max | 72 | -0.2068 | -0.3847 |
| all | prior_loss_mean | 72 | 0.4096 | 0.4501 |
| all | final_train_loss | 72 | -0.2195 | -0.3189 |
| lattice | lambda | 36 | -0.2779 | -0.4975 |
| lattice | lambda_eff_fro | 36 | -0.2777 | -0.4976 |
| lattice | lambda_eff_lambda_max | 36 | -0.2735 | -0.4983 |
| lattice | prior_loss_mean | 36 | 0.4429 | 0.5282 |
| lattice | final_train_loss | 36 | -0.2137 | -0.3864 |
| scalefree | lambda | 36 | -0.2336 | -0.3994 |
| scalefree | lambda_eff_fro | 36 | -0.2306 | -0.3922 |
| scalefree | lambda_eff_lambda_max | 36 | -0.2198 | -0.4037 |
| scalefree | prior_loss_mean | 36 | 0.3867 | 0.4347 |
| scalefree | final_train_loss | 36 | -0.2158 | -0.1686 |

## Diagnostic Answers

- Are true/permuted/random graph priors matched in Laplacian scale? Partly. True and permuted are exactly matched in scale because permutation relabels node states against the same Laplacian spectrum. Random controls match node count, edge count, and trace, but not degree dispersion, Frobenius norm, lambda_max, or spectral gap.
  lattice: permuted vs true fro +0.0%, lambda_max +0.0%; random vs true fro +8.4%, lambda_max +32.3%.
  scalefree: permuted vs true fro +0.0%, lambda_max +0.0%; random vs true fro -10.1%, lambda_max -32.7%.
- Could random/permuted controls outperform because effective smoothing strength differs? Random controls can, in principle, because their spectral scale is not matched beyond edge count and trace. Permuted controls cannot be explained by Laplacian scale: their spectrum is identical to the true graph, so their wins point to node-label specificity being weak or to optimization/regularization effects. Observed H=32 control wins include: lattice lambda=0.001 permuted_graph 0.0285 < graph 0.0381; lattice lambda=0.001 random_graph 0.0301 < graph 0.0381; lattice lambda=0.005 permuted_graph 0.0171 < graph 0.0219; lattice lambda=0.005 random_graph 0.0194 < graph 0.0219; lattice lambda=0.01 permuted_graph 0.0093 < graph 0.0182; lattice lambda=0.01 random_graph 0.0098 < graph 0.0182; scalefree lambda=0.005 permuted_graph 0.0157 < graph 0.0337; scalefree lambda=0.005 random_graph 0.0147 < graph 0.0337.
- Does H=32 correlate better with lambda or lambda_eff? Across all Cycle 4 graph-prior runs, Spearman rho is lambda=-0.4435, lambda*fro=-0.4080, lambda*lambda_max=-0.3847. By rank correlation the strongest listed predictor is `prior_loss_mean`; the differences should be read cautiously because topology and lambda are confounded and the response is non-monotone.
- Is lambda sensitivity likely topology-specific or regularization-scale driven? The evidence points more toward regularization scale plus graph-control mismatch than a stable true-topology effect. Topology matters because scale metrics differ by graph family, but true specificity does not survive lambda changes and permuted controls sometimes win at identical Laplacian scale.

## Implementation Notes

- `permuted_graph` is analyzed as a relabeled true-graph Laplacian: spectral scale is unchanged even though node-label alignment is destroyed in the loss.
- `random_graph` uses the same random-edge seed formula as training: `3001 + seed * 100000 + frame_idx`, with frame indices reconstructed from `n_transitions`, `stride`, `horizon`, and `seed`.
- HO graph edge weights are unit weights in the training loader, so trace is `2 * n_edges` for all true/permuted/random graphs with the same topology edge count.
