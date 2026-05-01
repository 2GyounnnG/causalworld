# Cycle 6 Graph-Dynamics Alignment Diagnostics

Analysis only: no training, ISO17, rMD17 top-up, or new experiment launch commands are run.

Inputs:
- `experiments/results/cycle2_rmd17_multimolecule/cycle2_rmd17_multimolecule_results.json`
- `experiments/results/cycle3_ho_networks/cycle3_ho_networks_results.json`
- `experiments/results/cycle4_ho_lambda_robustness/cycle4_ho_lambda_robustness_results.json`
- `experiments/results/cycle5_ho_lattice_bridge/cycle5_ho_lattice_bridge_results.json`

Method:
- `X_t` is the centered raw coordinate matrix for each frame.
- `dX_t` is the centered one-step coordinate change, `X_{t+1} - X_t`.
- HO true graphs use the stored unit-weight topology edges.
- rMD17 true graphs use the same cutoff graph and inverse-distance weights as the node-wise graph prior.
- `permuted_graph` uses the true Laplacian with the training permutation seed formula and a deterministic NumPy permutation; this preserves the scale-matched control interpretation.
- `random_graph` uses the same random-edge seed formula used during training.

## Alignment Gap Summary

Positive `D_dX` gaps mean the true graph is smoother than the control graph for temporal changes. Positive `R_low` gaps mean temporal changes put more energy in the true graph low-frequency basis.

| domain | item | D_dX gap perm-control | D_dX gap random-control | R_low_dX2 gap perm | R_low_dX2 gap random | true smoother vs both? |
| --- | --- | --- | --- | --- | --- | --- |
| HO | lattice | -0.752421 | -0.768985 | -0.011168 | -0.016212 | NO |
| HO | random | -1.0349 | -1.0610 | -0.009295 | -0.017623 | NO |
| HO | scalefree | -3.2220 | -3.1893 | -0.005288 | -0.007787 | NO |
| rMD17 | aspirin | 2.1991 | 2.2560 | 0.162895 | 0.158951 | YES |
| rMD17 | ethanol | 0.795017 | 0.808203 | 0.233862 | 0.249449 | YES |
| rMD17 | malonaldehyde | 0.762419 | 0.769573 | 0.204301 | 0.232618 | YES |
| rMD17 | naphthalene | 1.5666 | 1.5413 | 0.066382 | 0.061357 | YES |
| rMD17 | toluene | 1.3362 | 1.3149 | -0.056252 | -0.048196 | YES |

## Correlations With Observed Specificity

`S_graph = E_control - E_graph`; positive values mean the true graph prior has lower rollout error than the control.

| scope | predictor | n | Pearson r | Spearman rho |
| --- | --- | --- | --- | --- |
| all | gap_D_dX_norm_control_minus_true | 84 | -0.029827 | -0.058033 |
| all | gap_D_X_norm_control_minus_true | 84 | 0.067663 | 0.034713 |
| all | gap_R_low_dX_2_true_minus_control | 84 | -0.002901 | 0.096406 |
| all | gap_frobenius_control_minus_true | 84 | -0.149111 | -0.050054 |
| all | gap_lambda_max_control_minus_true | 84 | -0.142241 | -0.116970 |
| all | graph_prior_loss_mean | 84 | -0.026682 | -0.205940 |
| H=32 | gap_D_dX_norm_control_minus_true | 42 | -0.036921 | -0.028058 |
| H=32 | gap_D_X_norm_control_minus_true | 42 | 0.074585 | 0.035792 |
| H=32 | gap_R_low_dX_2_true_minus_control | 42 | -0.001288 | 0.088616 |
| H=32 | gap_frobenius_control_minus_true | 42 | -0.173485 | -0.065413 |
| H=32 | gap_lambda_max_control_minus_true | 42 | -0.165678 | -0.116097 |
| H=32 | graph_prior_loss_mean | 42 | -0.028751 | -0.259300 |
| rMD17 | gap_D_dX_norm_control_minus_true | 20 | 0.636411 | 0.727552 |
| rMD17 | gap_D_X_norm_control_minus_true | 20 | 0.697025 | 0.781892 |
| rMD17 | gap_R_low_dX_2_true_minus_control | 20 | -0.303679 | -0.461890 |
| rMD17 | gap_frobenius_control_minus_true | 20 | -0.243334 | -0.181133 |
| rMD17 | gap_lambda_max_control_minus_true | 20 | 0.080505 | 0.099623 |
| rMD17 | graph_prior_loss_mean | 20 | 0.657730 | 0.797081 |
| HO | gap_D_dX_norm_control_minus_true | 64 | -0.151167 | -0.209035 |
| HO | gap_D_X_norm_control_minus_true | 64 | -0.147803 | -0.191766 |
| HO | gap_R_low_dX_2_true_minus_control | 64 | 0.104430 | 0.102098 |
| HO | gap_frobenius_control_minus_true | 64 | -0.146905 | -0.035867 |
| HO | gap_lambda_max_control_minus_true | 64 | -0.144699 | -0.150111 |
| HO | graph_prior_loss_mean | 64 | -0.047774 | -0.498739 |

## Diagnostic Answers

1. Are true dynamics smoother on true graph than on controls? At H=32 comparison granularity, 10/42 rows have positive temporal-change alignment gaps, and 3/15 positive-specificity rows also have positive temporal-change gaps. The answer is mixed, not universal.
2. Does temporal-change smoothness predict observed specificity? Weakly at best in the pooled table: H=32 Spearman rho for `gap_D_dX_norm_control_minus_true` is -0.028058.
3. Is `D_dX` more predictive than static `D_X`? H=32 Spearman rho is -0.028058 for `D_dX` versus 0.035792 for `D_X`; use the sign and magnitude together, because source/lambda reuse creates repeated alignment rows.
4. Is low-frequency energy ratio predictive? H=32 Spearman rho for `R_low_dX(K=2)` gap is 0.088616; this is not strong enough to stand alone as a decision rule.
5. Are graph spectrum-only metrics insufficient? Yes. H=32 Spearman rho for Frobenius/lambda_max gaps is -0.065413/-0.116097, and permuted controls are spectrum-matched by construction, so spectrum-only metrics cannot explain permuted-control outcomes.
6. Decision rule: the proposed `only when D_dX is smoother on the true graph` rule is not validated by these raw-coordinate diagnostics. It would correctly warn against many generic-smoothing cases, but it would also reject the Cycle 5 high-lambda lattice specificity point. A safer rule is: claim topology specificity only with paired rollout evidence against scale-matched controls; use positive `D_dX` alignment as supporting mechanistic evidence when present, not as a sufficient or currently necessary condition.

Key caveat: these diagnostics are computed on raw centered coordinates, while the prior acts on learned node states. A learned encoder can rotate, rescale, or filter dynamics before the Laplacian penalty is applied, so raw-coordinate alignment is a mechanism probe rather than a complete explanation.

## Files

- `analysis_out/cycle6_alignment_summary.csv`
- `analysis_out/cycle6_alignment_vs_specificity.csv`
