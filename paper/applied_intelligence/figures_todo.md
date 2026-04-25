# Figures Todo

## Figure 1: Euclidean vs Spectral Latent Priors

- Source CSV: conceptual figure; no CSV required.
- Intended message: Euclidean priors regularize latent covariance generically, while spectral priors impose graph-structured geometry.
- Caption draft: "Conceptual comparison of Euclidean and spectral latent priors. The Euclidean prior encourages isotropic latent geometry, whereas the spectral prior aligns latent variation with graph Laplacian structure. The experiments show that this structure is beneficial when aligned with dynamics but can destabilize long-horizon rollout when misaligned."

## Figure 2: rMD17 Aspirin Horizon Scaling

- Source CSV: `analysis_out/aggregate_by_prior_horizon.csv`
- Intended message: Spectral priors reduce rollout error across horizons and give the strongest H=16 result in the completed aspirin 10-seed comparison.
- Caption draft: "rMD17 aspirin rollout error by horizon for no prior, Euclidean prior, and spectral prior. The spectral prior achieves the lowest H=16 error under the current evaluation protocol."

## Figure 3: Laplacian Ablation H=16 / Horizon Plot

- Source CSV: `analysis_out/aggregate_laplacian_ablation.csv`
- Intended message: Per-frame Laplacian is best, but fixed Laplacian controls still outperform the no-prior baseline, indicating that the spectral gain is not solely due to per-frame recomputation.
- Caption draft: "Spectral Laplacian ablation on rMD17 aspirin. Fixed mean and fixed frame-0 Laplacians preserve substantial improvement over the no-prior baseline, while the per-frame Laplacian obtains the lowest error."

## Figure 4: Weight Sweep H=16 by Prior and Weight

- Source CSV: `analysis_out/aggregate_weight_sweep.csv`
- Intended message: Prior weight matters. Euclidean improves when tuned, and the best spectral setting outperforms the best Euclidean setting, while spectral `w=0.01` is unstable.
- Caption draft: "rMD17 aspirin H=16 rollout error across prior weights. Tuned Euclidean regularization improves over no prior, while the best spectral setting is strongest. The spectral `w=0.01` setting reveals long-horizon instability."

## Figure 5: Wolfram Spectral Seed-Level Instability / Mean vs Median

- Source CSV: `analysis_out/wolfram_horizon_ratios.csv` and `analysis_out/aggregate_wolfram.csv`
- Intended message: Wolfram flat is a boundary case. Euclidean has the best H=16 mean, while spectral has a heavy-tailed failure mode caused by a few exploding seeds.
- Caption draft: "Wolfram flat H=16 instability analysis. Spectral prior performance is heavy-tailed: most seeds remain near baseline scale, but seeds 1, 2, and 8 explode at long horizon, causing the spectral mean to diverge from its median."

