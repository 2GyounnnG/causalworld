# Audit: Priors And Laplacians

## Summary

- PASS: 19
- WARNING: 2

## Checks

| Check | Status | Evidence | Recommendation |
| --- | --- | --- | --- |
| train_rmd17.py parses | PASS | AST parse succeeded |  |
| rMD17 Euclidean covariance is batch-level | PASS | scripts/train_rmd17.py collects batch_latents and applies euclidean_cov_penalty(torch.stack(batch_latents, dim=0)) once per minibatch. |  |
| Wolfram Euclidean covariance is batch-level | PASS | train.py appends model.encode(obs) across the minibatch and applies euclidean_cov_penalty(latent_batch). |  |
| Euclidean covariance needs at least two latents | PASS | model.euclidean_cov_penalty returns zero for B < 2, which is why training scripts must bypass per-sample model.loss for Euclidean. |  |
| rMD17 Euclidean path is separate from none and spectral | PASS | Euclidean first computes base transition loss with prior='none', then adds config.prior_weight * batch covariance penalty. |  |
| prior_weight=0 disables prior penalty | PASS | tiny smoke test: spectral prior_loss=1.58139, total difference vs none=0 |  |
| Config preserves per_frame default | PASS | Config.laplacian_mode defaults to 'per_frame', preserving existing callers. |  |
| per_frame recomputes Laplacian from current sample | PASS | The spectral batch loop builds L from obs_raw when fixed_laplacian is None. |  |
| fixed_frame0 uses one fixed Laplacian | PASS | fixed_frame0 builds L once from RMD17Trajectory(config.molecule)[0] before the model is trained/evaluated. |  |
| fixed_mean uses one averaged Laplacian | PASS | fixed_mean builds dense first-frame samples up to min(500, n_frames), averages them, and reuses the tensor. |  |
| fixed Laplacian modes do not inspect current batch frame for L | PASS | The batch loop reuses fixed_laplacian instead of calling build_molecular_laplacian(obs_raw) in fixed modes. |  |
| Laplacian ablation runner requests all modes | PASS | scripts/run_rmd17_laplacian_ablation.py imports build_molecular_laplacian and passes laplacian_mode=mode into Config. |  |
| rMD17 seed handling | PASS | train_one_seed sets NumPy, Torch, and CUDA seeds. |  |
| Wolfram seed handling | PASS | train.set_seed covers Python random, NumPy, Torch, and CUDA. |  |
| rMD17 output records seed/config | PASS | train_one_seed returns config.__dict__, which includes seed and laplacian_mode for new runs. |  |
| rMD17 evaluation horizons | PASS | Config defaults to H=1,2,4,8,16 and eval transition collection uses max horizon. |  |
| Wolfram evaluation horizons | PASS | train.py defines HORIZONS = [1, 2, 4, 8, 16]. |  |
| rMD17 train/eval sampling separation | WARNING | Eval transitions use a different seed and coarser stride, but collect_rmd17_transitions does not explicitly exclude training frame indices. | For a reviewer-facing leakage audit, add a non-training analysis check that compares sampled train/eval frame_idx sets for each seed. |
| rMD17 train/eval frame-index overlap | WARNING | Reconstructed train/eval frame_idx sets for 17 molecule/seed configs from result metadata; 17 configs overlap. aspirin seed=0: 30 overlaps, first=[3000, 3700, 14100, 16400, 17000]; aspirin seed=1: 33 overlaps, first=[1400, 6900, 8000, 11000, 12100]; aspirin seed=2: 41 overlaps, first=[4300, 5200, 8400, 15200, 21200]; aspirin seed=3: 44 overlaps, first=[3400, 4400, 5800, 6200, 6500]; aspirin seed=4: 46 overlaps, first=[300, 700, 2800, 4000, 11100]; aspirin seed=5: 41 overlaps, first=[4700, 5600, 6600, 6900, 9200] | Exclude training frame_idx values from eval sampling, and persist sampled train/eval frame_idx sets into run metadata for future reviewer-facing leakage audits. |
| Wolfram train/eval sampling separation | PASS | Wolfram runners collect eval episodes with seed=10_000 + seed. |  |
| rMD17 transition sampling reproducibility | PASS | collect_rmd17_transitions samples candidate frame indices with np.random.default_rng(seed) and records frame_idx. |  |

## Commit-Style Patch Section

No FAIL checks were found, so no patch section is proposed.
