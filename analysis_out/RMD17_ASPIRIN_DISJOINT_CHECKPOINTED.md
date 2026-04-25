# rMD17 Aspirin Disjoint Checkpointed

## Status

- aspirin / none: 5/5 seeds complete
- aspirin / euclidean: 5/5 seeds complete
- aspirin / spectral: 5/5 seeds complete
- checkpoints present for completed runs: 15
- frame-overlap audit: PASS

## H=16 Aggregate

| molecule | prior | n | mean | std | median | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aspirin | euclidean | 5 | 0.273 | 0.125 | 0.245 | 0.165 | 0.486 |
| aspirin | none | 5 | 0.374 | 0.240 | 0.437 | 0.092 | 0.665 |
| aspirin | spectral | 5 | 0.112 | 0.075 | 0.086 | 0.062 | 0.243 |

## Frame-Overlap Audit

| molecule | prior | seed | train_eval_start_overlap_count | train_eval_overlap_count | checkpoint_path |
| --- | --- | --- | --- | --- | --- |
| aspirin | euclidean | 0 | 0 | 0 | checkpoints/rmd17_aspirin_disjoint_checkpointed/aspirin_flat_euclidean_seed0_w0.1_modeper_frame.pt |
| aspirin | euclidean | 1 | 0 | 0 | checkpoints/rmd17_aspirin_disjoint_checkpointed/aspirin_flat_euclidean_seed1_w0.1_modeper_frame.pt |
| aspirin | euclidean | 2 | 0 | 0 | checkpoints/rmd17_aspirin_disjoint_checkpointed/aspirin_flat_euclidean_seed2_w0.1_modeper_frame.pt |
| aspirin | euclidean | 3 | 0 | 0 | checkpoints/rmd17_aspirin_disjoint_checkpointed/aspirin_flat_euclidean_seed3_w0.1_modeper_frame.pt |
| aspirin | euclidean | 4 | 0 | 0 | checkpoints/rmd17_aspirin_disjoint_checkpointed/aspirin_flat_euclidean_seed4_w0.1_modeper_frame.pt |
| aspirin | none | 0 | 0 | 0 | checkpoints/rmd17_aspirin_disjoint_checkpointed/aspirin_flat_none_seed0_w0.1_modeper_frame.pt |
| aspirin | none | 1 | 0 | 0 | checkpoints/rmd17_aspirin_disjoint_checkpointed/aspirin_flat_none_seed1_w0.1_modeper_frame.pt |
| aspirin | none | 2 | 0 | 0 | checkpoints/rmd17_aspirin_disjoint_checkpointed/aspirin_flat_none_seed2_w0.1_modeper_frame.pt |
| aspirin | none | 3 | 0 | 0 | checkpoints/rmd17_aspirin_disjoint_checkpointed/aspirin_flat_none_seed3_w0.1_modeper_frame.pt |
| aspirin | none | 4 | 0 | 0 | checkpoints/rmd17_aspirin_disjoint_checkpointed/aspirin_flat_none_seed4_w0.1_modeper_frame.pt |
| aspirin | spectral | 0 | 0 | 0 | checkpoints/rmd17_aspirin_disjoint_checkpointed/aspirin_flat_spectral_seed0_w0.1_modeper_frame.pt |
| aspirin | spectral | 1 | 0 | 0 | checkpoints/rmd17_aspirin_disjoint_checkpointed/aspirin_flat_spectral_seed1_w0.1_modeper_frame.pt |
| aspirin | spectral | 2 | 0 | 0 | checkpoints/rmd17_aspirin_disjoint_checkpointed/aspirin_flat_spectral_seed2_w0.1_modeper_frame.pt |
| aspirin | spectral | 3 | 0 | 0 | checkpoints/rmd17_aspirin_disjoint_checkpointed/aspirin_flat_spectral_seed3_w0.1_modeper_frame.pt |
| aspirin | spectral | 4 | 0 | 0 | checkpoints/rmd17_aspirin_disjoint_checkpointed/aspirin_flat_spectral_seed4_w0.1_modeper_frame.pt |

## Notes

- These are new strict disjoint-frame, checkpointed rMD17 runs.
- They do not overwrite old rMD17 aspirin 10-seed result files.
- Lower rollout error is better.
