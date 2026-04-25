# rMD17 Multimolecule Disjoint Checkpointed

## Status

- ethanol / none: 5/5 seeds complete
- ethanol / euclidean: 5/5 seeds complete
- ethanol / spectral: 5/5 seeds complete
- malonaldehyde / none: 5/5 seeds complete
- malonaldehyde / euclidean: 5/5 seeds complete
- malonaldehyde / spectral: 5/5 seeds complete
- checkpoints present for completed runs: 30
- frame-overlap audit: PASS

## H=16 Aggregate

| molecule | prior | n | mean | std | median | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ethanol | euclidean | 5 | 0.115 | 0.080 | 0.106 | 0.042 | 0.245 |
| ethanol | none | 5 | 0.115 | 0.080 | 0.106 | 0.042 | 0.245 |
| ethanol | spectral | 5 | 0.031 | 0.016 | 0.033 | 0.008 | 0.051 |
| malonaldehyde | euclidean | 5 | 0.084 | 0.046 | 0.060 | 0.043 | 0.144 |
| malonaldehyde | none | 5 | 0.079 | 0.052 | 0.060 | 0.026 | 0.144 |
| malonaldehyde | spectral | 5 | 0.036 | 0.025 | 0.024 | 0.013 | 0.069 |

## Frame-Overlap Audit

| molecule | prior | seed | train_eval_start_overlap_count | train_eval_overlap_count | checkpoint_path |
| --- | --- | --- | --- | --- | --- |
| ethanol | euclidean | 0 | 0 | 0 | checkpoints/rmd17_multimolecule_disjoint_checkpointed/ethanol_flat_euclidean_seed0_w0.1_modeper_frame.pt |
| ethanol | euclidean | 1 | 0 | 0 | checkpoints/rmd17_multimolecule_disjoint_checkpointed/ethanol_flat_euclidean_seed1_w0.1_modeper_frame.pt |
| ethanol | euclidean | 2 | 0 | 0 | checkpoints/rmd17_multimolecule_disjoint_checkpointed/ethanol_flat_euclidean_seed2_w0.1_modeper_frame.pt |
| ethanol | euclidean | 3 | 0 | 0 | checkpoints/rmd17_multimolecule_disjoint_checkpointed/ethanol_flat_euclidean_seed3_w0.1_modeper_frame.pt |
| ethanol | euclidean | 4 | 0 | 0 | checkpoints/rmd17_multimolecule_disjoint_checkpointed/ethanol_flat_euclidean_seed4_w0.1_modeper_frame.pt |
| ethanol | none | 0 | 0 | 0 | checkpoints/rmd17_multimolecule_disjoint_checkpointed/ethanol_flat_none_seed0_w0.1_modeper_frame.pt |
| ethanol | none | 1 | 0 | 0 | checkpoints/rmd17_multimolecule_disjoint_checkpointed/ethanol_flat_none_seed1_w0.1_modeper_frame.pt |
| ethanol | none | 2 | 0 | 0 | checkpoints/rmd17_multimolecule_disjoint_checkpointed/ethanol_flat_none_seed2_w0.1_modeper_frame.pt |
| ethanol | none | 3 | 0 | 0 | checkpoints/rmd17_multimolecule_disjoint_checkpointed/ethanol_flat_none_seed3_w0.1_modeper_frame.pt |
| ethanol | none | 4 | 0 | 0 | checkpoints/rmd17_multimolecule_disjoint_checkpointed/ethanol_flat_none_seed4_w0.1_modeper_frame.pt |
| ethanol | spectral | 0 | 0 | 0 | checkpoints/rmd17_multimolecule_disjoint_checkpointed/ethanol_flat_spectral_seed0_w0.1_modeper_frame.pt |
| ethanol | spectral | 1 | 0 | 0 | checkpoints/rmd17_multimolecule_disjoint_checkpointed/ethanol_flat_spectral_seed1_w0.1_modeper_frame.pt |
| ethanol | spectral | 2 | 0 | 0 | checkpoints/rmd17_multimolecule_disjoint_checkpointed/ethanol_flat_spectral_seed2_w0.1_modeper_frame.pt |
| ethanol | spectral | 3 | 0 | 0 | checkpoints/rmd17_multimolecule_disjoint_checkpointed/ethanol_flat_spectral_seed3_w0.1_modeper_frame.pt |
| ethanol | spectral | 4 | 0 | 0 | checkpoints/rmd17_multimolecule_disjoint_checkpointed/ethanol_flat_spectral_seed4_w0.1_modeper_frame.pt |
| malonaldehyde | euclidean | 0 | 0 | 0 | checkpoints/rmd17_multimolecule_disjoint_checkpointed/malonaldehyde_flat_euclidean_seed0_w0.1_modeper_frame.pt |
| malonaldehyde | euclidean | 1 | 0 | 0 | checkpoints/rmd17_multimolecule_disjoint_checkpointed/malonaldehyde_flat_euclidean_seed1_w0.1_modeper_frame.pt |
| malonaldehyde | euclidean | 2 | 0 | 0 | checkpoints/rmd17_multimolecule_disjoint_checkpointed/malonaldehyde_flat_euclidean_seed2_w0.1_modeper_frame.pt |
| malonaldehyde | euclidean | 3 | 0 | 0 | checkpoints/rmd17_multimolecule_disjoint_checkpointed/malonaldehyde_flat_euclidean_seed3_w0.1_modeper_frame.pt |
| malonaldehyde | euclidean | 4 | 0 | 0 | checkpoints/rmd17_multimolecule_disjoint_checkpointed/malonaldehyde_flat_euclidean_seed4_w0.1_modeper_frame.pt |
| malonaldehyde | none | 0 | 0 | 0 | checkpoints/rmd17_multimolecule_disjoint_checkpointed/malonaldehyde_flat_none_seed0_w0.1_modeper_frame.pt |
| malonaldehyde | none | 1 | 0 | 0 | checkpoints/rmd17_multimolecule_disjoint_checkpointed/malonaldehyde_flat_none_seed1_w0.1_modeper_frame.pt |
| malonaldehyde | none | 2 | 0 | 0 | checkpoints/rmd17_multimolecule_disjoint_checkpointed/malonaldehyde_flat_none_seed2_w0.1_modeper_frame.pt |
| malonaldehyde | none | 3 | 0 | 0 | checkpoints/rmd17_multimolecule_disjoint_checkpointed/malonaldehyde_flat_none_seed3_w0.1_modeper_frame.pt |
| malonaldehyde | none | 4 | 0 | 0 | checkpoints/rmd17_multimolecule_disjoint_checkpointed/malonaldehyde_flat_none_seed4_w0.1_modeper_frame.pt |
| malonaldehyde | spectral | 0 | 0 | 0 | checkpoints/rmd17_multimolecule_disjoint_checkpointed/malonaldehyde_flat_spectral_seed0_w0.1_modeper_frame.pt |
| malonaldehyde | spectral | 1 | 0 | 0 | checkpoints/rmd17_multimolecule_disjoint_checkpointed/malonaldehyde_flat_spectral_seed1_w0.1_modeper_frame.pt |
| malonaldehyde | spectral | 2 | 0 | 0 | checkpoints/rmd17_multimolecule_disjoint_checkpointed/malonaldehyde_flat_spectral_seed2_w0.1_modeper_frame.pt |
| malonaldehyde | spectral | 3 | 0 | 0 | checkpoints/rmd17_multimolecule_disjoint_checkpointed/malonaldehyde_flat_spectral_seed3_w0.1_modeper_frame.pt |
| malonaldehyde | spectral | 4 | 0 | 0 | checkpoints/rmd17_multimolecule_disjoint_checkpointed/malonaldehyde_flat_spectral_seed4_w0.1_modeper_frame.pt |

## Notes

- These are new strict disjoint-frame, checkpointed rMD17 runs.
- They do not overwrite old rMD17 aspirin 10-seed result files.
- Lower rollout error is better.
