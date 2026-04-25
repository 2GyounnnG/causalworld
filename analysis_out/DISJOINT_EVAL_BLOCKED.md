# Disjoint rMD17 Evaluation Blocked

The disjoint rMD17 aspirin evaluation was not generated because no trained
checkpoint files were found for the completed 10-seed run.

Searched for checkpoint-like files with suffixes `.pt`, `.pth`, `.ckpt`, and
`.checkpoint`, including aspirin/rMD17/prior/seed naming patterns. No matches
were found.

See `analysis_out/CHECKPOINTING_FOR_DISJOINT_EVAL.md` for the forward path.

## Missing Checkpoints

The required checkpoint set is:

- molecule=`aspirin`, encoder=`flat`, prior=`none`, seeds=`0,1,2,3,4,5,6,7,8,9`
- molecule=`aspirin`, encoder=`flat`, prior=`euclidean`, seeds=`0,1,2,3,4,5,6,7,8,9`
- molecule=`aspirin`, encoder=`flat`, prior=`spectral`, seeds=`0,1,2,3,4,5,6,7,8,9`

## Rerun Command

The completed comparison was produced by the rMD17 10-seed runner:

```bash
/home/user/miniforge3/envs/causalworld/bin/python scripts/run_rmd17_10seed.py
```

However, the current training path records metrics and configs but does not save
model checkpoints. A checkpoint-saving rerun would be needed before
`scripts/evaluate_rmd17_disjoint.py` can load trained models and compute a
disjoint eval without retraining.

The checkpointed rerun command, if explicitly needed for reviewer-facing
disjoint evaluation, is:

```bash
python scripts/run_rmd17_10seed_checkpointed.py
```

## Recommendation

For future reviewer-facing leakage audits, persist both:

- trained model checkpoints per molecule/encoder/prior/seed
- sampled train/eval `frame_idx` sets in run metadata
