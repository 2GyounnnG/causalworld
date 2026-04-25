# Checkpointing For Disjoint rMD17 Evaluation

Disjoint evaluation is blocked for the already completed rMD17 aspirin 10-seed
run because those runs saved metrics and configs, but not trained model
checkpoints. Without the trained weights, a disjoint train/eval frame audit
cannot re-evaluate the existing models without retraining.

Future rMD17 runs can now opt into checkpoint and frame-index persistence.
`scripts/train_rmd17.py` supports this through:

- `save_checkpoint=True`
- `checkpoint_dir`
- `save_frame_indices=True`

Checkpointed runs save:

- trained model state
- run config
- final loss
- rollout errors
- git commit when available
- sampled train/eval frame indices in result metadata

The checkpointed aspirin 10-seed runner writes checkpoints under:

```text
checkpoints/rmd17/aspirin_10seed_checkpointed/
```

and writes a separate result file:

```text
rmd17_aspirin_10seed_checkpointed_results.json
```

It does not overwrite the existing completed 10-seed result file.

Do not run this rerun unless it is explicitly needed for reviewer-facing
disjoint evaluation.

Run the checkpointed rerun later with:

```bash
python scripts/run_rmd17_10seed_checkpointed.py
```

After that rerun exists, a disjoint evaluator can load the saved checkpoints,
exclude the persisted training `frame_idx` values, and recompute rollout metrics
on disjoint evaluation frames.
