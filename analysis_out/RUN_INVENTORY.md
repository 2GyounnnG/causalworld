# Run Inventory

Repo root inspected: `/home/user/projects/causalworld`

## Repo Tree

```
.codex
.git
.gitignore
1.0
20,
__pycache__
analysis_out
backups
causal_graph.png
codex_bootstrap.txt
codex_session.log
data
diagnose.py
diagnostics.md
diagnostics_results.json
full_summary.json
horizon_gap.png
hypergraph_env.py
logs
long_horizon.json
long_horizon.png
midscale_summary.json
model.py
overnight_brief.txt
overnight_log.txt
overnight_summary.md
requirements-60ti.txt
requirements.txt
results.json
rmd17_aspirin_10seed_results.json
rmd17_aspirin_10seed_results.pre_restart_20260422_214952.json
rmd17_aspirin_10seed_results.pre_resume_20260422_221851.json
rmd17_aspirin_laplacian_ablation.json
rmd17_aspirin_results.json
rmd17_aspirin_smoke.json
rmd17_aspirin_weight_sweep.json
rmd17_ethanol_results.json
rmd17_malonaldehyde_results.json
run_in_tmux.sh
run_in_tmux.sh.bak
scripts
train.py
validation_10seed_flat.json
validation_10seed_flat.png
validation_10seed_flat_200ep.json
validation_10seed_hypergraph.json
validation_10seed_hypergraph.png
validation_10seed_hypergraph_200ep.json
validation_wolfram_flat_10seed_200ep.json
validation_wolfram_flat_10seed_200ep.pre_restart_20260422_214952.json
weight_sweep.png
analysis_out/AUDIT_PRIORS_AND_LAPLACIANS.md
analysis_out/CHECKPOINTING_FOR_DISJOINT_EVAL.md
analysis_out/DISJOINT_EVAL_BLOCKED.md
analysis_out/RESULTS_SUMMARY_FOR_PAPER.md
analysis_out/RUN_INVENTORY.md
analysis_out/WEIGHT_SWEEP_INSTABILITY.md
analysis_out/WOLFRAM_INSTABILITY.md
analysis_out/aggregate_by_prior_horizon.csv
analysis_out/aggregate_laplacian_ablation.csv
analysis_out/aggregate_weight_sweep.csv
analysis_out/aggregate_wolfram.csv
analysis_out/duplicate_manifest_rows.csv
analysis_out/failed_or_incomplete_runs.csv
analysis_out/manifest.csv
analysis_out/manifest_raw.csv
analysis_out/plots
analysis_out/weight_sweep_horizon_ratios.csv
analysis_out/wolfram_horizon_ratios.csv
backups/rmd17_aspirin_laplacian_ablation.20260424_005456.json
backups/rmd17_aspirin_weight_sweep.20260424_005456.json
backups/validation_wolfram_flat_10seed_200ep.20260424_005456.json
data/rmd17_raw
logs/flat_200ep_20260421_021613.log
logs/hyper_200ep_20260421_092926.log
logs/hyper_200ep_20260421_094750.log
logs/hyper_200ep_20260421_095059.log
logs/rmd17_10s_20260422_045621.log
logs/rmd17_aspirin_20260421_102329.log
logs/rmd17_aspirin_20260421_142952.log
logs/rmd17_ethanol_20260421_102501.log
logs/rmd17_ethanol_20260421_142952.log
logs/rmd17_euc_redo2_20260423_114516.log
logs/rmd17_euc_redo_20260423_025149.log
logs/rmd17_lap_ablation_20260423_212621.log
logs/rmd17_lap_ablation_resume_20260424_005456.log
logs/rmd17_malonaldehyde_20260421_102512.log
logs/rmd17_malonaldehyde_20260421_142952.log
logs/rmd17_spec_resume_20260422_221936.log
logs/rmd17_weight_sweep_20260423_212615.log
logs/rmd17_weight_sweep_resume_20260424_005456.log
logs/wolfram_200ep_20260422_050446.log
logs/wolfram_resume2_20260423_114521.log
logs/wolfram_resume_20260423_025154.log
logs/wolfram_resume_20260424_005456.log
scripts/aggregate_results.py
scripts/analyze_weight_instability.py
scripts/analyze_wolfram_instability.py
scripts/audit_priors_and_laplacians.py
scripts/build_manifest.py
scripts/inspect_collapse.py
scripts/plot_results.py
scripts/resume_rmd17_aspirin_spectral.py
scripts/rmd17_loader.py
scripts/run_10seed_flat.py
scripts/run_10seed_flat_200ep.py
scripts/run_10seed_hypergraph.py
scripts/run_10seed_hypergraph_200ep.py
scripts/run_analysis_pipeline.sh
scripts/run_long_horizon.py
scripts/run_rmd17_10seed.py
scripts/run_rmd17_10seed_checkpointed.py
scripts/run_rmd17_euclidean_redo.py
scripts/run_rmd17_laplacian_ablation.py
scripts/run_rmd17_weight_sweep.py
scripts/run_wolfram_flat_10seed_200ep.py
scripts/run_wolfram_resume.py
scripts/train_rmd17.py
analysis_out/plots/rmd17_aspirin_horizon_scaling.pdf
analysis_out/plots/rmd17_aspirin_horizon_scaling.png
analysis_out/plots/rmd17_laplacian_ablation.pdf
analysis_out/plots/rmd17_laplacian_ablation.png
analysis_out/plots/rmd17_weight_sweep.pdf
analysis_out/plots/rmd17_weight_sweep.png
analysis_out/plots/wolfram_horizon_scaling.pdf
analysis_out/plots/wolfram_horizon_scaling.png
data/rmd17_raw/rmd17
data/rmd17_raw/rmd17.tar.bz2
```

## Current Result Inventory

- `rmd17_aspirin_10seed`: 150 rollout metric rows
- `rmd17_aspirin_3seed_pilot`: 45 rollout metric rows
- `rmd17_ethanol_3seed_pilot`: 45 rollout metric rows
- `rmd17_laplacian_ablation`: 75 rollout metric rows
- `rmd17_malonaldehyde_3seed_pilot`: 45 rollout metric rows
- `rmd17_weight_sweep`: 180 rollout metric rows
- `wolfram_flat_10seed`: 150 rollout metric rows
- `wolfram_flat_10seed_200ep`: 150 rollout metric rows
- `wolfram_hypergraph_10seed`: 150 rollout metric rows
- `wolfram_long_horizon`: 63 rollout metric rows

## Incomplete, Failed, Or Duplicate Signals

- Duplicate rollout rows removed from `manifest.csv`: 295 (affected experiments: `rmd17_aspirin_10seed`, `wolfram_flat_10seed_200ep`, `wolfram_hypergraph_10seed`)
- `rmd17_aspirin_10seed_results.pre_restart_20260422_214952.json`: duplicate_snapshot (150/?) archival snapshot; excluded from aggregate conclusions
- `rmd17_aspirin_10seed_results.pre_restart_20260422_214952.json`: incomplete (125/150) expected 3 priors x 10 seeds x 5 horizons; checked by (prior, seed, horizon)
- `rmd17_aspirin_10seed_results.pre_resume_20260422_221851.json`: duplicate_snapshot (150/?) archival snapshot; excluded from aggregate conclusions
- `rmd17_aspirin_10seed_results.pre_resume_20260422_221851.json`: incomplete (125/150) expected 3 priors x 10 seeds x 5 horizons; checked by (prior, seed, horizon)
- `validation_10seed_flat_200ep.json`: duplicate_snapshot (45/?) archival snapshot; excluded from aggregate conclusions
- `validation_wolfram_flat_10seed_200ep.pre_restart_20260422_214952.json`: duplicate_snapshot (30/?) archival snapshot; excluded from aggregate conclusions
- `validation_wolfram_flat_10seed_200ep.pre_restart_20260422_214952.json`: incomplete (30/150) expected 3 priors x 10 seeds x 5 horizons; checked by (prior, seed, horizon)
- `codex_session.log`: failed (starts=0; saves=0; lines=845/?) Script done on 2026-04-21 01:48:38+08:00 [COMMAND_EXIT_CODE="130"]
- `logs/flat_200ep_20260421_021613.log`: incomplete (starts=1; saves=0; lines=2123/?) [flat|euclidean|seed=3] epoch 119/200 total=0.056 prior=3.799 H16_eval=0.26
- `logs/hyper_200ep_20260421_092926.log`: failed (starts=1; saves=0; lines=7/?) === Finished at Tue Apr 21 09:29:26 CST 2026 (exit=$?) ===
- `logs/hyper_200ep_20260421_094750.log`: failed (starts=1; saves=0; lines=7/?) === Finished at Tue Apr 21 09:47:50 CST 2026 (exit=$?) ===
- `logs/hyper_200ep_20260421_095059.log`: incomplete (starts=1; saves=0; lines=676/?) [hyper|none|seed=1] epoch 71/200 total=0.27 prior=0.00 H16_eval=0.84
- `logs/rmd17_10s_20260422_045621.log`: incomplete (starts=27; saves=25; lines=366/?) [aspirin|flat|spectral|seed=5] epoch  45/50 loss=0.0067 H8_eval=0.0837
- `logs/rmd17_lap_ablation_20260423_212621.log`: incomplete (starts=8; saves=6; lines=89/?) [aspirin|flat|spectral|seed=1] epoch  25/50 loss=0.0065 H8_eval=0.0542
- `logs/rmd17_lap_ablation_resume_20260424_005456.log`: incomplete (starts=15; saves=15; lines=196/?)   -> saved partial results to rmd17_aspirin_laplacian_ablation.json
- `logs/rmd17_spec_resume_20260422_221936.log`: incomplete (starts=5; saves=5; lines=79/?) DONE rmd17 aspirin spectral resume
- `logs/rmd17_weight_sweep_20260423_212615.log`: incomplete (starts=8; saves=6; lines=93/?) [aspirin|flat|spectral|seed=0] epoch  45/50 loss=0.0016 H8_eval=0.0346
- `logs/rmd17_weight_sweep_resume_20260424_005456.log`: incomplete (starts=27; saves=27; lines=352/?)   -> saved partial results to rmd17_aspirin_weight_sweep.json
- `logs/wolfram_200ep_20260422_050446.log`: incomplete (starts=1; saves=2; lines=96/?) [flat|spectral|seed=2] epoch 180/200 total=0.001 trans=0.000 prior=0.080 rollout_H8=0.06
- `logs/wolfram_resume2_20260423_114521.log`: incomplete (starts=1; saves=2; lines=79/?) [flat|euclidean|seed=6] epoch 40/200 total=0.038 trans=0.000 prior=3.754 rollout_H8=0.06
- `logs/wolfram_resume_20260423_025154.log`: incomplete (starts=1; saves=2; lines=69/?) [flat|none|seed=4] epoch 40/200 total=0.000 trans=0.000 prior=0.000 rollout_H8=0.04
- `logs/wolfram_resume_20260424_005456.log`: incomplete (starts=0; saves=4; lines=126/?) [seed=9] saved partial results to validation_wolfram_flat_10seed_200ep.json

## Notes

- Raw experiment outputs were not modified.
- Snapshot files are retained in the manifest as `duplicate_snapshot` and excluded from aggregate conclusions.
- Current partial experiments are still represented in aggregates, with incomplete status recorded separately.
