# Defensive Preflight Experiment Plan for JCS Revision

This note defines reviewer-facing follow-up experiments without modifying existing results or rewriting the manuscript. The intent is defensive: clarify when a quick graph-prior signal is only a screening result, when it persists under a standard budget, and whether the permuted-graph control is fair under the node ordering used by the encoder.

## Reviewer Vulnerabilities

1. N-body quick candidate topology signal reverses under a standard budget.
2. Spectrum-matched permuted graph control may be unfair if the encoder is node-order sensitive.
3. The preflight protocol claims low-cost screening but does not quantify compute savings.

## Mode Definitions

- Quick mode is triage. It asks whether a prior is worth further attention under a deliberately small training budget.
- Standard mode is a persistence check. It asks whether a quick signal survives when the base model has more opportunity to fit the dynamics.
- Audit mode is a mechanism check. It inspects learned latent graph-frequency behavior and should be used before topology-specific claims.
- A quick candidate topology signal is not final topology-specific evidence. It can motivate standard checks and audits, but it should not be reported as a definitive topology attribution by itself.

## 1. N-body Robustness Runner

Script: `scripts/run_nbody_robustness_plan.py`

Default design:

- dataset: `nbody_distance`
- particles: 36
- distance-k values: 4, 8, 12
- seeds: 0, 1, 2
- priors: `none`, `graph`, `permuted_graph`, calibrated `temporal_smooth`
- quick epochs: 5
- standard epochs: 20
- horizons: 16, 32
- outputs: `analysis_out/preflight_runs/nbody_robustness/`

Dry-run command:

```bash
python scripts/run_nbody_robustness_plan.py --dry-run
```

Actual run command:

```bash
python scripts/run_nbody_robustness_plan.py
```

The runner is resumable at the output-directory level: it skips a condition when `summary.csv` already contains a `classification` row unless `--force` is passed.

## 2. Node-Order Sanity Check

Script: `scripts/run_node_order_sanity_check.py`

Default design:

- dataset: `nbody_distance`
- conditions:
  - original node order
  - consistently permuted nodes, where node features and graph labels are permuted together
  - mismatched original features with permuted graph labels as a stress-control condition
- prior: `graph`
- seeds: 0, 1, 2
- epochs: 5
- outputs: `analysis_out/preflight_runs/node_order_sanity/`

Dry-run command:

```bash
python scripts/run_node_order_sanity_check.py --dry-run
```

Actual run command:

```bash
python scripts/run_node_order_sanity_check.py
```

Interpretation:

- If original and consistently permuted-node rollouts match closely, the encoder/training path is not showing a material node-order artifact for this check.
- If they differ materially, the permuted-graph control needs more careful framing because node ordering may influence the encoder path.
- The mismatched condition is expected to be harder or unfair; it is included only to expose the magnitude of label mismatch effects.

## 3. Compute Accounting

Script: `scripts/account_preflight_compute.py`

Default behavior:

- scans `analysis_out/preflight_runs/**/run_config.json`
- reads sibling `summary.csv` files where available
- estimates cost as `number of trainings x epochs x train_transitions`
- writes `paper/tables/preflight_compute_cost.md`
- compares observed quick preflight, standard check, audit-mode runs, and a configurable hypothetical full sweep

Command:

```bash
python scripts/account_preflight_compute.py
```

The output table reports relative transition-epoch units, not wall-clock time. This is the appropriate scale estimate for the manuscript claim that preflight is a low-cost screening protocol rather than a replacement for full sweeps.
