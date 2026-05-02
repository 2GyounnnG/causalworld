# causalworld

Code and assets for research on graph priors and latent dynamics.

## JCS submission

This repository accompanies the manuscript

> **A Model-Conditioned Preflight Protocol for Prior Selection in Node-Wise Scientific Latent Dynamics**
> Ruiqi Wang
> *Submitted to the Journal of Computational Science (Elsevier), 2026*

The submission state is on the `cycle0-relational-structure` branch and tagged `v1.0-jcs-submission`.

```bash
git checkout v1.0-jcs-submission
```

### Manuscript

| Path | Contents |
|------|----------|
| `paper/jcs_main.tex` | LaTeX source |
| `paper/jcs_main.pdf` | Compiled manuscript |
| `paper/cover_letter.md` | Cover letter |
| `paper/references.bib` | Bibliography |
| `paper/figures/` | Figures referenced from `jcs_main.tex` |
| `paper/figures/figure_data/` | Source data for selected figures |
| `paper/scripts/` | Figure generation scripts |
| `paper/tables/*_summary.md` | Aggregated multi-seed result tables |
| `paper/tables/headline_attribution_table.md` | Headline 13-cell attribution table |
| `paper/tables/effect_size_unified.md` | Unified effect-size table |
| `paper/tables/master_multiseed_summary.md` | Master multi-seed summary across all regimes |

### Implementation

| Path | Contents |
|------|----------|
| `scripts/graph_prior_preflight_check.py` | Adapter-based preflight entry point |
| `scripts/preflight_multiseed_common.py` | Multi-seed common utilities |
| `scripts/run_*_5seed.py`, `run_*_3seed.py` | Per-regime multi-seed campaign drivers |
| `scripts/aggregate_*.py` | Per-regime aggregation scripts |
| `scripts/run_full_campaign.sh` | Full multi-seed campaign launcher |
| `scripts/train_cycle0.py`, `train_cycle3_ho_networks.py` | Training entry points used by adapters |
| `analysis_out/preflight_runs/` | Per-regime, per-seed raw run artifacts |
| `analysis_out/*_REPORT.md` | Per-regime multi-seed summary reports |

### Reproducing manuscript numbers

Each measurement cell reported in the paper corresponds to a directory under
`analysis_out/preflight_runs/`. Each directory contains `run_config.json`,
`summary.csv`, and the raw rollout artifacts for every seed. Aggregated
summary tables are in `paper/tables/`.

Example — re-aggregate the N-body 5-seed cell from raw artifacts:

```bash
conda activate causalworld
python scripts/aggregate_nbody_robustness_5seed.py
```

Example — re-generate Figure 4 (N-body standard-budget rollout):

```bash
python paper/scripts/make_fig_nbody_standard.py
```

### Other branches

The `main` branch contains earlier ablation work that is unrelated to this
manuscript and is not maintained as part of the JCS submission.

## Contact

Ruiqi Wang — riw010@ucsd.edu
Department of Physics, University of California San Diego
