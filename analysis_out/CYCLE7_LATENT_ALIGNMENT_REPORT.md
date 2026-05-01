# Cycle 7 Latent Alignment Diagnostics

Analysis only: no training, ISO17, or rMD17 top-up commands are run.

## Checkpoint Audit

Python torch import available: NO
Checkpoint files discovered under `checkpoints/`: 197
Checkpoint filename encoder tokens: flat=197

Eligible Cycle 2-5 graph-prior node-wise runs:
| source | eligible runs |
| --- | --- |
| cycle2_rmd17_multimolecule | 45 |
| cycle3_ho_networks | 45 |
| cycle4_ho_lambda_robustness | 72 |
| cycle5_ho_lattice_bridge | 75 |

Checkpoint status:
| status | runs |
| --- | --- |
| missing_checkpoint_path | 237 |

Result: no requested Cycle 2/3/4/5 node-wise GNN run records a checkpoint path. Existing `.pt` files are not referenced by these cycle result payloads, and the visible checkpoint naming is dominated by older `flat`/spectral runs rather than the node-wise GNN Cycle 2-5 runs.

## Observed Specificity Available For Alignment

H=32 positive true-graph specificity comparisons in existing results: 15.
| source | domain | item | lambda | control | S_graph H=32 |
| --- | --- | --- | --- | --- | --- |
| cycle2_rmd17_multimolecule | rMD17 | aspirin | 0.1 | permuted_graph | 0.001098 |
| cycle2_rmd17_multimolecule | rMD17 | aspirin | 0.1 | random_graph | -0.000068 |
| cycle2_rmd17_multimolecule | rMD17 | ethanol | 0.1 | permuted_graph | -0.006124 |
| cycle2_rmd17_multimolecule | rMD17 | ethanol | 0.1 | random_graph | -0.004883 |
| cycle2_rmd17_multimolecule | rMD17 | malonaldehyde | 0.1 | permuted_graph | -0.003758 |
| cycle2_rmd17_multimolecule | rMD17 | malonaldehyde | 0.1 | random_graph | -0.004083 |
| cycle2_rmd17_multimolecule | rMD17 | naphthalene | 0.1 | permuted_graph | 0.001231 |
| cycle2_rmd17_multimolecule | rMD17 | naphthalene | 0.1 | random_graph | 0.002335 |
| cycle2_rmd17_multimolecule | rMD17 | toluene | 0.1 | permuted_graph | -0.003506 |
| cycle2_rmd17_multimolecule | rMD17 | toluene | 0.1 | random_graph | -0.003017 |
| cycle5_ho_lattice_bridge | HO | lattice | 0.001 | permuted_graph | -0.004939 |
| cycle5_ho_lattice_bridge | HO | lattice | 0.001 | random_graph | -0.008596 |
| cycle5_ho_lattice_bridge | HO | lattice | 0.005 | permuted_graph | -0.014237 |
| cycle5_ho_lattice_bridge | HO | lattice | 0.005 | random_graph | -0.013498 |
| cycle5_ho_lattice_bridge | HO | lattice | 0.01 | permuted_graph | -0.005165 |
| cycle5_ho_lattice_bridge | HO | lattice | 0.01 | random_graph | -0.006423 |
| cycle5_ho_lattice_bridge | HO | lattice | 0.05 | permuted_graph | -0.001280 |
| cycle5_ho_lattice_bridge | HO | lattice | 0.05 | random_graph | -0.001041 |
| cycle5_ho_lattice_bridge | HO | lattice | 0.1 | permuted_graph | 0.002866 |
| cycle5_ho_lattice_bridge | HO | lattice | 0.1 | random_graph | 0.002965 |

## Diagnostic Answers

- Does latent-space `D_dH` explain high-lambda lattice specificity? Not with the current artifacts. Cycle 5 contains the high-lambda lattice specificity result, but the trained node-wise models were not checkpointed, so `H_t` and `H_{t+1}` cannot be recovered without retraining.
- Is latent alignment more predictive than raw-coordinate alignment from Cycle 6? Unknown. The needed latent-state measurements are unavailable for the requested cycles.
- Does true graph produce lower latent Dirichlet energy than permuted/random controls? Not evaluated. The summary CSV records every eligible run as missing a recoverable checkpoint.
- Should the paper include latent alignment as mechanism or only as negative diagnostic? Do not include it as positive mechanism from current artifacts. At most, mention that raw-coordinate alignment was a negative diagnostic and that latent-state alignment would require checkpointed reruns or future checkpointing.

## Decision

Cycle 7 is blocked by missing checkpoint artifacts, not by a statistical result. The paper should not claim a latent-alignment mechanism unless node-wise checkpoints are produced by a future analysis-only rerun of saved models or by newly checkpointed experiments.

## Files

- `analysis_out/cycle7_latent_alignment_summary.csv`
