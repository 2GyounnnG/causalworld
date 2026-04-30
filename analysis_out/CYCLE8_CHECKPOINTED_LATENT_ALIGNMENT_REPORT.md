# Cycle 8 Checkpointed Latent Alignment Report

Experiment: `cycle8_checkpointed_lattice_latent_alignment`
Analysis includes only HO lattice, GNN encoder, priors graph/permuted_graph/random_graph, lambda=0.1, seeds 0-4.

## Run Integrity

Success/failure count: 15 ok / 0 failed
Checkpoint availability: 15/15
Latent artifact availability: 15/15

## Rollout Error

| prior | H=16 | H=32 |
| --- | --- | --- |
| graph | 0.0056 +/- 0.0018 (n=5) | 0.0169 +/- 0.0054 (n=5) |
| permuted_graph | 0.0066 +/- 0.0016 (n=5) | 0.0198 +/- 0.0059 (n=5) |
| random_graph | 0.0067 +/- 0.0016 (n=5) | 0.0199 +/- 0.0062 (n=5) |

## Latent Dirichlet Energy

| prior | D_true(Delta_H) | D_prior(Delta_H) | D_perm(Delta_H) | D_rand(Delta_H) | D_true_norm(Delta_H) |
| --- | --- | --- | --- | --- | --- |
| graph | 0.0000 +/- 0.0000 (n=5) | 0.0000 +/- 0.0000 (n=5) | 0.0001 +/- 0.0000 (n=5) | 0.0001 +/- 0.0000 (n=5) | 2.8106 +/- 0.0413 (n=5) |
| permuted_graph | 0.0001 +/- 0.0000 (n=5) | 0.0001 +/- 0.0000 (n=5) | 0.0001 +/- 0.0000 (n=5) | 0.0001 +/- 0.0000 (n=5) | 3.5109 +/- 0.1120 (n=5) |
| random_graph | 0.0001 +/- 0.0000 (n=5) | 0.0001 +/- 0.0000 (n=5) | 0.0001 +/- 0.0000 (n=5) | 0.0001 +/- 0.0000 (n=5) | 3.5237 +/- 0.0898 (n=5) |

## Low-Frequency Ratio On True L

| prior | R_low K=2 | R_low K=4 | R_low K=8 |
| --- | --- | --- | --- |
| graph | 0.1036 +/- 0.0088 (n=5) | 0.1866 +/- 0.0118 (n=5) | 0.3153 +/- 0.0145 (n=5) |
| permuted_graph | 0.0640 +/- 0.0069 (n=5) | 0.1216 +/- 0.0087 (n=5) | 0.2114 +/- 0.0116 (n=5) |
| random_graph | 0.0637 +/- 0.0067 (n=5) | 0.1207 +/- 0.0080 (n=5) | 0.2099 +/- 0.0112 (n=5) |

## Paired Latent Alignment

| control model | paired delta D_true_norm(control - graph) | graph lower count |
| --- | --- | --- |
| permuted_graph | 0.7003 +/- 0.1304 (n=5) | 5/5 |
| random_graph | 0.7131 +/- 0.1123 (n=5) | 5/5 |

## Correlation With H=32 Rollout

| latent metric | n | Pearson r | Spearman rho |
| --- | --- | --- | --- |
| D_true_Delta_H_norm | 15 | 0.1422 | 0.0607 |
| D_prior_Delta_H_norm | 15 | 0.2400 | 0.1607 |
| D_perm_Delta_H_norm | 15 | 0.2495 | 0.4643 |
| D_rand_Delta_H_norm | 15 | -0.1592 | -0.1071 |

## Conclusion

Can latent alignment explain high-lambda lattice specificity? YES. The strict explanation criterion used here is that the graph-prior model has both lower H=32 rollout error and lower `D_true_norm(Delta_H)` than both control-prior models.
If the paired latent-energy deltas are positive and the rollout table reproduces the graph advantage, latent alignment is a plausible mechanism. If rollout specificity reproduces without lower true-graph latent energy, latent alignment should remain future work or a negative diagnostic.

## Files

- `analysis_out/cycle8_latent_alignment_summary.csv`
