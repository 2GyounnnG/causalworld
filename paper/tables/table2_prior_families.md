# Table 2. Prior Families and Attribution Roles

| prior family | definition | attribution role | manuscript use |
| --- | --- | --- | --- |
| `none` | No auxiliary prior. | Establishes whether any prior improves rollout. | Baseline for all preflight recommendations. |
| `graph_laplacian` | Candidate-graph Laplacian smoothness on learned node-wise latents, `Tr(H^T L H)`. | Tests whether candidate graph-frequency smoothing helps the tested model. | Main prior under evaluation, not introduced as a new prior. |
| `permuted_graph` | Spectrum-matched permuted Laplacian, `L_perm = P^T L P`. | Preserves Laplacian spectrum while removing node-label semantics. | Decisive control for topology attribution. |
| `random_graph` | Optional graph with matched coarse size or edge statistics. | Tests whether an unrelated graph can produce similar smoothing. | Secondary control; weaker than spectrum-matched permutation. |
| `temporal_smooth` | Graph-free smoothness on learned latent temporal deltas, e.g. `sum_t ||H_{t+1} - H_t||_F^2`. | Tests whether temporal regularization explains graph-prior gains. | Must be calibrated when compared with graph priors. |
| future covariance/energy priors | Planned auxiliary baselines for additional scientific structure. | Broaden the preflight family set. | Mentioned as future extensions, not current core evidence. |

Source: `paper/jcs_draft_methods.md`, `paper/tool_readme_draft.md`.
