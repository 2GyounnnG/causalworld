# Node-Order Sanity Check

Created: `2026-05-01T09:04:06.597898+00:00`

Purpose: test whether the spectrum-matched permuted-graph control is confounded by node-order sensitivity.
The consistent condition permutes node features and graph labels together; a large rollout shift versus original would indicate order sensitivity in the encoder/training path.
The mismatched condition keeps original features with a permuted graph and is an optional unfair-control stress test.

| condition | seeds | H=16 | H=32 |
| --- | --- | --- | --- |
| consistent_permuted_nodes | 3 | 0.229157 | 0.872711 |
| mismatched_original_x_permuted_l | 3 | 0.22962 | 0.870892 |
| original | 3 | 0.229157 | 0.872711 |

Interpretation:
- quick mode is triage only; it asks whether a prior deserves more attention.
- standard mode is a persistence check under a larger training budget.
- audit mode is a mechanism check on learned latent smoothing.
- a quick candidate topology signal is not final topology-specific evidence.
