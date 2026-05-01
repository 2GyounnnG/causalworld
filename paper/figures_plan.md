# Figures Plan

## Main Figures

1. Long-horizon smoothing summary: H=32 rollout error for GNN none, graph, permuted_graph, and random_graph across Cycle 2 molecules and Cycle 3 HO topologies.
2. Covariance exclusion: aspirin and HO H=32 bars showing none/covariance/graph controls, highlighting that covariance does not recover graph-style gains.
3. rMD17 specificity panel: graph minus permuted/random H=32 paired deltas by molecule, showing weak or negative exact-graph evidence.
4. HO lattice lambda bridge: Cycle 5 H=32 paired deltas versus lambda, with lambda=0.1 separated as the only supported specificity point.
5. Scale-matched control schematic/table: true vs permuted Laplacian spectra identical; random graph spectral scale differs.

## Supplementary Figures

- Full rollout curves over H=1,2,4,8,16,32 for each cycle.
- Cycle 4 lambda robustness heatmap by topology, lambda, and control.
- Cycle 5 seed-wise paired scatter for lattice graph versus permuted/random controls.
- Table of decision-rule outcomes for every H=32 specificity comparison in `analysis_out/master_graph_specificity_table.csv`.
