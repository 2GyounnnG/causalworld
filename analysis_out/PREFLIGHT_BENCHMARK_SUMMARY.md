# Graph Prior Preflight Benchmark Summary

Executive summary: Existing completed preflight runs show that the tool is useful as a control-oriented screen, not as a standalone proof that a graph prior uses scientific topology. The benchmark set now spans representative node-wise physical prediction regimes with candidate graph structure. The clearest repeated pattern is budget-dependent: true graph priors can help low-budget rollout, but at longer training budgets no-prior or spectrum-matched smoothing controls can catch up or outperform.

No new experiments, ISO17, rMD17 top-up, METR-LA standard run, full sweeps, or model-logic changes were performed. This update only reads existing outputs and writes:

- `analysis_out/PREFLIGHT_BENCHMARK_SUMMARY.md`
- `analysis_out/preflight_benchmark_summary.csv`

## Benchmark Table

| case | role | graph source | epochs | H32 none | H32 graph | H32 permuted | graph gain vs none | true vs permuted | classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HO lattice | controlled positive case | true lattice | 5 | 1.084877 | 1.013924 | 1.067758 | +6.54% | +5.04% | topology_aligned_latent_smoothing |
| Graph heat lattice | graph-generated cautionary case | true generator graph | 5 | 0.603606 | 0.690180 | 0.605792 | -14.34% | -13.93% | no_graph_gain |
| Graph low-frequency lattice | synthetic alignment stress test | constructed true low-frequency Laplacian basis | 5; analysis 20/50 | 0.720953 | 0.691928 | 0.670257 | +4.03% | -3.23% | generic_smoothing |
| Spring-mass lattice quick | second-order mechanics, low budget | true spring lattice | 5 | 0.686395 | 0.573988 | 0.584110 | +16.38% | +1.73% | topology_aligned_latent_smoothing |
| Spring-mass lattice standard | second-order mechanics, longer budget | true spring lattice | 20 | 0.080921 | 0.104796 | 0.123826 | -29.51% | +15.37% | no_graph_gain |
| Graph-wave lattice quick | PDE-like wave dynamics, low budget | true wave lattice | 5 | 0.682496 | 0.571246 | 0.575382 | +16.30% | +0.72% | topology_aligned_latent_smoothing |
| Graph-wave lattice standard | PDE-like wave dynamics, longer budget | true wave lattice | 20 | 0.082617 | 0.095294 | 0.102072 | -15.34% | +6.64% | no_graph_gain |
| N-body distance quick | long-range particle interaction, low budget | initial distance-kNN weighted graph | 5 | 0.821576 | 0.543263 | 0.560362 | +33.88% | +3.05% | topology_aligned_latent_smoothing |
| N-body distance standard | long-range particle interaction, longer budget | initial distance-kNN weighted graph | 20 | 0.156127 | 0.113463 | 0.089004 | +27.33% | -27.48% | generic_smoothing |
| METR-LA correlation T=2000 | main real traffic application case | correlation_topk sensor graph | 5 | 0.406830 | 0.446130 | 0.433669 | -9.66% | -2.87% | no_graph_gain |
| METR-LA correlation smoke | short traffic smoke check | correlation_topk sensor graph | 1 | 3.755799 | 3.706100 | 3.697807 | +1.32% | -0.22% | generic_smoothing |

## Physical Prediction Coverage

The preflight benchmark now covers representative node-wise physical prediction regimes with candidate graph structure. It is not universal coverage of all physics, and the candidate graph may be incomplete or only approximate.

Coverage now includes:

- local coupled oscillator dynamics via HO lattice.
- second-order spring mechanics via `spring_mass_lattice`.
- graph-discretized wave/PDE-like dynamics via `graph_wave_lattice`.
- graph diffusion/heat dynamics via `graph_heat_lattice`.
- spectral low-frequency graph dynamics via `graph_lowfreq_lattice`.
- long-range particle interactions via `nbody_distance`.
- molecular dynamics via earlier rMD17 cycles.
- real traffic spatiotemporal forecasting via METR-LA correlation graph.

This breadth is useful for stress-testing attribution claims because the regimes differ in whether the candidate graph is the actual simulator graph, a low-frequency construction, an approximate distance graph, or a data-derived correlation graph.

## Emergent Pattern: Early Topology-Aligned Regularization, Late Catch-Up

Across the newer physical adapters, a consistent budget-dependent pattern appears.

- Spring-mass: at 5 epochs the true graph is best at H=32 (`0.573988` versus none `0.686395` and permuted `0.584110`); at 20 epochs no-prior is best (`0.080921` versus graph `0.104796`).
- Graph-wave: at 5 epochs the true graph is best at H=32 (`0.571246` versus none `0.682496` and permuted `0.575382`); at 20 epochs no-prior is best (`0.082617` versus graph `0.095294`).
- N-body distance: at 5 epochs the true graph is best at H=32 (`0.543263` versus none `0.821576` and permuted `0.560362`); at 20 epochs the permuted graph is best (`0.089004` versus true graph `0.113463`).

Interpretation: true graph priors can improve low-budget rollout as topology-aligned regularizers, but with more training the advantage may disappear, and graph-aware architecture or spectrum-matched smoothing controls can catch up or outperform.

## Case Notes

### HO Lattice

Graph source: true lattice. Role: controlled positive case. Expected behavior was topology-aligned latent smoothing at high prior weight, and the completed run matches that expectation.

At H=32, no-prior rollout error was `1.084877`, true graph was `1.013924`, and permuted graph was `1.067758`. The final classification was `topology_aligned_latent_smoothing`. The latent audit also supports the classification: graph `D_true_norm(Delta_H)` was `3.644385` versus `4.054084` for permuted, and graph `R_low K=4` was `0.102023` versus `0.067782` for permuted.

### Graph Heat Lattice

Graph source: true generator graph. Role: graph-generated cautionary case. Expected behavior before audit was maybe topology-aligned, but the actual result was `no_graph_gain` under the current latent model.

The failure audit conclusion is that the oracle heat model is accurate, raw `D_dX` did not favor the true graph, and lower lambda partly reduced harm but did not beat none or permuted. H=32 errors were `0.603606` for none, `0.690180` for graph, and `0.605792` for permuted.

Interpretation: graph-generated dynamics does not automatically imply latent graph prior utility.

### Graph Low-Frequency Lattice

Graph source: constructed true low-frequency Laplacian basis. Role: synthetic alignment stress test.

The 5-epoch preflight raw Stage 0 strongly favored the true graph: true `D_dX_norm` was `0.078922`, versus `0.459954` for permuted and `0.376278` for random. Despite that strong raw alignment, the 5-epoch classification was `generic_smoothing`: graph beat none at H=32 (`0.691928` versus `0.720953`) but did not beat permuted (`0.670257`).

The failure analysis shows training-budget dependence. At 20 epochs, graph beat permuted at H=32 (`0.0580` versus `0.0780`). At 50 epochs, permuted beat graph (`0.0021` versus `0.0038`) despite graph retaining more true-smooth latent deltas.

### Spring-Mass Lattice

Graph source: true spring lattice. Role: second-order mechanical benchmark.

The 5-epoch run classified as `topology_aligned_latent_smoothing`, with graph best at H=32. The 20-epoch run classified as `no_graph_gain`: the graph still beat the permuted control, but no-prior had the lowest H=32 rollout error. This is a clean example of early topology-aligned regularization followed by no-prior catch-up.

### Graph-Wave Lattice

Graph source: true wave lattice. Role: graph-discretized wave/PDE-like benchmark.

The 5-epoch run classified as `topology_aligned_latent_smoothing`, again with graph best at H=32. The 20-epoch run classified as `no_graph_gain`: no-prior was best, even though graph remained better than permuted. This mirrors spring-mass and suggests the prior can be most useful when training budget is tight.

### N-Body Distance

Graph source: initial-position weighted distance-kNN graph. Role: long-range particle interaction benchmark.

The 5-epoch run classified as `topology_aligned_latent_smoothing`. At 20 epochs, the true graph still beat no-prior, but the permuted graph was best, so the classification became `generic_smoothing`. This is consistent with the caveat that long-range interactions are not necessarily captured by a fixed local Laplacian; smoothing structure can help without proving topology-specific use.

### METR-LA Correlation Graph

Graph source: `correlation_topk` sensor graph, not official road adjacency. Role: real traffic application case.

The main T=2000/train160 run is preferred over the smoke check. Stage 0 showed the correlation graph was smoother than permuted and random controls in raw temporal changes: true `D_dX_norm` was `0.237967`, permuted was `0.304427`, and random was `0.473690`.

Stage 1 did not translate that raw alignment into rollout gain. H=32 errors were `0.406830` for none, `0.446130` for graph, and `0.433669` for permuted. Graph gain versus none was `-9.66%`, true-vs-permuted gain was `-2.87%`, and final classification was `no_graph_gain`.

The 1-epoch smoke run is included in the CSV only as a short functionality check; it is not the main METR-LA result.

## What The Preflight Check Can And Cannot Claim

The preflight check can quickly identify whether the true graph prior improves rollout error over a no-prior GNN and whether that improvement survives a spectrum-matched permuted graph control. It can also provide mechanistic evidence from raw graph-dynamics alignment and, when latent traces are available, learned latent temporal smoothness.

The preflight check cannot by itself prove that a model uses the intended scientific topology. A true graph beating no-prior but failing against the permuted control is best interpreted as generic smoothing. A true graph beating both controls is stronger evidence, but the latent audit should still be inspected before making a topology-use claim.

## Recommended Usage Modes

- Quick mode: 5 epochs, detects graph gain and cheap positive candidates.
- Standard mode: around 20 epochs, tests whether topology-specificity survives more training.
- Extended mode: longer training, tests persistence/control catch-up when a topology-specific signal appears at shorter budgets.

## Current Limitations

- Raw-coordinate alignment is not sufficient.
- Latent audit requires saved latent traces/checkpoints.
- Classification depends on model, training budget, and prior weight.
- Some candidate graphs are approximate or data-derived, not ground-truth physical edges.
- METR-LA run used a correlation-derived graph, not official road adjacency.

## Decision Rule

Before claiming that a graph prior uses true scientific topology, compare it against a spectrum-matched permuted graph. If true graph does not beat the permuted control, interpret the gain as generic smoothing or no graph gain. If true graph beats the permuted control, inspect learned latent temporal smoothness before making a topology-use claim.
