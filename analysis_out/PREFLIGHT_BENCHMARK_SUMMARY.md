# Graph Prior Preflight Benchmark Summary

Executive summary: Existing completed preflight runs show that the tool is useful as a control-oriented screen, not as a standalone proof that a graph prior uses scientific topology. The HO lattice run is the clearest positive case: true graph beats both no-prior and spectrum-matched permuted controls, and latent deltas are smoother in the true graph basis. The graph heat, graph low-frequency, and METR-LA cases show the important caveats: graph-generated or raw-aligned dynamics can still fail to produce rollout gain, and topology-specific behavior can depend on training budget.

No new training, ISO17, rMD17 top-up, METR-LA standard/20-epoch run, or model-logic changes were performed. This report only reads existing outputs and writes:

- `analysis_out/PREFLIGHT_BENCHMARK_SUMMARY.md`
- `analysis_out/preflight_benchmark_summary.csv`

## Benchmark Table

| case | role | graph source | epochs | H32 none | H32 graph | H32 permuted | graph gain vs none | true vs permuted | classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HO lattice | controlled positive case | true lattice | 5 | 1.084877 | 1.013924 | 1.067758 | +6.54% | +5.04% | topology_aligned_latent_smoothing |
| Graph heat lattice | graph-generated cautionary case | true generator graph | 5 | 0.603606 | 0.690180 | 0.605792 | -14.34% | -13.93% | no_graph_gain |
| Graph low-frequency lattice | synthetic alignment stress test | constructed true low-frequency Laplacian basis | 5; analysis 20/50 | 0.720953 | 0.691928 | 0.670257 | +4.03% | -3.23% | generic_smoothing |
| METR-LA correlation graph T=2000 | real traffic application case | correlation_topk sensor graph | 5 | 0.406830 | 0.446130 | 0.433669 | -9.66% | -2.87% | no_graph_gain |

## Case Notes

### HO lattice

Graph source: true lattice. Role: controlled positive case. Expected behavior was topology-aligned latent smoothing at high prior weight, and the completed run matches that expectation.

At H=32, no-prior rollout error was `1.084877`, true graph was `1.013924`, and permuted graph was `1.067758`. The final classification was `topology_aligned_latent_smoothing`. The latent audit also supports the classification: graph `D_true_norm(Delta_H)` was `3.644385` versus `4.054084` for permuted, and graph `R_low K=4` was `0.102023` versus `0.067782` for permuted.

Files used: `analysis_out/preflight_runs/ho_lattice/preflight_report.md`, `analysis_out/preflight_runs/ho_lattice/summary.csv`.

### Graph heat lattice

Graph source: true generator graph. Role: graph-generated cautionary case. Expected behavior before audit was maybe topology-aligned, but the actual result was `no_graph_gain` under the current latent model.

The failure audit conclusion is that the oracle heat model is accurate, raw `D_dX` did not favor the true graph, and lower lambda partly reduced harm but did not beat none or permuted. H=32 errors were `0.603606` for none, `0.690180` for graph, and `0.605792` for permuted. Best lower-lambda graph H=32 in the audit was `0.6288`, still worse than none and permuted.

Interpretation: graph-generated dynamics does not automatically imply latent graph prior utility.

Files used: `analysis_out/preflight_runs/graph_heat_lattice/preflight_report.md`, `analysis_out/preflight_runs/graph_heat_lattice/summary.csv`, `analysis_out/preflight_runs/graph_heat_lattice/FAILURE_AUDIT.md`.

### Graph low-frequency lattice

Graph source: constructed true low-frequency Laplacian basis. Role: synthetic alignment stress test.

The 5-epoch preflight raw Stage 0 strongly favored the true graph: true `D_dX_norm` was `0.078922`, versus `0.459954` for permuted and `0.376278` for random. Despite that strong raw alignment, the 5-epoch classification was `generic_smoothing`: graph beat none at H=32 (`0.691928` versus `0.720953`) but did not beat permuted (`0.670257`).

The failure analysis shows training-budget dependence. At 20 epochs, graph beat permuted at H=32 (`0.0580` versus `0.0780`). At 50 epochs, permuted beat graph (`0.0021` versus `0.0038`) despite graph retaining more true-smooth latent deltas: graph `D_true_norm(Delta_H)` was lower at both 20 epochs (`0.5387` versus `0.7497`) and 50 epochs (`0.6458` versus `0.8690`).

Interpretation: raw alignment is insufficient; topology-specific signal is training-budget dependent.

Files used: `analysis_out/preflight_runs/graph_lowfreq_lattice/preflight_report.md`, `analysis_out/preflight_runs/graph_lowfreq_lattice/summary.csv`, `analysis_out/preflight_runs/graph_lowfreq_lattice/FAILURE_ANALYSIS.md`, `analysis_out/preflight_runs/graph_lowfreq_lattice/failure_analysis_summary.csv`.

### METR-LA correlation graph T=2000

Graph source: `correlation_topk` sensor graph, not official road adjacency. Role: real traffic application case.

Stage 0 showed the correlation graph was smoother than permuted and random controls in raw temporal changes: true `D_dX_norm` was `0.237967`, permuted was `0.304427`, and random was `0.473690`.

Stage 1 did not translate that raw alignment into rollout gain. H=32 errors were `0.406830` for none, `0.446130` for graph, and `0.433669` for permuted. Graph gain versus none was `-9.66%`, true-vs-permuted gain was `-2.87%`, and final classification was `no_graph_gain`.

Interpretation: raw graph-dynamics alignment does not guarantee graph-prior rollout gain in a real traffic dataset.

Files used: `analysis_out/preflight_runs/metr_la_corr_T2000_train160/preflight_report.md`, `analysis_out/preflight_runs/metr_la_corr_T2000_train160/summary.csv`.

## What The Preflight Check Can And Cannot Claim

The preflight check can quickly identify whether the true graph prior improves rollout error over a no-prior GNN and whether that improvement survives a spectrum-matched permuted graph control. It can also provide mechanistic evidence from raw graph-dynamics alignment and, when latent traces are available, learned latent temporal smoothness.

The preflight check cannot by itself prove that a model uses the intended scientific topology. A true graph beating no-prior but failing against the permuted control is best interpreted as generic smoothing. A true graph beating both controls is stronger evidence, but the latent audit should still be inspected before making a topology-use claim.

## Recommended Usage Modes

- Quick mode: 5 epochs, detects graph gain only. Use it to find obvious no-graph-gain cases and cheap positive candidates.
- Standard mode: around 20 epochs, tests topology-specificity. Use paired true-vs-permuted comparison as the main decision point.
- Extended mode: longer training, tests persistence/control catch-up. Use it when a topology-specific signal appears at standard budget and you need to know whether it persists.

## Current Limitations

- Raw-coordinate alignment is not sufficient.
- Latent audit requires saved latent traces/checkpoints.
- Classification depends on model, training budget, and prior weight.
- METR-LA run used a correlation-derived graph, not official road adjacency.

## Decision Rule

Before claiming that a graph prior uses true scientific topology, compare it against a spectrum-matched permuted graph. If true graph does not beat the permuted control, interpret the gain as generic smoothing or no graph gain. If true graph beats the permuted control, inspect learned latent temporal smoothness before making a topology-use claim.
