# Unified Effect-Size Table

Effect sizes are graph-arm-standard-error contrasts, `(comparator_mean - graph_mean) / (std_graph / sqrt(n_graph))`. Positive values mean the graph prior has lower H32 rollout error. CI overlap uses reported 95% intervals. HO audit graph-vs-control effects use paired bootstrap deltas from the specificity audit because the audit cell has no no-prior arm. Footnote: `topology_specific` = controlled audit-positive class; HO audit is the only such case in our evidence.

| regime | cell | n_seeds | mean_h32_graph | effect_size_vs_none | ci_overlap_vs_none | effect_size_vs_permuted | ci_overlap_vs_permuted | effect_size_vs_random | ci_overlap_vs_random | effect_size_vs_temporal | ci_overlap_vs_temporal | final_classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ho_audit_5seed | ho_lattice_audit | 5 | 0.0169 | N/A | N/A | 3.083 | non-overlap | 2.745 | non-overlap | N/A | N/A | topology_specific |
| nbody_robustness_5seed | distance_k_04/standard_ep20 | 5 | 0.0641 | 6.014 | non-overlap | 0.535 | overlap | N/A | N/A | 0.139 | overlap | generic_regularization |
| nbody_robustness_5seed | distance_k_08/standard_ep20 | 5 | 0.0814 | 4.768 | non-overlap | -0.370 | overlap | N/A | N/A | -0.801 | overlap | generic_regularization |
| nbody_robustness_5seed | distance_k_12/standard_ep20 | 5 | 0.0772 | 4.824 | non-overlap | 0.906 | overlap | N/A | N/A | -1.265 | overlap | generic_regularization |
| rmd17_cycle2_3seed | aspirin | 3 | 0.0306 | 8.367 | non-overlap | 0.147 | overlap | -0.013 | overlap | N/A | N/A | generic_regularization |
| rmd17_cycle2_3seed | ethanol | 3 | 0.0449 | 5.196 | overlap | -2.935 | overlap | -2.358 | overlap | N/A | N/A | generic_regularization |
| rmd17_cycle2_3seed | malonaldehyde | 3 | 0.0346 | 3.979 | overlap | -0.445 | overlap | -0.480 | overlap | N/A | N/A | generic_regularization |
| rmd17_cycle2_3seed | naphthalene | 3 | 0.0210 | 6.785 | non-overlap | 0.191 | overlap | 0.381 | overlap | N/A | N/A | generic_regularization |
| rmd17_cycle2_3seed | toluene | 3 | 0.0323 | 2.359 | overlap | -0.289 | overlap | -0.247 | overlap | N/A | N/A | generic_regularization |
| graph_heat_lattice | legacy_ep5_no_quick_tag | 1 | 0.6902 | N/A | non-overlap | N/A | non-overlap | N/A | N/A | N/A | N/A | no_effect |
| graph_wave_5seed | standard_ep20 | 5 | 0.0782 | 0.352 | overlap | -0.469 | overlap | N/A | N/A | -0.559 | overlap | no_effect |
| metr_la_3seed | corr_T2000_train160_ep5 | 3 | 0.5243 | -0.736 | overlap | -0.132 | overlap | N/A | N/A | -0.089 | overlap | no_effect |
| spring_mass_5seed | standard_ep20 | 5 | 0.0800 | 0.088 | overlap | 0.189 | overlap | N/A | N/A | -0.024 | overlap | no_effect |
