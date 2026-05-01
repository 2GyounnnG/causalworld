# Appendix A. Preflight Protocol Details

This appendix will provide the complete label definitions and decision thresholds used by the preflight protocol.

# Appendix B. Benchmark Regimes and Dataset Adapters

This appendix will list all dataset adapters, graph sources, trajectory shapes, and whether the graph is known, approximate, generated, or data-derived.

# Appendix C. Prior Families and Calibration

The graph Laplacian prior on learned node-wise latent states is

```math
R_G(H; L) = Tr(H^T L H).
```

The spectrum-matched permuted graph control is

```math
L_perm = P^T L P,
```

where `P` is a permutation matrix. This preserves the Laplacian eigenvalues while disrupting node-label semantics.

The graph-free temporal smoothness prior is

```math
R_T(H_{1:T}) = sum_t ||H_{t+1} - H_t||_F^2.
```

For calibrated prior-strength matching, let `alpha_G` be the graph prior weight, and let `R_G^0` and `R_T^0` denote initial graph and temporal prior losses under the same calibration pass. The calibrated temporal weight is

```math
alpha_T = alpha_G R_G^0 / (R_T^0 + epsilon),
```

so that

```math
alpha_T R_T^0 approx alpha_G R_G^0.
```

# Appendix D. Evidence Map

This appendix will map each manuscript claim to supporting figures, tables, reports, and caveats.

# Appendix E. Reproducibility and Software Interface

Example preflight command using the existing spring-mass lattice adapter:

```bash
conda run -n causalworld python scripts/graph_prior_preflight_check.py \
  --dataset spring_mass_lattice \
  --prior-weight 0.1 \
  --epochs 5 \
  --train-transitions 96 \
  --eval-transitions 32 \
  --horizons 16 32 \
  --include-temporal-prior \
  --calibrate-prior-strength \
  --out-dir analysis_out/preflight_runs/spring_mass_lattice_ep5_temporal_calibrated
```
