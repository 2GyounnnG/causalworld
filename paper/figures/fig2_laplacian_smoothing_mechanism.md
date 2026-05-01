# Figure 2 Draft. Laplacian Smoothing Mechanism and Permuted Control

## Panel A: Graph-Frequency Smoothing Identity

```math
R_G(H; L) = Tr(H^T L H)
```

With eigendecomposition

```math
L = U Lambda U^T
```

and graph Fourier coefficients

```math
H = U A,
```

the penalty becomes

```math
Tr(H^T L H) = Tr(A^T Lambda A) = sum_k lambda_k ||a_k||_2^2.
```

Interpretation: larger Laplacian eigenvalues correspond to higher graph frequencies, so the prior penalizes high-frequency latent variation more strongly.

## Panel B: Spectrum-Matched Permuted Control

```math
L_perm = P^T L P
```

where `P` is a permutation matrix.

Key properties:

- `L_perm` has the same eigenvalues as `L`.
- The graph-frequency penalty scale is preserved.
- Node-label semantics are disrupted.
- If graph and permuted graph perform similarly, the gain is better interpreted as generic graph smoothing.

## Panel C: Attribution Rule

```text
graph beats none
  -> prior may help prediction

graph beats permuted_graph
  -> candidate topology may add value beyond spectrum-matched smoothing

graph beats calibrated temporal_smooth
  -> graph structure may add value beyond graph-free temporal regularization

graph beats controls + latent audit positive
  -> topology-aligned latent smoothing under tested model condition
```

Message: graph-prior gains require controls because Laplacian regularization is first a smoothing operator; topology attribution is a stronger claim than predictive improvement.
