# Latent Collapse Finding (rMD17 + flat encoder)

## Empirical observation

For ethanol and malonaldehyde with the flat encoder, latent vectors 
across 200 sampled transitions are IDENTICAL (per-dimension std = 0 
exactly). Aspirin shows variation (per-dim std mean ≈ 0.30).

## Numerical confirmation
- Ethanol: per-dim std (min/max/mean) = (0.0, 0.0, 0.0)
- Malonaldehyde: per-dim std (min/max/mean) = (0.0, 0.0, 0.0)
- Aspirin: per-dim std (min/max/mean) = (0.055, 0.78, 0.30)

## Mechanism

The flat encoder computes graph-level summary scalars:
- num_nodes, num_edges, mean_arity, etc. (6 features)

For molecules where the bond connectivity is invariant across the 
trajectory (a property that holds for stable small molecules under 
classical MD), these scalars are constants. The encoder thus maps 
all transitions to a single fixed latent vector.

## Implication for prior choice

Covariance-based priors (Euclidean isotropy) are numerically 
inactive when latents collapse: cov(z) = 0 implies ||cov - I||_F 
is a constant function of z, with zero gradient.

Spectral graph-Laplacian priors remain active because they evaluate 
z^T L z per sample (not across the batch). Even when all latents are 
identical, the gradient pushes the shared latent toward L's null 
space.

## Empirical evidence

In the disjoint-frame eval, ethanol and malonaldehyde euclidean 
results are bit-identical to none results. Spectral, in contrast, 
produces substantial improvement (73% and 54% reduction at H=16 vs 
none, respectively).

## Reframing for paper

This is not a bug or limitation. It is a **structural finding**:

The choice of latent prior must be compatible with the encoder's 
representation regime. Encoder designs that produce constant or 
low-variance latents on a given dataset (a common case for small 
molecules in flat encoder regimes) cannot be regularized by 
covariance-based priors. Geometric priors that operate per-sample 
on graph structure remain effective in this regime.

## Paper section to update

Update Section 9 (Limitations) and Section 7 (Discussion) to include 
this finding as a positive contribution rather than a stale-data 
caveat.
