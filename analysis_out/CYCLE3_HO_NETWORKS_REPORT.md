# Cycle 3 Controlled HO Networks Report

Results file: `experiments/results/cycle3_ho_networks/cycle3_ho_networks_results.json`
Schema version: `cycle3_ho_networks_v1`
Success/failure count: 75 ok / 0 failed
Missing expected runs: 0
Runs with non-finite rollout diagnostics: none
Graph prior metadata node-wise: YES

## Rollout Error Mean +/- Std

| topology | prior | H=1 | H=2 | H=4 | H=8 | H=16 | H=32 |
|---|---|---:|---:|---:|---:|---:|---:|
| lattice | none | 0.0008 +/- 0.0004 (n=5) | 0.0015 +/- 0.0007 (n=5) | 0.0030 +/- 0.0014 (n=5) | 0.0058 +/- 0.0030 (n=5) | 0.0123 +/- 0.0069 (n=5) | 0.0385 +/- 0.0215 (n=5) |
| lattice | covariance | 0.0008 +/- 0.0004 (n=5) | 0.0015 +/- 0.0007 (n=5) | 0.0030 +/- 0.0014 (n=5) | 0.0059 +/- 0.0030 (n=5) | 0.0125 +/- 0.0069 (n=5) | 0.0391 +/- 0.0212 (n=5) |
| lattice | graph | 0.0003 +/- 0.0001 (n=5) | 0.0006 +/- 0.0002 (n=5) | 0.0012 +/- 0.0004 (n=5) | 0.0025 +/- 0.0009 (n=5) | 0.0056 +/- 0.0018 (n=5) | 0.0169 +/- 0.0054 (n=5) |
| lattice | permuted_graph | 0.0004 +/- 0.0001 (n=5) | 0.0007 +/- 0.0002 (n=5) | 0.0014 +/- 0.0004 (n=5) | 0.0029 +/- 0.0008 (n=5) | 0.0066 +/- 0.0016 (n=5) | 0.0198 +/- 0.0059 (n=5) |
| lattice | random_graph | 0.0004 +/- 0.0001 (n=5) | 0.0007 +/- 0.0002 (n=5) | 0.0014 +/- 0.0004 (n=5) | 0.0029 +/- 0.0008 (n=5) | 0.0067 +/- 0.0016 (n=5) | 0.0199 +/- 0.0062 (n=5) |
| random | none | 0.0007 +/- 0.0002 (n=5) | 0.0015 +/- 0.0004 (n=5) | 0.0029 +/- 0.0007 (n=5) | 0.0056 +/- 0.0014 (n=5) | 0.0114 +/- 0.0029 (n=5) | 0.0361 +/- 0.0127 (n=5) |
| random | covariance | 0.0008 +/- 0.0002 (n=5) | 0.0015 +/- 0.0004 (n=5) | 0.0029 +/- 0.0007 (n=5) | 0.0057 +/- 0.0015 (n=5) | 0.0116 +/- 0.0035 (n=5) | 0.0359 +/- 0.0179 (n=5) |
| random | graph | 0.0003 +/- 0.0001 (n=5) | 0.0006 +/- 0.0002 (n=5) | 0.0013 +/- 0.0005 (n=5) | 0.0026 +/- 0.0010 (n=5) | 0.0057 +/- 0.0024 (n=5) | 0.0172 +/- 0.0090 (n=5) |
| random | permuted_graph | 0.0004 +/- 0.0001 (n=5) | 0.0007 +/- 0.0001 (n=5) | 0.0014 +/- 0.0002 (n=5) | 0.0030 +/- 0.0005 (n=5) | 0.0066 +/- 0.0011 (n=5) | 0.0187 +/- 0.0066 (n=5) |
| random | random_graph | 0.0004 +/- 0.0001 (n=5) | 0.0007 +/- 0.0001 (n=5) | 0.0014 +/- 0.0003 (n=5) | 0.0029 +/- 0.0006 (n=5) | 0.0065 +/- 0.0015 (n=5) | 0.0186 +/- 0.0069 (n=5) |
| scalefree | none | 0.0009 +/- 0.0003 (n=5) | 0.0018 +/- 0.0005 (n=5) | 0.0035 +/- 0.0009 (n=5) | 0.0067 +/- 0.0018 (n=5) | 0.0131 +/- 0.0040 (n=5) | 0.0373 +/- 0.0187 (n=5) |
| scalefree | covariance | 0.0009 +/- 0.0003 (n=5) | 0.0019 +/- 0.0005 (n=5) | 0.0036 +/- 0.0010 (n=5) | 0.0069 +/- 0.0019 (n=5) | 0.0133 +/- 0.0040 (n=5) | 0.0370 +/- 0.0182 (n=5) |
| scalefree | graph | 0.0003 +/- 0.0001 (n=5) | 0.0006 +/- 0.0002 (n=5) | 0.0012 +/- 0.0005 (n=5) | 0.0024 +/- 0.0010 (n=5) | 0.0052 +/- 0.0022 (n=5) | 0.0167 +/- 0.0089 (n=5) |
| scalefree | permuted_graph | 0.0003 +/- 0.0001 (n=5) | 0.0007 +/- 0.0001 (n=5) | 0.0013 +/- 0.0003 (n=5) | 0.0027 +/- 0.0006 (n=5) | 0.0057 +/- 0.0014 (n=5) | 0.0157 +/- 0.0051 (n=5) |
| scalefree | random_graph | 0.0003 +/- 0.0001 (n=5) | 0.0007 +/- 0.0001 (n=5) | 0.0013 +/- 0.0003 (n=5) | 0.0027 +/- 0.0006 (n=5) | 0.0057 +/- 0.0014 (n=5) | 0.0158 +/- 0.0048 (n=5) |

## Graph Prior Gain Beyond GNN None

| topology | H=1 | H=2 | H=4 | H=8 | H=16 | H=32 |
|---|---:|---:|---:|---:|---:|---:|
| lattice | 0.0008 -> 0.0003 (+58.9% gain) | 0.0015 -> 0.0006 (+58.6% gain) | 0.0030 -> 0.0012 (+57.8% gain) | 0.0058 -> 0.0025 (+56.2% gain) | 0.0123 -> 0.0056 (+54.2% gain) | 0.0385 -> 0.0169 (+56.0% gain) |
| random | 0.0007 -> 0.0003 (+57.4% gain) | 0.0015 -> 0.0006 (+57.0% gain) | 0.0029 -> 0.0013 (+55.9% gain) | 0.0056 -> 0.0026 (+53.5% gain) | 0.0114 -> 0.0057 (+49.8% gain) | 0.0361 -> 0.0172 (+52.2% gain) |
| scalefree | 0.0009 -> 0.0003 (+67.0% gain) | 0.0018 -> 0.0006 (+66.9% gain) | 0.0035 -> 0.0012 (+66.3% gain) | 0.0067 -> 0.0024 (+64.7% gain) | 0.0131 -> 0.0052 (+60.6% gain) | 0.0373 -> 0.0167 (+55.2% gain) |

## Graph Specificity: Graph vs Permuted Graph

| topology | H=1 | H=2 | H=4 | H=8 | H=16 | H=32 |
|---|---:|---:|---:|---:|---:|---:|
| lattice | 0.0003 vs 0.0004 (+11.5% gain) | 0.0006 vs 0.0007 (+11.7% gain) | 0.0012 vs 0.0014 (+12.2% gain) | 0.0025 vs 0.0029 (+13.3% gain) | 0.0056 vs 0.0066 (+15.1% gain) | 0.0169 vs 0.0198 (+14.5% gain) |
| random | 0.0003 vs 0.0004 (+12.0% gain) | 0.0006 vs 0.0007 (+12.1% gain) | 0.0013 vs 0.0014 (+12.2% gain) | 0.0026 vs 0.0030 (+12.5% gain) | 0.0057 vs 0.0066 (+12.9% gain) | 0.0172 vs 0.0187 (+7.8% gain) |
| scalefree | 0.0003 vs 0.0003 (+10.5% gain) | 0.0006 vs 0.0007 (+10.7% gain) | 0.0012 vs 0.0013 (+11.2% gain) | 0.0024 vs 0.0027 (+11.6% gain) | 0.0052 vs 0.0057 (+9.5% gain) | 0.0167 vs 0.0157 (-6.4% loss) |

## Graph Specificity: Graph vs Random Graph

| topology | H=1 | H=2 | H=4 | H=8 | H=16 | H=32 |
|---|---:|---:|---:|---:|---:|---:|
| lattice | 0.0003 vs 0.0004 (+11.8% gain) | 0.0006 vs 0.0007 (+12.0% gain) | 0.0012 vs 0.0014 (+12.4% gain) | 0.0025 vs 0.0029 (+13.5% gain) | 0.0056 vs 0.0067 (+15.4% gain) | 0.0169 vs 0.0199 (+14.9% gain) |
| random | 0.0003 vs 0.0004 (+12.0% gain) | 0.0006 vs 0.0007 (+11.9% gain) | 0.0013 vs 0.0014 (+11.8% gain) | 0.0026 vs 0.0029 (+11.6% gain) | 0.0057 vs 0.0065 (+11.5% gain) | 0.0172 vs 0.0186 (+7.2% gain) |
| scalefree | 0.0003 vs 0.0003 (+9.1% gain) | 0.0006 vs 0.0007 (+9.4% gain) | 0.0012 vs 0.0013 (+10.1% gain) | 0.0024 vs 0.0027 (+11.0% gain) | 0.0052 vs 0.0057 (+9.5% gain) | 0.0167 vs 0.0158 (-5.8% loss) |

## Paired Graph-Specificity Analysis

Paired deltas are computed seed-wise as control rollout error minus graph rollout error. Positive values mean the true graph has lower error than the control.

| topology | control | horizon | mean delta +/- std | graph wins | bootstrap 95% CI |
|---|---|---:|---:|---:|---:|
| lattice | permuted_graph | H=1 | 0.0000 +/- 0.0000 | 4/5 | [0.0000, 0.0001] |
| lattice | permuted_graph | H=2 | 0.0001 +/- 0.0001 | 4/5 | [0.0000, 0.0001] |
| lattice | permuted_graph | H=4 | 0.0002 +/- 0.0001 | 4/5 | [0.0001, 0.0003] |
| lattice | permuted_graph | H=8 | 0.0004 +/- 0.0002 | 5/5 | [0.0002, 0.0006] |
| lattice | permuted_graph | H=16 | 0.0010 +/- 0.0005 | 5/5 | [0.0006, 0.0013] |
| lattice | permuted_graph | H=32 | 0.0029 +/- 0.0021 | 4/5 | [0.0011, 0.0042] |
| lattice | random_graph | H=1 | 0.0000 +/- 0.0000 | 4/5 | [0.0000, 0.0001] |
| lattice | random_graph | H=2 | 0.0001 +/- 0.0001 | 4/5 | [0.0000, 0.0001] |
| lattice | random_graph | H=4 | 0.0002 +/- 0.0001 | 4/5 | [0.0001, 0.0003] |
| lattice | random_graph | H=8 | 0.0004 +/- 0.0003 | 4/5 | [0.0002, 0.0006] |
| lattice | random_graph | H=16 | 0.0010 +/- 0.0005 | 5/5 | [0.0005, 0.0013] |
| lattice | random_graph | H=32 | 0.0030 +/- 0.0024 | 4/5 | [0.0010, 0.0046] |
| random | permuted_graph | H=1 | 0.0000 +/- 0.0001 | 4/5 | [-0.0000, 0.0001] |
| random | permuted_graph | H=2 | 0.0001 +/- 0.0001 | 4/5 | [-0.0000, 0.0002] |
| random | permuted_graph | H=4 | 0.0002 +/- 0.0003 | 4/5 | [-0.0000, 0.0004] |
| random | permuted_graph | H=8 | 0.0004 +/- 0.0006 | 3/5 | [-0.0000, 0.0008] |
| random | permuted_graph | H=16 | 0.0009 +/- 0.0014 | 3/5 | [-0.0002, 0.0019] |
| random | permuted_graph | H=32 | 0.0015 +/- 0.0064 | 4/5 | [-0.0040, 0.0056] |
| random | random_graph | H=1 | 0.0000 +/- 0.0001 | 4/5 | [0.0000, 0.0001] |
| random | random_graph | H=2 | 0.0001 +/- 0.0001 | 4/5 | [0.0000, 0.0002] |
| random | random_graph | H=4 | 0.0002 +/- 0.0002 | 4/5 | [0.0000, 0.0004] |
| random | random_graph | H=8 | 0.0003 +/- 0.0004 | 4/5 | [0.0001, 0.0007] |
| random | random_graph | H=16 | 0.0007 +/- 0.0010 | 4/5 | [-0.0001, 0.0015] |
| random | random_graph | H=32 | 0.0013 +/- 0.0053 | 4/5 | [-0.0034, 0.0047] |
| scalefree | permuted_graph | H=1 | 0.0000 +/- 0.0001 | 4/5 | [-0.0000, 0.0001] |
| scalefree | permuted_graph | H=2 | 0.0001 +/- 0.0002 | 3/5 | [-0.0001, 0.0002] |
| scalefree | permuted_graph | H=4 | 0.0001 +/- 0.0004 | 3/5 | [-0.0002, 0.0004] |
| scalefree | permuted_graph | H=8 | 0.0003 +/- 0.0009 | 3/5 | [-0.0005, 0.0010] |
| scalefree | permuted_graph | H=16 | 0.0005 +/- 0.0022 | 3/5 | [-0.0012, 0.0023] |
| scalefree | permuted_graph | H=32 | -0.0010 +/- 0.0084 | 2/5 | [-0.0075, 0.0055] |
| scalefree | random_graph | H=1 | 0.0000 +/- 0.0001 | 3/5 | [-0.0001, 0.0001] |
| scalefree | random_graph | H=2 | 0.0001 +/- 0.0002 | 3/5 | [-0.0001, 0.0002] |
| scalefree | random_graph | H=4 | 0.0001 +/- 0.0004 | 3/5 | [-0.0002, 0.0004] |
| scalefree | random_graph | H=8 | 0.0003 +/- 0.0009 | 3/5 | [-0.0005, 0.0010] |
| scalefree | random_graph | H=16 | 0.0005 +/- 0.0023 | 3/5 | [-0.0012, 0.0023] |
| scalefree | random_graph | H=32 | -0.0009 +/- 0.0086 | 2/5 | [-0.0075, 0.0059] |

## Euclidean Conditioning: Covariance vs Graph Controls

| topology | covariance vs graph H=32 | covariance vs permuted H=32 | covariance vs random H=32 |
|---|---:|---:|---:|
| lattice | 0.0391 -> 0.0169 (+56.7% gain) | 0.0391 -> 0.0198 (+49.4% gain) | 0.0391 -> 0.0199 (+49.1% gain) |
| random | 0.0359 -> 0.0172 (+52.0% gain) | 0.0359 -> 0.0187 (+47.9% gain) | 0.0359 -> 0.0186 (+48.2% gain) |
| scalefree | 0.0370 -> 0.0167 (+54.8% gain) | 0.0370 -> 0.0157 (+57.6% gain) | 0.0370 -> 0.0158 (+57.3% gain) |

## Topology Classification

| topology | class | covariance gain H=32 | graph gain H=32 | graph vs permuted H=32 | graph vs random H=32 | rationale |
|---|---|---:|---:|---:|---:|---|
| lattice | true-graph-specific | -1.6% loss | +56.0% gain | +14.5% gain | +14.9% gain | true graph beats both graph controls by >=10% |
| random | random-control-equivalent | +0.5% gain | +52.2% gain | +7.8% gain | +7.2% gain | true, permuted, and random graph controls are within +/-10% |
| scalefree | random-control-equivalent | +0.8% gain | +55.2% gain | -6.4% loss | -5.8% loss | true, permuted, and random graph controls are within +/-10% |

## Recommendation

rMD17 top-up recommendation: do not top up rMD17 seeds yet; first resolve the generic-control mechanism.
ISO17 recommendation: do not run ISO17 yet; it would likely amplify the same ambiguity at higher cost.
Paper thesis: emphasize generic smoothing/conditioning unless a later controlled result recovers true-topology specificity.
