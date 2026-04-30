# Cycle 2 rMD17 Multi-Molecule Attribution Report

Results file: `experiments/results/cycle2_rmd17_multimolecule/cycle2_rmd17_multimolecule_results.json`
Schema version: `cycle2_rmd17_multimolecule_v1`
Success/failure count: 75 ok / 0 failed
Missing expected runs: 0
Runs with non-finite rollout diagnostics: none
Graph prior metadata node-wise: YES

## Rollout Error Mean +/- Std

| molecule | encoder | prior | H=1 | H=2 | H=4 | H=8 | H=16 | H=32 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| aspirin | mlp_global | none | 0.0260 +/- 0.0069 (n=3) | 0.0274 +/- 0.0053 (n=3) | 0.0304 +/- 0.0058 (n=3) | 0.0446 +/- 0.0101 (n=3) | 0.0806 +/- 0.0232 (n=3) | 0.1953 +/- 0.0645 (n=3) |
| aspirin | gnn_node | none | 0.0095 +/- 0.0014 (n=3) | 0.0093 +/- 0.0011 (n=3) | 0.0105 +/- 0.0017 (n=3) | 0.0159 +/- 0.0041 (n=3) | 0.0299 +/- 0.0078 (n=3) | 0.0934 +/- 0.0230 (n=3) |
| aspirin | gnn_node | graph | 0.0034 +/- 0.0007 (n=3) | 0.0033 +/- 0.0005 (n=3) | 0.0039 +/- 0.0009 (n=3) | 0.0053 +/- 0.0008 (n=3) | 0.0100 +/- 0.0019 (n=3) | 0.0306 +/- 0.0130 (n=3) |
| aspirin | gnn_node | permuted_graph | 0.0035 +/- 0.0005 (n=3) | 0.0034 +/- 0.0004 (n=3) | 0.0040 +/- 0.0009 (n=3) | 0.0057 +/- 0.0009 (n=3) | 0.0108 +/- 0.0004 (n=3) | 0.0317 +/- 0.0078 (n=3) |
| aspirin | gnn_node | random_graph | 0.0035 +/- 0.0005 (n=3) | 0.0034 +/- 0.0004 (n=3) | 0.0041 +/- 0.0009 (n=3) | 0.0058 +/- 0.0009 (n=3) | 0.0110 +/- 0.0009 (n=3) | 0.0305 +/- 0.0083 (n=3) |
| ethanol | mlp_global | none | 0.0189 +/- 0.0027 (n=3) | 0.0203 +/- 0.0043 (n=3) | 0.0214 +/- 0.0038 (n=3) | 0.0270 +/- 0.0066 (n=3) | 0.0464 +/- 0.0145 (n=3) | 0.1133 +/- 0.0447 (n=3) |
| ethanol | gnn_node | none | 0.0097 +/- 0.0031 (n=3) | 0.0096 +/- 0.0021 (n=3) | 0.0117 +/- 0.0029 (n=3) | 0.0135 +/- 0.0024 (n=3) | 0.0230 +/- 0.0038 (n=3) | 0.0557 +/- 0.0163 (n=3) |
| ethanol | gnn_node | graph | 0.0046 +/- 0.0014 (n=3) | 0.0050 +/- 0.0013 (n=3) | 0.0061 +/- 0.0015 (n=3) | 0.0079 +/- 0.0016 (n=3) | 0.0150 +/- 0.0006 (n=3) | 0.0449 +/- 0.0036 (n=3) |
| ethanol | gnn_node | permuted_graph | 0.0044 +/- 0.0015 (n=3) | 0.0049 +/- 0.0014 (n=3) | 0.0058 +/- 0.0018 (n=3) | 0.0073 +/- 0.0021 (n=3) | 0.0134 +/- 0.0022 (n=3) | 0.0388 +/- 0.0081 (n=3) |
| ethanol | gnn_node | random_graph | 0.0046 +/- 0.0013 (n=3) | 0.0050 +/- 0.0012 (n=3) | 0.0060 +/- 0.0014 (n=3) | 0.0076 +/- 0.0014 (n=3) | 0.0141 +/- 0.0005 (n=3) | 0.0400 +/- 0.0040 (n=3) |
| malonaldehyde | mlp_global | none | 0.0238 +/- 0.0030 (n=3) | 0.0229 +/- 0.0033 (n=3) | 0.0252 +/- 0.0030 (n=3) | 0.0317 +/- 0.0047 (n=3) | 0.0520 +/- 0.0123 (n=3) | 0.1329 +/- 0.0386 (n=3) |
| malonaldehyde | gnn_node | none | 0.0129 +/- 0.0029 (n=3) | 0.0125 +/- 0.0026 (n=3) | 0.0154 +/- 0.0044 (n=3) | 0.0171 +/- 0.0056 (n=3) | 0.0276 +/- 0.0075 (n=3) | 0.0686 +/- 0.0189 (n=3) |
| malonaldehyde | gnn_node | graph | 0.0055 +/- 0.0014 (n=3) | 0.0056 +/- 0.0008 (n=3) | 0.0064 +/- 0.0011 (n=3) | 0.0081 +/- 0.0007 (n=3) | 0.0130 +/- 0.0027 (n=3) | 0.0346 +/- 0.0148 (n=3) |
| malonaldehyde | gnn_node | permuted_graph | 0.0053 +/- 0.0012 (n=3) | 0.0056 +/- 0.0007 (n=3) | 0.0063 +/- 0.0010 (n=3) | 0.0080 +/- 0.0009 (n=3) | 0.0124 +/- 0.0020 (n=3) | 0.0308 +/- 0.0082 (n=3) |
| malonaldehyde | gnn_node | random_graph | 0.0054 +/- 0.0012 (n=3) | 0.0057 +/- 0.0007 (n=3) | 0.0063 +/- 0.0010 (n=3) | 0.0081 +/- 0.0008 (n=3) | 0.0124 +/- 0.0021 (n=3) | 0.0305 +/- 0.0091 (n=3) |
| naphthalene | mlp_global | none | 0.0090 +/- 0.0005 (n=3) | 0.0096 +/- 0.0006 (n=3) | 0.0101 +/- 0.0008 (n=3) | 0.0129 +/- 0.0017 (n=3) | 0.0236 +/- 0.0052 (n=3) | 0.0630 +/- 0.0145 (n=3) |
| naphthalene | gnn_node | none | 0.0052 +/- 0.0011 (n=3) | 0.0057 +/- 0.0006 (n=3) | 0.0077 +/- 0.0020 (n=3) | 0.0124 +/- 0.0033 (n=3) | 0.0236 +/- 0.0064 (n=3) | 0.0637 +/- 0.0219 (n=3) |
| naphthalene | gnn_node | graph | 0.0019 +/- 0.0005 (n=3) | 0.0021 +/- 0.0003 (n=3) | 0.0025 +/- 0.0003 (n=3) | 0.0036 +/- 0.0006 (n=3) | 0.0074 +/- 0.0026 (n=3) | 0.0210 +/- 0.0109 (n=3) |
| naphthalene | gnn_node | permuted_graph | 0.0019 +/- 0.0003 (n=3) | 0.0020 +/- 0.0002 (n=3) | 0.0024 +/- 0.0002 (n=3) | 0.0039 +/- 0.0006 (n=3) | 0.0080 +/- 0.0027 (n=3) | 0.0222 +/- 0.0125 (n=3) |
| naphthalene | gnn_node | random_graph | 0.0019 +/- 0.0003 (n=3) | 0.0020 +/- 0.0002 (n=3) | 0.0025 +/- 0.0002 (n=3) | 0.0039 +/- 0.0007 (n=3) | 0.0080 +/- 0.0029 (n=3) | 0.0234 +/- 0.0122 (n=3) |
| toluene | mlp_global | none | 0.0148 +/- 0.0020 (n=3) | 0.0128 +/- 0.0017 (n=3) | 0.0150 +/- 0.0029 (n=3) | 0.0196 +/- 0.0035 (n=3) | 0.0368 +/- 0.0049 (n=3) | 0.1039 +/- 0.0233 (n=3) |
| toluene | gnn_node | none | 0.0055 +/- 0.0010 (n=3) | 0.0067 +/- 0.0017 (n=3) | 0.0082 +/- 0.0018 (n=3) | 0.0132 +/- 0.0035 (n=3) | 0.0236 +/- 0.0062 (n=3) | 0.0609 +/- 0.0226 (n=3) |
| toluene | gnn_node | graph | 0.0030 +/- 0.0004 (n=3) | 0.0032 +/- 0.0005 (n=3) | 0.0035 +/- 0.0004 (n=3) | 0.0046 +/- 0.0004 (n=3) | 0.0089 +/- 0.0028 (n=3) | 0.0323 +/- 0.0210 (n=3) |
| toluene | gnn_node | permuted_graph | 0.0032 +/- 0.0004 (n=3) | 0.0034 +/- 0.0005 (n=3) | 0.0037 +/- 0.0004 (n=3) | 0.0049 +/- 0.0004 (n=3) | 0.0093 +/- 0.0024 (n=3) | 0.0288 +/- 0.0125 (n=3) |
| toluene | gnn_node | random_graph | 0.0031 +/- 0.0004 (n=3) | 0.0033 +/- 0.0005 (n=3) | 0.0036 +/- 0.0004 (n=3) | 0.0048 +/- 0.0004 (n=3) | 0.0092 +/- 0.0023 (n=3) | 0.0293 +/- 0.0136 (n=3) |

## Graph Prior Gain Beyond GNN None

| molecule | H=1 | H=2 | H=4 | H=8 | H=16 | H=32 |
|---|---:|---:|---:|---:|---:|---:|
| aspirin | 0.0095 -> 0.0034 (+63.9% gain) | 0.0093 -> 0.0033 (+64.8% gain) | 0.0105 -> 0.0039 (+63.2% gain) | 0.0159 -> 0.0053 (+66.7% gain) | 0.0299 -> 0.0100 (+66.5% gain) | 0.0934 -> 0.0306 (+67.2% gain) |
| ethanol | 0.0097 -> 0.0046 (+53.1% gain) | 0.0096 -> 0.0050 (+48.4% gain) | 0.0117 -> 0.0061 (+47.4% gain) | 0.0135 -> 0.0079 (+41.8% gain) | 0.0230 -> 0.0150 (+34.8% gain) | 0.0557 -> 0.0449 (+19.4% gain) |
| malonaldehyde | 0.0129 -> 0.0055 (+57.8% gain) | 0.0125 -> 0.0056 (+55.2% gain) | 0.0154 -> 0.0064 (+58.2% gain) | 0.0171 -> 0.0081 (+52.3% gain) | 0.0276 -> 0.0130 (+53.0% gain) | 0.0686 -> 0.0346 (+49.6% gain) |
| naphthalene | 0.0052 -> 0.0019 (+63.3% gain) | 0.0057 -> 0.0021 (+63.8% gain) | 0.0077 -> 0.0025 (+68.2% gain) | 0.0124 -> 0.0036 (+71.1% gain) | 0.0236 -> 0.0074 (+68.8% gain) | 0.0637 -> 0.0210 (+67.0% gain) |
| toluene | 0.0055 -> 0.0030 (+45.9% gain) | 0.0067 -> 0.0032 (+52.5% gain) | 0.0082 -> 0.0035 (+57.4% gain) | 0.0132 -> 0.0046 (+64.9% gain) | 0.0236 -> 0.0089 (+62.2% gain) | 0.0609 -> 0.0323 (+47.0% gain) |

## Graph Specificity: Graph vs Permuted Graph

| molecule | H=1 | H=2 | H=4 | H=8 | H=16 | H=32 |
|---|---:|---:|---:|---:|---:|---:|
| aspirin | 0.0034 vs 0.0035 (+3.1% gain) | 0.0033 vs 0.0034 (+3.6% gain) | 0.0039 vs 0.0040 (+3.9% gain) | 0.0053 vs 0.0057 (+6.9% gain) | 0.0100 vs 0.0108 (+7.5% gain) | 0.0306 vs 0.0317 (+3.5% gain) |
| ethanol | 0.0046 vs 0.0044 (-3.1% loss) | 0.0050 vs 0.0049 (-2.4% loss) | 0.0061 vs 0.0058 (-5.4% loss) | 0.0079 vs 0.0073 (-7.3% loss) | 0.0150 vs 0.0134 (-11.6% loss) | 0.0449 vs 0.0388 (-15.8% loss) |
| malonaldehyde | 0.0055 vs 0.0053 (-2.3% loss) | 0.0056 vs 0.0056 (+0.5% gain) | 0.0064 vs 0.0063 (-2.8% loss) | 0.0081 vs 0.0080 (-1.2% loss) | 0.0130 vs 0.0124 (-4.6% loss) | 0.0346 vs 0.0308 (-12.2% loss) |
| naphthalene | 0.0019 vs 0.0019 (-2.4% loss) | 0.0021 vs 0.0020 (-4.3% loss) | 0.0025 vs 0.0024 (-0.7% loss) | 0.0036 vs 0.0039 (+7.1% gain) | 0.0074 vs 0.0080 (+7.4% gain) | 0.0210 vs 0.0222 (+5.5% gain) |
| toluene | 0.0030 vs 0.0032 (+6.5% gain) | 0.0032 vs 0.0034 (+5.8% gain) | 0.0035 vs 0.0037 (+4.7% gain) | 0.0046 vs 0.0049 (+4.9% gain) | 0.0089 vs 0.0093 (+4.1% gain) | 0.0323 vs 0.0288 (-12.2% loss) |

## Graph Specificity: Graph vs Random Graph

| molecule | H=1 | H=2 | H=4 | H=8 | H=16 | H=32 |
|---|---:|---:|---:|---:|---:|---:|
| aspirin | 0.0034 vs 0.0035 (+3.4% gain) | 0.0033 vs 0.0034 (+4.1% gain) | 0.0039 vs 0.0041 (+4.8% gain) | 0.0053 vs 0.0058 (+8.3% gain) | 0.0100 vs 0.0110 (+8.7% gain) | 0.0306 vs 0.0305 (-0.2% loss) |
| ethanol | 0.0046 vs 0.0046 (+0.0% gain) | 0.0050 vs 0.0050 (-0.2% loss) | 0.0061 vs 0.0060 (-2.8% loss) | 0.0079 vs 0.0076 (-3.1% loss) | 0.0150 vs 0.0141 (-6.6% loss) | 0.0449 vs 0.0400 (-12.2% loss) |
| malonaldehyde | 0.0055 vs 0.0054 (-1.7% loss) | 0.0056 vs 0.0057 (+0.8% gain) | 0.0064 vs 0.0063 (-2.6% loss) | 0.0081 vs 0.0081 (-1.0% loss) | 0.0130 vs 0.0124 (-4.4% loss) | 0.0346 vs 0.0305 (-13.4% loss) |
| naphthalene | 0.0019 vs 0.0019 (-1.0% loss) | 0.0021 vs 0.0020 (-3.0% loss) | 0.0025 vs 0.0025 (-0.0% loss) | 0.0036 vs 0.0039 (+7.4% gain) | 0.0074 vs 0.0080 (+8.3% gain) | 0.0210 vs 0.0234 (+10.0% gain) |
| toluene | 0.0030 vs 0.0031 (+5.1% gain) | 0.0032 vs 0.0033 (+4.4% gain) | 0.0035 vs 0.0036 (+3.6% gain) | 0.0046 vs 0.0048 (+3.5% gain) | 0.0089 vs 0.0092 (+3.1% gain) | 0.0323 vs 0.0293 (-10.3% loss) |

## Architecture Gain: MLP None vs GNN None

| molecule | H=1 | H=2 | H=4 | H=8 | H=16 | H=32 |
|---|---:|---:|---:|---:|---:|---:|
| aspirin | 0.0260 -> 0.0095 (+63.5% gain) | 0.0274 -> 0.0093 (+65.8% gain) | 0.0304 -> 0.0105 (+65.3% gain) | 0.0446 -> 0.0159 (+64.4% gain) | 0.0806 -> 0.0299 (+62.9% gain) | 0.1953 -> 0.0934 (+52.2% gain) |
| ethanol | 0.0189 -> 0.0097 (+48.5% gain) | 0.0203 -> 0.0096 (+52.5% gain) | 0.0214 -> 0.0117 (+45.3% gain) | 0.0270 -> 0.0135 (+49.9% gain) | 0.0464 -> 0.0230 (+50.4% gain) | 0.1133 -> 0.0557 (+50.8% gain) |
| malonaldehyde | 0.0238 -> 0.0129 (+45.8% gain) | 0.0229 -> 0.0125 (+45.3% gain) | 0.0252 -> 0.0154 (+38.9% gain) | 0.0317 -> 0.0171 (+46.1% gain) | 0.0520 -> 0.0276 (+47.0% gain) | 0.1329 -> 0.0686 (+48.4% gain) |
| naphthalene | 0.0090 -> 0.0052 (+42.3% gain) | 0.0096 -> 0.0057 (+40.2% gain) | 0.0101 -> 0.0077 (+23.0% gain) | 0.0129 -> 0.0124 (+3.9% gain) | 0.0236 -> 0.0236 (-0.3% loss) | 0.0630 -> 0.0637 (-1.2% loss) |
| toluene | 0.0148 -> 0.0055 (+62.7% gain) | 0.0128 -> 0.0067 (+48.0% gain) | 0.0150 -> 0.0082 (+45.4% gain) | 0.0196 -> 0.0132 (+32.7% gain) | 0.0368 -> 0.0236 (+35.9% gain) | 0.1039 -> 0.0609 (+41.3% gain) |

## Molecule Classification

| molecule | class | arch gain H=32 | graph gain H=32 | graph vs permuted H=32 | graph vs random H=32 | rationale |
|---|---|---:|---:|---:|---:|---|
| aspirin | random-control-equivalent | +52.2% gain | +67.2% gain | +3.5% gain | -0.2% loss | true, permuted, and random graph controls are within +/-10% |
| ethanol | graph-smoothing-dominated | +50.8% gain | +19.4% gain | -15.8% loss | -12.2% loss | graph controls improve over GNN none without true-graph specificity |
| malonaldehyde | graph-smoothing-dominated | +48.4% gain | +49.6% gain | -12.2% loss | -13.4% loss | graph controls improve over GNN none without true-graph specificity |
| naphthalene | random-control-equivalent | -1.2% loss | +67.0% gain | +5.5% gain | +10.0% gain | true, permuted, and random graph controls are within +/-10% |
| toluene | graph-smoothing-dominated | +41.3% gain | +47.0% gain | -12.2% loss | -10.3% loss | graph controls improve over GNN none without true-graph specificity |

## Recommendation

Top up from 3 to 5 seeds: aspirin, malonaldehyde, naphthalene, toluene.
- aspirin: random-control-equivalent; true, permuted, and random graph controls are within +/-10%; graph H=32 CV=0.43 over n=3
- malonaldehyde: graph-smoothing-dominated; graph controls improve over GNN none without true-graph specificity; graph H=32 CV=0.43 over n=3
- naphthalene: random-control-equivalent; true, permuted, and random graph controls are within +/-10%; near graph-specificity threshold; graph H=32 CV=0.52 over n=3
- toluene: graph-smoothing-dominated; graph controls improve over GNN none without true-graph specificity; graph H=32 CV=0.65 over n=3
True graph specificity is not supported; the evidence favors generic graph smoothing or control equivalence.
Cycle 3 recommendation: run controlled HO networks before ISO17 to separate graph smoothing from true graph semantics.
