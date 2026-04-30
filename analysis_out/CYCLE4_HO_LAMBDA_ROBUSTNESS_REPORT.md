# Cycle 4 HO Lambda Robustness Report

Results file: `experiments/results/cycle4_ho_lambda_robustness/cycle4_ho_lambda_robustness_results.json`
Schema version: `cycle4_ho_lambda_robustness_v1`
Success/failure count: 72 ok / 0 failed
Missing expected runs: 0
Runs with non-finite rollout diagnostics: none
Graph prior metadata node-wise: YES

## Rollout Error Mean +/- Std

| topology | lambda | prior | H=1 | H=2 | H=4 | H=8 | H=16 | H=32 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| lattice | 0.001 | graph | 0.0006 +/- 0.0002 (n=3) | 0.0012 +/- 0.0003 (n=3) | 0.0023 +/- 0.0006 (n=3) | 0.0049 +/- 0.0009 (n=3) | 0.0117 +/- 0.0014 (n=3) | 0.0381 +/- 0.0033 (n=3) |
| lattice | 0.001 | permuted_graph | 0.0006 +/- 0.0004 (n=3) | 0.0012 +/- 0.0007 (n=3) | 0.0025 +/- 0.0014 (n=3) | 0.0050 +/- 0.0027 (n=3) | 0.0103 +/- 0.0054 (n=3) | 0.0285 +/- 0.0157 (n=3) |
| lattice | 0.001 | random_graph | 0.0006 +/- 0.0003 (n=3) | 0.0012 +/- 0.0006 (n=3) | 0.0024 +/- 0.0012 (n=3) | 0.0049 +/- 0.0024 (n=3) | 0.0103 +/- 0.0050 (n=3) | 0.0301 +/- 0.0159 (n=3) |
| lattice | 0.005 | graph | 0.0005 +/- 0.0001 (n=3) | 0.0010 +/- 0.0002 (n=3) | 0.0019 +/- 0.0004 (n=3) | 0.0038 +/- 0.0008 (n=3) | 0.0082 +/- 0.0019 (n=3) | 0.0219 +/- 0.0062 (n=3) |
| lattice | 0.005 | permuted_graph | 0.0004 +/- 0.0000 (n=3) | 0.0007 +/- 0.0001 (n=3) | 0.0015 +/- 0.0002 (n=3) | 0.0029 +/- 0.0005 (n=3) | 0.0062 +/- 0.0014 (n=3) | 0.0171 +/- 0.0062 (n=3) |
| lattice | 0.005 | random_graph | 0.0004 +/- 0.0001 (n=3) | 0.0008 +/- 0.0001 (n=3) | 0.0016 +/- 0.0002 (n=3) | 0.0031 +/- 0.0005 (n=3) | 0.0065 +/- 0.0014 (n=3) | 0.0194 +/- 0.0079 (n=3) |
| lattice | 0.01 | graph | 0.0003 +/- 0.0001 (n=3) | 0.0007 +/- 0.0001 (n=3) | 0.0013 +/- 0.0002 (n=3) | 0.0026 +/- 0.0004 (n=3) | 0.0057 +/- 0.0013 (n=3) | 0.0182 +/- 0.0102 (n=3) |
| lattice | 0.01 | permuted_graph | 0.0002 +/- 0.0001 (n=3) | 0.0005 +/- 0.0001 (n=3) | 0.0010 +/- 0.0003 (n=3) | 0.0018 +/- 0.0005 (n=3) | 0.0036 +/- 0.0011 (n=3) | 0.0093 +/- 0.0032 (n=3) |
| lattice | 0.01 | random_graph | 0.0002 +/- 0.0001 (n=3) | 0.0005 +/- 0.0001 (n=3) | 0.0010 +/- 0.0003 (n=3) | 0.0019 +/- 0.0006 (n=3) | 0.0037 +/- 0.0012 (n=3) | 0.0098 +/- 0.0049 (n=3) |
| lattice | 0.05 | graph | 0.0003 +/- 0.0001 (n=3) | 0.0005 +/- 0.0002 (n=3) | 0.0011 +/- 0.0004 (n=3) | 0.0022 +/- 0.0008 (n=3) | 0.0048 +/- 0.0020 (n=3) | 0.0166 +/- 0.0114 (n=3) |
| lattice | 0.05 | permuted_graph | 0.0003 +/- 0.0001 (n=3) | 0.0006 +/- 0.0001 (n=3) | 0.0011 +/- 0.0003 (n=3) | 0.0023 +/- 0.0006 (n=3) | 0.0051 +/- 0.0016 (n=3) | 0.0166 +/- 0.0081 (n=3) |
| lattice | 0.05 | random_graph | 0.0003 +/- 0.0001 (n=3) | 0.0006 +/- 0.0001 (n=3) | 0.0012 +/- 0.0003 (n=3) | 0.0024 +/- 0.0006 (n=3) | 0.0053 +/- 0.0017 (n=3) | 0.0173 +/- 0.0082 (n=3) |
| scalefree | 0.001 | graph | 0.0006 +/- 0.0001 (n=3) | 0.0012 +/- 0.0001 (n=3) | 0.0023 +/- 0.0003 (n=3) | 0.0044 +/- 0.0005 (n=3) | 0.0085 +/- 0.0016 (n=3) | 0.0230 +/- 0.0077 (n=3) |
| scalefree | 0.001 | permuted_graph | 0.0007 +/- 0.0004 (n=3) | 0.0014 +/- 0.0008 (n=3) | 0.0028 +/- 0.0016 (n=3) | 0.0056 +/- 0.0033 (n=3) | 0.0127 +/- 0.0071 (n=3) | 0.0354 +/- 0.0163 (n=3) |
| scalefree | 0.001 | random_graph | 0.0007 +/- 0.0003 (n=3) | 0.0013 +/- 0.0007 (n=3) | 0.0028 +/- 0.0014 (n=3) | 0.0058 +/- 0.0031 (n=3) | 0.0134 +/- 0.0067 (n=3) | 0.0393 +/- 0.0149 (n=3) |
| scalefree | 0.005 | graph | 0.0005 +/- 0.0001 (n=3) | 0.0010 +/- 0.0001 (n=3) | 0.0020 +/- 0.0002 (n=3) | 0.0041 +/- 0.0005 (n=3) | 0.0095 +/- 0.0016 (n=3) | 0.0337 +/- 0.0120 (n=3) |
| scalefree | 0.005 | permuted_graph | 0.0004 +/- 0.0001 (n=3) | 0.0009 +/- 0.0002 (n=3) | 0.0017 +/- 0.0005 (n=3) | 0.0032 +/- 0.0010 (n=3) | 0.0063 +/- 0.0018 (n=3) | 0.0157 +/- 0.0028 (n=3) |
| scalefree | 0.005 | random_graph | 0.0004 +/- 0.0001 (n=3) | 0.0009 +/- 0.0001 (n=3) | 0.0017 +/- 0.0003 (n=3) | 0.0031 +/- 0.0005 (n=3) | 0.0059 +/- 0.0009 (n=3) | 0.0147 +/- 0.0024 (n=3) |
| scalefree | 0.01 | graph | 0.0004 +/- 0.0001 (n=3) | 0.0008 +/- 0.0003 (n=3) | 0.0015 +/- 0.0005 (n=3) | 0.0030 +/- 0.0009 (n=3) | 0.0059 +/- 0.0016 (n=3) | 0.0130 +/- 0.0028 (n=3) |
| scalefree | 0.01 | permuted_graph | 0.0003 +/- 0.0001 (n=3) | 0.0006 +/- 0.0001 (n=3) | 0.0013 +/- 0.0002 (n=3) | 0.0025 +/- 0.0004 (n=3) | 0.0052 +/- 0.0006 (n=3) | 0.0140 +/- 0.0014 (n=3) |
| scalefree | 0.01 | random_graph | 0.0004 +/- 0.0001 (n=3) | 0.0008 +/- 0.0003 (n=3) | 0.0015 +/- 0.0005 (n=3) | 0.0030 +/- 0.0010 (n=3) | 0.0066 +/- 0.0020 (n=3) | 0.0235 +/- 0.0113 (n=3) |
| scalefree | 0.05 | graph | 0.0004 +/- 0.0001 (n=3) | 0.0007 +/- 0.0002 (n=3) | 0.0014 +/- 0.0004 (n=3) | 0.0028 +/- 0.0007 (n=3) | 0.0063 +/- 0.0014 (n=3) | 0.0233 +/- 0.0101 (n=3) |
| scalefree | 0.05 | permuted_graph | 0.0004 +/- 0.0001 (n=3) | 0.0007 +/- 0.0002 (n=3) | 0.0014 +/- 0.0005 (n=3) | 0.0028 +/- 0.0009 (n=3) | 0.0059 +/- 0.0020 (n=3) | 0.0175 +/- 0.0063 (n=3) |
| scalefree | 0.05 | random_graph | 0.0003 +/- 0.0001 (n=3) | 0.0007 +/- 0.0002 (n=3) | 0.0014 +/- 0.0005 (n=3) | 0.0027 +/- 0.0009 (n=3) | 0.0058 +/- 0.0018 (n=3) | 0.0174 +/- 0.0054 (n=3) |

## Paired Graph-Specificity Deltas

Deltas are paired by seed as control rollout error minus graph rollout error. Positive values mean graph has lower error than the control.

| topology | lambda | control | horizon | mean delta +/- std | graph wins | bootstrap 95% CI |
|---|---:|---|---:|---:|---:|---:|
| lattice | 0.001 | permuted_graph | H=1 | 0.0000 +/- 0.0003 | 1/3 | [-0.0002, 0.0004] |
| lattice | 0.001 | permuted_graph | H=2 | 0.0001 +/- 0.0007 | 1/3 | [-0.0004, 0.0008] |
| lattice | 0.001 | permuted_graph | H=4 | 0.0001 +/- 0.0013 | 1/3 | [-0.0009, 0.0016] |
| lattice | 0.001 | permuted_graph | H=8 | 0.0000 +/- 0.0027 | 1/3 | [-0.0023, 0.0030] |
| lattice | 0.001 | permuted_graph | H=16 | -0.0014 +/- 0.0062 | 1/3 | [-0.0074, 0.0050] |
| lattice | 0.001 | permuted_graph | H=32 | -0.0096 +/- 0.0188 | 1/3 | [-0.0283, 0.0092] |
| lattice | 0.001 | random_graph | H=1 | 0.0000 +/- 0.0003 | 1/3 | [-0.0002, 0.0003] |
| lattice | 0.001 | random_graph | H=2 | 0.0001 +/- 0.0005 | 1/3 | [-0.0004, 0.0006] |
| lattice | 0.001 | random_graph | H=4 | 0.0001 +/- 0.0010 | 2/3 | [-0.0009, 0.0011] |
| lattice | 0.001 | random_graph | H=8 | -0.0001 +/- 0.0021 | 2/3 | [-0.0023, 0.0020] |
| lattice | 0.001 | random_graph | H=16 | -0.0014 +/- 0.0055 | 1/3 | [-0.0074, 0.0033] |
| lattice | 0.001 | random_graph | H=32 | -0.0079 +/- 0.0187 | 1/3 | [-0.0280, 0.0091] |
| lattice | 0.005 | permuted_graph | H=1 | -0.0001 +/- 0.0001 | 1/3 | [-0.0002, 0.0000] |
| lattice | 0.005 | permuted_graph | H=2 | -0.0002 +/- 0.0002 | 1/3 | [-0.0004, 0.0000] |
| lattice | 0.005 | permuted_graph | H=4 | -0.0004 +/- 0.0005 | 1/3 | [-0.0008, 0.0001] |
| lattice | 0.005 | permuted_graph | H=8 | -0.0009 +/- 0.0010 | 1/3 | [-0.0017, 0.0003] |
| lattice | 0.005 | permuted_graph | H=16 | -0.0020 +/- 0.0029 | 1/3 | [-0.0039, 0.0013] |
| lattice | 0.005 | permuted_graph | H=32 | -0.0048 +/- 0.0102 | 1/3 | [-0.0108, 0.0070] |
| lattice | 0.005 | random_graph | H=1 | -0.0001 +/- 0.0001 | 1/3 | [-0.0001, 0.0000] |
| lattice | 0.005 | random_graph | H=2 | -0.0002 +/- 0.0002 | 1/3 | [-0.0002, 0.0000] |
| lattice | 0.005 | random_graph | H=4 | -0.0003 +/- 0.0003 | 1/3 | [-0.0005, 0.0001] |
| lattice | 0.005 | random_graph | H=8 | -0.0007 +/- 0.0009 | 1/3 | [-0.0013, 0.0003] |
| lattice | 0.005 | random_graph | H=16 | -0.0018 +/- 0.0026 | 1/3 | [-0.0033, 0.0013] |
| lattice | 0.005 | random_graph | H=32 | -0.0025 +/- 0.0092 | 1/3 | [-0.0111, 0.0072] |
| lattice | 0.01 | permuted_graph | H=1 | -0.0001 +/- 0.0000 | 0/3 | [-0.0001, -0.0001] |
| lattice | 0.01 | permuted_graph | H=2 | -0.0002 +/- 0.0001 | 0/3 | [-0.0002, -0.0001] |
| lattice | 0.01 | permuted_graph | H=4 | -0.0003 +/- 0.0001 | 0/3 | [-0.0004, -0.0002] |
| lattice | 0.01 | permuted_graph | H=8 | -0.0008 +/- 0.0003 | 0/3 | [-0.0010, -0.0004] |
| lattice | 0.01 | permuted_graph | H=16 | -0.0021 +/- 0.0010 | 0/3 | [-0.0031, -0.0010] |
| lattice | 0.01 | permuted_graph | H=32 | -0.0088 +/- 0.0081 | 0/3 | [-0.0181, -0.0029] |
| lattice | 0.01 | random_graph | H=1 | -0.0001 +/- 0.0000 | 0/3 | [-0.0001, -0.0001] |
| lattice | 0.01 | random_graph | H=2 | -0.0002 +/- 0.0000 | 0/3 | [-0.0002, -0.0001] |
| lattice | 0.01 | random_graph | H=4 | -0.0003 +/- 0.0001 | 0/3 | [-0.0004, -0.0002] |
| lattice | 0.01 | random_graph | H=8 | -0.0007 +/- 0.0002 | 0/3 | [-0.0009, -0.0005] |
| lattice | 0.01 | random_graph | H=16 | -0.0020 +/- 0.0007 | 0/3 | [-0.0026, -0.0011] |
| lattice | 0.01 | random_graph | H=32 | -0.0083 +/- 0.0059 | 0/3 | [-0.0151, -0.0038] |
| lattice | 0.05 | permuted_graph | H=1 | 0.0000 +/- 0.0000 | 2/3 | [-0.0000, 0.0000] |
| lattice | 0.05 | permuted_graph | H=2 | 0.0000 +/- 0.0001 | 2/3 | [-0.0001, 0.0001] |
| lattice | 0.05 | permuted_graph | H=4 | 0.0000 +/- 0.0001 | 2/3 | [-0.0001, 0.0001] |
| lattice | 0.05 | permuted_graph | H=8 | 0.0001 +/- 0.0002 | 2/3 | [-0.0002, 0.0003] |
| lattice | 0.05 | permuted_graph | H=16 | 0.0004 +/- 0.0009 | 2/3 | [-0.0006, 0.0011] |
| lattice | 0.05 | permuted_graph | H=32 | 0.0000 +/- 0.0046 | 2/3 | [-0.0050, 0.0042] |
| lattice | 0.05 | random_graph | H=1 | 0.0000 +/- 0.0000 | 2/3 | [-0.0000, 0.0000] |
| lattice | 0.05 | random_graph | H=2 | 0.0000 +/- 0.0001 | 2/3 | [-0.0001, 0.0001] |
| lattice | 0.05 | random_graph | H=4 | 0.0001 +/- 0.0001 | 2/3 | [-0.0001, 0.0002] |
| lattice | 0.05 | random_graph | H=8 | 0.0002 +/- 0.0003 | 2/3 | [-0.0001, 0.0005] |
| lattice | 0.05 | random_graph | H=16 | 0.0005 +/- 0.0011 | 2/3 | [-0.0006, 0.0016] |
| lattice | 0.05 | random_graph | H=32 | 0.0007 +/- 0.0059 | 2/3 | [-0.0052, 0.0065] |
| scalefree | 0.001 | permuted_graph | H=1 | 0.0001 +/- 0.0004 | 1/3 | [-0.0002, 0.0006] |
| scalefree | 0.001 | permuted_graph | H=2 | 0.0002 +/- 0.0008 | 1/3 | [-0.0003, 0.0011] |
| scalefree | 0.001 | permuted_graph | H=4 | 0.0005 +/- 0.0015 | 1/3 | [-0.0005, 0.0022] |
| scalefree | 0.001 | permuted_graph | H=8 | 0.0013 +/- 0.0029 | 1/3 | [-0.0007, 0.0046] |
| scalefree | 0.001 | permuted_graph | H=16 | 0.0042 +/- 0.0057 | 3/3 | [0.0003, 0.0108] |
| scalefree | 0.001 | permuted_graph | H=32 | 0.0123 +/- 0.0136 | 2/3 | [-0.0011, 0.0261] |
| scalefree | 0.001 | random_graph | H=1 | 0.0001 +/- 0.0003 | 1/3 | [-0.0002, 0.0004] |
| scalefree | 0.001 | random_graph | H=2 | 0.0002 +/- 0.0007 | 1/3 | [-0.0003, 0.0009] |
| scalefree | 0.001 | random_graph | H=4 | 0.0005 +/- 0.0013 | 1/3 | [-0.0005, 0.0020] |
| scalefree | 0.001 | random_graph | H=8 | 0.0014 +/- 0.0027 | 2/3 | [-0.0007, 0.0045] |
| scalefree | 0.001 | random_graph | H=16 | 0.0049 +/- 0.0054 | 3/3 | [0.0016, 0.0111] |
| scalefree | 0.001 | random_graph | H=32 | 0.0163 +/- 0.0141 | 3/3 | [0.0005, 0.0278] |
| scalefree | 0.005 | permuted_graph | H=1 | -0.0001 +/- 0.0001 | 1/3 | [-0.0001, 0.0001] |
| scalefree | 0.005 | permuted_graph | H=2 | -0.0001 +/- 0.0002 | 1/3 | [-0.0003, 0.0001] |
| scalefree | 0.005 | permuted_graph | H=4 | -0.0003 +/- 0.0005 | 1/3 | [-0.0006, 0.0003] |
| scalefree | 0.005 | permuted_graph | H=8 | -0.0008 +/- 0.0012 | 1/3 | [-0.0017, 0.0005] |
| scalefree | 0.005 | permuted_graph | H=16 | -0.0032 +/- 0.0031 | 1/3 | [-0.0056, 0.0003] |
| scalefree | 0.005 | permuted_graph | H=32 | -0.0180 +/- 0.0140 | 0/3 | [-0.0268, -0.0018] |
| scalefree | 0.005 | random_graph | H=1 | -0.0001 +/- 0.0001 | 1/3 | [-0.0001, 0.0000] |
| scalefree | 0.005 | random_graph | H=2 | -0.0001 +/- 0.0002 | 1/3 | [-0.0003, 0.0001] |
| scalefree | 0.005 | random_graph | H=4 | -0.0003 +/- 0.0003 | 1/3 | [-0.0006, 0.0001] |
| scalefree | 0.005 | random_graph | H=8 | -0.0009 +/- 0.0008 | 0/3 | [-0.0016, -0.0000] |
| scalefree | 0.005 | random_graph | H=16 | -0.0036 +/- 0.0022 | 0/3 | [-0.0053, -0.0011] |
| scalefree | 0.005 | random_graph | H=32 | -0.0190 +/- 0.0143 | 0/3 | [-0.0297, -0.0027] |
| scalefree | 0.01 | permuted_graph | H=1 | -0.0001 +/- 0.0001 | 0/3 | [-0.0002, -0.0000] |
| scalefree | 0.01 | permuted_graph | H=2 | -0.0002 +/- 0.0002 | 0/3 | [-0.0004, -0.0000] |
| scalefree | 0.01 | permuted_graph | H=4 | -0.0003 +/- 0.0004 | 0/3 | [-0.0007, -0.0001] |
| scalefree | 0.01 | permuted_graph | H=8 | -0.0005 +/- 0.0006 | 0/3 | [-0.0012, -0.0001] |
| scalefree | 0.01 | permuted_graph | H=16 | -0.0007 +/- 0.0011 | 1/3 | [-0.0019, 0.0001] |
| scalefree | 0.01 | permuted_graph | H=32 | 0.0010 +/- 0.0016 | 3/3 | [0.0000, 0.0029] |
| scalefree | 0.01 | random_graph | H=1 | -0.0000 +/- 0.0000 | 1/3 | [-0.0000, 0.0000] |
| scalefree | 0.01 | random_graph | H=2 | -0.0000 +/- 0.0001 | 1/3 | [-0.0001, 0.0000] |
| scalefree | 0.01 | random_graph | H=4 | -0.0001 +/- 0.0001 | 1/3 | [-0.0002, 0.0001] |
| scalefree | 0.01 | random_graph | H=8 | -0.0000 +/- 0.0003 | 1/3 | [-0.0002, 0.0003] |
| scalefree | 0.01 | random_graph | H=16 | 0.0007 +/- 0.0009 | 2/3 | [-0.0001, 0.0017] |
| scalefree | 0.01 | random_graph | H=32 | 0.0105 +/- 0.0100 | 3/3 | [0.0029, 0.0218] |
| scalefree | 0.05 | permuted_graph | H=1 | -0.0000 +/- 0.0000 | 1/3 | [-0.0000, 0.0000] |
| scalefree | 0.05 | permuted_graph | H=2 | -0.0000 +/- 0.0001 | 1/3 | [-0.0001, 0.0001] |
| scalefree | 0.05 | permuted_graph | H=4 | -0.0000 +/- 0.0001 | 1/3 | [-0.0002, 0.0001] |
| scalefree | 0.05 | permuted_graph | H=8 | -0.0001 +/- 0.0002 | 2/3 | [-0.0003, 0.0001] |
| scalefree | 0.05 | permuted_graph | H=16 | -0.0004 +/- 0.0007 | 1/3 | [-0.0008, 0.0004] |
| scalefree | 0.05 | permuted_graph | H=32 | -0.0058 +/- 0.0067 | 1/3 | [-0.0131, 0.0001] |
| scalefree | 0.05 | random_graph | H=1 | -0.0000 +/- 0.0000 | 1/3 | [-0.0000, 0.0000] |
| scalefree | 0.05 | random_graph | H=2 | -0.0000 +/- 0.0001 | 1/3 | [-0.0001, 0.0001] |
| scalefree | 0.05 | random_graph | H=4 | -0.0001 +/- 0.0001 | 1/3 | [-0.0002, 0.0001] |
| scalefree | 0.05 | random_graph | H=8 | -0.0001 +/- 0.0002 | 1/3 | [-0.0004, 0.0001] |
| scalefree | 0.05 | random_graph | H=16 | -0.0005 +/- 0.0005 | 1/3 | [-0.0008, 0.0001] |
| scalefree | 0.05 | random_graph | H=32 | -0.0059 +/- 0.0065 | 0/3 | [-0.0133, -0.0011] |

## Specificity Summary

Support requires H=32 paired evidence against both controls: positive mean delta, 3/3 graph wins, and bootstrap CI lower bound above zero.

| topology | lambda | graph_specificity_supported? | H=32 evidence |
|---|---:|---|---|
| lattice | 0.001 | NO | perm delta=-0.0096, wins=1/3, CI=[-0.0283, 0.0092]; rand delta=-0.0079, wins=1/3, CI=[-0.0280, 0.0091] |
| lattice | 0.005 | NO | perm delta=-0.0048, wins=1/3, CI=[-0.0108, 0.0070]; rand delta=-0.0025, wins=1/3, CI=[-0.0111, 0.0072] |
| lattice | 0.01 | NO | perm delta=-0.0088, wins=0/3, CI=[-0.0181, -0.0029]; rand delta=-0.0083, wins=0/3, CI=[-0.0151, -0.0038] |
| lattice | 0.05 | NO | perm delta=0.0000, wins=2/3, CI=[-0.0050, 0.0042]; rand delta=0.0007, wins=2/3, CI=[-0.0052, 0.0065] |
| scalefree | 0.001 | NO | perm delta=0.0123, wins=2/3, CI=[-0.0011, 0.0261]; rand delta=0.0163, wins=3/3, CI=[0.0005, 0.0278] |
| scalefree | 0.005 | NO | perm delta=-0.0180, wins=0/3, CI=[-0.0268, -0.0018]; rand delta=-0.0190, wins=0/3, CI=[-0.0297, -0.0027] |
| scalefree | 0.01 | YES | perm delta=0.0010, wins=3/3, CI=[0.0000, 0.0029]; rand delta=0.0105, wins=3/3, CI=[0.0029, 0.0218] |
| scalefree | 0.05 | NO | perm delta=-0.0058, wins=1/3, CI=[-0.0131, 0.0001]; rand delta=-0.0059, wins=0/3, CI=[-0.0133, -0.0011] |

## Recommendation

Lattice specificity is not supported across the tested lambda values.
Scalefree does not cleanly remain random-control-equivalent; inspect lambda-specific exceptions before making the claim.
Recommendation: do not run ISO17 yet; resolve lambda sensitivity before expanding to ISO17.
