# Master Attribution Summary

Analysis-only master summary for Cycles 1-5. No training, ISO17, rMD17 top-up, or Cycle 6 experiment commands were run.

Inputs:
- `analysis_out/CYCLE1_ASPIRIN_PILOT_REPORT.md`
- `analysis_out/CYCLE2_RMD17_MULTIMOLECULE_REPORT.md`
- `analysis_out/CYCLE3_HO_NETWORKS_REPORT.md`
- `analysis_out/CYCLE4_HO_LAMBDA_ROBUSTNESS_REPORT.md`
- `analysis_out/CYCLE4_GRAPH_SPECTRA_AND_LOSS_DIAGNOSTICS.md`
- `analysis_out/CYCLE5_HO_LATTICE_BRIDGE_REPORT.md`
- `experiments/results/cycle1_aspirin_pilot/cycle1_aspirin_pilot_results.json`
- `experiments/results/cycle2_rmd17_multimolecule/cycle2_rmd17_multimolecule_results.json`
- `experiments/results/cycle3_ho_networks/cycle3_ho_networks_results.json`
- `experiments/results/cycle4_ho_lambda_robustness/cycle4_ho_lambda_robustness_results.json`
- `experiments/results/cycle5_ho_lattice_bridge/cycle5_ho_lattice_bridge_results.json`

Run integrity:
- `cycle1_aspirin_pilot`: 55 ok / 0 failed
- `cycle2_rmd17_multimolecule`: 75 ok / 0 failed
- `cycle3_ho_networks`: 75 ok / 0 failed
- `cycle4_ho_lambda_robustness`: 72 ok / 0 failed
- `cycle5_ho_lattice_bridge`: 75 ok / 0 failed

## Claim 1: Graph-Style Node-Wise Smoothing Improves Long-Horizon Rollout

- Cycle 1 aspirin: GNN none H=32 0.1008 -> graph 0.0330 (+67.3% gain). Permuted/random graph controls are close at 0.0350/0.0336, so the robust part is graph-style smoothing, not exact topology.
- Cycle 2 rMD17: graph-prior H=32 gains over GNN none span aspirin +67.2%, ethanol +19.4%, malonaldehyde +49.6%, naphthalene +67.0%, toluene +47.0%.
- Cycle 3 HO: graph-prior H=32 gains over GNN none are lattice +56.0%, random +52.2%, scalefree +55.2%.

Interpretation: the long-horizon improvement is consistent and large whenever node-wise graph-style smoothing is present, including when the graph is permuted or randomized.

## Claim 2: Covariance Conditioning Does Not Explain Graph-Style Gains

- Cycle 1 aspirin GNN covariance does not reproduce the graph gain: none 0.1008, covariance 0.1192, graph 0.0330. Covariance is -18.2% versus none, while graph is +67.3%.
- Cycle 3 HO covariance is also far from graph-style priors at H=32: covariance -> graph gains are lattice +56.7%, random +52.0%, scalefree +54.8%.

Interpretation: Euclidean covariance conditioning is not the operative explanation for the graph-prior rollout gains.

## Claim 3: Exact Molecular Graph Specificity Is Weak On rMD17

- aspirin: graph vs permuted +3.5%, graph vs random -0.2% at H=32.
- ethanol: graph vs permuted -15.8%, graph vs random -12.2% at H=32.
- malonaldehyde: graph vs permuted -12.2%, graph vs random -13.4% at H=32.
- naphthalene: graph vs permuted +5.5%, graph vs random +10.0% at H=32.
- toluene: graph vs permuted -12.2%, graph vs random -10.3% at H=32.

Interpretation: no rMD17 molecule gives stable, two-control evidence for exact molecular graph specificity. Naphthalene is closest, but it is still within the control-equivalent band against the permuted control and only borderline against random.

## Claim 4: HO Lattice Can Show True Topology Specificity At High Prior Weight

- Cycle 3 lattice at implicit lambda=0.1: graph beats permuted/random at H=32 with deltas 0.0029/0.0030, wins 4/5 and 4/5.
- Cycle 5 lattice bridge at explicit lambda=0.1 reproduces the same result: graph mean 0.0169, permuted 0.0198, random 0.0199; bootstrap CIs [0.0011, 0.0042] and [0.0010, 0.0046].

Interpretation: Cycle 3 was not contradicted by Cycle 4; Cycle 4 simply did not test the Cycle 3 default lambda=0.1.

## Claim 5: Topology Specificity Is Not Lambda-Robust

- Cycle 4 lattice tested lambda values 0.001, 0.005, 0.01, and 0.05; none supported lattice specificity. Cycle 5 confirmed the same no-support result for those four lambdas with five seeds.
- Cycle 5 supports lattice specificity only at lambda=0.1. Lower lambda settings often favor permuted/random controls at H=32.
- Cycle 4 scalefree has an isolated lambda=0.01 support row under n=3, but adjacent lambda values reverse or weaken it; this is not robust enough for a broad topology-specific claim.

Interpretation: specificity is high-lambda-conditional, not a lambda-robust property of the method.

## Claim 6: Permuted Controls Are Scale-Matched

- Cycle 4 graph spectra diagnostics show true and permuted graph Laplacians are spectrum-matched: lattice fro/lambda_max +0.0%/+0.0%; scalefree fro/lambda_max +0.0%/+0.0%.
- Random controls match node count, edge count, and trace, but not full spectral scale: lattice random fro/lambda_max +8.4%/+32.3%; scalefree random fro/lambda_max -10.1%/-32.7%.

Interpretation: whenever a permuted graph matches or beats the true graph, that result cannot be dismissed as a Laplacian-scale artifact. It directly weakens node-label/topology semantics.

## Recommended Paper Thesis

The defensible thesis is: graph-style node-wise regularization is a strong long-horizon rollout stabilizer, but most observed gains are attributable to generic graph smoothing rather than exact molecular topology semantics. HO lattice provides a controlled positive example of true topology specificity, but only at high prior weight lambda=0.1; therefore the paper should not claim lambda-robust true-topology specificity. The main claim should emphasize generic graph-structured smoothing, with topology specificity presented as conditional and limited.

Generated companion tables:
- `analysis_out/master_rollout_table.csv`
- `analysis_out/master_graph_specificity_table.csv`
- `analysis_out/master_decision_rules.md`
- `paper/figures_plan.md`
