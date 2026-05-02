Dear Editor,

We submit our manuscript, “A Model-Conditioned Preflight Protocol for Prior Selection in Node-Wise Scientific Latent Dynamics,” for consideration in the Journal of Computational Science.

Graph priors are widely used in scientific latent-dynamics models, including molecular dynamics, particle simulations, traffic networks, and graph-discretized physical fields. In practice, a candidate graph topology is often available before training, and practitioners commonly compare a graph-prior model against a no-prior baseline. When the graph-prior model improves, the gain is often attributed to the candidate topology itself. Our work shows that, under the protocols and model conditions tested here, this attribution can be unsafe.

We propose a model-conditioned preflight protocol that decomposes graph-prior efficacy into three operational attribution classes: topology-aligned support, where the candidate graph improves over both no-prior training and a spectrum-matched permuted graph (SMPG) control; generic regularization, where the graph prior improves over no-prior training but does not survive SMPG controls; and no effect.

Across 13 measurement cells spanning seven regimes—HO lattices, N-body distance graphs, METR-LA traffic, graph-heat and graph-wave lattices, spring-mass systems, and five rMD17 molecules—the protocol identifies one controlled audit-positive case, eight generic-regularization cells, and four no-effect cells. Across the eight generic-regularization cells, effect sizes against the no-prior baseline range from 2.4 to 8.4, whereas effect sizes against the SMPG control range from -2.9 to 0.9 with confidence-interval overlap in every cell. Without SMPG and calibrated temporal-smoothing controls, these cases would have appeared to support topology-specific graph-prior effects.

The contribution is methodological rather than architectural: we provide a practical preflight workflow for prior selection and attribution under a fixed model condition. We do not introduce a new graph prior, claim universal physics prediction, frame the task as causal discovery, or present a state-of-the-art benchmark.

We believe the manuscript is a good fit for the Journal of Computational Science because it addresses a recurring methodological hazard in computational physics, molecular machine learning, and graph-based scientific modeling. The protocol is implemented as an adapter-based screening tool with standardized run artifacts, and the code is available at:
https://github.com/2GyounnnG/causalworld

The manuscript has not been published elsewhere and is not under consideration by any other journal. The author declares no conflicts of interest. Generative AI tools were not used to generate scientific results, numerical analyses, or raw experimental artifacts. All quantitative figures in the manuscript are produced from raw run artifacts using matplotlib.

Thank you for your consideration.

Sincerely,
Ruiqi Wang
Department of Physics
University of California San Diego
riw010@ucsd.edu
