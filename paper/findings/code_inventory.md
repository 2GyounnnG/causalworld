# Code Inventory

This inventory is based on the repository code, not on paper prose. Line numbers point to the start of the relevant function or class.

## Section 1: Top-level pipeline

1. **Choose experiment family.** Wolfram-style synthetic hypergraph runs use `train.py`; rMD17 molecular trajectory runs use `scripts/train_rmd17.py`.
2. **Create observations.** Wolfram runs generate rewrite episodes from `CausalWorldEnv`; rMD17 runs load `.npz` molecular frames and convert each frame to a PyTorch Geometric `HeteroData` graph.
3. **Flatten or sample transitions.** Wolfram episodes become `(obs, action, next_obs, reward, done, causal_graph)` transitions. rMD17 samples `(frame_t, frame_t+horizon)` pairs with recorded `frame_idx`.
4. **Adapt graph schema.** rMD17 atom/bond graphs are converted into the same `"node"` / `"hyperedge"` schema expected by `WorldModel`; Wolfram observations already use that schema.
5. **Encode state.** `WorldModel` uses either `HypergraphEncoder` or `FlatMLPEncoder` to map each observation to a latent vector.
6. **Predict latent transition.** `TransitionNetwork` takes latent state plus scalar action and predicts next latent, reward, and done logit.
7. **Apply base losses and prior.** Base training loss is next-latent MSE plus reward/done losses and small latent regularizers. Optional priors are Euclidean covariance or spectral Laplacian penalties.
8. **Roll out in latent space.** Evaluation rolls the latent state forward for horizons such as 1, 2, 4, 8, and 16, then compares predicted latent to the encoder's latent for the true future observation.
9. **Save artifacts.** Runners save JSON results, optional checkpoints, frame-index metadata, aggregate CSVs, reports, and plots.

## Section 2: Core functions inventory

### 1. WorldModel forward pass and loss computation

- **Name**: `WorldModel`
- **File**: `model.py:364`
- **Purpose**: Wrapper around an encoder and transition network. It defines the main learned world model used by both Wolfram and rMD17 experiments.
- **Inputs**: Constructor takes `encoder`, `hidden_dim`, `latent_dim`, `action_dim`, and `transition_hidden_dim`.
- **Outputs**: A PyTorch module with `.encode`, `.forward`, `.loss`, and `.rollout_latent`.
- **Key implementation detail**: Only `"hypergraph"` and `"flat"` are valid `encoder` values; `CausalGraphEncoder` exists but is not selectable through `WorldModel`.
- **Called by**: `train.train_one`, `scripts.train_rmd17.train_one_seed`, validation runners, diagnostics, checkpoint loading.
- **Critical for**: All reported rollout errors and all prior comparisons.

- **Name**: `WorldModel.forward`
- **File**: `model.py:394`
- **Purpose**: Encodes one observation and predicts the next latent state, reward, and done logit for one action.
- **Inputs**: `observation: HeteroData`, `action` as tensor/int/float.
- **Outputs**: Dict with `latent`, `next_latent_pred`, `reward_pred`, and `done_logit`.
- **Key implementation detail**: This is per-observation unless the encoder itself returns batched latents. The normal training loops iterate over samples inside a minibatch.
- **Called by**: `WorldModel.loss`; also demo code calls the model directly.
- **Critical for**: The basic claim that the model learns latent dynamics.

- **Name**: `WorldModel.loss`
- **File**: `model.py:406`
- **Purpose**: Computes one transition's training losses.
- **Inputs**: Current observation, action, next observation, reward, optional done, regularizer weights, prior name, prior weight, and optional Laplacian.
- **Outputs**: Dict containing `total`, `transition`, `reward`, `done`, `latent_l2`, `smoothness`, and `prior` tensors.
- **Key implementation detail**: The target next latent is detached, so gradients flow through the current-state encoder and transition network, not through the target encoding. Euclidean prior inside this function is degenerate for a single sample; the main training loops handle Euclidean covariance at minibatch level instead.
- **Called by**: `train.train_one`, `scripts.train_rmd17.train_one_seed`, `scripts/run_10seed_flat.py:286`, diagnostics, and audit scripts.
- **Critical for**: All training-loss claims; especially whether spectral and Euclidean priors are actually applied.

- **Name**: `TransitionNetwork`
- **File**: `model.py:204`
- **Purpose**: Predicts next latent, reward, and done from latent plus action.
- **Inputs**: `latent_dim`, `action_dim`, `hidden_dim`, `residual`; forward takes latent tensor and action.
- **Outputs**: `(next_latent, reward, done_logit)`.
- **Key implementation detail**: Uses a residual latent transition by default: `next_latent = latent + delta`.
- **Called by**: `WorldModel.forward`, `WorldModel.rollout_latent`.
- **Critical for**: Multi-step rollout behavior and long-horizon error curves.

- **Name**: `WorldModel.rollout_latent`
- **File**: `model.py:478`
- **Purpose**: Applies the transition network repeatedly in latent space.
- **Inputs**: Initial latent tensor and a list of actions.
- **Outputs**: Tensor stack of latents including the initial latent and each predicted step.
- **Key implementation detail**: Rollout never decodes back to physical state; evaluation compares predicted latent to an encoded true future observation.
- **Called by**: `train.evaluate_rollout`, `scripts.train_rmd17.evaluate_rollout`, `scripts.train_rmd17.quick_rollout_eval`.
- **Critical for**: Horizon-scaling claims.

- **Name**: `latent_l2_regularizer`
- **File**: `model.py:277`
- **Purpose**: Penalizes large latent values.
- **Inputs**: Latent tensor.
- **Outputs**: Mean squared latent magnitude.
- **Key implementation detail**: Always included in `WorldModel.loss` with default weight `1e-4`.
- **Called by**: `WorldModel.loss`.
- **Critical for**: Stabilizing all model training; not a headline paper claim.

- **Name**: `temporal_smoothness_regularizer`
- **File**: `model.py:283`
- **Purpose**: Penalizes large predicted latent movements.
- **Inputs**: Current latent and predicted next latent.
- **Outputs**: Mean squared latent difference.
- **Key implementation detail**: Always included in `WorldModel.loss` with default weight `1e-3`.
- **Called by**: `WorldModel.loss`.
- **Critical for**: Stabilizing rollout training; not isolated in reported prior comparisons.

### 2. Encoder variants

- **Name**: `HypergraphEncoder`
- **File**: `model.py:27`
- **Purpose**: Encodes a bipartite hypergraph observation into one latent vector using heterogeneous message passing.
- **Inputs**: `HeteroData` with `"node"` and `"hyperedge"` node types plus `"member_of"` and `"has_member"` edge types.
- **Outputs**: Latent tensor of shape `(latent_dim,)`.
- **Key implementation detail**: Uses two `HeteroConv` layers with `SAGEConv` in both node-to-hyperedge and hyperedge-to-node directions, then separately mean-pools node and hyperedge embeddings.
- **Called by**: `WorldModel(encoder="hypergraph")`; Wolfram hypergraph runners; possible rMD17 configs if `encoder="hypergraph"`.
- **Critical for**: Encoder-comparison claims, especially flat vs hypergraph validation.

- **Name**: `FlatMLPEncoder`
- **File**: `model.py:100`
- **Purpose**: Encodes hand-built graph summary statistics into a latent vector.
- **Inputs**: `HeteroData` in `"node"` / `"hyperedge"` schema.
- **Outputs**: Latent tensor of shape `(latent_dim,)`.
- **Key implementation detail**: Uses six scalar features: number of nodes, number of hyperedges, mean arity, max arity, fraction of hyperedges with known origin, and number of rewrite events inferred from origins.
- **Called by**: `WorldModel(encoder="flat")`; all main rMD17 runners; Wolfram flat runners.
- **Critical for**: Main rMD17 results and flat-encoder baseline/spectral comparisons.

- **Name**: `CausalGraphEncoder`
- **File**: `model.py:153`
- **Purpose**: Encodes event-causal graph summary statistics into a latent vector.
- **Inputs**: `networkx.DiGraph`.
- **Outputs**: Latent tensor of shape `(latent_dim,)`.
- **Key implementation detail**: It is not connected to `WorldModel`; it appears in demo/diagnostic code only.
- **Called by**: `model._demo`; no production training path found.
- **Critical for**: No current paper claim unless the paper describes a separate causal-graph encoder.

### 3. Prior penalties

- **Name**: `euclidean_cov_penalty`
- **File**: `model.py:308`
- **Purpose**: Penalizes deviation of latent covariance from the identity matrix.
- **Inputs**: Latent batch tensor `(B, D)` or one latent vector.
- **Outputs**: Frobenius norm of `Cov(z) - I`, or zero if batch size is less than 2.
- **Key implementation detail**: Needs a real batch. The main training scripts intentionally collect `batch_latents` and apply this once per minibatch.
- **Called by**: `WorldModel.loss` for direct Euclidean mode; `train.train_one`; `scripts.train_rmd17.train_one_seed`; `scripts/run_10seed_flat.py:286`; diagnostics/audit.
- **Critical for**: Euclidean baseline correctness and any claim that spectral beats an isotropic covariance prior.

- **Name**: `spectral_laplacian_penalty`
- **File**: `model.py:326`
- **Purpose**: Computes the Laplacian quadratic `z^T L z` averaged over samples.
- **Inputs**: Latent batch or vector and a `(latent_dim, latent_dim)` Laplacian.
- **Outputs**: Scalar penalty tensor.
- **Key implementation detail**: Works per sample, so training loops can safely call it inside `WorldModel.loss` for each transition and average across the minibatch.
- **Called by**: `WorldModel.loss`.
- **Critical for**: Spectral-prior paper claim.

### 4. Laplacian construction

- **Name**: `build_causal_laplacian`
- **File**: `model.py:338`
- **Purpose**: Converts a causal event graph into a latent-space Laplacian matrix.
- **Inputs**: `networkx.DiGraph`, `latent_dim`.
- **Outputs**: Torch tensor of shape `(latent_dim, latent_dim)`.
- **Key implementation detail**: Symmetrizes the graph, computes the normalized Laplacian, then reconstructs a latent-sized matrix from the largest eigenpairs with zero padding. Returns identity if the graph has fewer than two nodes.
- **Called by**: `train._build_fixed_laplacian`, `train.train_one`, `scripts.train_rmd17.build_molecular_laplacian`, `scripts.train_rmd17.build_graph_source_laplacian`, diagnostics, validation runners.
- **Critical for**: Definition of the spectral prior in both Wolfram and rMD17 experiments.

- **Name**: `build_molecular_laplacian`
- **File**: `scripts/train_rmd17.py:37`
- **Purpose**: Builds a latent-space Laplacian from an rMD17 atom-distance graph.
- **Inputs**: rMD17 `HeteroData` atom graph and `latent_dim`.
- **Outputs**: Torch tensor of shape `(latent_dim, latent_dim)`.
- **Key implementation detail**: Converts bidirectional atom edges into one undirected NetworkX graph and weights each bond-like edge as inverse distance before delegating to `build_causal_laplacian`.
- **Called by**: `build_graph_source_laplacian`, rMD17 fixed-mean Laplacian code, `scripts/run_rmd17_laplacian_ablation.py`.
- **Critical for**: rMD17 bond-graph spectral prior and Laplacian ablation.

- **Name**: `build_graph_source_laplacian`
- **File**: `scripts/train_rmd17.py:72`
- **Purpose**: Selects which graph supplies the rMD17 spectral Laplacian.
- **Inputs**: Raw atom observation, `latent_dim`, `graph_source`, `seed`, `frame_idx`, and device.
- **Outputs**: Device-placed Laplacian tensor.
- **Key implementation detail**: Supports `"bond"`, `"identity"`, `"random"`, and `"complete"`. Random graphs preserve the same number of atoms and undirected bonds, with seed derived from experiment seed and frame index.
- **Called by**: `scripts.train_rmd17.train_one_seed`, prior-attribution runs.
- **Critical for**: Prior-attribution claim that graph structure, not just regularization strength, matters.

- **Name**: `train._build_fixed_laplacian`
- **File**: `train.py:291`
- **Purpose**: Precomputes fixed Wolfram Laplacians for ablations.
- **Inputs**: Wolfram `Config`, train transitions, device.
- **Outputs**: `None` for per-step mode or fixed Laplacian tensor.
- **Key implementation detail**: `"fixed_initial"` uses the first training transition graph; `"fixed_average"` averages Laplacians from the first 50 flattened transitions.
- **Called by**: `train.train_one`.
- **Critical for**: Wolfram fixed-vs-per-step Laplacian ablation.

### 5. Training loops

- **Name**: `train_one_seed`
- **File**: `scripts/train_rmd17.py:385`
- **Purpose**: Runs one complete rMD17 training/evaluation job for one molecule, prior, encoder, and seed.
- **Inputs**: rMD17 `Config`.
- **Outputs**: Dict with `rollout_errors`, `final_loss`, `config`, `wall_time_sec`, optional `metadata`, and optional `checkpoint_path`.
- **Key implementation detail**: rMD17 uses scalar zero action, reward target `0.0`, and done target `0.0`; molecular energy is loaded but not used as the reward target.
- **Called by**: rMD17 runners including 10-seed, disjoint checkpointed, weight sweep, Laplacian ablation, prior attribution, and extended molecules.
- **Critical for**: Main molecular results and reviewer-facing reproducibility.

- **Name**: `train_one`
- **File**: `train.py:326`
- **Purpose**: Trains one Wolfram model configuration over already-collected synthetic transitions.
- **Inputs**: Wolfram `Config`, training transitions, evaluation episodes, device, horizons.
- **Outputs**: `(model, history)`.
- **Key implementation detail**: Spectral prior can be per-step or fixed; Euclidean prior is accumulated across the minibatch outside `WorldModel.loss`.
- **Called by**: `train.run_experiment`, Wolfram validation/weight/fixed-Laplacian runners.
- **Critical for**: Synthetic Wolfram horizon-scaling and prior comparisons.

- **Name**: `scripts/run_10seed_flat.py::train_cell`
- **File**: `scripts/run_10seed_flat.py:286`
- **Purpose**: A self-contained Wolfram training cell used by overnight validation wrappers.
- **Inputs**: Task settings, pilot defaults, prior, seed, train transitions, eval episodes, batch size, epochs.
- **Outputs**: Final rollout metrics by horizon.
- **Key implementation detail**: Duplicates the essential `train_one` logic in-runner, including minibatch Euclidean accumulation and per-transition spectral Laplacians.
- **Called by**: `run_cell_with_oom_retry`, `run_task`, flat/hypergraph/long-horizon wrappers.
- **Critical for**: Overnight validation artifacts if those are used in the paper.

### 6. Data loading

- **Name**: `RMD17Trajectory`
- **File**: `scripts/rmd17_loader.py:83`
- **Purpose**: Lazy dataset wrapper over one molecule's rMD17 `.npz` trajectory.
- **Inputs**: Molecule name, cutoff, optional data root.
- **Outputs**: Dataset object returning per-frame `HeteroData`; `get_pair` returns frame pairs.
- **Key implementation detail**: Loads `nuclear_charges`, `coords`, `energies`, and `forces`; only coordinates and charges are used for graph construction in the main path.
- **Called by**: `collect_rmd17_transitions`, `collect_disjoint_eval_transitions`, rMD17 evaluation, fixed Laplacian setup.
- **Critical for**: Correct molecular data provenance.

- **Name**: `atoms_to_pyg`
- **File**: `scripts/rmd17_loader.py:25`
- **Purpose**: Converts one molecular frame to a PyTorch Geometric heterogeneous atom graph.
- **Inputs**: Coordinates `(N_atoms, 3)`, atomic numbers `(N_atoms,)`, distance cutoff.
- **Outputs**: `HeteroData` with atom features, positions, atomic numbers, and directed atom-to-atom edges with distance attributes.
- **Key implementation detail**: Edges are all atom pairs within cutoff, excluding self edges; this is not a chemical-bond parser.
- **Called by**: `RMD17Trajectory.__getitem__`.
- **Critical for**: What "molecular graph" means in the spectral prior.

- **Name**: `collect_rmd17_transitions`
- **File**: `scripts/rmd17_loader.py:127`
- **Purpose**: Samples rMD17 frame pairs for training or standard evaluation.
- **Inputs**: Molecule, number of transitions, stride, horizon, seed, cutoff.
- **Outputs**: List of transition dicts with `obs`, `next_obs`, `horizon`, `frame_idx`, `energy`, and `molecule`.
- **Key implementation detail**: Candidate start indices are multiples of `stride`; sampled indices are sorted after random selection.
- **Called by**: `scripts.train_rmd17.train_one_seed`, smoke tests, collapse inspection.
- **Critical for**: Sampling reproducibility and train/eval leakage analysis.

- **Name**: `collect_disjoint_eval_transitions`
- **File**: `scripts/train_rmd17.py:216`
- **Purpose**: Samples rMD17 evaluation starts whose used frames avoid training frames.
- **Inputs**: Molecule, `n_transitions`, stride, evaluation horizons, seed, forbidden frame indices, cutoff.
- **Outputs**: List of transition dicts using the maximum evaluation horizon.
- **Key implementation detail**: A candidate is valid only if its start frame and all requested target frames are disjoint from the forbidden set.
- **Called by**: `scripts.train_rmd17.train_one_seed` when `Config.disjoint_eval=True`.
- **Critical for**: Reviewer-facing claims that evaluation frames are strictly disjoint from training frames.

- **Name**: `transition_used_frames`
- **File**: `scripts/train_rmd17.py:204`
- **Purpose**: Reconstructs all start and target frame indices used by a transition set.
- **Inputs**: Transition list and horizons.
- **Outputs**: Sorted list of frame indices.
- **Key implementation detail**: Requires `frame_idx` metadata; silently skips transitions without it.
- **Called by**: `train_one_seed`, `frame_index_metadata`.
- **Critical for**: Disjoint-evaluation auditability.

- **Name**: `frame_index_metadata`
- **File**: `scripts/train_rmd17.py:266`
- **Purpose**: Saves train/eval frame-index overlap metadata.
- **Inputs**: Train transitions, eval transitions, train horizon, eval horizons.
- **Outputs**: Dict containing start frames, used frames, overlap counts, and examples.
- **Key implementation detail**: Records both start-frame overlap and any used-frame overlap.
- **Called by**: `train_one_seed`.
- **Critical for**: Defending data split integrity.

- **Name**: `collect_episodes`
- **File**: `train.py:188`
- **Purpose**: Generates Wolfram-style rewrite episodes.
- **Inputs**: Rewrite rule, initial state, number of episodes, max steps, seed, environment profile.
- **Outputs**: List of episode dicts with observations, actions, and transitions.
- **Key implementation detail**: Each episode uses `seed + episode_index` and a fresh initial state from the selected profile.
- **Called by**: `train.run_experiment`, Wolfram runners.
- **Critical for**: Synthetic data generation and seed separation.

- **Name**: `flatten_transitions`
- **File**: `train.py:248`
- **Purpose**: Flattens Wolfram episode transitions into one training list.
- **Inputs**: Iterable of episode dicts.
- **Outputs**: List of transition dicts.
- **Key implementation detail**: Episode boundaries are discarded; fixed-average Laplacian then averages across the first 50 flattened transitions.
- **Called by**: `train.run_experiment`, Wolfram runners, summaries.
- **Critical for**: Training order and fixed Laplacian behavior.

- **Name**: `adapt_obs_for_world_model`
- **File**: `scripts/train_rmd17.py:97`
- **Purpose**: Converts rMD17 atom graphs into the hypergraph schema expected by `WorldModel`.
- **Inputs**: Atom/bond `HeteroData`.
- **Outputs**: `HeteroData` with `"node"`, `"hyperedge"`, and incidence edge types.
- **Key implementation detail**: Node feature is atomic number only; hyperedges represent undirected atom pairs with arity/origin features, while distances are stored separately as `hyperedge.distance` but not used by current encoders.
- **Called by**: `scripts.train_rmd17.move_obs_to_device`.
- **Critical for**: Whether rMD17 models actually see molecular geometry.

### 7. Evaluation

- **Name**: `train.evaluate_rollout`
- **File**: `train.py:256`
- **Purpose**: Evaluates Wolfram latent rollout error over full episodes.
- **Inputs**: Model, eval episodes, horizons, device.
- **Outputs**: Dict from horizon to mean normalized L2 latent error.
- **Key implementation detail**: Always starts from the first observation in each episode and uses the first `horizon` actions.
- **Called by**: `train.train_one`, `train.run_experiment`, Wolfram runners.
- **Critical for**: Wolfram horizon-scaling plots and tables.

- **Name**: `scripts.train_rmd17.evaluate_rollout`
- **File**: `scripts/train_rmd17.py:323`
- **Purpose**: Evaluates rMD17 latent rollout error over sampled frame starts.
- **Inputs**: Model, eval transitions, horizons, device, latent dim.
- **Outputs**: Dict from horizon to mean normalized L2 latent error.
- **Key implementation detail**: Reloads the trajectory by molecule and compares the rollout to `traj[idx + horizon]`; actions are always zeros.
- **Called by**: `train_one_seed`.
- **Critical for**: Main rMD17 H=1/2/4/8/16 claims.

- **Name**: `quick_rollout_eval`
- **File**: `scripts/train_rmd17.py:357`
- **Purpose**: Computes a small rMD17 rollout metric during training.
- **Inputs**: Model, sample transitions, one horizon, device, latent dim.
- **Outputs**: Mean normalized latent error.
- **Key implementation detail**: Uses at most whatever sample slice the caller passes; `train_one_seed` passes the first 20 eval transitions for H=8 progress logging.
- **Called by**: `train_one_seed`.
- **Critical for**: Progress monitoring only, not final paper numbers.

### 8. Other supporting machinery

- **Name**: `RewriteRule`
- **File**: `hypergraph_env.py:46`
- **Purpose**: Defines variable-based hypergraph rewrite rules and finds valid matches.
- **Inputs**: Left-hand-side pattern, right-hand-side pattern, optional name.
- **Outputs**: Rule object; `find_matches` returns concrete `RewriteMatch` objects.
- **Key implementation detail**: Pattern matching enforces injective edge use but not necessarily unique node bindings beyond variable consistency.
- **Called by**: `HypergraphState.apply_rule`, `CausalWorldEnv.step`, `collect_episodes`.
- **Critical for**: Synthetic Wolfram environment correctness.

- **Name**: `HypergraphState`
- **File**: `hypergraph_env.py:139`
- **Purpose**: Mutable hypergraph plus rewrite-event history.
- **Inputs**: Initial hyperedges and optional seed.
- **Outputs**: State object with conversion to PyG data and causal graph.
- **Key implementation detail**: Tracks `edge_origins` so causal dependencies between rewrite events can be recovered.
- **Called by**: `CausalWorldEnv`, `train.make_initial_state`, demos.
- **Critical for**: Causal-graph Laplacian source in Wolfram runs.

- **Name**: `CausalWorldEnv`
- **File**: `hypergraph_env.py:311`
- **Purpose**: Small RL-style environment around one hypergraph rewrite rule.
- **Inputs**: Rewrite rule, initial state, max steps, seed.
- **Outputs**: `reset`, `step`, and `sample_action` behavior.
- **Key implementation detail**: Reward is negative current hyperedge count; done is only max-step based.
- **Called by**: `collect_episodes`.
- **Critical for**: Synthetic environment data generation.

- **Name**: `save_checkpoint`
- **File**: `scripts/train_rmd17.py:291`
- **Purpose**: Saves rMD17 model state, config, final loss, rollout errors, and git commit.
- **Inputs**: Config, model, final loss, rollout errors.
- **Outputs**: Checkpoint path.
- **Key implementation detail**: Model tensors are moved to CPU before saving.
- **Called by**: `train_one_seed` when `save_checkpoint=True`.
- **Critical for**: Reproducibility and post-hoc checkpoint analysis.

- **Name**: `load_checkpoint_model`
- **File**: `scripts/train_rmd17.py:307`
- **Purpose**: Restores an rMD17 checkpoint into a `WorldModel`.
- **Inputs**: Checkpoint path and optional device.
- **Outputs**: `(Config, WorldModel, checkpoint_dict)`.
- **Key implementation detail**: Reconstructs the model from saved config fields before loading the state dict.
- **Called by**: Available for analysis; no main runner found in current search.
- **Critical for**: Checkpoint reuse and auditability.

## Section 3: External libraries used

Versions are from `requirements.txt`.

### Deep learning

| Library | Version | Purpose |
| --- | --- | --- |
| `torch` | `2.12.0.dev20260407+cu128` | Neural networks, tensors, autograd, Adam optimizer, CUDA, checkpoint serialization, and `torch.utils.data.Dataset`. |
| `torch_geometric` | `2.7.0` | Heterogeneous graph data containers and graph neural network layers. |
| `torchvision` | `0.27.0.dev20260407+cu128` | Installed but not imported by project code found in this pass. |
| `torchaudio` | `2.11.0.dev20260407+cu128` | Installed but not imported by project code found in this pass. |
| `triton` | `3.7.0+git9c288bc5` | Installed as Torch GPU compiler/runtime support; not directly imported. |

PyTorch Geometric classes/functions used:

| Symbol | Where | Purpose |
| --- | --- | --- |
| `HeteroData` | `hypergraph_env.py:18`, `model.py:16`, `train.py:20`, `scripts/rmd17_loader.py:7`, `scripts/train_rmd17.py:21` | Stores heterogeneous graph observations. |
| `HeteroConv` | `model.py:17`, used in `HypergraphEncoder` at `model.py:34` and `model.py:45` | Runs relation-specific message passing. |
| `SAGEConv` | `model.py:17`, used in `HypergraphEncoder` at `model.py:36`, `model.py:39`, `model.py:47`, `model.py:50` | GraphSAGE convolution for node-to-hyperedge and reverse relations. |
| `global_mean_pool` | `model.py:17`, used at `model.py:89`; also diagnostics at `diagnose.py:24` | Pools node and hyperedge embeddings into graph-level vectors. |
| `Batch` | `diagnose.py:23` | Diagnostic batching for hypergraph encoder analysis. |
| `to_networkx` | `scripts/train_rmd17.py:22` | Imported but not used in the inspected code. |

### Graph

| Library | Version | Purpose |
| --- | --- | --- |
| `networkx` | `3.6.1` | Hypergraph causal graph representation, molecular graph conversion, random/complete graph ablations, graph density/DAG metrics, normalized Laplacians. |

### Numerical

| Library | Version | Purpose |
| --- | --- | --- |
| `numpy` | `2.4.3` | Sampling frame indices, array operations, eigen-decomposition, summary statistics, molecular distance matrices. |
| `scipy` | `1.17.1` | Not directly imported, but NetworkX normalized Laplacian routines commonly rely on SciPy sparse machinery at runtime. |

### Data loading

| Library | Version | Purpose |
| --- | --- | --- |
| `torch.utils.data.Dataset` | from `torch` | Base class for `RMD17Trajectory`. |

### Plotting and analysis

| Library | Version | Purpose |
| --- | --- | --- |
| `matplotlib` | `3.10.8` | Result plots, paper figures, diagnostics, and demo causal graph visualization. |

### Other installed packages

These appear in `requirements.txt` but were not direct imports in the project Python files inspected: `aiohappyeyeballs`, `aiohttp`, `aiosignal`, `attrs`, `certifi`, `charset-normalizer`, `contourpy`, `cuda-bindings`, `cuda-pathfinder`, `cuda-toolkit`, `cycler`, `filelock`, `fonttools`, `frozenlist`, `fsspec`, `idna`, `Jinja2`, `kiwisolver`, `MarkupSafe`, `mpmath`, `multidict`, `nvidia-*` CUDA packages, `packaging`, `pillow`, `propcache`, `psutil`, `pyparsing`, `python-dateutil`, `requests`, `setuptools`, `six`, `sympy`, `tqdm`, `typing_extensions`, `urllib3`, `xxhash`, `yarl`.

## Section 4: Configuration knobs

### rMD17 `Config` (`scripts/train_rmd17.py:149`)

| Field | Default | Valid values | Experiments that override it |
| --- | --- | --- | --- |
| `molecule` | `"aspirin"` | One of `scripts.rmd17_loader.MOLECULES`: aspirin, azobenzene, benzene, ethanol, malonaldehyde, naphthalene, paracetamol, salicylic, toluene, uracil | Almost all rMD17 runners; extended runs use benzene/toluene/naphthalene/salicylic; multimolecule uses ethanol/malonaldehyde; attribution uses aspirin/ethanol/malonaldehyde. |
| `encoder` | `"flat"` | `"flat"` or `"hypergraph"` | rMD17 runners set `"flat"` explicitly; no main rMD17 hypergraph runner found. |
| `prior` | `"spectral"` | `"none"`, `"euclidean"`, `"spectral"` | 10-seed, checkpointed, disjoint, multimolecule, weight sweep run all three; ablation/attribution mostly spectral. |
| `graph_source` | `"bond"` | `"bond"`, `"random"`, `"complete"`, `"identity"` when `prior="spectral"` | Prior attribution sets random/complete/identity; extended and main runs use/default bond. |
| `prior_weight` | `0.1` | Nonnegative float; code does not enforce range | Weight sweep uses `0.001`, `0.01`, `0.1`, `1.0`; none configs often set `0.0`; disjoint/extended/attribution use `0.1`. |
| `latent_dim` | `16` | Positive int; must match Laplacian size | No rMD17 runner override found. |
| `hidden_dim` | `32` | Positive int | No rMD17 runner override found. |
| `n_transitions` | `2000` | Positive int no greater than sampled candidates | Smoke mode sets `200`; otherwise default in main runners. |
| `stride` | `10` | Integer `>= 1` | Defaults used; eval sampling internally uses `stride * 10`. |
| `horizon` | `1` | Integer `>= 1` | Defaults used for training horizon. |
| `eval_horizons` | `(1, 2, 4, 8, 16)` | Tuple/list of positive ints | Runners usually pass the same tuple explicitly. |
| `num_epochs` | `50` | Positive int | Smoke sets `3`; all main rMD17 runners use `50`. |
| `batch_size` | `32` | Positive int | No rMD17 runner override found. |
| `lr` | `1e-3` | Positive float | No rMD17 runner override found. |
| `seed` | `0` | Int | All multi-seed runners override. |
| `device` | `"cuda"` | Any valid `torch.device` string; code mainly expects `"cuda"` or `"cpu"` | Runners choose CPU fallback or require CUDA. |
| `transition_hidden_dim` | `128` | Positive int | No rMD17 runner override found. |
| `laplacian_mode` | `"per_frame"` | `"per_frame"`, `"fixed_frame0"`, `"fixed_mean"` | `run_rmd17_laplacian_ablation.py` runs all three. |
| `save_checkpoint` | `False` | Bool | Checkpointed/disjoint/extended/attribution runs set `True`. |
| `checkpoint_dir` | `"checkpoints/rmd17"` | Path string | Checkpointed/disjoint/extended/attribution runners override. |
| `save_frame_indices` | `True` | Bool | Checkpointed/disjoint/extended/attribution keep true; can be disabled by custom configs. |
| `disjoint_eval` | `False` | Bool | Disjoint checkpointed, multimolecule disjoint, prior attribution, and extended molecules set `True`. |
| `eval_n_transitions` | `200` | Positive int | No main runner override found. |

### Wolfram `Config` (`train.py:31`)

| Field | Default | Valid values | Experiments that override it |
| --- | --- | --- | --- |
| `encoder` | Required | `"flat"` or `"hypergraph"` | `run_experiment` runs both; flat/hypergraph validation runners set one encoder. |
| `prior` | Required | `"none"`, `"euclidean"`, `"spectral"` | `run_experiment` and validation runners run all three; weight/fixed-laplacian runners focus on spectral. |
| `prior_weight` | Required | Nonnegative float | Pilot defaults use `none=0.0`, `euclidean=1e-2`, `spectral=1e-2`; weight sweep uses `0.001`, `0.005`, `0.01`, `0.05`; none uses `0.0`. |
| `latent_dim` | `16` | Positive int | 200-epoch validation runners load/use 16; no alternative found in main runners. |
| `hidden_dim` | `32` | Positive int | Flat/hypergraph runners use 32. |
| `transition_hidden_dim` | `128` | Positive int | Defaults used. |
| `num_epochs` | `200` | Positive int | Smoke uses `1` or `10`; pilot/main Wolfram uses `200`. |
| `batch_size` | `32` | Positive int | CLI can override in `train.py`; OOM retry runners may halve it. |
| `lr` | `1e-3` | Positive float | Defaults used. |
| `seed` | `0` | Int | All multi-seed runners override. |
| `laplacian_mode` | `"per_step"` | `"per_step"`, `"fixed_initial"`, `"fixed_average"` | `run_wolfram_fixed_laplacian.py` runs all three; normal runs use per-step. |

Other Wolfram knobs not in `Config` but central to experiments:

| Knob | Default/source | Valid values | Experiments that override it |
| --- | --- | --- | --- |
| `HORIZONS` | `[1, 2, 4, 8, 16]` at `train.py:27` | Positive ints | `run_long_horizon.py` uses `[1, 2, 4, 8, 16, 32, 64]`. |
| `env_profile` | `"minimal"` in `run_experiment` CLI default | `"minimal"` or `"branching"` | CLI and pilot-derived runners may use stored profile. |
| `max_steps` | `16` from `build_environment` | Positive int | Long-horizon wrapper uses `64`. |
| `n_train` | `200` in normal `run_experiment` | Positive int | CLI can override; smoke uses `12`. |
| `n_eval` | `50` in normal `run_experiment` | Positive int | CLI can override; smoke uses `6`. |

## Section 5: File map

```text
.
|-- model.py
|   Core encoders, transition network, priors, Laplacian construction, and WorldModel.
|-- train.py
|   Wolfram-style synthetic training/evaluation pipeline.
|-- hypergraph_env.py
|   Rewrite-rule environment, hypergraph state, causal event graph, and PyG conversion.
|-- diagnose.py
|   Diagnostic experiments for collapse, weight sensitivity, seed stability, and encoder capacity.
|-- scripts/
|   |-- rmd17_loader.py
|   |   rMD17 .npz loader and atom-frame-to-HeteroData conversion.
|   |-- train_rmd17.py
|   |   rMD17 training loop, molecular Laplacians, disjoint eval, checkpoint save/load.
|   |-- run_rmd17_10seed.py
|   |   Basic aspirin 10-seed rMD17 runner.
|   |-- run_rmd17_10seed_checkpointed.py
|   |   Checkpointed aspirin 10-seed rMD17 runner.
|   |-- run_rmd17_aspirin_disjoint_checkpointed.py
|   |   Strict disjoint aspirin rMD17 runner with analysis refresh.
|   |-- run_rmd17_multimolecule_disjoint_checkpointed.py
|   |   Strict disjoint ethanol/malonaldehyde rMD17 runner.
|   |-- run_rmd17_extended_molecules.py
|   |   Extended molecule rMD17 runner for benzene, toluene, naphthalene, salicylic.
|   |-- run_rmd17_prior_attribution.py
|   |   rMD17 graph-source attribution runner: random, complete, identity.
|   |-- run_rmd17_weight_sweep.py
|   |   rMD17 prior-weight sweep.
|   |-- run_rmd17_laplacian_ablation.py
|   |   rMD17 per-frame vs fixed Laplacian ablation.
|   |-- run_rmd17_euclidean_redo.py
|   |   Recomputes stale Euclidean rMD17 results after the batch-covariance fix.
|   |-- resume_rmd17_aspirin_spectral.py
|   |   Resume helper for missing aspirin spectral seeds.
|   |-- run_wolfram_flat_10seed_200ep.py
|   |   Flat Wolfram 10-seed 200-epoch validation runner.
|   |-- run_wolfram_resume.py
|   |   Resume helper for Wolfram flat validation.
|   |-- run_wolfram_weight_sweep.py
|   |   Wolfram spectral weight sweep.
|   |-- run_wolfram_fixed_laplacian.py
|   |   Wolfram per-step/fixed Laplacian ablation.
|   |-- run_10seed_flat.py
|   |   Overnight validation harness for flat Wolfram runs.
|   |-- run_10seed_hypergraph.py
|   |   Wrapper around the overnight harness for hypergraph encoder.
|   |-- run_long_horizon.py
|   |   Wrapper around the overnight harness for H up to 64.
|   |-- aggregate_results.py
|   |   Aggregates manifest/results CSVs and writes summary tables.
|   |-- analyze_rmd17_disjoint_checkpointed.py
|   |   Aggregates strict-disjoint rMD17 checkpointed outputs.
|   |-- analyze_wolfram_instability.py
|   |   Analyzes Wolfram horizon/instability artifacts.
|   |-- analyze_weight_instability.py
|   |   Analyzes prior-weight instability artifacts.
|   |-- audit_priors_and_laplacians.py
|   |   Code audit for prior and Laplacian implementation details.
|   |-- build_manifest.py
|   |   Builds run/result manifest tables.
|   |-- plot_results.py
|   |   Produces aggregate plots.
|   `-- make_applied_intelligence_figures.py
|       Produces manuscript figures from aggregate artifacts.
|-- paper/
|   |-- applied_intelligence/
|   |   Manuscript, tables, figures, references, and submission package.
|   `-- findings/
|       Reviewer-facing findings notes, including this inventory.
|-- analysis_out/
|   Aggregated CSVs, audits, reports, and plot outputs.
|-- final_results_snapshot/
|   Snapshot of final aggregate artifacts.
|-- requirements.txt
|   CUDA/Torch environment with pinned package versions.
`-- requirements-60ti.txt
    Alternate/light requirements file without explicit Torch packages.
```

## Section 6: Known design decisions and why

- **Chosen**: Spectral penalty is computed per sample, then averaged by training loops.
  **Alternatives**: Build a batch-level graph prior or one Laplacian over all batch samples.
  **Why**: The implemented penalty `z^T L z` is well-defined for one latent and one graph Laplacian, so per-transition Laplacians can differ across samples. This is directly supported by `spectral_laplacian_penalty` and the training loops.

- **Chosen**: Euclidean covariance penalty is accumulated across the minibatch outside `WorldModel.loss`.
  **Alternatives**: Call `WorldModel.loss(... prior="euclidean")` per transition, or batch observations through the encoder.
  **Why**: `euclidean_cov_penalty` returns zero for batch size less than 2, so per-sample use would silently disable the prior. The rMD17 code documents this explicitly at `scripts/train_rmd17.py:524`.

- **Chosen**: rMD17 standard evaluation uses `stride * 10`; strict disjoint evaluation also uses `stride * 10`.
  **Alternatives**: Same stride as training, contiguous held-out block, or explicit train/eval frame split.
  **Why**: [INFERRED] The loader comment says stride reduces temporal autocorrelation; using a coarser eval stride likely further reduces correlation and eval cost. Strict disjoint mode adds explicit forbidden-frame filtering.

- **Chosen**: rMD17 disjoint eval tracks `frame_idx` through transition dicts and metadata.
  **Alternatives**: Only rely on different random seeds, or split by array ranges.
  **Why**: The audit found overlap under the older non-disjoint strategy, so preserving `frame_idx` enables concrete overlap checks and reviewer-facing evidence.

- **Chosen**: rMD17 atom graphs are adapted into the existing hypergraph schema rather than adding a molecular encoder.
  **Alternatives**: Implement a separate atom/bond GNN, use coordinates directly, or use a molecular graph library.
  **Why**: [INFERRED] This reuses `WorldModel`, `HypergraphEncoder`, `FlatMLPEncoder`, and the same prior machinery across Wolfram and rMD17.

- **Chosen**: rMD17 action, reward, and done are all effectively constants during training/eval.
  **Alternatives**: Predict energy/force/coordinate changes, include timestep or molecular dynamics action, or remove reward/done heads for rMD17.
  **Why**: [INFERRED] The model is being used as an autonomous latent dynamics predictor over time-indexed trajectory frames, not as a controlled dynamics model.

- **Chosen**: `latent_dim=16` in both main configs.
  **Alternatives**: Tune latent dimension or report a sweep.
  **Why**: [INFERRED] No code comment or runner rationale was found. The value is consistent across Wolfram and rMD17, which makes comparisons simpler.

- **Chosen**: rMD17 trains for 50 epochs while Wolfram default runs train for 200 epochs.
  **Alternatives**: Equalize epochs, equalize optimization steps, or tune per dataset.
  **Why**: [INFERRED] rMD17 has many more sampled transitions per run (`2000`) and heavier graph construction; Wolfram uses fewer episodes/transitions and 200 epochs in validation runners. No explicit rationale was found.

- **Chosen**: Laplacian fixed modes average early flattened transitions/frames.
  **Alternatives**: Average a random subset, all frames, or validation-independent graph templates.
  **Why**: [INFERRED] This gives a stable ablation reference while keeping computation bounded. Wolfram uses first 50 transitions; rMD17 fixed mean uses up to first 500 frames.

- **Chosen**: Rollout error is normalized L2 distance in latent space.
  **Alternatives**: Coordinate RMSE, energy/force error, decoded state error, or graph edit distance.
  **Why**: [INFERRED] The model has no decoder, so latent agreement with the encoder's true future state is the available rollout metric.

## Section 7: Things Codex is NOT sure about

- [HIGH] **rMD17 encoder input may discard the key physical signal.** `atoms_to_pyg` stores coordinates and distances, but `adapt_obs_for_world_model` feeds atomic number as node feature and arity/origin as hyperedge feature. Distances are saved as `hyperedge.distance` but current encoders do not read it. For a fixed molecule, many flat features may be nearly constant except cutoff-edge changes.

- [HIGH] **rMD17 rewards and done labels are constant zeros.** `train_one_seed` loads molecular energy into transitions but passes `reward=0.0` and `done=0.0` into `WorldModel.loss`. This may be intentional for latent-only dynamics, but the reward head then learns a dummy target.

- [HIGH] **Standard rMD17 eval is not guaranteed disjoint.** Non-disjoint mode uses a different seed and coarser stride but does not exclude training frame indices. The existing audit reports overlap in reconstructed metadata. Reviewer-facing claims should prefer runs with `disjoint_eval=True`.

- [HIGH] **`WorldModel.loss(prior="euclidean")` can silently do nothing for one sample.** The main loops avoid this, but direct callers may think they are applying a Euclidean prior when batch size is one.

- [MED] **`build_causal_laplacian` uses largest eigenpairs.** The doc says "top-k eigenvectors"; the code takes eigenvectors from the largest eigenvalues. In spectral graph methods, "smooth" modes often mean smallest nontrivial eigenvalues. I am not sure whether largest modes are intended here.

- [MED] **`build_causal_laplacian` reuses graph-node eigenvectors as latent dimensions by position.** If graph nodes exceed `latent_dim`, only the first `latent_dim` components of each eigenvector are copied. This is a projection choice, but I did not find a justification.

- [MED] **`CausalGraphEncoder` appears unused.** It may be a leftover from an earlier design or intended diagnostic component, but it is not selectable in `WorldModel`.

- [MED] **`collect_disjoint_eval_transitions` stores `next_obs` at max horizon only.** Final evaluation reloads the trajectory for every horizon and does not use this stored `next_obs`, so the field name can mislead readers.

- [MED] **rMD17 graph edges are cutoff-neighbor edges, not chemical bonds.** Function names use "bond" and `"bonded"`, but `atoms_to_pyg` creates edges for all atom pairs within a distance cutoff. This is valid if described as a distance graph, risky if described as chemical bonds.

- [LOW] **`scripts/train_rmd17.py` has unused imports.** `torch.nn.functional as F`, `to_networkx`, and `matplotlib.pyplot` are imported but not used in the current file. This is harmless but suggests some old code paths were removed.

- [LOW] **Some validation runners duplicate training logic.** `scripts/run_10seed_flat.py::train_cell` repeats the core train loop instead of calling `train.train_one`. It appears intentionally self-contained for overnight runs, but duplicated logic can drift.

- [LOW] **`requirements-60ti.txt` omits Torch packages.** The main `requirements.txt` pins Torch/Torch Geometric/CUDA packages, while the alternate file does not. I am not sure which environment file is canonical for reproduction.
