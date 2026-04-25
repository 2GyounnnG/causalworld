# save as: scripts/inspect_collapse.py
# Run after rMD17 training stabilizes
import torch, json, numpy as np
from scripts.rmd17_loader import RMD17Trajectory, collect_rmd17_transitions
from scripts.train_rmd17 import move_obs_to_device, Config
from model import WorldModel

device = torch.device("cuda")
# retrain a quick model to reproduce the state
# ... (or load from a saved checkpoint if available)

# Check 1: latent collapse
traj = RMD17Trajectory("malonaldehyde")
obs_samples = [move_obs_to_device(traj[i], device) for i in [0, 1000, 5000, 20000, 50000]]
model = WorldModel(encoder="flat", hidden_dim=32, latent_dim=16, action_dim=1)
model.load_state_dict(torch.load("???"))  # if saved
model.to(device).eval()
with torch.no_grad():
    zs = torch.stack([model.encode(o) for o in obs_samples])
print("latent std across diverse frames:", zs.std(dim=0).mean().item())
# if this is < 0.01, latent has collapsed

# Check 2: overlap between train and eval indices
train_trans = collect_rmd17_transitions("malonaldehyde", 2000, stride=10, horizon=1, seed=0)
eval_trans = collect_rmd17_transitions("malonaldehyde", 200, stride=100, horizon=16, seed=1000)
train_idx = set(t["frame_idx"] for t in train_trans)
eval_idx = set(t["frame_idx"] for t in eval_trans)
overlap = train_idx & eval_idx
print(f"train indices: {len(train_idx)}, eval indices: {len(eval_idx)}, overlap: {len(overlap)}")
print(f"train range: {min(train_idx)}-{max(train_idx)}")
print(f"eval range: {min(eval_idx)}-{max(eval_idx)}")