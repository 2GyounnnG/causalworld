# Overnight Validation Summary

- Start time: 2026-04-20T15:40:37+08:00
- End time: 2026-04-20T15:40:37+08:00
- Total wall time: 0.0 sec
- GPU: NVIDIA GeForce RTX 5060 Ti requested, but CUDA was not visible to PyTorch in this shell

## Task A results table (with std bands)

No artifact was produced.

## Task A+ results table (with std bands)

No artifact was produced.

## Task B results table (with std bands)

No artifact was produced.

## Interpretation

Flat encoder (A): no Task A artifact was available because the bootstrap CUDA gate failed before smoke testing.

Hypergraph encoder (A+): no Task A+ artifact was available because the bootstrap CUDA gate failed before smoke testing. The cross-encoder spectral-advantage claim cannot be assessed from this run.

Long horizon (B): no Task B artifact was available because the bootstrap CUDA gate failed before smoke testing.

## Decisions Made Without Approval

- Followed `codex_bootstrap.txt` and ran the required conda/CUDA verification before smoke or training.
- Did not install or reinstall dependencies.
- Did not run smoke testing, real training, or git commit because the bootstrap requires CUDA availability before proceeding.

## Bootstrap Output

```
/home/user/miniforge3/envs/causalworld/bin/python
/home/user/miniforge3/envs/causalworld/lib/python3.11/site-packages/torch/cuda/__init__.py:1061: UserWarning: Can't initialize NVML
  raw_cnt = _raw_device_count_nvml()
torch 2.11.0+cu130
cuda_available False
torch_cuda 13.0
device_count 0
device_name no gpu
```

## Errors

```
Bootstrap failed because torch.cuda.is_available() returned False and device_name was "no gpu". The required condition was cuda_available True with a device name containing NVIDIA GeForce RTX 5060 Ti.
```
