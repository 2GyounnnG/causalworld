from __future__ import annotations

from typing import Dict, Optional

import torch


def effective_rank(z_batch: torch.Tensor) -> float:
    if z_batch.dim() == 1:
        z_batch = z_batch.unsqueeze(0)
    B, D = z_batch.shape
    if B < 2:
        return float(D)
    z_batch = z_batch.detach().to(torch.float32)
    z_c = z_batch - z_batch.mean(dim=0, keepdim=True)
    C = (z_c.T @ z_c) / max(B - 1, 1)
    eigvals = torch.linalg.eigvalsh(C.cpu()).clamp_min(1e-12)
    p = eigvals / eigvals.sum()
    return float(torch.exp(-(p * torch.log(p)).sum()).item())


def covariance_condition_number(z_batch: torch.Tensor) -> float:
    if z_batch.dim() == 1:
        z_batch = z_batch.unsqueeze(0)
    B, _ = z_batch.shape
    if B < 2:
        return 1.0
    z_batch = z_batch.detach().to(torch.float32)
    z_c = z_batch - z_batch.mean(dim=0, keepdim=True)
    C = (z_c.T @ z_c) / max(B - 1, 1)
    eigvals = torch.linalg.eigvalsh(C.cpu()).clamp_min(1e-12)
    return float((eigvals.max() / eigvals.min()).item())


def projection_gaussianity(
    z_batch: torch.Tensor,
    num_slices: int = 8,
    seed: int = 0,
) -> float:
    if z_batch.dim() == 1:
        z_batch = z_batch.unsqueeze(0)
    B, D = z_batch.shape
    if B < 2 or num_slices <= 0:
        return 0.0
    z_batch = z_batch.detach().to(torch.float32)
    z_c = z_batch - z_batch.mean(dim=0, keepdim=True)
    z_std = z_c / (z_c.std(dim=0, unbiased=False, keepdim=True) + 1e-8)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    directions = torch.randn(
        num_slices,
        D,
        generator=generator,
        dtype=z_std.dtype,
    )
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
    directions = directions.to(device=z_std.device)

    statistics = []
    i = torch.arange(1, B + 1, device=z_std.device, dtype=z_std.dtype)
    weights = 2 * i - 1
    for direction in directions:
        s = z_std @ direction
        s = (s - s.mean()) / (s.std(unbiased=False) + 1e-8)
        u = torch.special.ndtr(torch.sort(s).values).clamp(1e-8, 1.0 - 1e-8)
        A2 = -B - (weights * (torch.log(u) + torch.log1p(-torch.flip(u, dims=[0])))).sum() / B
        statistics.append(A2)
    return float(torch.stack(statistics).mean().item())


def spectral_alignment(z_batch: torch.Tensor, L: torch.Tensor) -> float:
    if L is None:
        return float("nan")
    if torch.all(L.detach() == 0):
        return float("nan")
    if z_batch.dim() == 1:
        z_batch = z_batch.unsqueeze(0)
    B, _ = z_batch.shape
    z_batch = z_batch.detach().to(torch.float32)
    L = L.detach().to(torch.float32)
    z_c = z_batch - z_batch.mean(dim=0, keepdim=True)
    Sigma_z = (z_c.T @ z_c) / max(B - 1, 1)
    _, U_L = torch.linalg.eigh(L.cpu())
    M = U_L.T @ Sigma_z.cpu() @ U_L
    denominator = M.abs().sum()
    if float(denominator.item()) == 0.0:
        return float("nan")
    return float((torch.diag(M).abs().sum() / denominator).item())


def graph_stationarity(
    L_t: torch.Tensor,
    L_tp1: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    if torch.all(L_t.detach() == 0) and torch.all(L_tp1.detach() == 0):
        return 0.0
    numerator = torch.linalg.norm((L_tp1 - L_t).detach().to(torch.float32), ord="fro")
    denominator = torch.linalg.norm(L_t.detach().to(torch.float32), ord="fro") + eps
    return float((numerator / denominator).item())


def compute_all_diagnostics(
    z_batch: torch.Tensor,
    L: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    return {
        "effective_rank": effective_rank(z_batch),
        "condition_number": covariance_condition_number(z_batch),
        "projection_gaussianity": projection_gaussianity(z_batch),
        "spectral_alignment": spectral_alignment(z_batch, L),
    }
