from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .subspace_training import ProjectedOptimizer, SubspaceProjector


def unit_norm(x: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(x.dtype).eps if x.is_floating_point() else 1e-8
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)


def remove_parallel_grad_(weight: torch.Tensor) -> None:
    if weight.grad is None:
        return
    w = unit_norm(weight.detach())
    parallel_component = (weight.grad * w).sum(dim=-1, keepdim=True)
    weight.grad.sub_(parallel_component * w)


def normalize_direction_parameter_(w: torch.Tensor, projector: SubspaceProjector | None = None) -> None:
    with torch.no_grad():
        value = w.detach().float()
        if projector is not None:
            value = projector.project(value)
        w.copy_(unit_norm(value).to(w.device))


class LinearProbeNoBias(torch.nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(d_model) * 0.01)

    def forward(self, x):
        return x.float() @ self.weight.float()


def train_probe_direction(
    X: torch.Tensor,
    y: torch.Tensor,
    cfg: dict,
    projector: SubspaceProjector | None = None,
) -> torch.Tensor:
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    X, y = X.to(device), y.to(device)
    probe = LinearProbeNoBias(X.shape[-1]).to(device)
    if projector is not None:
        normalize_direction_parameter_(probe.weight, projector)
    base_opt = torch.optim.AdamW(
        probe.parameters(),
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 0.0),
    )
    opt = (
        ProjectedOptimizer(
            base_opt,
            [probe.weight],
            projector,
            project_gradients=bool(cfg.get("project_gradients", False)),
        )
        if projector is not None
        else base_opt
    )
    bs = cfg.get("batch_size", 4096)
    gen_device = device if str(device).startswith("cuda") else "cpu"
    g = torch.Generator(device=gen_device).manual_seed(cfg.get("seed", 0))
    for _ in tqdm(range(cfg.get("epochs", 50)), desc="probe epochs"):
        perm = torch.randperm(len(X), generator=g, device=gen_device).to(device)
        for start in range(0, len(X), bs):
            idx = perm[start : start + bs]
            loss = F.binary_cross_entropy_with_logits(probe(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            remove_parallel_grad_(probe.weight)
            opt.step()
            normalize_direction_parameter_(probe.weight, projector)
    out = probe.weight.detach().cpu().float()
    if projector is not None:
        out = projector.project(out)
    return unit_norm(out).squeeze(0)
