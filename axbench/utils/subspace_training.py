from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch


EPS = 1e-8


def orthonormal_columns(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    x = x.detach().float()
    if x.ndim == 1:
        x = x[:, None]
    if x.numel() == 0:
        return x.reshape(x.shape[0], 0)
    q, r = torch.linalg.qr(x, mode="reduced")
    diag = torch.diagonal(r).abs() if r.ndim == 2 else torch.empty(0)
    keep = diag > float(eps)
    return q[:, keep]


@dataclass(frozen=True)
class SubspaceProjector:
    """Orthogonal projector P = Q Q^T for hidden-state directions."""

    basis: torch.Tensor

    @classmethod
    def from_basis(cls, basis: torch.Tensor, *, orthonormalize: bool = True) -> "SubspaceProjector":
        q = orthonormal_columns(basis) if orthonormalize else basis.detach().float()
        if q.ndim != 2:
            raise ValueError(f"basis must be rank-2, got shape={tuple(q.shape)}")
        return cls(q.cpu().float())

    @property
    def dim(self) -> int:
        return int(self.basis.shape[0])

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    def basis_on(self, x: torch.Tensor) -> torch.Tensor:
        return self.basis.to(device=x.device, dtype=x.dtype)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank == 0:
            return torch.zeros_like(x)
        if int(x.shape[-1]) != self.dim:
            raise ValueError(f"last dimension {int(x.shape[-1])} does not match projector dim {self.dim}")
        q = self.basis_on(x)
        flat = x.reshape(-1, self.dim)
        projected = (flat @ q) @ q.T
        return projected.reshape_as(x)

    def project_(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x.copy_(self.project(x.detach()))
        return x

    def project_grad_(self, param: torch.Tensor) -> None:
        if param.grad is not None:
            param.grad.copy_(self.project(param.grad.detach()))

    def project_parameters_(self, params: Iterable[torch.Tensor]) -> None:
        for param in params:
            self.project_(param)

    def project_gradients_(self, params: Iterable[torch.Tensor]) -> None:
        for param in params:
            self.project_grad_(param)


class ProjectedOptimizer:
    """Optimizer adapter for projected-gradient direction training."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        params: Iterable[torch.Tensor],
        projector: SubspaceProjector,
        *,
        project_gradients: bool = False,
    ):
        self.optimizer = optimizer
        self.params = list(params)
        self.projector = projector
        self.project_gradients = bool(project_gradients)
        self.projector.project_parameters_(self.params)

    def zero_grad(self, *args, **kwargs):
        return self.optimizer.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        if self.project_gradients:
            self.projector.project_gradients_(self.params)
        out = self.optimizer.step(*args, **kwargs)
        self.projector.project_parameters_(self.params)
        return out

    def __getattr__(self, name: str):
        return getattr(self.optimizer, name)
