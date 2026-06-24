from __future__ import annotations

import torch
import torch.nn.functional as F

from .realizable_basis import EPS, suffix_batch_fn_for_position


def weighted_center(values: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    return values.float() - (probs.float() * values.float()).sum(dim=-1, keepdim=True)


def score_realizability_decomposition(
    logits0: torch.Tensor,
    jv_logits: torch.Tensor,
    readout: torch.Tensor,
    *,
    eps: float = EPS,
) -> dict[str, torch.Tensor]:
    """Compute C, B, A, R from base logits, logit velocity, and token readout."""
    logits0 = logits0.float().reshape(-1)
    readout = readout.to(logits0.device).float().reshape(-1)
    jv_logits = jv_logits.to(logits0.device).float()
    if jv_logits.ndim == 1:
        jv_logits = jv_logits.unsqueeze(0)
    if readout.numel() != logits0.numel() or jv_logits.shape[-1] != logits0.numel():
        raise ValueError("logits0, jv_logits, and readout must share the same vocab dimension.")

    probs = F.softmax(logits0, dim=-1)
    centered_readout = weighted_center(readout, probs)
    centered_jv = weighted_center(jv_logits, probs)

    capacity = (probs.unsqueeze(0) * centered_jv.square()).sum(dim=-1).clamp_min(0.0).sqrt()
    availability = (probs * centered_readout.square()).sum().clamp_min(0.0).sqrt()
    realizability = (probs.unsqueeze(0) * centered_readout.unsqueeze(0) * centered_jv).sum(dim=-1)
    denom = (capacity * availability).clamp_min(float(eps))
    alignment = torch.where(
        (capacity > float(eps)) & (availability > float(eps)),
        realizability / denom,
        torch.zeros_like(realizability),
    )
    return {
        "capacity_C": capacity,
        "availability_B": availability.expand_as(capacity),
        "alignment_A": alignment,
        "realizability_R": realizability,
    }


def jvp_logits_for_direction(owner, record: dict, direction: torch.Tensor) -> torch.Tensor:
    """Compute J_h v logits for one cached prediction-site record."""
    suffix = suffix_batch_fn_for_position(owner, record["input_ids"], record["attention_mask"], int(record["position"]))
    h0 = record["h0"].to(owner.device).float().reshape(1, -1)
    v = direction.to(owner.device).float().reshape_as(h0)
    _, jv = torch.autograd.functional.jvp(suffix, (h0,), (v,), create_graph=False, strict=False)
    return jv.detach().reshape(-1).cpu().float()


__all__ = [
    "jvp_logits_for_direction",
    "score_realizability_decomposition",
    "weighted_center",
]
