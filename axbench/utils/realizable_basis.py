from __future__ import annotations

import random
from collections import Counter
from typing import Any, Sequence

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .subspace_training import SubspaceProjector, orthonormal_columns


EPS = 1e-8


def _training_arg(owner: Any, name: str, default: Any = None) -> Any:
    training_args = getattr(owner, "training_args", None)
    return getattr(training_args, name, default) if training_args is not None else default


def _cfg(owner: Any, kwargs: dict, name: str, default: Any = None) -> Any:
    return kwargs.get(name, _training_arg(owner, name, default))


def _text_column(rows: pd.DataFrame) -> str:
    for name in ("completion", "output"):
        if name in rows.columns:
            return name
    raise ValueError("Realizable basis readout requires a completion/output column.")


def _calibration_text_column(rows: pd.DataFrame) -> str:
    for name in ("text",):
        if name in rows.columns:
            return name
    raise ValueError("Realizable basis calibration requires a text column.")


def validate_realizable_basis_examples(rows: pd.DataFrame) -> None:
    required_any = {
        "completion/output": {"completion", "output"},
        "label/labels": {"label", "labels"},
    }
    missing = []
    for name, options in required_any.items():
        if not options.intersection(rows.columns):
            missing.append(name)
    required = {"prefix", "text"}
    missing.extend(sorted(required - set(rows.columns)))
    if missing:
        raise ValueError(
            "Internal realizable basis construction requires Or-style rows with "
            "prefix, completion/output, text, and label/labels columns. "
            f"Missing: {missing}. Pass a projector/realizable_basis/realizable_basis_path "
            "or build a basis dataframe before calling this path."
        )


def _tokenizer_length_kwargs(max_length: int | None) -> dict:
    if max_length is None:
        return {"truncation": False}
    return {"truncation": True, "max_length": max_length}


def prefix_lengths(tokenizer, prefixes: Sequence[str], max_length: int | None) -> list[int]:
    lens = []
    for prefix in prefixes:
        ids = tokenizer(prefix, add_special_tokens=True, **_tokenizer_length_kwargs(max_length))["input_ids"]
        lens.append(len(ids))
    return lens


def prefix_last_positions_from_offsets(
    tokenizer,
    texts: Sequence[str],
    prefixes: Sequence[str],
    max_length: int | None,
) -> list[int]:
    try:
        enc = tokenizer(
            list(texts),
            return_offsets_mapping=True,
            padding=True,
            add_special_tokens=True,
            **_tokenizer_length_kwargs(max_length),
        )
    except (NotImplementedError, TypeError):
        return [max(0, x - 1) for x in prefix_lengths(tokenizer, prefixes, max_length)]

    out: list[int] = []
    for batch_idx, prefix in enumerate(prefixes):
        prefix_chars = len(str(prefix))
        offsets = enc["offset_mapping"][batch_idx]
        attention = enc["attention_mask"][batch_idx]
        candidates = []
        for token_idx, (offset, keep) in enumerate(zip(offsets, attention)):
            if not bool(keep):
                continue
            start, end = int(offset[0]), int(offset[1])
            if end > start and end <= prefix_chars:
                candidates.append(token_idx)
        if candidates:
            out.append(candidates[-1])
        else:
            out.append(max(0, prefix_lengths(tokenizer, [prefix], max_length)[0] - 1))
    return out


def calibration_start_positions(
    tokenizer,
    rows: pd.DataFrame,
    text_col: str,
    max_length: int | None,
) -> list[int]:
    texts = rows[text_col].astype(str).tolist()
    prefixes = rows["prefix"].astype(str).tolist()
    return prefix_last_positions_from_offsets(tokenizer, texts, prefixes, max_length)


def valid_token_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer, exclude_bos: bool = True) -> torch.Tensor:
    mask = attention_mask.bool().clone()
    bad_ids = {getattr(tokenizer, "pad_token_id", None), getattr(tokenizer, "eos_token_id", None)}
    if exclude_bos:
        bad_ids.add(getattr(tokenizer, "bos_token_id", None))
    for token_id in bad_ids:
        if token_id is not None:
            mask &= input_ids.ne(token_id)
    return mask


def prediction_position_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer, exclude_bos: bool = True) -> torch.Tensor:
    cur = valid_token_mask(input_ids, attention_mask, tokenizer, exclude_bos=exclude_bos)
    nxt = torch.zeros_like(cur)
    nxt[:, :-1] = valid_token_mask(input_ids[:, 1:], attention_mask[:, 1:], tokenizer, exclude_bos=True)
    return cur & nxt


def completion_prediction_mask(
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    texts: Sequence[str],
    prefixes: Sequence[str],
    max_length: int | None,
) -> torch.Tensor:
    base = prediction_position_mask(input_ids, attention_mask, tokenizer, exclude_bos=True)
    prefix_last = prefix_last_positions_from_offsets(tokenizer, list(texts), list(prefixes), max_length)
    mask = torch.zeros_like(base, dtype=torch.bool)
    for batch_idx, last_pos in enumerate(prefix_last):
        seq_len = int(attention_mask[batch_idx].sum().item())
        start = max(0, min(int(last_pos), seq_len - 1))
        mask[batch_idx, start : max(start, seq_len - 1)] = True
    return mask & base


def token_ids_for_text(tokenizer, text: str, *, remove_special_tokens: bool = True) -> list[int]:
    ids = tokenizer(str(text), add_special_tokens=False)["input_ids"]
    special = set(getattr(tokenizer, "all_special_ids", []) or []) if remove_special_tokens else set()
    return [int(token_id) for token_id in ids if int(token_id) not in special]


def build_log_count_readout(tokenizer, rows: pd.DataFrame, vocab_size: int, cfg: dict) -> torch.Tensor:
    smoothing = float(cfg.get("readout_smoothing", 0.1))
    clip = float(cfg.get("readout_clip", 8.0))
    text_col = _text_column(rows)
    labels = rows["labels"] if "labels" in rows.columns else rows["label"]
    pos_rows = rows[labels.astype(int).eq(1)]
    neg_rows = rows[~labels.astype(int).eq(1)]
    pos_counts = torch.zeros(int(vocab_size), dtype=torch.float32)
    neg_counts = torch.zeros(int(vocab_size), dtype=torch.float32)
    for text in pos_rows[text_col].astype(str).tolist():
        for token_id in token_ids_for_text(tokenizer, text, remove_special_tokens=True):
            if 0 <= token_id < int(vocab_size):
                pos_counts[token_id] += 1.0
    for text in neg_rows[text_col].astype(str).tolist():
        for token_id in token_ids_for_text(tokenizer, text, remove_special_tokens=True):
            if 0 <= token_id < int(vocab_size):
                neg_counts[token_id] += 1.0
    q_pos = (pos_counts + smoothing) / (pos_counts.sum() + smoothing * vocab_size).clamp_min(EPS)
    q_neg = (neg_counts + smoothing) / (neg_counts.sum() + smoothing * vocab_size).clamp_min(EPS)
    readout = (q_pos.log() - q_neg.log()).clamp(-clip, clip)
    return (readout - readout.mean()).float().cpu()


def fisher_apply(p: torch.Tensor, x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    p = p.float().clamp_min(float(eps))
    x = x.float()
    return p * (x - (p * x).sum())


def replacement_logits(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer: int,
    positions: torch.Tensor,
    hs: torch.Tensor,
) -> torch.Tensor:
    def replace_hook(module, inputs, outputs):
        del module, inputs
        if isinstance(outputs, tuple):
            hidden = outputs[0].clone()
            hidden[torch.arange(hidden.shape[0], device=hidden.device), positions] = hs.to(hidden.dtype)
            return (hidden,) + outputs[1:]
        hidden = outputs.clone()
        hidden[torch.arange(hidden.shape[0], device=hidden.device), positions] = hs.to(hidden.dtype)
        return hidden

    handle = model.model.layers[int(layer)].register_forward_hook(replace_hook, always_call=True)
    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()
    return outputs.logits


def suffix_batch_fn_for_position(owner: Any, input_ids_cpu: torch.Tensor, attention_mask_cpu: torch.Tensor, position: int):
    input_ids = input_ids_cpu.to(owner.device)
    attention_mask = attention_mask_cpu.to(owner.device)
    position = int(position)
    d_model = int(owner.model.config.hidden_size)

    def suffix_batch(hs: torch.Tensor) -> torch.Tensor:
        hs = hs.reshape(-1, d_model).to(owner.device).float()
        batch = int(hs.shape[0])
        toks = input_ids.expand(batch, -1).contiguous()
        attn = attention_mask.expand(batch, -1).contiguous()
        pos_t = torch.full((batch,), position, device=owner.device, dtype=torch.long)
        logits = replacement_logits(owner.model, toks, attn, owner.layer, pos_t, hs)
        return logits[torch.arange(batch, device=owner.device), pos_t].float()

    return suffix_batch


def vjp_logits_to_hidden_batch(owner: Any, suffix_batch_fn, h0: torch.Tensor, cotangents: torch.Tensor) -> torch.Tensor:
    cotangents = cotangents.detach().to(owner.device).float()
    batch = int(cotangents.shape[0])
    h = h0.detach().to(owner.device).float().reshape(1, -1).expand(batch, -1).clone().requires_grad_(True)
    z = suffix_batch_fn(h).float()
    val = (z * cotangents).sum()
    (grad,) = torch.autograd.grad(val, h, retain_graph=False, create_graph=False)
    return grad.detach().float()


@torch.no_grad()
def sample_calibration_sites(owner: Any, rows: pd.DataFrame, cfg: dict) -> list[dict]:
    rows = rows.reset_index(drop=True)
    text_col = _calibration_text_column(rows)
    label_col = "labels" if "labels" in rows.columns else "label"
    rng = random.Random(int(cfg.get("seed", 0)))
    selected: dict[int, list[dict]] = {0: [], 1: []}
    seen: Counter[int] = Counter()
    batch_size = int(cfg.get("basis_activation_batch_size", cfg.get("activation_batch_size", 8)))
    max_length = cfg.get("max_length", 1024)
    contexts_per_class = int(cfg.get("basis_contexts_per_class", 128))

    for start in tqdm(range(0, len(rows), batch_size), desc=f"realizable basis sites L{owner.layer}"):
        batch = rows.iloc[start : start + batch_size]
        texts = batch[text_col].astype(str).tolist()
        prefixes = batch["prefix"].astype(str).tolist()
        toks = owner.tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        ).to(owner.device)
        captured = {}

        def gather_hook(module, inputs, outputs):
            del module, inputs
            captured["resid"] = outputs[0] if isinstance(outputs, tuple) else outputs
            return outputs

        handle = owner.model.model.layers[int(owner.layer)].register_forward_hook(gather_hook, always_call=True)
        try:
            logits = owner.model(
                input_ids=toks["input_ids"],
                attention_mask=toks["attention_mask"],
                use_cache=False,
            ).logits
        finally:
            handle.remove()
        resid = captured["resid"].detach().float().cpu()
        logits_cpu = logits.detach().float().cpu()
        input_cpu = toks["input_ids"].detach().cpu()
        attn_cpu = toks["attention_mask"].detach().cpu()
        mask = completion_prediction_mask(
            owner.tokenizer,
            input_cpu,
            attn_cpu,
            texts,
            prefixes,
            max_length,
        )

        for local_idx, (_, row) in enumerate(batch.iterrows()):
            label = int(row[label_col])
            if label not in selected:
                continue
            for position in torch.where(mask[local_idx])[0].tolist():
                seen[label] += 1
                n_seen = int(seen[label])
                if len(selected[label]) < contexts_per_class:
                    slot = len(selected[label])
                else:
                    slot = rng.randrange(n_seen)
                    if slot >= contexts_per_class:
                        continue
                record = {
                    "input_ids": input_cpu[local_idx : local_idx + 1].clone(),
                    "attention_mask": attn_cpu[local_idx : local_idx + 1].clone(),
                    "position": int(position),
                    "h0": resid[local_idx, int(position)].clone(),
                    "logits0": logits_cpu[local_idx, int(position)].clone(),
                    "label": label,
                    "text": str(row[text_col]),
                }
                if slot == len(selected[label]):
                    selected[label].append(record)
                else:
                    selected[label][slot] = record
        del resid, logits, toks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = selected[0] + selected[1]
    rng.shuffle(out)
    return out


def local_realizability_gradient(owner: Any, record: dict, readout: torch.Tensor) -> torch.Tensor:
    suffix_batch = suffix_batch_fn_for_position(owner, record["input_ids"], record["attention_mask"], int(record["position"]))
    logits0 = record["logits0"].detach().to(owner.device).float()
    p = F.softmax(logits0, dim=-1).clamp_min(EPS)
    cotangent = fisher_apply(p, readout.to(owner.device).float(), EPS).reshape(1, -1)
    grad = vjp_logits_to_hidden_batch(owner, suffix_batch, record["h0"], cotangent).reshape(-1).detach().cpu().float()
    return grad


def covariance_realizable_basis(owner: Any, contexts: list[dict], readout: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    grads = []
    for record in tqdm(contexts, desc=f"realizable basis gradients L{owner.layer}"):
        grad = local_realizability_gradient(owner, record, readout)
        norm = float(torch.linalg.vector_norm(grad).item())
        if torch.isfinite(torch.tensor(norm)) and norm > EPS:
            grads.append(grad)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not grads:
        raise RuntimeError("No nonzero local realizability gradients were found.")
    A = torch.stack(grads, dim=0).float()
    _, singular_values, Vh = torch.linalg.svd(A, full_matrices=False)
    rank_eff = min(int(rank), int(Vh.shape[0]))
    Q = orthonormal_columns(Vh[:rank_eff].T.contiguous().float())[:, :rank_eff]
    eigenvalues = singular_values.square() / float(max(1, A.shape[0]))
    return Q.cpu(), eigenvalues.detach().cpu().float()


def projector_from_basis_tensor(owner: Any, basis: torch.Tensor) -> SubspaceProjector:
    hidden_size = int(owner.model.config.hidden_size)
    if basis.ndim != 2:
        raise ValueError(f"realizable basis must be rank-2, got shape={tuple(basis.shape)}")
    if int(basis.shape[0]) != hidden_size and int(basis.shape[1]) == hidden_size:
        basis = basis.T
    if int(basis.shape[0]) != hidden_size:
        raise ValueError(
            f"realizable basis first dimension must match hidden size {hidden_size}, "
            f"got shape={tuple(basis.shape)}"
        )
    return SubspaceProjector.from_basis(basis, orthonormalize=True)


def resolve_or_build_projector(owner: Any, examples: pd.DataFrame, kwargs: dict) -> SubspaceProjector:
    projector = kwargs.get("projector", kwargs.get("realizable_projector", None))
    if projector is not None:
        if isinstance(projector, SubspaceProjector):
            return projector
        return projector_from_basis_tensor(owner, torch.as_tensor(projector))

    basis = kwargs.get("realizable_basis", kwargs.get("basis", None))
    if basis is not None:
        return projector_from_basis_tensor(owner, torch.as_tensor(basis))

    path = kwargs.get("realizable_basis_path", _training_arg(owner, "realizable_basis_path", None))
    if path is not None:
        obj = torch.load(path, map_location=torch.device("cpu"))
        if isinstance(obj, SubspaceProjector):
            return obj
        if isinstance(obj, dict):
            for key in ("basis", "Q", "q", "projector_basis"):
                if key in obj:
                    obj = obj[key]
                    break
            else:
                raise ValueError(f"No basis tensor found in realizable basis file: {path}")
        return projector_from_basis_tensor(owner, torch.as_tensor(obj))

    readout_examples = kwargs.get("readout_examples", examples)
    cache_key = (
        id(examples),
        id(readout_examples),
        int(owner.layer),
        int(_cfg(owner, kwargs, "basis_rank", 32)),
        int(_cfg(owner, kwargs, "basis_contexts_per_class", 128)),
        int(_cfg(owner, kwargs, "prefix_length", kwargs.get("prefix_length", 1))),
    )
    cache = getattr(owner, "_realizable_projector_cache", {})
    if cache_key in cache:
        return cache[cache_key]

    validate_realizable_basis_examples(examples)
    validate_realizable_basis_examples(readout_examples)

    cfg = {
        "seed": getattr(owner, "seed", 0),
        "max_length": int(_cfg(owner, kwargs, "max_length", 1024)),
        "activation_batch_size": int(_cfg(owner, kwargs, "activation_batch_size", _training_arg(owner, "batch_size", 8))),
        "basis_activation_batch_size": int(_cfg(owner, kwargs, "basis_activation_batch_size", _cfg(owner, kwargs, "activation_batch_size", 8))),
        "basis_contexts_per_class": int(_cfg(owner, kwargs, "basis_contexts_per_class", 128)),
        "prefix_length": int(kwargs.get("prefix_length", _cfg(owner, kwargs, "prefix_length", 1))),
        "readout_smoothing": float(_cfg(owner, kwargs, "readout_smoothing", 0.1)),
        "readout_clip": float(_cfg(owner, kwargs, "readout_clip", 8.0)),
    }
    vocab_size = int(getattr(owner.model.config, "vocab_size", owner.model.lm_head.weight.shape[0]))
    readout = build_log_count_readout(owner.tokenizer, readout_examples, vocab_size, cfg)
    contexts = sample_calibration_sites(owner, examples, cfg)
    Q, _ = covariance_realizable_basis(owner, contexts, readout, int(_cfg(owner, kwargs, "basis_rank", 32)))
    projector = SubspaceProjector.from_basis(Q, orthonormalize=True)
    cache[cache_key] = projector
    owner._realizable_projector_cache = cache
    return projector
