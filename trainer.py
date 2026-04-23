""

import logging
from typing import Dict, Tuple
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)






def compute_policy_logps(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_weights: torch.Tensor,
    bf16: bool = True,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ""




    logits_to_keep = (
        labels.shape[-1] - (torch.ne(labels, -100).int().argmax(-1))
    ).max().item()

    amp_dtype = torch.bfloat16 if bf16 else torch.float32
    with torch.amp.autocast("cuda", enabled=bf16, dtype=amp_dtype):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

    logits = outputs.logits  # [B, L, V]



    logits = logits[:, -(logits_to_keep + 1):, :]  # [B, K+1, V]


    if temperature != 1.0:
        logits = logits / temperature


    shift_logits  = logits[:, :-1, :]                  # [B, K, V]
    shift_targets = input_ids[:, -logits_to_keep:]     # [B, K]
    shift_labels  = labels[:, -logits_to_keep:]        # [B, K]
    shift_weights = loss_weights[:, -logits_to_keep:]  # [B, K]

    B, K, V = shift_logits.shape

    per_token_logps = -F.cross_entropy(
        shift_logits.float().view(-1, V),
        shift_targets.clamp(min=0).view(-1),
        reduction="none",
    ).view(B, K)

    token_weights = shift_weights * (shift_labels != -100).float()  # [B, K]




    _ENT_SEQ_CHUNK = 64
    chunk_entropies = []
    with torch.no_grad():
        for seq_chunk in shift_logits.float().split(_ENT_SEQ_CHUNK, dim=1):
            logps = F.log_softmax(seq_chunk, dim=-1)       # [B, chunk, V]
            ent = -(torch.exp(logps) * logps).sum(-1)      # [B, chunk]
            chunk_entropies.append(ent)
    token_entropy = torch.cat(chunk_entropies, dim=1)      # [B, K]

    return per_token_logps, token_weights, token_entropy






def grpo_loss(
    per_token_logps: torch.Tensor,
    old_per_token_logps: torch.Tensor,
    token_weights: torch.Tensor,
    advantages: torch.Tensor,
    token_entropy: torch.Tensor,        # [B, K]  Shannon entropy per token
    epsilon: float = 0.2,
    ref_logps: torch.Tensor = None,
    kl_beta: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    ""

    log_ratio = per_token_logps - old_per_token_logps.detach()
    coef_1 = torch.exp(log_ratio)  # [B, K]


    coef_2 = torch.clamp(coef_1, 1 - epsilon, 1 + epsilon)

    adv_expanded = advantages.unsqueeze(1)  # [B, 1]
    per_token_loss = -torch.min(coef_1 * adv_expanded, coef_2 * adv_expanded)  # [B, K]





    ref_kl_value = 0.0
    if ref_logps is not None and kl_beta > 0:
        ref_diff = ref_logps.detach() - per_token_logps
        per_token_kl = torch.exp(ref_diff) - ref_diff - 1  # [B, K], always >= 0
        per_token_loss = per_token_loss + kl_beta * per_token_kl
        with torch.no_grad():
            ref_kl_value = (
                (per_token_kl.detach() * token_weights).sum(-1)
                / token_weights.sum(-1).clamp(min=1e-8)
            ).mean().item()


    loss = (
        (per_token_loss * token_weights).sum(-1)
        / token_weights.sum(-1).clamp(min=1e-8)
    ).mean()


    with torch.no_grad():




        mean_kl = (
            (log_ratio.detach() * token_weights).sum(-1)
            / token_weights.sum(-1).clamp(min=1e-8)
        ).mean().item()



        clip_frac = (
            ((coef_1 < 1 - epsilon) | (coef_1 > 1 + epsilon)).float()
            * token_weights
        ).sum() / token_weights.sum().clamp(min=1e-8)


        entropy = (
            (token_entropy.detach() * token_weights).sum(-1)
            / token_weights.sum(-1).clamp(min=1e-8)
        ).mean().item()

    return loss, {"kl": mean_kl, "clip_frac": clip_frac.item(), "entropy": entropy, "ref_kl": ref_kl_value}






def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    config,
    is_opt_step: bool = True,
    accumulation_scale: float = 1.0,
    ref_model: nn.Module = None,
) -> Dict[str, float]:
    ""
    model.train()
    device = next(model.parameters()).device
    is_ddp = isinstance(model, DDP)

    input_ids      = batch["input_ids"].to(device)
    labels         = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    loss_weights   = batch["loss_weights"].to(device)
    advantages     = batch["advantages"].to(device)



    total_samples = input_ids.size(0)
    train_micro_batch_size = config.train_micro_batch_size
    temperature   = config.temperature

    total_loss = 0.0
    total_kl   = 0.0
    total_clip = 0.0
    total_entr = 0.0
    total_ref_kl = 0.0
    kl_beta = config.kl_beta

    for i in range(0, total_samples, train_micro_batch_size):
        end = min(i + train_micro_batch_size, total_samples)
        w   = (end - i) / total_samples
        is_last_chunk = (end >= total_samples)


        ref_logps = None
        if ref_model is not None and kl_beta > 0:
            with torch.no_grad():
                ref_logps, _, _ = compute_policy_logps(
                    ref_model, input_ids[i:end], labels[i:end],
                    attention_mask[i:end], loss_weights[i:end], bf16=config.bf16,
                    temperature=temperature,
                )




        if not is_ddp:
            sync_ctx = nullcontext()
        elif not is_opt_step or not is_last_chunk:
            sync_ctx = model.no_sync()
        else:
            sync_ctx = nullcontext()

        with sync_ctx:
            per_token_logps, token_weights, token_entropy = compute_policy_logps(
                model, input_ids[i:end], labels[i:end], attention_mask[i:end],
                loss_weights[i:end], bf16=config.bf16, temperature=temperature,
            )










            old_per_token_logps = per_token_logps.detach()

            loss, loss_metrics = grpo_loss(
                per_token_logps=per_token_logps,
                old_per_token_logps=old_per_token_logps,
                token_weights=token_weights,
                advantages=advantages[i:end],
                token_entropy=token_entropy,
                epsilon=config.epsilon,
                ref_logps=ref_logps,
                kl_beta=kl_beta,
            )

            (loss * w * accumulation_scale).backward()

        total_loss += loss.item() * w
        total_kl   += loss_metrics["kl"]        * w
        total_clip += loss_metrics["clip_frac"] * w
        total_entr += loss_metrics["entropy"]   * w
        total_ref_kl += loss_metrics["ref_kl"]  * w

    grad_norm = 0.0
    if is_opt_step:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return {
        "loss":      total_loss,
        "kl":        total_kl,
        "clip_frac": total_clip,
        "entropy":   total_entr,
        "ref_kl":    total_ref_kl,
        "grad_norm": grad_norm,
    }
