"""
ReCrit GRPO training core logic.

Implements:
  - compute_policy_logps(): compute policy log probabilities
  - grpo_loss():            PPO-clip loss with token-level importance sampling
  - train_step():           one training-step update

Key design choices kept aligned with ms-swift:
  - logits_to_keep: compute logprobs only on the completion region and skip the prompt (saves memory)
  - temperature: logits /= temperature (consistent with ms-swift grpo_trainer.py:1619)
  - on-policy old_logprobs: use per_token_logps.detach() instead of vLLM logprobs
      (mathematically equivalent when num_iterations=1, while avoiding vLLM/PyTorch numerical drift)
  - reverse KL: exp(ref-cur) - (ref-cur) - 1 (consistent with ms-swift grpo_trainer.py:1126)
  - loss aggregation: (sum(loss*token_weights) / sum(token_weights)).mean()
      (consistent with ms-swift grpo_trainer.py:1228; here token_weights extends the original completion_mask
        defaulting to the same 0/1 mask while allowing turn_loss_weights to assign different weights to different turns)
"""

import logging
from typing import Dict, Tuple
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy log-probability computation
# ---------------------------------------------------------------------------

def compute_policy_logps(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_weights: torch.Tensor,
    bf16: bool = True,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run the forward pass and compute policy log probabilities, token-level valid weights, and Shannon entropy.

    Aligned with ms-swift in the following ways:
      - logits_to_keep optimization: keep only completion-region logits and skip the prompt
          (see the logits_to_keep computation in ms-swift grpo_trainer.py:880)
      - temperature scaling: logits /= temperature
          (see ms-swift grpo_trainer.py:1619)

    Args:
        model:          language model
        input_ids:      [B, L]  long
        labels:         [B, L]  long (-100 for non-assistant tokens)
        attention_mask: [B, L]  long
        temperature:    sampling temperature (affects both log-probability and entropy computation)

    Returns:
        per_token_logps: [B, K]  float, where K = logits_to_keep
        token_weights:   [B, K]  float, effective loss weights for assistant tokens
        token_entropy:   [B, K]  float, Shannon entropy per token

    Explanation of logits_to_keep:
        For each sequence in the batch, find the first position p where labels != -100.
        Then logits_to_keep = L - p, meaning only logits from p to the end are kept.
        Take the maximum over the batch to ensure all completion positions are covered.
        Skipped prompt positions receive token_weights = 0 everywhere and therefore do not affect the loss.
        Here token_weights can be viewed as a strict extension of the original completion_mask:
          - by default it is equivalent to a 0/1 completion mask
          - when turn_loss_weights is enabled, assistant tokens are multiplied by different per-turn coefficients
    """
    # logits_to_keep: keep only the completion region and skip the prompt (aligned with ms-swift)
    # labels layout: [-100, ..., -100, tok1, ..., tokN, -100, ..., -100]
    #               ^^^prompt^^^     ^^^completion^^^   ^^^padding^^^
    # argmax finds the first position != -100, and L minus that position gives the number of tokens to keep
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

    # Keep only the completion region plus one extra position (the shift needs the preceding logit)
    # logits[t] predicts input_ids[t+1], so K+1 logits are needed to obtain K logprobs
    logits = logits[:, -(logits_to_keep + 1):, :]  # [B, K+1, V]

    # Temperature scaling (aligned with ms-swift; no effect when temperature=1.0)
    if temperature != 1.0:
        logits = logits / temperature

    # Shift rule: logits[t] predicts input_ids[t+1]
    shift_logits  = logits[:, :-1, :]                  # [B, K, V]
    shift_targets = input_ids[:, -logits_to_keep:]     # [B, K]
    shift_labels  = labels[:, -logits_to_keep:]        # [B, K]
    shift_weights = loss_weights[:, -logits_to_keep:]  # [B, K]

    B, K, V = shift_logits.shape
    # Use the fused cross_entropy kernel so the full log_softmax tensor is never materialized, saving [B, K, V] memory
    per_token_logps = -F.cross_entropy(
        shift_logits.float().view(-1, V),
        shift_targets.clamp(min=0).view(-1),
        reduction="none",
    ).view(B, K)

    token_weights = shift_weights * (shift_labels != -100).float()  # [B, K]

    # Shannon entropy: -∑ p_i log p_i
    # Reference: entropy_from_logits in ms-swift swift/rlhf_trainers/utils.py:657
    # Chunk along the sequence dimension (64 positions per chunk) to avoid materializing the full [B, K, V] tensor and causing OOM
    _ENT_SEQ_CHUNK = 64
    chunk_entropies = []
    with torch.no_grad():
        for seq_chunk in shift_logits.float().split(_ENT_SEQ_CHUNK, dim=1):
            logps = F.log_softmax(seq_chunk, dim=-1)       # [B, chunk, V]
            ent = -(torch.exp(logps) * logps).sum(-1)      # [B, chunk]
            chunk_entropies.append(ent)
    token_entropy = torch.cat(chunk_entropies, dim=1)      # [B, K]

    return per_token_logps, token_weights, token_entropy


# ---------------------------------------------------------------------------
# GRPO loss (token-level importance sampling; see ms-swift grpo_trainer.py with loss_type='grpo')
# ---------------------------------------------------------------------------

def grpo_loss(
    per_token_logps: torch.Tensor,      # [B, K]  current policy
    old_per_token_logps: torch.Tensor,  # [B, K]  on-policy logprob (already shifted and aligned)
    token_weights: torch.Tensor,        # [B, K]  effective loss weights for assistant tokens
    advantages: torch.Tensor,           # [B]     GRPO advantages
    token_entropy: torch.Tensor,        # [B, K]  Shannon entropy per token
    epsilon: float = 0.2,               # symmetric PPO-clip coefficient
    ref_logps: torch.Tensor = None,     # [B, K]  reference-model logprobs (for the KL penalty)
    kl_beta: float = 0.0,              # KL penalty coefficient beta
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    GRPO PPO-clip loss with token-level importance sampling, plus an optional KL(pi_ref || pi) penalty.

    Aligned with ms-swift in the following ways:
        coef_1 = exp(log_ratio)                           per token
        coef_2 = clamp(coef_1, 1 - epsilon, 1 + epsilon)
        per_token_loss = -min(coef_1 * adv, coef_2 * adv)
        loss = mean_over_samples( sum(loss * token_weights) / sum(token_weights) )

    KL penalty (when ref_logps is not None and kl_beta > 0):
        Use reverse KL as in ms-swift (always >= 0):
        per_token_kl = exp(log π_ref - log π_θ) - (log π_ref - log π_θ) - 1
        per_token_loss += β * per_token_kl

    Two different quantities that are both casually called "KL" appear here:
      1. old-policy approx_kl (reported below as metrics["kl"])
         - computed from log_ratio = log pi_theta - log pi_old
         - mainly a diagnostic for the PPO / importance-sampling line
         - under on-policy num_iterations=1, old_per_token_logps = per_token_logps.detach(),
             so it is usually close to 0, and clip_frac is usually close to 0 as well
      2. ref-model reverse KL (reported below as metrics["ref_kl"])
         - computed as the reverse KL between the frozen reference model and the current policy
         - this is the actual regularizer added to the loss: per_token_loss += kl_beta * per_token_kl
         - at the very first training step, when the policy still matches the reference exactly, it is also close to 0;
             once optimizer.step() moves the policy away from the reference, it gradually becomes non-zero

    Note: in on-policy mode (old_per_token_logps = per_token_logps.detach()),
    the IS ratio stays at 1.0 and PPO-clip is inactive. This is the expected behavior when num_iterations=1.
    """
    # ── Token-Level Importance-Sampling Weights ──────────────────────────────────────────────────
    log_ratio = per_token_logps - old_per_token_logps.detach()
    coef_1 = torch.exp(log_ratio)  # [B, K]

    # ── PPO-clip ──────────────────────────────────────────────────────────────
    coef_2 = torch.clamp(coef_1, 1 - epsilon, 1 + epsilon)

    adv_expanded = advantages.unsqueeze(1)  # [B, 1]
    per_token_loss = -torch.min(coef_1 * adv_expanded, coef_2 * adv_expanded)  # [B, K]

    # ── KL(pi_ref || pi) Penalty (Reverse KL, Aligned With ms-swift) ──────────────────
    # Note: this is the KL against the frozen reference model, not the old-policy approx_kl from the PPO line.
    # It is added to the loss directly and anchors the current policy near the initial reference, preventing policy drift.
    # Formula: exp(log pi_ref - log pi_theta) - (log pi_ref - log pi_theta) - 1, which is always >= 0
    ref_kl_value = 0.0
    if ref_logps is not None and kl_beta > 0:
        ref_diff = ref_logps.detach() - per_token_logps  # log(π_ref/π)
        per_token_kl = torch.exp(ref_diff) - ref_diff - 1  # [B, K], always >= 0
        per_token_loss = per_token_loss + kl_beta * per_token_kl
        with torch.no_grad():
            ref_kl_value = (
                (per_token_kl.detach() * token_weights).sum(-1)
                / token_weights.sum(-1).clamp(min=1e-8)
            ).mean().item()

    # ── Loss Aggregation (GRPO Convention: Sequence sum/len first, then batch mean) ────────────────
    loss = (
        (per_token_loss * token_weights).sum(-1)
        / token_weights.sum(-1).clamp(min=1e-8)
    ).mean()

    # ── Diagnostic Metrics ──────────────────────────────────────────────────────────────
    with torch.no_grad():
        # approx_kl: E[log(pi/pi_old)], the standard approximate KL divergence used in the PPO literature.
        # It is returned as metrics["kl"] and mainly tells us whether the PPO / IS line is actually moving.
        # It is not the same as the reference-model KL above; under on-policy num_iterations=1,
        # because old = current.detach(), this value is usually close to 0.
        mean_kl = (
            (log_ratio.detach() * token_weights).sum(-1)
            / token_weights.sum(-1).clamp(min=1e-8)
        ).mean().item()

        # clip_frac is tied to the old-policy ratio. When the ratio stays at 1 (common for num_iterations=1),
        # this value is usually close to 0 as well.
        clip_frac = (
            ((coef_1 < 1 - epsilon) | (coef_1 > 1 + epsilon)).float()
            * token_weights
        ).sum() / token_weights.sum().clamp(min=1e-8)

        # Shannon entropy: -sum p_i log p_i (the true policy entropy computed by compute_policy_logps)
        entropy = (
            (token_entropy.detach() * token_weights).sum(-1)
            / token_weights.sum(-1).clamp(min=1e-8)
        ).mean().item()

    return loss, {"kl": mean_kl, "clip_frac": clip_frac.item(), "entropy": entropy, "ref_kl": ref_kl_value}


# ---------------------------------------------------------------------------
# Single-step training
# ---------------------------------------------------------------------------

def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    config,
    is_opt_step: bool = True,
    accumulation_scale: float = 1.0,
    ref_model: nn.Module = None,
) -> Dict[str, float]:
    """
    Execute one forward/backward pass and optionally call optimizer.step().

    Within a batch, the B samples are split into chunks of config.train_micro_batch_size for forward/backward passes,
    gradients accumulate naturally across chunks, and clipping + stepping are performed once at the end.

    Key points aligned with ms-swift:
      - old_logprobs uses per_token_logps.detach() (on-policy, num_iterations=1)
        ms-swift computes old_logprobs through a training-model forward pass inside _prepare_batch_inputs,
        and for num_iterations=1 that result is mathematically equivalent to per_token_logps.detach().
      - ref_logprobs uses a forward pass of the frozen reference model (consistent with ms-swift)
      - temperature is passed into compute_policy_logps (consistent with ms-swift)

    Args:
        is_opt_step: whether to execute optimizer.step() + zero_grad.
                     There are two cases where it is True:
                       1. the accumulation count reaches gradient_accumulation_steps
                       2. the final batch of the epoch (force a step so DDP gradients are not left unsynchronized)
        accumulation_scale: loss scaling factor (= 1 / gradient_accumulation_steps),
                           ensuring that the average gradient after accumulating N steps matches the single-step case.
        ref_model: frozen reference model used for the KL penalty. If None, no KL penalty is computed.

    DDP note: use no_sync() so that all-reduce is performed only on the final chunk,
    avoiding repeated averaging of gradients from intermediate chunks.
    When is_opt_step=False, every chunk uses no_sync(), because
    all-reduce should be delayed to the final chunk of the final accumulation step.
    """
    model.train()
    device = next(model.parameters()).device
    is_ddp = isinstance(model, DDP)

    input_ids      = batch["input_ids"].to(device)
    labels         = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    loss_weights   = batch["loss_weights"].to(device)
    advantages     = batch["advantages"].to(device)
    # old_logprobs_aligned is no longer used: in on-policy mode it is replaced with per_token_logps.detach()
    # (avoids numerical mismatch between vLLM and PyTorch and is equivalent to ms-swift at num_iterations=1)

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
        w   = (end - i) / total_samples   # weight of this chunk within the full batch
        is_last_chunk = (end >= total_samples)

        # ── Reference-Model Logprobs (For The KL Penalty) ────────────────────────────────
        ref_logps = None
        if ref_model is not None and kl_beta > 0:
            with torch.no_grad():
                ref_logps, _, _ = compute_policy_logps(
                    ref_model, input_ids[i:end], labels[i:end],
                    attention_mask[i:end], loss_weights[i:end], bf16=config.bf16,
                    temperature=temperature,
                )

        # DDP synchronization policy:
        #   - is_opt_step=False (not the final accumulation step): every chunk uses no_sync
        #   - is_opt_step=True (the final accumulation step): only the final chunk performs all-reduce
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

            # On-policy: old_logprobs = current-policy logprobs
            # Model weights do not change during accumulation (optimizer has not stepped yet),
            # so per_token_logps.detach() is mathematically equivalent to the result ms-swift gets by using the training model for a separate forward pass
            # to compute old_logprobs separately (and is more efficient because it saves one forward pass).
            # Therefore:
            #   - old-policy approx_kl (metrics["kl"]) is usually close to 0
            #   - clip_frac is usually close to 0
            # The quantity that can still provide regularization is ref-model KL (metrics["ref_kl"]),
            # but at the very first training step, when the policy has not yet moved away from the reference, it is also close to 0.
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
