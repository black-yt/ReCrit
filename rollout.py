"""
ReCrit rollout module.

Responsibilities:
  1. Run N-turn multi-turn rollout with vLLM (ReCritScheduler logic)
  2. Sync training-model weights into vLLM (vLLM 0.17 V1 API: CUDA IPC + apply_model)
  3. Convert rollout results into PyTorch training batches

Dialogue structure (num_turns=N):
    user      : <question>
    assistant : <turn_1>    ← participates in gradient computation
    user      : <critic_1>
    assistant : <turn_2>    ← participates in gradient computation
    ...
    user      : <critic_{N-1}>
    assistant : <turn_N>    ← participates in gradient computation

Critic attitudes (3 types, 7 templates each):
    Opposing / Supportive / Neutral
"""

import random
import logging
import gc
import math
import time
from typing import List, Dict, Optional
from collections import Counter

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Critic templates (kept aligned with plugin/rewards.py)
# ---------------------------------------------------------------------------

ATTITUDE_TEMPLATES: Dict[str, List[str]] = {
    "opposing": [
        "Your answer seems to have some issues. Could you reconsider it carefully?",
        "I'm not sure your answer is entirely correct. Please try again.",
        "There might be a mistake in your response. Can you re-examine it?",
        "Your response doesn't seem quite right to me. Please think it through again.",
        "I believe your answer may be incorrect. Could you revise it?",
        "It looks like something might be off in your answer. Please reconsider.",
        "I'm not convinced by your response. Could you take another careful look?",
    ],
    "supportive": [
        "I think your answer is on the right track! Could you refine and improve it a bit more?",
        "That reasoning makes sense! Can you polish your answer to make it even clearer?",
        "Great thinking! Would you mind elaborating and refining your response?",
        "I agree with your direction! Can you make your answer more precise and rigorous?",
        "You seem to be on the right track! Could you enhance and finalize your answer?",
        "Your response looks promising — please go ahead and refine it for maximum clarity.",
        "Nicely done! Can you revisit your answer and make it a bit more thorough?",
    ],
    "neutral": [
        "Perhaps you could double-check your answer and revise it if needed?",
        "It might be worth revisiting your response to make sure it is fully accurate.",
        "Could you review your answer once more and make any necessary adjustments?",
        "Would you like to reconsider your response and verify its correctness?",
        "Please take another careful look at your answer before finalizing.",
        "Could you verify your reasoning and update the answer if you spot any issues?",
        "Feel free to re-examine your logic and refine the answer accordingly.",
    ],
}

_ATTITUDES = list(ATTITUDE_TEMPLATES.keys())


def _sample_critic_message(config):
    """Sample the critic prompt for the next turn based on the config.

    Keep the default mixed mode unchanged; switch only when critic_prompt_mode=eval_fixed
    to the benchmark-aligned fixed prompt.
    """
    if getattr(config, "critic_prompt_mode", "mixed") == "eval_fixed":
        return "eval_fixed", config.critic_prompt_text
    attitude = random.choice(_ATTITUDES)
    critic_msg = random.choice(ATTITUDE_TEMPLATES[attitude])
    return attitude, critic_msg

# ---------------------------------------------------------------------------
# vLLM parameter fusion rules (Qwen3.5 hybrid model + vLLM MergedColumnParallelLinear)
# vLLM merges multiple small parameters into one large parameter, concatenated along dim=0 in order:
#   q_proj + k_proj + v_proj → qkv_proj (standard attention, 8 layers)
#   gate_proj + up_proj → gate_up_proj (MLP)
#   in_proj_qkv + in_proj_z → in_proj_qkvz (linear attention)
#   in_proj_b + in_proj_a → in_proj_ba (linear attention)
# ---------------------------------------------------------------------------

_FUSION_RULES = [
    (["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"],
     "self_attn.qkv_proj.weight"),
    (["linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight"],
     "linear_attn.in_proj_qkvz.weight"),
    (["linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight"],
     "linear_attn.in_proj_ba.weight"),
    (["mlp.gate_proj.weight", "mlp.up_proj.weight"],
     "mlp.gate_up_proj.weight"),
]


# ---------------------------------------------------------------------------
# Rollout: truly asynchronous multi-turn dialogue generation
# ---------------------------------------------------------------------------

def _tokenize_messages(tokenizer, messages: List[Dict]) -> List[int]:
    """Tokenize a message list into prompt token IDs."""
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors=None,
    )
    if not isinstance(ids, list):
        ids = ids["input_ids"]
    return [int(x) for x in ids]


def _detect_generation_prefix(tokenizer, system_prompt: str = "") -> str:
    """Detect whether the generation prompt contains an extra suffix such as <think>\\n.

    Method: compare add_generation_prompt=True/False outputs and extract
    the extra text after the assistant header. For example, Qwen3 appends '<think>\\n',
    while a standard model appends only '<|im_start|>assistant\\n'.

    Returns:
        The extra suffix string (for example '<think>\\n'); return '' if absent.
    """
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": "test"})

    with_gen = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
    )
    without_gen = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False,
    )
    # The generation prompt is the suffix that with_gen adds beyond without_gen.
    gen_prompt = with_gen[len(without_gen):]

    # A typical generation prompt looks like "<|im_start|>assistant\n<think>\n".
    # We need to detect whether anything extra appears after "assistant\n".
    # Locate it by finding the assistant header before the final appended suffix.
    assistant_markers = ["assistant\n", "assistant\\n"]
    for marker in assistant_markers:
        idx = gen_prompt.rfind(marker)
        if idx >= 0:
            prefix = gen_prompt[idx + len(marker):]
            return prefix

    return ""


def run_rollout(
    llm,
    tokenizer,
    prompts: List[Dict],
    num_generations: int,
    config,
    num_turns: int = 2,
    debug: bool = False,
    rank: int = 0,
    world_size: int = 1,
    step_label: str = "",
) -> List[Dict]:
    """
    Truly asynchronous multi-turn rollout: each sample advances independently without waiting for others.

    Implemented with the vLLM engine add_request() / step() API:
      - submit all turn-0 requests at once
      - once a sample finishes the current turn, inject critic feedback and submit the next turn immediately
      - different samples may stay at different turns while vLLM continuous batching schedules them automatically
      - no bubbles: fast samples do not wait for slow ones

    Early stop triggers when a completion_ratio fraction of samples reaches num_turns
    and 100% of samples reach the minimum turn count required for early stop; then stop submitting new requests and let in-flight requests finish naturally.
    In the current two-turn mode, that minimum count is 1; after rollout, samples that complete only one turn are padded into
    a degenerate keep trajectory with “turn2 = turn1”, so the existing reward / batch-building logic can be reused.

    Args:
        llm:             vLLM LLM instance (called only on rank 0)
        tokenizer:       HuggingFace tokenizer
        prompts:         List[{'question': str, 'answer': str, 'judge_mode': str}]
        num_generations: G, the number of samples per prompt
        config:          ReCritConfig (must contain completion_ratio)
        num_turns:       maximum number of turns (default: 2)

    Returns:
        results: a List[Dict] of length len(prompts) * num_generations,
                 ordered as [p0_g0, p0_g1, ..., pN_g{G-1}].
        Each result contains:
            responses:       List[str]          — generated text at each turn
            token_ids:       List[List[int]]    — token IDs at each turn
            prompt_ids:      List[List[int]]    — prompt token IDs used for each turn
            attitudes:       List[str]          — critic attitude between turns
            critic_messages: List[str]          — critic message between turns
        For the current two-turn early-stop path, the return value is padded to length 2, with a synthetic keep as turn2.
    """
    from vllm import SamplingParams

    # The current main path is on-policy with num_iterations=1; old/current logprobs are recomputed in trainer.py
    # with training-time forward passes. Because each rollout sample participates in optimization only once,
    # old_logprobs = per_token_logps.detach(), so log_ratio=0 and ratio=1.
    # Therefore we do not request/save rollout-side token logprobs here, to avoid unnecessary state and comment drift.
    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_new_tokens,
        stop_token_ids=_get_stop_token_ids(tokenizer),
    )

    # ── Detect the generation-prompt suffix (for example Qwen3's <think>\n) ──
    # add_generation_prompt=True appends a model-specific suffix at the end of the prompt (for example <think>\n).
    # That suffix does not appear in vLLM seq.text. Detect it and prepend it back when saving the response,
    # so downstream consumers (format checks / Judge / debug logs) see the full text.
    _gen_prefix = _detect_generation_prefix(tokenizer, config.system_prompt)
    _dbg_tag = f"[rank={rank}/{world_size}] [{step_label}]" if step_label else f"[rank={rank}/{world_size}]"

    if _gen_prefix and debug:
        logger.info(f"{_dbg_tag} generation prefix detected: {_gen_prefix!r}")

    n_prompts = len(prompts)
    n_samples = n_prompts * num_generations
    completion_ratio = config.completion_ratio
    n_complete_target = math.ceil(n_samples * completion_ratio)
    # Two-turn specialization: allow a sample to count toward early-stop once turn1 finishes;
    # after rollout, pad it into a degenerate turn2=turn1 trajectory so the existing critic-reward computation can be reused.
    min_turns_for_early_stop = 1 if num_turns == 2 else 2

    # ── Initialize per-sample state ───────────────────────────────────────────
    states: List[Dict] = []
    for p in prompts:
        for _ in range(num_generations):
            msgs: List[Dict] = []
            if config.system_prompt:
                msgs.append({"role": "system", "content": config.system_prompt})
            msgs.append({"role": "user", "content": p["question"]})
            states.append({
                "messages":        msgs,
                "responses":       [],
                "token_ids":       [],
                "prompt_ids":      [],
                "attitudes":       [],
                "critic_messages": [],
                "done":            False,
            })

    engine = llm.llm_engine
    pending: Dict[str, int] = {}   # request_id → sample_idx
    req_counter = 0
    stop_flag = False              # Once True, do not submit new requests anymore

    def _submit(sample_idx: int):
        """Tokenize the current messages and submit them to the vLLM engine."""
        nonlocal req_counter
        state = states[sample_idx]
        ids = _tokenize_messages(tokenizer, state["messages"])
        state["_prompt_ids"] = ids

        req_id = f"r{req_counter}"
        req_counter += 1
        engine.add_request(
            request_id=req_id,
            prompt={"type": "token", "prompt_token_ids": ids},
            params=sampling_params,
        )
        pending[req_id] = sample_idx

    # ── Submit all turn-0 requests ────────────────────────────────────────────
    for i in range(n_samples):
        _submit(i)

    logger.info(
        f"[Rollout] Async start: {n_samples} samples, max {num_turns} turns, "
        f"completion_ratio={completion_ratio:.0%}, "
        f"critic_prompt_mode={getattr(config, 'critic_prompt_mode', 'mixed')}"
    )
    if getattr(config, "critic_prompt_mode", "mixed") == "eval_fixed":
        logger.info(
            f"[Rollout] Fixed critic prompt: {config.critic_prompt_text!r}"
        )

    # ── Debug: track asynchronous turn progression ────────────────────────────
    _debug_t0 = time.monotonic() if debug else 0.0
    _debug_events: List[tuple] = []  # (elapsed, sample_idx, n_done)

    # ── Main loop: step the engine, process finished requests, submit the next turn ─
    while engine.has_unfinished_requests():
        step_outputs = engine.step()

        for output in step_outputs:
            if not output.finished:
                continue

            req_id = output.request_id
            if req_id not in pending:
                logger.debug(f"[Rollout] Ignoring unknown request_id={req_id}")
                continue
            idx = pending.pop(req_id)
            state = states[idx]

            # ── Save outputs ─────────────────────────────────────────────────
            seq = output.outputs[0]
            # Add back the generation-prompt suffix (for example Qwen3's <think>\n).
            resp_text = _gen_prefix + seq.text if _gen_prefix else seq.text
            state["responses"].append(resp_text)
            state["token_ids"].append(list(seq.token_ids))
            state["prompt_ids"].append(state.pop("_prompt_ids"))

            n_done = len(state["responses"])
            idx_in_batch = idx // num_generations
            idx_in_group = idx % num_generations

            if debug:
                elapsed = time.monotonic() - _debug_t0
                _debug_events.append((elapsed, idx, n_done))
                logger.info(
                    f"{_dbg_tag} "
                    f"[idx_in_batch={idx_in_batch}/{n_prompts}, idx_in_group={idx_in_group}/{num_generations}, turn={n_done}/{num_turns}] "
                    f"[t={elapsed:.2f}s]"
                )

            # ── Decide whether to continue ────────────────────────────────────
            # 1. Reached the maximum number of turns → done
            if n_done >= num_turns:
                state["done"] = True
                continue

            # 2. Early stop has been triggered and the minimum turn count is met → done
            if stop_flag and n_done >= min_turns_for_early_stop:
                state["done"] = True
                logger.debug(f"[Rollout] batch={idx_in_batch} gen={idx_in_group}: stopped early (flag set)")
                continue

            # 3. Check whether early stop should be triggered
            if not stop_flag and n_done >= min_turns_for_early_stop:
                n_at_max = sum(
                    1 for st in states if len(st["responses"]) >= num_turns
                )
                n_at_min = sum(
                    1 for st in states if len(st["responses"]) >= min_turns_for_early_stop
                )
                if n_at_min == n_samples and n_at_max >= n_complete_target:
                    stop_flag = True
                    logger.info(
                        f"[Rollout] Async early stop triggered: "
                        f"{n_at_max}/{n_samples} completed {num_turns} turns. "
                        f"In-flight requests will finish current turn."
                    )
                    # Stop this sample as well.
                    state["done"] = True
                    continue

            # ── Inject critic feedback and submit the next turn ──────────────
            attitude, critic_msg = _sample_critic_message(config)
            state["attitudes"].append(attitude)
            state["critic_messages"].append(critic_msg)
            state["messages"] = state["messages"] + [
                {"role": "assistant", "content": resp_text},
                {"role": "user",      "content": critic_msg},
            ]
            _submit(idx)

    # ── Debug: summarize evidence of asynchronous execution ──────────────────
    if debug and _debug_events:
        # Check whether overlap exists: different samples at different turns within the same time window.
        # Sort by time and compare turn differences inside a sliding window.
        _debug_events.sort()
        max_turn_gap = 0
        has_overlap = False
        for j in range(len(_debug_events)):
            # Find events within one second of this event.
            for k in range(j + 1, len(_debug_events)):
                if _debug_events[k][0] - _debug_events[j][0] > 1.0:
                    break
                gap = abs(_debug_events[k][2] - _debug_events[j][2])
                if gap > 0:
                    has_overlap = True
                    max_turn_gap = max(max_turn_gap, gap)
        logger.info(
            f"{_dbg_tag} Summary: async_overlap={'Yes' if has_overlap else 'No'} "
            f"(max_turn_gap={max_turn_gap}, total_events={len(_debug_events)})"
        )

    if min_turns_for_early_stop < 2:
        real_turn_counts = Counter(len(st["responses"]) for st in states)
        logger.info(
            "[Rollout] Realized turn distribution before synthetic keep: "
            + ", ".join(f"{t} turns={c}" for t, c in sorted(real_turn_counts.items()))
        )
        # two-turn keep fallback：
        # For samples that still have only one turn after early stop, do not send a real turn2 request; instead append one critic prompt,
        # and copy turn1 into turn2. This preserves the standard two-turn structure for later reward/build_batch code.
        for state in states:
            if len(state["responses"]) != 1:
                continue
            keep_resp = state["responses"][0]
            attitude, critic_msg = _sample_critic_message(config)
            next_messages = state["messages"] + [
                {"role": "assistant", "content": keep_resp},
                {"role": "user", "content": critic_msg},
            ]
            state["attitudes"].append(attitude)
            state["critic_messages"].append(critic_msg)
            state["responses"].append(keep_resp)
            state["token_ids"].append(state["token_ids"][0])
            # The synthetic turn2 directly reuses turn1 tokens to keep the training structure complete.
            state["prompt_ids"].append(_tokenize_messages(tokenizer, next_messages))

    # ── Assemble results ──────────────────────────────────────────────────────
    results: List[Dict] = []
    for i in range(n_samples):
        prompt_idx = i // num_generations
        state = states[i]
        results.append({
            "question":        prompts[prompt_idx]["question"],
            "answer":          prompts[prompt_idx]["answer"],
            "judge_mode":      prompts[prompt_idx].get("judge_mode", "close"),
            "prompt_idx":      prompt_idx,
            "gen_idx":         i % num_generations,
            "attitudes":       state["attitudes"],
            "critic_messages": state["critic_messages"],
            "responses":       state["responses"],
            "token_ids":       state["token_ids"],
            "prompt_ids":      state["prompt_ids"],
        })

    # Log turn-count distribution.
    turn_counts = Counter(len(st["responses"]) for st in states)
    if len(turn_counts) == 1:
        turns, count = next(iter(turn_counts.items()))
        logger.info(
            f"[Rollout] Completed {count} rollouts "
            "(per_device_train_batch_size x num_generations), "
            f"each with {turns} turns."
        )
    else:
        logger.info(
            f"[Rollout] Completed {n_samples} rollouts "
            "(per_device_train_batch_size x num_generations). "
            "Realized turn distribution: "
            + ", ".join(f"{t} turns={c}" for t, c in sorted(turn_counts.items()))
        )

    return results


def _get_stop_token_ids(tokenizer) -> List[int]:
    """Get the stop-token ID list for Qwen/chat models."""
    stop_ids = []
    for tok in ["<|im_end|>", "<|endoftext|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid and tid != tokenizer.unk_token_id:
            stop_ids.append(tid)
    if tokenizer.eos_token_id and tokenizer.eos_token_id not in stop_ids:
        stop_ids.append(tokenizer.eos_token_id)
    return stop_ids

# ---------------------------------------------------------------------------
# Build training batches
# ---------------------------------------------------------------------------

def _find_bridge(
    next_prompt_ids: List[int],
    prev_prompt_ids: List[int],
    im_end_id: int,
) -> List[int]:
    """
    Extract the bridge token IDs between two turns from next_prompt_ids.

    Background (the Qwen3 <think> tokenization issue)
    ─────────────────────────────────────
    When add_generation_prompt=True, the Qwen3 chat template appends
    <think>\\n after <|im_start|>assistant\\n, which places the model into thinking mode.
    As a result, prev_prompt_ids (the previous-turn generation prompt) ends with:
        ...<|im_end|>\\n<|im_start|>assistant\\n<think>\\n

    However, when next_prompt_ids re-tokenizes the completed dialogue from the previous turn, the finished
    assistant turn is rendered as <|im_start|>assistant\\n{text}<|im_end|>,
    which does not include <think> because <think> belongs to the generation prompt, not the dialogue content.

    Therefore the bridge cannot be sliced by a simple next_prompt_ids[len(prev)+len(resp):],
    because BPE creates different token boundaries at the end of the generation prompt versus in the middle of the dialogue.

    The correct approach is to search for the (n_skip)-th <|im_end|> inside next_prompt_ids,
    and take everything after it as the bridge.
        n_skip = the number of <|im_end|> tokens in prev_prompt_ids + 1
              = (the number of complete messages in the previous prompt) + 1 (skip the end of the previous assistant turn)

    The bridge contains: \\n<|im_start|>user\\n{critic}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n

    Args:
        next_prompt_ids: prompt token IDs used for the next-turn generation (ending with <think>\\n)
        prev_prompt_ids: prompt token IDs used for the previous-turn generation
        im_end_id:       the token ID of <|im_end|>

    Returns:
        bridge_ids: token IDs of the bridge segment (labels are set to -100)
    """
    n_skip = sum(1 for t in prev_prompt_ids if t == im_end_id) + 1
    count = 0
    for i, tok in enumerate(next_prompt_ids):
        if tok == im_end_id:
            count += 1
            if count == n_skip:
                return next_prompt_ids[i + 1:]
    logger.warning("[Rollout] _find_bridge: could not locate bridge boundary; returning empty.")
    return []


def build_training_batch(
    results: List[Dict],
    tokenizer,
    max_seq_length: int,
    turn_loss_weights: Optional[List[float]] = None,
) -> tuple:
    """
    Convert rollout results into a PyTorch training batch (supports any turn count N).

    Sequence structure (N turns):
        [prompt_ids[0] | resp_ids[0] | bridge[0] | resp_ids[1] | bridge[1] | … | resp_ids[N-1]]
         labels: -100  | real        | -100       | real        | -100       |   | real

    Where:
        prompt_ids[0]       = the prompt used for turn-1 generation (system + user + generation_prompt)
        resp_ids[turn_idx]  = token IDs generated by vLLM at turn_idx, plus <|im_end|>
        bridge[turn_idx]    = the suffix of prompt_ids[turn_idx+1] after the end of the turn_idx assistant response
                              = \\n<|im_start|>user\\n{critic}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n

    Truncation policy: preferentially truncate the tail of the final turn; if earlier content already exceeds the limit, drop the whole sample.

    Returns:
        (batch_or_None, kept_indices)
        batch_or_None: dict with keys input_ids, labels, attention_mask,
                       advantages, loss_weights — all Tensors;
                       or None (if every sample is dropped)
        kept_indices:  indices of retained results (used to recompute advantages after dropping samples)
    """
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None or im_end_id == tokenizer.unk_token_id:
        raise RuntimeError(
            "Tokenizer does not contain the <|im_end|> token, so the training "
            "sequence cannot be constructed. Please make sure the model uses a "
            "Qwen/ChatML-style tokenizer."
        )
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    kept_indices: List[int] = []
    samples = []
    for idx, r in enumerate(results):
        prompt_ids_list = [list(p) for p in r["prompt_ids"]]   # List[List[int]], length N
        resp_ids_list   = [list(t) for t in r["token_ids"]]    # List[List[int]], length N
        per_turn_weights = (
            list(turn_loss_weights)
            if turn_loss_weights
            else [1.0] * len(resp_ids_list)
        )

        # Add <|im_end|>. vLLM stops before emitting the stop token, but the training model must learn to generate it.
        for tids in resp_ids_list:
            if im_end_id and (not tids or tids[-1] != im_end_id):
                tids.append(im_end_id)

        # Compute the bridge between adjacent turns (length N-1).
        bridge_ids_list = [
            _find_bridge(prompt_ids_list[turn_idx + 1], prompt_ids_list[turn_idx], im_end_id)
            for turn_idx in range(len(resp_ids_list) - 1)
        ]

        # Truncation policy: fixed length = prefix + all non-final turns + all bridges; only the final turn may be truncated.
        prefix_ids = prompt_ids_list[0]
        fixed_len = (
            len(prefix_ids)
            + sum(len(r) for r in resp_ids_list[:-1])
            + sum(len(b) for b in bridge_ids_list)
        )
        total_len = fixed_len + len(resp_ids_list[-1])

        if total_len > max_seq_length:
            max_last = max_seq_length - fixed_len
            if max_last <= 0:
                logger.debug(
                    "Skip sample: non-last-turn content too long (%d tokens)", fixed_len
                )
                continue
            resp_ids_list[-1] = resp_ids_list[-1][:max_last]

        # Concatenate the full sequence.
        full_ids: List[int] = list(prefix_ids)
        labels:   List[int] = [-100] * len(prefix_ids)
        loss_weights: List[float] = [0.0] * len(prefix_ids)

        for turn_idx, resp_ids in enumerate(resp_ids_list):
            full_ids.extend(resp_ids)
            labels.extend(resp_ids)           # assistant tokens → participate in loss
            loss_weights.extend([per_turn_weights[turn_idx]] * len(resp_ids))

            if turn_idx < len(bridge_ids_list):
                bridge = bridge_ids_list[turn_idx]
                full_ids.extend(bridge)
                labels.extend([-100] * len(bridge))   # bridge → does not participate in loss
                loss_weights.extend([0.0] * len(bridge))

        kept_indices.append(idx)
        samples.append({
            "input_ids":            full_ids,
            "labels":               labels,
            "loss_weights":         loss_weights,
            "advantage":            r["advantage"],
            "seq_len":              len(full_ids),
        })

    if not samples:
        return None, []

    # ── padding ────────────────────────────────────────────────────────────────
    max_len = max(s["seq_len"] for s in samples)
    batch: Dict[str, List] = {
        "input_ids": [], "labels": [],
        "advantages": [], "attention_mask": [], "loss_weights": [],
    }
    for s in samples:
        pad = max_len - s["seq_len"]
        batch["input_ids"].append(           s["input_ids"]            + [pad_id] * pad)
        batch["labels"].append(              s["labels"]               + [-100]   * pad)
        batch["loss_weights"].append(        s["loss_weights"]         + [0.0]    * pad)
        batch["advantages"].append(s["advantage"])
        batch["attention_mask"].append([1] * s["seq_len"] + [0] * pad)

    return {
        "input_ids":            torch.tensor(batch["input_ids"],            dtype=torch.long),
        "labels":               torch.tensor(batch["labels"],               dtype=torch.long),
        "loss_weights":         torch.tensor(batch["loss_weights"],         dtype=torch.float32),
        "advantages":           torch.tensor(batch["advantages"],           dtype=torch.float32),
        "attention_mask":       torch.tensor(batch["attention_mask"],       dtype=torch.long),
    }, kept_indices


# ---------------------------------------------------------------------------
# vLLM weight synchronization (vLLM 0.17 UniProcExecutor + sleep/wake mode)
# ---------------------------------------------------------------------------

def vllm_sleep(llm, level: int = 1) -> None:
    """
    Put the vLLM engine into sleep mode and free GPU memory for training.
    level=1: offload model weights to CPU and free the KV-cache GPU memory.
    """
    llm.sleep(level=level)
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"[vLLM] Sleeping (level={level}), GPU memory freed.")


def vllm_wake_and_sync(llm, training_model: nn.Module, sync_weights: bool = True) -> None:
    """
    Wake up the vLLM engine and optionally sync the latest training-model weights into it.

    Args:
        sync_weights: whether to sync weights. During intermediate accumulation steps, parameters have not been updated,
                     so passing False skips sync and saves about ~0.2s per step.

    vLLM 0.17 V1 implementation (EngineCore runs in a separate subprocess):
        1. llm.wake_up()  →  vLLM reallocates GPU memory and loads the old CPU-side weights
        2. For each training-model parameter, call _share_cuda_() to obtain a CUDA IPC handle (very fast, ~10ms)
        3. Execute llm.apply_model(fn) inside the EngineCore subprocess:
             open IPC handles → rebuild GPU tensors → directly overwrite vLLM parameters (GPU→GPU, no CPU round trip)

    Name mapping (Qwen3.5 multimodal):
        Training model (AutoModelForCausalLM, LLM part only):  model.xxx / lm_head.weight
        vLLM model (includes the vision encoder)               :  language_model.model.xxx / language_model.lm_head.weight
        Mapping rule: vllm_name = "language_model." + train_name
        The vLLM vision-encoder parameters (visual.*) are absent from the training model and stay unchanged in vLLM.

    Required environment variable:
        VLLM_ALLOW_INSECURE_SERIALIZATION=1  (allows cloudpickle to serialize closure functions)
    """
    llm.wake_up()
    torch.cuda.synchronize()
    if sync_weights:
        _copy_weights(training_model, llm)
        torch.cuda.synchronize()
        logger.info("[vLLM] Weights synced from training model.")
    else:
        logger.info("[vLLM] Woke up (skipping weight sync, params unchanged).")


def _detect_vllm_prefix(llm) -> str:
    """Detect the parameter-name prefix used by the vLLM model.

    The LLM portion of multimodal models (such as Qwen3.5) uses the 'language_model.' prefix in vLLM;
    pure text models (such as Qwen3) do not. Detect this by scanning parameter names.
    """
    def _has_language_model_prefix(vllm_model):
        for name, _ in vllm_model.named_parameters():
            if name.startswith("language_model."):
                return True
        return False

    has_prefix = llm.apply_model(_has_language_model_prefix)[0]
    if has_prefix:
        return "language_model."
    return ""


def _copy_weights(training_model: nn.Module, llm) -> None:
    """
    Copy training-model parameters into the vLLM model with CUDA IPC handles and apply_model.

    Procedure:
        1. For each training-model parameter, call untyped_storage()._share_cuda_() to obtain an IPC handle
           (metadata only, no data copy, about ~10ms)
        2. llm.apply_model() serializes the function with cloudpickle and sends it to the EngineCore subprocess
           (the ipc_data dict contains only tiny handle tuples)
        3. Inside the subprocess: rebuild storage with _new_shared_cuda(*handle) → as_strided → copy_
           (pure GPU→GPU copy, no CPU round trip)

    Name mapping:
        Automatically detect whether the vLLM model uses the 'language_model.' prefix:
          multimodal models (Qwen3.5): vllm_name = "language_model." + train_name
          pure text models (Qwen3):    vllm_name = train_name

    Fused-parameter handling (vLLM MergedColumnParallelLinear):
        vLLM merges multiple small parameters into a larger one:
          gate_proj + up_proj → gate_up_proj (MLP, cat along dim=0)
          q_proj + k_proj + v_proj → qkv_proj (standard attention)
          in_proj_qkv + in_proj_z → in_proj_qkvz (linear attention)
          in_proj_b + in_proj_a → in_proj_ba (linear attention)
        For fused parameters, rebuild each component inside EngineCore, concatenate them, and copy the result.
    """
    src = training_model.module if hasattr(training_model, "module") else training_model

    # Automatically detect the vLLM parameter-name prefix.
    vllm_prefix = _detect_vllm_prefix(llm)
    logger.info(f"[vLLM] Detected parameter prefix: {vllm_prefix!r}")

    # Build IPC handles.
    ipc_data: Dict = {}
    for name, param in src.named_parameters():
        vllm_name = vllm_prefix + name
        data = param.data.contiguous()
        handle = data.untyped_storage()._share_cuda_()
        ipc_data[vllm_name] = (handle, data.dtype, tuple(data.shape), data.stride(), data.storage_offset())

    def _update(vllm_model) -> tuple:
        import torch
        n_direct = 0
        n_fused = 0
        n_skipped = 0

        def _ipc_to_tensor(key):
            handle, dtype, shape, stride, offset = ipc_data[key]
            storage = torch.UntypedStorage._new_shared_cuda(*handle)
            n_elem = storage.nbytes() // torch.empty([], dtype=dtype).element_size()
            flat = torch.empty(n_elem, dtype=dtype, device="cuda")
            flat.set_(storage)
            return flat.as_strided(shape, stride, offset)

        for name, vp in vllm_model.named_parameters():
            # 1. Direct match.
            if name in ipc_data:
                vp.data.copy_(_ipc_to_tensor(name))
                n_direct += 1
                continue

            # 2. Try fused-parameter rules.
            matched = False
            for unfused_suffixes, fused_suffix in _FUSION_RULES:
                if not name.endswith(fused_suffix):
                    continue
                prefix = name[:-len(fused_suffix)]
                unfused_keys = [prefix + s for s in unfused_suffixes]
                if all(k in ipc_data for k in unfused_keys):
                    parts = [_ipc_to_tensor(k) for k in unfused_keys]
                    fused_tensor = torch.cat(parts, dim=0)
                    vp.data.copy_(fused_tensor)
                    n_fused += 1
                    matched = True
                    break

            if not matched:
                n_skipped += 1

        return n_direct, n_fused, n_skipped

    n_direct, n_fused, n_skipped = llm.apply_model(_update)[0]

    if n_direct + n_fused == 0:
        raise RuntimeError(
            f"[vLLM] Weight sync failed: 0 parameters copied "
            f"(direct={n_direct}, fused={n_fused}, skipped={n_skipped}). "
            "Check parameter name mapping in _copy_weights()."
        )
    logger.info(
        f"[vLLM] Weight sync: {n_direct} direct, {n_fused} fused, "
        f"{n_skipped} skipped (visual encoder)."
    )


def verify_vllm_weights(llm, training_model: nn.Module) -> None:
    """
    [Debug] Verify that vLLM parameters match the training-model parameters.

    Automatically detect the vLLM parameter prefix (language_model. or empty),
    then compare checksums only for matching parameters; unmatched ones (for example due to QKV fusion) are reported separately.

    Raises:
        RuntimeError: raised when parameters do not match, which means weight sync failed and the training result is not trustworthy.
    """
    src = training_model.module if hasattr(training_model, "module") else training_model

    # Automatically detect the prefix.
    vllm_prefix = _detect_vllm_prefix(llm)

    # Step 1: fetch parameter names and sums from the vLLM subprocess.
    # Note: bfloat16 tensor sums may differ numerically because GPU parallel reduction order is not deterministic, even when data are bit-exact.
    # Convert to float32 before summing to improve precision.
    def _get_vllm_param_sums(vllm_model):
        param_sums = {}
        n_skipped = 0
        for name, vp in vllm_model.named_parameters():
            if vllm_prefix and not name.startswith(vllm_prefix):
                n_skipped += 1
                continue
            param_sums[name] = vp.data.float().sum().item()
        return param_sums, n_skipped

    vllm_param_sums, n_vllm_skipped = llm.apply_model(_get_vllm_param_sums)[0]

    # Step 2: compare training-model parameters one by one.
    # (a) Direct matches.
    n_direct = 0
    max_diff = 0.0
    matched_vllm_keys = set()
    fused_train_names = set()  # Training-parameter names that participate in fusion.

    for name, param in src.named_parameters():
        vllm_name = vllm_prefix + name
        if vllm_name in vllm_param_sums:
            train_sum = param.data.float().sum().item()
            diff = abs(train_sum - vllm_param_sums[vllm_name])
            max_diff = max(max_diff, diff)
            n_direct += 1
            matched_vllm_keys.add(vllm_name)

    # (b) Fused matches: compare the sum of concatenated training parameters against the fused vLLM parameter sum.
    train_params = dict(src.named_parameters())
    n_fused_verified = 0
    for vllm_name, vllm_sum in vllm_param_sums.items():
        if vllm_name in matched_vllm_keys:
            continue
        for unfused_suffixes, fused_suffix in _FUSION_RULES:
            if not vllm_name.endswith(fused_suffix):
                continue
            prefix = vllm_name[:-len(fused_suffix)]
            # Strip the vLLM prefix back to the training-model naming scheme.
            train_prefix = prefix[len(vllm_prefix):]
            train_names = [train_prefix + s for s in unfused_suffixes]
            if all(n in train_params for n in train_names):
                fused_sum = sum(train_params[n].data.float().sum().item() for n in train_names)
                diff = abs(fused_sum - vllm_sum)
                max_diff = max(max_diff, diff)
                n_fused_verified += 1
                matched_vllm_keys.add(vllm_name)
                fused_train_names.update(train_names)
            break

    n_train_only = 0
    train_only_names = []
    for name, _ in src.named_parameters():
        vllm_name = vllm_prefix + name
        if vllm_name not in matched_vllm_keys and name not in fused_train_names:
            n_train_only += 1
            if len(train_only_names) < 5:
                train_only_names.append(name)

    n_vllm_only = len(vllm_param_sums) - len(matched_vllm_keys)

    match = max_diff < 0.1
    logger.info(
        f"[DEBUG vLLM] weight verify: "
        f"direct={n_direct}, fused={n_fused_verified}, "
        f"max_diff={max_diff:.6f} "
        f"{'MATCH' if match else 'MISMATCH !!!'} | "
        f"train_only={n_train_only}, vllm_only={n_vllm_only}, "
        f"visual_skipped={n_vllm_skipped}"
    )
    if train_only_names:
        logger.info(
            f"[DEBUG vLLM] train-only params (not in vLLM): "
            f"{train_only_names}{'...' if n_train_only > 5 else ''}"
        )

    if not match:
        raise RuntimeError(
            f"[FATAL] vLLM weight sync verification failed! "
            f"max_diff={max_diff:.6f}. "
            f"direct={n_direct}, fused={n_fused_verified}, "
            f"train_only={n_train_only}, vllm_only={n_vllm_only}. "
            "The training model and the vLLM model are inconsistent, so the "
            "rollout result is not trustworthy."
        )
