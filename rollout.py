""

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
    ""
    if getattr(config, "critic_prompt_mode", "mixed") == "eval_fixed":
        return "eval_fixed", config.critic_prompt_text
    attitude = random.choice(_ATTITUDES)
    critic_msg = random.choice(ATTITUDE_TEMPLATES[attitude])
    return attitude, critic_msg










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






def _tokenize_messages(tokenizer, messages: List[Dict]) -> List[int]:
    ""
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors=None,
    )
    if not isinstance(ids, list):
        ids = ids["input_ids"]
    return [int(x) for x in ids]


def _detect_generation_prefix(tokenizer, system_prompt: str = "") -> str:
    ""
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

    gen_prompt = with_gen[len(without_gen):]




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
    ""
    from vllm import SamplingParams





    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_new_tokens,
        stop_token_ids=_get_stop_token_ids(tokenizer),
    )





    _gen_prefix = _detect_generation_prefix(tokenizer, config.system_prompt)
    _dbg_tag = f"[rank={rank}/{world_size}] [{step_label}]" if step_label else f"[rank={rank}/{world_size}]"

    if _gen_prefix and debug:
        logger.info(f"{_dbg_tag} generation prefix detected: {_gen_prefix!r}")

    n_prompts = len(prompts)
    n_samples = n_prompts * num_generations
    completion_ratio = config.completion_ratio
    n_complete_target = math.ceil(n_samples * completion_ratio)


    min_turns_for_early_stop = 1 if num_turns == 2 else 2


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
    pending: Dict[str, int] = {}
    req_counter = 0
    stop_flag = False

    def _submit(sample_idx: int):
        ""
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


    _debug_t0 = time.monotonic() if debug else 0.0
    _debug_events: List[tuple] = []  # (elapsed, sample_idx, n_done)


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


            seq = output.outputs[0]

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



            if n_done >= num_turns:
                state["done"] = True
                continue


            if stop_flag and n_done >= min_turns_for_early_stop:
                state["done"] = True
                logger.debug(f"[Rollout] batch={idx_in_batch} gen={idx_in_group}: stopped early (flag set)")
                continue


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

                    state["done"] = True
                    continue


            attitude, critic_msg = _sample_critic_message(config)
            state["attitudes"].append(attitude)
            state["critic_messages"].append(critic_msg)
            state["messages"] = state["messages"] + [
                {"role": "assistant", "content": resp_text},
                {"role": "user",      "content": critic_msg},
            ]
            _submit(idx)


    if debug and _debug_events:


        _debug_events.sort()
        max_turn_gap = 0
        has_overlap = False
        for j in range(len(_debug_events)):

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

            state["prompt_ids"].append(_tokenize_messages(tokenizer, next_messages))


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
    ""
    stop_ids = []
    for tok in ["<|im_end|>", "<|endoftext|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid and tid != tokenizer.unk_token_id:
            stop_ids.append(tid)
    if tokenizer.eos_token_id and tokenizer.eos_token_id not in stop_ids:
        stop_ids.append(tokenizer.eos_token_id)
    return stop_ids





def _find_bridge(
    next_prompt_ids: List[int],
    prev_prompt_ids: List[int],
    im_end_id: int,
) -> List[int]:
    ""
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
    ""
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
        prompt_ids_list = [list(p) for p in r["prompt_ids"]]
        resp_ids_list   = [list(t) for t in r["token_ids"]]
        per_turn_weights = (
            list(turn_loss_weights)
            if turn_loss_weights
            else [1.0] * len(resp_ids_list)
        )


        for tids in resp_ids_list:
            if im_end_id and (not tids or tids[-1] != im_end_id):
                tids.append(im_end_id)


        bridge_ids_list = [
            _find_bridge(prompt_ids_list[turn_idx + 1], prompt_ids_list[turn_idx], im_end_id)
            for turn_idx in range(len(resp_ids_list) - 1)
        ]


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


        full_ids: List[int] = list(prefix_ids)
        labels:   List[int] = [-100] * len(prefix_ids)
        loss_weights: List[float] = [0.0] * len(prefix_ids)

        for turn_idx, resp_ids in enumerate(resp_ids_list):
            full_ids.extend(resp_ids)
            labels.extend(resp_ids)
            loss_weights.extend([per_turn_weights[turn_idx]] * len(resp_ids))

            if turn_idx < len(bridge_ids_list):
                bridge = bridge_ids_list[turn_idx]
                full_ids.extend(bridge)
                labels.extend([-100] * len(bridge))
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






def vllm_sleep(llm, level: int = 1) -> None:
    ""
    llm.sleep(level=level)
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"[vLLM] Sleeping (level={level}), GPU memory freed.")


def vllm_wake_and_sync(llm, training_model: nn.Module, sync_weights: bool = True) -> None:
    ""
    llm.wake_up()
    torch.cuda.synchronize()
    if sync_weights:
        _copy_weights(training_model, llm)
        torch.cuda.synchronize()
        logger.info("[vLLM] Weights synced from training model.")
    else:
        logger.info("[vLLM] Woke up (skipping weight sync, params unchanged).")


def _detect_vllm_prefix(llm) -> str:
    ""
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
    ""
    src = training_model.module if hasattr(training_model, "module") else training_model


    vllm_prefix = _detect_vllm_prefix(llm)
    logger.info(f"[vLLM] Detected parameter prefix: {vllm_prefix!r}")


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

            if name in ipc_data:
                vp.data.copy_(_ipc_to_tensor(name))
                n_direct += 1
                continue


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
    ""
    src = training_model.module if hasattr(training_model, "module") else training_model


    vllm_prefix = _detect_vllm_prefix(llm)




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



    n_direct = 0
    max_diff = 0.0
    matched_vllm_keys = set()
    fused_train_names = set()

    for name, param in src.named_parameters():
        vllm_name = vllm_prefix + name
        if vllm_name in vllm_param_sums:
            train_sum = param.data.float().sum().item()
            diff = abs(train_sum - vllm_param_sums[vllm_name])
            max_diff = max(max_diff, diff)
            n_direct += 1
            matched_vllm_keys.add(vllm_name)


    train_params = dict(src.named_parameters())
    n_fused_verified = 0
    for vllm_name, vllm_sum in vllm_param_sums.items():
        if vllm_name in matched_vllm_keys:
            continue
        for unfused_suffixes, fused_suffix in _FUSION_RULES:
            if not vllm_name.endswith(fused_suffix):
                continue
            prefix = vllm_name[:-len(fused_suffix)]

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
