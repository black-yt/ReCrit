"""
Compare asynchronous multi-turn rollout based on engine.step() against
synchronous multi-turn rollout based on llm.generate().

Experiment design:
  - 2 prompts: SHORT (math) and LONG (essay generation)
  - 2 dialogue turns (turn 0 + critic + turn 1)
  - compare the completion timestamp of every sample at every turn

Expected result:
  - synchronous: SHORT and LONG finish turn 0 at the same time
  - asynchronous: SHORT finishes much earlier and does not wait for LONG
"""

import argparse
import time
import os
import sys

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def run_sync(llm, tokenizer, prompts, sampling_params, num_turns=2):
    """Synchronous baseline: llm.generate() is called once per turn for the full batch."""
    from copy import deepcopy

    N = len(prompts)
    messages_batch = [deepcopy(p["messages"]) for p in prompts]
    turn_finish_times = [[] for _ in range(N)]

    t0 = time.time()

    for turn_idx in range(num_turns):
        token_prompts = []
        for msgs in messages_batch:
            ids = tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True, return_tensors=None,
            )
            if not isinstance(ids, list):
                ids = ids["input_ids"]
            token_prompts.append({"prompt_token_ids": [int(x) for x in ids]})

        outputs = llm.generate(token_prompts, sampling_params=sampling_params, use_tqdm=False)
        batch_done_time = time.time() - t0

        for i, out in enumerate(outputs):
            n_tokens = len(out.outputs[0].token_ids)
            turn_finish_times[i].append(batch_done_time)
            print(f"  [SYNC]  sample={i} turn={turn_idx} "
                  f"tokens={n_tokens:4d}  time={batch_done_time:.2f}s")

            if turn_idx < num_turns - 1:
                messages_batch[i] = messages_batch[i] + [
                    {"role": "assistant", "content": out.outputs[0].text},
                    {"role": "user", "content": "Please reconsider your answer carefully."},
                ]

    total_time = time.time() - t0
    return total_time, turn_finish_times


def run_async(llm, tokenizer, prompts, sampling_params, num_turns=2):
    """Asynchronous mode: add_request() + step() advances each sample independently."""
    engine = llm.llm_engine
    N = len(prompts)

    states = [{"messages": list(p["messages"]), "turn": 0, "done": False} for p in prompts]
    pending = {}
    req_counter = 0
    turn_finish_times = [[] for _ in range(N)]

    def _submit(idx):
        nonlocal req_counter
        s = states[idx]
        ids = tokenizer.apply_chat_template(
            s["messages"], tokenize=True, add_generation_prompt=True, return_tensors=None,
        )
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        req_id = f"r{req_counter}"
        req_counter += 1
        engine.add_request(
            request_id=req_id,
            prompt={"prompt_token_ids": [int(x) for x in ids]},
            params=sampling_params,
        )
        pending[req_id] = idx

    t0 = time.time()
    for i in range(N):
        _submit(i)

    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        for output in step_outputs:
            if not output.finished:
                continue
            req_id = output.request_id
            if req_id not in pending:
                continue
            idx = pending.pop(req_id)
            s = states[idx]

            n_tokens = len(output.outputs[0].token_ids)
            s["turn"] += 1
            elapsed = time.time() - t0
            turn_finish_times[idx].append(elapsed)

            print(f"  [ASYNC] sample={idx} turn={s['turn']-1} "
                  f"tokens={n_tokens:4d}  time={elapsed:.2f}s")

            if s["turn"] >= num_turns:
                s["done"] = True
                continue

            s["messages"] = s["messages"] + [
                {"role": "assistant", "content": output.outputs[0].text},
                {"role": "user", "content": "Please reconsider your answer carefully."},
            ]
            _submit(idx)

    total_time = time.time() - t0
    return total_time, turn_finish_times


def main():
    parser = argparse.ArgumentParser(
        description="Compare bubble time between asynchronous and synchronous multi-turn generation"
    )
    parser.add_argument("--model_path", required=True, help="HuggingFace model path")
    parser.add_argument("--num_turns", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"Loading model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    prompts = [
        {
            "name": "SHORT",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 1+1? Answer in one word."},
            ],
        },
        {
            "name": "LONG",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content":
                    "Write a comprehensive 1000-word essay about the history of "
                    "artificial intelligence, covering major milestones from the "
                    "1950s to 2025. Include specific dates, researchers, and breakthroughs."},
            ],
        },
    ]

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=args.max_new_tokens,
        logprobs=1,
    )

    SEP = "=" * 70

    print(f"\n{SEP}")
    print("Experiment 1: synchronous mode (llm.generate)")
    print(SEP)

    llm_sync = LLM(
        model=args.model_path, tokenizer=args.model_path,
        dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True, max_model_len=8192,
    )
    sync_time, sync_turns = run_sync(
        llm_sync, tokenizer, prompts, sampling_params, num_turns=args.num_turns,
    )
    del llm_sync
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()

    print(f"\n{SEP}")
    print("Experiment 2: asynchronous mode (engine.step)")
    print(SEP)

    llm_async = LLM(
        model=args.model_path, tokenizer=args.model_path,
        dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True, max_model_len=8192,
    )
    async_time, async_turns = run_async(
        llm_async, tokenizer, prompts, sampling_params, num_turns=args.num_turns,
    )
    del llm_async
    gc.collect(); torch.cuda.empty_cache()

    print(f"\n{SEP}")
    print("Comparison")
    print(SEP)

    print(f"\n{'':4s}{'':10s}  {'sync':>10s}  {'async':>10s}")
    print(f"{'':4s}{'-'*10}  {'-'*10}  {'-'*10}")
    for i, p in enumerate(prompts):
        name = p["name"]
        for t in range(args.num_turns):
            st = sync_turns[i][t] if t < len(sync_turns[i]) else 0
            at = async_turns[i][t] if t < len(async_turns[i]) else 0
            print(f"  {name:8s} T{t}  {st:8.2f}s  {at:8.2f}s")

    print(f"\n  {'TOTAL':8s}      {sync_time:8.2f}s  {async_time:8.2f}s")
    speedup = sync_time / async_time if async_time > 0 else float("inf")
    print(f"  {'SPEEDUP':8s}      {'':8s}  {speedup:8.2f}x")

    bubble_sync = sum(
        max(sync_turns[j][t] for j in range(len(prompts))) -
        min(sync_turns[j][t] for j in range(len(prompts)))
        for t in range(args.num_turns)
    )
    print(f"\n  Synchronous bubble time (sum of max-min across turns): {bubble_sync:.2f}s")
    print(f"  SHORT T0->T1 gap in async mode: ", end="")
    if len(async_turns[0]) >= 2:
        gap = async_turns[0][1] - async_turns[0][0]
        print(f"{gap:.2f}s")
    else:
        print("N/A")

    print(f"\n  Expected behavior:")
    print("    Sync: SHORT and LONG finish T0 at the same time (SHORT waits for LONG).")
    print("    Async: SHORT finishes T0 much earlier than LONG (independent progress, no bubble).")
    print()


if __name__ == "__main__":
    main()
