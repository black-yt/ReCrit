"""
Test how vLLM reports logprobs for stop tokens.

Specifically checks whether, with stop_token_ids=[im_end_id], vLLM includes the
stop token in both seq.token_ids and seq.logprobs.
"""

import argparse
import os
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--vllm_max_model_len", type=int, default=4096)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_id = tokenizer.eos_token_id
    stop_ids = []
    for tok in ["<|im_end|>", "<|endoftext|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid and tid != tokenizer.unk_token_id:
            stop_ids.append(tid)
    if eos_id and eos_id not in stop_ids:
        stop_ids.append(eos_id)

    print(f"im_end_id = {im_end_id}")
    print(f"eos_token_id = {eos_id}")
    print(f"stop_token_ids = {stop_ids}")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.5,
        max_model_len=args.vllm_max_model_len,
        trust_remote_code=True,
    )

    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2? Answer briefly."},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors=None,
    )
    if not isinstance(prompt_ids, list):
        prompt_ids = prompt_ids["input_ids"]
    prompt_ids = [int(x) for x in prompt_ids]

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=256,
        logprobs=1,
        stop_token_ids=stop_ids,
    )

    engine = llm.llm_engine
    engine.add_request(
        request_id="test_0",
        prompt={"type": "token", "prompt_token_ids": prompt_ids},
        params=sp,
    )

    seq = None
    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        for output in step_outputs:
            if output.finished:
                seq = output.outputs[0]

    if seq is None:
        print("ERROR: no output received")
        return

    token_ids = list(seq.token_ids)
    logprobs = seq.logprobs

    print(f"\n{'='*60}")
    print(f"Generated {len(token_ids)} tokens")
    print(f"Last 5 token_ids: {token_ids[-5:]}")
    print(f"Last 5 tokens decoded: {[tokenizer.decode([t]) for t in token_ids[-5:]]}")
    print(f"logprobs length: {len(logprobs) if logprobs else 'None'}")

    has_stop_in_ids = any(t in stop_ids for t in token_ids[-3:])
    last_is_im_end = token_ids[-1] == im_end_id if token_ids else False

    print(f"\n--- Diagnostics ---")
    print(f"Stop token appears near the end of token_ids: {has_stop_in_ids}")
    print(f"Last token is <|im_end|>: {last_is_im_end}")

    if logprobs and len(logprobs) == len(token_ids):
        print(f"logprobs length == token_ids length: True (both are {len(token_ids)})")
        last_lp = logprobs[-1]
        if last_lp and token_ids[-1] in last_lp:
            lp_val = last_lp[token_ids[-1]].logprob
            print(f"Last token logprob: {lp_val:.4f}")
            print(
                "\nConclusion: vLLM returns the logprob of the stop token, "
                "so the current 0.0 padding in _align_lp is unnecessary."
            )
            print("            No additional fix is needed.")
        else:
            print("Last token logprob: MISSING")
            print("\nConclusion: vLLM returns the stop token but not its logprob; this needs handling.")
    elif logprobs and len(logprobs) < len(token_ids):
        print(f"logprobs length ({len(logprobs)}) < token_ids length ({len(token_ids)})")
        diff = len(token_ids) - len(logprobs)
        print(f"Missing {diff} logprob entry(ies), possibly the stop token.")
        print(
            "\nConclusion: vLLM does not return the stop-token logprob, "
            "and the current implementation pads it with 0.0."
        )
        print("            The IS ratio is biased at the stop token position, but only by 1 token/turn.")
    elif logprobs and len(logprobs) > len(token_ids):
        print(f"logprobs length ({len(logprobs)}) > token_ids length ({len(token_ids)})")
        print("Unexpected state: there are more logprobs than token_ids.")
    else:
        print("logprobs is None or empty")
        print("\nConclusion: vLLM did not return logprobs; check the SamplingParams configuration.")

    print(f"\n--- Detailed logprobs for the last 5 tokens ---")
    start = max(0, len(token_ids) - 5)
    for i in range(start, len(token_ids)):
        tid = token_ids[i]
        tok_text = tokenizer.decode([tid])
        if logprobs and i < len(logprobs) and logprobs[i]:
            if tid in logprobs[i]:
                lp = logprobs[i][tid].logprob
                print(f"  [{i}] id={tid:8d}  logprob={lp:+.4f}  text={tok_text!r}")
            else:
                print(f"  [{i}] id={tid:8d}  logprob=MISSING  text={tok_text!r}")
        else:
            print(f"  [{i}] id={tid:8d}  logprob=N/A      text={tok_text!r}")


if __name__ == "__main__":
    main()
