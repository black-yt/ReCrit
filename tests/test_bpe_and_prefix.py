"""
Test BPE consistency and the correctness of _detect_generation_prefix().

Checks:
  1. Whether prompt_ids[0] + resp_ids[0] matches the prefix of prompt_ids[1]
  2. Whether _detect_generation_prefix correctly extracts any extra generation suffix
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

SEP = "=" * 60


def test_detect_generation_prefix(tokenizer, system_prompt="You are a helpful assistant."):
    """Validate _detect_generation_prefix() on the current tokenizer."""
    from rollout import _detect_generation_prefix

    print(f"\n{SEP}")
    print("1. _detect_generation_prefix test")
    print(SEP)

    prefix = _detect_generation_prefix(tokenizer, system_prompt)
    print(f"  Detected generation prefix: {prefix!r}")

    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": "test"})

    with_gen = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    without_gen = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    gen_prompt = with_gen[len(without_gen):]
    print(f"  Full generation prompt: {gen_prompt!r}")

    if prefix:
        assert gen_prompt.endswith(prefix), (
            f"prefix {prefix!r} is not at the end of gen_prompt {gen_prompt!r}"
        )
        print("  Check: prefix is correctly located at the end of the generation prompt")
    else:
        print("  Check: no extra prefix (nothing beyond the standard assistant header)")

    for sp in ["", "You are a math tutor.", None]:
        p = _detect_generation_prefix(tokenizer, sp or "")
        label = repr(sp) if sp is not None else "None"
        print(f"  system_prompt={label:40s} -> prefix={p!r}")

    if prefix:
        with_gen_ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        without_gen_ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        print(f"  Prefix token IDs: {prefix_ids}")
        print(f"  with_gen trailing IDs: {with_gen_ids[-len(prefix_ids):]}")

        if with_gen_ids[-len(prefix_ids):] == prefix_ids:
            print("  Token-level check: PASS")
        else:
            print("  Token-level check: MISMATCH (possibly caused by BPE boundary effects)")
            gen_ids = with_gen_ids[len(without_gen_ids):]
            print(f"  Generation prompt IDs: {gen_ids}")
            prefix_decoded = tokenizer.decode(gen_ids[-len(prefix_ids):])
            print(f"  Decoded suffix: {prefix_decoded!r}")

    print(f"  _detect_generation_prefix: PASS")
    return True


def test_bpe_consistency(tokenizer, system_prompt="You are a helpful assistant."):
    """Compare concatenation-based tokenization against whole-sequence tokenization."""
    print(f"\n{SEP}")
    print("2. BPE consistency test (_find_bridge prerequisite)")
    print(SEP)

    from rollout import _find_bridge, _detect_generation_prefix

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    gen_prefix = _detect_generation_prefix(tokenizer, system_prompt)
    print(f"  im_end_id: {im_end_id}")
    print(f"  gen_prefix: {gen_prefix!r}")

    test_responses = [
        ("4", "Your answer seems wrong. Please reconsider."),
        ("Let me think step by step.\n2 + 2 = 4\n</think>\n4",
         "I'm not convinced. Could you try again?"),
        ("This is a math problem.\nThe answer is 42.\n</think>\n42",
         "Perhaps double-check your work?"),
        ("Result: x=3.14\n</think>\n3.14",
         "Are you sure about the precision?"),
    ]

    question = "What is 2+2?"
    all_pass = True

    for case_idx, (seq_text, critic_msg) in enumerate(test_responses):
        print(f"\n  Case {case_idx}: seq_text={seq_text[:50]!r}...")

        full_resp = gen_prefix + seq_text if gen_prefix else seq_text

        msgs_t0 = []
        if system_prompt:
            msgs_t0.append({"role": "system", "content": system_prompt})
        msgs_t0.append({"role": "user", "content": question})

        prompt_ids_0 = tokenizer.apply_chat_template(
            msgs_t0, tokenize=True, add_generation_prompt=True, return_tensors=None,
        )

        resp_ids_0 = tokenizer.encode(seq_text, add_special_tokens=False)

        msgs_t1 = list(msgs_t0) + [
            {"role": "assistant", "content": full_resp},
            {"role": "user", "content": critic_msg},
        ]
        prompt_ids_1 = tokenizer.apply_chat_template(
            msgs_t1, tokenize=True, add_generation_prompt=True, return_tensors=None,
        )

        concat = list(prompt_ids_0) + resp_ids_0 + [im_end_id]
        match_len = len(concat)

        if len(prompt_ids_1) < match_len:
            print(f"    FAIL: prompt_ids_1 ({len(prompt_ids_1)}) is shorter than concat ({match_len})")
            all_pass = False
            continue

        prefix_match = prompt_ids_1[:match_len] == concat
        if prefix_match:
            print(f"    BPE prefix match: PASS (len={match_len})")
        else:
            first_diff = None
            for i in range(min(match_len, len(prompt_ids_1))):
                if i >= len(concat) or concat[i] != prompt_ids_1[i]:
                    first_diff = i
                    break

            print("    BPE prefix match: FAIL")
            print(f"    First mismatch position: {first_diff}")
            if first_diff is not None:
                ctx = 3
                s = max(0, first_diff - ctx)
                e = min(match_len, first_diff + ctx + 1)
                concat_ctx = concat[s:e]
                prompt1_ctx = prompt_ids_1[s:e]
                print(f"    concat  [{s}:{e}]: {concat_ctx}")
                print(f"    prompt1 [{s}:{e}]: {prompt1_ctx}")
                print(f"    concat  decoded: {tokenizer.decode(concat_ctx)!r}")
                print(f"    prompt1 decoded: {tokenizer.decode(prompt1_ctx)!r}")
            all_pass = False

        bridge = _find_bridge(prompt_ids_1, prompt_ids_0, im_end_id)
        if bridge:
            bridge_text = tokenizer.decode(bridge)
            print(f"    bridge ({len(bridge)} tokens): {bridge_text[:80]!r}...")

            if critic_msg[:20] in bridge_text:
                print("    Bridge contains critic message: PASS")
            else:
                print("    Bridge contains critic message: FAIL")
                print(f"    Expected to contain: {critic_msg[:20]!r}")
                all_pass = False

            if gen_prefix and gen_prefix in bridge_text:
                print("    Bridge ends with gen_prefix: PASS")
            elif not gen_prefix:
                print("    Bridge suffix: no gen_prefix present (as expected)")
            else:
                print("    Bridge ends with gen_prefix: FAIL")
                all_pass = False
        else:
            print("    Bridge: empty (_find_bridge failed)")
            all_pass = False

    return all_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"Model: {args.model_path}")
    print(f"Tokenizer class: {type(tokenizer).__name__}")

    pass1 = test_detect_generation_prefix(tokenizer)
    pass2 = test_bpe_consistency(tokenizer)

    print(f"\n{SEP}")
    print(f"Summary: prefix={'PASS' if pass1 else 'FAIL'}, "
          f"BPE={'PASS' if pass2 else 'FAIL'}")
    print(SEP)

    if not (pass1 and pass2):
        sys.exit(1)


if __name__ == "__main__":
    main()
