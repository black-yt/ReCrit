"""
Validate _find_bridge() by extracting the bridge token IDs between two turns.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rollout import _find_bridge


def test_bridge_alignment(model_path: str) -> bool:
    """Check bridge extraction across multiple turn-1 / critic-message combinations."""
    from transformers import AutoTokenizer

    print(f"Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    print(f"<|im_end|> token id: {im_end_id}")

    system_prompt = "You are a helpful assistant."
    question = "What is the capital of France?"
    turn1_texts = [
        "The capital of France is Paris.",
        "Paris is the capital of France. It is a major European city.",
    ]
    critic_messages = [
        "Your answer seems to have some issues. Could you reconsider it carefully?",
        "I think your answer is on the right track! Could you refine and improve it a bit more?",
    ]

    def apply_template(messages):
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors=None,
        )
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        return [int(x) for x in ids]

    turn1_messages_base = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question},
    ]

    passed = 0
    failed = 0

    for turn1_text in turn1_texts:
        for critic_msg in critic_messages:
            turn1_prompt_ids = apply_template(turn1_messages_base)
            turn2_messages = turn1_messages_base + [
                {"role": "assistant", "content": turn1_text},
                {"role": "user",      "content": critic_msg},
            ]
            turn2_prompt_ids = apply_template(turn2_messages)

            bridge_ids = _find_bridge(turn2_prompt_ids, turn1_prompt_ids, im_end_id)
            bridge_text = tokenizer.decode(bridge_ids)

            critic_in_bridge = critic_msg in bridge_text
            bridge_starts_with_newline = bridge_text.startswith("\n")
            bridge_ends_with_gen_prompt = (
                "assistant" in bridge_text.split(critic_msg)[-1]
                if critic_msg in bridge_text else False
            )
            turn1_not_in_bridge = turn1_text not in bridge_text

            ok = all([
                critic_in_bridge,
                bridge_starts_with_newline,
                bridge_ends_with_gen_prompt,
                turn1_not_in_bridge,
            ])

            status = "PASS" if ok else "FAIL"
            print(f"[{status}] turn1='{turn1_text[:40]}' critic='{critic_msg[:30]}'")
            print(f"  turn1_prompt len={len(turn1_prompt_ids)}, "
                  f"turn2_prompt len={len(turn2_prompt_ids)}, "
                  f"bridge len={len(bridge_ids)}")
            print(f"  bridge_text={repr(bridge_text[:80])}")
            if not ok:
                print(f"  checks: critic={critic_in_bridge}, newline={bridge_starts_with_newline}, "
                      f"gen_prompt={bridge_ends_with_gen_prompt}, no_turn1={turn1_not_in_bridge}")

            if ok:
                passed += 1
            else:
                failed += 1

    total = passed + failed
    print(f"\nResults: {passed}/{total} passed, {failed}/{total} failed")

    if failed == 0:
        print("OK: _find_bridge correctly extracts bridge for all test cases.")
        return True
    else:
        print("FAIL: _find_bridge extraction has errors.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate the correctness of bridge-token extraction in _find_bridge()"
    )
    parser.add_argument("--model_path", required=True, help="HuggingFace model path")
    args = parser.parse_args()

    ok = test_bridge_alignment(args.model_path)
    sys.exit(0 if ok else 1)
