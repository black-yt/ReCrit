"""
Validate the logits_to_keep optimization used in compute_policy_logps().
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_logps_full(model, input_ids, labels, attention_mask, temperature=1.0):
    """Full computation that does not skip the prompt region."""
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    logits = outputs.logits  
    if temperature != 1.0:
        logits = logits / temperature

    shift_logits = logits[:, :-1, :]       
    shift_targets = input_ids[:, 1:]        
    shift_labels = labels[:, 1:]            

    B, L_minus_1, V = shift_logits.shape
    per_token_logps = -F.cross_entropy(
        shift_logits.float().view(-1, V),
        shift_targets.clamp(min=0).view(-1),
        reduction="none",
    ).view(B, L_minus_1)

    completion_mask = (shift_labels != -100).long()
    return per_token_logps, completion_mask


def compute_logps_optimized(model, input_ids, labels, attention_mask, temperature=1.0):
    """Optimized computation matching trainer.py::compute_policy_logps."""
    logits_to_keep = (
        labels.shape[-1] - (torch.ne(labels, -100).int().argmax(-1))
    ).max().item()

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    logits = outputs.logits  
    logits = logits[:, -(logits_to_keep + 1):, :]  

    if temperature != 1.0:
        logits = logits / temperature

    shift_logits = logits[:, :-1, :]                 
    shift_targets = input_ids[:, -logits_to_keep:]   
    shift_labels = labels[:, -logits_to_keep:]       

    B, K, V = shift_logits.shape
    per_token_logps = -F.cross_entropy(
        shift_logits.float().view(-1, V),
        shift_targets.clamp(min=0).view(-1),
        reduction="none",
    ).view(B, K)

    completion_mask = (shift_labels != -100).long()
    return per_token_logps, completion_mask, logits_to_keep


def make_test_batch(tokenizer, device):
    """Construct a toy batch with prompts and completions of different lengths."""
    prompt1 = "What is 2+2?"
    completion1 = "The answer is 4. This is a basic arithmetic problem."

    prompt2 = "Please explain the concept of machine learning in simple terms for a beginner."
    completion2 = "Machine learning is a type of AI."

    
    p1_ids = tokenizer.encode(prompt1, add_special_tokens=False)
    c1_ids = tokenizer.encode(completion1, add_special_tokens=False)
    p2_ids = tokenizer.encode(prompt2, add_special_tokens=False)
    c2_ids = tokenizer.encode(completion2, add_special_tokens=False)

    seq1 = p1_ids + c1_ids
    lab1 = [-100] * len(p1_ids) + c1_ids

    seq2 = p2_ids + c2_ids
    lab2 = [-100] * len(p2_ids) + c2_ids

    
    max_len = max(len(seq1), len(seq2))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def pad(seq, length, val):
        return seq + [val] * (length - len(seq))

    input_ids = torch.tensor([
        pad(seq1, max_len, pad_id),
        pad(seq2, max_len, pad_id),
    ], dtype=torch.long, device=device)

    labels = torch.tensor([
        pad(lab1, max_len, -100),
        pad(lab2, max_len, -100),
    ], dtype=torch.long, device=device)

    attention_mask = (input_ids != pad_id).long()

    print(f"Sequence 1: prompt={len(p1_ids)} tokens, completion={len(c1_ids)} tokens")
    print(f"Sequence 2: prompt={len(p2_ids)} tokens, completion={len(c2_ids)} tokens")
    print(f"Padded length: {max_len}")

    return input_ids, labels, attention_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()

    input_ids, labels, attention_mask = make_test_batch(tokenizer, device)

    with torch.no_grad():
        full_logps, full_mask = compute_logps_full(
            model, input_ids, labels, attention_mask, args.temperature
        )
        opt_logps, opt_mask, K = compute_logps_optimized(
            model, input_ids, labels, attention_mask, args.temperature
        )

    print(f"\nlogits_to_keep = {K}")
    print(f"Full per_token_logps shape: {full_logps.shape}")
    print(f"Optimized per_token_logps shape: {opt_logps.shape}")

    full_tail = full_logps[:, -K:]
    full_mask_tail = full_mask[:, -K:]

    mask_match = (full_mask_tail == opt_mask).all().item()
    print(f"\ncompletion_mask match: {mask_match}")

    comp_positions = opt_mask.bool()
    full_comp = full_tail[comp_positions]
    opt_comp = opt_logps[comp_positions]

    max_diff = (full_comp - opt_comp).abs().max().item()
    mean_diff = (full_comp - opt_comp).abs().mean().item()

    print(f"Maximum logps difference on completion positions: {max_diff:.2e}")
    print(f"Mean logps difference on completion positions: {mean_diff:.2e}")

    skipped_mask = full_mask[:, :-K] if full_logps.shape[1] > K else torch.tensor([])
    if skipped_mask.numel() > 0:
        all_prompt = (skipped_mask == 0).all().item()
        print(f"Skipped region is all prompt tokens (mask=0): {all_prompt}")
        if not all_prompt:
            n_nonzero = skipped_mask.sum().item()
            print(f"  Warning: skipped region contains {n_nonzero} non-zero mask position(s)!")
    else:
        print("No skipped region (logits_to_keep covers the full sequence)")

    tol = 1e-4
    passed = mask_match and max_diff < tol
    print(f"\n{'PASSED' if passed else 'FAILED'} (tolerance={tol})")
    if not passed:
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
