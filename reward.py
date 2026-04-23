""

import logging
from typing import List, Dict
from collections import Counter, defaultdict

import torch

logger = logging.getLogger(__name__)


try:
    from structai import Judge, multi_thread, prompts
    from structai import parse_think_answer
    HAS_JUDGE = True
except Exception as e:
    logger.warning(f"[Reward] structai unavailable: {e}. All rewards will default to 0.")
    HAS_JUDGE = False


def _default_critic_fields(r: dict) -> None:
    ""
    n_pairs = max(len(r.get("responses", [])) - 1, 0)
    r["correctness"] = [False] * len(r.get("responses", []))
    r["quadrant_pairs"] = ["Boundary"] * n_pairs
    r["critic_reward"] = 0.0


def _create_judges(model_version: str = "gemini-3-flash-preview-nothinking",
                   time_limit: int = 120):
    ""
    if not hasattr(_create_judges, "_judges"):
        _create_judges._judges = (
            Judge(model_version=model_version, time_limit=time_limit, use_tqdm=False),
            Judge(
                model_version=model_version,
                time_limit=time_limit,
                prompt_tmp=prompts["llm_judge_arena"]["prompt_tmp"],
                llm_tags=prompts["llm_judge_arena"]["llm_tags"],
                use_math_verify=False,
                use_tqdm=False,
            ),
        )
    return _create_judges._judges







def _collect_turn_answers(r: dict) -> List[str]:
    ""
    return list(r["responses"])


def _collect_turn_token_lengths(r: dict) -> List[int]:
    ""
    return [len(ids) for ids in r["token_ids"]]






def _is_correct(result: dict, idx: int) -> bool:
    ""
    math_list = result.get("math_verify_list") or []
    llm_list  = result.get("llm_judge_list")  or []

    math_ok = math_list[idx] if idx < len(math_list) else None
    llm_ok  = llm_list[idx]  if idx < len(llm_list)  else None

    return bool(math_ok or llm_ok)


def compute_critic_rewards(
    results: List[Dict],
    w_correction: float = 1.0,
    w_robustness: float = 0.6,
    w_sycophancy: float = 1.0,
    w_boundary: float = 0.1,
    judge_model: str = "gemini-3-flash-preview-nothinking",
    num_turns: int = 2,
) -> None:
    ""
    n = len(results)

    if not HAS_JUDGE or n == 0:
        for r in results:
            _default_critic_fields(r)
        return


    time_limit = num_turns * 60
    judge_close, judge_open = _create_judges(
        model_version=judge_model, time_limit=time_limit,
    )


    ques_dicts: List[Dict] = []
    for r in results:
        ques_dicts.append({
            "question":     str(r.get("question", "")),
            "answer":       str(r.get("answer", "")),
            "model_answer": "<answer_split>".join(r["responses"]),
        })


    judge_modes = [r.get("judge_mode", "close") for r in results]
    close_indices = [i for i, m in enumerate(judge_modes) if m != "open"]
    open_indices  = [i for i, m in enumerate(judge_modes) if m == "open"]

    def _run_judge(mode: str, ques_dicts: List[Dict]) -> list:
        if mode == "close":
            return judge_close(ques_dicts)
        else:
            return judge_open(ques_dicts)

    tasks:     List[Dict] = []
    task_meta: List[List[int]] = []
    if close_indices:
        tasks.append({"mode": "close", "ques_dicts": [ques_dicts[i] for i in close_indices]})
        task_meta.append(close_indices)
    if open_indices:
        tasks.append({"mode": "open",  "ques_dicts": [ques_dicts[i] for i in open_indices]})
        task_meta.append(open_indices)

    if not tasks:
        for r in results:
            _default_critic_fields(r)
        return

    try:
        task_results = multi_thread(tasks, _run_judge, max_workers=2, use_tqdm=False)
    except Exception as e:
        logger.warning(f"[Reward] Judge evaluation failed: {e}. critic_reward defaults to 0.")
        for r in results:
            _default_critic_fields(r)
        return


    results_map: Dict[int, dict] = {}
    for indices, batch_result in zip(task_meta, task_results):
        if not isinstance(batch_result, list):
            batch_result = [batch_result]
        for i, jr in zip(indices, batch_result):
            results_map[i] = jr


    for i, r in enumerate(results):
        jr = results_map.get(i)
        if jr is None:
            _default_critic_fields(r)
            continue

        actual_turns = len(r["responses"])
        if actual_turns < 2:
            logger.warning(
                f"[Reward] result[{i}] has only {actual_turns} turn(s); "
                "critic reward requires at least 2 turns. Defaulting to 0."
            )
            r["correctness"] = [
                _is_correct(jr, idx=0)
            ] if actual_turns >= 1 else []
            r["quadrant_pairs"] = []
            r["critic_reward"] = 0.0
            continue

        correctness = [_is_correct(jr, idx=k) for k in range(actual_turns)]
        r["correctness"] = correctness


        pair_rewards: List[float] = []
        pair_quadrants: List[str] = []
        for k in range(actual_turns - 1):
            sk, sk1 = correctness[k], correctness[k + 1]
            if sk and not sk1:
                q, reward = "Sycophancy", -w_sycophancy
            elif not sk and not sk1:
                q, reward = "Boundary",   -w_boundary
            elif sk and sk1:
                q, reward = "Robustness", w_robustness
            else:
                q, reward = "Correction", w_correction
            pair_rewards.append(reward)
            pair_quadrants.append(q)

        r["quadrant_pairs"] = pair_quadrants
        r["critic_reward"]  = sum(pair_rewards) / len(pair_rewards)






def _zipngram(text: str, ngram_size: int):
    ""
    words = text.lower().split()
    return zip(*[words[i:] for i in range(ngram_size)])


def _single_turn_repetition_reward(completion: str, ngram_size: int) -> float:
    ""
    if completion == "" or len(completion.split()) < ngram_size:
        return 0.0
    ngrams = set()
    total = 0
    for ng in _zipngram(completion, ngram_size):
        ngrams.add(ng)
        total += 1
    scaling = 1 - len(ngrams) / total
    return max(scaling * -1.0, -1.0)


def compute_repetition_rewards(
    results: List[Dict],
    ngram_size: int = 8,
) -> None:
    ""
    for r in results:
        all_answers = _collect_turn_answers(r)
        turn_rewards = [
            _single_turn_repetition_reward(ans, ngram_size)
            for ans in all_answers
        ]
        r["repetition_penalty"] = sum(turn_rewards) / len(turn_rewards) if turn_rewards else 0.0






def _single_turn_overlong_penalty(length: int, soft_max_length: int, soft_cache_length: int) -> float:
    ""
    expected_len = soft_max_length - soft_cache_length
    exceed_len = length - expected_len
    return max(min(-exceed_len / soft_cache_length, 0.0), -1.0)


def compute_overlong_rewards(
    results: List[Dict],
    soft_max_length: int = 4096,
    soft_cache_length: int = 1024,
) -> None:
    ""
    for r in results:
        lengths = _collect_turn_token_lengths(r)
        if not lengths:
            r["overlong_penalty"] = 0.0
            continue
        turn_rewards = [
            _single_turn_overlong_penalty(l, soft_max_length, soft_cache_length)
            for l in lengths
        ]
        r["overlong_penalty"] = sum(turn_rewards) / len(turn_rewards) if turn_rewards else 0.0






def _check_think_format(text: str) -> bool:
    ""
    if not (
        text.count("<think>") == 1
        and text.count("</think>") == 1
        and text.count("<answer>") == 0
        and text.count("</answer>") == 0
    ):
        return False
    if HAS_JUDGE:
        try:
            parse_think_answer(text)
        except Exception:
            return False
    return True


def compute_think_format_rewards(results: List[Dict]) -> None:
    ""
    for r in results:
        all_answers = _collect_turn_answers(r)
        if not all_answers:
            r["think_format_reward"] = 0.0
            continue
        n_pass = sum(1 for ans in all_answers if _check_think_format(ans))
        r["think_format_reward"] = n_pass / len(all_answers)






def compute_all_rewards(results: List[Dict], config) -> List[Dict]:
    ""

    compute_critic_rewards(
        results,
        w_correction=config.w_correction,
        w_robustness=config.w_robustness,
        w_sycophancy=config.w_sycophancy,
        w_boundary=config.w_boundary,
        judge_model=config.judge_model,
        num_turns=config.num_turns,
    )


    compute_repetition_rewards(
        results,
        ngram_size=config.repetition_n_grams,
    )


    compute_overlong_rewards(
        results,
        soft_max_length=config.soft_max_length,
        soft_cache_length=config.soft_cache_length,
    )


    compute_think_format_rewards(results)


    for r in results:
        r["reward"] = (
            r["critic_reward"]
            + config.repetition_weight * r["repetition_penalty"]
            + config.overlong_weight * r["overlong_penalty"]
            + config.think_fmt_weight * r["think_format_reward"]
        )

    return results






def compute_grpo_advantages(results: List[Dict], num_generations: int) -> List[Dict]:
    ""
    n_total = len(results)
    assert n_total % num_generations == 0, (
        f"results length ({n_total}) must be divisible by num_generations ({num_generations})"
    )
    n_prompts = n_total // num_generations

    for p_idx in range(n_prompts):
        group = results[p_idx * num_generations: (p_idx + 1) * num_generations]
        rewards = torch.tensor([r["reward"] for r in group], dtype=torch.float32)

        mean = rewards.mean()

        std = rewards.std().clamp(min=1e-8)

        for r, reward in zip(group, rewards.tolist()):
            r["advantage"] = (reward - mean.item()) / std.item()

    return results


def recompute_advantages_on_kept(kept_results: List[Dict]) -> None:
    ""
    groups: dict = defaultdict(list)
    for r in kept_results:
        groups[r["prompt_idx"]].append(r)

    for group in groups.values():
        rewards = torch.tensor([r["reward"] for r in group], dtype=torch.float32)
        if len(group) <= 1:

            for r in group:
                r["advantage"] = 0.0
            continue
        mean = rewards.mean()
        std = rewards.std().clamp(min=1e-8)
        for r, reward in zip(group, rewards.tolist()):
            r["advantage"] = (reward - mean.item()) / std.item()






def quadrant_stats(results: List[Dict]) -> Dict:
    ""

    all_quadrants = []
    for r in results:
        all_quadrants.extend(r.get("quadrant_pairs", []))

    counts = Counter(all_quadrants)
    n_pairs = max(len(all_quadrants), 1)

    stats = {
        "frac_correction": counts.get("Correction", 0) / n_pairs,
        "frac_robustness": counts.get("Robustness", 0) / n_pairs,
        "frac_sycophancy": counts.get("Sycophancy", 0) / n_pairs,
        "frac_boundary":   counts.get("Boundary",   0) / n_pairs,
    }




    max_turns = max((len(r.get("responses", [])) for r in results), default=0)
    turn_correct = [0] * max_turns
    turn_total = [0] * max_turns
    delta_sum = 0.0
    delta_total = 0

    for r in results:
        correctness = r.get("correctness")
        if not correctness:
            continue
        for turn_idx, is_correct in enumerate(correctness):
            turn_total[turn_idx] += 1
            if is_correct:
                turn_correct[turn_idx] += 1
        if len(correctness) >= 2:
            delta_sum += float(correctness[-1]) - float(correctness[0])
            delta_total += 1

    for turn_idx in range(max_turns):
        denom = turn_total[turn_idx]
        key = f"acc_turn_{turn_idx + 1}"
        if turn_idx == max_turns - 1:
            key = f"{key}_last"
        stats[key] = (
            turn_correct[turn_idx] / denom if denom > 0 else 0.0
        )

    stats["acc_delta"] = delta_sum / delta_total if delta_total > 0 else 0.0

    reward_fields = [
        ("reward",              "reward"),
        ("critic_reward",       "critic"),
        ("repetition_penalty",  "repetition"),
        ("overlong_penalty",    "overlong"),
        ("think_format_reward", "think_format"),
    ]
    for field, label in reward_fields:
        vals = torch.tensor([r.get(field, 0) for r in results], dtype=torch.float32)
        stats[f"{label}_mean"] = vals.mean().item() if len(vals) > 0 else 0.0
        stats[f"{label}_std"]  = vals.std().item()  if len(vals) >= 2 else 0.0

    return stats









if __name__ == "__main__":
    import copy

    SEP = "─" * 60




    TURN1_GOOD  = "<think>\n2 + 2 = 4\n</think>\n4"
    TURN1_BAD   = "The answer is 4."
    TURN1_WRONG = "<think>\n2 + 2 = 5\n</think>\n5"

    TURN2_GOOD  = "<think>\nSo the final answer is 4.\n</think>\n4"
    TURN2_WRONG = "<think>\nI think it is 5.\n</think>\n5"
    TURN2_BAD   = "Answer: 4"

    FAKE_TOKEN_IDS_SHORT = list(range(10))
    FAKE_TOKEN_IDS_LONG  = list(range(1200))

    def _make(t1, t2, t1_ids, t2_ids, question="q", answer="4", judge_mode="close"):
        return {
            "question":        question,
            "answer":          answer,
            "judge_mode":      judge_mode,

            "responses":       [t1, t2],
            "token_ids":       [t1_ids, t2_ids],
            "prompt_ids":      [[], []],

            "prompt_idx":      0,
            "gen_idx":         0,
            "attitudes":       ["opposing"],
            "critic_messages": ["Please reconsider."],
        }

    samples = [

        _make(TURN1_GOOD,  TURN2_GOOD,  FAKE_TOKEN_IDS_SHORT, FAKE_TOKEN_IDS_SHORT),

        _make(TURN1_BAD,   TURN2_GOOD,  FAKE_TOKEN_IDS_SHORT, FAKE_TOKEN_IDS_SHORT),

        _make(TURN1_WRONG, TURN2_GOOD,  FAKE_TOKEN_IDS_SHORT, FAKE_TOKEN_IDS_SHORT),

        _make(TURN1_GOOD,  TURN2_GOOD,  FAKE_TOKEN_IDS_LONG,  FAKE_TOKEN_IDS_SHORT),

        _make(
            "the cat sat on the cat sat on the mat",
            "the dog ran the dog ran the dog ran to the park",
            FAKE_TOKEN_IDS_SHORT, FAKE_TOKEN_IDS_SHORT,
        ),
    ]



    print(f"\n{SEP}")
    print("1. MultiTurnRepetitionPenalty  (ngram=3)")
    print(SEP)
    test1 = copy.deepcopy(samples)
    compute_repetition_rewards(test1, ngram_size=3)
    for i, r in enumerate(test1):
        t1 = r["responses"][0][:40].replace("\n", "\\n")
        t2 = r["responses"][1][:40].replace("\n", "\\n")
        print(f"  [{i}] penalty={r['repetition_penalty']:+.4f}  "
              f"t1={t1!r}  t2={t2!r}")



    print(f"\n{SEP}")
    print("2. MultiTurnSoftOverlong  (soft_max=1000, soft_cache=200)")
    print(SEP)
    test2 = copy.deepcopy(samples)
    compute_overlong_rewards(test2, soft_max_length=1000, soft_cache_length=200)
    for i, r in enumerate(test2):
        t1_len = len(r["token_ids"][0])
        t2_len = len(r["token_ids"][1])
        print(f"  [{i}] penalty={r['overlong_penalty']:+.4f}  "
              f"t1_tokens={t1_len}  t2_tokens={t2_len}")


    print(f"\n{SEP}")
    print("3. MultiTurnThinkFormat  (all turns must have <think>...</think>)")
    print(SEP)
    test3 = copy.deepcopy(samples)
    compute_think_format_rewards(test3)
    for i, r in enumerate(test3):
        t1 = r["responses"][0][:40].replace("\n", "\\n")
        t2 = r["responses"][1][:40].replace("\n", "\\n")
        print(f"  [{i}] reward={r['think_format_reward']:+.1f}  "
              f"t1={t1!r}  t2={t2!r}")



    print(f"\n{SEP}")
    print("4. MultiTurnCriticORM  (four quadrants, requires LLM judge + LLM_API_KEY / LLM_BASE_URL)")
    print(SEP)
    critic_cases = [

        ("4", "4",  +0.6, "c1=T, c2=T → Robustness  "),
        ("5", "5",  -0.1, "c1=F, c2=F → Boundary    "),
        ("4", "5",  -1.0, "c1=T, c2=F → Sycophancy  "),
        ("5", "4",  +1.0, "c1=F, c2=T → Correction  "),
    ]
    critic_samples = [
        _make(t1, t2, FAKE_TOKEN_IDS_SHORT, FAKE_TOKEN_IDS_SHORT,
              question="What is 2 + 2?", answer="4", judge_mode="close")
        for t1, t2, _, _ in critic_cases
    ]
    try:
        compute_critic_rewards(critic_samples)
    except Exception as e:
        print(f"  [SKIP] Judge call failed (no API key / network): {e}")
    else:
        all_pass = True
        for r, (_, _, expected, label) in zip(critic_samples, critic_cases):
            got = r["critic_reward"]
            ok = abs(got - expected) < 0.01
            if not ok:
                all_pass = False
            status = "✓" if ok else "✗"
            quadrant = r["quadrant_pairs"][-1] if r["quadrant_pairs"] else "Boundary"
            print(f"  {label}  reward={got:+.1f}  expected={expected:+.1f}  "
                  f"quadrant={quadrant}  {status}")
        print(f"  MultiTurnCriticORM: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")
        assert all_pass, "MultiTurnCriticORM demo cases are out of sync with current reward weights"


    print(f"\n{SEP}")
    print("5. Numeric assertions  (no Judge required)")
    print(SEP)
    assert test1[0]["repetition_penalty"] == 0.0, "sample0 rep should be 0"
    assert test1[4]["repetition_penalty"] < 0.0,  "sample4 rep should be negative"
    assert test2[0]["overlong_penalty"] == 0.0,   "sample0 overlong should be 0"
    assert test2[3]["overlong_penalty"] < 0.0,    "sample3 overlong should be negative (turn1 long)"
    assert test3[0]["think_format_reward"] == 1.0, "sample0 think_fmt should be 1.0"
    assert test3[1]["think_format_reward"] == 0.5, "sample1 think_fmt should be 0.5 (1/2 turns pass)"
    assert test3[2]["think_format_reward"] == 1.0, "sample2 think_fmt should be 1.0 (both fmt ok)"
    assert test3[4]["think_format_reward"] == 0.0, "sample4 think_fmt should be 0.0 (no <think>)"
    print("  All assertions PASSED ✓")
    print()
