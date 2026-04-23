"""
ReCrit dataset module.

Loads JSONL files in either ms-swift format or a simplified QA format and
returns a list of {"question": str, "answer": str, "judge_mode": str}.
"""

import json
import logging

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class QADataset(Dataset):
    """
    Load a JSONL dataset in one of two supported formats:
      1. ms-swift format used by the training pipeline:
            {"messages": [{"role": "user", "content": "..."}], "answer": "...", "judge_mode": "close"}
         The question is extracted from the first user message, preserving any
         existing "Format Requirements" block.
      2. Simplified alias-based format:
            {"question": "...", "answer": "...", "judge_mode": "close"}
            {"problem": "...", "solution": "..."}

    Fields:
        question:   Full user question, possibly including formatting constraints.
        answer:     Reference answer. For close mode this is typically short;
                    for open mode it may be a longer text.
        judge_mode: "close" or "open" (default: "close").

    Args:
        path: JSONL file path.
        judge_mode:
            "close"/"open" force every sample into that mode.
            "both" reads judge_mode from each sample and requires the field.
        add_format_prompt:
            Whether to append a format-requirements prompt to the question.
    """

    _Q_ALIASES = ("question", "problem", "query")
    _A_ALIASES = ("answer", "solution", "ground_truth")

    # Standard format prompts extracted from the actual training data.
    _FORMAT_PROMPT_CLOSE = (
        "\n\nFormat Requirements:\n"
        "1. Clearly show your reasoning process enclosed within <think> and </think>.\n"
        '2. After </think>, provide only the final answer. For multiple-choice questions, '
        'output only the option label(s) (e.g., A, B, C, D, BC, DI, ADI, etc.) without '
        'including the option content or any explanatory phrases such as "\\boxed{A}" or '
        '"The answer is C."'
    )
    _FORMAT_PROMPT_OPEN = (
        "\n\nFormat Requirements:\n"
        "1. Clearly show your reasoning process enclosed within <think> and </think>.\n"
        "2. After </think>, provide the short final answer."
    )

    def __init__(self, path: str, judge_mode: str = "both", add_format_prompt: bool = False):
        self.data = []
        n_missing_judge_mode = 0

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)

                # Extract the question.
                # Prefer the first user message in the ms-swift "messages" list.
                q = None
                if "messages" in obj and isinstance(obj["messages"], list):
                    for msg in obj["messages"]:
                        if msg.get("role") == "user":
                            q = msg.get("content", "")
                            break

                # Fall back to alias fields such as question/problem/query.
                if not q:
                    q = next((obj[k] for k in self._Q_ALIASES if k in obj), None)

                # Extract the answer.
                a = next((obj[k] for k in self._A_ALIASES if k in obj), None)

                if not q or not a:
                    continue

                # Resolve the effective judge mode.
                if judge_mode in ("close", "open"):
                    mode = judge_mode
                else:
                    # "both" means the mode must come from the sample itself.
                    mode = obj.get("judge_mode", "")
                    if not mode:
                        n_missing_judge_mode += 1
                        mode = "close"

                # Append format requirements only when they are not already present.
                if add_format_prompt:
                    fmt = self._FORMAT_PROMPT_CLOSE if mode == "close" else self._FORMAT_PROMPT_OPEN
                    if "Format Requirements" not in q:
                        q = q + fmt

                self.data.append(
                    {
                        "question": q,
                        "answer": a,
                        "judge_mode": mode,
                    }
                )

        if n_missing_judge_mode > 0:
            logger.warning(
                f"[Dataset] judge_mode='both', but {n_missing_judge_mode} sample(s) "
                "are missing the judge_mode field. They were defaulted to 'close'. "
                "Please ensure every sample includes judge_mode."
            )
        logger.info(
            f"[Dataset] Loaded {len(self.data)} samples from {path} "
            f"(judge_mode={judge_mode}, add_format_prompt={add_format_prompt})"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """Return the batch as list[dict] without additional collation."""
    return batch
