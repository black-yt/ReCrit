"""
ReCrit dataset module.

Loads JSONL files in ms-swift format or a simplified QA format,
and returns a list of {"question": str, "answer": str, "judge_mode": str}.
"""

import json
import logging

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class QADataset(Dataset):
    """
    Load a JSONL file in either of the following formats:
      1. ms-swift format (the actual training-data format):
            {"messages": [{"role": "user", "content": "..."}], "answer": "...", "judge_mode": "close"}
         question is extracted from messages[0]["content"], preserving the full Format Requirements block
      2. simplified alias-based format:
            {"question": "...", "answer": "...", "judge_mode": "close"}
            {"problem": "...", "solution": "..."}

    Fields:
        question   : full user question, including formatting constraints when present
        answer     : reference answer (typically short in close mode and possibly long-form in open mode)
        judge_mode : "close" or "open" (default: "close")

    Args:
        path:              JSONL file path
        judge_mode:        "close"/"open"/"both"
                             close/open - force all samples to use that mode
                             both       - use each sample's own judge_mode field (it must exist)
        add_format_prompt: whether to append a format-requirements prompt to the question
    """

    _Q_ALIASES = ("question", "problem", "query")
    _A_ALIASES = ("answer", "solution", "ground_truth")

    # Standard format-requirements prompts extracted from the training data
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

                # -- Extract question --
                # First try the first user message in messages (ms-swift format)
                q = None
                if "messages" in obj and isinstance(obj["messages"], list):
                    for msg in obj["messages"]:
                        if msg.get("role") == "user":
                            q = msg.get("content", "")
                            break
                # Otherwise fall back to question/problem/query fields
                if not q:
                    q = next((obj[k] for k in self._Q_ALIASES if k in obj), None)

                # -- Extract answer --
                a = next((obj[k] for k in self._A_ALIASES if k in obj), None)

                if not q or not a:
                    continue

                # -- Determine judge_mode --
                if judge_mode in ("close", "open"):
                    mode = judge_mode
                else:
                    # both: use the judge_mode field stored in the data
                    mode = obj.get("judge_mode", "")
                    if not mode:
                        n_missing_judge_mode += 1
                        mode = "close"  # default to close when the field is missing

                # -- Append format-requirements prompt --
                if add_format_prompt:
                    fmt = self._FORMAT_PROMPT_CLOSE if mode == "close" else self._FORMAT_PROMPT_OPEN
                    # Append only when the question does not already contain "Format Requirements" to avoid duplication
                    if "Format Requirements" not in q:
                        q = q + fmt

                self.data.append({
                    "question":   q,
                    "answer":     a,
                    "judge_mode": mode,
                })

        if n_missing_judge_mode > 0:
            logger.warning(
                f"[Dataset] judge_mode='both', but {n_missing_judge_mode} sample(s) "
                "are missing the judge_mode field. They were defaulted to 'close'. "
                "Please ensure every sample includes judge_mode."
            )
        logger.info(f"[Dataset] Loaded {len(self.data)} samples from {path} "
                    f"(judge_mode={judge_mode}, add_format_prompt={add_format_prompt})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """DataLoader collate: return list[dict] directly."""
    return batch
