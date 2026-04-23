""

import json
import logging

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class QADataset(Dataset):
    ""

    _Q_ALIASES = ("question", "problem", "query")
    _A_ALIASES = ("answer", "solution", "ground_truth")


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



                q = None
                if "messages" in obj and isinstance(obj["messages"], list):
                    for msg in obj["messages"]:
                        if msg.get("role") == "user":
                            q = msg.get("content", "")
                            break

                if not q:
                    q = next((obj[k] for k in self._Q_ALIASES if k in obj), None)


                a = next((obj[k] for k in self._A_ALIASES if k in obj), None)

                if not q or not a:
                    continue


                if judge_mode in ("close", "open"):
                    mode = judge_mode
                else:

                    mode = obj.get("judge_mode", "")
                    if not mode:
                        n_missing_judge_mode += 1
                        mode = "close"


                if add_format_prompt:
                    fmt = self._FORMAT_PROMPT_CLOSE if mode == "close" else self._FORMAT_PROMPT_OPEN

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
    ""
    return batch
