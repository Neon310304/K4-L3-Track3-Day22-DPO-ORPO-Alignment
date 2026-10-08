"""Measure actual reward-model input lengths for the saved generation and sanity pairs."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab22 import config as C
from lab22 import judge as J


def count_tokens(tokenizer, prompt, answer):
    conversation = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
    text = tokenizer.apply_chat_template(conversation, tokenize=False)
    if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
        text = text[len(tokenizer.bos_token):]
    return len(tokenizer(text)["input_ids"])


def main():
    from transformers import AutoTokenizer

    manifest = json.loads((ROOT / "submission/evidence/pipeline.json").read_text(encoding="utf-8"))
    limit = int(manifest["settings"]["judge_max_length"])
    records = [
        json.loads(line) for line in (C.EVAL_DIR / "side_by_side.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    payload = {}
    for name in C.JUDGE_RM_MODELS:
        tokenizer = AutoTokenizer.from_pretrained(name)

        generated = [count_tokens(tokenizer, row["prompt"], row[side]) for row in records for side in ("sft", "dpo")]
        sanity = [
            count_tokens(tokenizer, prompt, answer) for prompt, good, bad in J.SANITY_PAIRS for answer in (good, bad)
        ]
        payload[name] = {
            "max_length": limit, "generated_inputs": len(generated), "generated_max_tokens": max(generated),
            "generated_truncated_inputs": sum(tokens > limit for tokens in generated),
            "sanity_inputs": len(sanity), "sanity_max_tokens": max(sanity),
            "sanity_truncated_inputs": sum(tokens > limit for tokens in sanity),
        }
    (ROOT / "submission/evidence/judge-token-budgets.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
