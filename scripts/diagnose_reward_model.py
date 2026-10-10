"""Measure unchanged Vietnamese sanity pairs in a vanilla Transformers process."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab22 import judge as J


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Skywork/Skywork-Reward-V2-Qwen3-4B")
    parser.add_argument("--nf4", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["JUDGE_RM_4BIT"] = "1" if args.nf4 else "0"
    import torch

    started = time.perf_counter()
    scorer = J.make_rm_scorer(args.model)
    rows = []
    for index, (prompt, good, bad) in enumerate(J.SANITY_PAIRS):
        good_score, bad_score = scorer(prompt, good), scorer(prompt, bad)
        rows.append({"id": index, "prompt": prompt, "good_score": good_score, "bad_score": bad_score,
                     "correct": good_score > bad_score})
        print(f"sanity {index + 1}/{len(J.SANITY_PAIRS)}: correct={good_score > bad_score}", flush=True)
    payload = {
        "model": args.model, "nf4": args.nf4, "process_imports_unsloth": "unsloth" in sys.modules,
        "gpu": torch.cuda.get_device_name(0), "seconds": time.perf_counter() - started,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "bitsandbytes")},
        "pairs": rows, "sanity_accuracy": sum(row["correct"] for row in rows) / len(rows),
        "sanity_pairs_sha256": hashlib.sha256(
            json.dumps(J.SANITY_PAIRS, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Sanity accuracy: {payload['sanity_accuracy']:.1%}; passed={payload['sanity_accuracy'] >= 0.8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
