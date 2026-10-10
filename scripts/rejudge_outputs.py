"""Re-score the exact saved NB4 answers without loading or training the policy."""

from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab22 import judge

JUDGES = ["Skywork/Skywork-Reward-V2-Qwen3-4B", "Skywork/Skywork-Reward-V2-Llama-3.2-3B"]


def evaluate(records, names, scorer_factory, seed=42, progress=None, cleanup=None):
    notify = progress or (lambda message: None)
    sanity, per_judge, details, settings = {}, {}, {}, {}
    for name in names:
        notify(f"Loading {name}")
        score = scorer_factory(name)
        settings[name] = getattr(score, "metadata", {})
        try:
            pairs = []
            for index, (prompt, good, bad) in enumerate(judge.SANITY_PAIRS):
                good_score, bad_score = score(prompt, good), score(prompt, bad)
                if not all(math.isfinite(value) for value in (good_score, bad_score)):
                    raise ValueError(f"{name}: nonfinite sanity scores")
                pairs.append({"index": index, "good_score": good_score, "bad_score": bad_score,
                              "correct": good_score > bad_score})
            details[name] = pairs
            sanity[name] = sum(pair["correct"] for pair in pairs) / len(pairs)
            notify(f"{name}: sanity {sanity[name]:.1%}; threshold unchanged at 80%")
            rows = []
            for index, record in enumerate(records, 1):
                verdict = judge.rm_judge_pair(record["prompt"], record["sft"], record["dpo"], score)
                if verdict["winner"] == "failed":
                    raise ValueError(f"{name}: invalid score at {record['id']}")
                rows.append({**record, **verdict})
                notify(f"{name}: scored {index}/{len(records)}")
            per_judge[name] = rows
        finally:
            del score
            gc.collect()
            if cleanup:
                cleanup()
    panel = [name for name in names if sanity[name] >= 0.8]
    if not panel:
        raise ValueError("No judge passes sanity; refusing to publish a win rate")
    judged = [{**record, **judge.panel_record([per_judge[name][index] for name in panel])}
              for index, record in enumerate(records)]
    label = "rm-panel:" + "+".join(panel)
    raw = {"judge": label, "records": judged, "per_judge": per_judge,
           "sanity_details": details, "scorer_settings": settings}
    summary = {
        "judge": label, "sanity_accuracy": min(sanity[name] for name in panel), "sanity": sanity,
        "overall": judge.summarize(judged, seed=seed),
        **{category: judge.summarize([row for row in judged if row["category"] == category], seed=seed)
           for category in ("heldout", "helpfulness", "safety")},
        "per_judge": {
            name: judge.summarize([row for row in rows if row["category"] == "heldout"], seed=seed)
            for name, rows in per_judge.items()
        },
    }
    if len(names) >= 2:
        summary["judge_agreement"] = {
            "judges": names[:2], **judge.agreement(per_judge[names[0]], per_judge[names[1]]),
        }
    return raw, summary


def load_outputs(path):
    payload = path.read_bytes()
    records = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    categories = {category: sum(row["category"] == category for row in records)
                  for category in ("heldout", "helpfulness", "safety")}
    if categories != {"heldout": 50, "helpfulness": 4, "safety": 4} or len(records) != 58:
        raise ValueError("Expected the unmodified 8 fixed + 50 held-out answers")
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Duplicate answer IDs")
    if any(not row[key].strip() for row in records for key in ("prompt", "sft", "dpo")):
        raise ValueError("Empty saved prompt or answer")
    return records, hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=ROOT / "data/eval/side_by_side.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/eval/rechecked")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory; previous judge evidence must be preserved")
    records, outputs_sha = load_outputs(args.outputs)
    args.output_dir.mkdir(parents=True)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    import torch

    raw, summary = evaluate(records, JUDGES, judge.make_rm_scorer, progress=lambda message: print(message, flush=True),
                            cleanup=lambda: torch.cuda.empty_cache() if torch.cuda.is_available() else None)
    if hashlib.sha256(args.outputs.read_bytes()).hexdigest() != outputs_sha:
        raise ValueError("Saved answers changed during judging")
    raw["outputs_sha256"] = summary["outputs_sha256"] = outputs_sha
    for filename, payload in (("judge_results_rm.json", raw), ("judge_summary.json", summary)):
        (args.output_dir / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
    record = {
        "started_utc": started, "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "outputs_sha256": outputs_sha, "policy_retrained": False, "answers_regenerated": False,
        "judge_source_sha256": hashlib.sha256((ROOT / "lab22/judge.py").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sanity": summary["sanity"], "both_judges_pass": all(value >= 0.8 for value in summary["sanity"].values()),
        "environment": {key: os.environ[key] for key in (
            "JUDGE_RM_DTYPE", "JUDGE_RM_DEVICE_MAP", "JUDGE_RM_4BIT", "JUDGE_RM_MAX_LENGTH",
            "JUDGE_RM_MAX_GPU_GIB", "JUDGE_RM_MAX_CPU_GIB", "JUDGE_RM_OFFLOAD_DIR",
        ) if key in os.environ},
    }
    (args.output_dir / "recheck.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    print("Saved", args.output_dir, "without retraining or editing any answer.", flush=True)
    return 0 if record["both_judges_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
