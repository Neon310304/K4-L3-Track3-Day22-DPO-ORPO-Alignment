"""Cross-check submission evidence without training or calling a judge."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab22 import data as D
from lab22 import judge as J

STAGES = {
    "nb0": "00_dpo_loss_from_scratch", "nb1": "01_sft_mini",
    "nb2": "02_preference_data", "nb3": "03_dpo_train", "nb4": "04_compare_and_eval",
}


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def same_values(measured, saved):
    if isinstance(measured, dict):
        return isinstance(saved, dict) and all(
            key in saved and same_values(value, saved[key]) for key, value in measured.items()
        )
    if isinstance(measured, list):
        return isinstance(saved, list) and len(measured) == len(saved) and all(
            same_values(first, second) for first, second in zip(measured, saved)
        )
    if isinstance(measured, float):
        return isinstance(saved, (float, int)) and math.isclose(measured, saved, abs_tol=1e-9)
    return measured == saved


def audit():
    from datasets import Dataset

    checks = {}
    manifest = read_json("submission/evidence/pipeline.json")
    seed = manifest["settings"]["seed"]
    completed = {row["stage"]: row for row in manifest["stages"] if row["exit_code"] == 0}
    checks["required_stages_completed"] = all(stage in completed for stage in STAGES)
    checks["executed_sources_unchanged"] = all(
        hashlib.sha256((ROOT / "notebooks" / (stem + ".py")).read_text(encoding="utf-8").encode("utf-8")).hexdigest()
        == completed.get(stage, {}).get("source_sha256") for stage, stem in STAGES.items()
    )
    checks["notebooks_keep_successful_outputs"] = True
    for stem in STAGES.values():
        notebook = read_json("notebooks/" + stem + ".ipynb")
        code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code" and "".join(cell["source"]).strip()]
        checks["notebooks_keep_successful_outputs"] &= bool(code) and all(
            cell["execution_count"] is not None for cell in code
        )
        checks["notebooks_keep_successful_outputs"] &= any(cell.get("outputs") for cell in code)
        checks["notebooks_keep_successful_outputs"] &= not any(
            output["output_type"] == "error" for cell in code for output in cell.get("outputs", [])
        )
    train = list(Dataset.from_parquet(str(ROOT / "data/pref/train.parquet")))
    heldout = list(Dataset.from_parquet(str(ROOT / "data/pref/eval.parquet")))
    D.assert_disjoint(train, heldout)
    checks["full_preference_split"] = len(train) == 800 and len(heldout) == 100
    checks["no_train_eval_prompt_overlap"] = True
    checks["adapter_matches_saved_split"] = D.split_mismatch(ROOT / "data/pref", ROOT / "adapters/dpo") is None
    sft = read_json("adapters/sft-mini/sft_metrics.json")
    checks["full_sft_sample_count"] = sft["samples"] == 1000
    checks["response_only_sft_loss"] = (
        0 < sft["supervised_tokens"] < sft["total_tokens"] and sft["supervised_fraction"] < 0.95
    )
    sft_losses = [row["loss"] for row in sft["log_history"] if "loss" in row]
    checks["sft_loss_decreases"] = len(sft_losses) >= 2 and sft_losses[-1] < sft_losses[0]
    dpo = read_json("adapters/dpo/dpo_metrics.json")
    adapter = read_json("adapters/dpo/adapter_config.json")
    reference = Path(adapter["base_model_name_or_path"])
    checks["dpo_reference_is_sft"] = reference.as_posix().rstrip("/").endswith("models/sft-merged")
    checks["one_full_dpo_epoch"] = dpo["epochs"] == 1 and dpo["optimizer_steps"] == 100
    train_rewards = [row for row in dpo["log_history"] if "rewards/chosen" in row and "rewards/rejected" in row]
    eval_rewards = [
        row for row in dpo["log_history"] if "eval_rewards/chosen" in row and "eval_rewards/rejected" in row
    ]
    checks["train_and_heldout_rewards_saved"] = len(train_rewards) >= 2 and len(eval_rewards) >= 2
    from lab22 import modeling as MD

    eval_history = MD.reward_history(dpo["log_history"], prefix="eval_")
    train_history = MD.reward_history(dpo["log_history"])
    checks["diagnosis_matches_reward_history"] = MD.diagnose(
        eval_history if len(eval_history) >= 2 else train_history
    )[0] == dpo["diagnosis"]
    checks["heldout_evaluated_every_25_steps"] = {25, 50, 75, 100} <= {row["step"] for row in eval_rewards}
    checks["dpo_same_data_and_config"] = all(
        dpo[key] == manifest["settings"][key] for key in ("beta", "lr", "epochs", "seed")
    )
    outputs = [
        json.loads(line) for line in (ROOT / "data/eval/side_by_side.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    held_prompts = {D.normalize_prompt(row["prompt"][0]["content"]) for row in heldout}
    held_outputs = [row for row in outputs if row["category"] == "heldout"]
    checks["eight_fixed_and_fifty_heldout"] = len(outputs) == 58 and len(held_outputs) == 50
    checks["generation_uses_unseen_distinct_prompts"] = len({row["prompt"] for row in held_outputs}) == 50 and all(
        D.normalize_prompt(row["prompt"]) in held_prompts for row in held_outputs
    )
    saved = read_json("data/eval/judge_results_rm.json")
    summary = read_json("data/eval/judge_summary.json")
    checks["judge_matches_exact_outputs"] = (
        summary["outputs_sha256"] == saved["outputs_sha256"] == digest("data/eval/side_by_side.jsonl")
    )
    checks["judge_records_preserve_answers"] = len(saved["records"]) == len(outputs) and all(
        all(judged[key] == generated[key] for key in ("id", "prompt", "sft", "dpo", "category"))
        for judged, generated in zip(saved["records"], outputs)
    )
    checks["judge_summary_recomputed"] = same_values(J.summarize(saved["records"], seed=seed), summary["overall"])
    for category in ("heldout", "helpfulness", "safety"):
        measured = J.summarize([row for row in saved["records"] if row["category"] == category], seed=seed)
        checks["judge_summary_recomputed"] &= same_values(measured, summary[category])
    checks["two_real_judges_recorded"] = len(saved["per_judge"]) == 2 and len(summary["sanity"]) == 2
    panel = [name for name in saved["per_judge"] if summary["sanity"][name] >= 0.8] or list(saved["per_judge"])
    checks["panel_obeys_sanity_filter"] = summary["judge"] == saved["judge"] == "rm-panel:" + "+".join(panel)
    checks["panel_obeys_sanity_filter"] &= all(
        J.panel_record([saved["per_judge"][name][index] for name in panel])["winner"] == row["winner"]
        for index, row in enumerate(saved["records"])
    )
    checks["rm_verdicts_match_saved_scores"] = all(
        len(rows) == len(outputs) and all(
            all(row[key] == generated[key] for key in ("id", "prompt", "sft", "dpo", "category"))
            and J.rm_record(row["sft_score"], row["dpo_score"])["winner"] == row["winner"]
            for row, generated in zip(rows, outputs)
        )
        for rows in saved["per_judge"].values()
    )
    checks["per_judge_summary_recomputed"] = all(
        same_values(
            J.summarize([row for row in rows if row["category"] == "heldout"], seed=seed), summary["per_judge"][name]
        )
        for name, rows in saved["per_judge"].items()
    )
    names = list(saved["per_judge"])
    checks["judge_agreement_recomputed"] = same_values(
        {"judges": names[:2], **J.agreement(saved["per_judge"][names[0]], saved["per_judge"][names[1]])},
        summary["judge_agreement"],
    )
    payload = {"passed": all(checks.values()), "checks": checks,
               "artifacts_sha256": {path: digest(path) for path in (
                   "adapters/sft-mini/sft_metrics.json", "adapters/dpo/dpo_metrics.json",
                   "adapters/dpo/split.json", "data/pref/train.parquet", "data/pref/eval.parquet",
                   "data/eval/side_by_side.jsonl", "data/eval/judge_results_rm.json", "data/eval/judge_summary.json",
               )}}
    (ROOT / "submission/evidence/audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main():
    try:
        result = audit()
    except (OSError, KeyError, ValueError, AssertionError) as error:
        print(f"FAIL incomplete or inconsistent evidence: {error}")
        return 1
    for name, passed in result["checks"].items():
        print(f"{'OK' if passed else 'FAIL'} {name}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
