"""Run the required notebooks, preserving real outputs and per-stage evidence."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab22 import config as C

STAGES = {
    "nb0": "00_dpo_loss_from_scratch",
    "nb1": "01_sft_mini",
    "nb2": "02_preference_data",
    "nb3": "03_dpo_train",
    "nb4": "04_compare_and_eval",
}


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def source_hash(path):
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    args = parser.parse_args()
    evidence = ROOT / "submission/evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    manifest_path = evidence / "pipeline.json"
    settings = {
        "tier": C.COMPUTE_TIER, "model": C.BASE_MODEL, "max_length": C.MAX_LEN,
        "sft_samples": C.SFT_SLICE, "pref_train": C.PREF_TRAIN, "pref_eval": C.PREF_EVAL,
        "beta": C.DPO_BETA, "lr": C.DPO_LR, "epochs": C.DPO_EPOCHS,
        "seed": C.SEED, "judge_models": C.JUDGE_RM_MODELS, "judge_prompts": C.JUDGE_PROMPTS,
        "judge_4bit": os.environ.get("JUDGE_RM_4BIT", "0"),
        "judge_max_length": os.environ.get("JUDGE_RM_MAX_LENGTH", "4096"),
        "generation_batch": os.environ.get("GEN_BATCH_SIZE", "8"),
        "generation_max_new_tokens": C.GEN_MAX_NEW_TOKENS,
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "started_utc": timestamp(), "settings": settings, "stages": [],
    }
    if manifest["settings"] != settings:
        raise SystemExit("Experiment configuration changed; preserve the existing run before starting another.")
    for stage in args.stages:
        source = ROOT / "notebooks" / (STAGES[stage] + ".py")
        digest = source_hash(source)
        prior = [row for row in manifest["stages"] if row["stage"] == stage and row["exit_code"] == 0]
        if prior:
            if prior[-1]["source_sha256"] != digest:
                raise SystemExit(f"Executed source changed for {stage}; do not silently overwrite its evidence.")
            print(f"SKIP {stage}: already completed", flush=True)
            continue
        attempt = 1 + sum(row["stage"] == stage for row in manifest["stages"])
        log = evidence / f"{stage}-{attempt}.log"
        started = time.perf_counter()
        record = {"stage": stage, "started_utc": timestamp(), "source_sha256": digest,
                  "log": log.relative_to(ROOT).as_posix(),
                  "saved_tensors_on_cpu": os.environ.get("DPO_SAVE_ON_CPU", "0") == "1"}
        print(f"RUN {stage}: {log.relative_to(ROOT)}", flush=True)
        with log.open("w", encoding="utf-8") as destination:
            converted = subprocess.run([sys.executable, "-m", "jupytext", "--to", "notebook", str(source)],
                                       cwd=ROOT, stdout=destination, stderr=subprocess.STDOUT)
            executed = converted if converted.returncode else subprocess.run(
                [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                 "--ExecutePreprocessor.timeout=-1", str(source.with_suffix(".ipynb"))],
                cwd=ROOT, stdout=destination, stderr=subprocess.STDOUT,
            )
        record.update({"finished_utc": timestamp(), "seconds": time.perf_counter() - started,
                       "exit_code": executed.returncode})
        manifest["stages"].append(record)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"END {stage}: exit={executed.returncode}, seconds={record['seconds']:.1f}", flush=True)
        if executed.returncode:
            print("\n".join(log.read_text(encoding="utf-8").splitlines()[-45:]))
            return executed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
