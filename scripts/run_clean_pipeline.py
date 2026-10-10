"""Run the original make pipeline once, refusing reused experiment artifacts."""

from __future__ import annotations

import datetime
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_pipeline import STAGES, source_hash

from lab22 import config as C


def existing_artifacts(root):
    paths = [root / relative for relative in (
        "models/sft-merged", "adapters/sft-mini", "adapters/dpo",
        "data/pref/train.parquet", "data/pref/eval.parquet", "data/eval/side_by_side.jsonl",
        "data/eval/judge_summary.json", "submission/evidence/pipeline.json",
    )]
    paths.extend(root / "notebooks" / (stem + ".ipynb") for stem in STAGES.values())
    paths.extend((root / "submission/screenshots").glob("*.png"))
    return [path.relative_to(root).as_posix() for path in paths if path.exists()]


def execution_record(root, stage, stem, fallback_end):
    path = root / "notebooks" / (stem + ".ipynb")
    if not path.exists():
        return None
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code" and "".join(cell["source"]).strip()]
    successful = bool(cells) and all(cell.get("execution_count") is not None for cell in cells) and not any(
        output["output_type"] == "error" for cell in cells for output in cell.get("outputs", [])
    )
    timestamps = sorted(
        datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        for cell in cells for value in cell.get("metadata", {}).get("execution", {}).values()
    )
    started = timestamps[0] if timestamps else fallback_end
    finished = timestamps[-1] if timestamps else fallback_end
    return {
        "stage": stage, "source_sha256": source_hash(root / "notebooks" / (stem + ".py")),
        "started_utc": started.isoformat(), "finished_utc": finished.isoformat(),
        "seconds": (finished - started).total_seconds(), "exit_code": 0 if successful else 1,
        "timing_source": "nbconvert cell execution timestamps",
        "log": "submission/evidence/clean-pipeline.log",
        "saved_tensors_on_cpu": os.environ.get("DPO_SAVE_ON_CPU", "0") == "1",
    }


def pipeline_command(python):
    return [
        "make", f"PY={shlex.quote(python)}",
        f"JUPYTEXT={shlex.join([python, '-m', 'jupytext'])}",
        f"JUPYTER={shlex.join([python, '-m', 'jupyter'])}", "pipeline",
    ]


def main():
    leftovers = existing_artifacts(ROOT)
    if leftovers:
        print("Refusing to reuse results; use a fresh source-only checkout:", ", ".join(leftovers))
        return 1
    evidence = ROOT / "submission/evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    started = datetime.datetime.now(datetime.timezone.utc)
    clock_started = time.perf_counter()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    sources = [ROOT / "Makefile", *sorted((ROOT / "lab22").glob("*.py")),
               *(ROOT / "notebooks" / (stem + ".py") for stem in STAGES.values())]
    manifest = {
        "started_utc": started.isoformat(), "entrypoint": "make pipeline", "fresh_artifacts": True,
        "source_commit": revision,
        "cached_downloads_reused": any(cache.glob("models--*")) or any(cache.glob("datasets--*")),
        "core_sources_sha256": {path.relative_to(ROOT).as_posix(): source_hash(path) for path in sources},
        "instrumentation_sha256": source_hash(Path(__file__)),
        "settings": {
            "tier": C.COMPUTE_TIER, "model": C.BASE_MODEL, "max_length": C.MAX_LEN,
            "sft_samples": C.SFT_SLICE, "pref_train": C.PREF_TRAIN, "pref_eval": C.PREF_EVAL,
            "beta": C.DPO_BETA, "lr": C.DPO_LR, "epochs": C.DPO_EPOCHS, "seed": C.SEED,
            "judge_models": C.JUDGE_RM_MODELS, "judge_prompts": C.JUDGE_PROMPTS,
            "judge_4bit": os.environ.get("JUDGE_RM_4BIT", "0"),
            "judge_dtype": os.environ.get("JUDGE_RM_DTYPE", "auto"),
            "judge_device_map": os.environ.get("JUDGE_RM_DEVICE_MAP", "auto"),
            "judge_max_length": os.environ.get("JUDGE_RM_MAX_LENGTH", "4096"),
            "generation_batch": os.environ.get("GEN_BATCH_SIZE", "8"),
            "generation_max_new_tokens": C.GEN_MAX_NEW_TOKENS,
        },
        "stages": [],
    }
    manifest_path = evidence / "pipeline.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    command = pipeline_command(sys.executable)
    print("Running", " ".join(command), flush=True)
    with (evidence / "clean-pipeline.log").open("w", encoding="utf-8") as destination:
        process = subprocess.Popen(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in process.stdout:
            destination.write(line)
            destination.flush()
            print(line, end="", flush=True)
        exit_code = process.wait()
    ended = datetime.datetime.now(datetime.timezone.utc)
    manifest.update({"finished_utc": ended.isoformat(), "seconds": time.perf_counter() - clock_started,
                     "exit_code": exit_code})
    manifest["stages"] = [
        record for stage, stem in STAGES.items()
        if (record := execution_record(ROOT, stage, stem, ended)) is not None
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Clean make pipeline exit={exit_code}; stages saved={len(manifest['stages'])}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
