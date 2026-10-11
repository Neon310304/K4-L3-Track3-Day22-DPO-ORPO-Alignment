"""Recover NB4 from exact saved answers while preserving the failed attempt."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_clean_pipeline import execution_record, pipeline_environment
from scripts.rejudge_outputs import load_outputs


def main():
    evidence = ROOT / "submission/evidence"
    path = evidence / "pipeline.json"
    original_bytes = path.read_bytes()
    original = json.loads(original_bytes)
    completed = {row["stage"] for row in original["stages"] if row["exit_code"] == 0}
    assert completed >= {"nb0", "nb1", "nb2", "nb3"}, "Training must finish before NB4 recovery"
    assert "nb4" not in completed, "Do not overwrite an already successful NB4"
    _, answers_sha = load_outputs(ROOT / "data/eval/side_by_side.jsonl")
    started = datetime.datetime.now(datetime.timezone.utc)
    history = ROOT / "submission/history" / ("T4_NB4_FAILED_" + started.strftime("%Y%m%dT%H%M%SZ"))
    history.mkdir(parents=True, exist_ok=False)
    (history / "pipeline.json").write_bytes(original_bytes)
    for relative in ("submission/evidence/clean-pipeline.log", "submission/evidence/nb4-progress.log",
                     "submission/evidence/judge-recovery.log", "notebooks/04_compare_and_eval.ipynb"):
        source = ROOT / relative
        if source.is_file():
            shutil.copy2(source, history / source.name)
    (history / "README.md").write_text(
        "The original fresh run completed NB0-NB3 and generated the saved answers. "
        "NB4 then failed in Unsloth-patched reward inference. The raw make exit code "
        "and log are preserved here. The original NB4 notebook was created by jupytext; "
        "nbconvert did not save its failed execution. It is not a successful executed notebook.\n",
        encoding="utf-8",
    )
    env = pipeline_environment(sys.executable, ROOT)
    env["NB4_RESUME_OUTPUTS_SHA256"] = answers_sha
    command = ["make", f"PY={sys.executable}", f"JUPYTEXT={sys.executable} -m jupytext",
               f"JUPYTER={sys.executable} -m jupyter", "eval"]
    clock = time.perf_counter()
    with (evidence / "nb4-recovery.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in child.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        exit_code = child.wait()
    finished = datetime.datetime.now(datetime.timezone.utc)
    assert hashlib.sha256((ROOT / "data/eval/side_by_side.jsonl").read_bytes()).hexdigest() == answers_sha
    record = execution_record(ROOT, "nb4", "04_compare_and_eval", finished)
    recovery = {
        "started_utc": started.isoformat(), "finished_utc": finished.isoformat(),
        "seconds": time.perf_counter() - clock, "exit_code": exit_code,
        "entrypoint": "make eval", "policy_retrained": False, "answers_regenerated": False,
        "outputs_sha256": answers_sha, "original_pipeline_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "original_history": history.relative_to(ROOT).as_posix(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "overlay_source_commit": os.environ.get("NB4_RECOVERY_SOURCE_COMMIT"),
        "nb4_source_sha256": hashlib.sha256((ROOT / "notebooks/04_compare_and_eval.py").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (evidence / "nb4-recovery.json").write_text(json.dumps(recovery, indent=2) + "\n")
    if exit_code == 0 and record and record["exit_code"] == 0:
        record["log"] = "submission/evidence/nb4-recovery.log"
        final = dict(original)
        final["entrypoint"] = "make pipeline; make eval recovery"
        final["stages"] = [row for row in original["stages"] if row["stage"] != "nb4"] + [record]
        final["core_sources_sha256"] = dict(original["core_sources_sha256"])
        final["core_sources_sha256"]["notebooks/04_compare_and_eval.py"] = record["source_sha256"]
        final["recovery"] = recovery
        final["finished_utc"] = finished.isoformat()
        final["seconds"] = original["seconds"] + recovery["seconds"]
        final["exit_code"] = 0
        path.write_text(json.dumps(final, indent=2) + "\n")
    print("NB4 recovery exit:", exit_code, "original evidence:", history, flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
