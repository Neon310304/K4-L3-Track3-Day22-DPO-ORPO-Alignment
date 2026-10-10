"""Generate a small Colab launcher for the exact 4B core make pipeline."""

import argparse
import json
from pathlib import Path

from build_colab import code, md

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "colab/Lab22_T4_CLEAN_RUN.ipynb"
SOURCE_COMMIT = "53a00376e779fbf06471dfeb42032027bfd56653"
REPOSITORY = "https://github.com/Neon310304/K4-L3-Track3-Day22-DPO-ORPO-Alignment.git"


def render():
    setup = f'''import os
import subprocess
import sys
from pathlib import Path

REPOSITORY = {REPOSITORY!r}
SOURCE_COMMIT = {SOURCE_COMMIT!r}
WORK = Path("/content/day22-clean-t4")
if WORK.exists():
    raise RuntimeError(
        "Workspace already exists. Start a new T4 session; do not reuse old results."
    )
subprocess.run(["nvidia-smi"], check=True)
subprocess.run(["git", "clone", "--no-checkout", REPOSITORY, str(WORK)], check=True)
sources = ["lab22", "scripts", "Makefile", "requirements.txt", ".gitignore", ".gitattributes",
           "notebooks/*.py", "submission/screenshots/README.md"]
subprocess.run(["git", "checkout", SOURCE_COMMIT, "--", *sources], cwd=WORK, check=True)
subprocess.run(["git", "update-ref", "HEAD", SOURCE_COMMIT], cwd=WORK, check=True)
os.chdir(WORK)
os.environ.update({{
    "COMPUTE_TIER": "T4",
    "BASE_MODEL": "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
    "MAX_LEN": "768", "SFT_SLICE": "1000", "PREF_TRAIN": "800", "PREF_EVAL": "100",
    "DPO_BETA": "0.1", "DPO_LR": "5e-6", "DPO_EPOCHS": "1", "SEED": "42",
    "GEN_MAX_NEW_TOKENS": "384", "GEN_BATCH_SIZE": "1", "JUDGE_PROMPTS": "50",
    "JUDGE_PROVIDER": "rm", "JUDGE_RM_4BIT": "0", "JUDGE_RM_MAX_LENGTH": "4096",
    "JUDGE_RM_DTYPE": "float32", "JUDGE_RM_DEVICE_MAP": "auto",
    "JUDGE_RM_MAX_GPU_GIB": "11", "JUDGE_RM_MAX_CPU_GIB": "6",
    "JUDGE_RM_MODELS": "Skywork/Skywork-Reward-V2-Qwen3-4B,Skywork/Skywork-Reward-V2-Llama-3.2-3B",
    "DPO_SAVE_ON_CPU": "0", "MPLBACKEND": "Agg", "TOKENIZERS_PARALLELISM": "false",
    "HF_HOME": "/content/day22-hf-cache", "HF_HUB_DISABLE_XET": "1",
}})
(WORK / "submission/evidence").mkdir(parents=True, exist_ok=True)
Path(os.environ["HF_HOME"], "hub").mkdir(parents=True, exist_ok=True)
print("Fresh source-only checkout:", WORK, "at", SOURCE_COMMIT)
'''
    for relative in ("scripts/run_clean_pipeline.py", "scripts/diagnose_reward_model.py", "lab22/judge.py",
                     "scripts/setup_t4_environment.py"):
        body = (ROOT / relative).read_text(encoding="utf-8")
        setup += f"\n(WORK / {relative!r}).write_text({body!r}, encoding='utf-8')\n"
    setup += '''
LAB_PY = "/content/day22-venv/bin/python"
command = [sys.executable, "-u", "scripts/setup_t4_environment.py"]
with Path("submission/evidence/setup.log").open("w", encoding="utf-8") as log:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    if process.wait():
        raise RuntimeError("T4 environment setup failed; inspect submission/evidence/setup.log")
subprocess.run([LAB_PY, "-c", "import torch; assert torch.cuda.is_available(); "
                "gpu = torch.cuda.get_device_properties(0); "
                "assert 'T4' in gpu.name and gpu.total_memory >= 14 * 1024**3; print(gpu)"], check=True)
subprocess.run([LAB_PY, "scripts/capture_environment.py"], check=True)
with open("submission/evidence/pip-freeze.txt", "w") as stream:
    subprocess.run([LAB_PY, "-m", "pip", "freeze"], stdout=stream, check=True)
smoke = subprocess.run([LAB_PY, "scripts/verify.py", "--smoke"], capture_output=True, text=True)
Path("submission/evidence/smoke.log").write_text(smoke.stdout + smoke.stderr, encoding="utf-8")
print(smoke.stdout, smoke.stderr, flush=True)
smoke.check_returncode()
preflight = {}
for label, name in zip(("qwen", "llama"), os.environ["JUDGE_RM_MODELS"].split(",")):
    output = Path(f"submission/evidence/judge-preflight-{label}.json")
    command = [LAB_PY, "-u", "scripts/diagnose_reward_model.py", "--model", name, "--output", str(output)]
    with Path(f"submission/evidence/judge-preflight-{label}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise RuntimeError(f"Judge preflight failed: {name}; inspect saved log.")
    import json
    preflight[name] = json.loads(output.read_text(encoding="utf-8"))["sanity_accuracy"]
    assert preflight[name] >= 0.8, "Judge sanity must pass the unchanged 80% threshold before training."
print("Setup and both judge preflights passed:", preflight, flush=True)
'''
    run = '''assert len(preflight) == 2 and all(value >= 0.8 for value in preflight.values())
process = subprocess.Popen([LAB_PY, "-u", "scripts/run_clean_pipeline.py"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in process.stdout:
    print(line, end="", flush=True)
PIPELINE_EXIT_CODE = process.wait()
print("make pipeline exit code:", PIPELINE_EXIT_CODE, flush=True)
print("Continue to the export cell even if a stage failed, so the actual log is kept.", flush=True)
'''
    export = '''import hashlib
import json
import zipfile

subprocess.run([LAB_PY, "scripts/capture_environment.py"], check=True)
pipeline = json.loads(Path("submission/evidence/pipeline.json").read_text())
summary_path = Path("data/eval/judge_summary.json")
sanity = json.loads(summary_path.read_text()).get("sanity", {}) if summary_path.exists() else {}
result = {
    "pipeline_exit_code": pipeline.get("exit_code"),
    "fresh_artifacts": pipeline["fresh_artifacts"],
    "base_model": pipeline["settings"]["model"],
    "judge_nf4": pipeline["settings"]["judge_4bit"] == "1",
    "sanity": sanity,
    "both_judges_pass": len(sanity) == 2 and all(value >= 0.8 for value in sanity.values()),
    "reflection_status": "Needs update from this run; old 0.6B report was not copied.",
}
Path("submission/evidence/t4-result-status.json").write_text(json.dumps(result, indent=2) + "\\n")
print(json.dumps(result, indent=2))
print("If a judge still fails sanity, do not lower the threshold or claim the issue is fixed.")
paths = []
for folder, patterns in {
    "notebooks": ["*.ipynb"], "submission/screenshots": ["*.png"],
    "submission/evidence": ["*"], "data/eval": ["*.json", "*.jsonl"],
    "data/pref": ["*.parquet", "*.json"], "adapters/dpo": ["*.json"],
    "adapters/sft-mini": ["adapter_config.json", "sft_metrics.json"],
    "models/sft-merged": ["config.json", "generation_config.json"],
}.items():
    for pattern in patterns:
        paths.extend(path for path in Path(folder).glob(pattern) if path.is_file())
paths = sorted(set(paths))
reference = Path("models/sft-merged")
if (reference / "config.json").exists():
    weights = []
    for weight in sorted(reference.iterdir()):
        if weight.suffix not in {".safetensors", ".bin"}:
            continue
        digest = hashlib.sha256()
        with weight.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        weights.append({"name": weight.name, "bytes": weight.stat().st_size, "sha256": digest.hexdigest()})
    reference_evidence = Path("submission/evidence/sft-reference.json")
    reference_evidence.write_text(json.dumps({
        "path": str(reference.resolve()),
        "config_sha256": hashlib.sha256((reference / "config.json").read_bytes()).hexdigest(),
        "weight_files": weights, "weights_exported": False,
    }, indent=2) + "\\n")
    paths.append(reference_evidence)
hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
manifest_path = Path("submission/evidence/export-sha256.json")
manifest_path.write_text(json.dumps(hashes, indent=2) + "\\n")
paths.append(manifest_path)
archive = Path("/content/day22-t4-evidence.zip")
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
    for path in paths:
        zipped.write(path, arcname=str(path))
print("Saved", archive, "with", len(paths), "files; no .env/cache/model weights.")
from google.colab import files
files.download(str(archive))
print("Also save this executed launcher notebook via File > Download > .ipynb.")
'''
    return {
        "cells": [
            md("# Day22 — chạy sạch Qwen3-4B trên Colab T4\n\n"
               "Chọn **Runtime → Change runtime type → T4 GPU**, rồi **Run all**. "
               "Notebook chỉ chạy NB0–NB4 bằng `make pipeline` gốc, giữ 1.000 SFT và 800/100 preference. "
               "Hai judge chạy FP32/offload, không NF4; setup kiểm score hữu hạn và sanity 80% trước training.\n\n"
               "Không nhập API key. Kết quả mới được giữ riêng, không dùng báo cáo/số liệu 0.6B cũ. "
               "Sau khi chạy, tải ZIP và gửi về để cập nhật REFLECTION, kiểm tra gatekeeper và hoàn thiện bài. "
               "Nếu runtime bị ngắt, giữ log; khởi tạo session mới để chứng minh lượt chạy sạch."),
            md("## 1. Tạo nguồn sạch và môi trường mới"), code(setup),
            md("## 2. Chạy toàn bộ pipeline bắt buộc (có thể mất vài giờ)"), code(run),
            md("## 3. Kiểm trạng thái và tải ZIP bằng chứng (chạy cả khi pipeline lỗi)"), code(export),
        ],
        "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    notebook = render()
    if args.check:
        return 0 if TARGET.exists() and json.loads(TARGET.read_text(encoding="utf-8")) == notebook else 1
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(TARGET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
