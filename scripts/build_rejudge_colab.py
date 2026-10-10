"""Build a rejudge-only notebook for the existing T4 workspace."""

import argparse
import json
from pathlib import Path

from build_colab import code, md

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "colab/Lab22_REJUDGE_ONLY.ipynb"
CELL_TARGET = ROOT / "colab/Lab22_REJUDGE_CELL.txt"


def render():
    setup = '''import os
import subprocess
import sys
from pathlib import Path

WORK = Path("/content/day22-clean-t4")
assert (WORK / "data/eval/side_by_side.jsonl").exists(), (
    "Use the existing T4 runtime, or restore your saved evidence ZIP and dependencies first."
)
os.chdir(WORK)
subprocess.run(["nvidia-smi"], check=True)
os.environ.update({
    "JUDGE_RM_DTYPE": "float32", "JUDGE_RM_DEVICE_MAP": "auto", "JUDGE_RM_4BIT": "0",
    "JUDGE_RM_MAX_GPU_GIB": "10", "JUDGE_RM_MAX_CPU_GIB": "6", "JUDGE_RM_MAX_LENGTH": "4096",
    "JUDGE_RM_OFFLOAD_DIR": "/content/day22-rm-offload", "TOKENIZERS_PARALLELISM": "false",
})
print("Rejudge only: exact saved answers, no policy loading, generation or training.")
'''
    for relative in ("lab22/judge.py", "scripts/rejudge_outputs.py"):
        body = (ROOT / relative).read_text(encoding="utf-8")
        setup += f"\n(WORK / {relative!r}).write_text({body!r}, encoding='utf-8')\n"
    run = '''command = [sys.executable, "scripts/rejudge_outputs.py"]
log_path = Path("submission/evidence/rejudge.log")
log_path.parent.mkdir(parents=True, exist_ok=True)
with log_path.open("w", encoding="utf-8") as log:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    REJUDGE_EXIT = process.wait()
print("Rejudge exit code:", REJUDGE_EXIT)
print("0 = both pass sanity; 2 = measured but at least one fails; other errors: inspect log.")
print("Run the export cell even if scoring failed, to preserve the actual evidence.")
'''
    export = '''import hashlib
import json
import zipfile
from google.colab import files

root = Path("data/eval/rechecked")
status = json.loads((root / "recheck.json").read_text()) if (root / "recheck.json").exists() else {}
print(json.dumps(status, indent=2))
print("Do not claim both judges passed unless both_judges_pass is actually true.")
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
    Path("submission/evidence/sft-reference.json").write_text(json.dumps({
        "path": str(reference.resolve()),
        "config_sha256": hashlib.sha256((reference / "config.json").read_bytes()).hexdigest(),
        "weight_files": weights, "weights_exported": False,
    }, indent=2) + "\\n")
else:
    print("WARNING: original merged SFT config is missing; do not fabricate a config to pass verify.")
paths = []
for folder, patterns in {
    "notebooks": ["*.ipynb"], "submission/screenshots": ["*.png"],
    "submission/evidence": ["*"], "data/eval": ["*.json", "*.jsonl"],
    "data/eval/rechecked": ["*.json"], "data/pref": ["*.parquet", "*.json"],
    "adapters/dpo": ["adapter_config.json", "dpo_metrics.json", "split.json"],
    "adapters/sft-mini": ["adapter_config.json", "sft_metrics.json"],
    "models/sft-merged": ["config.json", "generation_config.json"],
    "lab22": ["judge.py"], "scripts": ["rejudge_outputs.py"],
}.items():
    for pattern in patterns:
        paths.extend(path for path in Path(folder).glob(pattern) if path.is_file())
paths = sorted(set(paths))
manifest = Path("submission/evidence/recheck-export-sha256.json")
hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if path != manifest}
manifest.write_text(json.dumps(hashes, indent=2) + "\\n")
paths = sorted(set(paths + [manifest]))
archive = Path("/content/day22-t4-recheck-evidence.zip")
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
    for path in paths:
        zipped.write(path, arcname=str(path))
print("Saved", archive, "with", len(paths), "files. This is evidence, not the final submission ZIP.")
files.download(str(archive))
print("Also save this executed notebook: File > Download > .ipynb.")
'''
    return {
        "cells": [
            md("# Lab22 — chỉ chấm lại NB4, không train lại\n\n"
               "Dùng **runtime T4 cũ còn giữ kết quả**. Cách đơn giản nhất: mở file "
               "`Lab22_REJUDGE_CELL.txt`, sao chép toàn bộ vào **một cell code mới ở cuối notebook cũ**, "
               "rồi chỉ chạy cell mới. Không Run all, không xoá runtime. "
               "Mở notebook mới có thể tạo runtime khác, không còn workspace đang giữ kết quả.\n\n"
               "Hai judge nạp lần lượt FP32 và offload, không NF4; sanity vẫn 80%. "
               "Chưa biết trước cả hai sẽ đạt. Không sửa/xoá thẻ công cụ trong câu trả lời. "
               "Gửi ZIP recheck về để cập nhật bài cuối; đừng nộp ZIP bằng chứng như bài hoàn chỉnh."),
            md("## 1. Cài bản sửa vào workspace đang có"), code(setup),
            md("## 2. Chấm lại đúng 58 câu đã sinh"), code(run),
            md("## 3. Tải bằng chứng, kể cả khi có lỗi"), code(export),
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
    cell_text = "\n\n".join("".join(cell["source"]) for cell in notebook["cells"]
                             if cell["cell_type"] == "code") + "\n"
    if args.check:
        return 0 if (
            TARGET.exists() and json.loads(TARGET.read_text(encoding="utf-8")) == notebook
            and CELL_TARGET.exists() and CELL_TARGET.read_text(encoding="utf-8") == cell_text
        ) else 1
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    CELL_TARGET.write_text(cell_text, encoding="utf-8")
    print(TARGET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
