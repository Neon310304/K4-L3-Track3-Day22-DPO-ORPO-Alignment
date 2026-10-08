"""Record actual runtime and cached Hub revisions, without credentials."""

import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab22 import config as C


def main():
    import torch

    evidence = ROOT / "submission/evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    packages = ("torch", "torchao", "unsloth", "unsloth_zoo", "trl", "transformers",
                "peft", "bitsandbytes", "datasets", "accelerate", "matplotlib", "jupytext", "nbconvert")
    payload = {
        "python": platform.python_version(), "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0),
        "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        "cuda_runtime": torch.version.cuda,
        "versions": {name: importlib.metadata.version(name) for name in packages},
        "base_model_requested": C.BASE_MODEL,
    }
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    payload["cached_revisions"] = {
        directory.name: {ref.relative_to(directory / "refs").as_posix(): ref.read_text().strip()
                         for ref in (directory / "refs").rglob("*") if ref.is_file()}
        for directory in sorted(cache.iterdir())
        if directory.name.startswith(("models--", "datasets--")) and directory.is_dir()
    }
    (evidence / "environment.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
