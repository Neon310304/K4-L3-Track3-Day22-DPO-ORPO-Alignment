"""Back up real LoRA weights and evidence locally before NB4; never publish this ZIP."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import zipfile
from pathlib import Path


PATTERNS = {
    "adapters/sft-mini": ("*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"),
    "adapters/dpo": ("*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"),
    "models/sft-merged": ("*.json", "*.model", "*.txt", "*.jinja"),
    "notebooks": ("0[0-3]_*.ipynb",),
    "data/pref": ("*.parquet", "*.json"),
    "submission/screenshots": ("*.png",),
    "submission/evidence": ("*.json", "*.log", "pip-freeze.txt"),
}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def export_checkpoint(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Keep existing backup: {destination}")
    for adapter in ("sft-mini", "dpo"):
        folder = root / "adapters" / adapter
        if not (folder / "adapter_config.json").is_file() or not (folder / "adapter_model.safetensors").is_file():
            raise ValueError(f"Completed adapter missing: {adapter}")
    paths = set()
    for folder, patterns in PATTERNS.items():
        for pattern in patterns:
            paths.update(p for p in (root / folder).glob(pattern) if p.is_file())
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Backup path escapes workspace: {path}")
    files = {
        path.relative_to(root).as_posix(): {"bytes": path.stat().st_size, "sha256": digest(path)}
        for path in sorted(paths)
    }
    manifest = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "Private local recovery backup after NB3, before final NB4 results",
        "complete_submission": False,
        "contains_lora_weights": True,
        "contains_merged_weights": False,
        "recovery_note": "Reload the pinned base model and SFT LoRA with the recorded environment, then recreate merged_16bit before loading DPO. This archive alone is not a merged policy checkpoint.",
        "files": files,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Exclusive creation prevents a concurrent process from replacing a prior backup.
        with destination.open("xb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
            for path in sorted(paths):
                archive.write(path, arcname=path.relative_to(root).as_posix())
            archive.writestr("training-backup-manifest.json", json.dumps(manifest, indent=2) + "\n")
    except FileExistsError:
        raise
    except Exception:
        # Remove only this incomplete file, never an earlier completed backup.
        destination.unlink(missing_ok=True)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export_checkpoint(args.root, args.output)
    print(f"Saved private recovery backup: {args.output}; {len(result['files'])} files; includes LoRA weights.")
    print("Keep outside GitHub and the weight-free submission ZIP.")


if __name__ == "__main__":
    main()
