"""Install the pinned T4 stack in Python 3.12 without compiling xformers.

The Colab launcher stays on its host Python; nbconvert kernels and GPU jobs
run in this environment. The CUDA 11.8 wheel comes from the PyTorch index.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REQUIREMENTS = [
    "torch==2.7.0+cu118", "torchvision==0.22.0+cu118",
    "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.30-cp312-cp312-manylinux_2_28_x86_64.whl",
    "torchao==0.13.0", "unsloth==2026.10.2", "unsloth_zoo==2026.10.2",
    "trl==1.13.0", "transformers==5.17.0", "peft>=0.18,<1", "accelerate>=1.10,<2",
    "bitsandbytes>=0.48,<1", "datasets>=4.7,<5", "matplotlib>=3.9,<4", "pandas>=2.2,<4",
    "pyarrow>=17", "jupytext>=1.16,<2", "nbconvert>=7,<8", "ipykernel>=6,<8",
    "pytest>=8.3,<10", "lm-eval[ifeval,math]==0.4.13",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=Path("/content/day22-venv"))
    args = parser.parse_args()
    subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
    subprocess.run([
        sys.executable, "-m", "uv", "venv", "--python", "3.12", "--seed", str(args.venv),
    ], check=True)
    python = str(args.venv / "bin/python")
    subprocess.run([
        python, "-m", "pip", "install", "--only-binary=xformers",
        "--extra-index-url", "https://download.pytorch.org/whl/cu118", *REQUIREMENTS,
    ], check=True)
    subprocess.run([python, "-m", "ipykernel", "install", "--prefix", str(args.venv), "--name", "python3"],
                   check=True)
    # Colab's system IPython config selects google.colab._kernel.Kernel, which
    # is unavailable in this isolated interpreter. CLI arguments take priority.
    specification = args.venv / "share/jupyter/kernels/python3/kernel.json"
    kernel = json.loads(specification.read_text(encoding="utf-8"))
    kernel["argv"].insert(1, "--IPKernelApp.kernel_class=ipykernel.ipkernel.IPythonKernel")
    # The option belongs to ipykernel, not the Python interpreter.
    kernel["argv"].remove("--IPKernelApp.kernel_class=ipykernel.ipkernel.IPythonKernel")
    kernel["argv"].append("--IPKernelApp.kernel_class=ipykernel.ipkernel.IPythonKernel")
    specification.write_text(json.dumps(kernel, indent=2) + "\n", encoding="utf-8")
    print("T4 interpreter:", python, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
