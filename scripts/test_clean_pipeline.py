import ast
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from build_clean_colab import render
from run_clean_pipeline import execution_record, existing_artifacts, pipeline_command, pipeline_environment


def test_clean_run_refuses_preexisting_training_or_evaluation_artifacts(tmp_path):
    assert not existing_artifacts(tmp_path)
    result = tmp_path / "data/eval/side_by_side.jsonl"
    result.parent.mkdir(parents=True)
    result.write_text("{}\n")
    assert "data/eval/side_by_side.jsonl" in existing_artifacts(tmp_path)


def test_clean_colab_uses_4b_unquantized_judges_and_exports_no_weights():
    notebook = render()
    cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for cell in cells:
        ast.parse(cell)
    assert '"BASE_MODEL": "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"' in cells[0]
    assert '"JUDGE_RM_4BIT": "0"' in cells[0]
    assert '"PREF_TRAIN": "800"' in cells[0]
    assert '"PREF_EVAL": "100"' in cells[0]
    assert '"SFT_SLICE": "1000"' in cells[0]
    assert "scripts/run_clean_pipeline.py" in cells[1]
    assert "value >= 0.8" in cells[2]
    assert '"*.safetensors"' not in cells[2]
    assert '"models/sft-merged": ["config.json", "generation_config.json"]' in cells[2]
    assert '"weights_exported": False' in cells[2]
    assert "both_judges_pass" in cells[2]
    assert "scripts/setup_t4_environment.py" in cells[0]
    assert '[LAB_PY, "-u", "scripts/run_clean_pipeline.py"]' in cells[1]
    assert '"JUDGE_RM_DTYPE": "float32"' in cells[0]
    assert 'assert preflight[name] >= 0.8' in cells[0]
    assert "scripts/export_training_checkpoint.py" in cells[0]
    assert "to notebooks/03_dpo_train.ipynb" in cells[1]
    assert "day22-training-checkpoint-" in cells[1]
    assert "files.download(str(checkpoint))" in cells[1]


def test_clean_colab_file_matches_generator():
    path = Path(__file__).resolve().parents[1] / "colab/Lab22_T4_CLEAN_RUN.ipynb"
    assert json.loads(path.read_text(encoding="utf-8")) == render()


def test_export_can_repeat_without_duplicate_files_or_hashing_its_old_manifest(tmp_path, monkeypatch):
    import hashlib
    import types
    import zipfile

    monkeypatch.chdir(tmp_path)
    evidence = tmp_path / "submission/evidence"
    reference = tmp_path / "models/sft-merged"
    evidence.mkdir(parents=True)
    reference.mkdir(parents=True)
    (evidence / "pipeline.json").write_text(json.dumps({
        "exit_code": 0, "fresh_artifacts": True,
        "settings": {"model": "test-model", "judge_4bit": "0"},
    }))
    (reference / "config.json").write_text('{"model_type":"qwen3"}')
    (reference / "model.safetensors").write_bytes(b"private weight fixture")
    (tmp_path / ".env").write_text("PRIVATE_FIXTURE=do-not-export")
    downloads = []
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(download=downloads.append)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    # Redirect the Colab archive path to the isolated fixture directory.
    source = "".join(render()["cells"][6]["source"]).replace(
        'Path("/content/day22-t4-evidence.zip")', 'Path("day22-t4-evidence.zip")',
    )
    context = {
        "Path": Path, "LAB_PY": sys.executable,
        "subprocess": types.SimpleNamespace(run=lambda *args, **kwargs: None),
    }
    for _ in range(2):
        exec(source, context)
        with zipfile.ZipFile(tmp_path / "day22-t4-evidence.zip") as archive:
            assert archive.testzip() is None
            names = archive.namelist()
            assert len(names) == len(set(names))
            assert not any(name.endswith(".safetensors") or name == ".env" for name in names)
            manifest = json.loads(archive.read("submission/evidence/export-sha256.json"))
            assert set(manifest) == set(names) - {"submission/evidence/export-sha256.json"}
            assert all(hashlib.sha256(archive.read(name)).hexdigest() == digest
                       for name, digest in manifest.items())
            saved = json.loads(archive.read("submission/evidence/sft-reference.json"))
            assert saved["weight_files"][0]["sha256"] == hashlib.sha256(b"private weight fixture").hexdigest()
            assert saved["weights_exported"] is False
    assert downloads == ["day22-t4-evidence.zip"] * 2


def test_pipeline_command_does_not_assume_tools_are_beside_python():
    python = "/usr/bin/python3"
    values = dict(argument.split("=", 1) for argument in pipeline_command(python)[1:-1])
    assert shlex.split(values["JUPYTEXT"]) == [python, "-m", "jupytext"]
    assert shlex.split(values["JUPYTER"]) == [python, "-m", "jupyter"]
    assert pipeline_command(python)[-1] == "pipeline"


def test_pipeline_command_preserves_interpreter_paths_with_spaces():
    python = "/opt/colab environment/bin/python"
    values = dict(argument.split("=", 1) for argument in pipeline_command(python)[1:-1])
    assert shlex.split(values["PY"]) == [python]
    assert shlex.split(values["JUPYTEXT"]) == [python, "-m", "jupytext"]
    assert shlex.split(values["JUPYTER"]) == [python, "-m", "jupyter"]


@pytest.mark.skipif(shutil.which("make") is None, reason="GNU make is required")
def test_module_launchers_execute_a_notebook_through_original_makefile(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    shutil.copyfile(root / "Makefile", tmp_path / "Makefile")
    notebooks = tmp_path / "notebooks"
    notebooks.mkdir()
    (notebooks / "00_dpo_loss_from_scratch.py").write_text(
        '# %%\nprint("module launcher passed")\n', encoding="utf-8",
    )
    command = pipeline_command(sys.executable)
    command[-1] = "nb0"
    inherited = tmp_path / "host-ipython/profile_default"
    inherited.mkdir(parents=True)
    (inherited / "ipython_kernel_config.py").write_text(
        "c = get_config()\nc.IPKernelApp.kernel_class = 'host_only.missing.Kernel'\n",
    )
    monkeypatch.setenv("IPYTHONDIR", str(inherited.parent))
    subprocess.run(command, cwd=tmp_path, env=pipeline_environment(sys.executable, tmp_path),
                   check=True, capture_output=True, text=True, timeout=120)
    notebook = json.loads((notebooks / "00_dpo_loss_from_scratch.ipynb").read_text(encoding="utf-8"))
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert cells[0]["execution_count"] == 1
    assert "module launcher passed" in "".join(cells[0]["outputs"][0]["text"])


def test_execution_record_does_not_mark_a_partial_notebook_successful(tmp_path):
    import datetime

    folder = tmp_path / "notebooks"
    folder.mkdir()
    (folder / "example.py").write_text("print('test')\n")
    (folder / "example.ipynb").write_text(json.dumps({"cells": [
        {"cell_type": "code", "source": ["print('test')"], "execution_count": None, "outputs": []},
    ]}))
    record = execution_record(tmp_path, "nb0", "example", datetime.datetime.now(datetime.timezone.utc))
    assert record["exit_code"] == 1
