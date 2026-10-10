import ast
import json
import math
from pathlib import Path

import pytest
from build_rejudge_colab import render
from rejudge_outputs import evaluate, load_outputs

from lab22 import judge


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), None])
def test_invalid_reward_scores_fail_instead_of_becoming_ties(invalid):
    verdict = judge.rm_record(invalid, 1.0)
    assert verdict["winner"] == "failed"
    assert verdict["sft_score"] is None
    json.dumps(verdict, allow_nan=False)


def test_sanity_rejects_nonfinite_scores():
    with pytest.raises(ValueError, match="Nonfinite"):
        judge.sanity_accuracy(lambda prompt, answer: float("nan"))


def test_rejudge_preserves_answers_and_filters_low_sanity_judge():
    records = [{"id": "example", "category": "heldout", "prompt": "question", "sft": "old", "dpo": "new"}]
    good = {answer for prompt, answer, bad in judge.SANITY_PAIRS}

    def factory(name):
        def score(prompt, answer):
            if prompt == "question":
                return float(answer == "new")
            return float((answer in good) == (name == "good-judge"))
        return score

    raw, summary = evaluate(records, ["bad-judge", "good-judge"], factory)
    assert summary["sanity"] == {"bad-judge": 0.0, "good-judge": 1.0}
    assert summary["judge"] == "rm-panel:good-judge"
    assert raw["records"][0]["winner"] == "dpo"
    assert all(raw["records"][0][key] == value for key, value in records[0].items())
    assert len(raw["sanity_details"]["good-judge"]) == len(judge.SANITY_PAIRS)
    json.dumps(raw, allow_nan=False)


def test_rejudge_does_not_publish_nonfinite_scores_or_all_failed_sanity():
    with pytest.raises(ValueError, match="nonfinite"):
        evaluate([], ["invalid"], lambda name: lambda prompt, answer: math.nan)
    with pytest.raises(ValueError, match="No judge passes"):
        evaluate([], ["constant"], lambda name: lambda prompt, answer: 0.0)


def test_rejudge_refuses_incomplete_saved_answers(tmp_path):
    path = tmp_path / "answers.jsonl"
    path.write_text(json.dumps({"id": "example", "category": "heldout"}) + "\n")
    with pytest.raises(ValueError, match="8 fixed"):
        load_outputs(path)


def test_rejudge_notebook_has_no_training_or_answer_regeneration():
    notebook = render()
    cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for source in cells:
        ast.parse(source)
    assert '"JUDGE_RM_DTYPE": "float32"' in cells[0]
    assert '"JUDGE_RM_4BIT": "0"' in cells[0]
    assert "scripts/rejudge_outputs.py" in cells[1]
    assert "make pipeline" not in cells[1]
    assert "*.safetensors" not in cells[2]
    target = Path(__file__).resolve().parents[1] / "colab/Lab22_REJUDGE_ONLY.ipynb"
    assert json.loads(target.read_text(encoding="utf-8")) == notebook
