import ast
import math
from pathlib import Path

import pytest
import torch


def student_loss():
    source = Path(__file__).resolve().parents[1] / "notebooks/00_dpo_loss_from_scratch.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "my_dpo_loss")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["my_dpo_loss"]


def test_student_loss_matches_reference_and_log_two():
    from lab22.dpo_math import dpo_loss

    chosen = torch.tensor([-10.0, -20.0])
    rejected = torch.tensor([-15.0, -30.0])
    assert student_loss()(chosen, rejected, chosen, rejected).item() == pytest.approx(math.log(2), abs=1e-6)
    expected = dpo_loss(chosen + 1, rejected - 2, chosen, rejected, beta=0.5)[0]
    assert torch.allclose(student_loss()(chosen + 1, rejected - 2, chosen, rejected, beta=0.5), expected)


def test_student_loss_backpropagates_and_is_stable():
    chosen = torch.tensor([-1000.0, 1000.0], requires_grad=True)
    rejected = torch.zeros(2, requires_grad=True)
    loss = student_loss()(chosen, rejected, torch.zeros(2), torch.zeros(2), beta=1.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(chosen.grad).all() and torch.isfinite(rejected.grad).all()
    assert chosen.grad[0] < 0 and rejected.grad[0] > 0
