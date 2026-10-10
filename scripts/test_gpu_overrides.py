import sys
from types import SimpleNamespace

import pytest
import torch

from lab22 import judge


def fake_reward_stack(monkeypatch):
    calls = {}

    def load_model(name, **kwargs):
        calls.update(name=name, **kwargs)
        return SimpleNamespace(eval=lambda: SimpleNamespace(device="cpu"))

    def load_tokenizer(name):
        return SimpleNamespace()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=load_model),
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
        BitsAndBytesConfig=lambda **kwargs: kwargs,
    ))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    return calls


def test_reward_model_quantization_is_opt_in(monkeypatch):
    calls = fake_reward_stack(monkeypatch)
    monkeypatch.delenv("JUDGE_RM_4BIT", raising=False)
    monkeypatch.delenv("JUDGE_RM_MAX_LENGTH", raising=False)
    assert callable(judge.make_rm_scorer("test-reward-model"))
    assert "quantization_config" not in calls


def test_reward_model_nf4_keeps_same_judge(monkeypatch):
    calls = fake_reward_stack(monkeypatch)
    monkeypatch.setenv("JUDGE_RM_4BIT", "1")
    monkeypatch.setenv("JUDGE_RM_MAX_LENGTH", "2048")
    assert callable(judge.make_rm_scorer("test-reward-model"))
    assert calls["name"] == "test-reward-model"
    assert calls["quantization_config"]["load_in_4bit"] is True
    assert calls["quantization_config"]["bnb_4bit_quant_type"] == "nf4"


def test_reward_model_rejects_nonpositive_context(monkeypatch):
    monkeypatch.setenv("JUDGE_RM_MAX_LENGTH", "0")
    with pytest.raises(ValueError, match="positive"):
        judge.make_rm_scorer("unused")


def test_reward_model_defaults_to_fp32_on_gpu_without_native_bf16(monkeypatch):
    calls = fake_reward_stack(monkeypatch)
    monkeypatch.delenv("JUDGE_RM_DTYPE", raising=False)
    monkeypatch.delenv("JUDGE_RM_4BIT", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    judge.make_rm_scorer("test-reward-model")
    assert calls["dtype"] == torch.float32
    assert calls["device_map"] == "auto"


def test_reward_model_fp32_offload_reserves_memory_on_t4(monkeypatch):
    calls = fake_reward_stack(monkeypatch)
    monkeypatch.setenv("JUDGE_RM_DTYPE", "float32")
    monkeypatch.setenv("JUDGE_RM_DEVICE_MAP", "auto")
    monkeypatch.setenv("JUDGE_RM_MAX_GPU_GIB", "10")
    monkeypatch.setenv("JUDGE_RM_MAX_CPU_GIB", "6")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (14 * 1024**3, 15 * 1024**3))
    judge.make_rm_scorer("test-reward-model")
    assert calls["dtype"] == torch.float32
    assert calls["max_memory"] == {0: 10 * 1024**3, "cpu": 6 * 1024**3}


def test_reward_model_rejects_unknown_dtype(monkeypatch):
    monkeypatch.setenv("JUDGE_RM_DTYPE", "unknown")
    with pytest.raises(ValueError, match="JUDGE_RM_DTYPE"):
        judge.make_rm_scorer("unused")
