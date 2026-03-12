import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper
from tensorrt_llm._torch.models.modeling_utils import (
    _load_weights_impl,
    _load_weights_impl_v2,
    maybe_alias_or_copy_tensor,
)


class DummyConfig(PretrainedConfig):
    def __init__(self):
        super().__init__()
        self.architectures = ["DummyAliasModel"]
        self.tie_word_embeddings = False
        self.num_attention_heads = 1
        self.num_key_value_heads = 1


class TinyWeightModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_config = ModelConfig(pretrained_config=DummyConfig())
        self.config = self.model_config.pretrained_config
        self.layer = nn.Linear(4, 3, bias=False)


class DummyWeightMapper(BaseWeightMapper):
    def map_weights(self) -> None:
        self._mapping = {}

    def apply_callbacks(
        self, module: nn.Module, module_name: str, module_names_breakdown: list[str], weights: dict
    ) -> list[dict]:
        raise AssertionError("apply_callbacks should not be reached in this test")


def _configure_cpu_weight_load(monkeypatch):
    monkeypatch.setenv("TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL", "1")
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_args, **_kwargs: None)


def test_maybe_alias_or_copy_tensor_aliases_matching_tensor():
    dest = nn.Parameter(torch.zeros((2, 3), dtype=torch.float32))
    src = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    aliased = maybe_alias_or_copy_tensor(dest, src)

    assert aliased is True
    assert dest.data_ptr() == src.data_ptr()
    assert torch.equal(dest, src)


def test_maybe_alias_or_copy_tensor_copies_noncontiguous_tensor():
    dest = nn.Parameter(torch.zeros((2, 3), dtype=torch.float32))
    src = torch.arange(12, dtype=torch.float32).reshape(2, 6)[:, ::2]

    aliased = maybe_alias_or_copy_tensor(dest, src)

    assert aliased is False
    assert dest.data_ptr() != src.data_ptr()
    assert torch.equal(dest, src)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_maybe_alias_or_copy_tensor_aliases_matching_gpu_tensor():
    dest = nn.Parameter(torch.zeros((2, 3), dtype=torch.float32, device="cuda"))
    src = torch.arange(6, dtype=torch.float32, device="cuda").reshape(2, 3)

    aliased = maybe_alias_or_copy_tensor(dest, src)

    assert aliased is True
    assert dest.data_ptr() == src.data_ptr()
    assert torch.equal(dest, src)


def test_load_weights_impl_aliases_matching_tensor(monkeypatch):
    _configure_cpu_weight_load(monkeypatch)
    model = TinyWeightModel()
    src = torch.arange(model.layer.weight.numel(), dtype=model.layer.weight.dtype).reshape_as(
        model.layer.weight
    )

    _load_weights_impl(model, {"layer.weight": src})

    assert model.layer.weight.data_ptr() == src.data_ptr()
    assert torch.equal(model.layer.weight, src)


def test_load_weights_impl_v2_aliases_matching_tensor(monkeypatch):
    _configure_cpu_weight_load(monkeypatch)
    model = TinyWeightModel()
    mapper = DummyWeightMapper()
    src = torch.arange(model.layer.weight.numel(), dtype=model.layer.weight.dtype).reshape_as(
        model.layer.weight
    )

    _load_weights_impl_v2(model, {"layer.weight": src}, mapper)

    assert model.layer.weight.data_ptr() == src.data_ptr()
    assert torch.equal(model.layer.weight, src)
