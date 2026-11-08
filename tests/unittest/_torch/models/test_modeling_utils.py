# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_utils
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm.mapping import Mapping


def test_timing_metric_accumulates_and_records_failures(monkeypatch) -> None:
    perf_counter_values = iter([1.0, 1.25, 2.0, 2.5])
    monkeypatch.setattr(modeling_utils.time, "perf_counter", lambda: next(perf_counter_values))
    metrics = {}

    with modeling_utils.timing_metric("load_seconds", metrics):
        pass

    with pytest.raises(RuntimeError, match="load failed"):
        with modeling_utils.timing_metric("load_seconds", metrics):
            raise RuntimeError("load failed")

    assert metrics["load_seconds"] == pytest.approx(0.75)


def _tied_embedding_backbone(
    config: ModelConfig[PretrainedConfig],
    mapping: Mapping,
    mode: TensorParallelMode | None,
) -> modeling_utils.DecoderModel:
    backbone = modeling_utils.DecoderModel(config)
    # Layout checks do not run a forward pass or need a GPU collective workspace.
    backbone.embed_tokens = Embedding(
        24, 16, dtype=torch.float32, mapping=mapping, tensor_parallel_mode=mode, reduce_output=False
    )
    return backbone


@pytest.mark.cpu_only
@pytest.mark.parametrize("enable_attention_dp", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_tied_embedding_parallel_layout(enable_attention_dp: bool, rank: int) -> None:
    mapping = Mapping(world_size=2, tp_size=2, rank=rank, enable_attention_dp=enable_attention_dp)
    config = ModelConfig(
        pretrained_config=PretrainedConfig(tie_word_embeddings=True, torch_dtype=torch.float32),
        mapping=mapping,
    )
    backbone = _tied_embedding_backbone(config, mapping, TensorParallelMode.COLUMN)
    model = modeling_utils.DecoderModelForCausalLM(
        backbone, config=config, hidden_size=16, vocab_size=24
    )
    assert model.lm_head.weight is backbone.embed_tokens.weight
    assert model.lm_head.weight.shape == (24 if enable_attention_dp else 12, 16)
    if enable_attention_dp:
        assert model.lm_head.tp_mode is None
        assert backbone.embed_tokens.tp_mode is None
        assert model.lm_head.tp_size == 1
        assert backbone.embed_tokens.tp_size == 2


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "embedding_tp_size,embedding_mode,error",
    [
        (4, TensorParallelMode.COLUMN, "same TP size"),
        (2, TensorParallelMode.ROW, "same TP mode"),
        (1, None, "same TP mode"),
    ],
)
def test_tied_embedding_rejects_incompatible_layout(
    embedding_tp_size: int, embedding_mode: TensorParallelMode | None, error: str
) -> None:
    config = ModelConfig(
        pretrained_config=PretrainedConfig(tie_word_embeddings=True, torch_dtype=torch.float32),
        mapping=Mapping(world_size=2, tp_size=2),
    )
    backbone = _tied_embedding_backbone(
        config, Mapping(world_size=embedding_tp_size, tp_size=embedding_tp_size), embedding_mode
    )
    with pytest.raises(AssertionError, match=error):
        modeling_utils.DecoderModelForCausalLM(
            backbone, config=config, hidden_size=16, vocab_size=24
        )


@pytest.mark.parametrize(
    "name, expected",
    [
        ("model.layers.0.mlp", "model.layers.0.mlp"),
        ("model._orig_mod.layers.0.mlp", "model.layers.0.mlp"),
        ("_orig_mod.model.layers.0.mlp", "model.layers.0.mlp"),
        ("model._orig_mod", "model"),
        ("_orig_mod", ""),
        # Only whole components are the wrapper, never a substring.
        ("model.layers._orig_mod_x.mlp", "model.layers._orig_mod_x.mlp"),
    ],
)
def test_strip_torch_compile_wrapper(name, expected) -> None:
    assert modeling_utils.strip_torch_compile_wrapper(name) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the module walk pins the CUDA device")
def test_load_weights_impl_v2_matches_through_torch_compile_wrapper(monkeypatch) -> None:
    """A refit into a compiled model loads every module, wrapper or not.

    ``torch.compile`` inserts ``_orig_mod`` into the paths below the compiled
    scope; checkpoint keys never carry it. Before the loader stripped it, an
    ``allow_partial_loading=True`` reload matched nothing under the wrapper
    and silently kept the old weights.
    """

    class _Inner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(4, 4)])

    class _Stub(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model_config = ModelConfig(
                pretrained_config=PretrainedConfig(tie_word_embeddings=False), mapping=Mapping()
            )
            self.config = self.model_config.pretrained_config
            self.model = _Inner()
            self.lm_head = nn.Linear(4, 2, bias=False)

    stub = _Stub()
    # Compile the inner scope the way ModelEngine does for a
    # DecoderModelForCausalLM; the wrapper is only walked, never run.
    stub.model = torch.compile(stub.model, backend="eager")
    assert any("_orig_mod" in n for n, _ in stub.named_parameters())

    weights = {
        "model.layers.0.weight": torch.full((4, 4), 2.0),
        "model.layers.0.bias": torch.full((4,), 3.0),
        "lm_head.weight": torch.full((2, 4), 5.0),
    }
    monkeypatch.setenv("TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL", "True")
    mapper = HfWeightMapper()
    mapper.init_model_and_config(stub, stub.model_config)
    modeling_utils._load_weights_impl_v2(stub, weights, mapper, allow_partial_loading=True)

    inner = stub.model._orig_mod
    torch.testing.assert_close(inner.layers[0].weight.detach(), weights["model.layers.0.weight"])
    torch.testing.assert_close(inner.layers[0].bias.detach(), weights["model.layers.0.bias"])
    torch.testing.assert_close(stub.lm_head.weight.detach(), weights["lm_head.weight"])
