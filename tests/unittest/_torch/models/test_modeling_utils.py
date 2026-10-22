# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_utils
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
