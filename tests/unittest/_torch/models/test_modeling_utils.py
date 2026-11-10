# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_utils
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.linear import (
    Linear,
    TensorParallelMode,
    WeightMode,
    WeightsLoadingConfig,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo


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


def _fused_index_projection(declare_names: bool) -> nn.Module:
    """A fused, weightless Linear whose checkpoint parts are index_q_proj / index_k_proj."""
    projection = Linear(
        16,
        24,
        bias=False,
        dtype=torch.bfloat16,
        mapping=Mapping(),
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        weights_loading_config=WeightsLoadingConfig(weight_mode=WeightMode.FUSED_GATE_UP_LINEAR),
        skip_create_weights_in_init=True,
    )
    if declare_names:
        projection.fused_checkpoint_names = ("index_q_proj", "index_k_proj")
    return projection


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "entries, excluded, expected",
    [
        ("matching", (), "declared"),
        ("absent", (), None),
        ("mismatched", (), ValueError),
        ("zero_point", (), ValueError),
        ("missing_k", (), ValueError),
        ("matching", ("index_q_proj", "index_k_proj"), "excluded"),
        ("matching", ("index_q_proj",), ValueError),
        ("matching", ("index_qk_proj",), "excluded"),
    ],
)
def test_quantization_passes_match_declared_fused_checkpoint_names(
    entries: str, excluded: tuple[str, ...], expected: object
) -> None:
    """Per-layer entries and exclusions apply by the declared names, and the parts must agree."""
    prefix = "model.layers.3.self_attn"
    declared = QuantConfig(quant_algo=QuantAlgo.MXFP8, group_size=32)
    other = QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16)
    legacy = QuantConfig(quant_algo=QuantAlgo.FP8)
    quant_config_dict = {
        # The undeclared gate/up projection keeps its by-name match.
        "model.layers.3.mlp.gate_proj": legacy,
        # Another layer's entries never reach this projection.
        "model.layers.2.self_attn.index_q_proj": other,
        "model.layers.2.self_attn.index_k_proj": other,
    }
    if entries != "absent":
        quant_config_dict[f"{prefix}.index_q_proj"] = declared
    k_configs = {
        # Equal to the Q entry but a separate object, as a per-layer map holds it.
        "matching": QuantConfig(quant_algo=QuantAlgo.MXFP8, group_size=32),
        "mismatched": other,
        # Same algorithm and group size; any other differing field still conflicts.
        "zero_point": QuantConfig(quant_algo=QuantAlgo.MXFP8, group_size=32, has_zero_point=True),
    }
    if entries in k_configs:
        quant_config_dict[f"{prefix}.index_k_proj"] = k_configs[entries]
    global_config = QuantConfig(
        quant_algo=QuantAlgo.MIXED_PRECISION,
        kv_cache_quant_algo=QuantAlgo.FP8,
        exclude_modules=[f"{prefix}.{name}" for name in excluded] or None,
    )
    model_config = SimpleNamespace(quant_config=global_config, quant_config_dict=quant_config_dict)
    projection = _fused_index_projection(declare_names=True)
    # Without declared names a fused projection keeps the gate/up name match.
    undeclared = _fused_index_projection(declare_names=False)
    modules = [
        (f"{prefix}.index_qk_proj", projection),
        ("model.layers.3.mlp.gate_up_proj", undeclared),
    ]

    def run_passes() -> None:
        """Run both passes in __post_init__ order."""
        modeling_utils.apply_layerwise_quant_config(model_config, iter(modules))
        modeling_utils.apply_quant_config_exclude_modules(model_config, iter(modules))

    if expected is ValueError:
        with pytest.raises(ValueError, match="one fused projection"):
            run_passes()
        return
    run_passes()
    assert undeclared.quant_config is legacy
    if expected == "declared":
        assert projection.quant_config is declared
    elif expected == "excluded":
        assert projection.quant_config.quant_algo is None
        assert projection.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
    else:
        assert projection.quant_config is None
