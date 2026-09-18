# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Unit and integration tests for the MiniMax-M3 text bring-up.

Helper tests exercise config normalization, layer scheduling, and
routing-method scaling. ``test_text_checkpoint_loading`` loads the real
MiniMax-M3 checkpoint config / tokenizer / chat template, runs the
static keyspace coverage classifier on every key in the checkpoint's
safetensors index, and confirms that each ``language_model.*`` weight
is either mapped to a TRT-LLM text parameter (loaded) or intentionally
ignored with a documented reason. CUDA tests exercise attention module
construction and the multi-rank ADP negative-control path.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig
from utils.llm_data import llm_models_root

import tensorrt_llm._torch.models.modeling_minimaxm3 as modeling_minimaxm3
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import MiniMaxM3MsaSparseAttention
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.common import (
    MiniMaxM3SparseConfig,
    MiniMaxM3SparseMetadataParams,
    MiniMaxM3SparseParams,
    index_head_range,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_indexer import _group_max_reduce
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.minimaxm3_weight_mapper import (
    MiniMaxM3HfWeightMapper,
)
from tensorrt_llm._torch.models.modeling_minimaxm3 import (
    MiniMaxM3Attention,
    MiniMaxM3DecoderLayer,
    MiniMaxM3ForCausalLM,
    MiniMaxM3Model,
    MiniMaxM3MoE,
    MiniMaxM3QKVIndexerLinear,
    _build_swiglu_oai_dense_mlp,
    _load_qkv_index_proj_weights,
    _minimax_m3_swiglu_oai,
    _moe_routed_output_is_global,
    _strip_language_model_prefix,
    _validate_sparse_attention_runtime_config,
    _wrap_dict_as_config,
    get_moe_layer_ids,
    get_sparse_disable_index_value_layer_ids,
    get_sparse_layer_ids,
    get_text_config,
    is_minimax_m3_vl_config,
)
from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM
from tensorrt_llm._torch.models.modeling_utils import _load_weights_impl_v2
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.moe.fused_moe.interface import MoESchedulerKind
from tensorrt_llm._torch.moe.fused_moe.routing import (
    MiniMaxM2MoeRoutingMethod,
    MiniMaxM3MoeRoutingMethod,
)
from tensorrt_llm._torch.utils import AuxStreamType, EventType
from tensorrt_llm.llmapi import MiniMaxM3SparseAttentionConfig, RocketSparseAttentionConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NUM_HIDDEN_LAYERS = 7
_SPARSE_FREQ = [0, 0, 0, 1, 1, 1, 1]
_DISABLE_INDEX_VALUE = [0, 0, 0, 1, 1, 1, 1]
_MOE_LAYER_FREQ = [0, 0, 0, 1, 1, 1, 1]


class _M3CompositionGate(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class _M3CompositionExperts(nn.Module):
    def __init__(self, scheduler_kind: MoESchedulerKind) -> None:
        super().__init__()
        self.backend = SimpleNamespace(scheduler_kind=scheduler_kind)

    def forward(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        del router_logits, kwargs
        return torch.full_like(hidden_states, 3.0)


class _M3CompositionShared(nn.Module):
    def forward(self, hidden_states: torch.Tensor, lora_params: dict | None = None) -> torch.Tensor:
        del lora_params
        return torch.full_like(hidden_states, 2.0)


class _M3CompositionAllReduce(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs = []
        self.params = []

    def forward(
        self, value: torch.Tensor, *, all_reduce_params: object | None = None
    ) -> torch.Tensor:
        self.inputs.append(value.clone())
        self.params.append(all_reduce_params)
        return value * 4


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "scheduler_kind, expected, reduced_input",
    [
        (MoESchedulerKind.EXTERNAL_COMM, 20.0, 5.0),
        (MoESchedulerKind.FUSED_COMM, 11.0, 2.0),
    ],
)
def test_minimax_m3_moe_reduces_only_local_terms(
    monkeypatch: pytest.MonkeyPatch,
    scheduler_kind: MoESchedulerKind,
    expected: float,
    reduced_input: float,
) -> None:
    """An already-global routed result must not enter M3's TP AllReduce."""
    monkeypatch.setattr(
        "tensorrt_llm._torch.models.modeling_minimaxm3.maybe_execute_in_parallel",
        lambda routed, shared, *_args, **_kwargs: (routed(), shared()),
    )
    moe = MiniMaxM3MoE.__new__(MiniMaxM3MoE)
    nn.Module.__init__(moe)
    moe.gate = _M3CompositionGate()
    moe.experts = _M3CompositionExperts(scheduler_kind)
    moe.shared_experts = _M3CompositionShared()
    moe.routed_output_is_global = _moe_routed_output_is_global(moe.experts)
    moe.allreduce = _M3CompositionAllReduce()
    moe.event_dict = {EventType.Main: None, EventType.MoeShared: None}
    moe.aux_stream = None

    hidden_states = torch.ones((2, 4))
    output = moe(hidden_states, SimpleNamespace(all_rank_num_tokens=None))

    torch.testing.assert_close(output, torch.full_like(hidden_states, expected))
    assert len(moe.allreduce.inputs) == 1
    torch.testing.assert_close(
        moe.allreduce.inputs[0], torch.full_like(hidden_states, reduced_input)
    )
    assert moe.allreduce.params == [None]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "scheduler_kind, expected_post_fusion",
    [
        (MoESchedulerKind.EXTERNAL_COMM, True),
        (MoESchedulerKind.FUSED_COMM, False),
    ],
)
def test_minimax_m3_decoder_layer_sets_post_fusion_from_moe_scheduler(
    monkeypatch: pytest.MonkeyPatch,
    scheduler_kind: MoESchedulerKind,
    expected_post_fusion: bool,
) -> None:
    """A fused-communication MoE must not schedule a second boundary reduction."""

    monkeypatch.setenv("TRTLLM_MINIMAX_M3_EAGER_FUSION_DISABLED", "0")

    class _DecoderComponent(nn.Module):
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            super().__init__()

    class _DecoderMoE(_DecoderComponent):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.backend = SimpleNamespace(scheduler_kind=scheduler_kind)
            self.routed_output_is_global = _moe_routed_output_is_global(self)

    monkeypatch.setattr(
        "tensorrt_llm._torch.models.modeling_minimaxm3.MiniMaxM3Attention",
        _DecoderComponent,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.models.modeling_minimaxm3.MiniMaxM3MoE",
        _DecoderMoE,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.models.modeling_minimaxm3.RMSNorm",
        _DecoderComponent,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.models.modeling_minimaxm3.AllReduce",
        _DecoderComponent,
    )
    model_config = ModelConfig(
        pretrained_config=_make_text_config(),
        mapping=Mapping(world_size=2, rank=0, tp_size=2),
    )

    layer = MiniMaxM3DecoderLayer(
        model_config=model_config,
        layer_idx=3,
        aux_stream_dict={
            AuxStreamType.Attention: None,
            AuxStreamType.MoeShared: None,
        },
    )

    assert layer.enable_fusion
    assert layer.pre_feed_forward_fusion
    assert layer.post_feed_forward_fusion is expected_post_fusion


@pytest.mark.parametrize(
    "sparse_attention_config",
    [None, RocketSparseAttentionConfig()],
)
def test_validate_sparse_attention_runtime_config_rejects_wrong_backend(
    sparse_attention_config: MiniMaxM3SparseAttentionConfig | RocketSparseAttentionConfig | None,
) -> None:
    model_config = ModelConfig(
        pretrained_config=_make_text_config(),
        sparse_attention_config=sparse_attention_config,
    )

    with pytest.raises(ValueError, match="algorithm='minimax_m3'"):
        _validate_sparse_attention_runtime_config(model_config)


def test_validate_sparse_attention_runtime_config_accepts_minimax_m3() -> None:
    model_config = ModelConfig(
        pretrained_config=_make_text_config(),
        sparse_attention_config=MiniMaxM3SparseAttentionConfig(),
    )

    _validate_sparse_attention_runtime_config(model_config)


@pytest.mark.cpu_only
def test_validate_fused_projection_requires_fp8_main_kv_cache() -> None:
    sparse_config = MiniMaxM3SparseAttentionConfig(
        implementation="msa",
        indexer_kv_dtype="fp8",
        fuse_qkv_index_projection=True,
    )
    model_config = ModelConfig(
        pretrained_config=_make_text_config(),
        sparse_attention_config=sparse_config,
    )
    with pytest.raises(ValueError, match="requires an FP8 main KV cache"):
        _validate_sparse_attention_runtime_config(model_config)

    model_config.quant_config = QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8)
    _validate_sparse_attention_runtime_config(model_config)


@pytest.mark.cpu_only
@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("attention_dp", [False, True])
def test_fused_qkv_index_projection_preserves_index_head_groups(
    tp_size: int, attention_dp: bool
) -> None:
    cfg = MiniMaxM3SparseAttentionConfig(
        implementation="msa",
        indexer_kv_dtype="fp8",
        fuse_qkv_index_projection=True,
        num_attention_heads=64,
        num_key_value_heads=4,
    )
    sparse_params = cfg.to_sparse_params()
    metadata_params = cfg.to_sparse_metadata_params()
    checkpoint_cfg = cfg.model_copy(
        update={"num_attention_heads": None, "num_key_value_heads": None}
    )
    checkpoint = SimpleNamespace(num_attention_heads=64, num_key_value_heads=4)
    assert checkpoint_cfg.to_sparse_params(pretrained_config=checkpoint) == sparse_params
    assert checkpoint_cfg.to_sparse_metadata_params(pretrained_config=checkpoint) == metadata_params
    mapping = SimpleNamespace(tp_size=tp_size, tp_rank=0, enable_attention_dp=attention_dp)
    local_q_heads = 64 if attention_dp else 64 // tp_size
    local_kv_heads = 4 if attention_dp else max(4 // tp_size, 1)

    assert sparse_params.fuse_qkv_index_projection is True
    assert metadata_params.sharded_head_counts(mapping) == (local_q_heads, local_kv_heads)
    assert metadata_params.num_index_heads == 4
    assert metadata_params.sharded_index_head_count(mapping) == local_kv_heads
    kernel_cfg = MiniMaxM3SparseConfig.from_sparse_params(
        sparse_params, num_q_heads=local_q_heads, num_kv_heads=local_kv_heads, head_dim=128
    )
    assert kernel_cfg.num_index_heads == local_kv_heads

    compatibility_cfg = cfg.model_copy(update={"fuse_qkv_index_projection": False})
    assert compatibility_cfg.to_sparse_metadata_params() == metadata_params
    compatibility_kernel_cfg = MiniMaxM3SparseConfig.from_sparse_params(
        compatibility_cfg.to_sparse_params(),
        num_q_heads=local_q_heads,
        num_kv_heads=local_kv_heads,
        head_dim=128,
    )
    # Rank zero keeps its own index/KV pairs, not a max over other ranks'
    # index heads. Each head favors a distinct block to expose misgrouping.
    scores = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0])).unsqueeze(-1)
    fused_scores = _group_max_reduce(scores[:local_kv_heads], kernel_cfg)
    reference_scores = _group_max_reduce(scores[:local_kv_heads], compatibility_kernel_cfg)
    torch.testing.assert_close(fused_scores, reference_scores)
    expected_blocks = torch.arange(local_kv_heads)
    torch.testing.assert_close(fused_scores.argmax(dim=1).flatten(), expected_blocks)
    with pytest.raises(ValueError, match=r"requires the 'msa' implementation"):
        MiniMaxM3SparseAttentionConfig(implementation="triton", fuse_qkv_index_projection=True)
    with pytest.raises(ValueError, match=r"requires indexer_kv_dtype='fp8'"):
        MiniMaxM3SparseAttentionConfig(implementation="msa", fuse_qkv_index_projection=True)


@pytest.mark.cpu_only
def test_minimax_m3_uses_one_engine_speculative_base() -> None:
    assert issubclass(MiniMaxM3ForCausalLM, SpecDecOneEngineForCausalLM)


@pytest.mark.cpu_only
def test_setup_aliases_preserves_one_engine_draft_weight_loading() -> None:
    loaded = []

    class DraftModel:
        shares_target_kv_cache = True

        def load_weights_from_target_model(self, target) -> None:
            loaded.append(target)

    target = MiniMaxM3ForCausalLM.__new__(MiniMaxM3ForCausalLM)
    layers = [
        SimpleNamespace(input_layernorm=object()),
        SimpleNamespace(input_layernorm=object()),
    ]
    final_norm = object()
    object.__setattr__(target, "draft_model", DraftModel())
    object.__setattr__(target, "model", SimpleNamespace(layers=layers, norm=final_norm))

    target.setup_aliases()

    assert loaded == [target]
    assert layers[0].next_layer_layernorm is layers[1].input_layernorm
    assert layers[1].next_layer_layernorm is final_norm


@pytest.mark.cpu_only
def test_eagle_capture_precedes_next_layer_norm() -> None:
    class CaptureMetadata:
        def __init__(self) -> None:
            self.captured = None

        def is_layer_capture(self, layer_idx: int) -> bool:
            return layer_idx == 25

        def maybe_capture_hidden_states(self, layer_idx, hidden_states, residual) -> None:
            self.captured = (layer_idx, hidden_states.clone(), residual.clone())

    layer = SimpleNamespace(
        layer_idx=25,
        _apply_pre_feed_forward_norm=lambda hidden, residual: (hidden + 1, residual + 2),
        block_sparse_moe=lambda hidden, unused_metadata, **unused_kwargs: hidden + 3,
        _feed_forward_all_reduce_params=lambda: None,
        _apply_next_layer_layernorm=lambda hidden, residual: (hidden + 10, residual + 20),
    )
    spec_metadata = CaptureMetadata()
    hidden_states = torch.tensor([1.0])
    residual = torch.tensor([2.0])

    output, output_residual = MiniMaxM3DecoderLayer.forward_MoE(
        layer,
        hidden_states,
        SimpleNamespace(),
        residual,
        spec_metadata,
    )

    assert spec_metadata.captured is not None
    layer_idx, captured_hidden, captured_residual = spec_metadata.captured
    assert layer_idx == 25
    torch.testing.assert_close(captured_hidden, torch.tensor([5.0]))
    torch.testing.assert_close(captured_residual, torch.tensor([4.0]))
    torch.testing.assert_close(output, torch.tensor([15.0]))
    torch.testing.assert_close(output_residual, torch.tensor([24.0]))


@pytest.mark.cpu_only
def test_piecewise_attention_boundary_runs_horizontal_producer(monkeypatch) -> None:
    class FakeAttentionLayer:
        def __init__(self) -> None:
            self.producer_shapes = None

        def _fused_fp8_qkv_indexer_norm_rope_kv_insert(self, packed, position_ids, attn_metadata):
            self.producer_shapes = (
                tuple(packed.shape),
                tuple(position_ids.shape),
                attn_metadata.num_tokens,
            )
            return packed[:, :3].clone(), packed[:, :1].clone()

        def _dispatch_attention_backend(self, q, k, v, idx_q, idx_k, attn_metadata, output) -> None:
            assert k is None and v is None and idx_k is None
            assert idx_q.shape == (attn_metadata.num_tokens, 1)
            output.copy_(q)

    metadata = SimpleNamespace(num_tokens=2)
    layer = FakeAttentionLayer()
    monkeypatch.setattr(
        modeling_minimaxm3,
        "_extract_minimax_m3_attention_extra_attrs",
        lambda layer_idx: (metadata, layer),
    )
    packed = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    position_ids = torch.arange(4, dtype=torch.int32).reshape(1, 4)
    output = torch.full((4, 3), -1.0)

    modeling_minimaxm3.minimax_m3_attn_custom_op_inplace(
        None,
        None,
        None,
        None,
        None,
        packed,
        position_ids,
        "3",
        output,
    )

    assert layer.producer_shapes == ((2, 5), (1, 2), 2)
    torch.testing.assert_close(output[:2], packed[:2, :3])
    torch.testing.assert_close(output[2:], torch.full((2, 3), -1.0))


@pytest.mark.cpu_only
def test_piecewise_projection_fake_preserves_padded_hidden_rows(monkeypatch) -> None:
    projection = object.__new__(MiniMaxM3QKVIndexerLinear)
    nn.Module.__init__(projection)
    projection.local_output_sizes = (3, 4)
    layer = SimpleNamespace(qkv_proj=projection)
    monkeypatch.setattr(
        modeling_minimaxm3,
        "_extract_minimax_m3_attention_extra_attrs",
        lambda layer_idx: (SimpleNamespace(), layer),
    )
    hidden_states = torch.randn(256, 5)
    position_ids = torch.arange(6).reshape(1, 6)

    packed = modeling_minimaxm3._minimax_m3_qkv_index_proj_fake(hidden_states, position_ids, "3")

    # The real GEMM projects every padded hidden row. Position IDs remain an
    # input solely to carry the unpadded token symbol to the piecewise segment.
    assert packed.shape == (hidden_states.shape[0], 7)


@pytest.mark.cpu_only
def test_piecewise_fused_projection_preserves_input_token_dimension(monkeypatch) -> None:
    """Do not inherit a bucket-specialized token dimension from the GEMM output."""
    packed = torch.randn(2, 7)
    captured = {}

    def fake_boundary(q, k, v, idx_q, idx_k, packed_arg, position_ids, layer_idx, output):
        assert q is None and k is None and v is None
        assert idx_q is None and idx_k is None
        captured["packed"] = packed_arg
        captured["position_ids"] = position_ids
        captured["output_shape"] = tuple(output.shape)
        output.zero_()

    layer = SimpleNamespace(
        enable_fused_qkv_index_projection=True,
        qkv_proj=lambda hidden_states: packed,
        attn=object(),  # Compatibility path, without the captured FP8 producer.
        register_to_config=True,
        num_heads=1,
        head_dim=3,
        attn_activation_dtype=torch.float32,
        layer_idx_str="3",
        o_proj=lambda output, all_reduce_params: output,
    )
    monkeypatch.setattr(
        modeling_minimaxm3,
        "_extract_minimax_m3_attention_extra_attrs",
        lambda layer_idx: (SimpleNamespace(), layer),
    )
    monkeypatch.setattr(modeling_minimaxm3, "is_torch_compiling", lambda: True)
    monkeypatch.setattr(
        modeling_minimaxm3,
        "maybe_bcg_minimax_m3_attn_custom_op_inplace",
        fake_boundary,
    )
    hidden_states = torch.randn(4, 5)
    position_ids = torch.arange(6).reshape(1, 6)

    result = MiniMaxM3Attention._sparse_forward(
        layer,
        position_ids=position_ids,
        hidden_states=hidden_states,
        attn_metadata=SimpleNamespace(),
    )

    assert captured["packed"] is packed
    assert captured["position_ids"] is position_ids
    assert captured["output_shape"] == (position_ids.shape[-1], 3)
    assert result.shape == (position_ids.shape[-1], 3)


@pytest.mark.cpu_only
def test_piecewise_captured_producer_preserves_symbolic_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain an unbacked symbolic token count in both fake query outputs."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    layer = SimpleNamespace(q_size=1024, index_q_size=128)
    monkeypatch.setattr(
        modeling_minimaxm3,
        "_extract_minimax_m3_attention_extra_attrs",
        lambda layer_idx: (None, layer),
    )
    with FakeTensorMode(shape_env=ShapeEnv()) as mode:
        num_tokens = mode.shape_env.create_unbacked_symint()
        hidden = torch.empty((num_tokens, 512), dtype=torch.bfloat16)
        positions = torch.empty((1, num_tokens), dtype=torch.int32)
        kv_cache = torch.empty((2, 2, 1, 128, 128), dtype=torch.float8_e4m3fn)
        index_cache = torch.empty((2, 1, 128, 128), dtype=torch.float8_e4m3fn)
        slots = torch.empty((4096,), dtype=torch.int32)
        q, idx_q = torch.ops.trtllm.minimax_m3_fused_sparse_qkv_producer(
            hidden, positions, kv_cache, index_cache, slots, "3"
        )
    for output, width in ((q, 1024), (idx_q, 128)):
        assert output.shape[0].node.expr == num_tokens.node.expr
        assert output.shape[1] == width
        assert output.dtype == torch.float8_e4m3fn


@pytest.mark.cpu_only
def test_piecewise_captures_horizontal_producer_before_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the captured producer before eager sparse attention consumes caches."""
    from unittest.mock import Mock

    backend = object.__new__(MiniMaxM3MsaSparseAttention)
    backend.indexer_kv_dtype = "fp8"
    packed = torch.empty((4, 7), dtype=torch.bfloat16)
    q = torch.empty((4, 3), dtype=torch.float8_e4m3fn)
    idx_q = torch.empty((4, 1), dtype=torch.float8_e4m3fn)
    output = torch.empty((4, 3), dtype=torch.bfloat16)
    kv_cache = torch.empty(8, dtype=torch.float8_e4m3fn)
    index_cache = torch.empty_like(kv_cache)
    slots = torch.tensor([0, 1, -1, -1], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_tokens=2, msa_layer_cache_tensors={3: (kv_cache, index_cache)}, msa_out_cache_loc=slots
    )
    layer = SimpleNamespace(
        enable_fused_qkv_index_projection=True,
        register_to_config=True,
        attn=backend,
        _emit_fp8_main_qkv=lambda: True,
        layer_idx=3,
        layer_idx_str="3",
        qkv_proj=Mock(return_value=packed),
        _fused_fp8_qkv_indexer_norm_rope_kv_insert=Mock(return_value=(q, idx_q)),
        _forward_attention_core=Mock(return_value=output),
        o_proj=lambda output, all_reduce_params: output,
    )
    monkeypatch.setattr(modeling_minimaxm3, "is_torch_compiling", lambda: True)
    monkeypatch.setattr(
        modeling_minimaxm3,
        "_extract_minimax_m3_attention_extra_attrs",
        lambda layer_idx: (metadata, layer),
    )
    hidden = torch.empty((4, 5), dtype=torch.bfloat16)
    positions = torch.arange(4).reshape(1, 4)
    result = MiniMaxM3Attention._sparse_forward(layer, positions, hidden, metadata)
    assert result is output
    layer.qkv_proj.assert_called_once_with(hidden)
    layer._fused_fp8_qkv_indexer_norm_rope_kv_insert.assert_called_once_with(
        packed, positions, metadata, cache_tensors=(kv_cache, index_cache, slots)
    )
    layer._forward_attention_core.assert_called_once_with(q, None, None, idx_q, None, metadata)


@pytest.mark.cpu_only
def test_piecewise_captured_producer_rejects_unavailable_fusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject unsupported fused geometry without an eager fallback in capture."""
    from unittest.mock import Mock

    layer = SimpleNamespace(
        qkv_proj=Mock(side_effect=lambda hidden: hidden.clone()),
        _fused_fp8_qkv_indexer_norm_rope_kv_insert=Mock(return_value=None),
    )
    monkeypatch.setattr(
        modeling_minimaxm3, "_extract_minimax_m3_attention_extra_attrs", lambda _: (None, layer)
    )
    with pytest.raises(RuntimeError) as exc:
        torch.ops.trtllm.minimax_m3_fused_sparse_qkv_producer(
            torch.zeros(2, 4),
            torch.arange(2),
            torch.zeros(4),
            torch.zeros(4),
            torch.arange(2, dtype=torch.int32),
            "3",
        )
    assert str(exc.value) == (
        "MiniMax-M3 piecewise graph requires the fused FP8 sparse QKV producer."
    )
    layer._fused_fp8_qkv_indexer_norm_rope_kv_insert.assert_called_once()


@pytest.mark.cpu_only
@pytest.mark.parametrize("restore_inplace", [False, True])
def test_piecewise_captured_producer_declares_cache_mutations(
    monkeypatch: pytest.MonkeyPatch,
    restore_inplace: bool,
) -> None:
    """Real custom-op schema/AOT checks with CPU cache writes in place of CUDA math."""
    from torch._dynamo.backends.common import aot_autograd
    from torch._functorch.aot_autograd import make_boxed_func
    from torch._higher_order_ops.auto_functionalize import (
        auto_functionalized,
        auto_functionalized_v2,
    )

    from tensorrt_llm._torch.compilation.remove_copy_pass import remove_copy_for_mutates_args

    def producer(
        packed: torch.Tensor,
        positions: torch.Tensor,
        metadata: object,
        *,
        cache_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Emulate native cache writes using only the explicit tensor arguments."""
        kv_cache, index_cache, slots = cache_tensors
        # Deliberately use only the explicit caches, not the runtime metadata.
        valid = slots >= 0
        indices = slots[valid].long()
        kv_cache.index_copy_(0, indices, packed[valid, :3].float())
        index_cache.index_copy_(0, indices, packed[valid, :1].float() + 1)
        return packed[:, :3].to(torch.float8_e4m3fn), packed[:, :1].to(torch.float8_e4m3fn)

    layer = SimpleNamespace(
        q_size=3,
        index_q_size=1,
        qkv_proj=lambda hidden: hidden + 2,
        _fused_fp8_qkv_indexer_norm_rope_kv_insert=producer,
    )
    monkeypatch.setattr(
        modeling_minimaxm3, "_extract_minimax_m3_attention_extra_attrs", lambda _: (None, layer)
    )
    op = torch.ops.trtllm.minimax_m3_fused_sparse_qkv_producer.default
    assert {
        arg.name for arg in op._schema.arguments if arg.alias_info and arg.alias_info.is_write
    } == {"kv_cache", "index_k_cache"}
    hidden = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    positions = torch.arange(4)
    kv_cache = torch.zeros(8, 3)
    index_cache = torch.zeros(8, 1)
    slots = torch.tensor([1, 3, -1, -1], dtype=torch.int32)
    args = (hidden, positions, kv_cache, index_cache, slots, "3")
    assert all(result == "SUCCESS" for result in torch.library.opcheck(op, args).values())

    def run(
        hidden: torch.Tensor,
        positions: torch.Tensor,
        main: torch.Tensor,
        index: torch.Tensor,
        slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expose query results and immediate observations of both live caches."""
        q, idx_q = op(hidden, positions, main, index, slots, "3")
        return q.float(), idx_q.float(), main.clone(), index.clone()

    optimized_graphs = []

    def optimize(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]) -> object:
        """Apply the production in-place recovery pass and inspect its aliases."""
        remove_copy_for_mutates_args(gm.graph)
        gm.graph.lint()
        gm.recompile()
        producer_nodes = [node for node in gm.graph.nodes if node.target == op]
        assert len(producer_nodes) == 1
        # No full-pool functionalization clones between the graph inputs and
        # the producer: MSA's eager boundary must observe the original pools.
        assert producer_nodes[0].kwargs["kv_cache"].op == "placeholder"
        assert producer_nodes[0].kwargs["index_k_cache"].op == "placeholder"
        assert all(
            node.target not in (auto_functionalized, auto_functionalized_v2)
            for node in gm.graph.nodes
        )
        optimized_graphs.append(gm)
        return make_boxed_func(gm.forward)

    # Match the production backend's functionalization version.
    monkeypatch.setattr(torch._inductor.config, "enable_auto_functionalized_v2", False)
    compiled = torch.compile(
        run,
        backend=aot_autograd(fw_compiler=optimize) if restore_inplace else "aot_eager",
        fullgraph=True,
    )
    for offset in (0, 4):
        kv_cache.zero_()
        index_cache.zero_()
        q, idx_q, observed_main, observed_index = compiled(
            hidden + offset, positions, kv_cache, index_cache, slots
        )
        packed = hidden + offset + 2
        expected_main = torch.zeros_like(kv_cache)
        expected_index = torch.zeros_like(index_cache)
        expected_main[[1, 3]] = packed[:2, :3]
        expected_index[[1, 3]] = packed[:2, :1] + 1
        torch.testing.assert_close(kv_cache, expected_main)
        torch.testing.assert_close(index_cache, expected_index)
        torch.testing.assert_close(observed_main, expected_main)
        torch.testing.assert_close(observed_index, expected_index)
        torch.testing.assert_close(q, packed[:, :3].to(torch.float8_e4m3fn).float())
        torch.testing.assert_close(idx_q, packed[:, :1].to(torch.float8_e4m3fn).float())
        torch.testing.assert_close(slots, torch.tensor([1, 3, -1, -1], dtype=torch.int32))
    if restore_inplace:
        assert len(optimized_graphs) == 1


@pytest.mark.cpu_only
def test_piecewise_unfused_indexer_keeps_cache_write_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep index-cache mutation outside capture when projections are separate."""
    from unittest.mock import Mock

    backend = object.__new__(MiniMaxM3MsaSparseAttention)
    backend.indexer_kv_dtype = "fp8"
    q, k, v, idx_q, idx_k = [torch.empty((4, 128), dtype=torch.float8_e4m3fn) for _ in range(5)]
    metadata = SimpleNamespace(num_tokens=2)
    layer = SimpleNamespace(
        enable_fused_qkv_index_projection=False,
        register_to_config=True,
        attn=backend,
        _emit_fp8_main_qkv=lambda: True,
        qkv_proj=Mock(return_value=torch.empty((4, 384), dtype=torch.bfloat16)),
        index_qk_proj=Mock(return_value=torch.empty((4, 256), dtype=torch.bfloat16)),
        _fused_qk_norm_rope=Mock(
            side_effect=[torch.cat((q, k, v), dim=-1), torch.cat((idx_q, idx_k), dim=-1)]
        ),
        _fused_fp8_index_qk_norm_rope=Mock(),
        _split_main_qkv=lambda tensor: (q, k, v),
        _split_index_qk=lambda tensor: (idx_q, idx_k),
        num_heads=1,
        num_key_value_heads=1,
        head_dim=128,
        sparse_num_index_heads=1,
        sparse_index_dim=128,
        q_norm=object(),
        k_norm=object(),
        index_q_norm=object(),
        index_k_norm=object(),
        ln_events=(None, None),
        aux_stream=None,
        _forward_attention_core=Mock(return_value=torch.empty((4, 128), dtype=torch.bfloat16)),
        o_proj=lambda output, all_reduce_params: output,
    )
    monkeypatch.setattr(modeling_minimaxm3, "is_torch_compiling", lambda: True)
    monkeypatch.setattr(
        modeling_minimaxm3,
        "maybe_execute_in_parallel",
        lambda first, second, *args, **kwargs: (first(), second()),
    )
    MiniMaxM3Attention._sparse_forward(
        layer, torch.arange(4).reshape(1, 4), torch.empty((4, 128), dtype=torch.bfloat16), metadata
    )
    layer._fused_fp8_index_qk_norm_rope.assert_not_called()
    assert layer._fused_qk_norm_rope.call_count == 2
    assert all(call.kwargs["out_fp8"] for call in layer._fused_qk_norm_rope.call_args_list)
    layer._forward_attention_core.assert_called_once_with(q, k, v, idx_q, idx_k, metadata)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("indexer_dtype", "caches_prewritten"),
    [("fp8", False), ("fp8", True), ("bf16", False)],
    ids=["unfused-fp8", "fused-fp8", "bf16"],
)
def test_piecewise_attention_boundary_preserves_indexer_cache_contract(
    monkeypatch: pytest.MonkeyPatch, indexer_dtype: str, caches_prewritten: bool
) -> None:
    """Exercise the real MSA indexer after live-row slicing and cache insertion."""
    from unittest.mock import Mock

    dtype = torch.float8_e4m3fn if indexer_dtype == "fp8" else torch.bfloat16
    q, k, v, idx_q, idx_k = [torch.full((4, 128), value).to(dtype) for value in range(1, 6)]
    index_cache = torch.zeros((4, 1, 1, 128), dtype=dtype)
    expected_cache = torch.zeros_like(index_cache)
    expected_cache[:2, 0, 0].copy_(idx_k[:2])
    if caches_prewritten:
        index_cache.copy_(expected_cache)
    selected_blocks = torch.zeros(2, 1, 16, dtype=torch.int32)
    attn_metadata = SimpleNamespace(
        num_tokens=2,
        msa_decode_span=None,
        msa_idx_k_cache=Mock(return_value=index_cache),
        msa_write_idx_k=Mock(),
        msa_prefill_proxy_plan=None,
        msa_prefill_n_valid_blocks=None,
        msa_kv_indices=torch.tensor([0, 1], dtype=torch.int32),
        msa_qo_lens_cpu=torch.tensor([2], dtype=torch.int32),
        msa_kv_lens_cpu=torch.tensor([2], dtype=torch.int32),
        msa_qo_offset_cpu=torch.tensor([0], dtype=torch.int32),
    )

    def write_caches(
        live_k: torch.Tensor,
        live_v: torch.Tensor,
        live_idx_k: torch.Tensor,
        metadata: SimpleNamespace,
    ) -> None:
        """Replace only CUDA scatter math, retaining the live cache-write inputs."""
        assert metadata is attn_metadata
        torch.testing.assert_close(live_k.float(), k[:2].float())
        torch.testing.assert_close(live_v.float(), v[:2].float())
        torch.testing.assert_close(live_idx_k.float(), idx_k[:2].float())
        index_cache[:2, 0, 0].copy_(live_idx_k)

    def select_blocks(
        live_idx_q: torch.Tensor, cache: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        """Require the cache write to precede selection, without running CUDA scoring."""
        assert cache is index_cache
        torch.testing.assert_close(cache.float(), expected_cache.float())
        torch.testing.assert_close(live_idx_q.flatten(1).float(), idx_q[:2].float())
        return selected_blocks

    backend = object.__new__(MiniMaxM3MsaSparseAttention)
    backend.layer_idx = 3
    backend.indexer_kv_dtype = indexer_dtype
    backend.m3_config = SimpleNamespace(num_index_heads=1, sparse_index_dim=128)
    backend.write_layer_caches = Mock(side_effect=write_caches)
    backend.indexer = SimpleNamespace(select_blocks=Mock(side_effect=select_blocks))
    backend.forward = Mock()
    layer = MiniMaxM3Attention.__new__(MiniMaxM3Attention)
    layer.attn = backend
    layer.is_sparse_attention_layer = True
    output = torch.empty((4, 128), dtype=torch.bfloat16)
    monkeypatch.setattr(
        modeling_minimaxm3,
        "_extract_minimax_m3_attention_extra_attrs",
        lambda layer_idx: (attn_metadata, layer),
    )
    modeling_minimaxm3.minimax_m3_attn_custom_op_inplace(
        q,
        None if caches_prewritten else k,
        None if caches_prewritten else v,
        idx_q,
        None if caches_prewritten else idx_k,
        None,
        None,
        "3",
        output,
    )

    assert backend.write_layer_caches.call_count == (0 if caches_prewritten else 1)
    attn_metadata.msa_write_idx_k.assert_not_called()
    attn_metadata.msa_idx_k_cache.assert_called_once_with(3)
    backend.indexer.select_blocks.assert_called_once()
    backend.forward.assert_called_once()
    called_q, called_k, called_v, called_metadata = backend.forward.call_args.args
    assert called_k is None and called_v is None
    torch.testing.assert_close(called_q.float(), q[:2].float())
    assert called_metadata is attn_metadata
    forward_args = backend.forward.call_args.kwargs["forward_args"]
    assert forward_args.output.shape == (2, 128)
    assert forward_args.output.data_ptr() == output.data_ptr()
    assert forward_args.sparse_backend_args.topk_indices is selected_blocks


def test_model_init_validates_sparse_attention_runtime_config() -> None:
    model_config = ModelConfig(
        pretrained_config=_make_text_config(),
        sparse_attention_config=None,
    )

    with pytest.raises(ValueError, match="algorithm='minimax_m3'"):
        MiniMaxM3Model(model_config)


def _make_text_config():
    """Build a SimpleNamespace mimicking the real M3 text config (trimmed)."""
    sparse_attention_config = {
        "use_sparse_attention": True,
        "sparse_index_dim": 128,
        "sparse_num_index_heads": 4,
        "sparse_topk_blocks": 16,
        "sparse_block_size": 128,
        "sparse_init_block": 0,
        "sparse_local_block": 1,
        "sparse_score_type": "max",
        "sparse_disable_index_value": list(_DISABLE_INDEX_VALUE),
        "sparse_attention_freq": list(_SPARSE_FREQ),
    }
    return SimpleNamespace(
        model_type="minimax_m3",
        hidden_size=6144,
        intermediate_size=3072,
        num_hidden_layers=_NUM_HIDDEN_LAYERS,
        num_attention_heads=64,
        num_key_value_heads=4,
        head_dim=128,
        vocab_size=200064,
        max_position_embeddings=524288,
        rms_norm_eps=1e-06,
        use_gemma_norm=True,
        attention_output_gate=False,
        rope_theta=5000000,
        rotary_dim=64,
        partial_rotary_factor=0.5,
        hidden_act="swigluoai",
        use_qk_norm=True,
        qk_norm_type="per_head",
        tie_word_embeddings=False,
        dense_intermediate_size=12288,
        shared_intermediate_size=3072,
        num_local_experts=128,
        num_experts_per_tok=4,
        n_shared_experts=1,
        scoring_func="sigmoid",
        use_routing_bias=True,
        moe_layer_freq=list(_MOE_LAYER_FREQ),
        num_mtp_modules=1,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        routed_scaling_factor=2.0,
        sparse_attention_config=sparse_attention_config,
        architectures=["MiniMaxM3SparseForCausalLM"],
        torch_dtype="bfloat16",
    )


def _make_vl_config():
    return SimpleNamespace(
        model_type="minimax_m3_vl",
        text_config=_make_text_config(),
        vision_config=SimpleNamespace(
            hidden_size=1280,
            num_attention_heads=16,
            num_hidden_layers=32,
        ),
        torch_dtype="bfloat16",
        tie_word_embeddings=False,
        architectures=["MiniMaxM3SparseForConditionalGeneration"],
        image_token_index=200025,
        video_token_index=200026,
    )


# ---------------------------------------------------------------------------
# Shared helpers used by both CPU and CUDA tests
# ---------------------------------------------------------------------------


_DEFAULT_CHECKPOINT_PATH = f"{llm_models_root()}/MiniMax-M3"


def _checkpoint_path() -> str:
    return _DEFAULT_CHECKPOINT_PATH


def _has_cuda() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CPU-only unit tests
# ---------------------------------------------------------------------------


def test_is_minimax_m3_vl_config_detects_vl():
    assert is_minimax_m3_vl_config(_make_vl_config()) is True


def test_is_minimax_m3_vl_config_detects_text_only():
    assert is_minimax_m3_vl_config(_make_text_config()) is False


def test_is_minimax_m3_vl_config_falls_back_to_architectures():
    cfg = SimpleNamespace(
        model_type="custom",
        architectures=["MiniMaxM3SparseForConditionalGeneration"],
    )
    assert is_minimax_m3_vl_config(cfg) is True


def test_get_text_config_returns_text_subconfig():
    vl_cfg = _make_vl_config()
    text_cfg = get_text_config(vl_cfg)
    assert text_cfg is vl_cfg.text_config
    assert text_cfg.num_hidden_layers == _NUM_HIDDEN_LAYERS


def test_get_text_config_passthrough_for_text_only():
    text_cfg = _make_text_config()
    assert get_text_config(text_cfg) is text_cfg


def test_get_text_config_propagates_dtype_when_missing():
    vl_cfg = _make_vl_config()
    vl_cfg.text_config.torch_dtype = None
    out = get_text_config(vl_cfg)
    assert out.torch_dtype == "bfloat16"


def test_get_text_config_missing_text_attribute_raises():
    bad = SimpleNamespace(model_type="minimax_m3_vl")
    with pytest.raises(ValueError, match="text_config"):
        get_text_config(bad)


def test_get_sparse_layer_ids_splits_dense_and_sparse():
    dense, sparse = get_sparse_layer_ids(_make_text_config())
    assert dense == [0, 1, 2]
    assert sparse == [3, 4, 5, 6]


def test_get_sparse_layer_ids_falls_back_when_disabled():
    cfg = _make_text_config()
    cfg.sparse_attention_config["use_sparse_attention"] = False
    dense, sparse = get_sparse_layer_ids(cfg)
    assert dense == list(range(_NUM_HIDDEN_LAYERS))
    assert sparse == []


def test_get_sparse_layer_ids_falls_back_without_config():
    cfg = _make_text_config()
    cfg.sparse_attention_config = None
    dense, sparse = get_sparse_layer_ids(cfg)
    assert dense == list(range(_NUM_HIDDEN_LAYERS))
    assert sparse == []


def test_get_sparse_layer_ids_length_mismatch_raises():
    cfg = _make_text_config()
    cfg.sparse_attention_config["sparse_attention_freq"] = [0] * (_NUM_HIDDEN_LAYERS + 1)
    with pytest.raises(ValueError, match="sparse_attention_freq length"):
        get_sparse_layer_ids(cfg)


def test_get_sparse_disable_index_value_layer_ids_matches_sparse():
    ids = get_sparse_disable_index_value_layer_ids(_make_text_config())
    assert ids == [3, 4, 5, 6]


def test_get_sparse_disable_index_value_no_config():
    cfg = _make_text_config()
    cfg.sparse_attention_config = None
    assert get_sparse_disable_index_value_layer_ids(cfg) == []


def test_get_moe_layer_ids_splits_dense_and_moe():
    dense, moe = get_moe_layer_ids(_make_text_config())
    assert dense == [0, 1, 2]
    assert moe == [3, 4, 5, 6]


def test_get_moe_layer_ids_all_moe_without_freq():
    cfg = _make_text_config()
    cfg.moe_layer_freq = None
    dense, moe = get_moe_layer_ids(cfg)
    assert dense == []
    assert moe == list(range(_NUM_HIDDEN_LAYERS))


def test_get_moe_layer_ids_length_mismatch_raises():
    cfg = _make_text_config()
    cfg.moe_layer_freq = [0] * (_NUM_HIDDEN_LAYERS - 1)
    with pytest.raises(ValueError, match="moe_layer_freq length"):
        get_moe_layer_ids(cfg)


# ---------------------------------------------------------------------------
# attention module transforms
# ---------------------------------------------------------------------------
#
# These tests construct :class:`MiniMaxM3Attention` with a tiny synthetic
# geometry and ``skip_create_weights_in_init=True`` so the Linear modules
# exist (with ``.in_features`` / ``.out_features`` set) but no weights are
# allocated. The base :class:`Attention` constructor reaches into CUDA-only
# paths (e.g. backend selection), so the tests run under
# ``pytest.mark.gpu`` + ``skipif(not _has_cuda())``. Geometry is bounded to
# a few KB.
#
# Coverage:
#  * Dense / sparse attention construction shapes match the configured
#    head_dim, head counts, and sparse index branch dimensions.
#  * Partial RoPE only rotates ``rotary_dim`` of ``head_dim`` channels.
#  * Per-head Gemma Q/K RMSNorm: q_norm / k_norm are RMSNorm with
#    ``use_gemma=True`` and ``hidden_size=head_dim``; the
#    :meth:`apply_qk_norm` reshape matches an independent hand-written
#    reference.
#  * Sparse index branch: KV-group-sharded index_qk_proj with
#    output [idx_q | idx_k] = num_index_heads * sparse_index_dim + sparse_index_dim
#    (idx_k is one K per token).
#  * Dense layers do not expose any index branch attributes (negative
#    control).


def _make_attention_test_config():
    """Return ``(text_config, ModelConfig)`` for the attention tests.

    Geometry is a scaled-down M3-shaped config: hidden_size=128, head_dim=32,
    num_heads=4, num_kv_heads=2, num_index_heads=2, sparse_index_dim=32,
    rotary_dim=16 (= head_dim * 0.5 — partial RoPE), 1 dense + 3 sparse
    layers. With ``skip_create_weights_in_init=True`` no Linear weight
    tensors are allocated, only metadata.
    """
    n_layers = 4
    sparse_cfg = {
        "use_sparse_attention": True,
        "sparse_index_dim": 32,
        "sparse_num_index_heads": 2,
        "sparse_topk_blocks": 4,
        "sparse_block_size": 16,
        "sparse_init_block": 0,
        "sparse_local_block": 1,
        "sparse_score_type": "max",
        "sparse_disable_index_value": [0, 1, 1, 1],
        "sparse_attention_freq": [0, 1, 1, 1],
    }
    text_cfg = _wrap_dict_as_config(
        {
            "hidden_size": 128,
            "intermediate_size": 64,
            "num_hidden_layers": n_layers,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "vocab_size": 256,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-6,
            "use_gemma_norm": True,
            "rope_theta": 10000.0,
            "rotary_dim": 16,
            "partial_rotary_factor": 0.5,
            "qk_norm_type": "per_head",
            "use_qk_norm": True,
            "sparse_attention_config": sparse_cfg,
            "torch_dtype": torch.bfloat16,
        }
    )
    model_cfg = ModelConfig(
        pretrained_config=text_cfg,
        mapping=Mapping(),
        skip_create_weights_in_init=True,
    )
    return text_cfg, model_cfg


def _per_head_gemma_rms_norm_reference(x, weight, eps):
    """Hand-written reference for per-head Gemma RMSNorm.

    Matches :class:`RMSNorm.forward` with ``use_gemma=True``,
    ``residual=None``, ``is_nvfp4=False``: cast to float32 to compute
    variance, normalise, cast back to input dtype, then scale by
    ``(weight + 1)``. The per-head structure comes from reshaping the
    input to ``(-1, head_dim)`` before applying this function.
    """
    input_dtype = x.dtype
    x_f32 = x.to(torch.float32)
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    x_norm = (x_f32 * torch.rsqrt(variance + eps)).to(input_dtype)
    return (weight + 1.0) * x_norm


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 attention construction needs CUDA")
def test_minimax_m3_attention_dense_construction_matches_config():
    """Dense layer's QKV/O projection and per-head Q/K norm match config."""
    text_cfg, model_cfg = _make_attention_test_config()
    attn = MiniMaxM3Attention(
        model_config=model_cfg,
        layer_idx=0,
        is_sparse_attention_layer=False,
        disable_index_value=False,
    )

    head_dim = int(text_cfg.head_dim)
    num_heads = int(text_cfg.num_attention_heads)
    num_kv = int(text_cfg.num_key_value_heads)
    hidden = int(text_cfg.hidden_size)

    # Q/K/V projection (fused into qkv_proj).
    assert attn.num_heads == num_heads
    assert attn.num_key_value_heads == num_kv
    assert attn.head_dim == head_dim
    assert attn.head_dim_value == head_dim
    assert attn.q_size == num_heads * head_dim
    assert attn.kv_size == num_kv * head_dim
    assert attn.qkv_proj.in_features == hidden
    # With tp_size=1, out_features == q_size + 2 * kv_size.
    assert attn.qkv_proj.out_features == num_heads * head_dim + 2 * num_kv * head_dim

    # Output projection.
    assert attn.o_proj.in_features == num_heads * head_dim
    assert attn.o_proj.out_features == hidden

    # Per-head Gemma Q/K RMSNorm.
    assert attn.use_gemma_norm is True
    assert attn.qk_norm_type == "per_head"
    assert attn.q_norm.use_gemma is True
    assert attn.k_norm.use_gemma is True
    assert tuple(attn.q_norm.weight.shape) == (head_dim,)
    assert tuple(attn.k_norm.weight.shape) == (head_dim,)

    # Dense layers must not expose any index-branch attributes.
    assert attn.is_sparse_attention_layer is False
    for name in (
        "index_q_proj",
        "index_k_proj",
        "index_qk_proj",
        "index_q_norm",
        "index_k_norm",
    ):
        assert not hasattr(attn, name), f"dense layer should not declare {name!r}"


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 attention construction needs CUDA")
def test_minimax_m3_attention_partial_rope_dim_is_rotary_dim():
    """Partial RoPE rotates only ``rotary_dim`` of ``head_dim`` channels."""
    text_cfg, model_cfg = _make_attention_test_config()
    attn = MiniMaxM3Attention(
        model_config=model_cfg,
        layer_idx=0,
        is_sparse_attention_layer=False,
    )
    rotary_dim = int(text_cfg.rotary_dim)
    head_dim = int(text_cfg.head_dim)
    assert attn.pos_embd_params is not None
    assert attn.pos_embd_params.rope.dim == rotary_dim
    assert attn.pos_embd_params.rope.dim < head_dim, (
        f"partial RoPE expects rope.dim < head_dim, got {attn.pos_embd_params.rope.dim} >= {head_dim}"
    )
    # The base Attention class also stores the rotary embedding when
    # ``rope_fusion=False``; M3 sets ``rope_fusion=False`` explicitly.
    assert attn.rope_fusion is False
    assert attn.rotary_emb is not None


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 attention construction needs CUDA")
def test_minimax_m3_attention_apply_qk_norm_matches_reference():
    """Verify ``apply_qk_norm`` does per-head Gemma RMSNorm and reshape-back."""
    text_cfg, model_cfg = _make_attention_test_config()
    attn = MiniMaxM3Attention(
        model_config=model_cfg,
        layer_idx=0,
        is_sparse_attention_layer=False,
    )
    head_dim = int(text_cfg.head_dim)
    eps = float(text_cfg.rms_norm_eps)

    # Set non-zero norm weights so the test catches any reshape /
    # weight-broadcast bugs (zero weights would mask many errors).
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    q_weight = torch.randn(head_dim, dtype=dtype, device=device) * 0.2
    k_weight = torch.randn(head_dim, dtype=dtype, device=device) * 0.2
    attn.q_norm.weight = torch.nn.Parameter(q_weight)
    attn.k_norm.weight = torch.nn.Parameter(k_weight)

    seq = 3
    q = torch.randn(seq, attn.q_size, dtype=dtype, device=device)
    k = torch.randn(seq, attn.kv_size, dtype=dtype, device=device)

    q_out, k_out = attn.apply_qk_norm(q, k)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape

    # Hand-written reference: per-head reshape -> Gemma RMSNorm ->
    # reshape back. Identical computation, independent code path.
    q_ref = _per_head_gemma_rms_norm_reference(q.reshape(-1, head_dim), q_weight, eps).reshape(
        q.shape
    )
    k_ref = _per_head_gemma_rms_norm_reference(k.reshape(-1, head_dim), k_weight, eps).reshape(
        k.shape
    )

    # BF16 + possible flashinfer kernel: use a looser tolerance.
    torch.testing.assert_close(q_out, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_out, k_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 attention construction needs CUDA")
def test_minimax_m3_attention_sparse_construction_matches_config():
    """Sparse layer adds the index branch with the fused index projection.

    * At TP1, index_qk_proj retains all index heads, out =
      num_index_heads * sparse_index_dim (idx_q) + sparse_index_dim (idx_k),
      where idx_k is one replicated K per token.
    * index_q_norm / index_k_norm are per-head Gemma RMSNorm of width
      sparse_index_dim.
    """
    text_cfg, model_cfg = _make_attention_test_config()
    sparse_cfg = text_cfg.sparse_attention_config
    num_index_heads = int(sparse_cfg["sparse_num_index_heads"])
    sparse_index_dim = int(sparse_cfg["sparse_index_dim"])
    hidden = int(text_cfg.hidden_size)

    attn = MiniMaxM3Attention(
        model_config=model_cfg,
        layer_idx=3,
        is_sparse_attention_layer=True,
        disable_index_value=True,
    )
    assert attn.is_sparse_attention_layer is True
    assert attn.disable_index_value is True

    # Explicit shard ranges retain whole index heads and replicate index-K.
    assert attn.index_q_size == num_index_heads * sparse_index_dim
    assert attn.index_k_size == sparse_index_dim
    assert attn.index_qk_proj.in_features == hidden
    assert attn.index_qk_proj.out_features == num_index_heads * sparse_index_dim + sparse_index_dim
    assert attn.index_qk_proj.tp_mode == modeling_minimaxm3.TensorParallelMode.COLUMN
    # Only the fused projection exists.
    assert not hasattr(attn, "index_q_proj")
    assert not hasattr(attn, "index_k_proj")

    # Per-head Gemma RMSNorm of width sparse_index_dim.
    assert attn.index_q_norm.use_gemma is True
    assert attn.index_k_norm.use_gemma is True
    assert tuple(attn.index_q_norm.weight.shape) == (sparse_index_dim,)
    assert tuple(attn.index_k_norm.weight.shape) == (sparse_index_dim,)

    # The sparse forward path now dispatches through the MiniMax-M3
    # sparse algorithm. Calling forward without metadata must raise a
    # clear RuntimeError pointing at the missing kv_cache_manager
    # (rather than silently returning garbage or crashing inside the
    # algorithm).
    try:
        attn.forward()
    except RuntimeError as e:
        msg = str(e)
        assert "attn_metadata" in msg or "kv_cache_manager" in msg, msg
    else:  # pragma: no cover
        raise AssertionError("sparse forward must raise RuntimeError when attn_metadata is None")


@pytest.mark.cpu_only
@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_minimax_m3_five_way_projection_shard_geometry(tp_size: int) -> None:
    shard_geometry = MiniMaxM3QKVIndexerLinear._shard_geometry
    for tp_rank in range(tp_size):
        module = SimpleNamespace(
            tp_size=tp_size, tp_rank=tp_rank, total_num_kv_heads=4, total_num_index_heads=4
        )
        kv_world = min(tp_size, 4)
        kv_rank = tp_rank // max(tp_size // 4, 1)
        assert shard_geometry(module, "q") == (tp_size, tp_rank)
        assert shard_geometry(module, "k") == (kv_world, kv_rank)
        assert shard_geometry(module, "v") == (kv_world, kv_rank)
        assert shard_geometry(module, "index_q") == (kv_world, kv_rank)
        assert shard_geometry(module, "index_k") == (1, 0)


@pytest.mark.cpu_only
@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_minimax_m3_five_way_projection_shards_index_rows(monkeypatch, tp_size: int) -> None:
    def init_linear(self, in_features, out_features, **kwargs) -> None:
        nn.Module.__init__(self)
        self.tp_size = kwargs["mapping"].tp_size
        self.tp_rank = kwargs["mapping"].tp_rank
        self.tp_mode = kwargs["tensor_parallel_mode"]
        self.weights_loading_config = kwargs["weights_loading_config"]
        self.out_features = out_features // self.tp_size

    monkeypatch.setattr(modeling_minimaxm3.Linear, "__init__", init_linear)
    # Exercise the real checkpoint loader on CPU; only its destination device
    # is substituted, leaving the TP slicing and five-way packing intact.
    load_weight_shard = modeling_minimaxm3.load_weight_shard

    def load_cpu_shard(weight, world_size, rank, mode, *, device):
        return load_weight_shard(weight, world_size, rank, mode, device=torch.device("cpu"))

    monkeypatch.setattr(modeling_minimaxm3, "load_weight_shard", load_cpu_shard)
    for tp_rank in range(tp_size):
        projection = MiniMaxM3QKVIndexerLinear(
            hidden_size=2,
            head_dim=128,
            total_num_heads=64,
            total_num_kv_heads=4,
            total_num_index_heads=4,
            index_head_dim=128,
            dtype=torch.bfloat16,
            mapping=SimpleNamespace(tp_size=tp_size, tp_rank=tp_rank),
            quant_config=None,
            skip_create_weights_in_init=True,
            force_dynamic_quantization=False,
            disable_deep_gemm=False,
            use_custom_cublas_mm=False,
            use_cute_dsl_bf16_gemm=False,
            use_cute_dsl_blockscaling_mm=False,
        )
        kv_heads = max(4 // tp_size, 1)
        assert projection.local_output_sizes == (
            64 // tp_size * 128,
            kv_heads * 128,
            kv_heads * 128,
            kv_heads * 128,
            128,
        )
        assert projection.local_num_index_heads == kv_heads
        assert projection.out_features == sum(projection.local_output_sizes)
        shards = {
            name: {"weight": torch.arange(heads * 128 * 2).reshape(heads * 128, 2)}
            for name, heads in zip(projection._SHARD_NAMES, (64, 4, 4, 4, 1), strict=True)
        }
        loaded = []
        projection.load_weights = loaded.extend
        projection.load_five_way_weights(shards)
        packed = loaded[0]["weight"].split(projection.local_output_sizes)
        torch.testing.assert_close(packed[4], shards["index_k"]["weight"])
        kv_rank = tp_rank // max(tp_size // 4, 1)
        torch.testing.assert_close(
            packed[3], shards["index_q"]["weight"].chunk(min(tp_size, 4))[kv_rank]
        )
        torch.testing.assert_close(packed[1], shards["k"]["weight"].chunk(min(tp_size, 4))[kv_rank])
        assert projection.tp_size == tp_size and projection.tp_rank == tp_rank


@pytest.mark.cpu_only
@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("attention_dp", [False, True])
@pytest.mark.parametrize("num_index_heads", [4, 8])
def test_minimax_m3_index_tp_matches_unsharded_reference(
    tp_size: int, attention_dp: bool, num_index_heads: int
) -> None:
    """Checkpoint row ownership and selected blocks must be TP invariant."""
    generator = torch.Generator().manual_seed(2718)
    head_dim, hidden_dim, global_kv_heads = 2, 8, 4
    q_weight = torch.randn(num_index_heads * head_dim, hidden_dim, generator=generator)
    k_weight = torch.randn(head_dim, hidden_dim, generator=generator)
    hidden = torch.randn(5, hidden_dim, generator=generator)
    key_hidden = torch.randn(11, 3, hidden_dim, generator=generator)
    global_q = torch.nn.functional.linear(hidden, q_weight).reshape(5, num_index_heads, head_dim)
    keys = torch.nn.functional.linear(key_hidden, k_weight)
    global_scores = torch.einsum("thd,bkd->hbkt", global_q, keys).amax(dim=2)
    params = MiniMaxM3SparseParams(num_index_heads=num_index_heads, global_num_kv_heads=4)
    reference_cfg = MiniMaxM3SparseConfig.from_sparse_params(
        params, num_q_heads=64, num_kv_heads=4, head_dim=head_dim
    )
    reference_scores = _group_max_reduce(global_scores, reference_cfg)
    metadata = MiniMaxM3SparseMetadataParams(
        global_num_q_heads=64, global_num_kv_heads=4, num_index_heads=num_index_heads
    )
    effective_tp = 1 if attention_dp else tp_size
    local_kv_heads = max(global_kv_heads // effective_tp, 1)
    for rank in range(tp_size):
        mapping = SimpleNamespace(tp_size=tp_size, tp_rank=rank, enable_attention_dp=attention_dp)
        start, end = index_head_range(num_index_heads, global_kv_heads, mapping)
        kv_start = 0 if attention_dp else rank // max(tp_size // 4, 1) * local_kv_heads
        assert (start, end) == (
            kv_start * (num_index_heads // 4),
            (kv_start + local_kv_heads) * (num_index_heads // 4),
        )
        assert metadata.sharded_index_head_count(mapping) == end - start

        # Exercise Linear's real per-shard loader, as used by index_qk_proj:
        # index-Q uses the KV group's rows; index-K always uses all its rows.
        projection = object.__new__(modeling_minimaxm3.Linear)
        nn.Module.__init__(projection)
        projection.tp_size = effective_tp
        projection.tp_rank = 0 if attention_dp else rank
        projection.tp_mode = modeling_minimaxm3.TensorParallelMode.COLUMN
        projection.tp_sharding = {"gate": (start * head_dim, end * head_dim), "up": (0, head_dim)}
        projection.weights_loading_config = modeling_minimaxm3.WeightsLoadingConfig(
            weight_mode=modeling_minimaxm3.WeightMode.FUSED_GATE_UP_LINEAR
        )
        local_q_weight = projection.load_shard({"weight": q_weight}, "weight", name="gate")
        local_k_weight = projection.load_shard({"weight": k_weight}, "weight", name="up")
        assert (
            projection.calculate_local_out_features((num_index_heads + 1) * head_dim)
            == (end - start + 1) * head_dim
        )
        torch.testing.assert_close(local_k_weight, k_weight)
        local_q = torch.nn.functional.linear(hidden, local_q_weight).reshape(
            5, end - start, head_dim
        )
        torch.testing.assert_close(local_q, global_q[:, start:end])
        local_scores = torch.einsum("thd,bkd->hbkt", local_q, keys).amax(dim=2)
        local_cfg = MiniMaxM3SparseConfig.from_sparse_params(
            params, num_q_heads=64 // effective_tp, num_kv_heads=local_kv_heads, head_dim=head_dim
        )
        assert local_cfg.num_index_heads == end - start
        grouped_scores = _group_max_reduce(local_scores, local_cfg)
        expected = reference_scores[kv_start : kv_start + local_kv_heads]
        torch.testing.assert_close(grouped_scores, expected)
        torch.testing.assert_close(
            grouped_scores.topk(3, dim=1).indices, expected.topk(3, dim=1).indices
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "num_index_heads,num_kv_heads,tp_size", [(0, 4, 1), (6, 4, 2), (4, 4, 3), (4, 0, 2)]
)
def test_minimax_m3_index_head_range_rejects_invalid_geometry(
    num_index_heads: int, num_kv_heads: int, tp_size: int
) -> None:
    with pytest.raises(ValueError):
        index_head_range(
            num_index_heads,
            num_kv_heads,
            SimpleNamespace(tp_size=tp_size, tp_rank=0, enable_attention_dp=False),
        )


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 attention construction needs CUDA")
@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("attention_dp", [False, True])
def test_minimax_m3_unfused_index_projection_tp_construction(
    tp_size: int, attention_dp: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify the model wires Linear's shard overrides to effective attention TP."""
    import tensorrt_llm._torch.distributed as distributed

    # Simulate each rank on one GPU without creating distributed workspaces.
    # No forward runs here; projection construction and shard sizing stay real.
    monkeypatch.setattr(distributed, "AllReduce", lambda **_kwargs: nn.Identity())
    for rank in range(tp_size):
        text_cfg, model_cfg = _make_attention_test_config()
        text_cfg.hidden_size = 256
        text_cfg.num_attention_heads = 8
        text_cfg.num_key_value_heads = 4
        text_cfg.sparse_attention_config["sparse_num_index_heads"] = 4
        model_cfg.mapping = Mapping(
            world_size=tp_size, rank=rank, tp_size=tp_size, enable_attention_dp=attention_dp
        )
        attn = MiniMaxM3Attention(
            model_config=model_cfg,
            layer_idx=3,
            is_sparse_attention_layer=True,
            disable_index_value=True,
        )
        kv_heads = 4 if attention_dp else max(4 // tp_size, 1)
        kv_start = 0 if attention_dp else rank // max(tp_size // 4, 1) * kv_heads
        assert attn.sparse_num_index_heads == kv_heads
        assert attn.index_qk_proj.out_features == (kv_heads + 1) * 32
        assert attn.index_qk_proj.tp_sharding == {
            "gate": (kv_start * 32, (kv_start + kv_heads) * 32),
            "up": (0, 32),
        }
        assert attn.index_qk_proj.mapping == attn.qkv_proj.mapping


@pytest.mark.cpu_only
def test_minimax_m3_five_way_loader_returns_exact_generic_skip() -> None:
    projection = object.__new__(MiniMaxM3QKVIndexerLinear)
    nn.Module.__init__(projection)
    captured = {}
    projection.load_five_way_weights = lambda shards: captured.update(shards)

    model = nn.Module()
    model.sparse = nn.Module()
    model.sparse.qkv_proj = projection
    weights = {
        f"sparse.{name}_proj.weight": torch.empty(1)
        for name in ("q", "k", "v", "index_q", "index_k")
    }

    loaded_modules = _load_qkv_index_proj_weights(model, weights)

    assert loaded_modules == ["sparse.qkv_proj"]
    assert set(captured) == {"q", "k", "v", "index_q", "index_k"}
    assert all(set(shard) == {"weight"} for shard in captured.values())
    assert weights == {}

    mapper = MiniMaxM3HfWeightMapper()
    mapper.add_skip_modules(loaded_modules)
    mapper._model = SimpleNamespace(config=SimpleNamespace(tie_word_embeddings=False))
    assert mapper.should_skip_module("sparse.qkv_proj")
    assert not mapper.should_skip_module("dense.qkv_proj")


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 attention construction needs CUDA")
def test_minimax_m3_attention_apply_index_qk_norm_matches_reference():
    """Sparse index per-head Gemma QK norm matches the hand-written reference.

    ``apply_index_qk_norm`` reshapes ``idx_q`` (``num_index_heads`` heads)
    and ``idx_k`` (1 replicated head) to ``(-1, sparse_index_dim)`` rows,
    applies the per-head Gemma RMSNorm, and reshapes back. The test sets
    non-zero norm weights, drives synthetic input, and compares against
    the same pure-torch reference used for the main Q/K norm.
    """
    text_cfg, model_cfg = _make_attention_test_config()
    sparse_cfg = text_cfg.sparse_attention_config
    num_index_heads = int(sparse_cfg["sparse_num_index_heads"])
    sparse_index_dim = int(sparse_cfg["sparse_index_dim"])

    attn = MiniMaxM3Attention(
        model_config=model_cfg,
        layer_idx=3,
        is_sparse_attention_layer=True,
        disable_index_value=True,
    )

    torch.manual_seed(1)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    eps = float(text_cfg.rms_norm_eps)
    iq_weight = torch.randn(sparse_index_dim, dtype=dtype, device=device) * 0.3
    ik_weight = torch.randn(sparse_index_dim, dtype=dtype, device=device) * 0.3
    attn.index_q_norm.weight = torch.nn.Parameter(iq_weight)
    attn.index_k_norm.weight = torch.nn.Parameter(ik_weight)

    seq = 5
    idx_q = torch.randn(seq, num_index_heads * sparse_index_dim, dtype=dtype, device=device)
    idx_k = torch.randn(seq, sparse_index_dim, dtype=dtype, device=device)
    iq_out, ik_out = attn.apply_index_qk_norm(idx_q, idx_k)
    assert iq_out.shape == idx_q.shape
    assert ik_out.shape == idx_k.shape

    iq_ref = _per_head_gemma_rms_norm_reference(
        idx_q.reshape(-1, sparse_index_dim), iq_weight, eps
    ).reshape(idx_q.shape)
    ik_ref = _per_head_gemma_rms_norm_reference(
        idx_k.reshape(-1, sparse_index_dim), ik_weight, eps
    ).reshape(idx_k.shape)

    torch.testing.assert_close(iq_out, iq_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(ik_out, ik_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 attention construction needs CUDA")
def test_minimax_m3_attention_dense_apply_index_qk_norm_raises():
    """Dense layers must reject ``apply_index_qk_norm`` calls."""
    _, model_cfg = _make_attention_test_config()
    attn = MiniMaxM3Attention(
        model_config=model_cfg,
        layer_idx=0,
        is_sparse_attention_layer=False,
    )
    idx_q = torch.zeros(2, 64, dtype=torch.bfloat16, device="cuda")
    idx_k = torch.zeros(2, 32, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="only valid on sparse attention layers"):
        attn.apply_index_qk_norm(idx_q, idx_k)


def _make_fused_qk_norm_rope_test_config():
    """Return (text_config, ModelConfig) with a kernel-supported head_dim.

    fused_qk_norm_rope only compiles for head_dim in {64, 128, 256}, so the
    head_dim=32 config from _make_attention_test_config cannot drive the fused
    path. This variant keeps the real M3 head_dim=128, sparse_index_dim=128 and
    rotary_dim=64 geometry with few heads so the tensors stay small.
    """
    n_layers = 4
    sparse_cfg = {
        "use_sparse_attention": True,
        "sparse_index_dim": 128,
        "sparse_num_index_heads": 2,
        "sparse_topk_blocks": 4,
        "sparse_block_size": 16,
        "sparse_init_block": 0,
        "sparse_local_block": 1,
        "sparse_score_type": "max",
        "sparse_disable_index_value": [0, 1, 1, 1],
        "sparse_attention_freq": [0, 1, 1, 1],
    }
    text_cfg = _wrap_dict_as_config(
        {
            "hidden_size": 512,
            "intermediate_size": 128,
            "num_hidden_layers": n_layers,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "vocab_size": 256,
            "max_position_embeddings": 256,
            "rms_norm_eps": 1e-6,
            "use_gemma_norm": True,
            "rope_theta": 5000000.0,
            "rotary_dim": 64,
            "partial_rotary_factor": 0.5,
            "qk_norm_type": "per_head",
            "use_qk_norm": True,
            "sparse_attention_config": sparse_cfg,
            "torch_dtype": torch.bfloat16,
        }
    )
    model_cfg = ModelConfig(
        pretrained_config=text_cfg,
        mapping=Mapping(),
        skip_create_weights_in_init=True,
    )
    return text_cfg, model_cfg


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 fused QK-norm+RoPE needs CUDA")
def test_minimax_m3_fused_qk_norm_rope_main_matches_separate():
    """Fused main-branch helper matches separate norm plus partial RoPE.

    Covers the helper's wiring against the path it replaces: partial rotary dim,
    theta and neox flag from RopeParams, Gemma scaling, per-head norm weights,
    and the norm epsilon.
    """
    _, model_cfg = _make_fused_qk_norm_rope_test_config()
    attn = MiniMaxM3Attention(
        model_config=model_cfg,
        layer_idx=0,
        is_sparse_attention_layer=False,
    )
    device = torch.device("cuda")
    dtype = torch.bfloat16
    head_dim = attn.head_dim

    torch.manual_seed(0)
    attn.q_norm.weight = torch.nn.Parameter(torch.randn(head_dim, dtype=dtype, device=device) * 0.2)
    attn.k_norm.weight = torch.nn.Parameter(torch.randn(head_dim, dtype=dtype, device=device) * 0.2)

    seq = 6
    qkv = torch.randn(seq, attn.q_size + 2 * attn.kv_size, dtype=dtype, device=device)
    position_ids = torch.arange(seq, dtype=torch.int32, device=device) + 3

    # Fused path.
    fused = attn._fused_qk_norm_rope(
        qkv.clone(),
        position_ids,
        num_heads_q=attn.num_heads,
        num_heads_k=attn.num_key_value_heads,
        num_heads_v=attn.num_key_value_heads,
        head_dim=head_dim,
        q_norm=attn.q_norm,
        k_norm=attn.k_norm,
    )
    assert fused is not None, "bf16 qkv + position_ids must take the fused path"
    q_f, k_f, v_f = fused.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)

    # Separate fallback path.
    q_s, k_s, v_s = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    q_s, k_s = attn.apply_qk_norm(q_s, k_s)
    q_s, k_s = attn.rotary_emb(position_ids, [q_s, k_s])

    torch.testing.assert_close(q_f.contiguous(), q_s.contiguous(), rtol=5e-2, atol=1e-1)
    torch.testing.assert_close(k_f.contiguous(), k_s.contiguous(), rtol=5e-2, atol=1e-1)
    # V is untouched by both paths.
    torch.testing.assert_close(v_f.contiguous(), v_s.contiguous(), rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 fused QK-norm+RoPE needs CUDA")
def test_minimax_m3_fused_qk_norm_rope_index_matches_separate():
    """Fused index-branch helper matches the separate path.

    The index branch norms and rotates the concatenated idx_q (per-head) and
    idx_k (single replicated head) with num_heads_v=0, then splits back.
    """
    _, model_cfg = _make_fused_qk_norm_rope_test_config()
    attn = MiniMaxM3Attention(
        model_config=model_cfg,
        layer_idx=3,
        is_sparse_attention_layer=True,
        disable_index_value=True,
    )
    device = torch.device("cuda")
    dtype = torch.bfloat16
    sparse_index_dim = attn.sparse_index_dim
    num_index_heads = attn.sparse_num_index_heads

    torch.manual_seed(1)
    attn.index_q_norm.weight = torch.nn.Parameter(
        torch.randn(sparse_index_dim, dtype=dtype, device=device) * 0.3
    )
    attn.index_k_norm.weight = torch.nn.Parameter(
        torch.randn(sparse_index_dim, dtype=dtype, device=device) * 0.3
    )

    seq = 5
    idx_q = torch.randn(seq, num_index_heads * sparse_index_dim, dtype=dtype, device=device)
    idx_k = torch.randn(seq, sparse_index_dim, dtype=dtype, device=device)
    position_ids = torch.arange(seq, dtype=torch.int32, device=device) + 7

    # Fused path over concatenated [idx_q, idx_k].
    fused = attn._fused_qk_norm_rope(
        torch.cat([idx_q, idx_k], dim=-1),
        position_ids,
        num_heads_q=num_index_heads,
        num_heads_k=1,
        num_heads_v=0,
        head_dim=sparse_index_dim,
        q_norm=attn.index_q_norm,
        k_norm=attn.index_k_norm,
    )
    assert fused is not None
    iq_f, ik_f = fused.split([num_index_heads * sparse_index_dim, sparse_index_dim], dim=-1)

    # Separate fallback path.
    iq_s, ik_s = attn.apply_index_qk_norm(idx_q, idx_k)
    iq_s, ik_s = attn.rotary_emb(position_ids, [iq_s, ik_s])

    torch.testing.assert_close(iq_f.contiguous(), iq_s.contiguous(), rtol=5e-2, atol=1e-1)
    torch.testing.assert_close(ik_f.contiguous(), ik_s.contiguous(), rtol=5e-2, atol=1e-1)


@pytest.mark.cpu_only
def test_minimax_m3_fp8_indexer_rejects_different_qk_norm_epsilons() -> None:
    """The fused kernel has one epsilon, so Q/K norms must agree."""
    attn = MiniMaxM3Attention.__new__(MiniMaxM3Attention)
    nn.Module.__init__(attn)
    backend = MiniMaxM3MsaSparseAttention.__new__(MiniMaxM3MsaSparseAttention)
    backend.indexer_kv_dtype = "fp8"
    attn.attn = backend
    attn.rotary_emb = object()
    attn.pos_embd_params = SimpleNamespace(
        rope=SimpleNamespace(dim=64, theta=5000000.0),
        is_neox=True,
    )
    attn.use_gemma_norm = True
    attn.index_q_norm = SimpleNamespace(variance_epsilon=1e-6)
    attn.index_k_norm = SimpleNamespace(variance_epsilon=1e-5)

    with pytest.raises(ValueError, match=r"identical index Q/K RMSNorm epsilon"):
        attn._fused_fp8_index_qk_norm_rope(
            torch.empty(1, 640, dtype=torch.bfloat16),
            torch.zeros(1, dtype=torch.int32),
            SimpleNamespace(),
        )


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 fused QK-norm+RoPE needs CUDA")
def test_minimax_m3_fused_qk_norm_rope_fallbacks():
    """The fused helper returns None (fallback) for non-bf16 or no position_ids."""
    _, model_cfg = _make_fused_qk_norm_rope_test_config()
    attn = MiniMaxM3Attention(
        model_config=model_cfg,
        layer_idx=0,
        is_sparse_attention_layer=False,
    )
    device = torch.device("cuda")
    seq = 3
    total = attn.q_size + 2 * attn.kv_size
    position_ids = torch.arange(seq, dtype=torch.int32, device=device)

    # No position_ids means RoPE cannot run, so fall back.
    qkv_bf16 = torch.randn(seq, total, dtype=torch.bfloat16, device=device)
    assert (
        attn._fused_qk_norm_rope(
            qkv_bf16,
            None,
            num_heads_q=attn.num_heads,
            num_heads_k=attn.num_key_value_heads,
            num_heads_v=attn.num_key_value_heads,
            head_dim=attn.head_dim,
            q_norm=attn.q_norm,
            k_norm=attn.k_norm,
        )
        is None
    )

    # Non-bf16 activations hit the bf16-only guard, so fall back.
    qkv_fp16 = torch.randn(seq, total, dtype=torch.float16, device=device)
    assert (
        attn._fused_qk_norm_rope(
            qkv_fp16,
            position_ids,
            num_heads_q=attn.num_heads,
            num_heads_k=attn.num_key_value_heads,
            num_heads_v=attn.num_key_value_heads,
            head_dim=attn.head_dim,
            q_norm=attn.q_norm,
            k_norm=attn.k_norm,
        )
        is None
    )


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 attention construction needs CUDA")
def test_minimax_m3_expect_fused_qk_norm_rope_predicate():
    """bf16 activations plus position_ids require the fused kernel.

    M3 keeps bf16 attention activations under every quantization flavor, so a
    runtime fallback there trips the forward assertions. Non-bf16 activations or
    missing position_ids relax the expectation.
    """
    _, model_cfg = _make_fused_qk_norm_rope_test_config()
    attn = MiniMaxM3Attention(
        model_config=model_cfg,
        layer_idx=3,
        is_sparse_attention_layer=True,
        disable_index_value=True,
    )
    device = torch.device("cuda")
    position_ids = torch.arange(4, dtype=torch.int32, device=device)

    # bf16 activations require fusion.
    assert attn.attn_activation_dtype == torch.bfloat16
    assert attn._expect_fused_qk_norm_rope(position_ids) is True

    # No position_ids means RoPE cannot run, so a fallback is allowed.
    assert attn._expect_fused_qk_norm_rope(None) is False

    # Non-bf16 activations allow a fallback with no assertion.
    attn.attn_activation_dtype = torch.float16
    assert attn._expect_fused_qk_norm_rope(position_ids) is False


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 attention construction needs CUDA")
def test_minimax_m3_attention_real_config_index_branch_shapes():
    """Real M3 config → sparse-layer index branch has the checkpoint's shapes.

    Asserts the fused index projection in numbers:
      * index_qk_proj.out_features == 640 (4 * 128 + 128), replicated. Source
        weights (512, 6144) + (128, 6144) merge into (640, 6144) at load time.
      * index_q_norm / index_k_norm weights have shape (128,).
    """
    pytest.importorskip("transformers")
    cfg = AutoConfig.from_pretrained(_checkpoint_path(), trust_remote_code=True)
    text_cfg = get_text_config(cfg)
    # ``MiniMaxM3Model.__init__`` normalises ``torch_dtype`` to a real
    # ``torch.dtype`` before constructing layers (the HF config stores it
    # as the string ``"bfloat16"``). Mirror that here so the standalone
    # attention construction does not blow up inside the RMSNorm
    # ``torch.zeros(..., dtype=dtype)`` call.
    if isinstance(getattr(text_cfg, "torch_dtype", None), str):
        text_cfg.torch_dtype = torch.bfloat16
    elif getattr(text_cfg, "torch_dtype", None) is None:
        text_cfg.torch_dtype = torch.bfloat16

    model_cfg = ModelConfig(
        pretrained_config=text_cfg,
        mapping=Mapping(),
        skip_create_weights_in_init=True,
    )
    attn = MiniMaxM3Attention(
        model_config=model_cfg,
        layer_idx=3,
        is_sparse_attention_layer=True,
        disable_index_value=True,
    )

    sparse_cfg = text_cfg.sparse_attention_config
    num_index_heads = int(sparse_cfg["sparse_num_index_heads"])
    sparse_index_dim = int(sparse_cfg["sparse_index_dim"])
    assert num_index_heads == 4
    assert sparse_index_dim == 128

    assert attn.index_q_size == num_index_heads * sparse_index_dim
    assert attn.index_k_size == sparse_index_dim
    assert attn.index_qk_proj.in_features == int(text_cfg.hidden_size)
    assert attn.index_qk_proj.out_features == num_index_heads * sparse_index_dim + sparse_index_dim
    assert attn.index_qk_proj.tp_mode == modeling_minimaxm3.TensorParallelMode.COLUMN
    assert not hasattr(attn, "index_q_proj")
    assert not hasattr(attn, "index_k_proj")

    # Per-head Gemma index norms: width sparse_index_dim.
    assert tuple(attn.index_q_norm.weight.shape) == (sparse_index_dim,)
    assert tuple(attn.index_k_norm.weight.shape) == (sparse_index_dim,)
    assert attn.index_q_norm.use_gemma is True
    assert attn.index_k_norm.use_gemma is True

    # Main Q/K norm shapes follow head_dim, not hidden_size.
    head_dim = int(text_cfg.head_dim)
    assert tuple(attn.q_norm.weight.shape) == (head_dim,)
    assert tuple(attn.k_norm.weight.shape) == (head_dim,)

    # Partial RoPE rotates rotary_dim of head_dim.
    assert attn.pos_embd_params.rope.dim == int(text_cfg.rotary_dim)
    assert attn.pos_embd_params.rope.dim < head_dim


# ---------------------------------------------------------------------------
# Routing-method unit tests (CPU)
# ---------------------------------------------------------------------------


def test_minimax_m3_routing_method_applies_routed_scaling_factor():
    """MiniMaxM3MoeRoutingMethod multiplies renormalized weights by scaling."""
    num_experts = 8
    top_k = 2
    bias = torch.zeros(num_experts, dtype=torch.float32)

    def bias_fn():
        return bias

    torch.manual_seed(0)
    logits = torch.randn(4, num_experts, dtype=torch.float32) * 3.0

    base = MiniMaxM2MoeRoutingMethod(
        top_k=top_k,
        num_experts=num_experts,
        callable_e_score_correction_bias=bias_fn,
    )
    scaled = MiniMaxM3MoeRoutingMethod(
        top_k=top_k,
        num_experts=num_experts,
        callable_e_score_correction_bias=bias_fn,
        routed_scaling_factor=2.0,
    )

    base_idx, base_weights = base.apply(logits)
    scaled_idx, scaled_weights = scaled.apply(logits)

    torch.testing.assert_close(base_idx, scaled_idx)
    torch.testing.assert_close(scaled_weights, base_weights * 2.0, rtol=0, atol=0)


def test_minimax_m3_routing_method_default_scale_is_identity():
    num_experts = 8
    top_k = 2
    bias = torch.zeros(num_experts, dtype=torch.float32)

    torch.manual_seed(0)
    logits = torch.randn(4, num_experts, dtype=torch.float32) * 3.0

    base = MiniMaxM2MoeRoutingMethod(
        top_k=top_k,
        num_experts=num_experts,
        callable_e_score_correction_bias=lambda: bias,
    )
    same = MiniMaxM3MoeRoutingMethod(
        top_k=top_k,
        num_experts=num_experts,
        callable_e_score_correction_bias=lambda: bias,
        routed_scaling_factor=1.0,
    )
    _, base_weights = base.apply(logits)
    _, same_weights = same.apply(logits)
    torch.testing.assert_close(same_weights, base_weights, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="fused MiniMax-M3 routing requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 64, 8192])
def test_minimax_m3_fused_routing_matches_reference(
    num_tokens: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    num_experts = 128
    top_k = 4
    routed_scaling_factor = 2.0
    logits = torch.full((num_tokens, num_experts), -4.0, device="cuda", dtype=torch.float32)
    token_offsets = torch.arange(num_tokens, device="cuda", dtype=torch.int64).unsqueeze(1)
    expert_offsets = torch.arange(top_k, device="cuda", dtype=torch.int64).unsqueeze(0)
    selected_experts = (token_offsets + expert_offsets) % num_experts
    selected_logits = torch.tensor([4.0, 3.0, 2.0, 1.0], device="cuda").expand(num_tokens, -1)
    logits.scatter_(1, selected_experts, selected_logits)
    bias = torch.linspace(-0.01, 0.01, num_experts, device="cuda", dtype=torch.float32)

    fused = MiniMaxM3MoeRoutingMethod(
        top_k=top_k,
        num_experts=num_experts,
        callable_e_score_correction_bias=lambda: bias,
        routed_scaling_factor=routed_scaling_factor,
    )

    scores = torch.sigmoid(logits)
    _, reference_idx = torch.topk(scores + bias, k=top_k, dim=-1, sorted=False)
    reference_weights = scores.gather(1, reference_idx)
    reference_idx = reference_idx.to(torch.int32)
    reference_weights = (
        reference_weights / (reference_weights.sum(dim=-1, keepdim=True) + 1e-20)
    ) * routed_scaling_factor

    monkeypatch.setattr(
        MiniMaxM2MoeRoutingMethod,
        "apply",
        lambda *_args, **_kwargs: pytest.fail("production FP32 routing used the PyTorch fallback"),
    )
    fused_idx, fused_weights = fused.apply(logits)

    reference_order = reference_idx.argsort(dim=-1)
    fused_order = fused_idx.argsort(dim=-1)
    reference_idx = reference_idx.gather(1, reference_order)
    reference_weights = reference_weights.gather(1, reference_order)
    fused_idx = fused_idx.gather(1, fused_order)
    fused_weights = fused_weights.gather(1, fused_order)

    torch.testing.assert_close(fused_idx, reference_idx, rtol=0, atol=0)
    torch.testing.assert_close(fused_weights, reference_weights, rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="fused MiniMax-M3 routing requires CUDA")
def test_minimax_m3_fused_routing_cuda_graph_replay_tracks_inputs() -> None:
    num_tokens = 64
    num_experts = 128
    torch.manual_seed(1)
    logits = torch.empty(num_tokens, num_experts, device="cuda", dtype=torch.float32)
    bias = torch.randn(num_experts, device="cuda", dtype=torch.float32) * 0.1
    routing = MiniMaxM3MoeRoutingMethod(
        top_k=4,
        num_experts=num_experts,
        callable_e_score_correction_bias=lambda: bias,
        routed_scaling_factor=2.0,
    )

    logits.normal_()
    routing.apply(logits)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_idx, graph_weights = routing.apply(logits)

    logits.normal_(mean=1.0, std=2.0)
    reference_idx, reference_weights = routing.apply(logits.clone())
    graph.replay()

    torch.testing.assert_close(graph_idx, reference_idx, rtol=0, atol=0)
    torch.testing.assert_close(graph_weights, reference_weights, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 needs CUDA")
def test_text_norm_weights_real_loader_smoke(monkeypatch: pytest.MonkeyPatch):
    """real ``_load_weights_impl_v2`` populates norm parameters.

    Constructs a memory-safe stub containing the top-level ``model.norm``
    and the first decoder layer's ``input_layernorm`` and
    ``post_attention_layernorm`` (each a 6144-dim BF16
    :class:`RMSNorm`, ~12 KB on CUDA), reads the corresponding tensors
    from the real checkpoint via ``safetensors``, strips the
    ``language_model.`` prefix exactly as the M3 VL wrapper does, and
    invokes :func:`_load_weights_impl_v2` end-to-end. The test fails if any
    target parameter remains at its zero-initialisation, proving the
    canonical loader walks the module tree and copies the correct source
    keys for these BF16 parameters.

    Why this slice: ``input_layernorm`` / ``post_attention_layernorm`` /
    ``model.norm`` exercise the loader's ``filter_weights`` + per-module
    copy path on real tensor handles.
    """
    pytest.importorskip("safetensors")
    pytest.importorskip("transformers")

    checkpoint = _checkpoint_path()
    cfg = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    text_cfg = get_text_config(cfg)

    eps = float(text_cfg.rms_norm_eps)
    use_gemma = bool(getattr(text_cfg, "use_gemma_norm", False))
    hidden = int(text_cfg.hidden_size)
    dtype = torch.bfloat16
    device = torch.device("cuda")

    # Memory-safe stub matching the M3 module tree for the three norms.
    class _Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_layernorm = RMSNorm(
                hidden_size=hidden,
                eps=eps,
                dtype=dtype,
                device=device,
                use_gemma=use_gemma,
            )
            self.post_attention_layernorm = RMSNorm(
                hidden_size=hidden,
                eps=eps,
                dtype=dtype,
                device=device,
                use_gemma=use_gemma,
            )

    class _ModelInner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = RMSNorm(
                hidden_size=hidden,
                eps=eps,
                dtype=dtype,
                device=device,
                use_gemma=use_gemma,
            )
            self.layers = nn.ModuleList([_Layer()])

    class _LoaderStub(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model_config = ModelConfig(pretrained_config=text_cfg, mapping=Mapping())
            self.config = text_cfg
            self.model = _ModelInner()

    stub = _LoaderStub()

    # Sanity: the RMSNorm init zero-fills the gemma path; if any tensor is
    # already equal to its checkpoint value we would not be testing the copy.
    assert torch.all(stub.model.norm.weight == 0)
    assert torch.all(stub.model.layers[0].input_layernorm.weight == 0)
    assert torch.all(stub.model.layers[0].post_attention_layernorm.weight == 0)

    targets = [
        "language_model.model.norm.weight",
        "language_model.model.layers.0.input_layernorm.weight",
        "language_model.model.layers.0.post_attention_layernorm.weight",
    ]

    # Group source keys by safetensors shard for efficient reads.
    with open(os.path.join(checkpoint, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    by_shard: dict = {}
    for key in targets:
        shard = weight_map[key]
        by_shard.setdefault(shard, []).append(key)

    raw_weights: dict = {}
    for shard, keys in by_shard.items():
        with safe_open(os.path.join(checkpoint, shard), framework="pt", device="cpu") as sf:
            for k in keys:
                raw_weights[k] = sf.get_tensor(k)

    # Strip `language_model.` exactly as `MiniMaxM3VLForConditionalGeneration`
    # does at load time. Confirm the stripped keyspace matches the inner
    # loader's expectation.
    text_weights, ignored = _strip_language_model_prefix(raw_weights)
    assert ignored == {}, f"unexpectedly stripped {len(ignored)} entries: {ignored!r}"
    assert set(text_weights.keys()) == {
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
    }

    # Invoke the canonical loader. `_load_weights_impl_v2` walks the stub's
    # module tree and uses the generic per-parameter copy fallback because
    # RMSNorm does not define ``load_weights``. Disable the parallel
    # executor so a failure surfaces immediately rather than as a thread
    # traceback (the parallel path is exercised in production; for this
    # tiny 3-module slice the serial walk is what the test should observe).
    monkeypatch.setenv("TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL", "True")
    weight_mapper = MiniMaxM3HfWeightMapper()
    weight_mapper.init_model_and_config(stub, stub.model_config)
    _load_weights_impl_v2(
        stub,
        text_weights,
        weight_mapper,
        allow_partial_loading=True,
    )

    # The three norms should now hold the source tensors' values.
    torch.testing.assert_close(
        stub.model.norm.weight.detach().cpu().to(torch.bfloat16),
        raw_weights["language_model.model.norm.weight"].to(torch.bfloat16),
    )
    torch.testing.assert_close(
        stub.model.layers[0].input_layernorm.weight.detach().cpu().to(torch.bfloat16),
        raw_weights["language_model.model.layers.0.input_layernorm.weight"].to(torch.bfloat16),
    )
    torch.testing.assert_close(
        stub.model.layers[0].post_attention_layernorm.weight.detach().cpu().to(torch.bfloat16),
        raw_weights["language_model.model.layers.0.post_attention_layernorm.weight"].to(
            torch.bfloat16
        ),
    )

    # Independent sanity: at least one tensor must be non-zero, i.e. the
    # loader actually performed the copy.
    assert torch.any(stub.model.norm.weight != 0)


# ---------------------------------------------------------------------------
# Attention DP construction for the dense MLP / MoE shared expert
# ---------------------------------------------------------------------------
#
# Under ``enable_attention_dp=True`` each rank processes a rank-local
# set of tokens. The base ``Attention`` re-maps tp_size=1 internally so
# qkv_proj/o_proj are replicated. The MiniMax-M3 dense MLP / MoE shared
# expert (a ``GatedMLP`` built by ``_build_swiglu_oai_dense_mlp``) must
# follow the same pattern: ``overridden_tp_size=1`` + ``reduce_output=
# False`` so it runs replicated under ADP. A ROW-parallel all-reduce
# across ADP ranks would mix outputs from independent rank-local token
# sets and produce wrong results. This test pins the construction-level
# invariants of that contract.


@pytest.mark.gpu
@pytest.mark.skipif(
    not _has_cuda(), reason="MiniMax-M3 MLP construction needs CUDA backend selection"
)
def test_minimax_m3_swiglu_oai_dense_mlp_under_adp_is_replicated():
    """``_build_swiglu_oai_dense_mlp`` under ``enable_attention_dp=True``
    must produce a ``GatedMLP`` whose Linear layers are replicated
    (full-width in_features/out_features) with ``reduce_output=False``
    on ``down_proj``. Without this the dense MLP / shared expert would
    all-reduce across ADP ranks and mix outputs from independent
    rank-local token sets.
    """
    text_cfg = _wrap_dict_as_config(
        {
            "hidden_size": 128,
            "intermediate_size": 64,
            "swiglu_alpha": 1.702,
            "swiglu_limit": 7.0,
            "torch_dtype": torch.bfloat16,
        }
    )
    # Simulate ADP with tp_size=4; world_size matches so Mapping
    # validation passes.
    model_cfg = ModelConfig(
        pretrained_config=text_cfg,
        mapping=Mapping(world_size=4, tp_size=4, pp_size=1, rank=0, enable_attention_dp=True),
        skip_create_weights_in_init=True,
    )

    intermediate = 64
    hidden = int(text_cfg.hidden_size)
    mlp = _build_swiglu_oai_dense_mlp(
        model_config=model_cfg,
        intermediate_size=intermediate,
    )
    # Under ADP the gate_up_proj and down_proj must be replicated:
    # in_features and out_features keep their full size and the
    # Linear-internal tp_size is 1.
    assert mlp.gate_up_proj.in_features == hidden
    assert mlp.gate_up_proj.out_features == 2 * intermediate, (
        "ADP gate_up_proj must be full-width (replicated), not sharded by the global TP size"
    )
    assert mlp.gate_up_proj.tp_size == 1
    assert mlp.down_proj.in_features == intermediate, (
        "ADP down_proj must be full-width (replicated), not ROW-sharded across the global TP group"
    )
    assert mlp.down_proj.out_features == hidden
    assert mlp.down_proj.tp_size == 1
    assert mlp.down_proj.reduce_output is False, (
        "ADP down_proj must skip the cross-rank all-reduce; otherwise it "
        "mixes outputs across independent rank-local token sets"
    )


# ---------------------------------------------------------------------------
# Fused SwiGLU-OAI numeric equivalence
# ---------------------------------------------------------------------------
#
# The dense MLP and MoE shared expert express swigluoai as plain SwiGLU with
# (alpha, beta, limit) so it routes through the fused silu_and_mul kernel.


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="fused silu_and_mul Triton kernel requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_minimax_m3_swiglu_oai_fused_matches_reference(dtype):
    """Fused silu_and_mul with (alpha, beta, limit) matches the eager reference.

    Also checks that alpha=1 and beta=0 recover plain SwiGLU.
    """
    from tensorrt_llm._torch.modules.swiglu import swiglu

    torch.manual_seed(0)
    alpha, limit = 1.702, 7.0
    # Wide range so both clamp branches (gate upper, up symmetric) are hit.
    gate_up = torch.randn(64, 2 * 128, device="cuda", dtype=dtype) * 6.0

    ref = _minimax_m3_swiglu_oai(gate_up, alpha=alpha, limit=limit)
    fused = swiglu(gate_up, swiglu_alpha=alpha, swiglu_beta=1.0, swiglu_limit=limit)

    assert fused.shape == ref.shape
    assert fused.dtype == gate_up.dtype
    # fp32 accumulation inside the kernel; loose tol for bf16 rounding.
    atol = 2e-2 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(fused.float(), ref.float(), atol=atol, rtol=1e-2)

    # alpha=1 / beta=0 reduces to plain SwiGLU: silu(gate_clamped) * up_clamped.
    gate, up = gate_up.chunk(2, dim=-1)
    gate_c = gate.clamp(max=limit)
    up_c = up.clamp(min=-limit, max=limit)
    plain_ref = torch.nn.functional.silu(gate_c) * up_c
    plain = swiglu(gate_up, swiglu_limit=limit)
    torch.testing.assert_close(plain.float(), plain_ref.float(), atol=atol, rtol=1e-2)
