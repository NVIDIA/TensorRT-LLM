# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRTLLM FP4 MLA helper tests."""

from types import SimpleNamespace
from typing import Optional

import pytest
import torch

import tensorrt_llm
import tensorrt_llm._torch.attention_backend.fmha.fp4_mla as fp4_mla_fmha_backend
import tensorrt_llm._torch.attention_backend.fp4_mla as fp4_mla_backend
import tensorrt_llm._torch.attention_backend.trtllm as trtllm_backend
import tensorrt_llm._torch.modules.mla as mla_module
from tensorrt_llm._torch.attention_backend.fmha.fp4_mla import Fp4MlaFmha
from tensorrt_llm._torch.attention_backend.fp4_mla import (
    FP4_BLOCK_SIZE,
    FP4_MLA_ATTENTION_BACKEND_ENV,
    FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV,
    FP4_MLA_CUTEDSL_MUFU16_ENV,
    FP4_MLA_K_RESIDUAL_DIM,
    FP4_MLA_KV_GLOBAL_SCALE,
    FP4_MLA_P_GLOBAL_SCALE,
    FP4_MLA_PACKED_SINGLE_COPY_ENV,
    FP4_MLA_Q_GLOBAL_SCALE,
    FP4_MLA_Q_RESIDUAL_DIM,
    FP4_MLA_TOKENS_PER_BLOCK,
    HP_BLOCK_SIZE,
    Fp4MlaVLayout,
    _cutedsl_backend_available,
    _fp4_mla_attention_backend,
    _get_fp4_mla_global_scale,
    _get_fp4_mla_hp_pool_layout,
    load_fp4_mla_chunked_kv_cache,
    run_fp4_mla_attention_decode,
    scatter_fp4_mla_kv_cache,
)
from tensorrt_llm._torch.attention_backend.fp4_mla.cache_manager import Fp4MlaKVCacheManagerV2
from tensorrt_llm._torch.attention_backend.fp4_mla.fp4_mla_context import (
    _build_fp8_mla_context_metadata,
)
from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs, AttentionInputType
from tensorrt_llm._torch.kimi_k3_cache_policy import KIMI_K3_FP8_KV_LAYERS_ENV
from tensorrt_llm._torch.modules.mla import MLA
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm.llmapi.llm_args import KvCacheConfig as LlmKvCacheConfig
from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import PageIndexConverter

_DataType = tensorrt_llm.bindings.DataType
_CacheType = tensorrt_llm.bindings.internal.batch_manager.CacheType


def _swizzled_sf_offset(row_idx: int, col_idx: int, sf_per_token: int) -> int:
    padded_cols = ((sf_per_token + 3) // 4) * 4
    return (
        col_idx % 4
        + (col_idx // 4) * (4 * 128)
        + (row_idx % 32) * 16
        + ((row_idx % 128) // 32) * 4
        + (row_idx // 128) * (128 * padded_cols)
    )


def _is_cutedsl_unavailable() -> bool:
    return (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (10, 7)
        or not _cutedsl_backend_available()
    )


def _set_fp4_mla_v_layout_env(monkeypatch, v_layout: Fp4MlaVLayout) -> None:
    """Select ``v_layout`` for the FP4 MLA KV cache managers built next."""
    monkeypatch.setenv(
        FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV, str(int(v_layout is Fp4MlaVLayout.FUSED_V))
    )
    monkeypatch.setenv(
        FP4_MLA_PACKED_SINGLE_COPY_ENV,
        str(int(v_layout is Fp4MlaVLayout.PACKED_SINGLE_COPY)),
    )


@pytest.mark.parametrize(
    ("backend", "fused_v", "single_copy", "expected"),
    [
        ("cutedsl", False, False, Fp4MlaVLayout.DUAL_COPY),
        ("cutedsl", True, False, Fp4MlaVLayout.FUSED_V),
        ("cutedsl", False, True, Fp4MlaVLayout.PACKED_SINGLE_COPY),
        ("triton", False, False, Fp4MlaVLayout.DUAL_COPY),
        ("triton", True, False, Fp4MlaVLayout.DUAL_COPY),
    ],
    ids=["dual-copy", "fused-v", "single-copy", "triton", "triton-ignores-fused-v"],
)
def test_fp4_mla_v_layout_resolves_from_the_environment(
    monkeypatch, backend: str, fused_v: bool, single_copy: bool, expected: Fp4MlaVLayout
) -> None:
    monkeypatch.setenv(FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV, str(int(fused_v)))
    monkeypatch.setenv(FP4_MLA_PACKED_SINGLE_COPY_ENV, str(int(single_copy)))
    monkeypatch.delenv(fp4_mla_backend.FP4_MLA_PACKED_Q_ENV, raising=False)

    assert fp4_mla_backend.resolve_fp4_mla_v_layout(backend) is expected


@pytest.mark.parametrize(
    ("backend", "fused_v", "packed_q", "match"),
    [
        ("cutedsl", True, "1", "set at most one"),
        ("triton", False, "1", "requires the cutedsl FP4 MLA backend"),
        ("cutedsl", False, "0", "requires the packed decode"),
    ],
    ids=["with-fused-v", "triton", "packed-q-off"],
)
def test_fp4_mla_single_copy_layout_rejects_what_it_cannot_serve(
    monkeypatch, backend: str, fused_v: bool, packed_q: str, match: str
) -> None:
    monkeypatch.setenv(FP4_MLA_PACKED_SINGLE_COPY_ENV, "1")
    monkeypatch.setenv(FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV, str(int(fused_v)))
    monkeypatch.setenv(fp4_mla_backend.FP4_MLA_PACKED_Q_ENV, packed_q)

    with pytest.raises(ValueError, match=match):
        fp4_mla_backend.resolve_fp4_mla_v_layout(backend)


@pytest.mark.parametrize(
    ("backend", "packed_q"), [("triton", "1"), ("cutedsl", "0")], ids=["triton", "packed-q-off"]
)
def test_fp4_mla_context_only_cache_takes_the_single_copy_switch_unchecked(
    monkeypatch, backend: str, packed_q: str
) -> None:
    """A context-only (CTX) cache never decodes, so a CTX worker may share the generation
    workers' environment whatever its backend or packed-decode setting."""
    monkeypatch.setenv(FP4_MLA_PACKED_SINGLE_COPY_ENV, "1")
    monkeypatch.setenv(FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV, "0")
    monkeypatch.setenv(fp4_mla_backend.FP4_MLA_PACKED_Q_ENV, packed_q)

    assert (
        fp4_mla_backend.resolve_fp4_mla_v_layout(backend, context_only=True)
        is Fp4MlaVLayout.PACKED_SINGLE_COPY
    )
    with pytest.raises(ValueError, match="PACKED_SINGLE_COPY requires"):
        fp4_mla_backend.resolve_fp4_mla_v_layout(backend)


def test_fp4_mla_v_layout_is_read_from_the_manager_only() -> None:
    manager = SimpleNamespace(fp4_mla_v_layout=Fp4MlaVLayout.PACKED_SINGLE_COPY)
    assert fp4_mla_backend._fp4_mla_v_layout(manager) is Fp4MlaVLayout.PACKED_SINGLE_COPY
    for unconfigured in (None, SimpleNamespace(), SimpleNamespace(fp4_mla_v_layout="dual_copy")):
        with pytest.raises(RuntimeError, match="configured with an FP4 MLA V layout"):
            fp4_mla_backend._fp4_mla_v_layout(unconfigured)


def test_fp4_mla_request_validation_uses_sparse_runtime_params() -> None:
    metadata = SimpleNamespace(
        num_sparse_topk=0,
        kv_cache_manager=SimpleNamespace(dtype=_DataType.NVFP4, kv_factor=1),
        high_precision_kv_pool=object(),
        fp4_mla_v_scale_pool=object(),
        beam_width=1,
        helix_position_offsets=None,
    )
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.generation_only)

    Fp4MlaFmha._validate_request(None, None, None, metadata, forward_args)

    forward_args.sparse_runtime_params.sparse_attn_indices = torch.tensor([0])
    with pytest.raises(NotImplementedError, match="does not support sparse attention"):
        Fp4MlaFmha._validate_request(None, None, None, metadata, forward_args)


def test_fp4_mla_request_validation_still_requires_nvfp4_kv() -> None:
    metadata = SimpleNamespace(
        num_sparse_topk=0,
        kv_cache_manager=SimpleNamespace(dtype=object(), kv_factor=1),
        high_precision_kv_pool=object(),
        fp4_mla_v_scale_pool=object(),
        beam_width=1,
        helix_position_offsets=None,
    )
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.generation_only)

    with pytest.raises(RuntimeError, match="requires NVFP4 KV cache storage"):
        Fp4MlaFmha._validate_request(None, None, None, metadata, forward_args)


@pytest.mark.parametrize(
    "softmax_stats, expected_exception",
    [
        (None, RuntimeError),
        (torch.empty(2, 3, 2, dtype=torch.bfloat16), ValueError),
        (torch.empty(2, 3, 3, dtype=torch.float32), ValueError),
        (torch.empty(2, 2, 3, dtype=torch.float32).transpose(1, 2), ValueError),
    ],
)
def test_fp4_mla_helix_validates_softmax_stats(
    softmax_stats: torch.Tensor | None,
    expected_exception: type[Exception],
) -> None:
    num_tokens, num_heads = 2, 3
    metadata = SimpleNamespace(
        num_sparse_topk=0,
        kv_cache_manager=SimpleNamespace(dtype=_DataType.NVFP4, kv_factor=1),
        high_precision_kv_pool=object(),
        fp4_mla_v_scale_pool=object(),
        beam_width=1,
        helix_position_offsets=torch.arange(num_tokens, dtype=torch.int32),
        num_generations=num_tokens,
        num_tokens=num_tokens,
    )
    forward_args = AttentionForwardArgs(
        attention_input_type=AttentionInputType.generation_only,
        softmax_stats_tensor=softmax_stats,
    )
    fmha = SimpleNamespace(attn=SimpleNamespace(num_heads=num_heads))

    with pytest.raises(
        expected_exception,
        match="FP4 MLA Helix requires.*softmax stats|requires softmax_stats_tensor",
    ):
        Fp4MlaFmha._validate_request(fmha, None, None, metadata, forward_args)


def test_fp4_mla_helix_accepts_valid_softmax_stats() -> None:
    num_generations, tokens_per_generation, num_heads = 2, 6, 3
    num_tokens = num_generations * tokens_per_generation
    metadata = SimpleNamespace(
        num_sparse_topk=0,
        kv_cache_manager=SimpleNamespace(dtype=_DataType.NVFP4, kv_factor=1),
        high_precision_kv_pool=object(),
        fp4_mla_v_scale_pool=object(),
        beam_width=1,
        helix_position_offsets=torch.arange(num_tokens, dtype=torch.int32),
        num_generations=num_generations,
        num_tokens=num_tokens,
    )
    forward_args = AttentionForwardArgs(
        attention_input_type=AttentionInputType.generation_only,
        softmax_stats_tensor=torch.empty(
            num_tokens,
            num_heads,
            2,
            dtype=torch.float32,
        ),
    )
    fmha = SimpleNamespace(attn=SimpleNamespace(num_heads=num_heads))

    Fp4MlaFmha._validate_request(fmha, None, None, metadata, forward_args)


def test_fp4_mla_helix_stats_require_cutedsl_backend(monkeypatch) -> None:
    kv_lora_rank = 512
    qk_rope_head_dim = FP4_MLA_K_RESIDUAL_DIM
    head_dim = kv_lora_rank + qk_rope_head_dim
    metadata = SimpleNamespace(
        page_size=FP4_MLA_TOKENS_PER_BLOCK,
        num_seqs=1,
        num_contexts=0,
    )
    q = torch.empty(1, 1, head_dim, dtype=torch.bfloat16)
    output = torch.empty(1, 1, kv_lora_rank, dtype=torch.bfloat16)
    softmax_stats = torch.empty(1, 1, 2, dtype=torch.float32)

    monkeypatch.setattr(fp4_mla_backend, "_fp4_mla_attention_backend", lambda: "triton")

    with pytest.raises(
        NotImplementedError,
        match="softmax stats require the cutedsl attention backend",
    ):
        fp4_mla_backend.run_fp4_mla_attention_decode(
            metadata,
            layer_idx=0,
            local_layer=0,
            q=q,
            output=output,
            sm_scale=1.0,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            prequantized_q=None,
            prequantized_q_sf=None,
            q_batch_capacity=None,
            softmax_stats_tensor=softmax_stats,
        )


def test_fp4_mla_generation_forwards_helix_softmax_stats(monkeypatch) -> None:
    num_tokens, num_heads, kv_lora_rank, qk_rope_head_dim = 2, 3, 4, 2
    fused_head_dim = kv_lora_rank + qk_rope_head_dim
    q = torch.empty(num_tokens, num_heads * fused_head_dim, dtype=torch.bfloat16)
    output = torch.empty(num_tokens, num_heads * kv_lora_rank, dtype=torch.bfloat16)
    softmax_stats = torch.empty(num_tokens, num_heads, 2, dtype=torch.float32)
    expected_stats = torch.tensor([1.25, 2.5], dtype=torch.float32).expand_as(softmax_stats)
    metadata = SimpleNamespace(
        num_generations=num_tokens,
        helix_position_offsets=torch.tensor([384, 769], dtype=torch.int32),
        _helix_spec_tokens_valid=False,
        _fp4_mla_generation_cache_scattered=True,
        _fp4_mla_prequantized_q=object(),
        _fp4_mla_prequantized_q_sf=object(),
        _fp4_mla_q_batch_capacity=num_tokens,
    )
    attn = SimpleNamespace(
        get_fp4_mla_local_layer_idx=lambda observed: 5,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        num_heads=num_heads,
        q_scaling=1.0,
        qk_nope_head_dim=2,
        layer_idx=7,
    )
    forward_args = AttentionForwardArgs(
        attention_input_type=AttentionInputType.generation_only,
        latent_cache=torch.empty(num_tokens, fused_head_dim, dtype=torch.bfloat16),
        softmax_stats_tensor=softmax_stats,
    )
    params = SimpleNamespace(
        attn=attn,
        meta=metadata,
        fwd=forward_args,
        qkv_input=q,
        context_buf=output,
    )
    calls = []

    def record_decode(*args, **kwargs) -> None:
        calls.append((args, kwargs))
        kwargs["softmax_stats_tensor"].copy_(expected_stats)

    monkeypatch.setattr(fp4_mla_fmha_backend, "run_fp4_mla_attention_decode", record_decode)

    Fp4MlaFmha.run_mla_generation(None, params)

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[:3] == (metadata, 7, 5)
    assert args[3].shape == (num_tokens, num_heads, fused_head_dim)
    assert args[4].shape == (num_tokens, num_heads, kv_lora_rank)
    assert kwargs["softmax_stats_tensor"] is softmax_stats
    assert softmax_stats.is_contiguous()
    torch.testing.assert_close(softmax_stats, expected_stats)
    assert not metadata._fp4_mla_generation_cache_scattered
    assert metadata._fp4_mla_prequantized_q is None
    assert metadata._fp4_mla_prequantized_q_sf is None
    assert metadata._fp4_mla_q_batch_capacity is None


def test_fp4_mla_rope_generation_forwards_helix_cache_ownership(
    monkeypatch,
) -> None:
    num_tokens, num_heads = 2, 3
    kv_lora_rank, qk_rope_head_dim = 4, 2
    fused_q = torch.empty(
        num_tokens,
        num_heads,
        kv_lora_rank + qk_rope_head_dim,
        dtype=torch.bfloat16,
    )
    q_pe = torch.empty(
        num_tokens,
        num_heads,
        qk_rope_head_dim,
        dtype=torch.bfloat16,
    )
    latent_cache = torch.empty(
        num_tokens,
        kv_lora_rank + qk_rope_head_dim,
        dtype=torch.bfloat16,
    )
    helix_position_offsets = torch.tensor([384, 769], dtype=torch.int32)
    helix_is_inactive_rank = torch.tensor([True, False])
    metadata = SimpleNamespace(
        num_generations=num_tokens,
        num_ctx_tokens=0,
        helix_position_offsets=helix_position_offsets,
        helix_is_inactive_rank=helix_is_inactive_rank,
        _fp4_mla_generation_cache_scattered=False,
        _helix_spec_tokens_valid=False,
    )
    attn = SimpleNamespace(
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        rotary_cos_sin=torch.empty(1024, qk_rope_head_dim, 2, dtype=torch.float32),
        layer_idx=7,
        can_fuse_fp4_mla_q_quant=lambda *args: True,
        get_fp4_mla_local_layer_idx=lambda observed: 5,
    )
    calls = []

    def record_scatter(*args, **kwargs) -> bool:
        calls.append((args, kwargs))
        return True

    monkeypatch.setattr(trtllm_backend, "scatter_fp4_mla_kv_cache", record_scatter)

    trtllm_backend.TrtllmAttention._fp4_mla_rope_generation(
        attn,
        fused_q,
        q_pe,
        latent_cache,
        metadata,
        fuse_q_quant=True,
    )

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[:3] == (metadata, latent_cache, 7)
    assert kwargs["helix_position_offsets"] is helix_position_offsets
    assert kwargs["helix_is_inactive_rank"] is helix_is_inactive_rank
    assert kwargs["q_quant_input"] is fused_q
    assert metadata._fp4_mla_generation_cache_scattered

    metadata.helix_position_offsets = None
    metadata.helix_is_inactive_rank = None
    metadata._fp4_mla_generation_cache_scattered = False
    trtllm_backend.TrtllmAttention._fp4_mla_rope_generation(
        attn,
        fused_q,
        q_pe,
        latent_cache,
        metadata,
        fuse_q_quant=True,
    )
    assert len(calls) == 2
    _, non_helix_kwargs = calls[1]
    assert non_helix_kwargs["helix_position_offsets"] is None
    assert non_helix_kwargs["helix_is_inactive_rank"] is None
    assert metadata._fp4_mla_generation_cache_scattered

    metadata.num_generations = 1
    metadata.helix_position_offsets = helix_position_offsets
    metadata.helix_is_inactive_rank = torch.tensor([False])
    metadata.helix_local_slots = torch.tensor([384, 385], dtype=torch.int32)
    metadata._helix_spec_tokens_valid = True
    metadata._fp4_mla_generation_cache_scattered = False
    trtllm_backend.TrtllmAttention._fp4_mla_rope_generation(
        attn,
        fused_q,
        q_pe,
        latent_cache,
        metadata,
        fuse_q_quant=True,
    )
    assert len(calls) == 3
    _, spec_helix_kwargs = calls[2]
    assert spec_helix_kwargs["helix_position_offsets"] is helix_position_offsets
    assert spec_helix_kwargs["helix_is_inactive_rank"] is metadata.helix_is_inactive_rank
    assert metadata._fp4_mla_generation_cache_scattered


@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("has_latent_cache", [True, False])
def test_mla_fp4_helix_generation_reaches_fused_cache_update(
    monkeypatch, weight_dtype, has_latent_cache
) -> None:
    num_tokens, num_heads = 2, 2
    qk_nope_head_dim, qk_rope_head_dim = 2, 2
    kv_lora_rank = 4

    class ReachedFusedCacheUpdate(Exception):
        pass

    class FakeFp4Attention:
        uses_fp4_mla_attention = True
        has_fp4_kv_cache = True
        has_fp8_kv_cache = False

        def mla_rope_generation(self, *args, **kwargs) -> None:
            assert kwargs == {"fuse_fp4_q_quant": True}
            raise ReachedFusedCacheUpdate

    mla = SimpleNamespace(
        num_heads_tp=num_heads,
        qk_head_dim=qk_nope_head_dim + qk_rope_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        kv_lora_rank=kv_lora_rank,
        mqa=FakeFp4Attention(),
        mapping=SimpleNamespace(has_cp_helix=lambda: True),
        k_b_proj_trans=torch.empty(
            num_heads,
            kv_lora_rank,
            qk_nope_head_dim,
            dtype=weight_dtype,
        ),
        k_b_proj_trans_scale=None,
        k_b_proj_trans_dequant=None,
        use_cute_dsl_blockscaling_bmm=False,
        apply_rotary_emb=False,
        aux_stream=None,
        ln_events=[None, None],
        _ugpu_k_b_proj_trans_shards=None,
        _bmm_bf16_out=lambda *args: None,
    )
    metadata = SimpleNamespace(
        num_seqs=num_tokens,
        num_ctx_tokens=0,
        _helix_spec_tokens_valid=False,
    )

    monkeypatch.setattr(mla_module, "TrtllmAttention", FakeFp4Attention)

    monkeypatch.setattr(mla_module, "fp8_block_scaling_bmm_out", lambda *args: None)
    error = ReachedFusedCacheUpdate if has_latent_cache else RuntimeError
    match = None if has_latent_cache else "needs a latent cache"
    latent_cache = (
        torch.empty(num_tokens, kv_lora_rank + qk_rope_head_dim, dtype=torch.bfloat16)
        if has_latent_cache
        else None
    )

    with pytest.raises(error, match=match):
        MLA.forward_absorption_generation(
            mla,
            torch.empty(
                num_tokens,
                num_heads * (qk_nope_head_dim + qk_rope_head_dim),
                dtype=torch.bfloat16,
            ),
            compressed_kv=torch.empty(0),
            k_pe=torch.empty(num_tokens, qk_rope_head_dim, dtype=torch.bfloat16),
            attn_metadata=metadata,
            output=torch.empty(0),
            position_ids=torch.arange(num_tokens),
            latent_cache=latent_cache,
        )


def test_k3_fp8_split_k_context_selection_is_shape_gated(monkeypatch) -> None:
    import sys
    from types import ModuleType
    from unittest.mock import Mock

    import tensorrt_llm._torch.attention_backend.fmha.fp4_mla as fp4_mla_fmha_module

    class FakeAttention:
        is_mla_enable = True
        kv_lora_rank = 512
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        v_head_dim = 128
        num_heads = 96
        num_kv_heads = 96

    attn = FakeAttention()
    fmha = Fp4MlaFmha(attn)
    custom_op_module_name = "tensorrt_llm._torch.custom_ops.cute_dsl_k3_context_custom_ops"
    custom_op_module = ModuleType(custom_op_module_name)
    custom_op_module.IS_K3_CONTEXT_FMHA_AVAILABLE = True
    custom_op_module.K3_CONTEXT_FMHA_IMPORT_ERROR = None
    monkeypatch.setitem(sys.modules, custom_op_module_name, custom_op_module)
    monkeypatch.setattr(fp4_mla_fmha_module, "get_sm_version", lambda: 107)
    debug_once = Mock()
    monkeypatch.setattr(fp4_mla_fmha_module.logger, "debug_once", debug_once)
    warning_once = Mock()
    monkeypatch.setattr(fp4_mla_fmha_module.logger, "warning_once", warning_once)

    assert fmha.supports_k3_fp8_split_k_context()
    debug_once.assert_not_called()

    for heads in (48, 24, 12, 6, 3):
        attn.num_heads = attn.num_kv_heads = heads
        assert fmha.supports_k3_fp8_split_k_context()
    debug_once.assert_not_called()

    attn.num_heads, attn.num_kv_heads = 12, 6
    assert not fmha.supports_k3_fp8_split_k_context()
    assert "matching positive Q/KV head counts" in debug_once.call_args.args[0]
    assert "got 12/6" in debug_once.call_args.args[0]
    key = debug_once.call_args.kwargs["key"]
    assert not fmha.supports_k3_fp8_split_k_context()
    assert debug_once.call_args.kwargs["key"] == key

    attn.num_heads = attn.num_kv_heads = 0
    assert not fmha.supports_k3_fp8_split_k_context()

    attn.num_heads = attn.num_kv_heads = 96
    monkeypatch.setattr(fp4_mla_fmha_module, "get_sm_version", lambda: 103)
    assert not fmha.supports_k3_fp8_split_k_context()
    assert "requires SM107, got SM103" in debug_once.call_args.args[0]

    monkeypatch.setattr(fp4_mla_fmha_module, "get_sm_version", lambda: 107)
    attn.qk_nope_head_dim = 64
    assert not fmha.supports_k3_fp8_split_k_context()
    assert "requires head dimensions" in debug_once.call_args.args[0]
    attn.qk_nope_head_dim = 128
    custom_op_module.IS_K3_CONTEXT_FMHA_AVAILABLE = False
    assert not fmha.supports_k3_fp8_split_k_context()
    assert "lacks split-K epilogue merge support" in debug_once.call_args.args[0]
    warning_once.assert_not_called()
    custom_op_module.K3_CONTEXT_FMHA_IMPORT_ERROR = ImportError("missing CuTe runtime")
    assert not fmha.supports_k3_fp8_split_k_context()
    assert "import failed" in debug_once.call_args.args[0]
    assert "missing CuTe runtime" in debug_once.call_args.args[0]
    # An unusable kernel on SM107 (e.g. a public CuTe DSL) is reported as a warning.
    assert "missing CuTe runtime" in warning_once.call_args.args[0]


def test_k3_fp8_split_k_context_caches_fmha_instance(monkeypatch) -> None:
    from unittest.mock import Mock

    import tensorrt_llm._torch.attention_backend.trtllm as trtllm_module

    class FakeAttention:
        uses_fp4_mla_attention = True
        is_mla_enable = True
        kv_lora_rank = 512
        v_head_dim = 128
        head_dim = 128
        create_fmha_libs = trtllm_module.TrtllmAttention.create_fmha_libs
        supports_k3_fp8_split_k_context = (
            trtllm_module.TrtllmAttention.supports_k3_fp8_split_k_context
        )
        forward_k3_fp8_split_k_context_partition = (
            trtllm_module.TrtllmAttention.forward_k3_fp8_split_k_context_partition
        )

    class FakeFallback:
        @classmethod
        def is_available(cls, attn):
            return True

        def __init__(self, attn):
            pass

    monkeypatch.setattr(
        trtllm_module, "get_enabled_fmha_lib_classes", lambda: [Fp4MlaFmha, FakeFallback]
    )
    supports = Mock(return_value=True)
    forward = Mock()
    monkeypatch.setattr(Fp4MlaFmha, "supports_k3_fp8_split_k_context", supports)
    monkeypatch.setattr(Fp4MlaFmha, "forward_k3_fp8_split_k_context_partition", forward)
    attn = FakeAttention()
    attn.fmha_libs = []
    assert attn.supports_k3_fp8_split_k_context()
    first = attn._fp4_mla_fmha
    assert first is attn.fmha_libs[0]

    class NoIteration(list):
        def __iter__(self):
            raise AssertionError("FMHA selection must reuse the cached instance")

    attn.fmha_libs = NoIteration(attn.fmha_libs)
    args = (None,) * 9
    kwargs = dict(max_q_len=256, max_kv_len=256, is_causal=False)
    assert attn.supports_k3_fp8_split_k_context()
    attn.forward_k3_fp8_split_k_context_partition(*args, **kwargs)
    forward.assert_called_once_with(*args, **kwargs)

    attn.create_fmha_libs()
    assert attn._fp4_mla_fmha is not first
    assert attn._fp4_mla_fmha is attn.fmha_libs[0]
    from tensorrt_llm._torch.attention_backend.fp4_mla.fp4_mla_context import (
        _build_fp8_mla_context_attn,
    )

    fp8_attn = _build_fp8_mla_context_attn(attn)
    assert fp8_attn._fp4_mla_fmha is None
    assert fp8_attn.fmha_libs == []
    assert attn._fp4_mla_fmha is not None
    attn.uses_fp4_mla_attention = False
    attn.create_fmha_libs()
    assert attn._fp4_mla_fmha is None
    assert not attn.supports_k3_fp8_split_k_context()
    with pytest.raises(RuntimeError, match="unavailable for chunked prefill"):
        attn.forward_k3_fp8_split_k_context_partition(*args, **kwargs)


def test_fp8_mla_context_partition_metadata_separates_q_and_kv_lengths() -> None:
    prompt_lens_cuda = torch.tensor([32, 48, 1], dtype=torch.int32)
    prompt_lens_cpu = prompt_lens_cuda.clone()
    kv_lens_cuda = torch.tensor([128, 64], dtype=torch.int32)
    kv_lens_cpu = kv_lens_cuda.clone()
    meta = SimpleNamespace(
        num_contexts=2,
        num_ctx_tokens=80,
        prompt_lens_cuda_runtime=prompt_lens_cuda,
        prompt_lens_cpu_runtime=prompt_lens_cpu,
        host_request_types_runtime=torch.tensor([0, 0, 1], dtype=torch.int32),
        positions=torch.arange(80, dtype=torch.int32),
        _fp4_mla_fp8_context_state=object(),
    )
    scratch = SimpleNamespace(
        cache_manager_view=object(),
        block_offsets=object(),
        block_ids_per_seq=object(),
        host_total_kv_lens=torch.tensor([192, 0], dtype=torch.int64),
    )

    fp8_meta = _build_fp8_mla_context_metadata(
        meta,
        scratch,
        kv_lens_cuda=kv_lens_cuda,
        kv_lens_cpu=kv_lens_cpu,
    )

    assert fp8_meta.prompt_lens_cuda_runtime.data_ptr() == prompt_lens_cuda.data_ptr()
    assert fp8_meta.prompt_lens_cpu_runtime.data_ptr() == prompt_lens_cpu.data_ptr()
    assert fp8_meta.kv_lens_cuda_runtime is kv_lens_cuda
    assert fp8_meta.kv_lens_runtime is kv_lens_cpu
    assert fp8_meta.host_total_kv_lens is scratch.host_total_kv_lens
    assert fp8_meta.helix_position_offsets.shape == (80,)
    assert fp8_meta._fp4_mla_fp8_context_state is None


def _reset_triton_allocator() -> None:
    import triton

    def _null_allocator(size: int, alignment: int, stream):
        raise RuntimeError("Triton kernel requested scratch memory without an allocator.")

    triton.set_allocator(_null_allocator)


def test_fp4_mla_generation_hp_page_ids_skip_mixed_batch_context_rows() -> None:
    metadata = SimpleNamespace(
        num_contexts=2,
        num_seqs=5,
        _fp4_mla_device_page_table=True,
        fp4_mla_page_table_stride=3,
        _fp4_mla_hp_page_indices=torch.arange(15, dtype=torch.int32),
    )

    page_ids = fp4_mla_backend._fp4_mla_generation_hp_page_ids(metadata, 3)

    torch.testing.assert_close(page_ids, torch.arange(6, 15, dtype=torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA tensors")
def test_fp4_mla_cuda_graph_generation_lengths_records_capture_once(monkeypatch) -> None:
    corrected_kv_lens = torch.empty(2, dtype=torch.int32, device="cuda")
    generation_lens = torch.empty(2, dtype=torch.int32, device="cuda")
    metadata = SimpleNamespace(
        num_contexts=1,
        num_seqs=3,
        kv_lens_cuda_runtime=torch.tensor([9, 17, 25], dtype=torch.int32, device="cuda"),
        prompt_lens_cuda_runtime=torch.tensor([9, 1, 1], dtype=torch.int32, device="cuda"),
        fp4_mla_generation_kv_lens=corrected_kv_lens,
        fp4_mla_generation_append_lens=generation_lens,
        fp4_mla_generation_lengths_num_tokens=4,
        fp4_mla_generation_lengths_num_seqs=2,
        fp4_mla_generation_lengths_num_contexts=1,
        _fp4_mla_generation_lengths_capture_recorded=False,
        is_cuda_graph=True,
    )
    populate_calls = []

    def populate_generation_lengths(*args, **kwargs) -> None:
        populate_calls.append((args, kwargs))
        args[2].fill_(19)
        args[3].fill_(2)

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        fp4_mla_backend,
        "populate_fp4_mla_generation_lengths",
        populate_generation_lengths,
    )

    first = fp4_mla_backend._fp4_mla_uniform_generation_lengths(metadata, 4, 2)
    second = fp4_mla_backend._fp4_mla_uniform_generation_lengths(metadata, 4, 2)

    assert len(populate_calls) == 1
    args, kwargs = populate_calls[0]
    assert args[0].data_ptr() == metadata.kv_lens_cuda_runtime[1:].data_ptr()
    assert args[1].data_ptr() == metadata.prompt_lens_cuda_runtime[1:].data_ptr()
    assert args[2].data_ptr() == corrected_kv_lens.data_ptr()
    assert args[3].data_ptr() == generation_lens.data_ptr()
    assert kwargs == {"num_gen_tokens": 4, "num_gen": 2}
    assert metadata._fp4_mla_generation_lengths_capture_recorded
    assert first[0].data_ptr() == second[0].data_ptr() == corrected_kv_lens.data_ptr()
    assert first[1].data_ptr() == second[1].data_ptr() == generation_lens.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA tensors")
@pytest.mark.parametrize(
    "tokens_per_block,cp_size,cp_rank,tokens_per_seq",
    [(64, 16, 0, 8), (64, 16, 15, 8), (32, 4, 1, 1), (128, 2, 1, 6)],
)
def test_recompute_helix_spec_buffers_matches_elementwise_reference(
    tokens_per_block: int, cp_size: int, cp_rank: int, tokens_per_seq: int
) -> None:
    generator = torch.Generator().manual_seed(7)
    num_seqs = 5
    num_ctx_tokens = 3
    num_tokens = num_seqs * tokens_per_seq
    base = torch.randint(0, 300_000, (num_seqs, 1), generator=generator)
    # One verify group straddles a ledger boundary, one a page boundary.
    base[0, 0] = tokens_per_block * cp_size - max(1, tokens_per_seq // 2)
    base[1, 0] = 7 * tokens_per_block - max(1, tokens_per_seq // 2)
    positions = (base + torch.arange(tokens_per_seq)).reshape(-1).to(torch.int32).cuda()
    metadata = object.__new__(trtllm_backend.TrtllmAttentionMetadata)
    metadata.kv_cache_manager = SimpleNamespace(tokens_per_block=tokens_per_block)
    metadata.mapping = SimpleNamespace(cp_size=cp_size, cp_rank=cp_rank)
    metadata.num_contexts = 1
    untouched = 77
    metadata.helix_position_offsets = torch.cat(
        [
            torch.zeros(num_ctx_tokens, dtype=torch.int32, device="cuda"),
            positions,
            torch.full((2,), untouched, dtype=torch.int32, device="cuda"),
        ]
    )
    metadata.helix_local_slots = torch.full_like(metadata.helix_position_offsets, untouched)
    metadata.helix_kv_bounds = torch.full_like(metadata.helix_position_offsets, untouched)
    metadata.kv_lens_cuda = torch.full((num_seqs + 3,), untouched, dtype=torch.int32, device="cuda")

    metadata.recompute_helix_spec_buffers(num_ctx_tokens, num_tokens, tokens_per_seq)
    torch.cuda.synchronize()

    # Reference: the per-op formulation (helix_local_len_vec mirrors the cache manager).
    owner = torch.div(positions, tokens_per_block, rounding_mode="floor") % cp_size
    expected_slots = torch.where(
        owner == cp_rank, metadata.helix_local_len_vec(positions), torch.full_like(positions, -1)
    )
    expected_bounds = metadata.helix_local_len_vec(positions + 1)
    gen = slice(num_ctx_tokens, num_ctx_tokens + num_tokens)
    torch.testing.assert_close(metadata.helix_local_slots[gen], expected_slots)
    torch.testing.assert_close(metadata.helix_kv_bounds[gen], expected_bounds)
    torch.testing.assert_close(
        metadata.kv_lens_cuda[1 : 1 + num_seqs],
        expected_bounds.view(num_seqs, tokens_per_seq)[:, -1],
    )
    assert (metadata.helix_local_slots[:num_ctx_tokens] == untouched).all()
    assert (metadata.helix_kv_bounds[num_ctx_tokens + num_tokens :] == untouched).all()
    assert metadata.kv_lens_cuda[0].item() == untouched
    assert (metadata.kv_lens_cuda[1 + num_seqs :] == untouched).all()
    # Any group of the sequence that this rank does not own writes no slot.
    assert ((metadata.helix_local_slots[gen] == -1) == (owner != cp_rank)).all()


@pytest.mark.parametrize("layer_offset", [0, 1, 3])
def test_fp4_mla_v2_encoded_page_capacity_is_layer_invariant(layer_offset: int) -> None:
    manager = object.__new__(Fp4MlaKVCacheManagerV2)
    manager.impl = SimpleNamespace(
        get_page_index_converter=lambda *_: PageIndexConverter(
            scale=4,
            expansion=1,
            layer_offset=layer_offset,
        ),
        get_page_index_upper_bound=lambda *_: 20 - layer_offset,
    )

    assert manager._role_encoded_page_capacity(0, Role.KEY) == 17


def test_fp4_mla_v2_cache_size_accounts_for_hp_intercept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "triton")
    monkeypatch.delenv(KIMI_K3_FP8_KV_LAYERS_ENV, raising=False)
    model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        )
    )

    slope, intercept = Fp4MlaKVCacheManagerV2.get_cache_size_per_token(
        model_config,
        Mapping(world_size=2, tp_size=1, pp_size=2, rank=0),
        num_layers=2,
        tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
        max_batch_size=3,
        spec_config=SimpleNamespace(tokens_per_gen_step=4),
    )

    assert slope == 2 * (320 + 40 + 32)
    assert intercept == 3 * 2 * (HP_BLOCK_SIZE + 3) * 576 * 2 * 2


def test_fp4_mla_v2_runtime_sizing_accounts_for_pipeline_slots() -> None:
    manager = object.__new__(Fp4MlaKVCacheManagerV2)
    manager.max_batch_size = 3
    manager.mapping = SimpleNamespace(pp_size=2)
    manager.max_num_tokens = 100
    manager.tokens_per_block = FP4_MLA_TOKENS_PER_BLOCK
    manager.enable_swa_scratch_reuse = False
    manager._get_runtime_cache_size_layer_components = lambda: ([10, 8], [None, 19])

    quota = manager._get_quota_from_max_tokens(manager.max_num_tokens)

    hp_page_bytes = FP4_MLA_TOKENS_PER_BLOCK * 8
    expected_quota = manager.max_num_tokens * (10 + 8) + 6 * hp_page_bytes
    assert quota == expected_quota
    assert manager._get_max_tokens_from_quota(quota) == manager.max_num_tokens


@pytest.mark.parametrize(
    ("v_layout", "expected_v_head_dim"),
    [
        (Fp4MlaVLayout.DUAL_COPY, 512),
        (Fp4MlaVLayout.FUSED_V, None),
        (Fp4MlaVLayout.PACKED_SINGLE_COPY, None),
    ],
    ids=["mufu16-packed-v", "fused-v-canonical-cache", "single-copy-canonical-cache"],
)
def test_fp4_mla_manager_v2_packed_v_matches_cutedsl_variant(
    monkeypatch,
    v_layout: Fp4MlaVLayout,
    expected_v_head_dim: int | None,
) -> None:
    class _ConstructionCaptured(Exception):
        pass

    captured = {}

    def capture_base_init(manager, *args, **kwargs) -> None:
        captured["manager"] = manager
        raise _ConstructionCaptured

    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "cutedsl")
    _set_fp4_mla_v_layout_env(monkeypatch, v_layout)
    monkeypatch.setattr(KVCacheManagerV2, "__init__", capture_base_init)

    with pytest.raises(_ConstructionCaptured):
        Fp4MlaKVCacheManagerV2(
            LlmKvCacheConfig(dtype="nvfp4", enable_partial_reuse=False),
            _CacheType.SELFKONLY,
            head_dim=576,
            tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
            dtype=_DataType.NVFP4,
            pretrained_config=SimpleNamespace(kv_lora_rank=512),
        )

    assert captured["manager"].fp4_mla_v_layout is v_layout
    assert captured["manager"].mla_v_head_dim == expected_v_head_dim


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA KV cache pools")
@pytest.mark.parametrize(
    ("v_layout", "bytes_per_token"),
    [(Fp4MlaVLayout.DUAL_COPY, 648), (Fp4MlaVLayout.PACKED_SINGLE_COPY, 392)],
    ids=["dual-copy", "single-copy"],
)
def test_fp4_mla_manager_v2_pools_match_the_layout_estimate(
    monkeypatch, v_layout: Fp4MlaVLayout, bytes_per_token: int
) -> None:
    """K 320 + K scales 40 + V scales 32 B per token, plus the 256 B V^T sidecar only in
    dual copy: the allocated pages, the manager's count and the static estimator agree."""
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "cutedsl")
    monkeypatch.delenv(KIMI_K3_FP8_KV_LAYERS_ENV, raising=False)
    _set_fp4_mla_v_layout_env(monkeypatch, v_layout)
    manager = _create_fp4_mla_v2_manager(
        max_tokens=4 * FP4_MLA_TOKENS_PER_BLOCK,
        max_seq_len=4 * FP4_MLA_TOKENS_PER_BLOCK,
        max_batch_size=1,
    )
    try:
        assert manager.fp4_mla_v_layout is v_layout
        assert manager.get_layer_bytes_per_token(0, Role.ALL) == bytes_per_token
        has_v_sidecar = v_layout is Fp4MlaVLayout.DUAL_COPY
        assert (manager.get_mla_v_packed_pool(0) is not None) is has_v_sidecar
        assert (manager.get_mla_v_packed_pool_base() is not None) is has_v_sidecar
        local_layer = manager._fp4_mla_compact_to_local[0]
        extra_roles = {
            buffer.role
            for buffer in manager._extra_buffers_per_layer(
                tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK
            )[local_layer]
        }
        assert (Role.MLA_V_PACKED in extra_roles) is has_v_sidecar
        layer_id = manager._cache_manager_layer_ids[0]
        roles = [Role.KEY, Role.KEY_BLOCK_SCALE, Role.MLA_V_SCALE]
        if has_v_sidecar:
            roles.append(Role.MLA_V_PACKED)
        page_bytes = sum(int(manager.impl.get_page_stride(layer_id, role)) for role in roles)
        assert page_bytes == bytes_per_token * FP4_MLA_TOKENS_PER_BLOCK
        # Single copy multiplies the K latent codes (always finite in FP4) of a partial
        # page's unwritten tokens by these scales under P = 0: they start at zero.
        assert int(manager.get_mla_v_scale_pool_base().count_nonzero()) == 0
        slope, _ = Fp4MlaKVCacheManagerV2.get_cache_size_per_token(
            SimpleNamespace(
                pretrained_config=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64)
            ),
            Mapping(world_size=1, tp_size=1, rank=0),
            num_layers=1,
            tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
            max_batch_size=1,
        )
        assert slope == bytes_per_token
    finally:
        manager.shutdown()


@pytest.mark.parametrize("v_layout", list(Fp4MlaVLayout), ids=lambda v_layout: v_layout.value)
def test_fp4_mla_disagg_import_rebuilds_variant_sidecars(
    monkeypatch,
    v_layout: Fp4MlaVLayout,
) -> None:
    has_v_sidecar = v_layout is Fp4MlaVLayout.DUAL_COPY
    page_ids = torch.tensor([17, 29], dtype=torch.int32)
    page_valid_tokens = torch.tensor([FP4_MLA_TOKENS_PER_BLOCK, 1], dtype=torch.int32)
    v_scale_pool = torch.empty((2, 32, 1), dtype=torch.uint8)
    kv_caches = {
        7: torch.empty(1, dtype=torch.uint8),
        11: torch.empty(1, dtype=torch.uint8),
    }
    sf_caches = {
        7: torch.empty(1, dtype=torch.uint8),
        11: torch.empty(1, dtype=torch.uint8),
    }
    v_packed_pools = {
        0: torch.empty(1, dtype=torch.uint8),
        1: torch.empty(1, dtype=torch.uint8),
    }
    staged = []
    scale_rebuilds = []
    packed_rebuilds = []

    def stage_page_metadata(block_ids, *, prompt_len, page_size, device):
        staged.append((block_ids, prompt_len, page_size, device))
        return page_ids, page_valid_tokens

    def rebuild_v_scales(
        sf_cache,
        observed_v_scale_pool,
        observed_page_ids,
        observed_page_valid_tokens,
        **kwargs,
    ) -> None:
        scale_rebuilds.append(
            (
                sf_cache,
                observed_v_scale_pool,
                observed_page_ids,
                observed_page_valid_tokens,
                kwargs,
            )
        )

    def rebuild_packed_v(
        v_packed,
        kv_cache,
        observed_page_ids,
        **kwargs,
    ) -> None:
        packed_rebuilds.append((v_packed, kv_cache, observed_page_ids, kwargs))

    def get_v_packed_pool(local_layer: int) -> torch.Tensor:
        if not has_v_sidecar:
            raise AssertionError(f"{v_layout} import must not request a packed-V pool")
        return v_packed_pools[local_layer]

    manager = SimpleNamespace(
        dtype=_DataType.NVFP4,
        kv_factor=1,
        fp4_mla_v_layout=v_layout,
        mla_v_scale_head_dim=512,
        mla_v_head_dim=512 if has_v_sidecar else None,
        tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
        pp_layers=[7, 11],
        num_local_layers=2,
        get_fp4_mla_page_table_spec=lambda *_: SimpleNamespace(),
        get_batch_cache_indices=lambda request_ids, layer_idx=None: [[17, 29, 31]],
        get_mla_v_scale_pool=lambda: v_scale_pool,
        get_fp4_mla_cache_buffers=lambda layer_idx: (
            kv_caches[layer_idx],
            sf_caches[layer_idx],
        ),
        get_mla_v_packed_pool=get_v_packed_pool,
    )
    monkeypatch.setattr(
        fp4_mla_backend,
        "_stage_fp4_mla_import_page_metadata",
        stage_page_metadata,
    )
    monkeypatch.setattr(
        fp4_mla_backend,
        "_rebuild_fp4_mla_v_scales_from_k_scales",
        rebuild_v_scales,
    )
    monkeypatch.setattr(
        fp4_mla_backend,
        "_repack_cutedsl_v_packed_cache",
        rebuild_packed_v,
    )

    assert fp4_mla_backend.rebuild_fp4_mla_disagg_imported_cache(
        manager,
        request_id=41,
        prompt_len=FP4_MLA_TOKENS_PER_BLOCK + 1,
    )

    assert staged == [
        (
            [17, 29],
            FP4_MLA_TOKENS_PER_BLOCK + 1,
            FP4_MLA_TOKENS_PER_BLOCK,
            v_scale_pool.device,
        )
    ]
    assert len(scale_rebuilds) == 2
    for local_layer, layer_idx in enumerate(manager.pp_layers):
        sf_cache, observed_pool, observed_ids, observed_valid, kwargs = scale_rebuilds[local_layer]
        assert sf_cache is sf_caches[layer_idx]
        assert observed_pool is v_scale_pool
        assert observed_ids is page_ids
        assert observed_valid is page_valid_tokens
        assert kwargs == {
            "local_layer": local_layer,
            "v_head_dim": 512,
            "page_size": FP4_MLA_TOKENS_PER_BLOCK,
        }
    assert len(packed_rebuilds) == (2 if has_v_sidecar else 0)
    for local_layer, layer_idx in enumerate(manager.pp_layers[: len(packed_rebuilds)]):
        v_packed, kv_cache, observed_ids, kwargs = packed_rebuilds[local_layer]
        assert v_packed is v_packed_pools[local_layer]
        assert kv_cache is kv_caches[layer_idx]
        assert observed_ids is page_ids
        assert kwargs == {
            "v_head_dim": 512,
            "page_size": FP4_MLA_TOKENS_PER_BLOCK,
            "block_v": fp4_mla_backend.FP4_MLA_SCALE_ROW_GROUP,
        }


def test_fp4_mla_disagg_import_skips_hybrid_linear_attention_layers(monkeypatch) -> None:
    page_ids = torch.tensor([17, 29], dtype=torch.int32)
    page_valid_tokens = torch.tensor([FP4_MLA_TOKENS_PER_BLOCK, 1], dtype=torch.int32)
    v_scale_pool = torch.empty((2, 32, 1), dtype=torch.uint8)
    requested_page_layers = []
    rebuilt_cache_layers = []
    rebuilt_compact_layers = []

    def get_batch_cache_indices(request_ids, *, layer_idx):
        requested_page_layers.append((request_ids, layer_idx))
        return [[17, 29]]

    def get_fp4_mla_cache_buffers(layer_idx):
        rebuilt_cache_layers.append(layer_idx)
        return torch.empty(1, dtype=torch.uint8), torch.empty(1, dtype=torch.uint8)

    def rebuild_v_scales(_sf_cache, _v_scale_pool, _page_ids, _page_valid_tokens, **kwargs):
        rebuilt_compact_layers.append(kwargs["local_layer"])

    manager = SimpleNamespace(
        dtype=_DataType.NVFP4,
        kv_factor=1,
        mla_v_scale_head_dim=512,
        tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
        pp_layers=[3, 7, 11],
        num_local_layers=3,
        _fp4_mla_compact_to_local=[1, 2],
        get_fp4_mla_page_table_spec=lambda *_: SimpleNamespace(),
        get_batch_cache_indices=get_batch_cache_indices,
        get_mla_v_scale_pool=lambda: v_scale_pool,
        get_fp4_mla_cache_buffers=get_fp4_mla_cache_buffers,
    )
    monkeypatch.setattr(
        fp4_mla_backend,
        "_stage_fp4_mla_import_page_metadata",
        lambda *args, **kwargs: (page_ids, page_valid_tokens),
    )
    monkeypatch.setattr(
        fp4_mla_backend,
        "_rebuild_fp4_mla_v_scales_from_k_scales",
        rebuild_v_scales,
    )

    assert fp4_mla_backend.rebuild_fp4_mla_disagg_imported_cache(
        manager,
        request_id=41,
        prompt_len=FP4_MLA_TOKENS_PER_BLOCK + 1,
    )

    assert requested_page_layers == [([41], 7)]
    assert rebuilt_cache_layers == [7, 11]
    assert rebuilt_compact_layers == [0, 1]


def test_fp4_mla_helix_disagg_import_skips_empty_rank() -> None:
    manager = SimpleNamespace(
        dtype=_DataType.NVFP4,
        kv_factor=1,
        mla_v_scale_head_dim=512,
        get_fp4_mla_page_table_spec=lambda *_: SimpleNamespace(),
        get_batch_cache_indices=lambda *_args, **_kwargs: pytest.fail(
            "an empty Helix rank must not query cache pages"
        ),
    )

    assert fp4_mla_backend.rebuild_fp4_mla_disagg_imported_cache(
        manager,
        request_id=41,
        prompt_len=0,
    )


@pytest.mark.parametrize("reuse_ends_at_snapshots", [False, True])
def test_fp4_mla_partial_reuse_needs_snapshot_endpoints(
    monkeypatch, reuse_ends_at_snapshots: bool
) -> None:
    """Partial-block matches stay on only where reuse ends at committed snapshots,
    whose blocks keep the HP ring of their open tile (hybrid recurrent managers)."""
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "triton")
    monkeypatch.delenv(KIMI_K3_FP8_KV_LAYERS_ENV, raising=False)
    manager = object.__new__(Fp4MlaKVCacheManagerV2)
    kv_cache_config = Fp4MlaKVCacheManagerV2.configure_manager(
        manager,
        LlmKvCacheConfig(dtype="nvfp4", enable_block_reuse=True, enable_partial_reuse=True),
        _CacheType.SELFKONLY,
        dtype=_DataType.NVFP4,
        tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
        head_dim=576,
        pretrained_config=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64),
        reuse_ends_at_snapshots=reuse_ends_at_snapshots,
    )
    assert kv_cache_config.enable_partial_reuse == reuse_ends_at_snapshots


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp4_mla_manager_v2_registers_native_cache_and_hp_roles(monkeypatch) -> None:
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "triton")
    manager = Fp4MlaKVCacheManagerV2(
        LlmKvCacheConfig(
            max_tokens=512,
            dtype="nvfp4",
            enable_block_reuse=True,
            enable_partial_reuse=True,
            host_cache_size=0,
        ),
        _CacheType.SELFKONLY,
        num_layers=2,
        num_kv_heads=1,
        head_dim=576,
        tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
        max_seq_len=512,
        max_batch_size=2,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=_DataType.NVFP4,
        max_num_tokens=512,
        pretrained_config=SimpleNamespace(kv_lora_rank=512),
    )
    try:
        config = manager.kv_cache_manager_py_config
        assert not config.enable_partial_reuse
        assert len(config.layers) == 4
        assert [buffer.role for buffer in config.layers[0].buffers] == [
            Role.KEY,
            Role.KEY_BLOCK_SCALE,
            Role.MLA_V_SCALE,
        ]
        assert [buffer.role for buffer in config.layers[2].buffers] == [Role.MLA_HP_TAIL]
        assert config.layers[2].sliding_window_size == HP_BLOCK_SIZE

        page_spec = manager.get_fp4_mla_page_table_spec(0)
        assert page_spec.cache_pool_id != page_spec.hp_pool_id
        assert page_spec.cache_page_index_scale == 1
        assert page_spec.hp_page_index_scale == 1

        kv_cache, sf_cache = manager.get_fp4_mla_cache_buffers(0)
        v_scale_pool = manager.get_mla_v_scale_pool()
        hp_pool = manager.get_fp4_mla_hp_pool()
        assert kv_cache.shape[0] == sf_cache.shape[0] == v_scale_pool.shape[1]
        assert v_scale_pool.dtype == torch.float8_e4m3fn
        assert manager.get_mla_v_scale_pool_base().dtype == torch.uint8
        assert hp_pool.shape[1:] == (2, 1, HP_BLOCK_SIZE * 576)
        assert manager.get_disagg_transfer_roles() == frozenset(
            (Role.KEY, Role.KEY_BLOCK_SCALE, Role.MLA_HP_TAIL)
        )
    finally:
        manager.shutdown()


_FP4_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Pages dequantized per batch by the reference cache materialization (bounds peak memory).
_REFERENCE_DEQUANT_PAGES_PER_BATCH = 256


def _decode_e2m1_nibbles(nibbles: torch.Tensor, magnitudes: torch.Tensor) -> torch.Tensor:
    magnitude = magnitudes[(nibbles & 0x07).long()]
    sign = torch.where(
        (nibbles & 0x08) != 0,
        -torch.ones_like(magnitude),
        torch.ones_like(magnitude),
    )
    return magnitude * sign


def _dequant_fp4_swizzled_pages(
    fp4_pages: torch.Tensor,
    sf_pages: torch.Tensor,
    *,
    logical_dim: int,
    sf_per_token: int,
    global_scale: float,
) -> torch.Tensor:
    """Dequantize NVFP4 rows, one swizzled scale-factor block per leading index.

    fp4_pages is [pages, rows, packed bytes] and sf_pages is [pages, scale bytes]; the scale of
    (row, block) sits at _swizzled_sf_offset(row, block) inside its page. Element 2i of a block is
    the low nibble of byte i and element 2i + 1 the high nibble. Each value is the E2M1 magnitude
    times its sign, times the FP32 E4M3 scale, divided by global_scale: the same FP32 operations
    in the same order as a scalar loop, so the result is bit-identical to one. Columns past
    sf_per_token * FP4_BLOCK_SIZE are left uninitialized.
    """
    device = fp4_pages.device
    num_pages, num_rows = fp4_pages.shape[0], fp4_pages.shape[1]
    covered_dim = sf_per_token * FP4_BLOCK_SIZE
    assert covered_dim <= logical_dim
    out = torch.empty((num_pages, num_rows, logical_dim), dtype=torch.float32, device=device)
    if num_pages == 0 or num_rows == 0 or sf_per_token == 0:
        return out
    magnitudes = torch.tensor(_FP4_E2M1_MAGNITUDES, dtype=torch.float32, device=device)
    packed = fp4_pages.view(torch.uint8)[..., : covered_dim // 2]
    values = torch.stack(
        (
            _decode_e2m1_nibbles(packed & 0x0F, magnitudes),
            _decode_e2m1_nibbles((packed >> 4) & 0x0F, magnitudes),
        ),
        dim=-1,
    ).reshape(num_pages, num_rows, sf_per_token, FP4_BLOCK_SIZE)
    sf_offsets = _swizzled_sf_offset(
        torch.arange(num_rows, device=device).unsqueeze(1),
        torch.arange(sf_per_token, device=device).unsqueeze(0),
        sf_per_token,
    )
    sf_flat = sf_pages.view(torch.float8_e4m3fn).reshape(num_pages, -1)
    scales = sf_flat[:, sf_offsets.reshape(-1)].float()
    scales = scales.reshape(num_pages, num_rows, sf_per_token, 1)
    out[..., :covered_dim] = (values * scales / global_scale).reshape(
        num_pages, num_rows, covered_dim
    )
    return out


def _dequant_fp4_swizzled(
    fp4_tensor: torch.Tensor,
    sf_tensor: torch.Tensor,
    *,
    logical_dim: int,
    sf_per_token: int,
    global_scale: float,
) -> torch.Tensor:
    return _dequant_fp4_swizzled_pages(
        fp4_tensor.unsqueeze(0),
        sf_tensor.reshape(1, -1),
        logical_dim=logical_dim,
        sf_per_token=sf_per_token,
        global_scale=global_scale,
    )[0]


def _duplicate_tail_groups(tensor: torch.Tensor, residual_dim: int) -> torch.Tensor:
    prefix = tensor[..., :-residual_dim]
    tail = tensor[..., -residual_dim:].reshape(
        *tensor.shape[:-1], residual_dim // FP4_BLOCK_SIZE, FP4_BLOCK_SIZE
    )
    duplicated_tail = tail.repeat_interleave(2, dim=-2).reshape(
        *tensor.shape[:-1],
        residual_dim * 2,
    )
    return torch.cat((prefix, duplicated_tail), dim=-1)


def _expand_qk_residual_terms(
    q: torch.Tensor,
    k: torch.Tensor,
    k_residual: torch.Tensor,
    residual_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build [Q, Q_r, Q] x [K, K, K_r] after the shared prefix."""
    prefix_dim = k.shape[-1] - residual_dim
    residual_groups = residual_dim // FP4_BLOCK_SIZE
    q_tail = q[..., prefix_dim:].reshape(*q.shape[:-1], residual_groups, 2, FP4_BLOCK_SIZE)
    q_main = q_tail[..., 0, :].reshape(*q.shape[:-1], residual_dim)
    q_residual = q_tail[..., 1, :].reshape(*q.shape[:-1], residual_dim)
    k_main = k[..., prefix_dim:]
    return (
        torch.cat((q[..., :prefix_dim], q_main, q_residual, q_main), dim=-1),
        torch.cat((k[..., :prefix_dim], k_main, k_main, k_residual), dim=-1),
    )


def _create_fp4_mla_v2_manager(
    *,
    max_tokens: int,
    max_seq_len: int,
    max_batch_size: int,
    spec_config=None,
    enable_block_reuse: bool = False,
) -> Fp4MlaKVCacheManagerV2:
    return Fp4MlaKVCacheManagerV2(
        LlmKvCacheConfig(
            max_tokens=max_tokens,
            dtype="nvfp4",
            enable_block_reuse=enable_block_reuse,
            host_cache_size=0,
        ),
        _CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=576,
        tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=_DataType.NVFP4,
        spec_config=spec_config,
        max_num_tokens=max_tokens,
        pretrained_config=SimpleNamespace(kv_lora_rank=512),
    )


def _build_multi_seq_metadata(kv_cache_manager, *, seq_lens, page_size):
    device = torch.device("cuda")
    num_seqs = len(seq_lens)
    request_ids = list(range(num_seqs))
    block_ids_per_seq = kv_cache_manager.get_batch_cache_indices(request_ids, layer_idx=0)
    page_spec = kv_cache_manager.get_fp4_mla_page_table_spec(0)
    hp_block_ids_per_seq = kv_cache_manager._get_batch_cache_indices_by_pool_id(
        request_ids,
        pool_id=page_spec.hp_pool_id,
        is_kv_aggregate=False,
    )
    num_blocks = [(seq_len + page_size - 1) // page_size for seq_len in seq_lens]

    max_blocks_per_seq = max(num_blocks)
    page_rows = []
    hp_page_rows = []
    for seq_idx, seq_blocks in enumerate(block_ids_per_seq):
        active_blocks = seq_blocks[: num_blocks[seq_idx]]
        page_rows.extend(active_blocks + [0] * (max_blocks_per_seq - len(active_blocks)))
        active_hp_blocks = hp_block_ids_per_seq[seq_idx][: num_blocks[seq_idx]]
        hp_page_rows.extend(active_hp_blocks + [0] * (max_blocks_per_seq - len(active_hp_blocks)))
    paged_kv_indices = torch.tensor(
        page_rows,
        dtype=torch.int32,
        device=device,
    )
    paged_kv_indptr = (
        torch.arange(num_seqs + 1, dtype=torch.int32, device=device) * max_blocks_per_seq
    )
    hp_page_indices = torch.tensor(hp_page_rows, dtype=torch.int32, device=device)
    batch_indices = torch.cat(
        [
            torch.full(
                (seq_len,),
                seq_idx,
                dtype=torch.int32,
                device=device,
            )
            for seq_idx, seq_len in enumerate(seq_lens)
        ]
    )
    positions = torch.cat(
        [torch.arange(seq_len, dtype=torch.int32, device=device) for seq_len in seq_lens]
    )

    hp_pool = kv_cache_manager.get_fp4_mla_hp_pool()
    kv_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    prompt_lens_cuda = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    prompt_lens_cpu = torch.tensor(seq_lens, dtype=torch.int32)
    kv_global_scale = torch.tensor([FP4_MLA_KV_GLOBAL_SCALE], dtype=torch.float32, device=device)
    q_global_scale = torch.tensor([FP4_MLA_Q_GLOBAL_SCALE], dtype=torch.float32, device=device)

    return SimpleNamespace(
        kv_cache_manager=kv_cache_manager,
        batch_indices=batch_indices,
        positions=positions,
        paged_kv_indices=paged_kv_indices,
        _paged_kv_indices=paged_kv_indices,
        _fp4_mla_hp_page_indices=hp_page_indices,
        paged_kv_indptr=paged_kv_indptr,
        _paged_kv_indptr=paged_kv_indptr,
        paged_kv_indptr_decode=paged_kv_indptr.clone(),
        _fp4_mla_device_page_table=True,
        _fp4_mla_device_page_table_valid=True,
        fp4_mla_page_table_stride=max_blocks_per_seq,
        fp4_mla_context_repack_max_touched_pages=max_blocks_per_seq,
        page_size=page_size,
        num_context_blocks=num_seqs * max_blocks_per_seq,
        num_generation_blocks=0,
        num_contexts=num_seqs,
        num_seqs=num_seqs,
        num_blocks=None,
        high_precision_kv_pool=hp_pool,
        fp4_mla_v_scale_pool=kv_cache_manager.get_mla_v_scale_pool(),
        kv_lens_cuda_runtime=kv_lens,
        prompt_lens_cuda_runtime=prompt_lens_cuda,
        prompt_lens_cpu_runtime=prompt_lens_cpu,
        fp4_mla_generation_kv_lens=torch.empty(num_seqs, dtype=torch.int32, device=device),
        fp4_mla_generation_append_lens=torch.empty(num_seqs, dtype=torch.int32, device=device),
        fp4_mla_generation_lengths_num_tokens=-1,
        fp4_mla_generation_lengths_num_seqs=-1,
        fp4_mla_generation_lengths_num_contexts=-1,
        _fp4_mla_generation_lengths_capture_recorded=False,
        _fp4_mla_q_global_scale=q_global_scale,
        _fp4_mla_kv_global_scale=kv_global_scale,
        request_ids=request_ids,
        runtime_features=SimpleNamespace(has_speculative_draft_tokens=False),
        is_cuda_graph=False,
        is_warmup=False,
    )


def _materialize_reference_cache_storage(metadata, layer_idx: int, head_dim: int) -> torch.Tensor:
    kv_cache, sf_cache = metadata.kv_cache_manager.get_fp4_mla_cache_buffers(layer_idx)
    sf_cache = sf_cache.view(torch.float8_e4m3fn)
    storage_head_dim = kv_cache.shape[-1] * 2
    static_global_scale = float(_get_fp4_mla_global_scale(metadata, kv_cache.device).item())
    num_generation_sequences = metadata.num_seqs - metadata.num_contexts
    page_stride = metadata.fp4_mla_page_table_stride
    page_rows = metadata.paged_kv_indices.view(
        metadata.num_seqs,
        page_stride,
    )
    generation_page_rows = page_rows[metadata.num_contexts : metadata.num_seqs]
    indptr = metadata.paged_kv_indptr_decode[: num_generation_sequences + 1].cpu().tolist()
    expected_indptr = [seq_idx * page_stride for seq_idx in range(num_generation_sequences + 1)]
    assert indptr == expected_indptr
    assert generation_page_rows.shape == (num_generation_sequences, page_stride)

    active_blocks = (
        (
            metadata.kv_lens_cuda_runtime[metadata.num_contexts : metadata.num_seqs]
            + metadata.page_size
            - 1
        )
        // metadata.page_size
    ).tolist()
    for seq_idx, block_count in enumerate(active_blocks):
        assert block_count <= page_stride
        assert indptr[seq_idx + 1] - indptr[seq_idx] == page_stride
        assert block_count * metadata.page_size >= int(
            metadata.kv_lens_cuda_runtime[metadata.num_contexts + seq_idx].item()
        )

    if num_generation_sequences == 0:
        return torch.empty(
            (0, metadata.page_size, storage_head_dim),
            dtype=torch.float32,
            device=kv_cache.device,
        )
    page_ids = generation_page_rows.reshape(-1).to(device=kv_cache.device, dtype=torch.long)
    assert bool(((page_ids >= 0) & (page_ids < kv_cache.shape[0])).all().item())
    # Every distinct page once, batched (a per-row Python loop here took hours at wrap shapes).
    unique_pages, page_slots = torch.unique(page_ids, return_inverse=True)
    dequantized_pages = torch.empty(
        (unique_pages.numel(), kv_cache.shape[2], storage_head_dim),
        dtype=torch.float32,
        device=kv_cache.device,
    )
    for start in range(0, unique_pages.numel(), _REFERENCE_DEQUANT_PAGES_PER_BATCH):
        batch_pages = unique_pages[start : start + _REFERENCE_DEQUANT_PAGES_PER_BATCH]
        dequantized_pages[start : start + batch_pages.numel()] = _dequant_fp4_swizzled_pages(
            kv_cache.index_select(0, batch_pages)[:, 0, :, 0, :],
            sf_cache.index_select(0, batch_pages),
            logical_dim=storage_head_dim,
            sf_per_token=storage_head_dim // FP4_BLOCK_SIZE,
            global_scale=static_global_scale,
        )
    storage = dequantized_pages[page_slots]
    assert storage.shape[0] == indptr[-1] == num_generation_sequences * page_stride
    return storage


def _materialize_reference_cache_tokens(
    metadata,
    layer_idx: int,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    head_dim: int,
    *,
    include_k_residual: bool = False,
) -> torch.Tensor:
    kv_cache, sf_cache = metadata.kv_cache_manager.get_fp4_mla_cache_buffers(layer_idx)
    sf_cache = sf_cache.view(torch.float8_e4m3fn)
    storage_head_dim = kv_cache.shape[-1] * 2
    static_global_scale = float(_get_fp4_mla_global_scale(metadata, kv_cache.device).item())
    page_rows = metadata.paged_kv_indices.view(
        metadata.num_seqs,
        metadata.fp4_mla_page_table_stride,
    )
    dequantized_pages = {}
    tokens = []
    for batch_idx, position in zip(
        batch_indices.cpu().tolist(),
        positions.cpu().tolist(),
    ):
        page_idx, page_position = divmod(position, metadata.page_size)
        physical_page = int(page_rows[batch_idx, page_idx].item())
        if physical_page not in dequantized_pages:
            dequantized_pages[physical_page] = _dequant_fp4_swizzled(
                kv_cache[physical_page, 0, :, 0, :],
                sf_cache[physical_page],
                logical_dim=storage_head_dim,
                sf_per_token=storage_head_dim // FP4_BLOCK_SIZE,
                global_scale=static_global_scale,
            )
        token = dequantized_pages[physical_page][page_position, :head_dim].clone()
        if include_k_residual:
            residual_end = head_dim + FP4_MLA_K_RESIDUAL_DIM
            token[-FP4_MLA_K_RESIDUAL_DIM:] += dequantized_pages[physical_page][
                page_position,
                head_dim:residual_end,
            ]
        tokens.append(token)
    return torch.stack(tokens, dim=0)


@pytest.mark.skipif(
    _is_cutedsl_unavailable(),
    reason="requires Rubin SM107 and the CuTe DSL runtime",
)
@pytest.mark.parametrize("inactive_rank", [False, True], ids=["active", "inactive"])
def test_fp4_mla_helix_scatter_uses_global_rope_and_local_cache_position(
    monkeypatch,
    inactive_rank: bool,
) -> None:
    _reset_triton_allocator()
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "cutedsl")
    _set_fp4_mla_v_layout_env(monkeypatch, Fp4MlaVLayout.DUAL_COPY)
    local_kv_len = 6
    local_cache_position = local_kv_len - 1
    global_position = 70
    num_heads = 4
    (
        kv_cache_manager,
        metadata,
        _,
        kv_lora_rank,
        qk_rope_head_dim,
    ) = _build_fp4_mla_attention_decode_case(
        seq_lens=[local_kv_len],
        num_heads=num_heads,
        seed=31,
    )
    try:
        head_dim = kv_lora_rank + qk_rope_head_dim
        metadata.positions = torch.tensor([global_position], dtype=torch.int32, device="cuda")
        metadata.enable_helix = True
        metadata._helix_spec_tokens_valid = False
        metadata.helix_position_offsets = metadata.positions
        metadata.helix_is_inactive_rank = torch.tensor(
            [inactive_rank], dtype=torch.bool, device="cuda"
        )

        kv_cache, sf_cache = kv_cache_manager.get_fp4_mla_cache_buffers(0)
        kv_cache.zero_()
        sf_cache.zero_()
        metadata.high_precision_kv_pool.zero_()
        metadata.fp4_mla_v_scale_pool.zero_()
        v_packed_pool = kv_cache_manager.get_mla_v_packed_pool_base()
        assert v_packed_pool is not None
        v_packed_pool.zero_()

        generation_latent = (
            torch.randn(1, head_dim, dtype=torch.bfloat16, device="cuda") * 0.25
        ).clamp_(-1.0, 1.0)
        generation_latent[:, :FP4_BLOCK_SIZE] = 0.75
        generation_latent[:, 1:FP4_BLOCK_SIZE:2] = -0.75
        q_pe = (
            torch.randn(
                1,
                num_heads,
                qk_rope_head_dim,
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.25
        ).clamp_(-1.0, 1.0)
        original_q_pe = q_pe.clone()
        q_quant_input = torch.zeros(
            1,
            num_heads,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        q_rope_out = q_quant_input[..., kv_lora_rank:]
        rotary_cos_sin = torch.zeros(
            (FP4_MLA_TOKENS_PER_BLOCK, qk_rope_head_dim, 2),
            dtype=torch.float32,
            device="cuda",
        )
        rotary_cos_sin[..., 0] = 1.0
        rotary_cos_sin[global_position, :, 0] = 0.0
        rotary_cos_sin[global_position, :, 1] = 1.0

        assert scatter_fp4_mla_kv_cache(
            metadata,
            generation_latent,
            layer_idx=0,
            token_offset=0,
            phase="generation",
            local_layer=0,
            v_head_dim=kv_lora_rank,
            rotary_cos_sin=rotary_cos_sin,
            q_pe=q_pe,
            q_rope_out=q_rope_out,
            q_quant_input=q_quant_input,
            helix_position_offsets=metadata.helix_position_offsets,
            helix_is_inactive_rank=metadata.helix_is_inactive_rank,
        )
        torch.cuda.synchronize()

        def rotate_pairs(values: torch.Tensor) -> torch.Tensor:
            pairs = values.reshape(*values.shape[:-1], qk_rope_head_dim // 2, 2)
            return torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)

        torch.testing.assert_close(q_rope_out, rotate_pairs(original_q_pe), rtol=0, atol=0)
        expected_q_fp4, expected_q_sf = torch.ops.trtllm.fp4_quantize_with_residual(
            q_quant_input.reshape(-1, head_dim),
            metadata._fp4_mla_q_global_scale,
            FP4_MLA_Q_RESIDUAL_DIM,
            is_act=True,
        )
        torch.testing.assert_close(metadata._fp4_mla_prequantized_q, expected_q_fp4)
        q_sf_per_token = (head_dim + FP4_MLA_Q_RESIDUAL_DIM) // FP4_BLOCK_SIZE
        valid_q_sf_offsets = torch.tensor(
            [
                _swizzled_sf_offset(row_idx, col_idx, q_sf_per_token)
                for row_idx in range(num_heads)
                for col_idx in range(q_sf_per_token)
            ],
            device="cuda",
        )
        torch.testing.assert_close(
            metadata._fp4_mla_prequantized_q_sf.view(torch.uint8)[valid_q_sf_offsets],
            expected_q_sf.view(torch.uint8).flatten()[valid_q_sf_offsets],
        )
        assert metadata._fp4_mla_q_batch_capacity == 1
        if inactive_rank:
            for cache_tensor in (
                kv_cache,
                sf_cache,
                metadata.high_precision_kv_pool,
                metadata.fp4_mla_v_scale_pool,
                v_packed_pool,
            ):
                assert not torch.count_nonzero(cache_tensor)
            return

        expected_latent = generation_latent.clone()
        expected_latent[:, kv_lora_rank:] = rotate_pairs(generation_latent[:, kv_lora_rank:])
        batch_indices = torch.zeros(1, dtype=torch.int32, device="cuda")
        local_token = _materialize_reference_cache_tokens(
            metadata,
            layer_idx=0,
            batch_indices=batch_indices,
            positions=torch.tensor([local_cache_position], dtype=torch.int32, device="cuda"),
            head_dim=head_dim,
        )
        torch.testing.assert_close(local_token, expected_latent.float(), rtol=0.25, atol=0.2)
        global_slot = _materialize_reference_cache_tokens(
            metadata,
            layer_idx=0,
            batch_indices=batch_indices,
            positions=metadata.helix_position_offsets,
            head_dim=head_dim,
        )
        assert not torch.count_nonzero(global_slot)
    finally:
        torch.cuda.synchronize()
        _reset_triton_allocator()
        kv_cache_manager.shutdown()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _fp4_mla_context_cache_state(metadata, ends) -> dict:
    """Copy every cache role a context writes, plus each sequence's open HP tail."""
    manager = metadata.kv_cache_manager
    kv_cache, sf_cache = manager.get_fp4_mla_cache_buffers(0)
    state = {"kv": kv_cache.clone(), "sf": sf_cache.clone()}
    if metadata.fp4_mla_v_scale_pool is not None:
        state["v_scale"] = metadata.fp4_mla_v_scale_pool.view(torch.uint8).clone()
    v_packed = manager.get_mla_v_packed_pool_base()
    if v_packed is not None:
        state["v_packed"] = v_packed.clone()
    pool = metadata.high_precision_kv_pool
    ring, head_dim = _get_fp4_mla_hp_pool_layout(metadata, pool)
    stride = metadata.fp4_mla_page_table_stride
    for seq_idx, end in enumerate(ends):
        tile = end - end % HP_BLOCK_SIZE
        if tile == end:
            continue
        page = int(metadata._fp4_mla_hp_page_indices[seq_idx * stride + tile // metadata.page_size])
        slots = [position % ring for position in range(tile, end)]
        state[f"hp_tail_{seq_idx}"] = pool[page, 0, 0].view(ring, head_dim)[slots].clone()
    return state


def _clear_fp4_mla_context_cache(metadata) -> None:
    manager = metadata.kv_cache_manager
    kv_cache, sf_cache = manager.get_fp4_mla_cache_buffers(0)
    kv_cache.zero_()
    sf_cache.zero_()
    if metadata.fp4_mla_v_scale_pool is not None:
        metadata.fp4_mla_v_scale_pool.view(torch.uint8).zero_()
    v_packed = manager.get_mla_v_packed_pool_base()
    if v_packed is not None:
        v_packed.zero_()
    metadata.high_precision_kv_pool.zero_()


def _scatter_fp4_mla_context_chunks(
    metadata, latent, q_context, rotary_cos_sin, seq_lens, chunks, q_nope_head_dim
):
    """Scatter one context chunk [lo, hi) per sequence, as a forward would.

    ``latent`` / ``q_context`` hold the full sequences back to back; returns the
    chunk's latent (K RoPE written back with Q RoPE) and rotated Q.
    """
    device = latent.device
    offsets = [sum(seq_lens[:seq_idx]) for seq_idx in range(len(seq_lens))]
    rows = torch.cat(
        [
            torch.arange(offset + lo, offset + hi, device=device)
            for offset, (lo, hi) in zip(offsets, chunks)
        ]
    )
    lengths = [hi - lo for lo, hi in chunks]
    metadata.batch_indices = torch.cat(
        [
            torch.full((length,), seq_idx, dtype=torch.int32, device=device)
            for seq_idx, length in enumerate(lengths)
        ]
    )
    metadata.positions = torch.cat(
        [torch.arange(lo, hi, dtype=torch.int32, device=device) for lo, hi in chunks]
    )
    metadata.kv_lens_cuda_runtime = torch.tensor(
        [hi for _, hi in chunks], dtype=torch.int32, device=device
    )
    metadata.prompt_lens_cuda_runtime = torch.tensor(lengths, dtype=torch.int32, device=device)
    metadata.prompt_lens_cpu_runtime = torch.tensor(lengths, dtype=torch.int32)
    # The production repack bound (the pages this chunk appends to, not the whole
    # sequence) and lead-tile gate, as configure_fp4_mla_device_page_table sets them.
    host_kv_lens = torch.tensor([hi for _, hi in chunks], dtype=torch.int32)
    metadata.fp4_mla_context_repack_max_touched_pages = (
        fp4_mla_backend._fp4_mla_context_repack_touched_pages(
            host_kv_lens, metadata.prompt_lens_cpu_runtime
        )
    )
    metadata.fp4_mla_context_resumes_inside_tile = (
        fp4_mla_backend._fp4_mla_context_resumes_inside_tile(
            host_kv_lens, metadata.prompt_lens_cpu_runtime
        )
    )
    chunk_latent = latent[rows].contiguous()
    chunk_q = None if q_context is None else q_context[rows].contiguous()
    assert scatter_fp4_mla_kv_cache(
        metadata,
        chunk_latent,
        layer_idx=0,
        token_offset=0,
        phase="context",
        local_layer=0,
        v_head_dim=512,
        rotary_cos_sin=rotary_cos_sin,
        q_context=chunk_q,
        q_nope_head_dim=None if chunk_q is None else q_nope_head_dim,
    )
    return chunk_latent, chunk_q


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="requires Rubin SM107",
)
@pytest.mark.parametrize("backend", ["cutedsl", "triton"])
@pytest.mark.parametrize("enable_block_reuse", [False, True], ids=["direct_v", "repacked_v"])
@pytest.mark.parametrize("with_q_rope", [False, True], ids=["k_rope", "qk_rope"])
def test_fp4_mla_context_resumed_inside_tile_matches_single_context(
    monkeypatch, backend: str, enable_block_reuse: bool, with_q_rope: bool
) -> None:
    """A context that starts inside a 16-token tile (a prefix reused at any token,
    or a chunk that ended mid-tile) leaves the cache bit-identical to one context
    over the whole sequence: K codes and scales, V scales, packed V, and the HP tail.
    """
    if backend == "cutedsl" and not _cutedsl_backend_available():
        pytest.skip("CuTe DSL FP4 MLA backend is unavailable")
    _reset_triton_allocator()
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, backend)
    _set_fp4_mla_v_layout_env(monkeypatch, Fp4MlaVLayout.DUAL_COPY)
    # (resume point, end) per sequence: inside the tile of the end, at a page's
    # last tile across the page boundary, tile-aligned, inside the first tile
    # with the chunk's final partial tile in the same page (its HP ring writes
    # hit the slots the resumed tile reads), and ending exactly on a tile boundary.
    starts = [133, 250, 128, 1, 37]
    ends = [139, 300, 181, 47, 48]
    page_size = FP4_MLA_TOKENS_PER_BLOCK
    head_dim, rope_dim, q_nope_head_dim, num_heads = 576, 64, 128, 8
    torch.manual_seed(20261020)
    device = torch.device("cuda")
    manager = _create_fp4_mla_v2_manager(
        max_tokens=16 * page_size,
        max_seq_len=4 * page_size,
        max_batch_size=len(ends),
        enable_block_reuse=enable_block_reuse,
    )
    try:
        manager.add_dummy_requests(list(range(len(ends))), ends)
        metadata = _build_multi_seq_metadata(manager, seq_lens=ends, page_size=page_size)
        latent = (
            torch.randn(sum(ends), head_dim, dtype=torch.bfloat16, device=device) * 0.25
        ).clamp_(-1.0, 1.0)
        q_context = None
        if with_q_rope:
            q_context = torch.randn(
                sum(ends),
                num_heads * (q_nope_head_dim + rope_dim),
                dtype=torch.bfloat16,
                device=device,
            )
        angles = torch.rand(4 * page_size, rope_dim, device=device) * 6.283
        rotary_cos_sin = torch.stack((angles.cos(), angles.sin()), dim=-1).contiguous()

        _clear_fp4_mla_context_cache(metadata)
        single_latent, single_q = _scatter_fp4_mla_context_chunks(
            metadata,
            latent,
            q_context,
            rotary_cos_sin,
            ends,
            [(0, end) for end in ends],
            q_nope_head_dim,
        )
        torch.cuda.synchronize()
        expected = _fp4_mla_context_cache_state(metadata, ends)

        _clear_fp4_mla_context_cache(metadata)
        head_latent, head_q = _scatter_fp4_mla_context_chunks(
            metadata,
            latent,
            q_context,
            rotary_cos_sin,
            ends,
            [(0, start) for start in starts],
            q_nope_head_dim,
        )
        tail_latent, tail_q = _scatter_fp4_mla_context_chunks(
            metadata,
            latent,
            q_context,
            rotary_cos_sin,
            ends,
            list(zip(starts, ends)),
            q_nope_head_dim,
        )
        torch.cuda.synchronize()
        resumed = _fp4_mla_context_cache_state(metadata, ends)

        assert resumed.keys() == expected.keys()
        for name, tensor in expected.items():
            assert torch.equal(resumed[name].view(torch.uint8), tensor.view(torch.uint8)), name

        # Per-token outputs (K RoPE written back, rotated Q) do not depend on the split.
        def regroup(head, tail):
            head_parts = head.split([start for start in starts])
            tail_parts = tail.split([end - start for start, end in zip(starts, ends)])
            return torch.cat([part for pair in zip(head_parts, tail_parts) for part in pair])

        assert torch.equal(regroup(head_latent, tail_latent), single_latent)
        if with_q_rope:
            assert torch.equal(regroup(head_q, tail_q), single_q)
    finally:
        torch.cuda.synchronize()
        _reset_triton_allocator()
        manager.shutdown()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="requires Rubin SM107",
)
def test_fp4_mla_chunked_cache_gather_dequantizes_k_residual(monkeypatch) -> None:
    _reset_triton_allocator()
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "triton")
    seq_lens = [300, 180]
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    head_dim = kv_lora_rank + qk_rope_head_dim
    kv_cache_manager = _create_fp4_mla_v2_manager(
        max_tokens=640,
        max_seq_len=max(seq_lens),
        max_batch_size=len(seq_lens),
    )
    try:
        kv_cache_manager.add_dummy_requests(list(range(len(seq_lens))), seq_lens)
        metadata = _build_multi_seq_metadata(
            kv_cache_manager,
            seq_lens=seq_lens,
            page_size=FP4_MLA_TOKENS_PER_BLOCK,
        )
        latent = (
            torch.randn(sum(seq_lens), head_dim, dtype=torch.bfloat16, device="cuda") * 0.25
        ).clamp_(-1.0, 1.0)
        scatter_fp4_mla_kv_cache(
            metadata,
            latent,
            layer_idx=0,
            token_offset=0,
            phase="context",
            local_layer=0,
            v_head_dim=kv_lora_rank,
        )

        chunk_lens = [96, 80]
        chunk_offsets = torch.tensor([128, 64], dtype=torch.int64, device="cuda")
        cu_chunk_lens = torch.tensor([0, 96, 176], dtype=torch.int64, device="cuda")
        gathered_compressed_kv, gathered_k_pe = load_fp4_mla_chunked_kv_cache(
            metadata,
            layer_idx=0,
            num_ctx_cached_tokens=sum(chunk_lens),
            cu_chunked_seq_len=cu_chunk_lens,
            chunked_global_offset=chunk_offsets,
            chunked_max_seq_len=max(chunk_lens),
            out_dtype=torch.bfloat16,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        )
        batch_indices = torch.tensor(
            [0] * chunk_lens[0] + [1] * chunk_lens[1],
            dtype=torch.int32,
            device="cuda",
        )
        positions = torch.cat(
            [
                torch.arange(
                    offset,
                    offset + length,
                    dtype=torch.int32,
                    device="cuda",
                )
                for offset, length in zip(chunk_offsets.tolist(), chunk_lens)
            ]
        )
        reference = _materialize_reference_cache_tokens(
            metadata,
            layer_idx=0,
            batch_indices=batch_indices,
            positions=positions,
            head_dim=head_dim,
            include_k_residual=True,
        ).to(torch.bfloat16)

        torch.testing.assert_close(
            gathered_compressed_kv,
            reference[:, :kv_lora_rank],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            gathered_k_pe,
            reference[:, kv_lora_rank:],
            rtol=0,
            atol=0,
        )
    finally:
        torch.cuda.synchronize()
        _reset_triton_allocator()
        kv_cache_manager.shutdown()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _build_fp4_mla_attention_decode_case(
    *,
    seq_lens,
    num_heads,
    seed,
    query_len_per_seq=1,
    enable_helix=False,
    enable_block_reuse=False,
):
    torch.manual_seed(seed)
    device = torch.device("cuda")

    kv_lora_rank = 512
    qk_rope_head_dim = 64
    head_dim = kv_lora_rank + qk_rope_head_dim
    page_size = FP4_MLA_TOKENS_PER_BLOCK
    num_blocks = [(seq_len + page_size - 1) // page_size for seq_len in seq_lens]
    num_pages = sum(num_blocks)
    max_seq_len = max(page_size, max(seq_lens))
    max_tokens = max(page_size, num_pages * page_size)
    # The V2 manager splits its budget across pools, so an exactly sized budget
    # cannot always admit long MTP requests (the dummy request is silently
    # dropped); give the test manager headroom.
    max_tokens += max(max_tokens // 2, len(seq_lens) * 64 * page_size)
    max_seq_len += 32 * page_size
    context_seq_lens = [seq_len - query_len_per_seq for seq_len in seq_lens]
    if min(context_seq_lens) <= 0:
        raise ValueError("FP4 MLA decode cases require a non-empty context for every sequence.")
    spec_config = (
        MTPDecodingConfig(max_draft_len=query_len_per_seq - 1) if query_len_per_seq > 1 else None
    )

    kv_cache_manager = _create_fp4_mla_v2_manager(
        max_tokens=max_tokens,
        max_seq_len=max_seq_len,
        max_batch_size=len(seq_lens),
        spec_config=spec_config,
        enable_block_reuse=enable_block_reuse,
    )
    kv_cache_manager.add_dummy_requests(list(range(len(seq_lens))), seq_lens)
    expected_storage_head_dim = head_dim + (
        FP4_MLA_K_RESIDUAL_DIM if _fp4_mla_attention_backend() in ("triton", "cutedsl") else 0
    )
    kv_cache, sf_cache = kv_cache_manager.get_fp4_mla_cache_buffers(0)
    kv_cache.zero_()
    sf_cache.zero_()
    assert kv_cache.shape[-1] * 2 == expected_storage_head_dim
    assert sf_cache.shape[-1] == expected_storage_head_dim // FP4_BLOCK_SIZE

    metadata = _build_multi_seq_metadata(
        kv_cache_manager,
        seq_lens=seq_lens,
        page_size=page_size,
    )
    assert metadata.fp4_mla_v_scale_pool is not None
    persistent_pool_base = kv_cache_manager.get_mla_v_packed_pool_base()
    if persistent_pool_base is not None:
        persistent_pool_base.zero_()
    metadata.fp4_mla_v_scale_pool.zero_()

    metadata.kv_lens_cuda_runtime = torch.tensor(
        context_seq_lens,
        dtype=torch.int32,
        device=device,
    )
    metadata.prompt_lens_cuda_runtime = metadata.kv_lens_cuda_runtime.clone()
    metadata.prompt_lens_cpu_runtime = torch.tensor(context_seq_lens, dtype=torch.int32)
    metadata.batch_indices = torch.cat(
        [
            torch.full((seq_len,), seq_idx, dtype=torch.int32, device=device)
            for seq_idx, seq_len in enumerate(context_seq_lens)
        ]
    )
    metadata.positions = torch.cat(
        [torch.arange(seq_len, dtype=torch.int32, device=device) for seq_len in context_seq_lens]
    )
    context_latent = (
        torch.randn(sum(context_seq_lens), head_dim, dtype=torch.bfloat16, device=device) * 0.25
    ).clamp_(-1.0, 1.0)
    scatter_fp4_mla_kv_cache(
        metadata,
        context_latent,
        layer_idx=0,
        token_offset=0,
        phase="context",
        local_layer=0,
        v_head_dim=kv_lora_rank,
    )
    torch.cuda.synchronize()

    metadata.num_contexts = 0
    metadata.num_context_blocks = 0
    metadata.num_generation_blocks = len(seq_lens) * metadata.fp4_mla_page_table_stride
    metadata.kv_lens_cuda_runtime = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    metadata.prompt_lens_cuda_runtime = torch.full(
        (len(seq_lens),), query_len_per_seq, dtype=torch.int32, device=device
    )
    metadata.prompt_lens_cpu_runtime = torch.full(
        (len(seq_lens),), query_len_per_seq, dtype=torch.int32
    )
    num_queries = len(seq_lens) * query_len_per_seq
    metadata.batch_indices = torch.arange(
        len(seq_lens),
        dtype=torch.int32,
        device=device,
    ).repeat_interleave(query_len_per_seq)
    metadata.positions = torch.cat(
        [
            torch.arange(context_len, seq_len, dtype=torch.int32, device=device)
            for context_len, seq_len in zip(context_seq_lens, seq_lens)
        ]
    )
    metadata.num_tokens = num_queries
    metadata.num_ctx_tokens = 0
    metadata.max_num_tokens = num_queries
    metadata.max_num_sequences = len(seq_lens)
    metadata.max_total_draft_tokens = query_len_per_seq - 1
    metadata.runtime_features.has_speculative_draft_tokens = query_len_per_seq > 1
    generation_latent = (
        torch.randn(num_queries, head_dim, dtype=torch.bfloat16, device=device) * 0.25
    ).clamp_(-1.0, 1.0)
    generation_latent[:, :FP4_BLOCK_SIZE] = 0.75
    generation_latent[:, 1:FP4_BLOCK_SIZE:2] = -0.75
    q_nope = (
        torch.randn(num_queries, num_heads, kv_lora_rank, dtype=torch.bfloat16, device=device)
        * 0.25
    ).clamp_(-1.0, 1.0)
    q_pe = (
        torch.randn(
            num_queries,
            num_heads,
            qk_rope_head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.25
    ).clamp_(-1.0, 1.0)
    q_quant_input = torch.cat((q_nope, torch.zeros_like(q_pe)), dim=-1).contiguous()
    q_rope_out = q_quant_input[..., kv_lora_rank:]
    rotary_cos_sin = torch.zeros(
        (max_seq_len, qk_rope_head_dim, 2),
        dtype=torch.float32,
        device=device,
    )
    rotary_cos_sin[..., 0] = 1.0

    if enable_helix:
        # CP1 exercises the Helix per-token metadata contract while the
        # verify group below still crosses a physical page boundary.
        metadata.helix_position_offsets = metadata.positions.clone()
        metadata.helix_local_slots = metadata.positions.clone()
        metadata.helix_kv_bounds = metadata.positions + 1
        metadata.helix_is_inactive_rank = torch.zeros(
            len(seq_lens), dtype=torch.bool, device=device
        )
        metadata._helix_spec_tokens_valid = query_len_per_seq > 1

    assert scatter_fp4_mla_kv_cache(
        metadata,
        generation_latent,
        layer_idx=0,
        token_offset=0,
        phase="generation",
        local_layer=0,
        v_head_dim=kv_lora_rank,
        rotary_cos_sin=rotary_cos_sin,
        q_pe=q_pe,
        q_rope_out=q_rope_out,
        q_quant_input=q_quant_input,
        helix_position_offsets=getattr(metadata, "helix_position_offsets", None),
        helix_is_inactive_rank=getattr(metadata, "helix_is_inactive_rank", None),
    )
    torch.cuda.synchronize()
    assert metadata._fp4_mla_prequantized_q is not None
    assert metadata._fp4_mla_prequantized_q_sf is not None
    assert metadata._fp4_mla_q_batch_capacity == num_queries
    torch.testing.assert_close(q_rope_out, q_pe, rtol=0, atol=0)

    canonical_generation = _materialize_reference_cache_tokens(
        metadata,
        layer_idx=0,
        batch_indices=metadata.batch_indices,
        positions=metadata.positions,
        head_dim=head_dim,
    )
    torch.testing.assert_close(
        canonical_generation,
        generation_latent.float(),
        rtol=0.25,
        atol=0.2,
        msg="generation scatter did not write the distinct BF16 latent into canonical KV",
    )

    # Decode reference exercises the quantized cache without an HP overlay.
    metadata.high_precision_kv_pool.zero_()
    torch.cuda.synchronize()

    return kv_cache_manager, metadata, q_quant_input, kv_lora_rank, qk_rope_head_dim


def _fp4_mla_attention_decode_reference(
    metadata,
    q_nope,
    q_pe,
    *,
    sm_scale,
    kv_lora_rank,
    qk_rope_head_dim,
) -> torch.Tensor:
    head_dim = kv_lora_rank + qk_rope_head_dim
    storage = _materialize_reference_cache_storage(metadata, 0, head_dim)
    dequant_cache = storage[..., :head_dim]
    dequant_k_residual = None
    if _fp4_mla_attention_backend() in ("triton", "cutedsl"):
        dequant_k_residual = storage[..., head_dim : head_dim + FP4_MLA_K_RESIDUAL_DIM]
    num_heads = q_nope.shape[1]
    q_full = torch.cat((q_nope, q_pe), dim=-1).reshape(-1, head_dim)
    global_scale = metadata._fp4_mla_q_global_scale
    q_fp4, q_sf = torch.ops.trtllm.fp4_quantize_with_residual(
        q_full,
        global_scale,
        FP4_MLA_Q_RESIDUAL_DIM,
        is_act=True,
    )
    q_logical_dim = head_dim + FP4_MLA_Q_RESIDUAL_DIM
    q_dequant = _dequant_fp4_swizzled(
        q_fp4,
        q_sf.view(torch.float8_e4m3fn),
        logical_dim=q_logical_dim,
        sf_per_token=q_logical_dim // FP4_BLOCK_SIZE,
        global_scale=float(global_scale.item()),
    )

    p_dequant = None
    if hasattr(metadata, "_fp4_mla_attention_p_buf"):
        p_dequant = _dequant_fp4_swizzled(
            metadata._fp4_mla_attention_p_buf,
            metadata._fp4_mla_attention_p_sf_buf,
            logical_dim=metadata.page_size,
            sf_per_token=metadata.page_size // FP4_BLOCK_SIZE,
            global_scale=FP4_MLA_P_GLOBAL_SCALE,
        )

    indptr = metadata.paged_kv_indptr_decode.cpu().tolist()
    num_seqs = metadata.num_seqs - metadata.num_contexts
    indptr = indptr[: num_seqs + 1]
    kv_lens = (
        metadata.kv_lens_cuda_runtime[metadata.num_contexts : metadata.num_seqs].cpu().tolist()
    )
    assert len(kv_lens) == num_seqs
    assert len(indptr) == num_seqs + 1
    assert indptr[0] == 0
    assert indptr[-1] == storage.shape[0]
    query_len_per_seq = q_nope.shape[0] // num_seqs
    max_pages = max(indptr[seq_idx + 1] - indptr[seq_idx] for seq_idx in range(num_seqs))
    outputs = []
    softmax_stats = []
    for seq_idx in range(num_seqs):
        kv_len = kv_lens[seq_idx]
        page_count = indptr[seq_idx + 1] - indptr[seq_idx]
        assert page_count == metadata.fp4_mla_page_table_stride
        assert page_count * metadata.page_size >= kv_len
        full_cache = dequant_cache[indptr[seq_idx] : indptr[seq_idx + 1]].reshape(-1, head_dim)
        assert full_cache.shape[0] >= kv_len
        full_v_cache = full_cache[:, :kv_lora_rank]
        for query_offset in range(query_len_per_seq):
            query_idx = seq_idx * query_len_per_seq + query_offset
            effective_kv_len = kv_len - (query_len_per_seq - 1 - query_offset)
            cache = full_cache[:effective_kv_len]
            v_cache = full_v_cache[:effective_kv_len]
            q_start = query_idx * num_heads
            q = q_dequant[q_start : q_start + num_heads]
            if dequant_k_residual is None:
                logical_q = q
                logical_k = _duplicate_tail_groups(cache.float(), FP4_MLA_Q_RESIDUAL_DIM)
            else:
                full_k_residual = dequant_k_residual[indptr[seq_idx] : indptr[seq_idx + 1]].reshape(
                    -1, FP4_MLA_K_RESIDUAL_DIM
                )
                logical_q, logical_k = _expand_qk_residual_terms(
                    q,
                    cache.float(),
                    full_k_residual[:effective_kv_len],
                    FP4_MLA_Q_RESIDUAL_DIM,
                )
            scores = torch.matmul(logical_q, logical_k.transpose(0, 1)) * sm_scale
            row_max = scores.max(dim=-1).values
            row_sum = torch.exp(scores - row_max[:, None]).sum(dim=-1)
            softmax_stats.append(torch.stack((row_max, row_sum), dim=-1))
            probs = torch.softmax(scores, dim=-1)

            if p_dequant is None:
                p = probs
            else:
                p_pages = []
                for page_rel in range(indptr[seq_idx + 1] - indptr[seq_idx]):
                    page_start = page_rel * metadata.page_size
                    valid_tokens = max(min(effective_kv_len - page_start, metadata.page_size), 0)
                    if valid_tokens == 0:
                        continue
                    p_page = query_idx * max_pages + page_rel
                    p_start = p_page * num_heads
                    p_pages.append(p_dequant[p_start : p_start + num_heads, :valid_tokens])
                p = torch.cat(p_pages, dim=-1)

            outputs.append(torch.matmul(p, v_cache.float()))
    return torch.stack(outputs, dim=0), torch.stack(softmax_stats, dim=0)


def _assert_fp4_mla_attention_decode_accuracy(
    monkeypatch,
    *,
    backend: str,
    fused_v_transpose: bool,
    num_heads: int,
    seq_lens: list[int],
    seed: int,
    query_len_per_seq: int = 1,
    enable_helix: bool = False,
    enable_block_reuse: bool = False,
    packed_queries: bool = False,
    split_kv: Optional[int] = None,
) -> None:
    _reset_triton_allocator()
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, backend)
    monkeypatch.setenv(FP4_MLA_CUTEDSL_MUFU16_ENV, "1")
    _set_fp4_mla_v_layout_env(
        monkeypatch, Fp4MlaVLayout.FUSED_V if fused_v_transpose else Fp4MlaVLayout.DUAL_COPY
    )
    # Packed decode rows are the default; the legacy padded path stays covered
    # by the existing cases, which opt out explicitly.
    monkeypatch.setenv(fp4_mla_backend.FP4_MLA_PACKED_Q_ENV, str(int(packed_queries)))
    monkeypatch.setenv("TRTLLM_FP4_MLA_SPLIT_KV", "auto" if split_kv is None else str(split_kv))
    (
        kv_cache_manager,
        metadata,
        q,
        kv_lora_rank,
        qk_rope_head_dim,
    ) = _build_fp4_mla_attention_decode_case(
        seq_lens=seq_lens,
        num_heads=num_heads,
        seed=seed,
        query_len_per_seq=query_len_per_seq,
        enable_helix=enable_helix,
        enable_block_reuse=enable_block_reuse,
    )
    try:
        v_packed_pool = fp4_mla_backend._get_fp4_mla_v_packed_pool(metadata, 0)
        assert (v_packed_pool is None) is fused_v_transpose

        output = torch.empty_like(q[..., :kv_lora_rank])
        softmax_stats = None
        if backend == "cutedsl" and (query_len_per_seq == 1 or enable_helix):
            softmax_stats = torch.empty((*q.shape[:2], 2), dtype=torch.float32, device=q.device)
        sm_scale = 0.1
        run_fp4_mla_attention_decode(
            metadata,
            layer_idx=0,
            local_layer=0,
            q=q,
            output=output,
            sm_scale=sm_scale,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            prequantized_q=metadata._fp4_mla_prequantized_q,
            prequantized_q_sf=metadata._fp4_mla_prequantized_q_sf,
            q_batch_capacity=metadata._fp4_mla_q_batch_capacity,
            softmax_stats_tensor=softmax_stats,
        )
        torch.cuda.synchronize()

        ref_output, ref_softmax_stats = _fp4_mla_attention_decode_reference(
            metadata,
            q[..., :kv_lora_rank],
            q[..., kv_lora_rank:],
            sm_scale=sm_scale,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        )
        output_float = output.float()
        error = output_float - ref_output
        output_rows = output_float.reshape(output_float.shape[0], -1)
        reference_rows = ref_output.reshape(ref_output.shape[0], -1)
        error_rows = error.reshape(error.shape[0], -1)
        output_norms = torch.linalg.vector_norm(output_rows, dim=1)
        reference_norms = torch.linalg.vector_norm(reference_rows, dim=1)
        relative_l2 = torch.linalg.vector_norm(error_rows, dim=1) / reference_norms
        cosine = torch.sum(output_rows * reference_rows, dim=1) / (output_norms * reference_norms)
        assert bool(torch.all(reference_norms > 1e-3).item())
        assert bool(torch.all(output_norms > 1e-3).item())
        max_relative_l2 = float(relative_l2.max().item())
        min_cosine = float(cosine.min().item())
        assert max_relative_l2 < 0.15, (
            f"{backend} FP4 MLA attention relative L2 error is too large: {max_relative_l2}"
        )
        assert min_cosine > 0.99, (
            f"{backend} FP4 MLA attention cosine similarity is too small: {min_cosine}"
        )
        max_abs_error = torch.max(torch.abs(error)).item()
        torch.testing.assert_close(
            output_float,
            ref_output,
            atol=1.5e-1,
            rtol=1.5e-1,
            msg=(
                f"{backend} FP4 MLA attention decode output diverged from reference; "
                f"max_abs_error={max_abs_error}, max_relative_l2={max_relative_l2}, "
                f"min_cosine={min_cosine}"
            ),
        )
        if softmax_stats is not None:
            torch.testing.assert_close(
                softmax_stats[..., 0],
                ref_softmax_stats[..., 0],
                atol=1.5e-1,
                rtol=1.5e-1,
                msg=f"{backend} FP4 MLA softmax row maxima diverged from reference",
            )
            torch.testing.assert_close(
                softmax_stats[..., 1],
                ref_softmax_stats[..., 1],
                atol=2.5e-1,
                rtol=1.5e-1,
                msg=f"{backend} FP4 MLA softmax row sums diverged from reference",
            )
    finally:
        torch.cuda.synchronize()
        _reset_triton_allocator()
        kv_cache_manager.shutdown()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


@pytest.mark.skipif(
    _is_cutedsl_unavailable(),
    reason="requires Rubin SM107 and the CuTe DSL runtime",
)
@pytest.mark.parametrize(
    "fused_v_transpose",
    [False, True],
    ids=["mufu16", "mufu16-fused-v-transpose"],
)
@pytest.mark.parametrize("query_len_per_seq", [1, 4], ids=["mtp0", "mtp3"])
def test_fp4_mla_attention_decode_cutedsl_matches_reference(
    monkeypatch,
    fused_v_transpose: bool,
    query_len_per_seq: int,
) -> None:
    _assert_fp4_mla_attention_decode_accuracy(
        monkeypatch,
        backend="cutedsl",
        fused_v_transpose=fused_v_transpose,
        num_heads=128,
        seq_lens=[131, 512],
        seed=29,
        query_len_per_seq=query_len_per_seq,
    )


@pytest.mark.skipif(
    _is_cutedsl_unavailable(),
    reason="requires Rubin SM107 and the CuTe DSL runtime",
)
@pytest.mark.parametrize(
    "fused_v_transpose",
    [False, True],
    ids=["mufu16", "mufu16-fused-v-transpose"],
)
def test_fp4_mla_attention_decode_cutedsl_helix_mtp5_matches_reference(
    monkeypatch,
    fused_v_transpose: bool,
) -> None:
    _assert_fp4_mla_attention_decode_accuracy(
        monkeypatch,
        backend="cutedsl",
        fused_v_transpose=fused_v_transpose,
        num_heads=128,
        seq_lens=[131, 512],
        seed=37,
        query_len_per_seq=6,
        enable_helix=True,
    )


@pytest.mark.skipif(
    _is_cutedsl_unavailable(),
    reason="requires Rubin SM107 and the CuTe DSL runtime",
)
@pytest.mark.parametrize(
    "num_heads,query_len_per_seq,enable_helix,split_kv,seq_lens",
    [
        (128, 1, False, 1, [131, 512]),
        (128, 4, False, 2, [131, 512]),
        (96, 8, False, 1, [131, 2100]),
        (96, 8, True, 1, [131, 2100]),
        (96, 8, True, 4, [131, 2100]),
        (96, 1, True, 4, [131, 2100]),
        (12, 8, False, 4, [131, 900]),
        (12, 8, True, 2, [131, 900]),
        (6, 6, False, 3, [131, 700]),
        # Imbalanced batches: the balanced plan gives the long requests most of
        # the task slots (DEP-style traffic, see plan_packed_tasks).
        (96, 8, False, 4, [131, 20000, 600, 45000]),
        (96, 8, True, 4, [131, 20000, 600, 45000]),
        (128, 1, False, 16, [300, 70000, 2200]),
        (12, 8, False, 2, [131, 9000, 131, 131, 131, 131, 131, 24000]),
        # Wave-boundary batches: the plan cuts variable-length wrap-around pieces
        # (fp4_mla_packed_reduce.plan_wrap_pieces), so tasks start mid-request.
        (96, 8, False, 4, [10240] * 20),
        (96, 4, False, 3, [8192] * 40),
        (96, 8, False, 4, [9000, 15000, 4000, 22000, 7000, 30000, 12000, 5000] * 3),
        (12, 8, False, 3, [20000] * 64),
        # Past 64 requests the plan still simulates its candidates (PLAN_SIM_MAX_REQUESTS):
        # 72 even requests on 35 slots take wrap-around pieces, not one task each.
        (96, 4, False, 3, [16384] * 72),
        # One request over many more task slots than the former 16-slot cap: every row
        # reduces 64 (H12 q1: 2-tile tasks) or 32 (H96 q4, 512-column bands) partials.
        (12, 1, False, 128, [32768]),
        (96, 4, False, 32, [65536]),
    ],
    ids=[
        "h128-q1-s1",
        "h128-q4-s2",
        "h96-q8-s1",
        "h96-q8-helix-s1",
        "h96-q8-helix-s4",
        "h96-q1-helix-s4",
        "h12-q8-s4",
        "h12-q8-helix-s2",
        "h6-q6-s3",
        "h96-q8-skew-s4",
        "h96-q8-helix-skew-s4",
        "h128-q1-skew-s16",
        "h12-q8-skew8-s2",
        "h96-q8-wrap20-s4",
        "h96-q4-wrap40-s3",
        "h96-q8-wrapskew24-s4",
        "h12-q8-wrap64-s3",
        "h96-q4-wrap72-s3",
        "h12-q1-b1-s128",
        "h96-q4-b1-s32",
    ],
)
def test_fp4_mla_attention_decode_cutedsl_packed_matches_reference(
    monkeypatch,
    num_heads: int,
    query_len_per_seq: int,
    enable_helix: bool,
    split_kv: int,
    seq_lens: list[int],
) -> None:
    """Packed [query, head] rows, in-kernel KV splits, per-row (Helix) bounds."""
    _assert_fp4_mla_attention_decode_accuracy(
        monkeypatch,
        backend="cutedsl",
        fused_v_transpose=False,
        num_heads=num_heads,
        seq_lens=seq_lens,
        seed=41,
        query_len_per_seq=query_len_per_seq,
        enable_helix=enable_helix,
        packed_queries=True,
        split_kv=split_kv,
    )


@pytest.mark.skipif(
    _is_cutedsl_unavailable(),
    reason="requires Rubin SM107 and the CuTe DSL runtime",
)
@pytest.mark.parametrize(
    "switch_env",
    [
        "TRTLLM_FP4_MLA_PACKED_FEWER_WAITS",
        "TRTLLM_FP4_MLA_FHADD2_ROWSUM",
        "TRTLLM_FP4_MLA_DECODE_BULK_EPILOGUE",
        "TRTLLM_FP4_MLA_DECODE_WARM_FINAL_TILE",
        "TRTLLM_FP4_MLA_DECODE_EARLY_K0",
    ],
    ids=[
        "fewer_waits",
        "fhadd2_rowsum",
        "bulk_epilogue",
        "warm_final_tile",
        "early_k0",
    ],
)
@pytest.mark.parametrize(
    "num_heads,query_len_per_seq,split_kv,seq_lens",
    [
        (96, 4, 3, [8192] * 40),
        (96, 8, 4, [131, 20000, 600, 45000]),
        (12, 8, 2, [131, 9000, 131, 131, 131, 131, 131, 24000]),
        (96, 1, 1, [131, 2100]),
    ],
    ids=["h96-q4-wrap40-s3", "h96-q8-skew-s4", "h12-q8-skew8-s2", "h96-q1-s1"],
)
def test_fp4_mla_packed_decode_switch_is_bitwise(
    monkeypatch,
    num_heads: int,
    query_len_per_seq: int,
    split_kv: int,
    seq_lens: list[int],
    switch_env: str,
) -> None:
    """Compile-time switches that keep the math give the same output bits on and off.

    TRTLLM_FP4_MLA_PACKED_FEWER_WAITS only drops producer waits;
    TRTLLM_FP4_MLA_FHADD2_ROWSUM fuses the row-sum widen-and-add into one instruction;
    TRTLLM_FP4_MLA_DECODE_BULK_EPILOGUE writes the split partial with TMA tensor stores from
    shared memory instead of per-lane vector stores (the same values at the same addresses);
    TRTLLM_FP4_MLA_DECODE_WARM_FINAL_TILE runs every tile, final and masked ones included,
    through one copy of the producer code that masks the scores in TMEM before its loads (the
    same arithmetic); TRTLLM_FP4_MLA_DECODE_EARLY_K0 arms the Q and first K barriers before the
    grid-dependency wait and splits the first K tile's TMA issue between the two CTAs (the same
    bytes in the same shared memory).
    """
    _assert_packed_decode_switch_is_bitwise(
        monkeypatch, num_heads, query_len_per_seq, split_kv, seq_lens, switch_env
    )


@pytest.mark.skipif(
    _is_cutedsl_unavailable(),
    reason="requires Rubin SM107 and the CuTe DSL runtime",
)
@pytest.mark.parametrize(
    "num_heads,query_len_per_seq,split_kv,seq_lens",
    [
        (6, 6, 3, [131, 700]),
        (12, 4, 2, [4096, 2100]),
        (12, 1, 4, [8192]),
    ],
    ids=["h6-q6-36rows-s3", "h12-q4-48rows-s2", "h12-q1-12rows-s4"],
)
def test_fp4_mla_packed_decode_bulk_epilogue_partial_warps_is_bitwise(
    monkeypatch,
    num_heads: int,
    query_len_per_seq: int,
    split_kv: int,
    seq_lens: list[int],
) -> None:
    """TMA-store epilogue vs per-lane stores when the packed rows end inside a store warp.

    36, 48 and 12 rows: the last warp's boxes are clipped at valid_m by the TMA unit, and
    the output must match the per-lane stores bit for bit.
    """
    _assert_packed_decode_switch_is_bitwise(
        monkeypatch,
        num_heads,
        query_len_per_seq,
        split_kv,
        seq_lens,
        "TRTLLM_FP4_MLA_DECODE_BULK_EPILOGUE",
    )


@pytest.mark.skipif(
    _is_cutedsl_unavailable(),
    reason="requires Rubin SM107 and the CuTe DSL runtime",
)
@pytest.mark.parametrize(
    "num_heads,query_len_per_seq,split_kv,seq_lens",
    [
        (12, 1, 1, [256, 512, 768, 4096]),
        (96, 1, 1, [255, 256, 257]),
        (12, 4, 1, [258, 257, 513]),
        (12, 1, 1, [7, 300]),
        (12, 1, 1, [272, 2064]),
        (12, 4, 2, [1026]),
        (96, 8, 2, [1029, 2053]),
    ],
    ids=[
        "h12-q1-1-2-3-16tiles",
        "h96-q1-key-bound",
        "h12-q4-two-masked-tiles",
        "h12-q1-7-300-keys",
        "h12-q1-group-bounds",
        "h12-q4-keyless-rows-s2",
        "h96-q8-keyless-rows-s2",
    ],
)
def test_fp4_mla_packed_decode_warm_final_tile_edges_is_bitwise(
    monkeypatch,
    num_heads: int,
    query_len_per_seq: int,
    split_kv: int,
    seq_lens: list[int],
) -> None:
    """One producer-tile copy vs the separate unmasked, masked and final-tile copies.

    Tasks of 1, 2, 3 and 16 tiles that need no mask; 255, 256 and 257 keys (a final group
    cut one key short, none cut, one key in the second tile); MTP rows whose earliest key
    bound puts two masked tiles in a task (rows end at 254-257 keys); 7 and 300 keys (16-key
    groups and score halves with no visible key); bounds at a group edge inside a tile (272
    and 2064 keys); and MTP rows that see no key of a split (bounds 1023-1026 and 1022-1029
    or 2046-2053 against two-tile tasks).
    """
    _assert_packed_decode_switch_is_bitwise(
        monkeypatch,
        num_heads,
        query_len_per_seq,
        split_kv,
        seq_lens,
        "TRTLLM_FP4_MLA_DECODE_WARM_FINAL_TILE",
    )


@pytest.mark.skipif(
    _is_cutedsl_unavailable(),
    reason="requires Rubin SM107 and the CuTe DSL runtime",
)
@pytest.mark.parametrize(
    "num_heads,query_len_per_seq,split_kv,seq_lens,enable_helix",
    [
        (12, 1, 2, [7, 300, 2064], False),
        (96, 8, 4, [131, 20000, 600, 45000], True),
    ],
    ids=["h12-q1-s2", "h96-q8-helix-skew-s4"],
)
def test_fp4_mla_packed_decode_warm_final_tile_softmax_stats_is_bitwise(
    monkeypatch,
    num_heads: int,
    query_len_per_seq: int,
    split_kv: int,
    seq_lens: list[int],
    enable_helix: bool,
) -> None:
    """The exported softmax row maxima and sums match too, Helix per-row bounds included."""
    _assert_packed_decode_switch_is_bitwise(
        monkeypatch,
        num_heads,
        query_len_per_seq,
        split_kv,
        seq_lens,
        "TRTLLM_FP4_MLA_DECODE_WARM_FINAL_TILE",
        enable_helix=enable_helix,
        compare_softmax_stats=True,
    )


@pytest.mark.skipif(
    _is_cutedsl_unavailable(),
    reason="requires Rubin SM107 and the CuTe DSL runtime",
)
@pytest.mark.parametrize(
    "num_heads,query_len_per_seq,split_kv,seq_lens",
    [
        (128, 1, None, [131, 512]),
        (128, 4, None, [131, 512]),
        (6, 6, 3, [2050]),
    ],
    ids=["h128-q1", "h128-q4", "h6-q6-folded-s3"],
)
def test_fp4_mla_unpacked_decode_warm_final_tile_is_bitwise(
    monkeypatch,
    num_heads: int,
    query_len_per_seq: int,
    split_kv: Optional[int],
    seq_lens: list[int],
) -> None:
    """The unpacked decode runs the same producer tiles; folded MTP rows included.

    Six folded queries end at 2045-2050 keys: the last two tiles of the third split are
    masked.
    """
    _assert_packed_decode_switch_is_bitwise(
        monkeypatch,
        num_heads,
        query_len_per_seq,
        split_kv,
        seq_lens,
        "TRTLLM_FP4_MLA_DECODE_WARM_FINAL_TILE",
        packed_queries=False,
    )


@pytest.mark.skipif(
    _is_cutedsl_unavailable(),
    reason="requires Rubin SM107 and the CuTe DSL runtime",
)
@pytest.mark.parametrize(
    "num_heads,query_len_per_seq,split_kv,seq_lens,enable_helix,compare_softmax_stats",
    [
        (12, 1, 1, [4096, 4096, 4096], False, True),
        (12, 4, 1, [300, 513, 768, 257], False, False),
        (96, 8, 1, [256, 129], False, False),
        (96, 4, 2, [70000, 1000], False, False),
        (12, 8, 5, [131072, 131072], False, False),
        (96, 4, 3, [8192] * 40, False, False),
        (96, 8, 4, [131, 20000, 600, 45000], False, False),
        (12, 8, 2, [131, 9000, 131, 131, 131, 131, 131, 24000], False, False),
        (96, 1, 1, [131, 2100], False, True),
        (96, 8, 4, [131, 20000, 600, 45000], True, True),
        (12, 8, 2, [131, 900], True, True),
        (6, 6, 3, [131, 700], False, False),
    ],
    ids=[
        "h12-q1-16tiles-stats",
        "h12-q4-short",
        "h96-q8-1tile",
        "h96-q4-skew-s2",
        "h12-q8-128k-s5",
        "h96-q4-wrap40-s3",
        "h96-q8-skew-s4",
        "h12-q8-skew8-s2",
        "h96-q1-s1-stats",
        "h96-q8-helix-skew-s4",
        "h12-q8-helix-s2",
        "h6-q6-s3",
    ],
)
def test_fp4_mla_packed_decode_single_copy_is_bitwise(
    monkeypatch,
    num_heads: int,
    query_len_per_seq: int,
    split_kv: int,
    seq_lens: list[int],
    enable_helix: bool,
    compare_softmax_stats: bool,
) -> None:
    """A single-copy cache decodes to the same bits as a dual-copy one, through the backend.

    The same context and generation tokens go into a dual-copy and a single-copy KV cache
    manager. The single-copy cache update and decode never touch a V^T sidecar, write the
    same K, K-scale and V-scale bytes, and the packed decode that reads V from the K latent
    gives the dual-copy output (and Helix softmax stats) bit for bit. Shapes: one to three
    tiles per task (prologue only), exactly 16 tiles (the split final QK tile), a 128K task
    (stage wrap and refills), skewed and wave-wrapping batches, and Helix per-row bounds.
    """
    _assert_single_copy_decode_is_bitwise(
        monkeypatch,
        num_heads,
        query_len_per_seq,
        split_kv,
        seq_lens,
        enable_helix=enable_helix,
        compare_softmax_stats=compare_softmax_stats,
    )


@pytest.mark.skipif(
    _is_cutedsl_unavailable(),
    reason="requires Rubin SM107 and the CuTe DSL runtime",
)
def test_fp4_mla_single_copy_decode_rejects_rows_beyond_the_packed_layout(monkeypatch) -> None:
    """128 heads x 9 queries need 1152 packed rows (limit 1024): the fold-Q and split-KV
    paths would read the V^T sidecar a single-copy cache does not have, so decode refuses."""
    _reset_triton_allocator()
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "cutedsl")
    monkeypatch.setenv(FP4_MLA_CUTEDSL_MUFU16_ENV, "1")
    _set_fp4_mla_v_layout_env(monkeypatch, Fp4MlaVLayout.PACKED_SINGLE_COPY)
    kv_cache_manager, metadata, q, kv_lora_rank, qk_rope_head_dim = (
        _build_fp4_mla_attention_decode_case(
            seq_lens=[131, 300], num_heads=128, seed=53, query_len_per_seq=9
        )
    )
    try:
        assert metadata._fp4_mla_prequantized_q_rows_per_seq == 0
        with pytest.raises(RuntimeError, match="single-copy V requires the packed decode"):
            run_fp4_mla_attention_decode(
                metadata,
                layer_idx=0,
                local_layer=0,
                q=q,
                output=torch.empty_like(q[..., :kv_lora_rank]),
                sm_scale=0.1,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                prequantized_q=metadata._fp4_mla_prequantized_q,
                prequantized_q_sf=metadata._fp4_mla_prequantized_q_sf,
                q_batch_capacity=metadata._fp4_mla_q_batch_capacity,
            )
    finally:
        torch.cuda.synchronize()
        _reset_triton_allocator()
        kv_cache_manager.shutdown()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


@pytest.mark.skipif(
    _is_cutedsl_unavailable(),
    reason="requires Rubin SM107 and the CuTe DSL runtime",
)
@pytest.mark.parametrize(
    "num_heads,query_len_per_seq,seq_lens",
    [
        (12, 1, [131, 2100, 8193]),
        (12, 4, [131, 8195, 1026, 4099]),
        (96, 8, [131, 20003, 607, 45010]),
    ],
    ids=["h12-q1", "h12-q4-straddle", "h96-q8-straddle"],
)
def test_fp4_mla_generation_update_uniform_length_is_bitwise(
    monkeypatch,
    num_heads: int,
    query_len_per_seq: int,
    seq_lens: list[int],
) -> None:
    """The cache update takes the batch's uniform generation length as an argument.

    Loading it per request instead (TRTLLM_FP4_MLA_UPDATE_UNIFORM_GEN_LEN=0) writes the
    same KV codes, scales, packed V and packed Q bytes. The lengths put new tokens across a
    16-token tile and, at 1026 and 4099 tokens, across a page.
    """
    _reset_triton_allocator()
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "cutedsl")
    monkeypatch.setenv(FP4_MLA_CUTEDSL_MUFU16_ENV, "1")
    _set_fp4_mla_v_layout_env(monkeypatch, Fp4MlaVLayout.DUAL_COPY)
    monkeypatch.setenv(fp4_mla_backend.FP4_MLA_PACKED_Q_ENV, "1")
    written = []
    for switch in ("0", "1"):
        monkeypatch.setenv(fp4_mla_backend.FP4_MLA_UPDATE_UNIFORM_GEN_LEN_ENV, switch)
        kv_cache_manager, metadata, q, _, _ = _build_fp4_mla_attention_decode_case(
            seq_lens=seq_lens,
            num_heads=num_heads,
            seed=47,
            query_len_per_seq=query_len_per_seq,
        )
        try:
            kv, sf = kv_cache_manager.get_fp4_mla_cache_buffers(0)
            buffers = {
                "kv": kv,
                "k_sf": sf,
                "v_sf": metadata.fp4_mla_v_scale_pool,
                "q": q,
                "q_fp4": metadata._fp4_mla_prequantized_q,
                "q_sf": metadata._fp4_mla_prequantized_q_sf,
            }
            if hasattr(kv_cache_manager, "get_mla_v_packed_pool"):
                buffers["v_packed"] = kv_cache_manager.get_mla_v_packed_pool(0)
            torch.cuda.synchronize()
            written.append(
                {
                    name: t.detach().contiguous().view(torch.uint8).clone()
                    for name, t in buffers.items()
                }
            )
        finally:
            torch.cuda.synchronize()
            _reset_triton_allocator()
            kv_cache_manager.shutdown()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    for name, loaded in written[0].items():
        assert torch.equal(loaded, written[1][name]), name


def _assert_bitwise_equal(off: torch.Tensor, on: torch.Tensor, what: str) -> None:
    """torch.equal, reporting how many values differ, by how much and at which [token, head]."""
    differ = off != on
    if not bool(differ.any().item()):
        return
    rows = torch.nonzero(differ.reshape(*differ.shape[:2], -1).any(dim=-1))
    max_diff = (off.float() - on.float()).abs().max().item()
    raise AssertionError(
        f"{what}: {int(differ.sum().item())} of {differ.numel()} values differ (max |diff| "
        f"{max_diff:.3g}) in {rows.shape[0]} [token, head] rows, first {rows[:8].tolist()}"
    )


def _assert_packed_decode_switch_is_bitwise(
    monkeypatch,
    num_heads: int,
    query_len_per_seq: int,
    split_kv: Optional[int],
    seq_lens: list[int],
    switch_env: str,
    packed_queries: bool = True,
    enable_helix: bool = False,
    compare_softmax_stats: bool = False,
) -> None:
    _reset_triton_allocator()
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "cutedsl")
    monkeypatch.setenv(FP4_MLA_CUTEDSL_MUFU16_ENV, "1")
    _set_fp4_mla_v_layout_env(monkeypatch, Fp4MlaVLayout.DUAL_COPY)
    monkeypatch.setenv(fp4_mla_backend.FP4_MLA_PACKED_Q_ENV, str(int(packed_queries)))
    monkeypatch.setenv("TRTLLM_FP4_MLA_SPLIT_KV", "auto" if split_kv is None else str(split_kv))
    (
        kv_cache_manager,
        metadata,
        q,
        kv_lora_rank,
        qk_rope_head_dim,
    ) = _build_fp4_mla_attention_decode_case(
        seq_lens=seq_lens,
        num_heads=num_heads,
        seed=43,
        query_len_per_seq=query_len_per_seq,
        enable_helix=enable_helix,
    )
    try:
        outputs = []
        stats = []
        for switch in ("0", "1"):
            monkeypatch.setenv(switch_env, switch)
            output = torch.full_like(q[..., :kv_lora_rank], float("nan"))
            softmax_stats = None
            if compare_softmax_stats:
                softmax_stats = torch.full(
                    (*q.shape[:2], 2), float("nan"), dtype=torch.float32, device=q.device
                )
            run_fp4_mla_attention_decode(
                metadata,
                layer_idx=0,
                local_layer=0,
                q=q,
                output=output,
                sm_scale=0.1,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                prequantized_q=metadata._fp4_mla_prequantized_q,
                prequantized_q_sf=metadata._fp4_mla_prequantized_q_sf,
                q_batch_capacity=metadata._fp4_mla_q_batch_capacity,
                softmax_stats_tensor=softmax_stats,
            )
            torch.cuda.synchronize()
            outputs.append(output)
            stats.append(softmax_stats)
        assert not bool(torch.isnan(outputs[0]).any().item())
        _assert_bitwise_equal(outputs[0], outputs[1], "output")
        if compare_softmax_stats:
            assert not bool(torch.isnan(stats[0]).any().item())
            _assert_bitwise_equal(stats[0], stats[1], "softmax stats")
    finally:
        torch.cuda.synchronize()
        _reset_triton_allocator()
        kv_cache_manager.shutdown()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _fail_v_sidecar_access(*args, **kwargs) -> None:
    raise AssertionError("a single-copy FP4 MLA cache must not read or write a V^T sidecar")


def _written_fp4_mla_cache_bytes(kv_cache_manager, metadata, num_requests: int) -> dict:
    """K, K-scale and V-scale bytes of every request page, in request and page order."""
    kv_cache, sf_cache = kv_cache_manager.get_fp4_mla_cache_buffers(0)
    pages = [
        int(page)
        for request_pages in kv_cache_manager.get_batch_cache_indices(
            list(range(num_requests)), layer_idx=0
        )
        for page in request_pages
        if int(page) >= 0
    ]
    page_ids = torch.tensor(pages, dtype=torch.long, device=kv_cache.device)
    pools = (("k", kv_cache), ("k_sf", sf_cache), ("v_sf", metadata.fp4_mla_v_scale_pool[0]))
    return {name: pool.view(torch.uint8).index_select(0, page_ids).clone() for name, pool in pools}


def _assert_single_copy_decode_is_bitwise(
    monkeypatch,
    num_heads: int,
    query_len_per_seq: int,
    split_kv: int,
    seq_lens: list[int],
    *,
    enable_helix: bool = False,
    compare_softmax_stats: bool = False,
) -> None:
    _reset_triton_allocator()
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "cutedsl")
    monkeypatch.setenv(FP4_MLA_CUTEDSL_MUFU16_ENV, "1")
    monkeypatch.setenv(fp4_mla_backend.FP4_MLA_PACKED_Q_ENV, "1")
    monkeypatch.setenv("TRTLLM_FP4_MLA_SPLIT_KV", str(split_kv))
    results = {}
    for v_layout in (Fp4MlaVLayout.DUAL_COPY, Fp4MlaVLayout.PACKED_SINGLE_COPY):
        single_copy = v_layout is Fp4MlaVLayout.PACKED_SINGLE_COPY
        _set_fp4_mla_v_layout_env(monkeypatch, v_layout)
        if single_copy:
            for name in (
                "_get_cutedsl_persistent_v_packed_cache",
                "_repack_cutedsl_v_packed_cache",
            ):
                monkeypatch.setattr(fp4_mla_backend, name, _fail_v_sidecar_access)
        kv_cache_manager, metadata, q, kv_lora_rank, qk_rope_head_dim = (
            _build_fp4_mla_attention_decode_case(
                seq_lens=seq_lens,
                num_heads=num_heads,
                seed=43,
                query_len_per_seq=query_len_per_seq,
                enable_helix=enable_helix,
            )
        )
        try:
            assert kv_cache_manager.fp4_mla_v_layout is v_layout
            assert (kv_cache_manager.get_mla_v_packed_pool(0) is None) is single_copy
            output = torch.full_like(q[..., :kv_lora_rank], float("nan"))
            softmax_stats = None
            if compare_softmax_stats:
                softmax_stats = torch.full(
                    (*q.shape[:2], 2), float("nan"), dtype=torch.float32, device=q.device
                )
            run_fp4_mla_attention_decode(
                metadata,
                layer_idx=0,
                local_layer=0,
                q=q,
                output=output,
                sm_scale=0.1,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                prequantized_q=metadata._fp4_mla_prequantized_q,
                prequantized_q_sf=metadata._fp4_mla_prequantized_q_sf,
                q_batch_capacity=metadata._fp4_mla_q_batch_capacity,
                softmax_stats_tensor=softmax_stats,
            )
            torch.cuda.synchronize()
            results[v_layout] = (
                output,
                softmax_stats,
                _written_fp4_mla_cache_bytes(kv_cache_manager, metadata, len(seq_lens)),
            )
        finally:
            torch.cuda.synchronize()
            _reset_triton_allocator()
            kv_cache_manager.shutdown()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    dual_output, dual_stats, dual_cache = results[Fp4MlaVLayout.DUAL_COPY]
    single_output, single_stats, single_cache = results[Fp4MlaVLayout.PACKED_SINGLE_COPY]
    for name, written in dual_cache.items():
        assert torch.equal(written, single_cache[name]), f"{name} cache bytes differ"
    assert not bool(torch.isnan(dual_output).any().item())
    _assert_bitwise_equal(dual_output, single_output, "output")
    if compare_softmax_stats:
        assert not bool(torch.isnan(dual_stats).any().item())
        _assert_bitwise_equal(dual_stats, single_stats, "softmax stats")


def _compile_fhadd2_rowsum_probe(num_elements: int, threads: int):
    """Kernel writing, per element, the fused and the widen-and-add FP32 row-sum pairs."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from tensorrt_llm._torch.attention_backend.fp4_mla import fp4_mla_cutedsl_mufu16 as m16

    @cute.kernel
    def probe_kernel(p_pairs: cute.Pointer, p_acc: cute.Pointer, p_out: cute.Pointer):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        idx = bidx * cutlass.Int32(threads) + tidx
        pairs = cute.make_tensor(p_pairs, cute.make_layout((num_elements,)))
        acc = cute.make_tensor(p_acc, cute.make_layout((2 * num_elements,)))
        out = cute.make_tensor(p_out, cute.make_layout((4 * num_elements,)))
        packed = pairs[idx]
        acc0 = acc[cutlass.Int32(2) * idx]
        acc1 = acc[cutlass.Int32(2) * idx + cutlass.Int32(1)]
        fused0, fused1 = m16._add_f16x2_to_f32_pair(acc0, acc1, packed)
        wide0, wide1 = m16._unpack_f16x2_to_f32_pair(packed)
        split = m16.nvvm_add_packed_f32x2((acc0, acc1), (wide0, wide1))
        out[cutlass.Int32(4) * idx] = fused0
        out[cutlass.Int32(4) * idx + cutlass.Int32(1)] = fused1
        out[cutlass.Int32(4) * idx + cutlass.Int32(2)] = split[0]
        out[cutlass.Int32(4) * idx + cutlass.Int32(3)] = split[1]

    @cute.jit
    def probe_host(
        p_pairs: cute.Pointer, p_acc: cute.Pointer, p_out: cute.Pointer, stream: cuda.CUstream
    ):
        probe_kernel(p_pairs, p_acc, p_out).launch(
            grid=(num_elements // threads, 1, 1), block=(threads, 1, 1), stream=stream
        )

    return m16, probe_host


@pytest.mark.skipif(
    _is_cutedsl_unavailable(),
    reason="requires Rubin SM107 and the CuTe DSL runtime",
)
def test_fhadd2_rowsum_add_matches_widen_and_add_bitwise() -> None:
    """add.rn.f32x2.f16x2.f32x2 equals the exact FP16 widening plus add.rn.f32x2.

    Every FP16 bit pattern (subnormals, infinities and NaNs included) in both lanes,
    against FP32 accumulators that cover zeros, subnormals, ties, cancellation and
    overflow. NaN results only need to be NaN on both paths.
    """
    from cutlass import Float32, Int32
    from cutlass.cute import AddressSpace
    from cutlass.cute.runtime import make_ptr

    threads = 256
    acc_values = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        6.0,
        -6.0,
        0.75,
        1.0e-3,
        0.1,
        3.0,
        2.0**-24,
        2.0**-14,
        2.0**-126,
        1.0e-40,
        -1.0e-40,
        2.0**24,
        2.0**24 + 2.0,
        1.0 + 2.0**-23,
        123456.789,
        65504.0,
        -65504.0,
        3.0e38,
        -3.0e38,
        3.4028234663852886e38,
        float("inf"),
        float("-inf"),
        float("nan"),
        1.5,
        2.0**-10,
        7.0e-8,
        5.9604645e-08,
        1024.0,
    ]
    lo = torch.arange(1 << 16, dtype=torch.int64)
    hi = (lo * 40503) & 0xFFFF
    words = lo | (hi << 16)
    words = torch.where(words >= (1 << 31), words - (1 << 32), words).to(torch.int32)
    acc_pairs = torch.tensor(
        [[acc_values[j], acc_values[(j + 7) % len(acc_values)]] for j in range(len(acc_values))],
        dtype=torch.float32,
    )
    pairs = words.repeat(len(acc_values)).cuda()
    acc = acc_pairs.repeat_interleave(words.numel(), dim=0).contiguous().cuda()
    num_elements = pairs.numel()
    out = torch.empty((num_elements, 4), dtype=torch.float32, device="cuda")
    m16, probe_host = _compile_fhadd2_rowsum_probe(num_elements, threads)
    gmem = AddressSpace.gmem
    ptrs = (
        make_ptr(Int32, pairs.data_ptr(), gmem, assumed_align=16),
        make_ptr(Float32, acc.data_ptr(), gmem, assumed_align=16),
        make_ptr(Float32, out.data_ptr(), gmem, assumed_align=16),
    )
    stream = m16._current_cu_stream()
    probe = m16._compile_cutedsl(probe_host, *ptrs, stream)
    probe(*ptrs, stream)
    torch.cuda.synchronize()
    fused, split = out[:, :2].contiguous(), out[:, 2:].contiguous()
    same_bits = fused.view(torch.int32) == split.view(torch.int32)
    both_nan = torch.isnan(fused) & torch.isnan(split)
    mismatches = int((~(same_bits | both_nan)).sum().item())
    assert mismatches == 0, f"{mismatches} of {2 * num_elements} lanes differ"
    # The sweep reaches the corners it is meant to: FP16 subnormal inputs and overflow.
    assert bool(torch.isinf(split).any().item()) and bool((split == 0).any().item())


@pytest.mark.skipif(
    _is_cutedsl_unavailable(),
    reason="requires Rubin SM107 and the CuTe DSL runtime",
)
def test_fp4_mla_attention_decode_cutedsl_block_reuse_repack_matches_reference(
    monkeypatch,
) -> None:
    original_repack = fp4_mla_backend._repack_cutedsl_v_packed_cache
    repack_calls = []

    def record_repack(*args, **kwargs) -> None:
        original_repack(*args, **kwargs)
        repack_calls.append(
            {
                "page_indptr": kwargs["page_indptr"].detach().cpu().clone(),
                "kv_lens": kwargs["kv_lens"].detach().cpu().clone(),
                "generation_lens": kwargs["generation_lens"].detach().cpu().clone(),
                "max_touched_pages": kwargs["max_touched_pages"],
            }
        )

    monkeypatch.setattr(fp4_mla_backend, "_repack_cutedsl_v_packed_cache", record_repack)

    _assert_fp4_mla_attention_decode_accuracy(
        monkeypatch,
        backend="cutedsl",
        fused_v_transpose=False,
        num_heads=128,
        seq_lens=[131, 512],
        seed=29,
        query_len_per_seq=1,
        enable_block_reuse=True,
    )

    assert len(repack_calls) == 2
    expected_indptr = torch.tensor([0, 4, 8], dtype=torch.int32)
    torch.testing.assert_close(repack_calls[0]["page_indptr"], expected_indptr)
    torch.testing.assert_close(repack_calls[1]["page_indptr"], expected_indptr)
    torch.testing.assert_close(
        repack_calls[0]["kv_lens"],
        torch.tensor([130, 511], dtype=torch.int32),
    )
    torch.testing.assert_close(
        repack_calls[0]["generation_lens"],
        torch.tensor([130, 511], dtype=torch.int32),
    )
    torch.testing.assert_close(
        repack_calls[1]["kv_lens"],
        torch.tensor([131, 512], dtype=torch.int32),
    )
    torch.testing.assert_close(
        repack_calls[1]["generation_lens"],
        torch.ones(2, dtype=torch.int32),
    )
    assert repack_calls[0]["max_touched_pages"] == 4
    assert repack_calls[1]["max_touched_pages"] == 1


def test_fp4_mla_context_repack_touched_pages_follow_the_chunk_not_the_prefix() -> None:
    touched_pages = fp4_mla_backend._fp4_mla_context_repack_touched_pages
    cold = torch.tensor([130, 511], dtype=torch.int32)
    assert touched_pages(cold, cold.clone()) == 4
    # A 3000-token chunk appends to 24-25 pages after any prefix: one compile bucket.
    chunk = torch.tensor([3000], dtype=torch.int32)
    for prefix in (37000, 37100, 40000, 100003):
        kv_lens = torch.tensor([prefix + 3000], dtype=torch.int32)
        assert touched_pages(kv_lens, chunk) == 32
    # The bound covers every sequence's page window, including a start mid-page.
    generator = torch.Generator().manual_seed(20261020)
    append_lens = torch.randint(1, 20000, (64,), generator=generator, dtype=torch.int32)
    kv_lens = append_lens + torch.randint(0, 200000, (64,), generator=generator, dtype=torch.int32)
    bound = touched_pages(kv_lens, append_lens)
    windows = (kv_lens - 1) // 128 - (kv_lens - append_lens) // 128 + 1
    assert bound >= int(windows.max()) and bound & (bound - 1) == 0
    assert touched_pages(torch.tensor([129]), torch.tensor([2])) == 2
    assert touched_pages(torch.tensor([128]), torch.tensor([1])) == 1


def test_fp4_mla_context_repack_bucket_limit_covers_any_forward() -> None:
    limit = fp4_mla_backend._fp4_mla_context_repack_bucket_limit
    touched_pages = fp4_mla_backend._fp4_mla_context_repack_touched_pages
    assert limit(16384) == 256
    assert limit(128) == 2
    assert limit(1) == 2
    # An append of up to max_num_tokens tokens after any prefix fits under the limit.
    generator = torch.Generator().manual_seed(20261020)
    for max_num_tokens in (1, 127, 128, 129, 8192, 16384):
        append_lens = torch.randint(
            1, max_num_tokens + 1, (256,), generator=generator, dtype=torch.int32
        )
        kv_lens = append_lens + torch.randint(
            0, 300000, (256,), generator=generator, dtype=torch.int32
        )
        assert touched_pages(kv_lens, append_lens) <= limit(max_num_tokens)


def test_fp4_mla_context_lead_tile_launch_runs_only_for_a_start_inside_a_tile() -> None:
    resumes_inside_tile = fp4_mla_backend._fp4_mla_context_resumes_inside_tile
    # Starts 0 and 3968: every context begins on a tile boundary.
    assert not resumes_inside_tile(torch.tensor([300, 4096]), torch.tensor([300, 128]))
    # Starts 0 and 37: the second context leads its tile with ring tokens.
    assert resumes_inside_tile(torch.tensor([300, 1037]), torch.tensor([300, 1000]))
    assert not resumes_inside_tile(torch.tensor([129]), torch.tensor([1]))
    assert resumes_inside_tile(torch.tensor([130], dtype=torch.int32), torch.tensor([1]))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="requires Rubin SM107",
)
@pytest.mark.parametrize(
    ("seq_len", "expected_slot"),
    [(17, 16), (33, 13)],
    ids=["tile-boundary", "ring-wrap"],
)
def test_fp4_mla_context_tail_uses_draft_slack_ring(
    monkeypatch,
    seq_len: int,
    expected_slot: int,
) -> None:
    _reset_triton_allocator()
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "triton")
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    head_dim = kv_lora_rank + qk_rope_head_dim
    ring_size = HP_BLOCK_SIZE + 3
    spec_config = MTPDecodingConfig(max_draft_len=3)
    kv_cache_manager = _create_fp4_mla_v2_manager(
        max_tokens=FP4_MLA_TOKENS_PER_BLOCK,
        max_seq_len=FP4_MLA_TOKENS_PER_BLOCK,
        max_batch_size=1,
        spec_config=spec_config,
    )
    try:
        kv_cache_manager.add_dummy_requests([0], [seq_len])
        metadata = _build_multi_seq_metadata(
            kv_cache_manager,
            seq_lens=[seq_len],
            page_size=FP4_MLA_TOKENS_PER_BLOCK,
        )
        latent = torch.randn(
            seq_len,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )

        scatter_fp4_mla_kv_cache(
            metadata,
            latent,
            layer_idx=0,
            token_offset=0,
            phase="context",
            local_layer=0,
            v_head_dim=kv_lora_rank,
        )
        torch.cuda.synchronize()

        hp_page = int(metadata._fp4_mla_hp_page_indices[0].item())
        hp_ring = metadata.high_precision_kv_pool.view(
            metadata.high_precision_kv_pool.shape[0], 1, 1, ring_size, head_dim
        )
        torch.testing.assert_close(
            hp_ring[hp_page, 0, 0, expected_slot],
            latent[-1],
            rtol=0,
            atol=0,
        )
        assert torch.count_nonzero(hp_ring[hp_page, 0, 0, 0]).item() == 0
    finally:
        torch.cuda.synchronize()
        _reset_triton_allocator()
        kv_cache_manager.shutdown()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="requires Rubin SM107",
)
def test_fp4_mla_context_scatter_fuses_qk_rope(monkeypatch) -> None:
    _reset_triton_allocator()
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "triton")
    seq_len = 17
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    num_heads = 4
    head_dim = kv_lora_rank + qk_rope_head_dim
    kv_cache_manager = _create_fp4_mla_v2_manager(
        max_tokens=FP4_MLA_TOKENS_PER_BLOCK,
        max_seq_len=FP4_MLA_TOKENS_PER_BLOCK,
        max_batch_size=1,
    )
    try:
        kv_cache_manager.add_dummy_requests([0], [seq_len])
        metadata = _build_multi_seq_metadata(
            kv_cache_manager,
            seq_lens=[seq_len],
            page_size=FP4_MLA_TOKENS_PER_BLOCK,
        )
        latent = torch.randn(
            seq_len,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        q = torch.randn(
            seq_len,
            num_heads * (qk_nope_head_dim + qk_rope_head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        original_latent = latent.clone()
        original_q = q.clone().view(
            seq_len,
            num_heads,
            qk_nope_head_dim + qk_rope_head_dim,
        )
        rotary_cos_sin = torch.zeros(
            (FP4_MLA_TOKENS_PER_BLOCK, qk_rope_head_dim, 2),
            dtype=torch.float32,
            device="cuda",
        )
        rotary_cos_sin[..., 1] = 1.0

        scatter_fp4_mla_kv_cache(
            metadata,
            latent,
            layer_idx=0,
            token_offset=0,
            phase="context",
            local_layer=0,
            v_head_dim=kv_lora_rank,
            rotary_cos_sin=rotary_cos_sin,
            q_context=q,
            q_nope_head_dim=qk_nope_head_dim,
        )
        torch.cuda.synchronize()

        expected_k_pe = original_latent[:, kv_lora_rank:].reshape(
            seq_len,
            qk_rope_head_dim // 2,
            2,
        )
        expected_k_pe = torch.stack(
            (-expected_k_pe[..., 1], expected_k_pe[..., 0]),
            dim=-1,
        ).flatten(1)
        q_view = q.view(
            seq_len,
            num_heads,
            qk_nope_head_dim + qk_rope_head_dim,
        )
        expected_q_pe = original_q[..., qk_nope_head_dim:].reshape(
            seq_len,
            num_heads,
            qk_rope_head_dim // 2,
            2,
        )
        expected_q_pe = torch.stack(
            (-expected_q_pe[..., 1], expected_q_pe[..., 0]),
            dim=-1,
        ).flatten(-2)
        torch.testing.assert_close(
            latent[:, :kv_lora_rank],
            original_latent[:, :kv_lora_rank],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            latent[:, kv_lora_rank:],
            expected_k_pe,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            q_view[..., :qk_nope_head_dim],
            original_q[..., :qk_nope_head_dim],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            q_view[..., qk_nope_head_dim:],
            expected_q_pe,
            rtol=0,
            atol=0,
        )
    finally:
        torch.cuda.synchronize()
        _reset_triton_allocator()
        kv_cache_manager.shutdown()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


_V_REPACK_PAGE_SIZE = 128
_V_REPACK_HEAD_DIM = 512
_V_REPACK_PACKED_DIM = _V_REPACK_HEAD_DIM // 2


def _random_v_repack_production_cache(num_pages: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(20260817)
    return torch.randint(
        0,
        256,
        (num_pages, 1, _V_REPACK_PAGE_SIZE, 1, _V_REPACK_PACKED_DIM),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )


def _v_repack_sentinel_output(num_pages: int, sentinel: int) -> torch.Tensor:
    return torch.full(
        (num_pages * _V_REPACK_HEAD_DIM, _V_REPACK_PAGE_SIZE // 2),
        sentinel,
        dtype=torch.uint8,
        device="cuda",
    )


def _resolve_v_repack_generation_page_ids(
    page_ids: list[int],
    page_indptr: list[int],
    kv_lens: list[int],
    generation_lens: list[int],
    *,
    num_pages: int,
    max_touched_pages: int,
) -> list[int]:
    resolved = []
    for sequence_idx, (kv_len, generation_len) in enumerate(zip(kv_lens, generation_lens)):
        if kv_len <= 0 or generation_len <= 0:
            continue
        first_page = max(kv_len - generation_len, 0) // _V_REPACK_PAGE_SIZE
        last_page = (kv_len - 1) // _V_REPACK_PAGE_SIZE
        page_begin = page_indptr[sequence_idx]
        page_count = page_indptr[sequence_idx + 1] - page_begin
        for touched_page in range(max_touched_pages):
            logical_page = first_page + touched_page
            if logical_page > last_page or logical_page < 0 or logical_page >= page_count:
                continue
            physical_page = page_ids[page_begin + logical_page]
            if 0 <= physical_page < num_pages:
                resolved.append(physical_page)
    return resolved


@pytest.mark.skipif(
    _is_cutedsl_unavailable(),
    reason="requires Rubin SM107 and the CuTe DSL runtime",
)
def test_fp4_mla_v_repack_full_pages_matches_reference() -> None:
    from tensorrt_llm._torch.attention_backend.fp4_mla import fp4_mla_cutedsl_v_repack

    num_pages = 3
    kv_cache = _random_v_repack_production_cache(num_pages)
    actual = _v_repack_sentinel_output(num_pages, sentinel=0xA5)
    expected = actual.clone()

    fp4_mla_cutedsl_v_repack.fp4_mla_repack_v_cache_reference(
        expected,
        kv_cache,
        v_head_dim=_V_REPACK_HEAD_DIM,
        page_size=_V_REPACK_PAGE_SIZE,
    )
    fp4_mla_cutedsl_v_repack.fp4_mla_repack_v_cache(
        actual,
        kv_cache,
        v_head_dim=_V_REPACK_HEAD_DIM,
        page_size=_V_REPACK_PAGE_SIZE,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    _is_cutedsl_unavailable(),
    reason="requires Rubin SM107 and the CuTe DSL runtime",
)
def test_fp4_mla_v_repack_selected_pages_preserves_untouched_page() -> None:
    from tensorrt_llm._torch.attention_backend.fp4_mla import fp4_mla_cutedsl_v_repack

    num_pages = 3
    sentinel = 0x5A
    kv_cache = _random_v_repack_production_cache(num_pages)
    page_ids = torch.tensor([2, 0], dtype=torch.int32, device="cuda")
    actual = _v_repack_sentinel_output(num_pages, sentinel)
    expected = actual.clone()

    fp4_mla_cutedsl_v_repack.fp4_mla_repack_v_cache_reference(
        expected,
        kv_cache,
        page_ids,
        v_head_dim=_V_REPACK_HEAD_DIM,
        page_size=_V_REPACK_PAGE_SIZE,
    )
    fp4_mla_cutedsl_v_repack.fp4_mla_repack_v_cache(
        actual,
        kv_cache,
        page_ids,
        v_head_dim=_V_REPACK_HEAD_DIM,
        page_size=_V_REPACK_PAGE_SIZE,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)
    assert torch.all(actual[_V_REPACK_HEAD_DIM : 2 * _V_REPACK_HEAD_DIM] == sentinel)


@pytest.mark.skipif(
    _is_cutedsl_unavailable(),
    reason="requires Rubin SM107 and the CuTe DSL runtime",
)
def test_fp4_mla_v_repack_generation_csr_crosses_page_boundary() -> None:
    from tensorrt_llm._torch.attention_backend.fp4_mla import fp4_mla_cutedsl_v_repack

    num_pages = 4
    max_touched_pages = 2
    sentinel = 0xC3
    page_ids_host = [2, 0, 1]
    page_indptr_host = [0, 2, 3]
    kv_lens_host = [130, 128]
    generation_lens_host = [4, 0]
    resolved_page_ids_host = _resolve_v_repack_generation_page_ids(
        page_ids_host,
        page_indptr_host,
        kv_lens_host,
        generation_lens_host,
        num_pages=num_pages,
        max_touched_pages=max_touched_pages,
    )
    assert resolved_page_ids_host == [2, 0]

    kv_cache = _random_v_repack_production_cache(num_pages)
    page_ids = torch.tensor(page_ids_host, dtype=torch.int32, device="cuda")
    page_indptr = torch.tensor(page_indptr_host, dtype=torch.int32, device="cuda")
    kv_lens = torch.tensor(kv_lens_host, dtype=torch.int32, device="cuda")
    generation_lens = torch.tensor(generation_lens_host, dtype=torch.int32, device="cuda")
    resolved_page_ids = torch.tensor(
        resolved_page_ids_host,
        dtype=torch.int32,
        device="cuda",
    )
    actual = _v_repack_sentinel_output(num_pages, sentinel)
    expected = actual.clone()

    fp4_mla_cutedsl_v_repack.fp4_mla_repack_v_cache_reference(
        expected,
        kv_cache,
        resolved_page_ids,
        v_head_dim=_V_REPACK_HEAD_DIM,
        page_size=_V_REPACK_PAGE_SIZE,
    )
    fp4_mla_cutedsl_v_repack.fp4_mla_repack_v_cache(
        actual,
        kv_cache,
        page_ids,
        v_head_dim=_V_REPACK_HEAD_DIM,
        page_size=_V_REPACK_PAGE_SIZE,
        page_indptr=page_indptr,
        kv_lens=kv_lens,
        generation_lens=generation_lens,
        max_touched_pages=max_touched_pages,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)
    assert torch.all(actual[_V_REPACK_HEAD_DIM : 2 * _V_REPACK_HEAD_DIM] == sentinel)
    assert torch.all(actual[3 * _V_REPACK_HEAD_DIM : 4 * _V_REPACK_HEAD_DIM] == sentinel)


@pytest.mark.skipif(
    _is_cutedsl_unavailable(),
    reason="requires Rubin SM107 and the CuTe DSL runtime",
)
def test_fp4_mla_v_repack_precompile_compiles_every_bound_and_keeps_the_output(
    monkeypatch,
) -> None:
    from tensorrt_llm._torch.attention_backend.fp4_mla import fp4_mla_cutedsl_v_repack

    compile_cache = {}
    monkeypatch.setattr(fp4_mla_cutedsl_v_repack, "_V_REPACK_COMPILE_CACHE", compile_cache)
    num_pages = 4
    max_touched_pages = 2
    kv_cache = _random_v_repack_production_cache(num_pages)
    page_ids = torch.tensor([2, 0, 1], dtype=torch.int32, device="cuda")
    page_indptr = torch.tensor([0, 2, 3], dtype=torch.int32, device="cuda")
    kv_lens = torch.tensor([130, 128], dtype=torch.int32, device="cuda")
    generation_lens = torch.tensor([4, 0], dtype=torch.int32, device="cuda")
    resolved_page_ids = torch.tensor([2, 0], dtype=torch.int32, device="cuda")
    actual = _v_repack_sentinel_output(num_pages, 0xC3)
    expected = actual.clone()

    fp4_mla_cutedsl_v_repack.fp4_mla_repack_v_cache_reference(
        expected,
        kv_cache,
        resolved_page_ids,
        v_head_dim=_V_REPACK_HEAD_DIM,
        page_size=_V_REPACK_PAGE_SIZE,
    )
    fp4_mla_cutedsl_v_repack.fp4_mla_repack_v_cache(
        actual,
        kv_cache,
        page_ids,
        v_head_dim=_V_REPACK_HEAD_DIM,
        page_size=_V_REPACK_PAGE_SIZE,
        page_indptr=page_indptr,
        kv_lens=kv_lens,
        generation_lens=generation_lens,
        max_touched_pages=max_touched_pages,
        precompile_max_touched_pages=8,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)
    # Cache key: (page_size, v_head_dim, use_page_ids, resolve_generation_pages, bound, ...).
    assert sorted(key[4] for key in compile_cache if key[3]) == [1, 2, 4, 8]


@pytest.mark.skipif(_is_cutedsl_unavailable(), reason="MXFP8 CTX gather requires Rubin CuTe DSL")
@pytest.mark.parametrize(
    "lengths, offsets, global_scale, large_page_stride",
    [
        ([97, 32], [0, 127], 32.0, False),
        ([13, 0, 117], [127, 1, 255], 128.0, False),
        ([128], [0], 1.0, False),
        ([128], [0], 1.0, True),
    ],
)
def test_fp4_mla_context_mxfp8_gather(
    lengths, offsets, global_scale, large_page_stride, monkeypatch
):
    """Validate the reader; direct cache bytes do not test scatter or reuse lifecycle.

    Power-of-two global scales make division exact, so the independent PyTorch
    reference is bitwise comparable despite compiler FFMA contraction. Arbitrary
    scales require a reference that preserves the original PTX expression.
    """
    seq_lens = [length + offset for length, offset in zip(lengths, offsets)]
    manager = _create_fp4_mla_v2_manager(
        max_tokens=65536,
        max_seq_len=max(seq_lens),
        max_batch_size=len(lengths),
        enable_block_reuse=True,
    )
    try:
        assert manager.enable_block_reuse
        manager.add_dummy_requests(list(range(len(lengths))), seq_lens)
        assert all(index in manager.kv_cache_map for index in range(len(lengths)))
        metadata = _build_multi_seq_metadata(manager, seq_lens=seq_lens, page_size=128)
        metadata._fp4_mla_kv_global_scale.fill_(global_scale)
        kv, sf = manager.get_fp4_mla_cache_buffers(0)
        if large_page_stride:
            # Exercise both cache and SF scalar reads beyond signed 32-bit
            # offsets without filling the unused gap between two pages.
            kv = torch.empty_strided(
                (2, *kv.shape[1:]),
                ((1 << 31) + kv.stride(0), *kv.stride()[1:]),
                dtype=kv.dtype,
                device=kv.device,
            )
            sf = torch.empty_strided(
                (2, *sf.shape[1:]),
                ((1 << 31) + sf.stride(0), *sf.stride()[1:]),
                dtype=sf.dtype,
                device=sf.device,
            )
            monkeypatch.setattr(manager, "get_fp4_mla_cache_buffers", lambda _: (kv, sf))
        torch.manual_seed(107)
        kv.random_(0, 256)
        sf.view(torch.uint8).copy_(
            torch.randint(1, 127, sf.shape, dtype=torch.uint8, device="cuda")
        )
        # Native allocation remains active, but use nonidentity physical pages.
        metadata._paged_kv_indices.copy_(kv.shape[0] - 1 - metadata._paged_kv_indices)
        if large_page_stride:
            metadata._paged_kv_indices.fill_(1)
            assert kv.stride(0) >= 1 << 31 and sf.stride(0) >= 1 << 31
        row_ids, physical_ids = [], []
        for batch, (length, offset) in enumerate(zip(lengths, offsets)):
            positions = torch.arange(offset, offset + length, device="cuda")
            row_ids.append(positions % 128)
            physical_ids.append(
                metadata._paged_kv_indices[
                    batch * metadata.fp4_mla_page_table_stride + positions // 128
                ].long()
            )
        rows, physical = torch.cat(row_ids), torch.cat(physical_ids)
        page, row = int(physical[0]), int(rows[0])
        kv[page, 0, row, 0, :256] = 0  # Floor scale, including signed-zero inputs elsewhere.
        kv[page, 0, row, 0, 256] = 0x22
        kv[page, 0, row, 0, 288] = 0xAA
        sf_bytes = sf[page].view(torch.uint8).flatten()
        for col in (32, 36):
            sf_bytes[_swizzled_sf_offset(row, col, 40)] = 56
        # Saturation and near cancellation test rounding after adding the residual.
        kv[page, 0, row, 0, 264] = 0xF7
        kv[page, 0, row, 0, 296] = 0
        sf_bytes[_swizzled_sf_offset(row, 33, 40)] = 126
        kv[page, 0, row, 0, 272] = 0x55
        kv[page, 0, row, 0, 304] = 0xDD
        sf_bytes[_swizzled_sf_offset(row, 34, 40)] = 56
        sf_bytes[_swizzled_sf_offset(row, 38, 40)] = 55

        packed = kv[physical, 0, rows, 0, :320]
        codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(-1, 640).long()
        lut = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            device="cuda",
        )
        sf_indices = _swizzled_sf_offset(rows[:, None], torch.arange(40, device="cuda"), 40)
        sf_values = sf.view(torch.float8_e4m3fn).reshape(sf.shape[0], -1).float()
        scales = sf_values[physical[:, None], sf_indices].repeat_interleave(16, dim=1)
        # Keep FP32 division per term; a reciprocal multiply changes rounding.
        values = lut[codes] * scales / metadata._fp4_mla_kv_global_scale
        dense = values[:, :512].to(torch.bfloat16).contiguous()
        rope = (values[:, 512:576] + values[:, 576:]).to(torch.bfloat16).contiguous()
        expected_a, expected_sf = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(dense)
        expected_pe, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
            rope, torch.ones(1, device="cuda")
        )

        cu = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64, device="cuda"
        )
        chunk_offsets = torch.tensor(offsets, dtype=torch.int64, device="cuda")
        # Capacity exceeds active contexts; sentinels must never be dereferenced.
        metadata._paged_kv_indices = torch.cat(
            (
                metadata._paged_kv_indices,
                torch.full(
                    (3 * metadata.fp4_mla_page_table_stride,), -1, dtype=torch.int32, device="cuda"
                ),
            )
        )
        cu = torch.cat((cu, torch.full((3,), -1, dtype=torch.int64, device="cuda")))
        chunk_offsets = torch.cat(
            (chunk_offsets, torch.full((3,), -1, dtype=torch.int64, device="cuda"))
        )
        (actual_a, actual_sf), actual_pe = load_fp4_mla_chunked_kv_cache(
            metadata,
            0,
            num_ctx_cached_tokens=sum(lengths),
            cu_chunked_seq_len=cu,
            chunked_global_offset=chunk_offsets,
            chunked_max_seq_len=max(lengths),
            out_dtype=torch.bfloat16,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            output_format="mxfp8",
        )
        for expected, actual in (
            (expected_a, actual_a),
            (expected_sf, actual_sf),
            (expected_pe, actual_pe),
        ):
            # Includes every padded SF byte, not just rows consumed by the GEMM.
            assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
    finally:
        torch.cuda.synchronize()
        manager.shutdown()


@pytest.mark.parametrize("bias_kind", ["missing", "none", "present"])
def test_mla_strided_context_projection_optional_bias(monkeypatch, bias_kind):
    projection = SimpleNamespace(weight=torch.empty(16, 512, dtype=torch.bfloat16))
    if bias_kind != "missing":
        projection.bias = None if bias_kind == "none" else torch.zeros(16)
    module = SimpleNamespace(
        kv_b_proj=projection,
        num_heads_tp=1,
        qk_nope_head_dim=8,
        v_head_dim=8,
        kv_lora_rank=512,
    )
    monkeypatch.setattr(mla_module, "is_sm_100f", lambda: True)
    assert MLA._use_kvbproj_strided_out(module) == (bias_kind != "present")


@pytest.mark.parametrize("output_format", ["dense", "mxfp8"])
@pytest.mark.parametrize("qk_rope_head_dim", [32, 128])
def test_fp4_mla_context_gather_rejects_unsupported_rope_dim(output_format, qk_rope_head_dim):
    # Reject the shape before accessing metadata or allocating CUDA output.
    with pytest.raises(ValueError, match="qk_rope_head_dim=64"):
        load_fp4_mla_chunked_kv_cache(
            None,
            0,
            num_ctx_cached_tokens=1,
            cu_chunked_seq_len=None,
            chunked_global_offset=None,
            chunked_max_seq_len=1,
            out_dtype=torch.bfloat16,
            kv_lora_rank=512,
            qk_rope_head_dim=qk_rope_head_dim,
            output_format=output_format,
        )
