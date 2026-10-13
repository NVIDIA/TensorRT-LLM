# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRTLLM FP4 MLA helper tests."""

from types import SimpleNamespace

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
    FP4_MLA_K_RESIDUAL_DIM,
    FP4_MLA_KV_GLOBAL_SCALE,
    FP4_MLA_Q_GLOBAL_SCALE,
    FP4_MLA_TOKENS_PER_BLOCK,
    HP_BLOCK_SIZE,
    _get_fp4_mla_global_scale,
    load_fp4_mla_chunked_kv_cache,
    scatter_fp4_mla_kv_cache,
)
from tensorrt_llm._torch.attention_backend.fp4_mla.cache_manager import Fp4MlaKVCacheManagerV2
from tensorrt_llm._torch.attention_backend.fp4_mla.fp4_mla_context import (
    _build_fp8_mla_context_metadata,
)
from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs, AttentionInputType
from tensorrt_llm._torch.kimi_k3_cache_policy import KIMI_K3_BF16_KV_LAYERS_ENV
from tensorrt_llm._torch.modules.mla import MLA
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import Role
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


def test_fp4_mla_decode_rejects_helix_softmax_stats() -> None:
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

    with pytest.raises(
        NotImplementedError,
        match="does not produce Helix softmax stats",
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
    monkeypatch.delenv(KIMI_K3_BF16_KV_LAYERS_ENV, raising=False)
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


def test_fp4_mla_v2_cache_size_accounts_for_bf16_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "triton")
    monkeypatch.setenv(KIMI_K3_BF16_KV_LAYERS_ENV, "3")
    model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(
            model_type="kimi_linear",
            num_hidden_layers=4,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            linear_attn_config={"full_attn_layers": [2, 4]},
        )
    )

    slope, intercept = Fp4MlaKVCacheManagerV2.get_cache_size_per_token(
        model_config,
        Mapping(world_size=1, tp_size=1, pp_size=1, rank=0),
        num_layers=2,
        tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
        max_batch_size=3,
        spec_config=SimpleNamespace(tokens_per_gen_step=4),
    )

    fp4_bytes_per_token = 320 + 40 + 32
    assert slope == fp4_bytes_per_token + 576 * torch.bfloat16.itemsize
    assert intercept == 3 * (HP_BLOCK_SIZE + 3) * 576 * 2


@pytest.mark.parametrize("rank, num_bf16_layers", [(0, 0), (1, 2)])
def test_fp4_mla_v2_default_bf16_tail_sizing_uses_global_layers(
    monkeypatch: pytest.MonkeyPatch, rank: int, num_bf16_layers: int
) -> None:
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "triton")
    monkeypatch.delenv(KIMI_K3_BF16_KV_LAYERS_ENV, raising=False)
    model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(
            model_type="kimi_k3",
            num_hidden_layers=16,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            linear_attn_config={"full_attn_layers": list(range(2, 17, 2))},
        )
    )
    slope, intercept = Fp4MlaKVCacheManagerV2.get_cache_size_per_token(
        model_config,
        Mapping(world_size=2, tp_size=1, pp_size=2, rank=rank),
        num_layers=4,
        tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
        max_batch_size=3,
        spec_config=SimpleNamespace(tokens_per_gen_step=4),
    )

    num_fp4_layers = 4 - num_bf16_layers
    assert slope == num_fp4_layers * (320 + 40 + 32) + num_bf16_layers * 576 * 2
    assert intercept == 3 * 2 * num_fp4_layers * (HP_BLOCK_SIZE + 3) * 576 * 2


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


def test_fp4_mla_disagg_import_rebuilds_v_scales(monkeypatch) -> None:
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
    staged = []
    scale_rebuilds = []

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

    manager = SimpleNamespace(
        dtype=_DataType.NVFP4,
        kv_factor=1,
        mla_v_scale_head_dim=512,
        mla_v_head_dim=None,
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
    monkeypatch.setattr(fp4_mla_backend, "_fp4_mla_attention_backend", lambda: "triton")
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


def _dequant_fp4_swizzled(
    fp4_tensor: torch.Tensor,
    sf_tensor: torch.Tensor,
    *,
    logical_dim: int,
    sf_per_token: int,
    global_scale: float,
) -> torch.Tensor:
    fp4_values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=fp4_tensor.device,
    )
    fp4_bytes = fp4_tensor.view(torch.uint8)
    sf_flat = sf_tensor.view(torch.float8_e4m3fn).reshape(-1)
    out = torch.empty(
        (fp4_bytes.shape[0], logical_dim),
        dtype=torch.float32,
        device=fp4_tensor.device,
    )

    for row_idx in range(fp4_bytes.shape[0]):
        for sf_col in range(sf_per_token):
            start = sf_col * FP4_BLOCK_SIZE
            packed = fp4_bytes[row_idx, start // 2 : start // 2 + 8]
            low = packed & 0x0F
            high = (packed >> 4) & 0x0F
            vals = torch.empty(FP4_BLOCK_SIZE, dtype=torch.float32, device=fp4_tensor.device)
            low_sign = torch.where(
                (low & 0x08) != 0,
                -torch.ones_like(low, dtype=torch.float32),
                torch.ones_like(low, dtype=torch.float32),
            )
            high_sign = torch.where(
                (high & 0x08) != 0,
                -torch.ones_like(high, dtype=torch.float32),
                torch.ones_like(high, dtype=torch.float32),
            )
            vals[0::2] = fp4_values[(low & 0x07).long()] * low_sign
            vals[1::2] = fp4_values[(high & 0x07).long()] * high_sign
            sf_offset = _swizzled_sf_offset(row_idx, sf_col, sf_per_token)
            out[row_idx, start : start + FP4_BLOCK_SIZE] = (
                vals * sf_flat[sf_offset].float() / global_scale
            )

    return out


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

    pages = []
    dequantized_pages = {}
    for page_id in generation_page_rows.reshape(-1).cpu().tolist():
        assert 0 <= page_id < kv_cache.shape[0]
        if page_id not in dequantized_pages:
            fp4_page = kv_cache[page_id, 0, :, 0, :]
            sf_page = sf_cache[page_id]
            dequantized_pages[page_id] = _dequant_fp4_swizzled(
                fp4_page,
                sf_page,
                logical_dim=storage_head_dim,
                sf_per_token=storage_head_dim // FP4_BLOCK_SIZE,
                global_scale=static_global_scale,
            )
        pages.append(dequantized_pages[page_id])
    if num_generation_sequences == 0:
        return torch.empty(
            (0, metadata.page_size, storage_head_dim),
            dtype=torch.float32,
            device=kv_cache.device,
        )
    storage = torch.stack(pages, dim=0)
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


@pytest.mark.parametrize("qk_rope_head_dim", [32, 128])
def test_fp4_mla_context_gather_rejects_unsupported_rope_dim(qk_rope_head_dim):
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
        )
