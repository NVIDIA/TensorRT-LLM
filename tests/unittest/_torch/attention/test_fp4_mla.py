# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRTLLM FP4 MLA helper tests."""

from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm
import tensorrt_llm._torch.attention.backends.fmha.fp4_mla as fp4_mla_fmha_backend
import tensorrt_llm._torch.attention.backends.fp4_mla as fp4_mla_backend
import tensorrt_llm._torch.attention.backends.fp4_mla.cache_update as fp4_mla_cache_update
import tensorrt_llm._torch.attention.backends.fp4_mla.decode as fp4_mla_decode
import tensorrt_llm._torch.attention.backends.fp4_mla.metadata as fp4_mla_metadata
import tensorrt_llm._torch.attention.backends.fp4_mla.v_cache as fp4_mla_v_cache
import tensorrt_llm._torch.attention.backends.trtllm as trtllm_backend
from tensorrt_llm._torch.attention.backends.fmha.fp4_mla import Fp4MlaFmha
from tensorrt_llm._torch.attention.backends.fp4_mla import (
    FP4_BLOCK_SIZE,
    FP4_MLA_ATTENTION_BACKEND_ENV,
    FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV,
    FP4_MLA_K_RESIDUAL_DIM,
    FP4_MLA_KV_GLOBAL_SCALE,
    FP4_MLA_P_GLOBAL_SCALE,
    FP4_MLA_Q_GLOBAL_SCALE,
    FP4_MLA_Q_RESIDUAL_DIM,
    FP4_MLA_TOKENS_PER_BLOCK,
    HP_BLOCK_SIZE,
    _cutedsl_backend_available,
    _fp4_mla_attention_backend,
    _get_fp4_mla_global_scale,
    load_fp4_mla_chunked_kv_cache,
    run_fp4_mla_attention_decode,
    scatter_fp4_mla_kv_cache,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.cache_manager import Fp4MlaKVCacheManagerV2
from tensorrt_llm._torch.attention.backends.fp4_mla.fp4_mla_context import (
    _build_fp8_mla_context_metadata,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.state import Fp4MlaState
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention.mla import MLA
from tensorrt_llm._torch.kimi_k3_cache_policy import (
    KIMI_K3_FP8_KV_LAYERS_ENV,
    get_kimi_k3_fp8_kv_layer_ids,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    BAD_PAGE_INDEX,
    KVCacheManagerV2,
    Role,
)
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.llmapi.llm_args import (
    DecodingBaseConfig,
    DFlashDecodingConfig,
    DSparkDecodingConfig,
    MTPDecodingConfig,
)
from tensorrt_llm.llmapi.llm_args import KvCacheConfig as LlmKvCacheConfig
from tensorrt_llm.mapping import Mapping

_DataType = tensorrt_llm.bindings.DataType
_CacheType = tensorrt_llm.bindings.internal.batch_manager.CacheType


def _swizzled_sf_offset(
    row_idx: int | torch.Tensor, col_idx: int | torch.Tensor, sf_per_token: int
) -> int | torch.Tensor:
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


def test_fp4_mla_request_validation_uses_sparse_runtime_params() -> None:
    metadata = SimpleNamespace(
        num_sparse_topk=0,
        kv_cache_manager=SimpleNamespace(dtype=_DataType.NVFP4, kv_factor=1),
        beam_width=1,
        fp4_mla_state=Fp4MlaState(hp_pool=object(), v_scale_pool=object()),
    )
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.generation_only)

    Fp4MlaFmha._is_supported(
        SimpleNamespace(attn=SimpleNamespace(sparse_params=None)),
        torch.empty(1, 4),
        None,
        None,
        metadata,
        forward_args,
    )

    forward_args.sparse_runtime_params.sparse_attn_indices = torch.tensor([0])
    with pytest.raises(NotImplementedError, match="does not support sparse attention"):
        Fp4MlaFmha._is_supported(
            SimpleNamespace(attn=SimpleNamespace(sparse_params=None)),
            torch.empty(1, 4),
            None,
            None,
            metadata,
            forward_args,
        )


def _fp4_mla_validation_metadata(**overrides) -> SimpleNamespace:
    fields = dict(
        num_sparse_topk=0,
        kv_cache_manager=SimpleNamespace(dtype=_DataType.NVFP4, kv_factor=1),
        beam_width=1,
        fp4_mla_state=Fp4MlaState(hp_pool=object(), v_scale_pool=object()),
        helix_position_offsets=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_fp4_mla_request_validation_still_requires_nvfp4_kv() -> None:
    metadata = _fp4_mla_validation_metadata(
        kv_cache_manager=SimpleNamespace(dtype=object(), kv_factor=1)
    )
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.generation_only)

    with pytest.raises(RuntimeError, match="requires NVFP4 KV cache storage"):
        Fp4MlaFmha._is_supported(
            SimpleNamespace(attn=SimpleNamespace(sparse_params=None)),
            torch.empty(1, 4),
            None,
            None,
            metadata,
            forward_args,
        )


@pytest.mark.parametrize(
    "attention_input_type, context_only_cache, supported",
    [
        (AttentionInputType.context_only, True, True),
        (AttentionInputType.generation_only, True, False),
        (AttentionInputType.context_only, False, False),
    ],
    ids=["context_only_cache", "generation", "context_with_full_cache"],
)
def test_fp4_mla_missing_v_scale_pool_only_for_context_only_cache(
    attention_input_type: AttentionInputType, context_only_cache: bool, supported: bool
) -> None:
    # Only a disaggregated context worker omits the V-scale pool.
    metadata = _fp4_mla_validation_metadata(
        kv_cache_manager=SimpleNamespace(
            dtype=_DataType.NVFP4,
            kv_factor=1,
            fp4_mla_context_only_cache=context_only_cache,
        ),
        fp4_mla_state=Fp4MlaState(hp_pool=object(), v_scale_pool=None),
    )
    forward_args = AttentionForwardArgs(attention_input_type=attention_input_type)
    fmha = SimpleNamespace(attn=SimpleNamespace(sparse_params=None))
    is_context = attention_input_type == AttentionInputType.context_only
    k = torch.empty(1, 4) if is_context else None
    v = torch.empty(1, 4) if is_context else None

    if supported:
        assert Fp4MlaFmha._is_supported(fmha, torch.empty(1, 4), k, v, metadata, forward_args)
    else:
        with pytest.raises(RuntimeError, match="requires the V-scale pool"):
            Fp4MlaFmha._is_supported(fmha, torch.empty(1, 4), k, v, metadata, forward_args)


@pytest.mark.parametrize(
    "softmax_stats, expected_exception",
    [
        (None, RuntimeError),
        (torch.empty(2, 3, 2, dtype=torch.bfloat16), ValueError),
        (torch.empty(2, 3, 3, dtype=torch.float32), ValueError),
        (torch.empty(2, 2, 3, dtype=torch.float32).transpose(1, 2), ValueError),
    ],
    ids=["missing", "bf16", "wrong-shape", "non-contiguous"],
)
def test_fp4_mla_helix_validates_softmax_stats(
    softmax_stats: torch.Tensor | None,
    expected_exception: type[Exception],
) -> None:
    num_tokens, num_heads = 2, 3
    metadata = _fp4_mla_validation_metadata(
        helix_position_offsets=torch.arange(num_tokens, dtype=torch.int32),
        num_generations=num_tokens,
    )
    forward_args = AttentionForwardArgs(
        attention_input_type=AttentionInputType.generation_only,
        softmax_stats_tensor=softmax_stats,
    )
    fmha = SimpleNamespace(attn=SimpleNamespace(sparse_params=None, num_heads=num_heads))

    with pytest.raises(
        expected_exception,
        match="FP4 MLA Helix requires.*softmax stats|requires softmax_stats_tensor",
    ):
        Fp4MlaFmha._is_supported(
            fmha, torch.empty(num_tokens, 4), None, None, metadata, forward_args
        )


def test_fp4_mla_helix_accepts_valid_softmax_stats() -> None:
    num_generations, tokens_per_generation, num_heads = 2, 6, 3
    num_tokens = num_generations * tokens_per_generation
    metadata = _fp4_mla_validation_metadata(
        helix_position_offsets=torch.arange(num_tokens, dtype=torch.int32),
        num_generations=num_generations,
    )
    forward_args = AttentionForwardArgs(
        attention_input_type=AttentionInputType.generation_only,
        softmax_stats_tensor=torch.empty(num_tokens, num_heads, 2, dtype=torch.float32),
    )
    fmha = SimpleNamespace(attn=SimpleNamespace(sparse_params=None, num_heads=num_heads))

    assert Fp4MlaFmha._is_supported(
        fmha, torch.empty(num_tokens, 4), None, None, metadata, forward_args
    )


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

    # decode.py binds the backend selector at import time; patch that binding.
    monkeypatch.setattr(fp4_mla_decode, "_fp4_mla_attention_backend", lambda: "triton")

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
        fp4_mla_state=Fp4MlaState(
            generation_cache_scattered=True,
            prequantized_q=torch.empty(0),
            prequantized_q_sf=torch.empty(0),
            q_batch_capacity=num_tokens,
        ),
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
        query_input=q,
        output=output,
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
    state = metadata.fp4_mla_state
    assert not state.generation_cache_scattered
    assert state.prequantized_q is None
    assert state.prequantized_q_sf is None
    assert state.q_batch_capacity is None


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
    q_pe = torch.empty(num_tokens, num_heads, qk_rope_head_dim, dtype=torch.bfloat16)
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
        _helix_spec_tokens_valid=False,
    )
    attn = SimpleNamespace(
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        rope_params=SimpleNamespace(duplicate_data=True),
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

    def run_rope_generation() -> dict:
        trtllm_backend.TrtllmAttention._fp4_mla_rope_generation(
            attn,
            fused_q,
            q_pe,
            latent_cache,
            metadata,
            fuse_q_quant=True,
        )
        args, kwargs = calls[-1]
        assert args[:3] == (metadata, latent_cache, 7)
        assert kwargs["phase"] == "generation"
        assert kwargs["local_layer"] == 5
        assert kwargs["q_quant_input"] is fused_q
        return kwargs

    helix_kwargs = run_rope_generation()
    assert helix_kwargs["helix_position_offsets"] is helix_position_offsets
    assert helix_kwargs["helix_is_inactive_rank"] is helix_is_inactive_rank

    metadata.helix_position_offsets = None
    metadata.helix_is_inactive_rank = None
    non_helix_kwargs = run_rope_generation()
    assert non_helix_kwargs["helix_position_offsets"] is None
    assert non_helix_kwargs["helix_is_inactive_rank"] is None

    metadata.num_generations = 1
    metadata.helix_position_offsets = helix_position_offsets
    metadata.helix_is_inactive_rank = torch.tensor([False])
    metadata.helix_local_slots = torch.tensor([384, 385], dtype=torch.int32)
    metadata._helix_spec_tokens_valid = True
    spec_helix_kwargs = run_rope_generation()
    assert spec_helix_kwargs["helix_position_offsets"] is helix_position_offsets
    assert spec_helix_kwargs["helix_is_inactive_rank"] is metadata.helix_is_inactive_rank
    assert len(calls) == 3


@pytest.mark.parametrize(
    ("cp_size", "helix", "accepted"),
    [(1, False, True), (8, True, True), (8, False, False)],
    ids=["no_cp", "helix", "non_helix_cp"],
)
def test_validate_fp4_mla_config_accepts_only_helix_context_parallelism(
    monkeypatch: pytest.MonkeyPatch, cp_size: int, helix: bool, accepted: bool
) -> None:
    from tensorrt_llm._torch.pyexecutor import model_loader

    monkeypatch.setattr(model_loader, "supports_fp4_mla_attention", lambda config: True)
    model_config = SimpleNamespace(
        sparse_attention_config=None,
        pretrained_config=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64),
        quant_config=SimpleNamespace(quant_mode=SimpleNamespace(has_fp4_kv_cache=lambda: True)),
        mapping=SimpleNamespace(cp_size=cp_size, has_cp_helix=lambda: helix),
        spec_config=None,
    )
    llm_args = SimpleNamespace(kv_cache_config=SimpleNamespace(use_kv_cache_manager_v2=True))
    if accepted:
        model_loader.validate_fp4_mla_config(model_config, llm_args)
    else:
        with pytest.raises(ValueError, match="only through Helix"):
            model_loader.validate_fp4_mla_config(model_config, llm_args)


@pytest.mark.parametrize(
    ("spec_config", "keeps_separate_draft_kv_cache"),
    [
        (DSparkDecodingConfig(max_draft_len=5), True),
        (DFlashDecodingConfig(max_draft_len=7), True),
        (MTPDecodingConfig(num_nextn_predict_layers=1, max_draft_len=3), False),
    ],
    ids=["dspark", "dflash", "mtp"],
)
def test_validate_fp4_mla_config_keeps_standalone_drafter_kv_cache(
    monkeypatch: pytest.MonkeyPatch,
    spec_config: DecodingBaseConfig,
    keeps_separate_draft_kv_cache: bool,
) -> None:
    """A standalone drafter keeps its own draft KV pool under FP4 MLA.

    Without it the drafter falls back to a context arena that is dense in
    max_seq_len. Drafters whose layers share the target's FP4 MLA cache
    (one-model MTP) keep using it.
    """
    from tensorrt_llm._torch.pyexecutor import model_loader

    monkeypatch.setattr(model_loader, "supports_fp4_mla_attention", lambda config: True)
    assert spec_config.spec_dec_mode.use_one_engine()
    model_config = SimpleNamespace(
        sparse_attention_config=None,
        pretrained_config=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64),
        quant_config=SimpleNamespace(quant_mode=SimpleNamespace(has_fp4_kv_cache=lambda: True)),
        mapping=SimpleNamespace(cp_size=1, has_cp_helix=lambda: False),
        spec_config=spec_config,
    )
    llm_args = SimpleNamespace(kv_cache_config=SimpleNamespace(use_kv_cache_manager_v2=True))

    model_loader.validate_fp4_mla_config(model_config, llm_args)

    assert spec_config._allow_separate_draft_kv_cache is keeps_separate_draft_kv_cache


def test_mla_fp4_helix_generation_reaches_fused_cache_update() -> None:
    """Helix generation must reach the fused FP4 RoPE/Q-quant/cache update."""
    num_tokens, num_heads = 2, 2
    qk_nope_head_dim, qk_rope_head_dim = 2, 2
    kv_lora_rank = 4

    class ReachedFusedCacheUpdate(Exception):
        pass

    class FakeFp4Attention:
        has_fp4_kv_cache = True
        has_fp8_kv_cache = False

        def support_fp4_kv_cache(self) -> bool:
            return True

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
        sparse_params=None,
        quant_config=SimpleNamespace(
            layer_quant_mode=SimpleNamespace(has_fp4_kv_cache=lambda: True)
        ),
        kv_cache_dtype="nvfp4",
        k_b_proj_trans=torch.empty(
            num_heads,
            kv_lora_rank,
            qk_nope_head_dim,
            dtype=torch.bfloat16,
        ),
        apply_rotary_emb=False,
        aux_stream=None,
        ln_events=[None, None],
        _bmm_bf16_out=lambda *args: None,
        # MLA.__init__ default; the BF16 k_b_proj BMM passes it through.
        _locality_domain_k_b_proj_trans_shards=None,
    )
    metadata = SimpleNamespace(
        num_seqs=num_tokens,
        num_ctx_tokens=0,
        _helix_spec_tokens_valid=False,
    )

    with pytest.raises(ReachedFusedCacheUpdate):
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
            latent_cache=torch.empty(
                num_tokens,
                kv_lora_rank + qk_rope_head_dim,
                dtype=torch.bfloat16,
            ),
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
        fp4_mla_state=Fp4MlaState(
            positions=torch.arange(80, dtype=torch.int32), fp8_context_state=object()
        ),
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
    assert fp8_meta.fp4_mla_state is None


def _reset_triton_allocator() -> None:
    import triton

    def _null_allocator(size: int, alignment: int, stream):
        raise RuntimeError("Triton kernel requested scratch memory without an allocator.")

    triton.set_allocator(_null_allocator)


def test_fp4_mla_generation_hp_page_ids_skip_mixed_batch_context_rows() -> None:
    metadata = SimpleNamespace(
        num_contexts=2,
        num_seqs=5,
        fp4_mla_state=Fp4MlaState(
            num_sequences=5,
            device_page_table=True,
            page_table_stride=3,
            hp_page_indices=torch.arange(15, dtype=torch.int32),
        ),
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
        is_cuda_graph=True,
        fp4_mla_state=Fp4MlaState(
            num_sequences=3,
            generation_kv_lens=corrected_kv_lens,
            generation_append_lens=generation_lens,
            generation_lengths_num_tokens=4,
            generation_lengths_num_seqs=2,
            generation_lengths_num_contexts=1,
            generation_lengths_capture_recorded=False,
        ),
    )
    populate_calls = []

    def populate_generation_lengths(*args, **kwargs) -> None:
        populate_calls.append((args, kwargs))
        args[2].fill_(19)
        args[3].fill_(2)

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        fp4_mla_metadata,
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
    assert metadata.fp4_mla_state.generation_lengths_capture_recorded
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
    num_tokens = num_seqs * tokens_per_seq
    base = torch.randint(0, 300_000, (num_seqs, 1), generator=generator)
    # One verify group straddles a ledger boundary, one a page boundary.
    base[0, 0] = tokens_per_block * cp_size - max(1, tokens_per_seq // 2)
    base[1, 0] = 7 * tokens_per_block - max(1, tokens_per_seq // 2)
    positions = (base + torch.arange(tokens_per_seq)).reshape(-1).to(torch.int32).cuda()
    metadata = object.__new__(trtllm_backend.TrtllmAttentionMetadata)
    metadata.kv_cache_manager = SimpleNamespace(tokens_per_block=tokens_per_block)
    metadata.mapping = SimpleNamespace(cp_size=cp_size, cp_rank=cp_rank)
    metadata._num_contexts = 1
    metadata._num_generations = num_seqs
    untouched = 77
    # The helix buffers are generation-relative; kv_lens_cuda is batch-indexed.
    metadata.helix_position_offsets = torch.cat(
        [positions, torch.full((2,), untouched, dtype=torch.int32, device="cuda")]
    )
    metadata.helix_local_slots = torch.full_like(metadata.helix_position_offsets, untouched)
    metadata.helix_kv_bounds = torch.full_like(metadata.helix_position_offsets, untouched)
    metadata.kv_lens_cuda = torch.full((num_seqs + 3,), untouched, dtype=torch.int32, device="cuda")

    metadata.recompute_helix_spec_buffers(num_tokens, tokens_per_seq)
    torch.cuda.synchronize()

    # Reference: the per-op formulation (helix_local_len_vec mirrors the cache manager).
    owner = torch.div(positions, tokens_per_block, rounding_mode="floor") % cp_size
    expected_slots = torch.where(
        owner == cp_rank, metadata.helix_local_len_vec(positions), torch.full_like(positions, -1)
    )
    expected_bounds = metadata.helix_local_len_vec(positions + 1)
    gen = slice(0, num_tokens)
    torch.testing.assert_close(metadata.helix_local_slots[gen], expected_slots)
    torch.testing.assert_close(metadata.helix_kv_bounds[gen], expected_bounds)
    torch.testing.assert_close(
        metadata.kv_lens_cuda[1 : 1 + num_seqs],
        expected_bounds.view(num_seqs, tokens_per_seq)[:, -1],
    )
    assert (metadata.helix_local_slots[num_tokens:] == untouched).all()
    assert (metadata.helix_kv_bounds[num_tokens:] == untouched).all()
    assert metadata.kv_lens_cuda[0].item() == untouched
    assert (metadata.kv_lens_cuda[1 + num_seqs :] == untouched).all()
    # Any group of the sequence that this rank does not own writes no slot.
    assert ((metadata.helix_local_slots[gen] == -1) == (owner != cp_rank)).all()


@pytest.mark.parametrize("layer_offset", [0, 1, 3])
def test_fp4_mla_v2_encoded_page_capacity_is_layer_invariant(layer_offset: int) -> None:
    manager = object.__new__(Fp4MlaKVCacheManagerV2)
    manager.impl = SimpleNamespace(
        get_page_index_converter=lambda *_: SimpleNamespace(
            scale=4,
            expansion=1,
            layer_offset=layer_offset,
        ),
        get_page_index_upper_bound=lambda *_: 20 - layer_offset,
    )

    assert manager._role_encoded_page_capacity(0, Role.KEY) == 17


def test_fp4_mla_v2_cache_size_accounts_for_hp_intercept(monkeypatch) -> None:
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "triton")
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


def test_fp4_mla_v2_cache_size_accounts_for_fp8_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "triton")
    monkeypatch.setenv(KIMI_K3_FP8_KV_LAYERS_ENV, "3")
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
    assert slope == fp4_bytes_per_token + 576 * torch.float8_e4m3fn.itemsize
    assert intercept == 3 * (HP_BLOCK_SIZE + 3) * 576 * 2


@pytest.mark.parametrize("rank, num_fp8_layers", [(0, 0), (1, 2)])
def test_fp4_mla_v2_default_fp8_tail_sizing_uses_global_layers(
    monkeypatch: pytest.MonkeyPatch, rank: int, num_fp8_layers: int
) -> None:
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "triton")
    monkeypatch.delenv(KIMI_K3_FP8_KV_LAYERS_ENV, raising=False)
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

    # The last two MLA layers of the global schedule (13 and 15) live on the
    # second pipeline stage.
    num_fp4_layers = 4 - num_fp8_layers
    assert slope == num_fp4_layers * (320 + 40 + 32) + num_fp8_layers * 576
    assert intercept == 3 * 2 * num_fp4_layers * (HP_BLOCK_SIZE + 3) * 576 * 2


def test_kimi_k3_fp8_kv_layer_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(
        model_type="kimi_linear",
        num_hidden_layers=8,
        linear_attn_config={"full_attn_layers": [2, 4, 6, 8]},
    )
    monkeypatch.delenv(KIMI_K3_FP8_KV_LAYERS_ENV, raising=False)
    # NVFP4 defaults to the last two MLA layers; other dtypes keep none.
    assert get_kimi_k3_fp8_kv_layer_ids(config, nvfp4_enabled=True) == frozenset({5, 7})
    assert get_kimi_k3_fp8_kv_layer_ids(config) == frozenset()
    # Other model families never take the Kimi K3 default.
    assert (
        get_kimi_k3_fp8_kv_layer_ids(SimpleNamespace(model_type="deepseek_v3"), nvfp4_enabled=True)
        == frozenset()
    )

    monkeypatch.setenv(KIMI_K3_FP8_KV_LAYERS_ENV, "1")
    assert get_kimi_k3_fp8_kv_layer_ids(config, nvfp4_enabled=True) == frozenset({1})
    monkeypatch.setenv(KIMI_K3_FP8_KV_LAYERS_ENV, "")
    assert get_kimi_k3_fp8_kv_layer_ids(config, nvfp4_enabled=True) == frozenset()
    monkeypatch.setenv(KIMI_K3_FP8_KV_LAYERS_ENV, "2")
    with pytest.raises(ValueError, match="only MLA layers"):
        get_kimi_k3_fp8_kv_layer_ids(config, nvfp4_enabled=True)
    monkeypatch.setenv(KIMI_K3_FP8_KV_LAYERS_ENV, "9")
    with pytest.raises(ValueError, match="outside"):
        get_kimi_k3_fp8_kv_layer_ids(config, nvfp4_enabled=True)


@pytest.mark.parametrize("fp8_env", [None, "", "7"], ids=["unset", "empty", "explicit"])
def test_kimi_k3_deprecated_bf16_kv_layers_env(monkeypatch: pytest.MonkeyPatch, fp8_env) -> None:
    config = SimpleNamespace(
        model_type="kimi_linear",
        num_hidden_layers=8,
        linear_attn_config={"full_attn_layers": [2, 4, 6, 8]},
    )
    if fp8_env is None:
        monkeypatch.delenv(KIMI_K3_FP8_KV_LAYERS_ENV, raising=False)
    else:
        monkeypatch.setenv(KIMI_K3_FP8_KV_LAYERS_ENV, fp8_env)
    monkeypatch.setenv("TLLM_KIMI_K3_BF16_KV_LAYERS", "5")
    # The deprecated variable selects the FP8 layers only when the new one is unset.
    expected = {None: frozenset({5}), "": frozenset(), "7": frozenset({7})}[fp8_env]
    for nvfp4_enabled in (True, False):
        assert get_kimi_k3_fp8_kv_layer_ids(config, nvfp4_enabled=nvfp4_enabled) == expected
    # Other model families ignore the variable.
    assert (
        get_kimi_k3_fp8_kv_layer_ids(SimpleNamespace(model_type="deepseek_v3"), nvfp4_enabled=True)
        == frozenset()
    )


@pytest.mark.parametrize(
    "fp4_kv, sparse_config, expected_fp8_context",
    [
        (True, None, True),
        (True, SimpleNamespace(algorithm="skip_softmax"), False),
        (False, None, False),
    ],
    ids=["dense_nvfp4", "sparse_nvfp4", "bf16"],
)
def test_dense_fp4_mla_declares_fp8_context_workspace(
    monkeypatch: pytest.MonkeyPatch, fp4_kv: bool, sparse_config, expected_fp8_context: bool
) -> None:
    # An all-NVFP4 model still attends cached-prefix chunks through the FP8
    # context kernel, so it must declare that kernel's workspace.
    monkeypatch.setenv(KIMI_K3_FP8_KV_LAYERS_ENV, "")
    monkeypatch.setattr(trtllm_backend, "get_sm_version", lambda: 100)
    requested = {}

    def workspace_bytes_per_token(**kwargs):
        requested.update(kwargs)
        return 7 if kwargs["fp8_context_mla"] else 0

    monkeypatch.setattr(
        trtllm_backend,
        "thop",
        SimpleNamespace(get_context_mla_workspace_bytes_per_token=workspace_bytes_per_token),
    )
    model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(
            model_type="kimi_linear",
            num_hidden_layers=8,
            linear_attn_config={"full_attn_layers": [2, 4, 6, 8]},
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            qk_nope_head_dim=128,
            v_head_dim=128,
            num_attention_heads=16,
        ),
        quant_config=SimpleNamespace(
            quant_mode=SimpleNamespace(
                has_fp4_kv_cache=lambda: fp4_kv, has_fp8_kv_cache=lambda: False
            )
        ),
        sparse_attention_config=sparse_config,
    )

    rate = trtllm_backend.TrtllmAttention.runtime_workspace_bytes_per_token(
        model_config, Mapping(world_size=1, tp_size=1, pp_size=1)
    )

    assert rate == (7 if expected_fp8_context else 0)
    assert requested.get("fp8_context_mla", False) is expected_fp8_context


@pytest.mark.parametrize(
    "model_type, fp8_env, pp_rank, expected_fp8_context",
    [
        # The default keeps global MLA layer 7 in FP8, on the second stage.
        ("kimi_linear", None, 1, True),
        ("kimi_linear", None, 0, False),
        # An empty list keeps every MLA layer in NVFP4.
        ("kimi_linear", "", 1, False),
        # Other model families have no FP8 fallback layers.
        ("deepseek_v3", None, 1, False),
    ],
    ids=["k3_local_fallback", "k3_remote_fallback", "k3_no_fallback", "non_k3"],
)
def test_kimi_k3_fp8_fallback_declares_fp8_context_workspace(
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
    fp8_env,
    pp_rank: int,
    expected_fp8_context: bool,
) -> None:
    # A sparse config keeps the dense FP4 MLA reservation out, so only the
    # Kimi K3 FP8 fallback layers can request the FP8 context workspace.
    if fp8_env is None:
        monkeypatch.delenv(KIMI_K3_FP8_KV_LAYERS_ENV, raising=False)
    else:
        monkeypatch.setenv(KIMI_K3_FP8_KV_LAYERS_ENV, fp8_env)
    monkeypatch.delenv("TLLM_KIMI_K3_BF16_KV_LAYERS", raising=False)
    monkeypatch.setattr(trtllm_backend, "get_sm_version", lambda: 100)
    requested = {}

    def workspace_bytes_per_token(**kwargs):
        requested.update(kwargs)
        return 7 if kwargs["fp8_context_mla"] else 0

    monkeypatch.setattr(
        trtllm_backend,
        "thop",
        SimpleNamespace(get_context_mla_workspace_bytes_per_token=workspace_bytes_per_token),
    )
    model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(
            model_type=model_type,
            num_hidden_layers=8,
            linear_attn_config={"full_attn_layers": [2, 4, 6, 8]},
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            qk_nope_head_dim=128,
            v_head_dim=128,
            num_attention_heads=16,
        ),
        quant_config=SimpleNamespace(
            quant_mode=SimpleNamespace(
                has_fp4_kv_cache=lambda: True, has_fp8_kv_cache=lambda: False
            )
        ),
        sparse_attention_config=SimpleNamespace(algorithm="skip_softmax"),
    )

    rate = trtllm_backend.TrtllmAttention.runtime_workspace_bytes_per_token(
        model_config, _pp_mapping(2, pp_rank)
    )

    assert rate == (7 if expected_fp8_context else 0)
    assert requested.get("fp8_context_mla", False) is expected_fp8_context


def _pp_mapping(pp_size: int, rank: int = 0, pp_partition=None) -> Mapping:
    return Mapping(
        world_size=pp_size, tp_size=1, pp_size=pp_size, rank=rank, pp_partition=pp_partition
    )


@pytest.mark.parametrize(
    "num_hidden_layers, full_attn_layers, mapping, expected",
    [
        # One stage keeps the global tail.
        (8, [2, 4, 6, 8], None, {5, 7}),
        # A stage whose MLA layers are exactly the tail keeps its earlier one in NVFP4.
        (8, [2, 4, 6, 8], _pp_mapping(2), {7}),
        # A one-MLA-layer stage keeps it in NVFP4.
        (8, [2, 4, 6, 8], _pp_mapping(2, pp_partition=[6, 2]), {5}),
        (8, [2, 4, 6, 8], _pp_mapping(4), set()),
        # Stages with more MLA layers keep the full default.
        (16, list(range(2, 17, 2)), _pp_mapping(2), {13, 15}),
        # Too few MLA layers for the default tail.
        (4, [2, 4], None, {3}),
        (4, [4], None, set()),
        # A truncated config ignores full_attn_layers past its last layer.
        (4, [2, 4, 6, 8], None, {3}),
    ],
)
def test_kimi_k3_default_fp8_kv_layers_keep_an_nvfp4_mla_layer_per_stage(
    monkeypatch: pytest.MonkeyPatch,
    num_hidden_layers: int,
    full_attn_layers: list[int],
    mapping,
    expected: set[int],
) -> None:
    monkeypatch.delenv(KIMI_K3_FP8_KV_LAYERS_ENV, raising=False)
    config = SimpleNamespace(
        model_type="kimi_k3",
        num_hidden_layers=num_hidden_layers,
        linear_attn_config={"full_attn_layers": full_attn_layers},
    )

    assert get_kimi_k3_fp8_kv_layer_ids(config, nvfp4_enabled=True, mapping=mapping) == frozenset(
        expected
    )


def test_kimi_k3_default_fp8_kv_layers_agree_across_pp_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(KIMI_K3_FP8_KV_LAYERS_ENV, raising=False)
    config = SimpleNamespace(
        model_type="kimi_k3",
        num_hidden_layers=8,
        linear_attn_config={"full_attn_layers": [2, 4, 6, 8]},
    )

    selections = {
        get_kimi_k3_fp8_kv_layer_ids(config, nvfp4_enabled=True, mapping=_pp_mapping(2, rank))
        for rank in range(2)
    }

    assert selections == {frozenset({7})}


@pytest.mark.parametrize(
    "layer_ids, mapping, stage",
    [
        ("5,7", _pp_mapping(2), 1),
        ("7", _pp_mapping(2, pp_partition=[6, 2]), 1),
        ("1,3,5,7", None, 0),
    ],
)
def test_kimi_k3_explicit_fp8_kv_layers_reject_an_all_fp8_stage(
    monkeypatch: pytest.MonkeyPatch, layer_ids: str, mapping, stage: int
) -> None:
    monkeypatch.setenv(KIMI_K3_FP8_KV_LAYERS_ENV, layer_ids)
    config = SimpleNamespace(
        model_type="kimi_k3",
        num_hidden_layers=8,
        linear_attn_config={"full_attn_layers": [2, 4, 6, 8]},
    )

    with pytest.raises(ValueError, match=f"pipeline stage {stage}"):
        get_kimi_k3_fp8_kv_layer_ids(config, nvfp4_enabled=True, mapping=mapping)
    # Without NVFP4 the model-wide check reports the dtype mismatch instead.
    assert get_kimi_k3_fp8_kv_layer_ids(config, mapping=mapping) == frozenset(
        int(layer_id) for layer_id in layer_ids.split(",")
    )


def test_fp4_mla_v2_cache_size_keeps_an_nvfp4_layer_on_a_tail_only_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "triton")
    monkeypatch.delenv(KIMI_K3_FP8_KV_LAYERS_ENV, raising=False)
    model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(
            model_type="kimi_k3",
            num_hidden_layers=16,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            linear_attn_config={"full_attn_layers": list(range(2, 17, 2))},
        )
    )

    # The second stage owns layers 12-15, whose MLA layers 13 and 15 are the
    # global default tail; layer 13 stays NVFP4.
    slope, _ = Fp4MlaKVCacheManagerV2.get_cache_size_per_token(
        model_config,
        _pp_mapping(2, rank=1, pp_partition=[12, 4]),
        num_layers=2,
        tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
        max_batch_size=3,
        spec_config=SimpleNamespace(tokens_per_gen_step=4),
    )

    assert slope == (320 + 40 + 32) + 576


@pytest.mark.parametrize("chunked_loop_num", [0, 2])
def test_fp4_mla_chunked_prefill_attends_cached_partitions_then_current_chunk(
    monkeypatch: pytest.MonkeyPatch, chunked_loop_num: int
) -> None:
    import tensorrt_llm._torch.attention.mla as mla_module

    monkeypatch.setattr(mla_module, "maybe_compiled_cat", torch.cat)
    tokens, cached, heads, nope, rope, rank, v_dim = 4, 6, 2, 8, 4, 16, 8
    calls = []

    def load_chunked_kv_cache_for_mla(**kwargs):
        calls.append(("load", kwargs["num_ctx_cached_tokens"]))
        num = kwargs["num_ctx_cached_tokens"]
        return torch.zeros(num, rank), torch.zeros(num, rope)

    def forward_partition(q, k, v, metadata, forward_args, *, kv_lens_cuda, kv_lens_cpu):
        calls.append(("partition", forward_args.attention_mask, k.shape[0], int(kv_lens_cpu[0])))
        assert forward_args.attention_input_type == AttentionInputType.context_only
        assert forward_args.latent_cache is None
        assert k.shape == (k.shape[0], heads * (nope + rope))
        assert v.shape == (k.shape[0], heads * v_dim)
        partition_keys.append(k)
        return forward_args.output

    def merge(merged, temp, stats, temp_stats, merge_op, metadata):
        calls.append(("merge", merge_op))

    partition_keys = []
    mha = SimpleNamespace(
        append_fp4_mla_context_cache=lambda q, latent, metadata: calls.append(("append",)),
        load_chunked_kv_cache_for_mla=load_chunked_kv_cache_for_mla,
        forward_fp4_mla_context_partition=forward_partition,
        merge_attention_for_mla=merge,
    )
    module = SimpleNamespace(
        mha=mha,
        out_scale=None,
        kv_lora_rank=rank,
        qk_rope_head_dim=rope,
        qk_nope_head_dim=nope,
        qk_head_dim=nope + rope,
        num_heads_tp=heads,
        num_context_heads=heads,
        v_head_dim=v_dim,
        _project_context_kv=lambda x: torch.zeros(x.shape[0], heads * (nope + v_dim)),
    )
    metadata = SimpleNamespace(
        num_ctx_tokens=tokens,
        num_contexts=1,
        chunked_loop_num=chunked_loop_num,
        cu_chunked_seq_len=[torch.tensor([0, cached])] * chunked_loop_num,
        host_cu_chunked_seq_len=torch.tensor([[0, cached]] * max(chunked_loop_num, 1)),
        chunked_global_offset=[torch.zeros(1)] * chunked_loop_num,
        max_chunk_len_per_loop=[cached] * chunked_loop_num,
        chunked_seq_len=[torch.tensor([cached])] * chunked_loop_num,
        host_chunked_seq_len=[torch.tensor([cached])] * chunked_loop_num,
        prompt_lens_cuda_runtime=torch.tensor([tokens]),
        prompt_lens_cpu_runtime=torch.tensor([tokens]),
        merge_op_tensor=[f"op{i}" for i in range(chunked_loop_num + 1)],
        runtime_features=SimpleNamespace(chunked_prefill_buffer_batch_size=1),
    )
    q = torch.zeros(tokens, heads * (nope + rope))
    latent = torch.randn(tokens, 1, rank + rope)
    output = torch.empty(tokens, heads * v_dim)

    result = MLA.forward_context_with_chunked_prefill_fp4(
        module, q, torch.zeros(tokens, rank), latent, metadata, output
    )

    assert result is output
    # The chunk is appended to the FP4 cache once, before any partition.
    assert calls[0] == ("append",)
    expected = []
    for loop_idx in range(chunked_loop_num):
        expected += [
            ("load", cached),
            ("partition", PredefinedAttentionMask.FULL, cached, cached),
            ("merge", f"op{loop_idx}"),
        ]
    expected += [
        ("partition", PredefinedAttentionMask.CAUSAL, tokens, tokens),
        ("merge", f"op{chunked_loop_num}"),
    ]
    assert calls[1:] == expected
    # The current chunk attends with the RoPE part of its own latent cache.
    current_k = partition_keys[-1].view(tokens, heads, nope + rope)
    for head in range(heads):
        torch.testing.assert_close(current_k[:, head, nope:], latent[:, 0, rank:])


def test_fp4_mla_v2_runtime_sizing_accounts_for_pipeline_slots() -> None:
    manager = object.__new__(Fp4MlaKVCacheManagerV2)
    manager.layer_specs = None
    manager.max_batch_size = 3
    manager.mapping = SimpleNamespace(pp_size=2)
    manager.max_num_tokens = 100
    manager.tokens_per_block = FP4_MLA_TOKENS_PER_BLOCK
    manager.enable_swa_scratch_reuse = False
    manager._has_cp_helix = False
    manager._generation_kv_capacity_headroom = 1
    manager._get_runtime_cache_size_layer_components = lambda: ([10, 8], [None, 19])

    quota = manager._get_quota_from_max_tokens(manager.max_num_tokens)

    hp_page_bytes = FP4_MLA_TOKENS_PER_BLOCK * 8
    # The retained HP window can straddle two physical pages per request.
    expected_quota = manager.max_num_tokens * (10 + 8) + 6 * 2 * hp_page_bytes
    assert quota == expected_quota
    assert manager._get_max_tokens_from_quota(quota) == manager.max_num_tokens


@pytest.mark.parametrize(
    ("fused_v_transpose", "expected_v_head_dim"),
    [(False, 512), (True, None)],
    ids=["mufu16-packed-v", "fused-v-canonical-cache"],
)
def test_fp4_mla_manager_v2_packed_v_matches_cutedsl_variant(
    monkeypatch,
    fused_v_transpose: bool,
    expected_v_head_dim: int | None,
) -> None:
    class _ConstructionCaptured(Exception):
        pass

    captured = {}

    def capture_base_init(manager, *args, **kwargs) -> None:
        captured["manager"] = manager
        raise _ConstructionCaptured

    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "cutedsl")
    monkeypatch.setenv(FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV, str(int(fused_v_transpose)))
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

    assert captured["manager"].mla_v_head_dim == expected_v_head_dim


@pytest.mark.parametrize(
    "fused_v_transpose",
    [False, True],
    ids=["mufu16-packed-v", "fused-v-canonical-cache"],
)
def test_fp4_mla_disagg_import_rebuilds_variant_sidecars(
    monkeypatch,
    fused_v_transpose: bool,
) -> None:
    page_ids = torch.tensor([17, 29, 5, 18, 30, 6], dtype=torch.int32)
    page_valid_tokens = torch.tensor([FP4_MLA_TOKENS_PER_BLOCK, 1, 3] * 2, dtype=torch.int32)
    kv_cache = torch.empty(1, dtype=torch.uint8)
    sf_cache = torch.empty(1, dtype=torch.uint8)
    v_scale_pool = torch.empty((32, 1), dtype=torch.uint8)
    v_packed_pool = torch.empty(1, dtype=torch.uint8)
    staged = []
    scale_rebuilds = []
    packed_rebuilds = []

    def stage_page_metadata(block_ids, prompt_lens, layer_page_offsets, *, page_size, device):
        staged.append((block_ids, prompt_lens, layer_page_offsets, page_size, device))
        return page_ids, page_valid_tokens

    def rebuild_v_scales(
        observed_sf_cache,
        observed_v_scale_pool,
        observed_page_ids,
        observed_page_valid_tokens,
        **kwargs,
    ) -> None:
        scale_rebuilds.append(
            (
                observed_sf_cache,
                observed_v_scale_pool,
                observed_page_ids,
                observed_page_valid_tokens,
                kwargs,
            )
        )

    def rebuild_packed_v(
        v_packed,
        observed_kv_cache,
        observed_page_ids,
        **kwargs,
    ) -> None:
        packed_rebuilds.append((v_packed, observed_kv_cache, observed_page_ids, kwargs))

    def get_v_packed_pool_base() -> torch.Tensor:
        if fused_v_transpose:
            raise AssertionError("fused-V import must not request a packed-V pool")
        return v_packed_pool

    manager = SimpleNamespace(
        dtype=_DataType.NVFP4,
        kv_factor=1,
        mla_v_scale_head_dim=512,
        mla_v_head_dim=None if fused_v_transpose else 512,
        tokens_per_block=FP4_MLA_TOKENS_PER_BLOCK,
        pp_layers=[7, 11],
        num_local_layers=2,
        get_fp4_mla_page_table_spec=lambda *_: SimpleNamespace(),
        get_batch_cache_indices=lambda request_ids, layer_idx=None: [[17, 29, 31], [5]],
        get_fp4_mla_cache_pool_bases=lambda: (kv_cache, sf_cache),
        get_mla_v_scale_pool_base=lambda: v_scale_pool,
        get_mla_v_scale_page_offset=lambda compact_layer: compact_layer,
        get_mla_v_packed_pool_base=get_v_packed_pool_base,
    )
    monkeypatch.setattr(fp4_mla_v_cache, "_fp4_mla_attention_backend", lambda: "cutedsl")
    monkeypatch.setattr(
        fp4_mla_v_cache,
        "_fp4_mla_cutedsl_fused_v_transpose_enabled",
        lambda: fused_v_transpose,
    )
    monkeypatch.setattr(
        fp4_mla_v_cache,
        "_stage_fp4_mla_import_page_metadata",
        stage_page_metadata,
    )
    monkeypatch.setattr(
        fp4_mla_v_cache,
        "_rebuild_fp4_mla_v_scales_from_k_scales",
        rebuild_v_scales,
    )
    monkeypatch.setattr(
        fp4_mla_v_cache,
        "_repack_cutedsl_v_packed_cache",
        rebuild_packed_v,
    )

    assert fp4_mla_backend.rebuild_fp4_mla_disagg_imported_cache(
        manager,
        request_ids=[41, 42],
        prompt_lens=[FP4_MLA_TOKENS_PER_BLOCK + 1, 3],
    )

    # One worklist covers the prompt pages of every request in every layer.
    assert staged == [
        (
            [[17, 29], [5]],
            [FP4_MLA_TOKENS_PER_BLOCK + 1, 3],
            [0, 1],
            FP4_MLA_TOKENS_PER_BLOCK,
            v_scale_pool.device,
        )
    ]
    assert len(scale_rebuilds) == 1
    observed_sf_cache, observed_pool, observed_ids, observed_valid, kwargs = scale_rebuilds[0]
    assert observed_sf_cache is sf_cache
    assert observed_pool.shape == (1, *v_scale_pool.shape)
    assert observed_pool.data_ptr() == v_scale_pool.data_ptr()
    assert observed_ids is page_ids
    assert observed_valid is page_valid_tokens
    assert kwargs == {
        "local_layer": 0,
        "v_head_dim": 512,
        "page_size": FP4_MLA_TOKENS_PER_BLOCK,
    }
    if fused_v_transpose:
        assert packed_rebuilds == []
    else:
        assert len(packed_rebuilds) == 1
        v_packed, observed_kv_cache, observed_ids, kwargs = packed_rebuilds[0]
        assert v_packed is v_packed_pool
        assert observed_kv_cache is kv_cache
        assert observed_ids is page_ids
        assert kwargs == {
            "v_head_dim": 512,
            "page_size": FP4_MLA_TOKENS_PER_BLOCK,
            "block_v": fp4_mla_backend.FP4_MLA_SCALE_ROW_GROUP,
        }


def test_fp4_mla_disagg_import_skips_hybrid_linear_attention_layers(monkeypatch) -> None:
    requested_page_layers = []
    requested_offset_layers = []
    staged_offsets = []

    def get_batch_cache_indices(request_ids, *, layer_idx):
        requested_page_layers.append((request_ids, layer_idx))
        return [[17, 29]]

    def get_mla_v_scale_page_offset(compact_layer):
        requested_offset_layers.append(compact_layer)
        return 10 + compact_layer

    def stage_page_metadata(_block_ids, _prompt_lens, layer_page_offsets, **_kwargs):
        staged_offsets.append(layer_page_offsets)
        return torch.empty(0, dtype=torch.int32), torch.empty(0, dtype=torch.int32)

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
        get_fp4_mla_cache_pool_bases=lambda: (
            torch.empty(1, dtype=torch.uint8),
            torch.empty(1, dtype=torch.uint8),
        ),
        get_mla_v_scale_pool_base=lambda: torch.empty((32, 1), dtype=torch.uint8),
        get_mla_v_scale_page_offset=get_mla_v_scale_page_offset,
    )
    monkeypatch.setattr(fp4_mla_v_cache, "_fp4_mla_attention_backend", lambda: "triton")
    monkeypatch.setattr(
        fp4_mla_v_cache,
        "_stage_fp4_mla_import_page_metadata",
        stage_page_metadata,
    )
    monkeypatch.setattr(
        fp4_mla_v_cache,
        "_rebuild_fp4_mla_v_scales_from_k_scales",
        lambda *_args, **_kwargs: None,
    )

    assert fp4_mla_backend.rebuild_fp4_mla_disagg_imported_cache(
        manager,
        request_ids=[41],
        prompt_lens=[FP4_MLA_TOKENS_PER_BLOCK + 1],
    )

    # Pages come from the first attention layer, and only the two attention
    # layers contribute a page offset to the worklist.
    assert requested_page_layers == [([41], 7)]
    assert requested_offset_layers == [0, 1]
    assert staged_offsets == [[10, 11]]


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
        request_ids=[41],
        prompt_lens=[0],
    )


@pytest.mark.parametrize(
    "backend,fused_v_transpose",
    [("triton", False), ("cutedsl", False), ("cutedsl", True)],
    ids=["triton", "cutedsl", "cutedsl_fused_v_transpose"],
)
def test_fp4_mla_disagg_import_rebuild_matches_context_cache_update(
    monkeypatch, backend: str, fused_v_transpose: bool
) -> None:
    """The generation-side rebuild reproduces the sidecars the context update wrote."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("FP4 MLA (NVFP4 KV cache) requires Blackwell or newer")
    if backend == "cutedsl" and _is_cutedsl_unavailable():
        pytest.skip("CuTeDSL FP4 MLA requires SM107 and nvidia-cutlass-dsl")
    _reset_triton_allocator()
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, backend)
    monkeypatch.setenv(FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV, str(int(fused_v_transpose)))
    torch.manual_seed(0)
    page_size = FP4_MLA_TOKENS_PER_BLOCK
    # Partial last pages, a partial 16-token tile, and an exact page multiple.
    seq_lens = [page_size + 1, 2 * page_size, 37]
    num_pages = sum((seq_len + page_size - 1) // page_size for seq_len in seq_lens)
    manager = _create_fp4_mla_v2_manager(
        max_tokens=num_pages * page_size,
        max_seq_len=2 * page_size,
        max_batch_size=len(seq_lens),
    )
    try:
        manager.add_dummy_requests(list(range(len(seq_lens))), seq_lens)
        for pool in manager._iter_fp4_mla_physical_pool_views():
            pool.view(torch.uint8).zero_()
        metadata = _build_multi_seq_metadata(manager, seq_lens=seq_lens, page_size=page_size)
        latent = (
            torch.randn(sum(seq_lens), 576, dtype=torch.bfloat16, device="cuda") * 0.25
        ).clamp_(-1.0, 1.0)
        scatter_fp4_mla_kv_cache(
            metadata,
            latent,
            layer_idx=0,
            token_offset=0,
            phase="context",
            local_layer=0,
            v_head_dim=512,
        )
        torch.cuda.synchronize()

        v_scale = manager.get_mla_v_scale_pool_base()
        v_packed = manager.get_mla_v_packed_pool_base()
        assert (v_packed is None) == (backend == "triton" or fused_v_transpose)
        expected_v_scale = v_scale.clone()
        expected_v_packed = None if v_packed is None else v_packed.clone()
        assert expected_v_scale.any()

        v_scale.zero_()
        if v_packed is not None:
            v_packed.zero_()
        manager.rebuild_disagg_imported_cache(
            [
                SimpleNamespace(py_request_id=request_id, prompt_len=seq_len)
                for request_id, seq_len in enumerate(seq_lens)
            ]
        )
        torch.cuda.synchronize()

        assert torch.equal(v_scale, expected_v_scale)
        if v_packed is not None:
            assert torch.equal(v_packed, expected_v_packed)
    finally:
        torch.cuda.synchronize()
        _reset_triton_allocator()
        manager.shutdown()


def test_fp4_mla_hp_ring_content_depends_only_on_position(monkeypatch) -> None:
    """A context prefill leaves the same BF16 tail in rings of different sizes.

    A context worker without speculative decoding keeps a 16-slot HP ring and a
    generation worker with five draft tokens keeps 21 slots. The disaggregated
    transfer moves each retained position from slot ``p % 16`` to slot
    ``p % 21``; that reproduces the generation worker's own prefill only if the
    ring holds the same values at those slots, which this checks against a
    generation-shaped manager prefilling the same latents.
    """
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("FP4 MLA (NVFP4 KV cache) requires Blackwell or newer")
    _reset_triton_allocator()
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "triton")
    torch.manual_seed(0)
    page_size = FP4_MLA_TOKENS_PER_BLOCK
    # Partial tiles of 1 and 15 tokens, one ending in a later page, and an
    # exact tile multiple that leaves no tail.
    seq_lens = [page_size + 1, 2 * page_size - 1, 37, 48]
    num_pages = sum((seq_len + page_size - 1) // page_size for seq_len in seq_lens)
    managers = []
    try:
        for spec_config in (None, MTPDecodingConfig(max_draft_len=5)):
            manager = _create_fp4_mla_v2_manager(
                max_tokens=num_pages * page_size,
                max_seq_len=2 * page_size,
                max_batch_size=len(seq_lens),
                spec_config=spec_config,
            )
            managers.append(manager)
            manager.add_dummy_requests(list(range(len(seq_lens))), seq_lens)
            for pool in manager._iter_fp4_mla_physical_pool_views():
                pool.view(torch.uint8).zero_()
        ctx, gen = managers
        assert ctx.fp4_mla_hp_pool_size == HP_BLOCK_SIZE
        assert gen.fp4_mla_hp_pool_size == HP_BLOCK_SIZE + 5

        latent = (
            torch.randn(sum(seq_lens), 576, dtype=torch.bfloat16, device="cuda") * 0.25
        ).clamp_(-1.0, 1.0)
        for manager in managers:
            scatter_fp4_mla_kv_cache(
                _build_multi_seq_metadata(manager, seq_lens=seq_lens, page_size=page_size),
                latent,
                layer_idx=0,
                token_offset=0,
                phase="context",
                local_layer=0,
                v_head_dim=512,
            )
        torch.cuda.synchronize()

        def ring_tokens(manager, request_id, seq_len):
            page = manager._get_batch_cache_indices_by_pool_id(
                [request_id],
                pool_id=manager.get_fp4_mla_page_table_spec(0).hp_pool_id,
                is_kv_aggregate=False,
            )[0][(seq_len - 1) // page_size]
            ring = manager.get_fp4_mla_hp_pool()[page, 0, 0].view(-1, 576)
            tail = range(seq_len - seq_len % HP_BLOCK_SIZE, seq_len)
            return ring[[p % manager.fp4_mla_hp_pool_size for p in tail]], tail

        for request_id, seq_len in enumerate(seq_lens):
            ctx_tail, tail = ring_tokens(ctx, request_id, seq_len)
            gen_tail, _ = ring_tokens(gen, request_id, seq_len)
            assert len(tail) == seq_len % HP_BLOCK_SIZE
            # Every tail token was written, so the comparison is not vacuous.
            assert gen_tail.ne(0).any(dim=1).all()
            assert torch.equal(ctx_tail, gen_tail)
    finally:
        torch.cuda.synchronize()
        _reset_triton_allocator()
        for manager in managers:
            manager.shutdown()


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
    finally:
        manager.shutdown()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="FP4 MLA (NVFP4 KV cache) requires Blackwell or newer",
)
@pytest.mark.parametrize("device_page_table", ["0", "1"])
def test_fp4_mla_manager_v2_copies_per_pool_block_offsets(monkeypatch, device_page_table) -> None:
    if device_page_table == "0" and not prefer_pinned():
        pytest.skip("Native page-table expansion requires GPU-addressable host memory")
    monkeypatch.setenv("TLLM_KVCM_V2_DEVICE_PAGE_TABLE", device_page_table)
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "triton")
    seq_lens = [FP4_MLA_TOKENS_PER_BLOCK + 1, 3]
    manager = _create_fp4_mla_v2_manager(max_tokens=1024, max_seq_len=512, max_batch_size=2)
    try:
        manager.add_dummy_requests(list(range(len(seq_lens))), seq_lens)
        assert manager._use_per_layer_page_tables is False
        assert manager.uses_device_page_table is (device_page_table == "1")
        if device_page_table == "0":
            assert all(
                tensor.is_pinned()
                for tensor in (
                    manager.host_kv_cache_block_offsets,
                    manager.index_scales,
                    manager.kv_offset,
                    manager.index_mapper.get_copy_index([0, 1], 0, 1),
                )
            )
        page_spec = manager.get_fp4_mla_page_table_spec(0)
        assert page_spec.cache_page_index_scale == 1
        dst = torch.full(
            (manager.num_attention_op_pools, len(seq_lens), 2, manager.max_blocks_per_seq),
            -1,
            dtype=torch.int32,
            device="cuda",
        )

        manager.copy_batch_block_offsets(dst, list(range(len(seq_lens))), 1, 0, len(seq_lens))
        torch.cuda.synchronize()

        block_ids = manager.get_batch_cache_indices(list(range(len(seq_lens))), layer_idx=0)
        for seq_idx, seq_len in enumerate(seq_lens):
            num_blocks = -(-seq_len // FP4_MLA_TOKENS_PER_BLOCK)
            assert (
                dst[page_spec.cache_pool_id, seq_idx, 0, :num_blocks].tolist()
                == block_ids[seq_idx][:num_blocks]
            )

        # Check every lifecycle pool and both planes, including padding, after
        # reordering and shrinking the batch. Reuse a capacity-sized output so
        # that rows beyond the active batch must remain untouched.
        for request_ids in ([1, 0], [1]):
            dst.fill_(-1)
            copy_idx = manager.index_mapper.get_copy_index(request_ids, 0, 1).long()
            base = manager.host_kv_cache_block_offsets[:, copy_idx, 0, :]
            scaled = base * manager.index_scales[:, None, None]
            expected = torch.stack(
                (
                    torch.where(base == BAD_PAGE_INDEX, 0, scaled),
                    torch.where(
                        base == BAD_PAGE_INDEX, 0, scaled + manager.kv_offset[:, None, None]
                    ),
                ),
                dim=2,
            )
            manager.copy_batch_block_offsets(dst, request_ids, 1, 0, len(request_ids))
            torch.cuda.synchronize()
            torch.testing.assert_close(dst[:, : len(request_ids)].cpu(), expected)
            assert torch.all(dst[:, len(request_ids) :] == -1)
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
    # Unpack all rows at once, preserving low-nibble/high-nibble order.
    packed = fp4_bytes[:, : (logical_dim + 1) // 2]
    nibbles = torch.stack((packed & 0x0F, packed >> 4), dim=-1)
    nibbles = nibbles.flatten(1)[:, :logical_dim]
    values = fp4_values[(nibbles & 0x07).long()]
    values = torch.where((nibbles & 0x08) != 0, -values, values)

    rows = torch.arange(fp4_bytes.shape[0], device=fp4_tensor.device)[:, None]
    cols = torch.arange(logical_dim, device=fp4_tensor.device)[None, :] // FP4_BLOCK_SIZE
    offsets = _swizzled_sf_offset(rows, cols, sf_per_token)
    return values * sf_flat.float()[offsets] / global_scale


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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="FP4 MLA (NVFP4 KV cache) requires Blackwell or newer",
)
def test_fp4_mla_pool_views_built_in_inference_mode_can_be_scrubbed(monkeypatch) -> None:
    """Cached FP4 MLA pool views stay writable outside inference mode.

    Warmup forwards run under ``torch.inference_mode`` and are the first to
    build and cache these views; the post-warmup KV-cache scrub then zeroes
    every physical pool outside it.
    """
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "triton")
    manager = _create_fp4_mla_v2_manager(max_tokens=1024, max_seq_len=512, max_batch_size=2)
    try:
        with torch.inference_mode():
            manager.get_fp4_mla_cache_buffers(0)
            manager.get_fp4_mla_cache_pool_bases()
            for _ in manager._iter_fp4_mla_physical_pool_views():
                pass
        cached = [
            tensor
            for value in manager._fp4_mla_view_cache.values()
            for tensor in (value if isinstance(value, tuple) else (value,))
        ]
        assert cached
        assert not any(tensor.is_inference() for tensor in cached)
        for view in manager._iter_fp4_mla_physical_pool_views():
            view.zero_()
        assert not manager.check_invalid_values_in_kv_cache(fill_with_zero=True)
    finally:
        manager.shutdown()


def _create_fp4_mla_v2_manager(
    *,
    max_tokens: int,
    max_seq_len: int,
    max_batch_size: int,
    spec_config: DecodingBaseConfig | None = None,
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


def _build_multi_seq_metadata(
    kv_cache_manager: Fp4MlaKVCacheManagerV2, *, seq_lens: list[int], page_size: int
) -> SimpleNamespace:
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
        page_size=page_size,
        num_contexts=num_seqs,
        num_seqs=num_seqs,
        kv_lens_cuda_runtime=kv_lens,
        prompt_lens_cuda_runtime=prompt_lens_cuda,
        prompt_lens_cpu_runtime=prompt_lens_cpu,
        request_ids=request_ids,
        runtime_features=SimpleNamespace(has_speculative_draft_tokens=False),
        is_cuda_graph=False,
        fp4_mla_state=Fp4MlaState(
            batch_indices=batch_indices,
            positions=positions,
            _paged_kv_indices=paged_kv_indices,
            hp_page_indices=hp_page_indices,
            _paged_kv_indptr=paged_kv_indptr,
            paged_kv_indptr_decode=paged_kv_indptr.clone(),
            device_page_table=True,
            device_page_table_valid=True,
            page_table_stride=max_blocks_per_seq,
            context_repack_max_touched_pages=max_blocks_per_seq,
            num_context_blocks=num_seqs * max_blocks_per_seq,
            num_generation_blocks=0,
            num_sequences=num_seqs,
            num_blocks=None,
            hp_pool=hp_pool,
            v_scale_pool=kv_cache_manager.get_mla_v_scale_pool(),
            generation_kv_lens=torch.empty(num_seqs, dtype=torch.int32, device=device),
            generation_append_lens=torch.empty(num_seqs, dtype=torch.int32, device=device),
            generation_lengths_num_tokens=-1,
            generation_lengths_num_seqs=-1,
            generation_lengths_num_contexts=-1,
            generation_lengths_capture_recorded=False,
            q_global_scale=q_global_scale,
            kv_global_scale=kv_global_scale,
        ),
    )


def _materialize_reference_cache_storage(metadata, layer_idx: int, head_dim: int) -> torch.Tensor:
    kv_cache, sf_cache = metadata.kv_cache_manager.get_fp4_mla_cache_buffers(layer_idx)
    sf_cache = sf_cache.view(torch.float8_e4m3fn)
    storage_head_dim = kv_cache.shape[-1] * 2
    static_global_scale = float(_get_fp4_mla_global_scale(metadata, kv_cache.device).item())
    num_generation_sequences = metadata.num_seqs - metadata.num_contexts
    page_stride = metadata.fp4_mla_state.page_table_stride
    page_rows = metadata.fp4_mla_state.paged_kv_indices.view(
        metadata.num_seqs,
        page_stride,
    )
    generation_page_rows = page_rows[metadata.num_contexts : metadata.num_seqs]
    indptr = (
        metadata.fp4_mla_state.paged_kv_indptr_decode[: num_generation_sequences + 1].cpu().tolist()
    )
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
    page_rows = metadata.fp4_mla_state.paged_kv_indices.view(
        metadata.num_seqs,
        metadata.fp4_mla_state.page_table_stride,
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
            # The K residual is stored after head_dim and added back onto the
            # RoPE tail, as the chunked-cache gather kernel does.
            residual_end = head_dim + FP4_MLA_K_RESIDUAL_DIM
            token[-FP4_MLA_K_RESIDUAL_DIM:] += dequantized_pages[physical_page][
                page_position,
                head_dim:residual_end,
            ]
        tokens.append(token)
    return torch.stack(tokens, dim=0)


@pytest.mark.skipif(
    _is_cutedsl_unavailable(),
    reason="requires Rubin SM107 and the CuTeDSL runtime",
)
@pytest.mark.parametrize("inactive_rank", [False, True], ids=["active", "inactive"])
def test_fp4_mla_helix_scatter_uses_global_rope_and_local_cache_position(
    monkeypatch,
    inactive_rank: bool,
) -> None:
    _reset_triton_allocator()
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "cutedsl")
    monkeypatch.setenv(FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV, "0")
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
        state = metadata.fp4_mla_state
        head_dim = kv_lora_rank + qk_rope_head_dim
        state.positions = torch.tensor([global_position], dtype=torch.int32, device="cuda")
        metadata._helix_spec_tokens_valid = False
        metadata.helix_position_offsets = state.positions
        metadata.helix_is_inactive_rank = torch.tensor(
            [inactive_rank], dtype=torch.bool, device="cuda"
        )

        kv_cache, sf_cache = kv_cache_manager.get_fp4_mla_cache_buffers(0)
        kv_cache.zero_()
        sf_cache.zero_()
        state.hp_pool.zero_()
        state.v_scale_pool.zero_()
        v_packed_pool = kv_cache_manager.get_mla_v_packed_pool_base()
        assert v_packed_pool is not None
        v_packed_pool.zero_()

        generation_latent = (
            torch.randn(1, head_dim, dtype=torch.bfloat16, device="cuda") * 0.25
        ).clamp_(-1.0, 1.0)
        generation_latent[:, :FP4_BLOCK_SIZE] = 0.75
        generation_latent[:, 1:FP4_BLOCK_SIZE:2] = -0.75
        q_pe = (
            torch.randn(1, num_heads, qk_rope_head_dim, dtype=torch.bfloat16, device="cuda") * 0.25
        ).clamp_(-1.0, 1.0)
        original_q_pe = q_pe.clone()
        q_quant_input = torch.zeros(1, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        q_rope_out = q_quant_input[..., kv_lora_rank:]
        # Identity rotation everywhere except the global position, which rotates
        # every RoPE pair by 90 degrees; any other rope position leaves q_pe as is.
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
            state.q_global_scale,
            FP4_MLA_Q_RESIDUAL_DIM,
            is_act=True,
        )
        torch.testing.assert_close(state.prequantized_q, expected_q_fp4)
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
            state.prequantized_q_sf.view(torch.uint8)[valid_q_sf_offsets],
            expected_q_sf.view(torch.uint8).flatten()[valid_q_sf_offsets],
        )
        assert state.q_batch_capacity == 1
        if inactive_rank:
            for cache_tensor in (
                kv_cache,
                sf_cache,
                state.hp_pool,
                state.v_scale_pool,
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


@pytest.mark.skipif(
    _is_cutedsl_unavailable(),
    reason="requires Rubin SM107 and the CuTeDSL runtime",
)
def test_fp4_mla_helix_generation_writes_hp_ring_at_global_position(monkeypatch) -> None:
    """Helix generation stores the BF16 tail at the token's global ring slot.

    The context writer and the disaggregated ring transfer address the HP
    ring by global position, while a Helix rank's cache position is
    rank-local. The generation update must use the global position.
    """
    _reset_triton_allocator()
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "cutedsl")
    monkeypatch.setenv(FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV, "0")
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
        seed=37,
    )
    try:
        state = metadata.fp4_mla_state
        head_dim = kv_lora_rank + qk_rope_head_dim
        ring_size = kv_cache_manager.fp4_mla_hp_pool_size
        assert global_position % ring_size != local_cache_position % ring_size
        state.positions = torch.tensor([global_position], dtype=torch.int32, device="cuda")
        metadata._helix_spec_tokens_valid = False
        metadata.helix_position_offsets = state.positions
        metadata.helix_is_inactive_rank = torch.tensor([False], dtype=torch.bool, device="cuda")

        kv_cache, sf_cache = kv_cache_manager.get_fp4_mla_cache_buffers(0)
        kv_cache.zero_()
        sf_cache.zero_()
        state.hp_pool.zero_()
        state.v_scale_pool.zero_()
        v_packed_pool = kv_cache_manager.get_mla_v_packed_pool_base()
        if v_packed_pool is not None:
            v_packed_pool.zero_()

        generation_latent = (
            torch.randn(1, head_dim, dtype=torch.bfloat16, device="cuda") * 0.25
        ).clamp_(-1.0, 1.0)
        generation_latent[:, :kv_lora_rank] += 0.5
        q_pe = (
            torch.randn(1, num_heads, qk_rope_head_dim, dtype=torch.bfloat16, device="cuda") * 0.25
        ).clamp_(-1.0, 1.0)
        q_quant_input = torch.zeros(1, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        q_rope_out = q_quant_input[..., kv_lora_rank:]
        rotary_cos_sin = torch.zeros(
            (FP4_MLA_TOKENS_PER_BLOCK, qk_rope_head_dim, 2),
            dtype=torch.float32,
            device="cuda",
        )
        rotary_cos_sin[..., 0] = 1.0

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

        hp_pool = state.hp_pool
        pool_head_dim = hp_pool.shape[-1] // ring_size
        ring = hp_pool.view(*hp_pool.shape[:-1], ring_size, pool_head_dim)
        written_slots = sorted(
            {
                int(slot)
                for slot in torch.nonzero(ring[..., :kv_lora_rank].abs().amax(dim=-1))[
                    :, -1
                ].tolist()
            }
        )
        assert written_slots == [global_position % ring_size]
        slot_rows = ring[..., global_position % ring_size, :kv_lora_rank].reshape(-1, kv_lora_rank)
        written = slot_rows[slot_rows.abs().amax(dim=-1) > 0]
        assert written.shape[0] >= 1
        torch.testing.assert_close(
            written[0].float(), generation_latent[0, :kv_lora_rank].float(), rtol=0, atol=0
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
    seq_lens: list[int],
    num_heads: int,
    seed: int,
    query_len_per_seq: int = 1,
    enable_helix: bool = False,
    enable_block_reuse: bool = False,
) -> tuple[Fp4MlaKVCacheManagerV2, SimpleNamespace, torch.Tensor, int, int]:
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
    assert metadata.fp4_mla_state.v_scale_pool is not None
    persistent_pool_base = kv_cache_manager.get_mla_v_packed_pool_base()
    if persistent_pool_base is not None:
        persistent_pool_base.zero_()
    metadata.fp4_mla_state.v_scale_pool.zero_()

    metadata.kv_lens_cuda_runtime = torch.tensor(
        context_seq_lens,
        dtype=torch.int32,
        device=device,
    )
    metadata.prompt_lens_cuda_runtime = metadata.kv_lens_cuda_runtime.clone()
    metadata.prompt_lens_cpu_runtime = torch.tensor(context_seq_lens, dtype=torch.int32)
    metadata.fp4_mla_state.batch_indices = torch.cat(
        [
            torch.full((seq_len,), seq_idx, dtype=torch.int32, device=device)
            for seq_idx, seq_len in enumerate(context_seq_lens)
        ]
    )
    metadata.fp4_mla_state.positions = torch.cat(
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
    metadata.fp4_mla_state.num_context_blocks = 0
    metadata.fp4_mla_state.num_generation_blocks = (
        len(seq_lens) * metadata.fp4_mla_state.page_table_stride
    )
    metadata.kv_lens_cuda_runtime = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    metadata.prompt_lens_cuda_runtime = torch.full(
        (len(seq_lens),), query_len_per_seq, dtype=torch.int32, device=device
    )
    metadata.prompt_lens_cpu_runtime = torch.full(
        (len(seq_lens),), query_len_per_seq, dtype=torch.int32
    )
    num_queries = len(seq_lens) * query_len_per_seq
    metadata.fp4_mla_state.batch_indices = torch.arange(
        len(seq_lens),
        dtype=torch.int32,
        device=device,
    ).repeat_interleave(query_len_per_seq)
    metadata.fp4_mla_state.positions = torch.cat(
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
        positions = metadata.fp4_mla_state.positions
        metadata.helix_position_offsets = positions.clone()
        metadata.helix_local_slots = positions.clone()
        metadata.helix_kv_bounds = positions + 1
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
    assert metadata.fp4_mla_state.prequantized_q is not None
    assert metadata.fp4_mla_state.prequantized_q_sf is not None
    assert metadata.fp4_mla_state.q_batch_capacity == num_queries
    torch.testing.assert_close(q_rope_out, q_pe, rtol=0, atol=0)

    canonical_generation = _materialize_reference_cache_tokens(
        metadata,
        layer_idx=0,
        batch_indices=metadata.fp4_mla_state.batch_indices,
        positions=metadata.fp4_mla_state.positions,
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
    metadata.fp4_mla_state.hp_pool.zero_()
    torch.cuda.synchronize()

    return kv_cache_manager, metadata, q_quant_input, kv_lora_rank, qk_rope_head_dim


def _fp4_mla_attention_decode_reference(
    metadata: SimpleNamespace,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    *,
    sm_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = kv_lora_rank + qk_rope_head_dim
    storage = _materialize_reference_cache_storage(metadata, 0, head_dim)
    dequant_cache = storage[..., :head_dim]
    dequant_k_residual = None
    if _fp4_mla_attention_backend() in ("triton", "cutedsl"):
        dequant_k_residual = storage[..., head_dim : head_dim + FP4_MLA_K_RESIDUAL_DIM]
    num_heads = q_nope.shape[1]
    q_full = torch.cat((q_nope, q_pe), dim=-1).reshape(-1, head_dim)
    global_scale = metadata.fp4_mla_state.q_global_scale
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
    if "_fp4_mla_attention_p_buf" in metadata.fp4_mla_state.workspaces:
        p_dequant = _dequant_fp4_swizzled(
            metadata.fp4_mla_state.workspaces["_fp4_mla_attention_p_buf"],
            metadata.fp4_mla_state.workspaces["_fp4_mla_attention_p_sf_buf"],
            logical_dim=metadata.page_size,
            sf_per_token=metadata.page_size // FP4_BLOCK_SIZE,
            global_scale=FP4_MLA_P_GLOBAL_SCALE,
        )

    indptr = metadata.fp4_mla_state.paged_kv_indptr_decode.cpu().tolist()
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
        assert page_count == metadata.fp4_mla_state.page_table_stride
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
    split_kv: int | None = None,
) -> None:
    _reset_triton_allocator()
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, backend)
    monkeypatch.setenv(
        FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE_ENV,
        str(int(fused_v_transpose)),
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
            prequantized_q=metadata.fp4_mla_state.prequantized_q,
            prequantized_q_sf=metadata.fp4_mla_state.prequantized_q_sf,
            q_batch_capacity=metadata.fp4_mla_state.q_batch_capacity,
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
    reason="requires Rubin SM107 and the CuTeDSL runtime",
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
    reason="requires Rubin SM107 and the CuTeDSL runtime",
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
    reason="requires Rubin SM107 and the CuTeDSL runtime",
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
    reason="requires Rubin SM107 and the CuTeDSL runtime",
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

    monkeypatch.setattr(fp4_mla_cache_update, "_repack_cutedsl_v_packed_cache", record_repack)

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

        hp_page = int(metadata.fp4_mla_state.hp_page_indices[0].item())
        # hp_pool is [pages, FP4 MLA layers, kv_factor, ring * head_dim]; with
        # MTP the draft layer adds a second FP4 layer. Layer 0 is the target.
        hp_pool = metadata.fp4_mla_state.hp_pool
        hp_ring = hp_pool.view(*hp_pool.shape[:3], ring_size, head_dim)
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
    reason="requires Rubin SM107 and the CuTeDSL runtime",
)
def test_fp4_mla_v_repack_full_pages_matches_reference() -> None:
    from tensorrt_llm._torch.attention.backends.fp4_mla import fp4_mla_cutedsl_v_repack

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
    reason="requires Rubin SM107 and the CuTeDSL runtime",
)
def test_fp4_mla_v_repack_selected_pages_preserves_untouched_page() -> None:
    from tensorrt_llm._torch.attention.backends.fp4_mla import fp4_mla_cutedsl_v_repack

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
    reason="requires Rubin SM107 and the CuTeDSL runtime",
)
def test_fp4_mla_v_repack_generation_csr_crosses_page_boundary() -> None:
    from tensorrt_llm._torch.attention.backends.fp4_mla import fp4_mla_cutedsl_v_repack

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
