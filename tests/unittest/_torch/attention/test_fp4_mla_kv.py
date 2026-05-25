# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Roundtrip tests for the FP4 MLA KV-cache kernels.

Exercises ``scatter_fp4_mla_kv_cache`` and ``get_fp4_mla_decode_cache``
(plus ``update_hp_kv_for_fp4_mla``) as a pair on a tiny V1 ``KVCacheManager``.
The goal is to catch stride / page-id / SF-layout bugs without standing up a
real model or FlashInfer wrapper.
"""

import os
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.fp4_mla_kv import (
    FLASHINFER_FP4_MLA_ATTENTION_ENV,
    FP4_BLOCK_SIZE,
    FP4_MLA_KV_GLOBAL_SCALE,
    FP4_MLA_P_GLOBAL_SCALE,
    FP4_MLA_Q_RESIDUAL_DIM,
    FP4_MLA_TOKENS_PER_BLOCK,
    HP_BLOCK_SIZE,
    get_fp4_mla_decode_cache,
    get_fp4_mla_v_scale_pool_shape,
    get_fp4_mla_v_scale_pool_size,
    get_fp4_mla_v_scale_pool_view,
    is_flashinfer_fp4_mla_attention_enabled,
    run_fp4_mla_attention_decode,
    scatter_fp4_mla_kv_cache,
    update_hp_kv_for_fp4_mla,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping

_DataType = tensorrt_llm.bindings.DataType
_CacheType = tensorrt_llm.bindings.internal.batch_manager.CacheType
_TEST_GLOBAL_SCALE = FP4_MLA_KV_GLOBAL_SCALE


def _swizzled_sf_offset(row_idx: int, col_idx: int, sf_per_token: int) -> int:
    padded_cols = ((sf_per_token + 3) // 4) * 4
    return (
        col_idx % 4
        + (col_idx // 4) * (4 * 128)
        + (row_idx % 32) * 16
        + ((row_idx % 128) // 32) * 4
        + (row_idx // 128) * (128 * padded_cols)
    )


def _is_pre_blackwell() -> bool:
    return not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10


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
            start = sf_col * 16
            packed = fp4_bytes[row_idx, start // 2 : start // 2 + 8]
            low = packed & 0x0F
            high = (packed >> 4) & 0x0F
            vals = torch.empty(16, dtype=torch.float32, device=fp4_tensor.device)
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
            out[row_idx, start : start + 16] = vals * sf_flat[sf_offset].float() / global_scale

    return out


def _duplicate_tail_groups(tensor: torch.Tensor, residual_dim: int) -> torch.Tensor:
    prefix = tensor[..., :-residual_dim]
    tail = tensor[..., -residual_dim:].reshape(*tensor.shape[:-1], residual_dim // 16, 16)
    duplicated_tail = tail.repeat_interleave(2, dim=-2).reshape(
        *tensor.shape[:-1],
        residual_dim * 2,
    )
    return torch.cat((prefix, duplicated_tail), dim=-1)


def test_flashinfer_fp4_mla_attention_env(monkeypatch):
    monkeypatch.delenv(FLASHINFER_FP4_MLA_ATTENTION_ENV, raising=False)
    assert not is_flashinfer_fp4_mla_attention_enabled()

    monkeypatch.setenv(FLASHINFER_FP4_MLA_ATTENTION_ENV, "on")
    assert is_flashinfer_fp4_mla_attention_enabled()

    monkeypatch.setenv(FLASHINFER_FP4_MLA_ATTENTION_ENV, "0")
    assert not is_flashinfer_fp4_mla_attention_enabled()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp4_mla_v_scale_pool_view_shape():
    device = torch.device("cuda")
    num_layers = 2
    num_pages = 3
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    head_dim = kv_lora_rank + qk_rope_head_dim
    page_size = FP4_MLA_TOKENS_PER_BLOCK

    allocated_page_elems = get_fp4_mla_v_scale_pool_size(head_dim, page_size)
    pool = torch.empty(
        (num_layers, num_pages, allocated_page_elems),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    metadata = SimpleNamespace(page_size=page_size, fp4_mla_v_scale_pool=pool)

    view = get_fp4_mla_v_scale_pool_view(metadata, v_head_dim=kv_lora_rank)

    assert tuple(view.shape) == get_fp4_mla_v_scale_pool_shape(
        num_layers, num_pages, kv_lora_rank, page_size
    )
    assert view.data_ptr() == pool.data_ptr()
    assert view.numel() == num_layers * num_pages * kv_lora_rank * (page_size // 16)


def _build_metadata(kv_cache_manager, *, num_tokens, page_size, num_layers):
    """Build a minimal metadata namespace that satisfies the kernels' field
    expectations. Single sequence, single layer slice, no draft tokens."""
    device = torch.device("cuda")
    num_blocks = (num_tokens + page_size - 1) // page_size

    # Sequence 0 owns pages block_ids[:num_blocks] in the cache pool.
    block_ids = kv_cache_manager.get_batch_cache_indices([0])[0][:num_blocks]
    paged_kv_indices = torch.tensor(block_ids, dtype=torch.int32, device=device)
    paged_kv_indptr = torch.tensor([0, num_blocks], dtype=torch.int32, device=device)
    # Single-sequence decode: compact page range is [0, num_blocks).
    paged_kv_indptr_decode = torch.tensor([0, num_blocks], dtype=torch.int32, device=device)
    batch_indices = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    positions = torch.arange(num_tokens, dtype=torch.int32, device=device)

    # HP pool: one sequence slot, HP_BLOCK_SIZE * head_dim BF16 values per
    # (seq_slot, layer) cell.
    head_dim = kv_cache_manager.head_dim
    hp_pool = torch.zeros(
        (1, num_layers, 1, HP_BLOCK_SIZE * head_dim),
        dtype=torch.bfloat16,
        device=device,
    )

    seq_slots = torch.zeros(1, dtype=torch.int32, device=device)
    seq_slots_cpu = torch.zeros(1, dtype=torch.int32, device="cpu")
    kv_lens = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    prompt_lens_cuda = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    prompt_lens_cpu = torch.tensor([num_tokens], dtype=torch.int32)
    global_scale = torch.tensor([_TEST_GLOBAL_SCALE], dtype=torch.float32, device=device)

    return SimpleNamespace(
        kv_cache_manager=kv_cache_manager,
        batch_indices=batch_indices,
        positions=positions,
        paged_kv_indices=paged_kv_indices,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indptr_decode=paged_kv_indptr_decode,
        page_size=page_size,
        num_context_blocks=0,
        num_generation_blocks=num_blocks,
        num_contexts=1,
        num_seqs=1,
        high_precision_kv_pool=hp_pool,
        fp4_mla_v_scale_pool=kv_cache_manager.get_mla_v_scale_pool(),
        hp_pool_owners={},
        seq_slots=seq_slots,
        seq_slots_cpu=seq_slots_cpu,
        kv_lens_cuda_runtime=kv_lens,
        prompt_lens_cuda_runtime=prompt_lens_cuda,
        prompt_lens_cpu_runtime=prompt_lens_cpu,
        _fp4_mla_global_scale=global_scale,
        request_ids=[0],
        is_cuda_graph=False,
        is_warmup=False,
    )


def _build_multi_seq_metadata(kv_cache_manager, *, seq_lens, page_size, num_layers):
    device = torch.device("cuda")
    num_seqs = len(seq_lens)
    request_ids = list(range(num_seqs))
    block_ids_per_seq = kv_cache_manager.get_batch_cache_indices(request_ids)
    num_blocks = [(seq_len + page_size - 1) // page_size for seq_len in seq_lens]

    paged_kv_indices = torch.tensor(
        [
            block_id
            for seq_idx, seq_blocks in enumerate(block_ids_per_seq)
            for block_id in seq_blocks[: num_blocks[seq_idx]]
        ],
        dtype=torch.int32,
        device=device,
    )
    indptr = [0]
    for block_count in num_blocks:
        indptr.append(indptr[-1] + block_count)
    paged_kv_indptr = torch.tensor(indptr, dtype=torch.int32, device=device)

    batch_indices = torch.cat(
        [
            torch.full((seq_len,), seq_idx, dtype=torch.int32, device=device)
            for seq_idx, seq_len in enumerate(seq_lens)
        ]
    )
    positions = torch.cat(
        [torch.arange(seq_len, dtype=torch.int32, device=device) for seq_len in seq_lens]
    )

    head_dim = kv_cache_manager.head_dim
    hp_pool = torch.zeros(
        (num_seqs, num_layers, 1, HP_BLOCK_SIZE * head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    seq_slots = torch.arange(num_seqs, dtype=torch.int32, device=device)
    seq_slots_cpu = torch.arange(num_seqs, dtype=torch.int32, device="cpu")
    kv_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    prompt_lens_cuda = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    prompt_lens_cpu = torch.tensor(seq_lens, dtype=torch.int32)
    global_scale = torch.tensor([_TEST_GLOBAL_SCALE], dtype=torch.float32, device=device)

    return SimpleNamespace(
        kv_cache_manager=kv_cache_manager,
        batch_indices=batch_indices,
        positions=positions,
        paged_kv_indices=paged_kv_indices,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indptr_decode=paged_kv_indptr.clone(),
        page_size=page_size,
        num_context_blocks=0,
        num_generation_blocks=sum(num_blocks),
        num_contexts=num_seqs,
        num_seqs=num_seqs,
        num_blocks=num_blocks,
        high_precision_kv_pool=hp_pool,
        fp4_mla_v_scale_pool=kv_cache_manager.get_mla_v_scale_pool(),
        hp_pool_owners={},
        seq_slots=seq_slots,
        seq_slots_cpu=seq_slots_cpu,
        kv_lens_cuda_runtime=kv_lens,
        prompt_lens_cuda_runtime=prompt_lens_cuda,
        prompt_lens_cpu_runtime=prompt_lens_cpu,
        _fp4_mla_global_scale=global_scale,
        request_ids=request_ids,
        is_cuda_graph=False,
        is_warmup=False,
    )


def _build_fp4_mla_attention_decode_case(*, seq_lens, num_heads, seed):
    torch.manual_seed(seed)
    device = torch.device("cuda")

    kv_lora_rank = 512
    qk_rope_head_dim = 64
    head_dim = kv_lora_rank + qk_rope_head_dim
    page_size = FP4_MLA_TOKENS_PER_BLOCK
    num_layers = 1
    num_blocks = [(seq_len + page_size - 1) // page_size for seq_len in seq_lens]
    max_seq_len = max(page_size, max(seq_lens))
    max_tokens = max(page_size, sum(num_blocks) * page_size)

    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    kv_cache_manager = KVCacheManager(
        KvCacheConfig(max_tokens=max_tokens, enable_block_reuse=False),
        _CacheType.SELFKONLY,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=head_dim,
        tokens_per_block=page_size,
        max_seq_len=max_seq_len,
        max_batch_size=len(seq_lens),
        mapping=mapping,
        dtype=_DataType.NVFP4,
    )
    kv_cache_manager.add_dummy_requests(list(range(len(seq_lens))), seq_lens)
    kv_cache_manager.get_buffers(0).view(torch.uint8).zero_()
    kv_cache_manager.get_block_scale_buffers(0).zero_()

    metadata = _build_multi_seq_metadata(
        kv_cache_manager,
        seq_lens=seq_lens,
        page_size=page_size,
        num_layers=num_layers,
    )
    assert metadata.fp4_mla_v_scale_pool is not None

    latent = (
        torch.randn(sum(seq_lens), head_dim, dtype=torch.bfloat16, device=device) * 0.25
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
    torch.cuda.synchronize()

    metadata.num_contexts = 0
    q_nope = (
        torch.randn(len(seq_lens), num_heads, kv_lora_rank, dtype=torch.bfloat16, device=device)
        * 0.25
    ).clamp_(-1.0, 1.0)
    q_pe = (
        torch.randn(
            len(seq_lens),
            num_heads,
            qk_rope_head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.25
    ).clamp_(-1.0, 1.0)

    return kv_cache_manager, metadata, q_nope, q_pe, kv_lora_rank, qk_rope_head_dim


def _fp4_mla_attention_decode_reference(
    metadata,
    q_nope,
    q_pe,
    *,
    sm_scale,
    kv_lora_rank,
    qk_rope_head_dim,
):
    head_dim = kv_lora_rank + qk_rope_head_dim
    high_precision_kv_pool = metadata.high_precision_kv_pool
    metadata.high_precision_kv_pool = None
    try:
        dequant_cache = get_fp4_mla_decode_cache(
            metadata,
            layer_idx=0,
            local_layer=0,
            head_dim=head_dim,
            dtype=torch.bfloat16,
        )
    finally:
        metadata.high_precision_kv_pool = high_precision_kv_pool

    num_heads = q_nope.shape[1]
    global_scale = metadata._fp4_mla_global_scale
    q_full = torch.cat((q_nope, q_pe), dim=-1).reshape(-1, head_dim)
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
        sf_per_token=q_logical_dim // 16,
        global_scale=_TEST_GLOBAL_SCALE,
    )

    p_dequant = _dequant_fp4_swizzled(
        metadata._fp4_mla_attention_p_buf,
        metadata._fp4_mla_attention_p_sf_buf,
        logical_dim=metadata.page_size,
        sf_per_token=metadata.page_size // 16,
        global_scale=FP4_MLA_P_GLOBAL_SCALE,
    )

    indptr = metadata.paged_kv_indptr_decode.cpu().tolist()
    kv_lens = metadata.kv_lens_cuda_runtime.cpu().tolist()
    outputs = []
    exact_probs = []
    quantized_probs = []
    for seq_idx in range(metadata.num_seqs):
        kv_len = kv_lens[seq_idx]
        cache = dequant_cache[indptr[seq_idx] : indptr[seq_idx + 1]].reshape(-1, head_dim)[:kv_len]
        logical_k = _duplicate_tail_groups(cache.float(), FP4_MLA_Q_RESIDUAL_DIM)
        q_start = seq_idx * num_heads
        q = q_dequant[q_start : q_start + num_heads]
        probs = torch.softmax(torch.matmul(q, logical_k.transpose(0, 1)) * sm_scale, dim=-1)

        p_pages = []
        for page_rel in range(indptr[seq_idx + 1] - indptr[seq_idx]):
            page_start = page_rel * metadata.page_size
            valid_tokens = max(min(kv_len - page_start, metadata.page_size), 0)
            if valid_tokens == 0:
                continue
            compact_page = indptr[seq_idx] + page_rel
            p_start = compact_page * num_heads
            p_pages.append(p_dequant[p_start : p_start + num_heads, :valid_tokens])
        p = torch.cat(p_pages, dim=-1)

        exact_probs.append(probs)
        quantized_probs.append(p)
        outputs.append(torch.matmul(probs, cache[:, :kv_lora_rank].float()))
    return torch.stack(outputs, dim=0), exact_probs, quantized_probs


def _cuda_event_benchmark(fn, *, warmup_iters=10, iters=100):
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("num_tokens", "page_size"),
    [(20, 16), (32, 16), (17, 16), (129, 128), (144, 128)],
    ids=[
        "tail4_page16",
        "aligned32_page16",
        "tail1_page16",
        "tail1_page128",
        "aligned144_page128",
    ],
)
def test_fp4_mla_scatter_gather_roundtrip(num_tokens: int, page_size: int, monkeypatch):
    """Write BF16 latent through scatter + HP update, then read via the
    dequant-gather + HP-overlay path and verify the two halves of the output:

    * Positions that fall in the FP4 region (before the last ``kv_len % 16``
      tokens) must match the input up to NVFP4 quant error.
    * Positions covered by the HP overlay (the last ``kv_len % 16`` tokens)
      must match the input exactly (BF16 roundtrip).
    """
    monkeypatch.delenv(FLASHINFER_FP4_MLA_ATTENTION_ENV, raising=False)
    torch.manual_seed(0)
    device = torch.device("cuda")

    # MLA shapes (DeepSeek-V3-Lite style, scaled down).
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    head_dim = kv_lora_rank + qk_rope_head_dim  # 576, divisible by 16.
    num_layers = 1
    max_seq_len = max(64, ((num_tokens + page_size - 1) // page_size) * page_size)
    max_batch_size = 1

    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    # max_tokens must cover at least ceil(max_seq_len / page_size) pages.
    kv_cache_config = KvCacheConfig(max_tokens=max_seq_len, enable_block_reuse=False)
    kv_cache_manager = KVCacheManager(
        kv_cache_config,
        _CacheType.SELFKONLY,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=head_dim,
        tokens_per_block=page_size,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        mapping=mapping,
        dtype=_DataType.NVFP4,
    )
    try:
        kv_cache_manager.add_dummy_requests([0], [num_tokens])

        # Zero the underlying data + scale pools so stale bytes can't mask bugs.
        data_buf = kv_cache_manager.get_buffers(0).view(torch.uint8)
        data_buf.zero_()
        sf_buf = kv_cache_manager.get_block_scale_buffers(0)
        assert sf_buf is not None, "V1 NVFP4 manager must expose block scales"
        sf_buf.zero_()

        metadata = _build_metadata(
            kv_cache_manager, num_tokens=num_tokens, page_size=page_size, num_layers=num_layers
        )

        # Stay inside the configured global-scale range so FP4 quantization
        # does not saturate; narrower latents keep the FP4 tolerance reasonable.
        latent = (
            torch.randn(num_tokens, head_dim, dtype=torch.bfloat16, device=device) * 1.5
        ).clamp_(-5.0, 5.0)

        # --- Write path -------------------------------------------------
        scatter_fp4_mla_kv_cache(metadata, latent, layer_idx=0, token_offset=0)
        update_hp_kv_for_fp4_mla(metadata, latent, local_layer=0, phase="context")

        # Switch the metadata into "decode" shape for the read path. The
        # decode kernels gather the entire 0..num_tokens range (as if every
        # token were in the KV history for an upcoming decode step).
        metadata.num_contexts = 0
        metadata.num_seqs = 1

        # --- Read path --------------------------------------------------
        combined = get_fp4_mla_decode_cache(
            metadata,
            layer_idx=0,
            local_layer=0,
            head_dim=head_dim,
            dtype=torch.bfloat16,
        )
        # combined shape: [num_blocks, page_size, head_dim].
        flat = combined.reshape(-1, head_dim)[:num_tokens]

        # --- Assertions -------------------------------------------------
        tail = num_tokens % HP_BLOCK_SIZE
        fp4_end = num_tokens - tail  # exclusive

        # FP4-dequantized region: allow NVFP4 quant error. With unit global
        # scale, worst-case absolute error is ~0.5 of the largest FP4 step
        # within the value's block; 1.0 is a safe bound for the clamped
        # latent range [-5, 5].
        if fp4_end > 0:
            torch.testing.assert_close(
                flat[:fp4_end].float(),
                latent[:fp4_end].float(),
                atol=1.0,
                rtol=0.5,
                msg=f"FP4 region mismatch for num_tokens={num_tokens}",
            )

        # HP overlay region: must be exact BF16 roundtrip.
        if tail > 0:
            torch.testing.assert_close(
                flat[fp4_end:].float(),
                latent[fp4_end:].float(),
                atol=0.0,
                rtol=0.0,
                msg=f"HP overlay mismatch for num_tokens={num_tokens}",
            )
    finally:
        kv_cache_manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("page_size", "ctx_tokens"),
    [(16, 32), (128, 128)],
    ids=["page16", "page128"],
)
def test_fp4_mla_hp_overlay_generation_phase(page_size: int, ctx_tokens: int, monkeypatch):
    """After an aligned context, perform one decode step and verify that the
    decode token surfaces through the HP overlay at the first tail slot."""
    monkeypatch.delenv(FLASHINFER_FP4_MLA_ATTENTION_ENV, raising=False)
    torch.manual_seed(1)
    device = torch.device("cuda")

    kv_lora_rank = 512
    qk_rope_head_dim = 64
    head_dim = kv_lora_rank + qk_rope_head_dim
    num_layers = 1
    max_seq_len = max(64, page_size * 2)
    # Aligned context -> no tail until the first decode token.
    total_tokens = ctx_tokens + 1

    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    kv_cache_manager = KVCacheManager(
        KvCacheConfig(max_tokens=max_seq_len, enable_block_reuse=False),
        _CacheType.SELFKONLY,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=head_dim,
        tokens_per_block=page_size,
        max_seq_len=max_seq_len,
        max_batch_size=1,
        mapping=mapping,
        dtype=_DataType.NVFP4,
    )
    try:
        kv_cache_manager.add_dummy_requests([0], [total_tokens])

        kv_cache_manager.get_buffers(0).view(torch.uint8).zero_()
        kv_cache_manager.get_block_scale_buffers(0).zero_()

        # Write the full context (32 tokens) in one scatter, then the single
        # decode token separately, matching the production flow.
        metadata = _build_metadata(
            kv_cache_manager, num_tokens=total_tokens, page_size=page_size, num_layers=num_layers
        )

        ctx_latent = (
            torch.randn(ctx_tokens, head_dim, dtype=torch.bfloat16, device=device) * 1.5
        ).clamp_(-5.0, 5.0)
        gen_latent = (torch.randn(1, head_dim, dtype=torch.bfloat16, device=device) * 1.5).clamp_(
            -5.0, 5.0
        )

        # Context scatter + HP update: kv_len temporarily = 32.
        metadata.kv_lens_cuda_runtime = torch.tensor([ctx_tokens], dtype=torch.int32, device=device)
        metadata.prompt_lens_cuda_runtime = torch.tensor(
            [ctx_tokens], dtype=torch.int32, device=device
        )
        metadata.prompt_lens_cpu_runtime = torch.tensor([ctx_tokens], dtype=torch.int32)
        metadata.num_contexts = 1
        metadata.num_seqs = 1
        # Only ctx_tokens are visible to the scatter kernel this call.
        metadata.positions = torch.arange(ctx_tokens, dtype=torch.int32, device=device)
        metadata.batch_indices = torch.zeros(ctx_tokens, dtype=torch.int32, device=device)
        scatter_fp4_mla_kv_cache(metadata, ctx_latent, layer_idx=0, token_offset=0)
        update_hp_kv_for_fp4_mla(metadata, ctx_latent, local_layer=0, phase="context")

        # Decode scatter + HP update: append the single gen token at position 32.
        metadata.kv_lens_cuda_runtime = torch.tensor(
            [total_tokens], dtype=torch.int32, device=device
        )
        metadata.positions = torch.tensor([ctx_tokens], dtype=torch.int32, device=device)
        metadata.batch_indices = torch.zeros(1, dtype=torch.int32, device=device)
        metadata.num_contexts = 0
        metadata.prompt_lens_cuda_runtime = torch.tensor([1], dtype=torch.int32, device=device)
        metadata.prompt_lens_cpu_runtime = torch.tensor([1], dtype=torch.int32)
        scatter_fp4_mla_kv_cache(metadata, gen_latent, layer_idx=0, token_offset=0)
        update_hp_kv_for_fp4_mla(metadata, gen_latent, local_layer=0, phase="generation")

        # Now read the full 33-token history back.
        num_blocks = (total_tokens + page_size - 1) // page_size
        metadata.num_generation_blocks = num_blocks
        metadata.paged_kv_indices = torch.tensor(
            kv_cache_manager.get_batch_cache_indices([0])[0][:num_blocks],
            dtype=torch.int32,
            device=device,
        )
        metadata.paged_kv_indptr_decode = torch.tensor(
            [0, num_blocks], dtype=torch.int32, device=device
        )

        combined = get_fp4_mla_decode_cache(
            metadata,
            layer_idx=0,
            local_layer=0,
            head_dim=head_dim,
            dtype=torch.bfloat16,
        )
        flat = combined.reshape(-1, head_dim)[:total_tokens]

        # Position 32 is the lone tail token; HP overlay must return exactly
        # gen_latent[0].
        torch.testing.assert_close(
            flat[ctx_tokens].float(),
            gen_latent[0].float(),
            atol=0.0,
            rtol=0.0,
            msg="HP overlay did not restore the decode token",
        )

        # Positions 0..31 come from FP4 dequant; accept NVFP4 roundtrip noise.
        torch.testing.assert_close(
            flat[:ctx_tokens].float(),
            ctx_latent.float(),
            atol=1.0,
            rtol=0.5,
            msg="FP4 region mismatch on context tokens",
        )
    finally:
        kv_cache_manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("attention_env", ["0", "1"], ids=["linear_sf", "swizzled_sf"])
def test_fp4_mla_scatter_last_page_no_oob(attention_env: str, monkeypatch):
    """Scatter must not write past a page's scale region in either SF layout.

    With ``tokens_per_block=128`` the scatter's ``BLOCK_SF=64`` Triton block
    has ``SF_PER_TOKEN=36`` valid lanes plus 28 masked-out lanes. For those
    masked lanes the unconstrained offset (linear or swizzled) can exceed
    the per-page stride. If masked-lane addresses are not pinned in-bounds,
    the last physical page's masked stores fall past the sf_cache allocation,
    which crashes with "illegal memory access" on Blackwell.

    The SF layout is gated by ``FLASHINFER_FP4_MLA_ATTENTION_ENV``; cover both
    settings so a regression in either path is caught. The test writes tokens
    that land exclusively on the LAST physical page and asserts:
      1. No bytes outside the target page are modified.
      2. The data round-trips correctly through the dequant path (so the
         valid lanes still wrote the right values).
    """
    monkeypatch.setenv(FLASHINFER_FP4_MLA_ATTENTION_ENV, attention_env)
    torch.manual_seed(3)
    device = torch.device("cuda")

    kv_lora_rank = 512
    qk_rope_head_dim = 64
    head_dim = kv_lora_rank + qk_rope_head_dim
    page_size = FP4_MLA_TOKENS_PER_BLOCK
    num_pages = 4
    max_seq_len = num_pages * page_size
    num_layers = 1

    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    kv_cache_manager = KVCacheManager(
        KvCacheConfig(max_tokens=max_seq_len, enable_block_reuse=False),
        _CacheType.SELFKONLY,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=head_dim,
        tokens_per_block=page_size,
        max_seq_len=max_seq_len,
        max_batch_size=1,
        mapping=mapping,
        dtype=_DataType.NVFP4,
    )
    try:
        kv_cache_manager.add_dummy_requests([0], [max_seq_len])
        kv_cache_manager.get_buffers(0).view(torch.uint8).zero_()
        sf_buf = kv_cache_manager.get_block_scale_buffers(0)
        assert sf_buf is not None

        block_ids = kv_cache_manager.get_batch_cache_indices([0])[0][:num_pages]
        last_physical_page = block_ids[-1]
        # Sentinel non-target pages so a stray cross-page write is observable.
        sf_buf.view(torch.uint8).fill_(0xA5)
        sf_buf[last_physical_page].zero_()
        kv_cache_manager.get_buffers(0).view(torch.uint8).fill_(0xA5)
        kv_cache_manager.get_buffers(0).view(torch.uint8)[last_physical_page].zero_()
        snapshot_sf = sf_buf.view(torch.uint8).clone()
        snapshot_kv = kv_cache_manager.get_buffers(0).view(torch.uint8).clone()

        # Write tokens that land on the last physical page only.
        last_start = (num_pages - 1) * page_size
        num_tokens = page_size
        paged_kv_indices = torch.tensor(block_ids, dtype=torch.int32, device=device)
        paged_kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
        batch_indices = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        positions = torch.arange(
            last_start, last_start + num_tokens, dtype=torch.int32, device=device
        )

        # Single-sequence read-back metadata (one big seq covering all pages).
        metadata = _build_metadata(
            kv_cache_manager,
            num_tokens=max_seq_len,
            page_size=page_size,
            num_layers=num_layers,
        )
        # Override scatter-only fields to write just the last-page slice.
        metadata.batch_indices = batch_indices
        metadata.positions = positions
        metadata.paged_kv_indices = paged_kv_indices
        metadata.paged_kv_indptr = paged_kv_indptr

        latent = (
            torch.randn(num_tokens, head_dim, dtype=torch.bfloat16, device=device) * 1.5
        ).clamp_(-5.0, 5.0)
        scatter_fp4_mla_kv_cache(metadata, latent, layer_idx=0, token_offset=0)
        torch.cuda.synchronize()

        # Bytes for any non-target page must be unchanged.
        sf_after = sf_buf.view(torch.uint8)
        kv_after = kv_cache_manager.get_buffers(0).view(torch.uint8)
        for pid in range(sf_after.shape[0]):
            if pid == last_physical_page:
                continue
            torch.testing.assert_close(
                sf_after[pid],
                snapshot_sf[pid],
                atol=0,
                rtol=0,
                msg=f"scatter wrote to sf of non-target page {pid}",
            )
            torch.testing.assert_close(
                kv_after[pid],
                snapshot_kv[pid],
                atol=0,
                rtol=0,
                msg=f"scatter wrote to kv of non-target page {pid}",
            )

        # Read back via dequant and verify round-trip correctness.
        metadata.num_contexts = 0
        metadata.num_seqs = 1
        combined = get_fp4_mla_decode_cache(
            metadata,
            layer_idx=0,
            local_layer=0,
            head_dim=head_dim,
            dtype=torch.bfloat16,
        ).reshape(-1, head_dim)
        recovered = combined[last_start : last_start + num_tokens]
        torch.testing.assert_close(
            recovered.float(),
            latent.float(),
            atol=1.0,
            rtol=0.5,
            msg="last-page FP4 round-trip mismatch",
        )
    finally:
        kv_cache_manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp4_mla_dequant_invalid_page_ids_no_oob(monkeypatch):
    """Dequant should reject a short page-id slice and guard invalid physical
    pages in-kernel instead of doing unchecked page-stride arithmetic."""
    monkeypatch.delenv(FLASHINFER_FP4_MLA_ATTENTION_ENV, raising=False)
    device = torch.device("cuda")

    kv_lora_rank = 512
    qk_rope_head_dim = 64
    head_dim = kv_lora_rank + qk_rope_head_dim
    page_size = FP4_MLA_TOKENS_PER_BLOCK
    num_layers = 1

    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    kv_cache_manager = KVCacheManager(
        KvCacheConfig(max_tokens=page_size, enable_block_reuse=False),
        _CacheType.SELFKONLY,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=head_dim,
        tokens_per_block=page_size,
        max_seq_len=page_size,
        max_batch_size=1,
        mapping=mapping,
        dtype=_DataType.NVFP4,
    )
    try:
        kv_cache_manager.add_dummy_requests([0], [page_size])
        kv_cache_manager.get_buffers(0).view(torch.uint8).zero_()
        kv_cache_manager.get_block_scale_buffers(0).zero_()

        metadata = _build_metadata(
            kv_cache_manager,
            num_tokens=page_size,
            page_size=page_size,
            num_layers=num_layers,
        )
        metadata.num_contexts = 0
        metadata.num_seqs = 0
        metadata.num_context_blocks = 0
        metadata.num_generation_blocks = 1
        metadata.paged_kv_indices = torch.empty(0, dtype=torch.int32, device=device)

        with pytest.raises(RuntimeError, match="needs 1 decode page ids"):
            get_fp4_mla_decode_cache(
                metadata,
                layer_idx=0,
                local_layer=0,
                head_dim=head_dim,
                dtype=torch.bfloat16,
            )

        invalid_page = kv_cache_manager.get_buffers(0).shape[0]
        metadata.paged_kv_indices = torch.tensor([invalid_page], dtype=torch.int32, device=device)
        combined = get_fp4_mla_decode_cache(
            metadata,
            layer_idx=0,
            local_layer=0,
            head_dim=head_dim,
            dtype=torch.bfloat16,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(
            combined,
            torch.zeros_like(combined),
            atol=0,
            rtol=0,
        )
    finally:
        kv_cache_manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp4_mla_real_scatter_writes_shared_2d_scales(monkeypatch):
    """Real FP4 scatter shares 16x16 scales only where K and V overlap."""
    monkeypatch.setenv(FLASHINFER_FP4_MLA_ATTENTION_ENV, "1")
    torch.manual_seed(4)
    device = torch.device("cuda")

    kv_lora_rank = 512
    qk_rope_head_dim = 64
    head_dim = kv_lora_rank + qk_rope_head_dim
    page_size = FP4_MLA_TOKENS_PER_BLOCK
    num_tokens = 32
    num_layers = 1

    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    kv_cache_manager = KVCacheManager(
        KvCacheConfig(max_tokens=page_size, enable_block_reuse=False),
        _CacheType.SELFKONLY,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=head_dim,
        tokens_per_block=page_size,
        max_seq_len=page_size,
        max_batch_size=1,
        mapping=mapping,
        dtype=_DataType.NVFP4,
    )
    try:
        kv_cache_manager.add_dummy_requests([0], [num_tokens])
        kv_cache_manager.get_buffers(0).view(torch.uint8).zero_()
        kv_cache_manager.get_block_scale_buffers(0).zero_()

        metadata = _build_metadata(
            kv_cache_manager,
            num_tokens=num_tokens,
            page_size=page_size,
            num_layers=num_layers,
        )
        assert metadata.fp4_mla_v_scale_pool is not None

        row = (
            torch.arange(num_tokens, dtype=torch.float32, device=device) % HP_BLOCK_SIZE + 1.0
        ).view(num_tokens, 1)
        col = (
            torch.arange(head_dim, dtype=torch.float32, device=device) % FP4_MLA_TOKENS_PER_BLOCK
            + 1.0
        ).view(1, head_dim)
        latent = (row * col / 512.0).to(torch.bfloat16)

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

        physical_page = kv_cache_manager.get_batch_cache_indices([0])[0][0]
        sf_per_token = head_dim // 16
        sf_per_page = page_size // 16
        k_page = (
            kv_cache_manager.get_block_scale_buffers(0)
            .view(torch.float8_e4m3fn)[physical_page]
            .reshape(-1)
            .view(torch.uint8)
        )
        v_page = (
            get_fp4_mla_v_scale_pool_view(metadata, v_head_dim=kv_lora_rank)[0, physical_page]
            .reshape(-1)
            .view(torch.uint8)
        )

        for token_block in range(num_tokens // HP_BLOCK_SIZE):
            token_base = token_block * HP_BLOCK_SIZE
            for dim_block in (0, 10, 31):
                k_offsets = torch.tensor(
                    [
                        _swizzled_sf_offset(row_idx, dim_block, sf_per_token)
                        for row_idx in range(token_base, token_base + HP_BLOCK_SIZE)
                    ],
                    dtype=torch.long,
                    device=device,
                )
                k_bytes = k_page[k_offsets]
                assert bool((k_bytes[0] != 0).item())
                torch.testing.assert_close(
                    k_bytes,
                    k_bytes[0].expand_as(k_bytes),
                    atol=0,
                    rtol=0,
                    msg=f"K scales are not shared for dim block {dim_block}",
                )

                v_offsets = torch.tensor(
                    [
                        _swizzled_sf_offset(dim_block * 16 + row_idx, token_block, sf_per_page)
                        for row_idx in range(16)
                    ],
                    dtype=torch.long,
                    device=device,
                )
                v_bytes = v_page[v_offsets]
                torch.testing.assert_close(
                    v_bytes,
                    k_bytes[0].expand_as(v_bytes),
                    atol=0,
                    rtol=0,
                    msg=f"K/V scales disagree for dim block {dim_block}",
                )

            tail_dim_block = kv_lora_rank // FP4_BLOCK_SIZE
            tail_offsets = torch.tensor(
                [
                    _swizzled_sf_offset(row_idx, tail_dim_block, sf_per_token)
                    for row_idx in range(token_base, token_base + HP_BLOCK_SIZE)
                ],
                dtype=torch.long,
                device=device,
            )
            tail_bytes = k_page[tail_offsets]
            assert bool((tail_bytes[0] != 0).item())
            assert int(torch.unique(tail_bytes).numel()) > 1, "K-only tail scales must be per-token"
    finally:
        kv_cache_manager.shutdown()


@pytest.mark.skipif(_is_pre_blackwell(), reason="requires Blackwell FP4 tensor cores")
def test_fp4_mla_attention_decode_residual_qk_duplicates_k_tail(monkeypatch):
    """Residual-Q QK must use the same cached K tail for main and residual groups."""
    monkeypatch.setenv(FLASHINFER_FP4_MLA_ATTENTION_ENV, "1")
    torch.manual_seed(6)
    device = torch.device("cuda")

    kv_lora_rank = 512
    qk_rope_head_dim = 64
    head_dim = kv_lora_rank + qk_rope_head_dim
    page_size = FP4_MLA_TOKENS_PER_BLOCK
    num_tokens = page_size
    num_layers = 1
    num_heads = 1

    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    kv_cache_manager = KVCacheManager(
        KvCacheConfig(max_tokens=page_size, enable_block_reuse=False),
        _CacheType.SELFKONLY,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=head_dim,
        tokens_per_block=page_size,
        max_seq_len=page_size,
        max_batch_size=1,
        mapping=mapping,
        dtype=_DataType.NVFP4,
    )
    try:
        kv_cache_manager.add_dummy_requests([0], [num_tokens])
        kv_cache_manager.get_buffers(0).view(torch.uint8).zero_()
        kv_cache_manager.get_block_scale_buffers(0).zero_()

        metadata = _build_metadata(
            kv_cache_manager,
            num_tokens=num_tokens,
            page_size=page_size,
            num_layers=num_layers,
        )
        assert metadata.fp4_mla_v_scale_pool is not None

        token_pattern = torch.linspace(
            -0.4,
            0.4,
            num_tokens,
            dtype=torch.float32,
            device=device,
        ).view(num_tokens, 1)
        dim_pattern = torch.linspace(
            -0.7,
            0.7,
            FP4_MLA_Q_RESIDUAL_DIM,
            dtype=torch.float32,
            device=device,
        ).view(1, FP4_MLA_Q_RESIDUAL_DIM)
        latent = torch.zeros(num_tokens, head_dim, dtype=torch.bfloat16, device=device)
        latent[:, -FP4_MLA_Q_RESIDUAL_DIM:] = (token_pattern + dim_pattern).to(torch.bfloat16)

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

        metadata.num_contexts = 0
        metadata.num_seqs = 1

        q_nope = torch.zeros(1, num_heads, kv_lora_rank, dtype=torch.bfloat16, device=device)
        q_pe = (
            torch.linspace(
                -0.9,
                0.9,
                qk_rope_head_dim,
                dtype=torch.float32,
                device=device,
            )
            .view(1, num_heads, qk_rope_head_dim)
            .to(torch.bfloat16)
        )
        output = torch.empty(1, num_heads, kv_lora_rank, dtype=torch.bfloat16, device=device)
        sm_scale = 0.1

        run_fp4_mla_attention_decode(
            metadata,
            layer_idx=0,
            local_layer=0,
            q_nope=q_nope,
            q_pe=q_pe,
            output=output,
            sm_scale=sm_scale,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        )
        torch.cuda.synchronize()

        q_full = torch.cat((q_nope, q_pe), dim=-1).reshape(num_heads, head_dim)
        q_fp4, q_sf = torch.ops.trtllm.fp4_quantize_with_residual(
            q_full,
            metadata._fp4_mla_global_scale,
            FP4_MLA_Q_RESIDUAL_DIM,
            is_act=True,
        )
        q_logical_dim = head_dim + FP4_MLA_Q_RESIDUAL_DIM
        q_dequant = _dequant_fp4_swizzled(
            q_fp4,
            q_sf,
            logical_dim=q_logical_dim,
            sf_per_token=q_logical_dim // 16,
            global_scale=_TEST_GLOBAL_SCALE,
        )
        dequant_cache = get_fp4_mla_decode_cache(
            metadata,
            layer_idx=0,
            local_layer=0,
            head_dim=head_dim,
            dtype=torch.bfloat16,
        ).reshape(-1, head_dim)[:num_tokens]
        logical_k = _duplicate_tail_groups(dequant_cache.float(), FP4_MLA_Q_RESIDUAL_DIM)
        ref_scores = torch.matmul(q_dequant, logical_k.transpose(0, 1)) * sm_scale
        ref_probs = torch.softmax(ref_scores, dim=-1)
        probs = metadata._fp4_mla_attention_p_prob_buf[:num_heads, :num_tokens]

        torch.testing.assert_close(
            probs,
            ref_probs,
            atol=2e-2,
            rtol=2e-2,
            msg="FP4 MLA residual-Q probabilities did not match duplicated K-tail reference",
        )
    finally:
        kv_cache_manager.shutdown()


@pytest.mark.skipif(_is_pre_blackwell(), reason="requires Blackwell FP4 tensor cores")
def test_fp4_mla_attention_decode_multi_seq_matches_reference(monkeypatch):
    """Multiple decode sequences and heads must match a QK-softmax-PV reference."""
    monkeypatch.setenv(FLASHINFER_FP4_MLA_ATTENTION_ENV, "1")
    num_heads = 5
    seq_lens = [32, 128]

    (
        kv_cache_manager,
        metadata,
        q_nope,
        q_pe,
        kv_lora_rank,
        qk_rope_head_dim,
    ) = _build_fp4_mla_attention_decode_case(
        seq_lens=seq_lens,
        num_heads=num_heads,
        seed=7,
    )
    try:
        output = torch.empty_like(q_nope)
        sm_scale = 0.1
        run_fp4_mla_attention_decode(
            metadata,
            layer_idx=0,
            local_layer=0,
            q_nope=q_nope,
            q_pe=q_pe,
            output=output,
            sm_scale=sm_scale,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        )
        torch.cuda.synchronize()

        ref_output, exact_probs, quantized_probs = _fp4_mla_attention_decode_reference(
            metadata,
            q_nope,
            q_pe,
            sm_scale=sm_scale,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        )
        for seq_idx, (exact_prob, quantized_prob) in enumerate(zip(exact_probs, quantized_probs)):
            torch.testing.assert_close(
                quantized_prob,
                exact_prob,
                atol=8e-2,
                rtol=8e-2,
                msg=f"FP4 MLA attention probabilities diverged for sequence {seq_idx}",
            )
        torch.testing.assert_close(
            output.float(),
            ref_output,
            atol=1e-1,
            rtol=1e-1,
            msg="FP4 MLA attention decode output diverged from the QK-softmax-PV reference",
        )
    finally:
        kv_cache_manager.shutdown()


@pytest.mark.skipif(
    os.environ.get("TRTLLM_RUN_FP4_MLA_ATTENTION_BENCHMARK") != "1",
    reason=("Manual perf benchmark; set TRTLLM_RUN_FP4_MLA_ATTENTION_BENCHMARK=1 to run"),
)
@pytest.mark.skipif(_is_pre_blackwell(), reason="requires Blackwell FP4 tensor cores")
@pytest.mark.parametrize("batch_size", [16, 32, 64, 128, 256], ids=lambda x: f"bs{x}")
@pytest.mark.parametrize("seq_len", [8192], ids=lambda x: f"seq{x}")
def test_fp4_mla_attention_decode_perf_benchmark(
    batch_size,
    seq_len,
    monkeypatch,
):
    """Opt-in microbenchmark for ``run_fp4_mla_attention_decode``.

    Run manually with:
    ``TRTLLM_RUN_FP4_MLA_ATTENTION_BENCHMARK=1 pytest -s -k fp4_mla_attention_decode_perf``.
    """
    monkeypatch.setenv(FLASHINFER_FP4_MLA_ATTENTION_ENV, "1")
    num_heads = 128
    (
        kv_cache_manager,
        metadata,
        q_nope,
        q_pe,
        kv_lora_rank,
        qk_rope_head_dim,
    ) = _build_fp4_mla_attention_decode_case(
        seq_lens=[seq_len] * batch_size,
        num_heads=num_heads,
        seed=8,
    )
    try:
        output = torch.empty_like(q_nope)

        def run_decode():
            run_fp4_mla_attention_decode(
                metadata,
                layer_idx=0,
                local_layer=0,
                q_nope=q_nope,
                q_pe=q_pe,
                output=output,
                sm_scale=0.1,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
            )

        run_decode()
        torch.cuda.synchronize()
        avg_ms = _cuda_event_benchmark(run_decode, warmup_iters=10, iters=50)
        tokens = batch_size * seq_len
        qk_dim = kv_lora_rank + qk_rope_head_dim + FP4_MLA_Q_RESIDUAL_DIM
        pv_dim = kv_lora_rank
        matmul_flops = 2 * batch_size * num_heads * seq_len * (qk_dim + pv_dim)
        matmul_tflops = matmul_flops / avg_ms / 1e9
        print(
            "\nrun_fp4_mla_attention_decode "
            f"batch={batch_size} seq_len={seq_len} heads={num_heads}: "
            f"{avg_ms:.4f} ms, {tokens * num_heads / avg_ms / 1e3:.2f} M token-head/s, "
            f"{matmul_tflops:.2f} estimated matmul TFLOP/s"
        )
        assert torch.isfinite(output.float()).all()
    finally:
        kv_cache_manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp4_mla_shared_tile_rejects_unaligned_context_start(monkeypatch):
    """The no-dequant FP4 path requires context chunks to start on a 16-token boundary."""
    monkeypatch.setenv(FLASHINFER_FP4_MLA_ATTENTION_ENV, "1")
    torch.manual_seed(5)
    device = torch.device("cuda")

    kv_lora_rank = 512
    qk_rope_head_dim = 64
    head_dim = kv_lora_rank + qk_rope_head_dim
    page_size = FP4_MLA_TOKENS_PER_BLOCK
    num_layers = 1
    cached_tokens = 8
    new_tokens = 16
    total_tokens = cached_tokens + new_tokens

    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    kv_cache_manager = KVCacheManager(
        KvCacheConfig(max_tokens=page_size, enable_block_reuse=False),
        _CacheType.SELFKONLY,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=head_dim,
        tokens_per_block=page_size,
        max_seq_len=page_size,
        max_batch_size=1,
        mapping=mapping,
        dtype=_DataType.NVFP4,
    )
    try:
        kv_cache_manager.add_dummy_requests([0], [total_tokens])
        kv_cache_manager.get_buffers(0).view(torch.uint8).zero_()
        kv_cache_manager.get_block_scale_buffers(0).zero_()

        metadata = _build_metadata(
            kv_cache_manager,
            num_tokens=cached_tokens,
            page_size=page_size,
            num_layers=num_layers,
        )
        assert metadata.fp4_mla_v_scale_pool is not None

        new_latent = (
            torch.randn(new_tokens, head_dim, dtype=torch.bfloat16, device=device) * 1.5
        ).clamp_(-5.0, 5.0)
        metadata.kv_lens_cuda_runtime = torch.tensor(
            [total_tokens], dtype=torch.int32, device=device
        )
        metadata.prompt_lens_cuda_runtime = torch.tensor(
            [new_tokens], dtype=torch.int32, device=device
        )
        metadata.prompt_lens_cpu_runtime = torch.tensor([new_tokens], dtype=torch.int32)

        with pytest.raises(
            ValueError,
            match="start position.*16-token aligned",
        ):
            scatter_fp4_mla_kv_cache(
                metadata,
                new_latent,
                layer_idx=0,
                token_offset=0,
                phase="context",
                local_layer=0,
                v_head_dim=kv_lora_rank,
            )
    finally:
        kv_cache_manager.shutdown()
