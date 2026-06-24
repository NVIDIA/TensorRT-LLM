# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRTLLM FP4 MLA helper tests."""

import os
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm
import tensorrt_llm._torch.attention_backend.fmha.fp4_mla as fp4_mla_fmha
from tensorrt_llm._torch.attention_backend.fmha.fp4_mla import Fp4MlaFmha
from tensorrt_llm._torch.attention_backend.fp4_mla import (
    FP4_BLOCK_SIZE,
    FP4_MLA_ATTENTION_BACKEND_ENV,
    FP4_MLA_KV_GLOBAL_SCALE,
    FP4_MLA_P_GLOBAL_SCALE,
    FP4_MLA_Q_RESIDUAL_DIM,
    FP4_MLA_TOKENS_PER_BLOCK,
    HP_BLOCK_SIZE,
    _cutile_backend_available,
    _get_cutile_v_packed_cache,
    _maybe_update_cutile_v_packed_cache,
    get_fp4_mla_v_scale_pool_shape,
    get_fp4_mla_v_scale_pool_size,
    get_fp4_mla_v_scale_pool_view,
    repair_fp4_mla_hp_kv_for_mtp_rejection,
    run_fp4_mla_attention_decode,
    scatter_fp4_mla_kv_cache,
    update_hp_kv_for_fp4_mla,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization.mode import QuantMode

_DataType = tensorrt_llm.bindings.DataType
_CacheType = tensorrt_llm.bindings.internal.batch_manager.CacheType
_TEST_GLOBAL_SCALE = FP4_MLA_KV_GLOBAL_SCALE


def _fp4_mla_attn_for_availability(head_dim: int):
    return SimpleNamespace(
        is_mla_enable=True,
        quant_mode=int(QuantMode(0).set_fp4_kv_cache()),
        attention_chunk_size=None,
        predicted_tokens_per_seq=1,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=FP4_MLA_Q_RESIDUAL_DIM,
        v_head_dim=128,
        head_dim=head_dim,
    )


def test_fp4_mla_fmha_available_for_context_and_generation_head_dims(monkeypatch):
    monkeypatch.setattr(fp4_mla_fmha, "get_sm_version", lambda: 100)
    monkeypatch.setattr(fp4_mla_fmha, "is_sm_100f", lambda sm: True)
    monkeypatch.setattr(
        fp4_mla_fmha.torch,
        "ops",
        SimpleNamespace(trtllm=SimpleNamespace(fp4_quantize_with_residual=object())),
    )

    assert Fp4MlaFmha.is_available(_fp4_mla_attn_for_availability(128 + 64))
    assert Fp4MlaFmha.is_available(_fp4_mla_attn_for_availability(512 + 64))
    assert not Fp4MlaFmha.is_available(_fp4_mla_attn_for_availability(128))


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


def _is_cutile_unavailable() -> bool:
    return _is_pre_blackwell() or not _cutile_backend_available()


def _reset_triton_allocator() -> None:
    import triton
    import triton.runtime._allocation as triton_allocation

    triton.set_allocator(triton_allocation._NULL_ALLOCATOR)


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
        kv_lens_cuda_runtime=kv_lens,
        prompt_lens_cuda_runtime=prompt_lens_cuda,
        prompt_lens_cpu_runtime=prompt_lens_cpu,
        _fp4_mla_global_scale=global_scale,
        request_ids=request_ids,
        is_cuda_graph=False,
        is_warmup=False,
    )


def _materialize_reference_cache(metadata, layer_idx: int, head_dim: int) -> torch.Tensor:
    kv_cache = metadata.kv_cache_manager.get_buffers(layer_idx).view(torch.uint8)
    sf_cache = metadata.kv_cache_manager.get_block_scale_buffers(layer_idx).view(
        torch.float8_e4m3fn
    )
    pages = []
    src_page_ids = metadata.paged_kv_indices[
        metadata.num_context_blocks : metadata.num_context_blocks + metadata.num_generation_blocks
    ]
    for page_id in src_page_ids.tolist():
        fp4_page = kv_cache[page_id, 0, :, 0, :]
        sf_page = sf_cache[page_id]
        pages.append(
            _dequant_fp4_swizzled(
                fp4_page,
                sf_page,
                logical_dim=head_dim,
                sf_per_token=head_dim // FP4_BLOCK_SIZE,
                global_scale=_TEST_GLOBAL_SCALE,
            )
        )
    if not pages:
        return torch.empty(
            (0, metadata.page_size, head_dim), dtype=torch.float32, device=kv_cache.device
        )
    return torch.stack(pages, dim=0)


def _build_fp4_mla_attention_decode_case(*, seq_lens, num_heads, seed, query_len_per_seq=1):
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
    metadata.prompt_lens_cuda_runtime = torch.full(
        (len(seq_lens),), query_len_per_seq, dtype=torch.int32, device=device
    )
    metadata.prompt_lens_cpu_runtime = torch.full(
        (len(seq_lens),), query_len_per_seq, dtype=torch.int32
    )
    num_queries = len(seq_lens) * query_len_per_seq
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
    dequant_cache = _materialize_reference_cache(metadata, 0, head_dim)
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
        sf_per_token=q_logical_dim // FP4_BLOCK_SIZE,
        global_scale=_TEST_GLOBAL_SCALE,
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
    kv_lens = metadata.kv_lens_cuda_runtime.cpu().tolist()
    num_seqs = metadata.num_seqs - metadata.num_contexts
    query_len_per_seq = q_nope.shape[0] // num_seqs
    max_pages = max(indptr[seq_idx + 1] - indptr[seq_idx] for seq_idx in range(num_seqs))
    outputs = []
    exact_probs = []
    quantized_probs = []
    for seq_idx in range(num_seqs):
        kv_len = kv_lens[seq_idx]
        full_cache = dequant_cache[indptr[seq_idx] : indptr[seq_idx + 1]].reshape(-1, head_dim)
        for query_offset in range(query_len_per_seq):
            query_idx = seq_idx * query_len_per_seq + query_offset
            effective_kv_len = kv_len - (query_len_per_seq - 1 - query_offset)
            cache = full_cache[:effective_kv_len]
            logical_k = _duplicate_tail_groups(cache.float(), FP4_MLA_Q_RESIDUAL_DIM)
            q_start = query_idx * num_heads
            q = q_dequant[q_start : q_start + num_heads]
            probs = torch.softmax(torch.matmul(q, logical_k.transpose(0, 1)) * sm_scale, dim=-1)

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

            exact_probs.append(probs)
            quantized_probs.append(p)
            outputs.append(torch.matmul(p, cache[:, :kv_lora_rank].float()))
    return torch.stack(outputs, dim=0), exact_probs, quantized_probs


def _assert_fp4_mla_attention_decode_accuracy(
    monkeypatch,
    *,
    backend: str,
    num_heads: int,
    seq_lens: list[int],
    seed: int,
    check_probs: bool = False,
    query_len_per_seq: int = 1,
) -> None:
    _reset_triton_allocator()
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, backend)
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
        seed=seed,
        query_len_per_seq=query_len_per_seq,
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
        if check_probs:
            for seq_idx, (exact_prob, quantized_prob) in enumerate(
                zip(exact_probs, quantized_probs)
            ):
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
            atol=1.5e-1,
            rtol=1.5e-1,
            msg=f"{backend} FP4 MLA attention decode output diverged from reference",
        )
    finally:
        torch.cuda.synchronize()
        _reset_triton_allocator()
        kv_cache_manager.shutdown()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp4_mla_v_scale_pool_view_shape():
    device = torch.device("cuda")
    num_layers = 2
    num_pages = 3
    kv_lora_rank = 512
    page_size = FP4_MLA_TOKENS_PER_BLOCK

    allocated_page_elems = get_fp4_mla_v_scale_pool_size(kv_lora_rank, page_size)
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
    assert view.numel() == num_layers * num_pages * kv_lora_rank * (page_size // FP4_BLOCK_SIZE)


def _materialize_single_seq_cache_with_hp_tail(
    metadata, *, layer_idx: int, local_layer: int, head_dim: int, num_tokens: int
) -> torch.Tensor:
    flat = (
        _materialize_reference_cache(metadata, layer_idx, head_dim)
        .reshape(-1, head_dim)[:num_tokens]
        .to(torch.bfloat16)
    )
    tail = num_tokens % HP_BLOCK_SIZE
    if tail == 0:
        return flat

    seq_slot = int(metadata.seq_slots[0].item())
    pool_head_dim = metadata.high_precision_kv_pool.shape[-1] // HP_BLOCK_SIZE
    hp_view = metadata.high_precision_kv_pool[seq_slot, local_layer, 0, :].view(
        HP_BLOCK_SIZE, pool_head_dim
    )
    flat[-tail:] = hp_view[:tail, :head_dim]
    return flat


@pytest.mark.skipif(_is_pre_blackwell(), reason="requires Blackwell FP4 support")
@pytest.mark.parametrize(
    "num_tokens",
    [32, 129, 144],
    ids=["aligned32", "tail1", "aligned144"],
)
def test_fp4_mla_scatter_gather_roundtrip(num_tokens: int):
    torch.manual_seed(0)
    device = torch.device("cuda")
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    head_dim = kv_lora_rank + qk_rope_head_dim
    page_size = FP4_MLA_TOKENS_PER_BLOCK
    num_layers = 1
    max_seq_len = ((num_tokens + page_size - 1) // page_size) * page_size

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
        kv_cache_manager.add_dummy_requests([0], [num_tokens])
        kv_cache_manager.get_buffers(0).view(torch.uint8).zero_()
        kv_cache_manager.get_block_scale_buffers(0).zero_()
        metadata = _build_metadata(
            kv_cache_manager, num_tokens=num_tokens, page_size=page_size, num_layers=num_layers
        )
        latent = (
            torch.randn(num_tokens, head_dim, dtype=torch.bfloat16, device=device) * 0.25
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
        update_hp_kv_for_fp4_mla(metadata, latent, local_layer=0, phase="context")
        torch.cuda.synchronize()

        recovered = _materialize_single_seq_cache_with_hp_tail(
            metadata,
            layer_idx=0,
            local_layer=0,
            head_dim=head_dim,
            num_tokens=num_tokens,
        )
        tail = num_tokens % HP_BLOCK_SIZE
        fp4_end = num_tokens - tail
        if fp4_end > 0:
            torch.testing.assert_close(
                recovered[:fp4_end].float(), latent[:fp4_end].float(), atol=1.0, rtol=0.5
            )
        if tail > 0:
            torch.testing.assert_close(
                recovered[fp4_end:].float(), latent[fp4_end:].float(), atol=0, rtol=0
            )
    finally:
        kv_cache_manager.shutdown()


@pytest.mark.skipif(_is_pre_blackwell(), reason="requires Blackwell FP4 support")
def test_fp4_mla_hp_overlay_generation_phase():
    torch.manual_seed(1)
    device = torch.device("cuda")
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    head_dim = kv_lora_rank + qk_rope_head_dim
    page_size = FP4_MLA_TOKENS_PER_BLOCK
    ctx_tokens = page_size
    total_tokens = ctx_tokens + 1
    num_layers = 1

    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    kv_cache_manager = KVCacheManager(
        KvCacheConfig(max_tokens=page_size * 2, enable_block_reuse=False),
        _CacheType.SELFKONLY,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=head_dim,
        tokens_per_block=page_size,
        max_seq_len=page_size * 2,
        max_batch_size=1,
        mapping=mapping,
        dtype=_DataType.NVFP4,
    )
    try:
        kv_cache_manager.add_dummy_requests([0], [total_tokens])
        kv_cache_manager.get_buffers(0).view(torch.uint8).zero_()
        kv_cache_manager.get_block_scale_buffers(0).zero_()
        metadata = _build_metadata(
            kv_cache_manager, num_tokens=total_tokens, page_size=page_size, num_layers=num_layers
        )

        ctx_latent = (
            torch.randn(ctx_tokens, head_dim, dtype=torch.bfloat16, device=device) * 0.25
        ).clamp_(-1.0, 1.0)
        gen_latent = (torch.randn(1, head_dim, dtype=torch.bfloat16, device=device) * 0.25).clamp_(
            -1.0, 1.0
        )

        metadata.kv_lens_cuda_runtime = torch.tensor([ctx_tokens], dtype=torch.int32, device=device)
        metadata.prompt_lens_cuda_runtime = metadata.kv_lens_cuda_runtime
        metadata.prompt_lens_cpu_runtime = torch.tensor([ctx_tokens], dtype=torch.int32)
        metadata.positions = torch.arange(ctx_tokens, dtype=torch.int32, device=device)
        metadata.batch_indices = torch.zeros(ctx_tokens, dtype=torch.int32, device=device)
        scatter_fp4_mla_kv_cache(
            metadata,
            ctx_latent,
            layer_idx=0,
            token_offset=0,
            phase="context",
            local_layer=0,
            v_head_dim=kv_lora_rank,
        )
        update_hp_kv_for_fp4_mla(metadata, ctx_latent, local_layer=0, phase="context")

        metadata.kv_lens_cuda_runtime = torch.tensor(
            [total_tokens], dtype=torch.int32, device=device
        )
        metadata.prompt_lens_cuda_runtime = torch.tensor([1], dtype=torch.int32, device=device)
        metadata.prompt_lens_cpu_runtime = torch.tensor([1], dtype=torch.int32)
        metadata.positions = torch.tensor([ctx_tokens], dtype=torch.int32, device=device)
        metadata.batch_indices = torch.zeros(1, dtype=torch.int32, device=device)
        metadata.num_contexts = 0
        scatter_fp4_mla_kv_cache(
            metadata,
            gen_latent,
            layer_idx=0,
            token_offset=0,
            phase="generation",
            local_layer=0,
            v_head_dim=kv_lora_rank,
        )
        update_hp_kv_for_fp4_mla(metadata, gen_latent, local_layer=0, phase="generation")
        torch.cuda.synchronize()

        recovered = _materialize_single_seq_cache_with_hp_tail(
            metadata,
            layer_idx=0,
            local_layer=0,
            head_dim=head_dim,
            num_tokens=total_tokens,
        )
        torch.testing.assert_close(
            recovered[ctx_tokens].float(), gen_latent[0].float(), atol=0, rtol=0
        )
    finally:
        kv_cache_manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("preallocated_snapshot", [False, True])
def test_fp4_mla_hp_pool_restores_rejected_linear_mtp_tokens(preallocated_snapshot: bool):
    device = torch.device("cuda")
    head_dim = 8
    old_len = 30
    gen_len = 4
    accepted_len = 2
    initial_hp = (
        torch.arange(HP_BLOCK_SIZE * head_dim, dtype=torch.float32, device=device)
        .reshape(HP_BLOCK_SIZE, head_dim)
        .to(torch.bfloat16)
    )
    hp_pool = initial_hp.reshape(1, 1, 1, HP_BLOCK_SIZE * head_dim).clone()
    metadata = SimpleNamespace(
        high_precision_kv_pool=hp_pool,
        fp4_mla_hp_snapshot_pool=None,
        seq_slots=torch.tensor([0], dtype=torch.int32, device=device),
        batch_indices=torch.zeros(gen_len, dtype=torch.int32, device=device),
        positions=torch.arange(old_len, old_len + gen_len, dtype=torch.int32, device=device),
        kv_lens_cuda_runtime=torch.tensor([old_len + gen_len], dtype=torch.int32, device=device),
        prompt_lens_cuda_runtime=torch.tensor([gen_len], dtype=torch.int32, device=device),
        prompt_lens_cpu_runtime=torch.tensor([gen_len], dtype=torch.int32),
        num_contexts=0,
        num_seqs=1,
        num_tokens=gen_len,
        is_warmup=False,
    )
    if preallocated_snapshot:
        metadata.fp4_mla_hp_snapshot_pool = torch.empty_like(hp_pool)
    gen_latent = (
        torch.arange(gen_len * head_dim, dtype=torch.float32, device=device)
        .reshape(gen_len, head_dim)
        .add_(1000)
        .to(torch.bfloat16)
    )

    update_hp_kv_for_fp4_mla(metadata, gen_latent, local_layer=0, phase="generation")
    repair_fp4_mla_hp_kv_for_mtp_rejection(
        metadata,
        torch.tensor([accepted_len], dtype=torch.int32, device=device),
    )

    hp_view = hp_pool.view(HP_BLOCK_SIZE, head_dim)
    accepted_slots = [(old_len + idx) % HP_BLOCK_SIZE for idx in range(accepted_len)]
    rejected_slots = [(old_len + idx) % HP_BLOCK_SIZE for idx in range(accepted_len, gen_len)]
    for idx, slot in enumerate(accepted_slots):
        torch.testing.assert_close(hp_view[slot].float(), gen_latent[idx].float())
    for idx, slot in enumerate(rejected_slots, start=accepted_len):
        torch.testing.assert_close(
            hp_view[slot].float(),
            initial_hp[slot].float(),
            msg=f"rejected token {idx} was not restored",
        )
    if preallocated_snapshot:
        assert getattr(metadata, "_fp4_mla_mtp_hp_snapshots")
    else:
        assert getattr(metadata, "_fp4_mla_mtp_hp_snapshots") is None


@pytest.mark.skipif(_is_pre_blackwell(), reason="requires Blackwell FP4 support")
def test_fp4_mla_real_scatter_writes_shared_2d_scales():
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
            kv_cache_manager, num_tokens=num_tokens, page_size=page_size, num_layers=num_layers
        )
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
        sf_per_token = head_dim // FP4_BLOCK_SIZE
        sf_per_page = page_size // FP4_BLOCK_SIZE
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
                torch.testing.assert_close(k_bytes, k_bytes[0].expand_as(k_bytes), atol=0, rtol=0)

                v_offsets = torch.tensor(
                    [
                        _swizzled_sf_offset(
                            dim_block * FP4_BLOCK_SIZE + row_idx, token_block, sf_per_page
                        )
                        for row_idx in range(FP4_BLOCK_SIZE)
                    ],
                    dtype=torch.long,
                    device=device,
                )
                v_bytes = v_page[v_offsets]
                torch.testing.assert_close(v_bytes, k_bytes[0].expand_as(v_bytes), atol=0, rtol=0)

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
            assert int(torch.unique(tail_bytes).numel()) > 1
    finally:
        kv_cache_manager.shutdown()


@pytest.mark.skipif(_is_pre_blackwell(), reason="requires Blackwell FP4 support")
def test_fp4_mla_scatter_last_page_no_oob():
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
        kv_cache = kv_cache_manager.get_buffers(0).view(torch.uint8)
        sf_buf = kv_cache_manager.get_block_scale_buffers(0)
        assert sf_buf is not None

        block_ids = kv_cache_manager.get_batch_cache_indices([0])[0][:num_pages]
        last_physical_page = block_ids[-1]
        sf_buf.view(torch.uint8).fill_(0xA5)
        sf_buf[last_physical_page].zero_()
        kv_cache.fill_(0xA5)
        kv_cache[last_physical_page].zero_()
        snapshot_sf = sf_buf.view(torch.uint8).clone()
        snapshot_kv = kv_cache.clone()

        last_start = (num_pages - 1) * page_size
        paged_kv_indices = torch.tensor(block_ids, dtype=torch.int32, device=device)
        paged_kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
        metadata = _build_metadata(
            kv_cache_manager, num_tokens=max_seq_len, page_size=page_size, num_layers=num_layers
        )
        metadata.batch_indices = torch.zeros(page_size, dtype=torch.int32, device=device)
        metadata.positions = torch.arange(
            last_start, last_start + page_size, dtype=torch.int32, device=device
        )
        metadata.paged_kv_indices = paged_kv_indices
        metadata.paged_kv_indptr = paged_kv_indptr
        metadata.kv_lens_cuda_runtime = torch.tensor(
            [max_seq_len], dtype=torch.int32, device=device
        )
        metadata.prompt_lens_cuda_runtime = torch.tensor(
            [page_size], dtype=torch.int32, device=device
        )
        metadata.prompt_lens_cpu_runtime = torch.tensor([page_size], dtype=torch.int32)

        latent = (
            torch.randn(page_size, head_dim, dtype=torch.bfloat16, device=device) * 0.25
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

        sf_after = sf_buf.view(torch.uint8)
        kv_after = kv_cache
        for pid in range(sf_after.shape[0]):
            if pid == last_physical_page:
                continue
            torch.testing.assert_close(sf_after[pid], snapshot_sf[pid], atol=0, rtol=0)
            torch.testing.assert_close(kv_after[pid], snapshot_kv[pid], atol=0, rtol=0)

        recovered = _materialize_reference_cache(metadata, 0, head_dim).reshape(-1, head_dim)[
            last_start : last_start + page_size
        ]
        torch.testing.assert_close(recovered.float(), latent.float(), atol=1.0, rtol=0.5)
    finally:
        kv_cache_manager.shutdown()


def _fp4_mla_attention_decode_residual_qk_duplicates_k_tail_impl(monkeypatch):
    monkeypatch.setenv("TRTLLM_FP4_MLA_TRITON_PREPACK_V", "0")
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
            kv_cache_manager, num_tokens=num_tokens, page_size=page_size, num_layers=num_layers
        )

        token_pattern = torch.linspace(
            -0.4, 0.4, num_tokens, dtype=torch.float32, device=device
        ).view(num_tokens, 1)
        dim_pattern = torch.linspace(
            -0.7, 0.7, FP4_MLA_Q_RESIDUAL_DIM, dtype=torch.float32, device=device
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
        metadata.num_contexts = 0
        metadata.prompt_lens_cuda_runtime = torch.ones(1, dtype=torch.int32, device=device)
        metadata.prompt_lens_cpu_runtime = torch.ones(1, dtype=torch.int32)
        q_nope = torch.zeros(1, num_heads, kv_lora_rank, dtype=torch.bfloat16, device=device)
        q_pe = (
            torch.linspace(-0.9, 0.9, qk_rope_head_dim, dtype=torch.float32, device=device)
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
            sf_per_token=q_logical_dim // FP4_BLOCK_SIZE,
            global_scale=_TEST_GLOBAL_SCALE,
        )
        dequant_cache = _materialize_reference_cache(metadata, 0, head_dim).reshape(-1, head_dim)[
            :num_tokens
        ]
        logical_k = _duplicate_tail_groups(dequant_cache.float(), FP4_MLA_Q_RESIDUAL_DIM)
        ref_scores = torch.matmul(q_dequant, logical_k.transpose(0, 1)) * sm_scale
        ref_probs = torch.softmax(ref_scores, dim=-1)
        p_dequant = _dequant_fp4_swizzled(
            metadata._fp4_mla_attention_p_buf,
            metadata._fp4_mla_attention_p_sf_buf,
            logical_dim=metadata.page_size,
            sf_per_token=metadata.page_size // FP4_BLOCK_SIZE,
            global_scale=FP4_MLA_P_GLOBAL_SCALE,
        )
        probs = p_dequant[:num_heads, :num_tokens]
        torch.testing.assert_close(
            probs,
            ref_probs,
            atol=8e-2,
            rtol=8e-2,
            msg="FP4 MLA residual-Q probabilities did not match duplicated K-tail reference",
        )
    finally:
        torch.cuda.synchronize()
        _reset_triton_allocator()
        kv_cache_manager.shutdown()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


@pytest.mark.skipif(_is_pre_blackwell(), reason="requires Blackwell FP4 tensor cores")
def test_fp4_mla_attention_decode_matches_reference(monkeypatch):
    _assert_fp4_mla_attention_decode_accuracy(
        monkeypatch,
        backend="triton",
        num_heads=5,
        seq_lens=[32, 128],
        seed=7,
        check_probs=True,
    )


@pytest.mark.skipif(_is_pre_blackwell(), reason="requires Blackwell FP4 tensor cores")
def test_fp4_mla_attention_decode_linear_mtp_matches_reference(monkeypatch):
    _assert_fp4_mla_attention_decode_accuracy(
        monkeypatch,
        backend="triton",
        num_heads=5,
        seq_lens=[32, 128],
        seed=11,
        check_probs=True,
        query_len_per_seq=3,
    )


@pytest.mark.skipif(_is_pre_blackwell(), reason="requires Blackwell FP4 tensor cores")
def test_fp4_mla_attention_decode_residual_qk_duplicates_k_tail(monkeypatch):
    _fp4_mla_attention_decode_residual_qk_duplicates_k_tail_impl(monkeypatch)


@pytest.mark.skipif(
    _is_cutile_unavailable(),
    reason="requires Blackwell FP4 tensor cores and Triton tl.ext",
)
def test_fp4_mla_attention_decode_cutile_matches_reference(monkeypatch):
    _assert_fp4_mla_attention_decode_accuracy(
        monkeypatch,
        backend="cutile",
        num_heads=128,
        seq_lens=[32, 128],
        seed=7,
        check_probs=False,
    )


@pytest.mark.skipif(
    _is_cutile_unavailable(),
    reason="requires Blackwell FP4 tensor cores and Triton tl.ext",
)
def test_fp4_mla_attention_decode_cutile_shared_v_pack_matches_reference(monkeypatch):
    monkeypatch.setenv("TRTLLM_FP4_MLA_PERSISTENT_V_PACK", "1")
    monkeypatch.setenv("TRTLLM_FP4_MLA_SHARE_V_PACK_STORAGE", "1")
    _assert_fp4_mla_attention_decode_accuracy(
        monkeypatch,
        backend="cutile",
        num_heads=128,
        seq_lens=[128, 128],
        seed=17,
        check_probs=False,
    )


@pytest.mark.skipif(
    _is_cutile_unavailable(),
    reason="requires Blackwell FP4 tensor cores and Triton tl.ext",
)
def test_fp4_mla_attention_decode_cutile_grouped_tail_matches_reference(monkeypatch):
    monkeypatch.setenv("TRTLLM_FP4_MLA_PERSISTENT_V_PACK", "1")
    monkeypatch.setenv("TRTLLM_FP4_MLA_SHARE_V_PACK_STORAGE", "1")
    monkeypatch.setenv("TRTLLM_FP4_MLA_GROUP_PAGES", "8")
    _assert_fp4_mla_attention_decode_accuracy(
        monkeypatch,
        backend="cutile",
        num_heads=128,
        seq_lens=[9 * FP4_MLA_TOKENS_PER_BLOCK, 9 * FP4_MLA_TOKENS_PER_BLOCK],
        seed=19,
        check_probs=False,
    )


@pytest.mark.skipif(
    _is_cutile_unavailable(),
    reason="requires Blackwell FP4 tensor cores and Triton tl.ext",
)
def test_fp4_mla_attention_decode_cutile_linear_mtp_matches_reference(monkeypatch):
    _assert_fp4_mla_attention_decode_accuracy(
        monkeypatch,
        backend="cutile",
        num_heads=128,
        seq_lens=[32, 128],
        seed=13,
        check_probs=False,
        query_len_per_seq=3,
    )


@pytest.mark.skipif(
    _is_cutile_unavailable(),
    reason="requires Blackwell FP4 tensor cores and Triton tl.ext",
)
def test_fp4_mla_attention_decode_cutile_grouped_mtp_matches_reference(monkeypatch):
    monkeypatch.setenv("TRTLLM_FP4_MLA_PERSISTENT_V_PACK", "1")
    monkeypatch.setenv("TRTLLM_FP4_MLA_SHARE_V_PACK_STORAGE", "1")
    monkeypatch.setenv("TRTLLM_FP4_MLA_GROUP_PAGES", "8")
    _assert_fp4_mla_attention_decode_accuracy(
        monkeypatch,
        backend="cutile",
        num_heads=128,
        seq_lens=[9 * FP4_MLA_TOKENS_PER_BLOCK, 9 * FP4_MLA_TOKENS_PER_BLOCK],
        seed=23,
        check_probs=False,
        query_len_per_seq=4,
    )


@pytest.mark.skipif(
    _is_cutile_unavailable(),
    reason="requires Blackwell FP4 support and Triton tl.ext",
)
def test_fp4_mla_cutile_shared_v_pack_storage_is_layer_tagged(monkeypatch):
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, "cutile")
    monkeypatch.setenv("TRTLLM_FP4_MLA_PERSISTENT_V_PACK", "1")
    monkeypatch.setenv("TRTLLM_FP4_MLA_SHARE_V_PACK_STORAGE", "1")

    device = torch.device("cuda")
    metadata = SimpleNamespace()
    num_pages = 4
    page_size = FP4_MLA_TOKENS_PER_BLOCK
    v_head_dim = 512
    head_dim = v_head_dim + 64
    kv_cache = torch.randint(
        0,
        256,
        (num_pages, 1, page_size, 1, head_dim // 2),
        dtype=torch.uint8,
        device=device,
    )
    page_ids = torch.arange(num_pages, dtype=torch.int32, device=device)
    v_sf = torch.empty(
        (2, num_pages, v_head_dim, page_size // FP4_BLOCK_SIZE),
        dtype=torch.float8_e4m3fn,
        device=device,
    )

    _maybe_update_cutile_v_packed_cache(
        metadata,
        0,
        kv_cache,
        page_ids,
        v_head_dim=v_head_dim,
        page_size=page_size,
        local_layer=0,
        v_sf=v_sf[0],
    )
    torch.cuda.synchronize()
    shared = metadata._fp4_mla_attention_v_packed_buf
    shared_ptr = shared.data_ptr()
    assert not hasattr(metadata, "_fp4_mla_attention_v_packed_buf_l0")
    assert (
        _get_cutile_v_packed_cache(
            metadata,
            0,
            kv_cache,
            v_head_dim=v_head_dim,
            page_size=page_size,
            local_layer=0,
            v_sf=v_sf[0],
            page_ids=page_ids,
        )
        is not None
    )
    assert (
        _get_cutile_v_packed_cache(
            metadata,
            1,
            kv_cache,
            v_head_dim=v_head_dim,
            page_size=page_size,
            local_layer=1,
            v_sf=v_sf[1],
            page_ids=page_ids,
        )
        is None
    )

    _maybe_update_cutile_v_packed_cache(
        metadata,
        1,
        kv_cache,
        page_ids,
        v_head_dim=v_head_dim,
        page_size=page_size,
        local_layer=1,
        v_sf=v_sf[1],
    )
    torch.cuda.synchronize()
    assert metadata._fp4_mla_attention_v_packed_buf.data_ptr() == shared_ptr
    assert not hasattr(metadata, "_fp4_mla_attention_v_packed_buf_l1")
    assert (
        _get_cutile_v_packed_cache(
            metadata,
            0,
            kv_cache,
            v_head_dim=v_head_dim,
            page_size=page_size,
            local_layer=0,
            v_sf=v_sf[0],
            page_ids=page_ids,
        )
        is None
    )
    assert (
        _get_cutile_v_packed_cache(
            metadata,
            1,
            kv_cache,
            v_head_dim=v_head_dim,
            page_size=page_size,
            local_layer=0,
            v_sf=v_sf[0],
            page_ids=page_ids,
        )
        is None
    )
    assert (
        _get_cutile_v_packed_cache(
            metadata,
            1,
            kv_cache,
            v_head_dim=v_head_dim,
            page_size=page_size,
            local_layer=1,
            v_sf=v_sf[1],
            page_ids=page_ids,
        )
        is not None
    )


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


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


def _estimate_fp4_mla_attention_decode_mbu_bytes(
    *,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    page_size: int,
) -> int:
    fp4_block_size = 16
    head_block_size = 128
    bf16_bytes = 2
    fp32_bytes = 4

    q_input_dim = kv_lora_rank + qk_rope_head_dim
    qk_dim = q_input_dim + FP4_MLA_Q_RESIDUAL_DIM
    pages_per_seq = _ceil_div(seq_len, page_size)
    padded_seq_len = pages_per_seq * page_size
    head_blocks = _ceil_div(num_heads, head_block_size)
    q_rows = batch_size * num_heads

    qk_fp4_bytes_per_token = qk_dim // 2 + _ceil_div(qk_dim, fp4_block_size)
    v_fp4_bytes_per_token = kv_lora_rank // 2 + _ceil_div(kv_lora_rank, fp4_block_size)
    p_bytes_per_seq = padded_seq_len // 2 + _ceil_div(padded_seq_len, fp4_block_size)

    q_setup_bytes = q_rows * q_input_dim * bf16_bytes * 3 + q_rows * qk_fp4_bytes_per_token
    qk_cache_bytes = 2 * batch_size * head_blocks * padded_seq_len * qk_fp4_bytes_per_token
    qk_q_bytes = 2 * batch_size * num_heads * pages_per_seq * qk_fp4_bytes_per_token
    stats_bytes = batch_size * num_heads * 2 * fp32_bytes * (1 + pages_per_seq)
    p_prob_bytes = batch_size * num_heads * padded_seq_len * fp32_bytes * 2
    p_quant_bytes = 2 * batch_size * num_heads * p_bytes_per_seq
    pv_cache_bytes = batch_size * head_blocks * padded_seq_len * v_fp4_bytes_per_token
    output_bytes = q_rows * kv_lora_rank * bf16_bytes

    return (
        q_setup_bytes
        + qk_cache_bytes
        + qk_q_bytes
        + stats_bytes
        + p_prob_bytes
        + p_quant_bytes
        + pv_cache_bytes
        + output_bytes
    )


@pytest.mark.skipif(
    os.environ.get("TRTLLM_RUN_FP4_MLA_ATTENTION_BENCHMARK") != "1",
    reason="manual perf benchmark",
)
@pytest.mark.skipif(_is_pre_blackwell(), reason="requires Blackwell FP4 tensor cores")
@pytest.mark.parametrize("batch_size", [16, 32, 64, 128, 256], ids=lambda x: f"bs{x}")
@pytest.mark.parametrize("seq_len", [8192], ids=lambda x: f"seq{x}")
def test_fp4_mla_attention_decode_perf_benchmark(batch_size, seq_len, monkeypatch):
    backend = os.environ.get(FP4_MLA_ATTENTION_BACKEND_ENV, "triton")
    monkeypatch.setenv(FP4_MLA_ATTENTION_BACKEND_ENV, backend)
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
        mbu_bytes = _estimate_fp4_mla_attention_decode_mbu_bytes(
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            page_size=metadata.page_size,
        )
        mbu_tbps = mbu_bytes / avg_ms / 1e9
        print(
            "\nrun_fp4_mla_attention_decode "
            f"backend={backend} batch={batch_size} seq_len={seq_len} heads={num_heads}: "
            f"{avg_ms:.4f} ms, {tokens * num_heads / avg_ms / 1e3:.2f} M token-head/s, "
            f"{matmul_tflops:.2f} estimated matmul TFLOP/s, "
            f"estimated MBU={mbu_tbps:.2f} TB/s"
        )
        assert torch.isfinite(output.float()).all()
    finally:
        kv_cache_manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp4_mla_shared_tile_rejects_unaligned_context_start():
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
            kv_cache_manager, num_tokens=new_tokens, page_size=page_size, num_layers=num_layers
        )
        metadata.positions = torch.arange(
            cached_tokens, total_tokens, dtype=torch.int32, device=device
        )
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

        with pytest.raises(ValueError, match="start position.*16-token aligned"):
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
