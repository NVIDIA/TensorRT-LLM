# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Standalone benchmark for the FP4 MLA decode kernel.

Compares the Triton and Rubin CuTeDSL FP4 backends against the
trtllm-gen fp8 baselines. ``trtllm_fp8`` uses FlashInfer's wrapper, while
``trtllm_fp8_rubin`` uses TensorRT-LLM's native attention op.
Run with:
    python tests/microbenchmarks/bench_fp4_mla_decode.py [--batch B] [--seq S] [--heads H] [--q-len Q]

--q-len (alias --mtp-len) sets the number of query tokens per sequence (>1 for
MTP / speculative decoding); it defaults to 1 (plain decode).

Examples (no model weights required)::

    python tests/microbenchmarks/bench_fp4_mla_decode.py --backend cutedsl --batch 16 --seq 32768 --cuda-graph
    export TRTLLM_FP4_MLA_CUTEDSL_FUSED_V_TRANSPOSE=1
    python tests/microbenchmarks/bench_fp4_mla_decode.py --backend cutedsl --batch 16 \
        --seq 32768 --q-len 4 --cuda-graph --generation-step

CuTeDSL requires Rubin SM107 and its CTM/CuTeDSL runtime. Triton requires
FP4-capable GPU kernels. ``full`` times decode (with prequantized Q), or the
scatter + fused Q/RoPE update + decode when ``--generation-step`` is set.
Byte-rate estimates are diagnostic, not measured HBM bandwidth. This is a
standalone developer benchmark, not a registered CI performance gate.
"""

import argparse
import os
import time
from collections.abc import Callable
from types import SimpleNamespace

import torch

import tensorrt_llm
from tensorrt_llm._torch.attention.backends import fp4_mla as fp4_mla_backend
from tensorrt_llm._torch.attention.backends.fp4_mla import (
    FP4_BLOCK_SIZE,
    FP4_MLA_ATTENTION_BACKEND_ENV,
    FP4_MLA_K_RESIDUAL_DIM,
    FP4_MLA_KV_GLOBAL_SCALE,
    FP4_MLA_P_GLOBAL_SCALE,
    FP4_MLA_Q_GLOBAL_SCALE,
    FP4_MLA_Q_RESIDUAL_DIM,
    FP4_MLA_TOKENS_PER_BLOCK,
    _cutedsl_backend_available,
    _fp4_mla_attention_backend,
    _fp4_mla_cutedsl_fused_v_transpose_enabled,
    _get_fp4_mla_global_scale,
    run_fp4_mla_attention_decode,
    scatter_fp4_mla_kv_cache,
)
from tensorrt_llm._torch.attention.backends.fp4_mla.cache_manager import Fp4MlaKVCacheManagerV2
from tensorrt_llm._torch.attention.backends.fp4_mla.state import Fp4MlaState
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MTPDecodingConfig
from tensorrt_llm.mapping import Mapping

_DataType = tensorrt_llm.bindings.DataType
_CacheType = tensorrt_llm.bindings.internal.batch_manager.CacheType
_BENCH_KV_GLOBAL_SCALE = FP4_MLA_KV_GLOBAL_SCALE


def _swizzled_sf_offset(row_idx: int, col_idx: int, sf_per_token: int) -> int:
    padded_cols = ((sf_per_token + 3) // 4) * 4
    return (
        col_idx % 4
        + (col_idx // 4) * (4 * 128)
        + (row_idx % 32) * 16
        + ((row_idx % 128) // 32) * 4
        + (row_idx // 128) * (128 * padded_cols)
    )


def _dequant_fp4_swizzled(
    fp4_tensor: torch.Tensor,
    sf_tensor: torch.Tensor,
    *,
    logical_dim: int,
    sf_per_token: int,
    global_scale: float,
) -> torch.Tensor:
    """Decode E2M1 values and swizzled scales with tensor operations per page/batch."""
    fp4_values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=fp4_tensor.device,
    )
    fp4_bytes = fp4_tensor.view(torch.uint8)[:, : logical_dim // 2]
    num_rows = fp4_bytes.shape[0]
    codes = torch.stack((fp4_bytes & 0x0F, fp4_bytes >> 4), dim=-1).reshape(num_rows, logical_dim)
    values = fp4_values[codes.long()].reshape(num_rows, sf_per_token, FP4_BLOCK_SIZE)

    # Broadcast the scalar reference layout over all rows and scale groups.
    row = torch.arange(num_rows, device=fp4_tensor.device)[:, None]
    col = torch.arange(sf_per_token, device=fp4_tensor.device)[None, :]
    padded_cols = ((sf_per_token + 3) // 4) * 4
    sf_offsets = (
        col % 4
        + (col // 4) * (4 * 128)
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + (row // 128) * (128 * padded_cols)
    )
    sf_flat = sf_tensor.view(torch.float8_e4m3fn).reshape(-1).float()
    scales = sf_flat[sf_offsets]
    return (values * scales[..., None] / global_scale).reshape(num_rows, logical_dim)


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
    q: torch.Tensor, k: torch.Tensor, k_residual: torch.Tensor, residual_dim: int
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


def _build_multi_seq_metadata(
    kv_cache_manager: Fp4MlaKVCacheManagerV2, *, seq_lens: list[int], page_size: int, layer_idx: int
) -> SimpleNamespace:
    device = torch.device("cuda")
    num_seqs = len(seq_lens)
    request_ids = list(range(num_seqs))
    block_ids_per_seq = kv_cache_manager.get_batch_cache_indices(
        request_ids,
        layer_idx=layer_idx,
    )
    page_spec = kv_cache_manager.get_fp4_mla_page_table_spec(layer_idx)
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
    kv_global_scale = torch.tensor([_BENCH_KV_GLOBAL_SCALE], dtype=torch.float32, device=device)
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
            num_context_blocks=num_seqs * max_blocks_per_seq,
            num_generation_blocks=0,
            num_sequences=num_seqs,
            num_blocks=None,
            hp_pool=hp_pool,
            v_scale_pool=kv_cache_manager.get_mla_v_scale_pool(),
            q_global_scale=q_global_scale,
            kv_global_scale=kv_global_scale,
        ),
    )


def _materialize_reference_cache_storage(
    metadata: SimpleNamespace, layer_idx: int, head_dim: int
) -> torch.Tensor:
    kv_cache, sf_cache = metadata.kv_cache_manager.get_fp4_mla_cache_buffers(layer_idx)
    sf_cache = sf_cache.view(torch.float8_e4m3fn)
    storage_head_dim = kv_cache.shape[-1] * 2
    static_global_scale = float(_get_fp4_mla_global_scale(metadata, kv_cache.device).item())
    pages = []
    dequantized_pages = {}
    page_rows = metadata.fp4_mla_state.paged_kv_indices.view(
        metadata.num_seqs,
        metadata.fp4_mla_state.page_table_stride,
    )
    # Preserve the fixed row stride used by paged_kv_indptr_decode, including
    # padding for shorter requests. The reference masks padding by KV length.
    src_page_ids = page_rows.reshape(-1)
    for page_id in src_page_ids.tolist():
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
    if not pages:
        return torch.empty(
            (0, metadata.page_size, storage_head_dim),
            dtype=torch.float32,
            device=kv_cache.device,
        )
    return torch.stack(pages, dim=0)


def _materialize_reference_cache(
    metadata: SimpleNamespace, layer_idx: int, head_dim: int
) -> torch.Tensor:
    return _materialize_reference_cache_storage(metadata, layer_idx, head_dim)[..., :head_dim]


def _materialize_reference_k_residual(
    metadata: SimpleNamespace, layer_idx: int, *, head_dim: int
) -> torch.Tensor:
    storage = _materialize_reference_cache_storage(metadata, layer_idx, head_dim)
    return storage[..., head_dim : head_dim + FP4_MLA_K_RESIDUAL_DIM]


def _build_fp4_mla_attention_decode_case(
    *,
    seq_lens: list[int],
    num_heads: int,
    seed: int,
    query_len_per_seq: int = 1,
    num_layers: int = 1,
    layer_idx: int = 0,
    local_layer: int | None = None,
) -> tuple[Fp4MlaKVCacheManagerV2, SimpleNamespace, torch.Tensor, torch.Tensor, int, int]:
    local_layer = layer_idx if local_layer is None else local_layer
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

    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    spec_config = (
        MTPDecodingConfig(max_draft_len=query_len_per_seq - 1) if query_len_per_seq > 1 else None
    )
    kv_cache_manager = Fp4MlaKVCacheManagerV2(
        KvCacheConfig(
            max_tokens=max_tokens,
            dtype="nvfp4",
            enable_block_reuse=False,
            host_cache_size=0,
        ),
        _CacheType.SELFKONLY,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=head_dim,
        tokens_per_block=page_size,
        max_seq_len=max_seq_len,
        max_batch_size=len(seq_lens),
        mapping=mapping,
        dtype=_DataType.NVFP4,
        spec_config=spec_config,
        max_num_tokens=max_tokens,
        pretrained_config=SimpleNamespace(kv_lora_rank=kv_lora_rank),
    )
    kv_cache_manager.add_dummy_requests(list(range(len(seq_lens))), seq_lens)
    expected_storage_head_dim = head_dim + (
        FP4_MLA_K_RESIDUAL_DIM if _fp4_mla_attention_backend() in ("triton", "cutedsl") else 0
    )
    for current_layer in range(num_layers):
        kv_cache, sf_cache = kv_cache_manager.get_fp4_mla_cache_buffers(current_layer)
        kv_cache.zero_()
        sf_cache.zero_()
        assert kv_cache.shape[-1] * 2 == expected_storage_head_dim
        assert sf_cache.shape[-1] == expected_storage_head_dim // FP4_BLOCK_SIZE

    metadata = _build_multi_seq_metadata(
        kv_cache_manager,
        seq_lens=seq_lens,
        page_size=page_size,
        layer_idx=layer_idx,
    )
    assert metadata.fp4_mla_state.v_scale_pool is not None
    persistent_pool_base = kv_cache_manager.get_mla_v_packed_pool_base()
    if persistent_pool_base is not None:
        persistent_pool_base.zero_()
    metadata.fp4_mla_state.v_scale_pool.zero_()

    latent = (
        torch.randn(sum(seq_lens), head_dim, dtype=torch.bfloat16, device=device) * 0.25
    ).clamp_(-1.0, 1.0)
    scatter_fp4_mla_kv_cache(
        metadata,
        latent,
        layer_idx=layer_idx,
        token_offset=0,
        phase="context",
        local_layer=local_layer,
        v_head_dim=kv_lora_rank,
    )
    # Decode reference exercises the quantized cache without an HP overlay.
    metadata.fp4_mla_state.hp_pool.zero_()
    torch.cuda.synchronize()

    metadata.num_contexts = 0
    metadata.fp4_mla_state.num_context_blocks = 0
    metadata.fp4_mla_state.num_generation_blocks = (
        len(seq_lens) * metadata.fp4_mla_state.page_table_stride
    )
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
    metadata: SimpleNamespace,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    *,
    sm_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    layer_idx: int = 0,
    local_layer: int = 0,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    head_dim = kv_lora_rank + qk_rope_head_dim
    dequant_cache = _materialize_reference_cache(metadata, layer_idx, head_dim)
    dequant_k_residual = None
    if _fp4_mla_attention_backend() in ("triton", "cutedsl"):
        dequant_k_residual = _materialize_reference_k_residual(
            metadata,
            layer_idx,
            head_dim=head_dim,
        )
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
            probs = torch.softmax(
                torch.matmul(logical_q, logical_k.transpose(0, 1)) * sm_scale,
                dim=-1,
            )

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
            outputs.append(torch.matmul(p, v_cache.float()))
    return torch.stack(outputs, dim=0), exact_probs, quantized_probs


BACKEND_CHOICES = (
    "trtllm_fp8",
    "trtllm_fp8_rubin",
    "triton",
    "cutedsl",
)


def _bench(fn: Callable[[], object], warmup: int = 10, iters: int = 50) -> float:
    for _ in range(warmup):
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


def _bench_with_untimed_setup(
    fn: Callable[[], object],
    setup: Callable[[], object],
    warmup: int = 10,
    iters: int = 50,
    queue_delay_cycles: int = 0,
) -> float:
    """Benchmark ``fn`` after a same-stream setup excluded from event timing."""
    for _ in range(warmup):
        setup()
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        setup()
        if queue_delay_cycles:
            torch.cuda._sleep(queue_delay_cycles)
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return sum(start.elapsed_time(end) for start, end in zip(starts, ends)) / iters


def _capture_cuda_graph(
    fn: Callable[[], object], device: torch.device
) -> tuple[torch.cuda.CUDAGraph, torch.cuda.Stream]:
    """Warm and capture the GPU work submitted by ``fn`` on a side stream."""
    current_stream = torch.cuda.current_stream(device)
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(current_stream)
    with torch.cuda.stream(capture_stream):
        fn()
    current_stream.wait_stream(capture_stream)
    torch.cuda.synchronize(device)

    cuda_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cuda_graph, stream=capture_stream):
        fn()
    torch.cuda.synchronize(device)
    return cuda_graph, capture_stream


def _bench_cuda_graph(
    replay: Callable[[], object], device: torch.device, warmup: int = 10, iters: int = 50
) -> tuple[float, float]:
    """Return per-replay CUDA-event and host-wall times in milliseconds."""
    for _ in range(warmup):
        replay()
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        replay()
    end.record()
    torch.cuda.synchronize(device)
    event_ms = start.elapsed_time(end) / iters

    wall_start_ns = time.perf_counter_ns()
    for _ in range(iters):
        replay()
    torch.cuda.synchronize(device)
    wall_ms = (time.perf_counter_ns() - wall_start_ns) / 1_000_000.0 / iters
    return event_ms, wall_ms


PAGE_SIZE = FP4_MLA_TOKENS_PER_BLOCK


def _fp4_mla_lazy_rebase_stats(exact_probs: list[torch.Tensor]) -> SimpleNamespace:
    """Classify lazy resident-O rebases from exact quantized-QK probabilities."""
    kernel = fp4_mla_backend._fp4_mla_cutedsl_kernel_module()
    kv_tile = kernel.KV_TILE
    rebase_threshold = kernel.SMEM_P4_LAZY_ANCHOR_REBASE_LOG2

    rebase_counts = []
    max_anchor_delta_log2 = 0.0
    for probabilities in exact_probs:
        log2_probabilities = torch.log2(probabilities.float())
        num_rows = log2_probabilities.shape[0]
        true_row_max = torch.full(
            (num_rows,), -torch.inf, dtype=torch.float32, device=probabilities.device
        )
        row_anchor = torch.full_like(true_row_max, -torch.inf)
        row_rebases = torch.zeros_like(true_row_max, dtype=torch.int32)
        for tile_start in range(0, log2_probabilities.shape[1], kv_tile):
            tile_row_max = (
                log2_probabilities[:, tile_start : tile_start + kv_tile].max(dim=1).values
            )
            candidate_row_max = torch.maximum(true_row_max, tile_row_max)
            finite_pair = torch.isfinite(row_anchor) & torch.isfinite(candidate_row_max)
            first_finite = ~torch.isfinite(row_anchor) & torch.isfinite(candidate_row_max)
            anchor_delta_log2 = torch.where(
                finite_pair,
                candidate_row_max - row_anchor,
                torch.zeros_like(candidate_row_max),
            )
            finite_delta = anchor_delta_log2[finite_pair]
            if finite_delta.numel():
                max_anchor_delta_log2 = max(max_anchor_delta_log2, float(finite_delta.max().item()))
            rebase = finite_pair & (anchor_delta_log2 > rebase_threshold)
            row_rebases += rebase.to(torch.int32)
            row_anchor = torch.where(first_finite | rebase, candidate_row_max, row_anchor)
            true_row_max = candidate_row_max
        rebase_counts.append(row_rebases.cpu())

    if not rebase_counts:
        raise ValueError("exact_probs must contain at least one query")
    counts = torch.cat(rebase_counts)
    return SimpleNamespace(
        total_rows=int(counts.numel()),
        rebase_rows=int((counts > 0).sum().item()),
        rebase_total=int(counts.sum().item()),
        rebase_max_per_row=int(counts.max().item()),
        max_anchor_delta_log2=max_anchor_delta_log2,
        threshold_log2=float(rebase_threshold),
    )


_CUTEDSL_LAUNCHER_TENSOR_NAMES = (
    "q_fp4",
    "q_sf",
    "kv_cache",
    "sf_cache",
    "v_packed",
    "v_sf",
    "global_scale",
    "src_page_ids",
    "paged_kv_indptr_decode",
    "kv_lens",
    "output",
)


def _launcher_tensor_pointers(tensors: tuple[object, ...]) -> dict[str, int | None]:
    """Record stable launch addresses, including fused-V's absent sidecar."""
    if len(tensors) != len(_CUTEDSL_LAUNCHER_TENSOR_NAMES):
        raise RuntimeError(
            f"CuTeDSL launcher requires {len(_CUTEDSL_LAUNCHER_TENSOR_NAMES)} "
            f"positional tensor arguments, got {len(tensors)}"
        )
    pointers = {}
    for name, tensor in zip(_CUTEDSL_LAUNCHER_TENSOR_NAMES, tensors):
        if name == "v_packed" and tensor is None:
            pointers[name] = None
        elif isinstance(tensor, torch.Tensor):
            pointers[name] = int(tensor.data_ptr())
        else:
            raise TypeError(f"CuTeDSL launcher argument {name} must be a tensor")
    return pointers


class RetainedBenchmarkState:
    """Keep one benchmark allocation and captured Graph alive for A/B checks."""

    def __init__(
        self,
        kv_cache_manager: Fp4MlaKVCacheManagerV2,
        launcher_tensors: tuple[torch.Tensor | None, ...],
        keepalive: dict[str, object],
    ) -> None:
        self._kv_cache_manager = kv_cache_manager
        self._launcher_tensors = tuple(launcher_tensors)
        self._keepalive = keepalive
        self.launcher_tensor_ptrs = _launcher_tensor_pointers(self._launcher_tensors)

    def shutdown(self) -> None:
        """Synchronize, destroy Graph references, and release the retained manager."""
        manager = self._kv_cache_manager
        if manager is None:
            return
        output = self._keepalive.get("output")
        if isinstance(output, torch.Tensor) and output.is_cuda:
            torch.cuda.synchronize(output.device)
        self._keepalive["cuda_graph"] = None
        self._keepalive["capture_stream"] = None
        self._launcher_tensors = ()
        self._kv_cache_manager = None
        self._keepalive.clear()
        manager.shutdown()


def _seq_lens_for_batch(batch: int, seq: int) -> list[int]:
    return [seq] * batch


def _seq_label(seq_lens: list[int]) -> str:
    return f"{seq_lens[0]}-{seq_lens[-1]}" if len(seq_lens) > 1 else str(seq_lens[0])


def _causal_token_pairs(seq_lens: list[int], q_len: int) -> int:
    return sum(
        max(seq_len - (q_len - 1 - query_offset), 0)
        for seq_len in seq_lens
        for query_offset in range(q_len)
    )


def _kernel_io_bytes(
    seq_lens: list[int],
    heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    q_len: int = 1,
    k_residual_dim: int = 0,
) -> int:
    """Estimate logical input/output bytes; this is not measured HBM traffic."""
    batch = len(seq_lens)
    token_pairs = _causal_token_pairs(seq_lens, q_len)
    q_head_dim = kv_lora_rank + qk_rope_head_dim + FP4_MLA_Q_RESIDUAL_DIM
    k_head_dim = kv_lora_rank + qk_rope_head_dim + k_residual_dim
    q_sf_per_token = q_head_dim // FP4_BLOCK_SIZE
    k_sf_per_token = k_head_dim // FP4_BLOCK_SIZE
    pages = sum((seq_len + PAGE_SIZE - 1) // PAGE_SIZE for seq_len in seq_lens)
    sf_per_page = PAGE_SIZE // FP4_BLOCK_SIZE

    q_fp4 = batch * q_len * heads * q_head_dim // 2
    q_sf = batch * q_len * heads * q_sf_per_token
    kv_cache = token_pairs * k_head_dim // 2
    k_sf_cache = token_pairs * k_sf_per_token
    v_packed = token_pairs * kv_lora_rank // 2
    v_sf_cache = q_len * pages * kv_lora_rank * sf_per_page
    out = batch * q_len * heads * kv_lora_rank * 2  # bf16/half
    return q_fp4 + q_sf + kv_cache + k_sf_cache + v_packed + v_sf_cache + out


def _fixed_tile_request_bytes(
    seq_lens: list[int],
    heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    q_len: int = 1,
    k_residual_dim: int = 0,
) -> int:
    """Estimate static global requests made by the current K256 kernel.

    This implementation-specific diagnostic rounds every query to the launch
    maximum K256 tile count. It is neither logical algorithm traffic nor a
    hardware DRAM counter, and therefore must not be used as a stable gate.
    """
    batch = len(seq_lens)
    queries = batch * q_len
    q_head_dim = kv_lora_rank + qk_rope_head_dim + FP4_MLA_Q_RESIDUAL_DIM
    k_head_dim = kv_lora_rank + qk_rope_head_dim + k_residual_dim
    physical_k = ((max(seq_lens) + 255) // 256) * 256

    q_fp4 = queries * heads * q_head_dim // 2
    # Both CTAs independently request the complete Q scale vector.
    q_sf = 2 * queries * heads * (q_head_dim // FP4_BLOCK_SIZE)
    k_fp4 = queries * physical_k * k_head_dim // 2
    k_sf = queries * physical_k * (k_head_dim // FP4_BLOCK_SIZE)
    v_fp4 = queries * physical_k * kv_lora_rank // 2
    v_sf = queries * physical_k * (kv_lora_rank // FP4_BLOCK_SIZE)
    out = queries * heads * kv_lora_rank * 2
    return q_fp4 + q_sf + k_fp4 + k_sf + v_fp4 + v_sf + out


def _tma_smem_completion_bytes(
    seq_lens: list[int],
    heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    q_len: int = 1,
    k_residual_dim: int = 0,
) -> int:
    """Estimate TMA shared-memory destination bytes for diagnostics only.

    Relative to static global requests, K and V scale multicast contributes a
    destination in each CTA. This is not a global-memory or HBM byte count.
    """
    global_requests = _fixed_tile_request_bytes(
        seq_lens,
        heads,
        kv_lora_rank,
        qk_rope_head_dim,
        q_len,
        k_residual_dim,
    )
    queries = len(seq_lens) * q_len
    physical_k = ((max(seq_lens) + 255) // 256) * 256
    multicast_scale_destination = (
        queries
        * physical_k
        * (
            (kv_lora_rank + qk_rope_head_dim + k_residual_dim) // FP4_BLOCK_SIZE
            + kv_lora_rank // FP4_BLOCK_SIZE
        )
    )
    return global_requests + multicast_scale_destination


def _trtllm_mla_io_bytes(
    seq_lens: list[int],
    heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    q_len: int = 1,
    elem_bytes: int = 2,
) -> int:
    """Estimate logical bytes for the trtllm-gen MLA decode baseline.

    ``elem_bytes`` is the byte width of the Q and KV-cache elements (2 for bf16,
    1 for fp8).  The output is always written as bf16 (2 B/elem).
    """
    batch = len(seq_lens)
    token_pairs = _causal_token_pairs(seq_lens, q_len)
    head_dim = kv_lora_rank + qk_rope_head_dim
    q = batch * q_len * heads * head_dim * elem_bytes
    kv = token_pairs * head_dim * elem_bytes  # ckv + kpe paged caches
    out = batch * q_len * heads * kv_lora_rank * 2
    return q + kv + out


def run_one_trtllm(
    batch: int, seq: int, heads: int, q_len: int = 1, warmup: int = 0, iters: int = 1
) -> float:
    """Fp8 baseline using the FlashInfer trtllm-gen MLA decode kernel.

    Feeds fp8 (e4m3) Q and KV cache so the kernel uses fp8 tensor cores
    (output stays bf16).
    """
    import flashinfer

    device = torch.device("cuda")
    label = "trtllm_fp8"
    io_dtype = torch.float8_e4m3fn
    elem_bytes = 1
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    qk_nope_head_dim = 128  # DeepSeek-V3 default; only used for fused scale convention.
    head_dim_qk = kv_lora_rank + qk_rope_head_dim
    page_size = 64  # trtllm-gen MLA decode only supports page_size of 32 or 64.
    seq_lens_list = _seq_lens_for_batch(batch, seq)
    max_seq = max(seq_lens_list)
    blocks_per_seq = [(seq_len + page_size - 1) // page_size for seq_len in seq_lens_list]
    max_blocks_per_seq = max(blocks_per_seq)
    total_pages = sum(blocks_per_seq)

    torch.manual_seed(8)
    # query layout: [batch, q_len, heads, kv_lora_rank + qk_rope_head_dim]
    query = torch.randn(batch, q_len, heads, head_dim_qk, dtype=torch.bfloat16, device=device)
    # kv_cache layout: [num_pages, page_size, head_dim_ckv + head_dim_kpe]
    kv_cache = torch.randn(total_pages, page_size, head_dim_qk, dtype=torch.bfloat16, device=device)
    # Quantize to fp8 e4m3 (randn ~ N(0,1) is well within e4m3 range).
    query = query.to(io_dtype)
    kv_cache = kv_cache.to(io_dtype)

    block_tables = torch.zeros((batch, max_blocks_per_seq), dtype=torch.int32, device=device)
    page_start = 0
    for batch_idx, num_blocks in enumerate(blocks_per_seq):
        block_tables[batch_idx, :num_blocks] = torch.arange(
            page_start, page_start + num_blocks, dtype=torch.int32, device=device
        )
        page_start += num_blocks
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.int8, device=device).view(-1, 4)

    output = torch.empty(batch, q_len, heads, kv_lora_rank, dtype=torch.bfloat16, device=device)

    # bmm1_scale folds q_scale * k_scale * sm_scale / sqrt(head_dim_qk); a
    # representative value (q_scale = k_scale = 1.0 for the fp8 unit-scale tensors).
    def run() -> None:
        flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq,
            out=output,
            bmm1_scale=0.1,
            bmm2_scale=1.0,
            backend="trtllm-gen",
        )

    run()
    torch.cuda.synchronize()
    avg_ms = _bench(run, warmup=warmup, iters=iters)
    qk_dim = kv_lora_rank + qk_rope_head_dim
    pv_dim = kv_lora_rank
    flops = 2 * heads * _causal_token_pairs(seq_lens_list, q_len) * (qk_dim + pv_dim)
    tflops = flops / avg_ms / 1e9
    bytes_per_call = _trtllm_mla_io_bytes(
        seq_lens_list, heads, kv_lora_rank, qk_rope_head_dim, q_len, elem_bytes
    )
    gb_s = bytes_per_call / (avg_ms * 1e-3) / 1e9
    print(
        f"backend={label:>12s} bs={batch:>3d} seq={_seq_label(seq_lens_list):>11s} "
        f"heads={heads:>3d} qlen={q_len:>2d}: "
        f"{avg_ms:>7.3f} ms  {tflops:>6.2f} TFLOP/s  "
        f"Est.IO {gb_s:>6.1f} GB/s",
        flush=True,
    )
    return avg_ms


def run_one_trtllm_rubin(
    batch: int,
    seq: int,
    heads: int,
    q_len: int = 1,
    warmup: int = 0,
    iters: int = 1,
    queue_delay_cycles: int = 0,
    use_cuda_graph: bool = False,
    graph_timing: str = "event",
) -> float:
    """FP8 baseline using TensorRT-LLM's native SM107 trtllm-gen path."""
    import tensorrt_llm
    from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
    from tensorrt_llm._torch.attention.backends.interface import (
        AttentionInputType,
        MLAParams,
        PositionalEmbeddingParams,
        RopeParams,
    )
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
    from tensorrt_llm._utils import str_dtype_to_binding, torch_dtype_to_str
    from tensorrt_llm.functional import PositionEmbeddingType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    if q_len > seq:
        raise ValueError(f"q_len ({q_len}) must not exceed seq ({seq})")
    if queue_delay_cycles < 0:
        raise ValueError("queue delay cycles must be non-negative")
    if graph_timing not in ("event", "wall"):
        raise ValueError(f"unsupported CUDA Graph timing mode: {graph_timing}")

    device = torch.device("cuda")
    label = "trtllm_fp8_rubin"
    io_dtype = torch.float8_e4m3fn
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    qk_nope_head_dim = 128
    q_lora_rank = 1536
    v_head_dim = 128
    head_dim_qk = kv_lora_rank + qk_rope_head_dim
    page_size = 32
    seq_lens_list = _seq_lens_for_batch(batch, seq)
    request_ids = list(range(batch))
    past_seq = seq - q_len
    max_tokens = batch * ((seq + page_size - 1) // page_size) * page_size
    mapping = Mapping(world_size=1, tp_size=1, rank=0)

    kv_cache_manager = KVCacheManager(
        KvCacheConfig(max_tokens=max_tokens, enable_block_reuse=False),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=head_dim_qk,
        tokens_per_block=page_size,
        max_seq_len=seq,
        max_batch_size=batch,
        mapping=mapping,
        dtype=str_dtype_to_binding(torch_dtype_to_str(io_dtype)),
    )
    try:
        kv_cache_manager.add_dummy_requests(request_ids, [seq] * batch)
        kv_cache_manager.get_buffers(0).zero_()

        metadata = TrtllmAttention.Metadata(
            seq_lens=torch.full((batch,), q_len, dtype=torch.int32),
            request_ids=request_ids,
            max_num_requests=batch,
            num_contexts=0,
            prompt_lens=[past_seq] * batch,
            max_num_tokens=batch * q_len,
            kv_cache_manager=kv_cache_manager,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[past_seq] * batch,
            ),
            mapping=mapping,
        )
        metadata.prepare()

        attention = TrtllmAttention(
            layer_idx=0,
            num_heads=heads,
            head_dim=head_dim_qk,
            num_kv_heads=1,
            quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8.value),
            q_scaling=1.0,
            pos_embd_params=PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=RopeParams(
                    dim=qk_rope_head_dim,
                    max_positions=seq,
                    original_max_positions=seq,
                    duplicate_data=True,
                ),
                is_neox=False,
            ),
            mla_params=MLAParams(
                q_lora_rank=q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                qk_nope_head_dim=qk_nope_head_dim,
                v_head_dim=v_head_dim,
                predicted_tokens_per_seq=q_len,
            ),
        )
        # Force the THOP adapter. Its SM107 AttentionOp selects the native
        # TRTLLM-gen runner; other adapters may route back through FlashInfer.
        attention._fmha_manager.fmha_libs = [FallbackFmha(attention)]

        torch.manual_seed(8)
        num_tokens = batch * q_len
        fused_q = torch.randn(
            num_tokens,
            heads * head_dim_qk,
            dtype=torch.bfloat16,
            device=device,
        )
        q_pe = torch.randn(
            num_tokens,
            heads,
            qk_rope_head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        latent_cache = torch.randn(
            num_tokens,
            head_dim_qk,
            dtype=torch.bfloat16,
            device=device,
        )
        cu_q_seqlens = torch.empty(batch + 1, dtype=torch.int32, device=device)
        cu_kv_seqlens = torch.empty(batch + 1, dtype=torch.int32, device=device)
        fmha_scheduler_counter = torch.empty(1, dtype=torch.uint32, device=device)
        mla_bmm1_scale = torch.empty(2, dtype=torch.float32, device=device)
        mla_bmm2_scale = torch.empty(1, dtype=torch.float32, device=device)
        quant_q_buffer = torch.empty(
            num_tokens,
            heads * head_dim_qk,
            dtype=torch.uint8,
            device=device,
        )
        output = torch.empty(
            num_tokens,
            heads * kv_lora_rank,
            dtype=torch.bfloat16,
            device=device,
        )

        def prepare() -> None:
            attention.mla_rope_generation(
                fused_q,
                q_pe,
                latent_cache,
                metadata,
                cu_q_seqlens,
                cu_kv_seqlens,
                fmha_scheduler_counter,
                mla_bmm1_scale,
                mla_bmm2_scale,
                quant_q_buffer,
            )

        def run() -> None:
            attention.forward(
                fused_q,
                None,
                None,
                metadata,
                attention_input_type=AttentionInputType.generation_only,
                latent_cache=latent_cache,
                q_pe=q_pe,
                cu_q_seqlens=cu_q_seqlens,
                cu_kv_seqlens=cu_kv_seqlens,
                fmha_scheduler_counter=fmha_scheduler_counter,
                mla_bmm1_scale=mla_bmm1_scale,
                mla_bmm2_scale=mla_bmm2_scale,
                quant_q_buffer=quant_q_buffer,
                output=output,
            )

        graph_event_ms = None
        graph_wall_ms = None
        prepare()
        run()
        torch.cuda.synchronize()
        if not bool(torch.isfinite(output).all().item()):
            raise RuntimeError("native TRTLLM FP8 MLA output contains non-finite values")
        if use_cuda_graph:
            prepare()
            torch.cuda.synchronize(device)
            cuda_graph, capture_stream = _capture_cuda_graph(run, device)

            def replay_graph() -> None:
                cuda_graph.replay()

            output.fill_(float("nan"))
            torch.cuda.synchronize(device)
            replay_graph()
            torch.cuda.synchronize(device)
            if not bool(torch.isfinite(output).all().item()):
                raise RuntimeError("native TRTLLM FP8 CUDA Graph output contains non-finite values")
            graph_event_ms, graph_wall_ms = _bench_cuda_graph(
                replay_graph,
                device,
                warmup=warmup,
                iters=iters,
            )
            avg_ms = graph_event_ms if graph_timing == "event" else graph_wall_ms
            del capture_stream
        else:
            avg_ms = _bench_with_untimed_setup(
                run,
                prepare,
                warmup=warmup,
                iters=iters,
                queue_delay_cycles=queue_delay_cycles,
            )
    finally:
        kv_cache_manager.shutdown()

    qk_dim = kv_lora_rank + qk_rope_head_dim
    pv_dim = kv_lora_rank
    flops = 2 * heads * _causal_token_pairs(seq_lens_list, q_len) * (qk_dim + pv_dim)
    tflops = flops / avg_ms / 1e9
    bytes_per_call = _trtllm_mla_io_bytes(
        seq_lens_list,
        heads,
        kv_lora_rank,
        qk_rope_head_dim,
        q_len,
        elem_bytes=1,
    )
    gb_s = bytes_per_call / (avg_ms * 1e-3) / 1e9
    graph_label = f" cuda_graph=True graph_timing={graph_timing}" if use_cuda_graph else ""
    print(
        f"backend={label:>18s} bs={batch:>3d} seq={_seq_label(seq_lens_list):>11s} "
        f"heads={heads:>3d} qlen={q_len:>2d}{graph_label}: "
        f"{avg_ms:>7.3f} ms  {tflops:>6.2f} TFLOP/s  "
        f"Est.IO {gb_s:>6.1f} GB/s",
        flush=True,
    )
    if graph_event_ms is not None and graph_wall_ms is not None:
        print(
            f"cuda_graph event_us={graph_event_ms * 1000.0:.6f} "
            f"host_wall_us={graph_wall_ms * 1000.0:.6f}",
            flush=True,
        )
    elif queue_delay_cycles:
        print(
            f"fused queue delay={queue_delay_cycles} cycles (excluded from the event interval)",
            flush=True,
        )
    return avg_ms


class _FusedLaunchTimer:
    """Time only compiled CuTeDSL fused launches on the active CUDA stream."""

    def __init__(self, device: torch.device, iters: int, queue_delay_cycles: int = 0) -> None:
        self.device = device
        self.queue_delay_cycles = queue_delay_cycles
        self.enabled = False
        self.count = 0
        self.starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        self.ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    def wrap(self, compiled: Callable[..., object]) -> Callable[..., object]:
        def timed_launch(*args: object, **kwargs: object) -> object:
            if not self.enabled:
                return compiled(*args, **kwargs)
            if self.count >= len(self.starts):
                raise RuntimeError("observed more timed fused launches than expected")
            if self.queue_delay_cycles:
                torch.cuda._sleep(self.queue_delay_cycles)
            stream = torch.cuda.current_stream(self.device)
            self.starts[self.count].record(stream)
            result = compiled(*args, **kwargs)
            self.ends[self.count].record(stream)
            self.count += 1
            return result

        return timed_launch

    def mean_ms(self) -> float:
        if self.count != len(self.starts):
            raise RuntimeError(
                f"expected {len(self.starts)} timed fused launches, got {self.count}"
            )
        return (
            sum(start.elapsed_time(end) for start, end in zip(self.starts, self.ends)) / self.count
        )


def run_one(
    batch: int,
    seq: int,
    heads: int,
    backend: str,
    q_len: int = 1,
    warmup: int = 0,
    iters: int = 1,
    timing_scope: str = "full",
    fused_queue_delay_cycles: int = 0,
    report_correction: bool = False,
    use_cuda_graph: bool = False,
    generation_step: bool = False,
    retain_state: bool = False,
    graph_timing: str = "event",
) -> float | tuple[float, RetainedBenchmarkState]:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iters <= 0:
        raise ValueError("iters must be positive")
    if q_len <= 0:
        raise ValueError("query length must be positive")
    if generation_step and q_len > seq:
        raise ValueError("generation-step query length must not exceed the sequence length")
    if timing_scope not in ("full", "fused", "cupti"):
        raise ValueError(f"unsupported timing scope: {timing_scope}")
    if fused_queue_delay_cycles < 0:
        raise ValueError("fused queue delay cycles must be non-negative")
    if graph_timing not in ("event", "wall"):
        raise ValueError(f"unsupported CUDA Graph timing mode: {graph_timing}")
    if fused_queue_delay_cycles and timing_scope != "fused":
        raise ValueError("fused queue delay requires fused timing")
    if use_cuda_graph and fused_queue_delay_cycles:
        raise ValueError("fused queue delay is not used for CUDA Graph replay")
    if use_cuda_graph and backend != "cutedsl":
        raise ValueError("CUDA Graph replay requires the CuTeDSL backend")
    if use_cuda_graph and timing_scope == "cupti":
        raise ValueError("CUDA Graph replay does not support CUPTI timing")
    if generation_step and not (backend == "cutedsl" and use_cuda_graph and timing_scope == "full"):
        raise ValueError(
            "generation-step timing requires the CuTeDSL backend, CUDA Graph, and full timing"
        )
    if timing_scope in ("fused", "cupti") and backend != "cutedsl":
        raise ValueError(f"{timing_scope} timing is only supported by the CuTeDSL backend")
    if retain_state and backend != "cutedsl":
        raise ValueError("retained benchmark state requires the CuTeDSL backend")
    os.environ[FP4_MLA_ATTENTION_BACKEND_ENV] = backend
    seq_lens = _seq_lens_for_batch(batch, seq)
    (
        kv_cache_manager,
        metadata,
        q_nope,
        q_pe,
        kv_lora_rank,
        qk_rope_head_dim,
    ) = _build_fp4_mla_attention_decode_case(
        seq_lens=seq_lens,
        num_heads=heads,
        seed=8,
        query_len_per_seq=q_len,
    )
    metadata.is_cuda_graph = use_cuda_graph
    # The synthetic benchmark never changes its sequence or append lengths.
    # Bind the authoritative runtime tensors as an already-populated result so
    # timing excludes the production MTP length-correction helper.
    metadata.fp4_mla_state.generation_kv_lens = metadata.kv_lens_cuda_runtime[:batch]
    metadata.fp4_mla_state.generation_append_lens = metadata.prompt_lens_cuda_runtime[:batch]
    metadata.fp4_mla_state.generation_lengths_num_tokens = batch * q_len
    metadata.fp4_mla_state.generation_lengths_num_seqs = batch
    metadata.fp4_mla_state.generation_lengths_num_contexts = metadata.num_contexts
    metadata.fp4_mla_state.generation_lengths_capture_recorded = True
    if use_cuda_graph:
        # Synthetic benchmark page tables are immutable and never grow on replay.
        metadata.fp4_mla_state._paged_kv_indices = metadata.fp4_mla_state.paged_kv_indices
    generation_latent = None
    generation_rotary_cos_sin = None
    persistent_v_pack = False
    if generation_step:
        context_lens = torch.tensor(
            [seq_len - q_len for seq_len in seq_lens],
            dtype=torch.int32,
            device=q_nope.device,
        )
        metadata.num_ctx_tokens = 0
        metadata.fp4_mla_state.batch_indices = torch.arange(
            batch, dtype=torch.int32, device=q_nope.device
        ).repeat_interleave(q_len)
        metadata.fp4_mla_state.positions = (
            context_lens[:, None]
            + torch.arange(q_len, dtype=torch.int32, device=q_nope.device)[None, :]
        ).reshape(-1)
        persistent_v_packed = fp4_mla_backend._get_fp4_mla_v_packed_pool(metadata, 0)
        if persistent_v_packed is None and not _fp4_mla_cutedsl_fused_v_transpose_enabled():
            kv_cache_manager.shutdown()
            raise RuntimeError(
                "generation-step timing requires the manager-owned persistent V-packed sidecar"
            )
        persistent_v_pack = persistent_v_packed is not None
        generation_latent = (
            torch.randn(
                batch * q_len,
                kv_lora_rank + qk_rope_head_dim,
                dtype=torch.bfloat16,
                device=q_nope.device,
            )
            * 0.25
        ).clamp_(-1.0, 1.0)
        generation_rotary_cos_sin = torch.zeros(
            max(seq_lens) + q_len,
            qk_rope_head_dim,
            2,
            dtype=torch.float32,
            device=q_nope.device,
        )
        generation_rotary_cos_sin[..., 0] = 1.0
    launcher_module = None
    original_launcher = None
    launcher_wrapper = None
    core_launcher = None
    original_compile_fused = None
    launcher_tensors = None
    launcher_tensor_ptrs = None
    launcher_kwargs = None
    retained_state = None
    capture_stream = None
    cuda_graph = None
    graph_event_ms = None
    graph_wall_ms = None
    cupti_cuda_events = []
    cupti_fused_launches_per_iter = 1
    cupti_fused_slot_stats = []
    try:
        if report_correction:
            _, exact_probs, _ = _fp4_mla_attention_decode_reference(
                metadata,
                q_nope,
                q_pe,
                sm_scale=0.1,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
            )
            correction_stats = _fp4_mla_lazy_rebase_stats(exact_probs)
            print(
                "correction "
                f"total_rows={correction_stats.total_rows} "
                f"row_rebase_rows={correction_stats.rebase_rows} "
                f"row_rebase_total={correction_stats.rebase_total} "
                f"row_rebase_max={correction_stats.rebase_max_per_row} "
                f"max_anchor_delta_log2="
                f"{correction_stats.max_anchor_delta_log2:.6f} "
                f"threshold_log2={correction_stats.threshold_log2:.6f}",
                flush=True,
            )
        if backend == "cutedsl":
            cutedsl = fp4_mla_backend._fp4_mla_cutedsl_kernel_module()

            launcher_module = cutedsl
            original_launcher = cutedsl.run_trtllm_fp4_mla_decode_page_native_from_raw

            def record_launcher(*args: object, **kwargs: object) -> object:
                nonlocal launcher_kwargs, launcher_tensors, launcher_tensor_ptrs
                current_tensors = tuple(args)
                current_ptrs = _launcher_tensor_pointers(current_tensors)
                if launcher_tensor_ptrs is None:
                    launcher_tensors = current_tensors
                    launcher_tensor_ptrs = current_ptrs
                    launcher_kwargs = dict(kwargs)
                elif current_ptrs != launcher_tensor_ptrs:
                    raise RuntimeError(
                        "CuTeDSL launcher tensor pointers changed within one benchmark run"
                    )
                return original_launcher(*args, **kwargs)

            launcher_wrapper = record_launcher
    except (ImportError, TypeError, ValueError, RuntimeError):
        kv_cache_manager.shutdown()
        raise

    try:
        if launcher_module is not None and launcher_wrapper is not None:
            launcher_module.run_trtllm_fp4_mla_decode_page_native_from_raw = launcher_wrapper
        output = torch.full_like(q_nope, float("nan"))
        q = torch.cat((q_nope, q_pe), dim=-1).contiguous()
        baseline_q_fp4, baseline_q_sf = torch.ops.trtllm.fp4_quantize_with_residual(
            q.view(-1, q.shape[-1]),
            metadata.fp4_mla_state.q_global_scale,
            FP4_MLA_Q_RESIDUAL_DIM,
            is_act=True,
        )
        baseline_q_sf = baseline_q_sf.view(torch.float8_e4m3fn).reshape(-1)

        def decode() -> None:
            q_fp4 = getattr(metadata.fp4_mla_state, "prequantized_q", None)
            q_sf = getattr(metadata.fp4_mla_state, "prequantized_q_sf", None)
            q_batch_capacity = getattr(metadata.fp4_mla_state, "q_batch_capacity", None)
            if q_fp4 is None:
                q_fp4 = baseline_q_fp4
                q_sf = baseline_q_sf
                q_batch_capacity = q.shape[0]
            run_fp4_mla_attention_decode(
                metadata,
                layer_idx=0,
                local_layer=0,
                q=q,
                output=output,
                sm_scale=0.1,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                prequantized_q=q_fp4,
                prequantized_q_sf=q_sf,
                q_batch_capacity=q_batch_capacity,
            )

        run = decode
        if generation_step:
            assert generation_latent is not None
            assert generation_rotary_cos_sin is not None

            def generation_run() -> None:
                scatter_fp4_mla_kv_cache(
                    metadata,
                    generation_latent,
                    layer_idx=0,
                    token_offset=0,
                    phase="generation",
                    local_layer=0,
                    v_head_dim=kv_lora_rank,
                    rotary_cos_sin=generation_rotary_cos_sin,
                    q_pe=q[..., kv_lora_rank:],
                    q_rope_out=q[..., kv_lora_rank:],
                    q_quant_input=q,
                )
                decode()

            run = generation_run

        run()
        torch.cuda.synchronize()
        if launcher_module is not None and original_launcher is not None:
            core_launcher = original_launcher
            launcher_module.run_trtllm_fp4_mla_decode_page_native_from_raw = original_launcher
            original_launcher = None
        if not bool(torch.isfinite(output).all().item()):
            raise RuntimeError("FP4 MLA benchmark output contains non-finite values.")
        if use_cuda_graph:
            if timing_scope == "fused":
                if core_launcher is None or launcher_tensors is None or launcher_kwargs is None:
                    raise RuntimeError("core-only CUDA Graph capture missed the CuTeDSL launcher")

                def eager_run() -> None:
                    core_launcher(*launcher_tensors, **launcher_kwargs)

            else:
                eager_run = run
            cuda_graph, capture_stream = _capture_cuda_graph(eager_run, output.device)

            def replay_graph() -> None:
                cuda_graph.replay()

            run = replay_graph
            output.fill_(float("nan"))
            run()
            torch.cuda.synchronize(output.device)
            if not bool(torch.isfinite(output).all().item()):
                raise RuntimeError(
                    "FP4 MLA CUDA Graph replay did not overwrite the output sentinel."
                )
        if use_cuda_graph:
            graph_event_ms, graph_wall_ms = _bench_cuda_graph(
                run,
                output.device,
                warmup=warmup,
                iters=iters,
            )
            avg_ms = graph_event_ms if graph_timing == "event" else graph_wall_ms
        elif timing_scope == "fused":
            timer = _FusedLaunchTimer(
                output.device,
                iters,
                queue_delay_cycles=fused_queue_delay_cycles,
            )
            original_compile_fused = launcher_module._compile_fused
            compile_cache_size = len(launcher_module._FUSED_COMPILE_CACHE)

            def compile_timed_fused(*args: object, **kwargs: object) -> Callable[..., object]:
                return timer.wrap(original_compile_fused(*args, **kwargs))

            launcher_module._compile_fused = compile_timed_fused
            for _ in range(warmup):
                run()
            torch.cuda.synchronize(output.device)
            if len(launcher_module._FUSED_COMPILE_CACHE) != compile_cache_size:
                raise RuntimeError("CuTeDSL fused compile cache changed during warmup")
            timer.enabled = True
            try:
                for _ in range(iters):
                    run()
            finally:
                timer.enabled = False
            torch.cuda.synchronize(output.device)
            if len(launcher_module._FUSED_COMPILE_CACHE) != compile_cache_size:
                raise RuntimeError("CuTeDSL fused compile cache changed during timing")
            avg_ms = timer.mean_ms()
        elif timing_scope == "cupti":
            compile_cache_size = len(launcher_module._FUSED_COMPILE_CACHE)
            for _ in range(warmup):
                run()
            torch.cuda.synchronize(output.device)
            from torch.profiler import ProfilerActivity, profile

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                acc_events=True,
            ) as profiler:
                for _ in range(iters):
                    run()
                torch.cuda.synchronize(output.device)
            if len(launcher_module._FUSED_COMPILE_CACHE) != compile_cache_size:
                raise RuntimeError("CuTeDSL fused compile cache changed during CUPTI timing")
            cupti_cuda_events = [
                event
                for event in profiler.events()
                if event.device_type == torch.autograd.DeviceType.CUDA
            ]
            target_prefix = "kernel_cutlass_kernel_"
            fused_events = [
                event for event in cupti_cuda_events if event.name.startswith(target_prefix)
            ]
            fused_events.sort(key=lambda event: event.time_range.start)
            if not fused_events or len(fused_events) % iters != 0:
                raise RuntimeError(
                    "expected a fixed positive number of CUPTI fused events "
                    f"per iteration, got {len(fused_events)} events for {iters} iterations"
                )
            cupti_fused_launches_per_iter = len(fused_events) // iters
            for slot in range(cupti_fused_launches_per_iter):
                slot_events = fused_events[slot::cupti_fused_launches_per_iter]
                if len(slot_events) != iters:
                    raise RuntimeError(
                        f"expected {iters} CUPTI fused events for slot {slot}, "
                        f"got {len(slot_events)}"
                    )
                cupti_fused_slot_stats.append(
                    (
                        slot,
                        len(slot_events),
                        sum(event.self_device_time_total for event in slot_events),
                    )
                )
            # PyTorch profiler device times are reported in microseconds.
            avg_ms = sum(event.self_device_time_total for event in fused_events) / iters / 1000.0
            if avg_ms <= 0.0:
                raise RuntimeError(f"CUPTI reported non-positive fused time {avg_ms}")
        else:
            avg_ms = _bench(run, warmup=warmup, iters=iters)
        if not bool(torch.isfinite(output).all().item()):
            raise RuntimeError("FP4 MLA timed output contains non-finite values.")
        k_residual_dim = FP4_MLA_K_RESIDUAL_DIM if backend in ("triton", "cutedsl") else 0
        qk_dim = kv_lora_rank + qk_rope_head_dim + FP4_MLA_Q_RESIDUAL_DIM + k_residual_dim
        pv_dim = kv_lora_rank
        flops = 2 * heads * _causal_token_pairs(seq_lens, q_len) * (qk_dim + pv_dim)
        tflops = flops / avg_ms / 1e9
        min_tflops = float(os.environ.get("FP4_MLA_MIN_TFLOPS", "0"))
        if tflops < min_tflops:
            raise RuntimeError(f"FP4 MLA throughput {tflops:.2f} TFLOP/s is below {min_tflops:.2f}")
        # None of these estimates is a hardware counter or measured HBM bandwidth.
        logical_bytes = _kernel_io_bytes(
            seq_lens,
            heads,
            kv_lora_rank,
            qk_rope_head_dim,
            q_len,
            k_residual_dim,
        )
        fixed_tile_bytes = _fixed_tile_request_bytes(
            seq_lens,
            heads,
            kv_lora_rank,
            qk_rope_head_dim,
            q_len,
            k_residual_dim,
        )
        tma_smem_bytes = _tma_smem_completion_bytes(
            seq_lens,
            heads,
            kv_lora_rank,
            qk_rope_head_dim,
            q_len,
            k_residual_dim,
        )
        logical_gb_s = logical_bytes / (avg_ms * 1e-3) / 1e9
        fixed_tile_gb_s = fixed_tile_bytes / (avg_ms * 1e-3) / 1e9
        tma_smem_gb_s = tma_smem_bytes / (avg_ms * 1e-3) / 1e9
        graph_timing_label = f" graph_timing={graph_timing}" if use_cuda_graph else ""
        generation_label = (
            f" generation_step=True persistent_v_pack={persistent_v_pack}"
            if generation_step
            else ""
        )
        print(
            f"backend={backend:>12s} bs={batch:>3d} seq={_seq_label(seq_lens):>11s} "
            f"heads={heads:>3d} qlen={q_len:>2d}{graph_timing_label}{generation_label} "
            f"scope={timing_scope}: "
            f"{avg_ms:>10.6f} ms  {tflops:>6.2f} TFLOP/s  "
            f"Est.Logical {logical_gb_s:>6.1f} GB/s  "
            f"Est.GlobalReq {fixed_tile_gb_s:>6.1f} GB/s  "
            f"Est.TMA-SMEM {tma_smem_gb_s:>6.1f} GB/s",
            flush=True,
        )
        if graph_event_ms is not None and graph_wall_ms is not None:
            print(
                f"cuda_graph event_us={graph_event_ms * 1000.0:.6f} "
                f"host_wall_us={graph_wall_ms * 1000.0:.6f}",
                flush=True,
            )
        elif fused_queue_delay_cycles:
            print(
                f"fused queue delay={fused_queue_delay_cycles} cycles "
                "(excluded from the event interval)",
                flush=True,
            )
        if cupti_cuda_events:
            if cupti_fused_launches_per_iter > 1:
                print(
                    f"cupti fused launches/iteration={cupti_fused_launches_per_iter}",
                    flush=True,
                )
            for slot, count, total_us in cupti_fused_slot_stats:
                role = (
                    ("fast_partition", "slow_partition")[slot]
                    if cupti_fused_launches_per_iter == 2
                    else "single"
                )
                print(
                    f"cupti fused slot={slot} role={role} count={count} "
                    f"total_us={total_us:.3f} avg_us={total_us / count:.3f}",
                    flush=True,
                )
            event_totals = {}
            ignored_events = {"Activity Buffer Request", "Lazy Function Loading"}
            for event in cupti_cuda_events:
                if event.name in ignored_events:
                    continue
                count, total_us = event_totals.get(event.name, (0, 0.0))
                event_totals[event.name] = (
                    count + 1,
                    total_us + event.self_device_time_total,
                )
            for name, (count, total_us) in sorted(
                event_totals.items(),
                key=lambda item: item[1][1],
                reverse=True,
            )[:8]:
                print(
                    f"cupti count={count} total_us={total_us:.3f} "
                    f"avg_us={total_us / count:.3f} kernel={name[:120]}",
                    flush=True,
                )
        if retain_state:
            if launcher_tensors is None or launcher_tensor_ptrs is None:
                raise RuntimeError("retained state did not observe CuTeDSL launcher tensors")
            retained_state = RetainedBenchmarkState(
                kv_cache_manager,
                launcher_tensors,
                {
                    "metadata": metadata,
                    "q": q,
                    "q_nope": q_nope,
                    "q_pe": q_pe,
                    "output": output,
                    "generation_latent": generation_latent,
                    "cuda_graph": cuda_graph,
                    "capture_stream": capture_stream,
                },
            )
            return avg_ms, retained_state
        return avg_ms
    finally:
        if launcher_module is not None and original_launcher is not None:
            launcher_module.run_trtllm_fp4_mla_decode_page_native_from_raw = original_launcher
        if launcher_module is not None and original_compile_fused is not None:
            launcher_module._compile_fused = original_compile_fused
        if retained_state is None:
            kv_cache_manager.shutdown()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, nargs="+", default=None)
    p.add_argument("--seq", type=int, default=30080)
    p.add_argument("--heads", type=int, default=128)
    p.add_argument(
        "--q-len",
        "--mtp-len",
        dest="q_len",
        type=int,
        default=1,
        help="Query tokens per sequence (>1 for MTP / speculative decoding).",
    )
    p.add_argument(
        "--backend",
        default=None,
        choices=BACKEND_CHOICES,
        help="Backend to benchmark; default runs all backends supported on the active GPU.",
    )
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument(
        "--timing-scope",
        choices=("full", "fused", "cupti"),
        default="full",
        help=(
            "full times the complete integrated decode call; fused isolates the "
            "CuTeDSL or Rubin FP8 attention core; cupti uses exact CUDA activity "
            "timestamps for the fused kernel (CuTeDSL backend only)."
        ),
    )
    p.add_argument(
        "--fused-queue-delay-cycles",
        "--queue-delay-cycles",
        type=int,
        default=0,
        help=(
            "enqueue a same-stream GPU delay before each timed start event to "
            "hide host launch submission; the delay is excluded from timing"
        ),
    )
    p.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Capture the selected timed call once and benchmark Graph replay.",
    )
    p.add_argument(
        "--graph-timing",
        choices=("event", "wall"),
        default="event",
        help=(
            "select the primary CUDA Graph metric; event is GPU elapsed time, "
            "while wall includes host Graph launch overhead"
        ),
    )
    p.add_argument(
        "--generation-step",
        action="store_true",
        help=(
            "Capture generation KV scatter, high-precision KV update, persistent "
            "touched-page V repack, and decode together (requires CuTeDSL CUDA "
            "Graph full timing)."
        ),
    )
    p.add_argument(
        "--report-correction",
        action="store_true",
        help=(
            "Classify lazy resident-O correction activity from the exact "
            "quantized-QK reference before timing."
        ),
    )
    args = p.parse_args()

    if args.batch is not None and any(batch <= 0 for batch in args.batch):
        p.error("--batch values must be positive")
    if args.seq <= 0 or args.heads <= 0:
        p.error("--seq and --heads must be positive")
    if not 0 < args.q_len <= args.seq:
        p.error("--q-len must be positive and must not exceed --seq")
    if args.warmup < 0:
        p.error("--warmup must be non-negative")
    if args.iters <= 0:
        p.error("--iters must be positive")
    if args.fused_queue_delay_cycles < 0:
        p.error("--fused-queue-delay-cycles must be non-negative")
    if args.fused_queue_delay_cycles and args.timing_scope != "fused":
        p.error("--fused-queue-delay-cycles requires --timing-scope fused")
    if args.fused_queue_delay_cycles and args.cuda_graph:
        p.error("--fused-queue-delay-cycles is not used with --cuda-graph")
    graph_backends = {"cutedsl", "trtllm_fp8_rubin"}
    if args.cuda_graph and args.backend not in graph_backends:
        p.error("--cuda-graph requires --backend cutedsl or trtllm_fp8_rubin")
    if args.cuda_graph and args.timing_scope == "cupti":
        p.error("--cuda-graph does not support --timing-scope cupti")
    if args.cuda_graph and args.backend == "trtllm_fp8_rubin" and args.timing_scope != "fused":
        p.error("FP8 core-only --cuda-graph requires --timing-scope fused")
    if args.generation_step and not (
        args.backend == "cutedsl" and args.cuda_graph and args.timing_scope == "full"
    ):
        p.error("--generation-step requires --backend cutedsl --cuda-graph --timing-scope full")
    if args.report_correction and args.backend != "cutedsl":
        p.error("--report-correction requires --backend cutedsl")
    if args.timing_scope == "fused" and args.backend not in ("cutedsl", "trtllm_fp8_rubin"):
        p.error("--timing-scope fused requires --backend cutedsl or trtllm_fp8_rubin")
    if args.timing_scope == "cupti" and args.backend != "cutedsl":
        p.error("--timing-scope cupti requires --backend cutedsl")

    batches = args.batch if args.batch else [16, 30, 60, 120, 200, 300]
    if not torch.cuda.is_available():
        p.error("This benchmark requires a CUDA GPU.")
    is_rubin = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
    if args.backend:
        if args.backend == "cutedsl":
            if not is_rubin:
                p.error("--backend cutedsl requires a Rubin SM107 GPU.")
            if not _cutedsl_backend_available():
                p.error("--backend cutedsl requires the CTM and CuTeDSL runtime packages.")
        if args.backend == "trtllm_fp8_rubin" and not is_rubin:
            p.error("--backend trtllm_fp8_rubin requires a Rubin SM107 GPU.")
        backends = [args.backend]
    else:
        excluded_backends = {"cutedsl"}
        excluded_backends.add("trtllm_fp8" if is_rubin else "trtllm_fp8_rubin")
        backends = [backend for backend in BACKEND_CHOICES if backend not in excluded_backends]
        if is_rubin and _cutedsl_backend_available():
            backends.append("cutedsl")

    for b in batches:
        for be in backends:
            if be == "trtllm_fp8":
                run_one_trtllm(b, args.seq, args.heads, args.q_len, args.warmup, args.iters)
            elif be == "trtllm_fp8_rubin":
                run_one_trtllm_rubin(
                    b,
                    args.seq,
                    args.heads,
                    args.q_len,
                    args.warmup,
                    args.iters,
                    queue_delay_cycles=args.fused_queue_delay_cycles,
                    use_cuda_graph=args.cuda_graph,
                    graph_timing=args.graph_timing,
                )
            else:
                run_one(
                    b,
                    args.seq,
                    args.heads,
                    be,
                    args.q_len,
                    args.warmup,
                    args.iters,
                    args.timing_scope,
                    args.fused_queue_delay_cycles,
                    args.report_correction,
                    args.cuda_graph,
                    args.generation_step,
                    graph_timing=args.graph_timing,
                )


if __name__ == "__main__":
    main()
