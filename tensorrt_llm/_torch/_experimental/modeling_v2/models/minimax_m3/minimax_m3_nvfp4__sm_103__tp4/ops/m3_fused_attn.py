# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 fused short-sequence decode attention: ``m3_fused_decode_attn``.

The kernel is ``torch.ops.trtllm.minimax_m3_fused_decode_attn`` (``cpp/tensorrt_llm/kernels/minimaxM3/
fusedDecodeAttn.cu``); it is compiled for the geometries in ``GEOMETRIES``, each with the MXFP8 hand-off.

One kernel replaces the per-layer chain that follows the packed QKV GEMM in short-sequence decode mode (every
request attends at most 16 KV blocks of 128 tokens, so block selection is the identity):

* q/k/index-k RMSNorm + NeoX partial RoPE, NVFP4 K/V cache insert and FP8 index-K insert
  (``minimaxM3Nvfp4QKVIndexerNormRopeKVInsertKernel``),
* block-sparse GQA decode over the 16 blocks (``_gqa_sparse_decode_kernel``),
* the split-K LSE merge, optionally with the MXFP8 copy of the output (``_merge_topk_attn_out_kernel``).

Geometry: 16 query heads, one KV head, one index head, head_dim 128, rotary_dim 64, page 128.

Launch: clusters of ``16 / BPC`` CTAs; CTA ``r`` owns KV blocks ``[r * BPC, r * BPC + BPC)`` for ``QG`` query
tokens of one request (``decode_query_len / QG`` clusters per request). Each block keeps its own softmax partial,
exactly the Triton producer's chunk, so the merge sees the same 16 partials.

* Before ``griddepcontrol.wait``: request metadata, RoPE rows, norm weights, the packed K chunks (registers) and the
  packed V tile (``cp.async``). These are staged before the CUDA graph or written by earlier steps; nothing the QKV
  GEMM writes is read, and nothing is written, before the wait.
* After the wait: K is dequantized to fp16 (exact) while the packed-QKV loads are in flight; every CTA
  normalizes/ropes the query rows it needs (recomputed rather than exchanged); the CTA whose block holds a new
  token's position also normalizes, ropes and quantizes that token's K, V and index-K exactly as the C++ producer
  does, and writes the dequantized row into its fp16 tiles. The PDL dependents are released once every K/V byte
  has been read: an earlier release lets their weight streams delay the KV reads.
* S^T = K Q^T and O^T = V^T P^T run on ``tcgen05.mma`` (FP16 operands from shared memory, FP32 accumulators in
  TMEM). The softmax runs in the TMEM-reading warps with the producer's per-block math (block max, exp2, fp16 P,
  log-domain LSE), so each partial equals the Triton producer's chunk partial up to accumulation order. The
  producer warps dequantize V meanwhile.
* Partials go from registers to the merging CTA with ``st.async`` + mbarrier transaction counts (DSMEM); CTA ``r``
  merges its heads with the merge kernel's LSE weighting and writes the BF16 output (and the MXFP8 hand-off).

CUDA C++ rather than Triton or CuTe DSL: the design needs cluster DSMEM pushes, loads issued before the PDL wait,
tcgen05/TMEM, and bit-exact reuse of the C++ producer's norm/RoPE/NVFP4 arithmetic.
"""

from typing import Optional

import torch

from ..ops.attention.m3_fused_decode_attn import m3_fused_decode_attn as _m3_fused_decode_attn_op

NUM_HEADS = 16  # capacity: fixed: the tp4 query heads
HEAD_DIM = 128  # capacity: fixed: the checkpoint's head dim
PAGE_SIZE = 128
MAX_BLOCKS = 16  # capacity: fixed: the MSA top-16 blocks
PACKED_WIDTH = (NUM_HEADS + 4) * HEAD_DIM  # capacity: fixed: the tp4 packed QKV width

# (decode_query_len, tokens per cluster, KV blocks per CTA) geometries compiled into the op: what
# default_geometry() returns for every step shape the target routes here (at most 16 tokens).
GEOMETRIES = frozenset(
    [
        (1, 1, 1),
        (1, 1, 2),
        (2, 1, 1),
        (2, 1, 2),
        (3, 1, 1),
        (4, 1, 1),
        (5, 1, 1),
        (6, 1, 1),
        (6, 1, 2),
        (7, 1, 1),
        (8, 1, 1),
    ]
)

_module = None


def _load_module():
    """The compiled ops' namespace (``torch.ops.trtllm``), resolved once; ``_module`` stays None until then."""
    global _module
    if _module is None:
        try:
            torch.ops.trtllm.minimax_m3_fused_decode_attn
        except AttributeError as exc:
            raise RuntimeError(
                "torch.ops.trtllm.minimax_m3_fused_decode_attn is not registered: this TensorRT-LLM "
                "build lacks cpp/tensorrt_llm/kernels/minimaxM3/fusedDecodeAttn.cu"
            ) from exc
        _module = torch.ops.trtllm
    return _module


def default_geometry(decode_query_len: int, num_requests: int) -> tuple:
    """(tokens per cluster, KV blocks per CTA) for a step shape: one token per 16-CTA cluster up to 6 clusters,
    then KV blocks paired into 8-CTA clusters up to 16 clusters where that geometry is compiled."""
    clusters = decode_query_len * num_requests
    if clusters <= 6:
        return (1, 1)
    if (decode_query_len, 1, 2) in GEOMETRIES and clusters <= 16:
        return (1, 2)
    return (1, 1)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"m3_fused_decode_attn: {msg}")


def m3_fused_decode_attn_checks(
    num_tokens,
    num_reqs,
    decode_query_len,
    kv_data,
    kv_scale,
    index_k_cache,
    out_cache_loc,
    position_ids,
    block_table,
    seq_lens,
    kv_quant_scale,
    k_global_scale,
    v_global_scale,
    q_norm_weight,
    k_norm_weight,
    index_k_norm_weight,
    rotary_cos_sin,
    output,
) -> None:
    """Argument checks shared by the attention entry points (the packed or fused-projection QKV source)."""
    _check(
        1 <= decode_query_len <= 8, f"decode_query_len must be in [1, 8], got {decode_query_len}"
    )
    _check(
        num_tokens == num_reqs * decode_query_len,
        "token rows must equal num_requests * decode_query_len",
    )
    _check(
        kv_data.dtype == torch.uint8
        and kv_data.dim() == 5
        and tuple(kv_data.shape[1:]) == (2, 1, 128, 64)
        and kv_data.stride(4) == 1
        and kv_data.stride(3) == 64
        and kv_data.stride(0) % 16 == 0
        and kv_data.stride(1) % 16 == 0
        and kv_data.data_ptr() % 16 == 0,
        "kv_data must be uint8 [pages, 2, 1, 128, 64] with contiguous 16-byte aligned head blocks",
    )
    _check(
        kv_scale.dtype == torch.uint8
        and kv_scale.dim() == 5
        and tuple(kv_scale.shape[1:]) == (2, 1, 128, 8)
        and kv_scale.stride(4) == 1
        and kv_scale.stride(3) == 8
        and kv_scale.stride(0) % 16 == 0
        and kv_scale.stride(1) % 16 == 0
        and kv_scale.data_ptr() % 16 == 0
        and kv_scale.shape[0] == kv_data.shape[0],
        "kv_scale must be uint8 [pages, 2, 1, 128, 8] with contiguous 16-byte aligned head blocks",
    )
    _check(
        index_k_cache.dtype == torch.float8_e4m3fn
        and index_k_cache.dim() == 4
        and tuple(index_k_cache.shape[1:]) == (1, 128, 128)
        and index_k_cache.stride(3) == 1
        and index_k_cache.stride(2) == 128
        and index_k_cache.stride(0) % 4 == 0
        and index_k_cache.shape[0] == kv_data.shape[0],
        "index_k_cache must be fp8 [pages, 1, 128, 128]",
    )
    for name, t in (
        ("out_cache_loc", out_cache_loc),
        ("position_ids", position_ids),
        ("block_table", block_table),
        ("seq_lens", seq_lens),
    ):
        _check(t.dtype == torch.int32 and t.is_cuda, f"{name} must be CUDA int32")
    _check(
        out_cache_loc.is_contiguous() and out_cache_loc.numel() >= num_tokens,
        "out_cache_loc too short",
    )
    _check(
        position_ids.is_contiguous() and position_ids.numel() == num_tokens,
        "position_ids must have T entries",
    )
    _check(
        seq_lens.is_contiguous()
        and block_table.dim() == 2
        and block_table.stride(1) == 1
        and block_table.shape[0] >= num_reqs,
        "block_table must be [num_requests, max_blocks] row-major",
    )
    for name, t in (
        ("kv_quant_scale", kv_quant_scale),
        ("k_global_scale", k_global_scale),
        ("v_global_scale", v_global_scale),
    ):
        _check(
            t.dtype == torch.float32 and t.is_cuda and t.is_contiguous(),
            f"{name} must be CUDA fp32",
        )
    _check(kv_quant_scale.numel() >= 3, "kv_quant_scale must hold [Q, K, V]")
    for name, t in (
        ("q_norm_weight", q_norm_weight),
        ("k_norm_weight", k_norm_weight),
        ("index_k_norm_weight", index_k_norm_weight),
    ):
        _check(
            t.dtype == torch.bfloat16
            and t.numel() == HEAD_DIM
            and t.is_contiguous()
            and t.data_ptr() % 8 == 0,
            f"{name} must be contiguous bf16 [128]",
        )
    _check(
        rotary_cos_sin.dtype == torch.float32
        and rotary_cos_sin.is_contiguous()
        and rotary_cos_sin.dim() == 3
        and tuple(rotary_cos_sin.shape[1:]) == (2, 32),
        "rotary_cos_sin must be fp32 [max_pos, 2, 32]",
    )
    out2d = output.view(num_tokens, NUM_HEADS * HEAD_DIM)
    _check(
        output.dtype == torch.bfloat16 and out2d.stride(1) == 1 and out2d.data_ptr() % 8 == 0,
        "output must be bf16 [T, 2048] with contiguous rows",
    )


def m3_fused_decode_attn(
    packed: torch.Tensor,
    kv_data: torch.Tensor,
    kv_scale: torch.Tensor,
    index_k_cache: torch.Tensor,
    out_cache_loc: torch.Tensor,
    position_ids: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_quant_scale: torch.Tensor,
    k_global_scale: torch.Tensor,
    v_global_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    rotary_cos_sin: torch.Tensor,
    *,
    eps: float,
    sm_scale: float,
    decode_query_len: int,
    output: torch.Tensor,
    mxfp8_out: Optional[tuple] = None,
    use_pdl: bool = True,
    geometry: Optional[tuple] = None,
) -> None:
    """Fused norm/RoPE/KV-insert + short-sequence sparse decode + split-K merge, written into ``output``.

    packed:        [T, 2560] bf16 QKV+index GEMM output, T = num_requests * decode_query_len.
    kv_data:       uint8 [num_pages, 2, 1, 128, 64] HND NVFP4 pool (K plane 0, V plane 1); written.
    kv_scale:      uint8 [num_pages, 2, 1, 128, 8] E4M3 block scales (K linear, V 4x4 token quads); written.
    index_k_cache: float8_e4m3fn [num_pages, 1, 128, 128]; written.
    out_cache_loc, position_ids: int32 [>= T] / [T]; block_table int32 [num_requests, >= 1]; seq_lens int32
    [num_requests] (attended length including this step's tokens, at most 16 * 128).
    kv_quant_scale: fp32 [3] (Q, K, V) quantization scales; k/v_global_scale: fp32 [1] dequantization scales.
    output:        bf16 [T, 2048] (or [T, 16, 128], contiguous rows).
    mxfp8_out:     (e4m3 [>= T, 2048], uint8 swizzled UE8M0 scales) hand-off for the o_proj GEMM (required: only
                   the MXFP8 variants are compiled),
                   same layout as the merge kernel's MERGE_MXFP8 output.
    geometry:      (query tokens per cluster, KV blocks per CTA); default_geometry() when None.
    """
    num_tokens = int(packed.shape[0])
    num_reqs = int(seq_lens.shape[0])
    _check(
        1 <= decode_query_len <= 8, f"decode_query_len must be in [1, 8], got {decode_query_len}"
    )
    _check(
        num_tokens == num_reqs * decode_query_len,
        "packed rows must equal num_requests * decode_query_len",
    )
    _check(
        packed.dtype == torch.bfloat16
        and packed.dim() == 2
        and packed.shape[1] == PACKED_WIDTH
        and packed.stride(1) == 1
        and packed.stride(0) % 4 == 0
        and packed.data_ptr() % 8 == 0,
        "packed must be bf16 [T, 2560] with 8-byte aligned contiguous rows",
    )
    m3_fused_decode_attn_checks(
        num_tokens,
        num_reqs,
        decode_query_len,
        kv_data,
        kv_scale,
        index_k_cache,
        out_cache_loc,
        position_ids,
        block_table,
        seq_lens,
        kv_quant_scale,
        k_global_scale,
        v_global_scale,
        q_norm_weight,
        k_norm_weight,
        index_k_norm_weight,
        rotary_cos_sin,
        output,
    )
    out2d = output.view(num_tokens, NUM_HEADS * HEAD_DIM)
    _check(
        mxfp8_out is not None,
        "only the variants with the MXFP8 hand-off are compiled; pass mxfp8_out",
    )
    q8, sf = mxfp8_out
    _check(
        q8.dtype == torch.float8_e4m3fn
        and q8.dim() == 2
        and q8.shape[1] == NUM_HEADS * HEAD_DIM
        and q8.shape[0] >= num_tokens
        and q8.is_contiguous(),
        "mxfp8 data must be e4m3 [>= T, 2048]",
    )
    _check(sf.dtype == torch.uint8 and sf.is_contiguous(), "mxfp8 scales must be contiguous uint8")
    q8 = q8.view(torch.uint8)
    if num_tokens == 0:
        return
    qg, bpc = geometry if geometry is not None else default_geometry(decode_query_len, num_reqs)
    _check(
        (decode_query_len, qg, bpc) in GEOMETRIES,
        f"geometry {(decode_query_len, qg, bpc)} is not compiled",
    )
    _load_module()
    _m3_fused_decode_attn_op(
        packed,
        kv_data,
        kv_scale,
        index_k_cache.view(torch.uint8),
        out_cache_loc,
        position_ids,
        block_table,
        seq_lens,
        kv_quant_scale,
        k_global_scale,
        v_global_scale,
        q_norm_weight,
        k_norm_weight,
        index_k_norm_weight,
        rotary_cos_sin,
        float(eps),
        float(sm_scale),
        int(decode_query_len),
        qg,
        bpc,
        out2d,
        q8,
        sf,
        bool(use_pdl),
    )


__all__ = ["GEOMETRIES", "default_geometry", "m3_fused_decode_attn"]
