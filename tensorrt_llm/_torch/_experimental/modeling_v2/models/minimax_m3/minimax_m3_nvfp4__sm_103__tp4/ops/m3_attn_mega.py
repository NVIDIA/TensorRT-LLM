# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 attention megakernel: ``m3_attn_mega``, one launch for a one-request, one-token decode layer.

The QKV GEMV (bit-identical to ``m3_mx_gemv``: split 4 with contiguous k-tiles, one accumulator, the MXFP8
activation as produced by the all-reduce, the split-K partials reduced in the owner CTA over DSMEM in rank order
and rounded to bf16), the fused decode attention and the o_proj GEMV run in one persistent grid, joined by per-tile
ready counters instead of a PDL boundary.

The kernel is ``torch.ops.trtllm.minimax_m3_attn_mega`` (``cpp/tensorrt_llm/kernels/minimaxM3/attnMega.cu``),
compiled for geometry (1, 1, 1) with the MXFP8 hand-off only.

Grid: 7 clusters of 16 CTAs. The first attention cluster(s) run the attention; the last 80 CTAs are the QKV units
(row tile u / 4, split u % 4; the four splits of a tile are consecutive CTAs of one cluster). Each unit's owner CTA
of a 32-row slice stores it to ``packed_out`` and release-increments the tile's counter; every attention CTA waits
for 4 more increments on all 20 tiles than at its last call (a per-CTA target it alone keeps), then reads the rows.

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

from ..ops.attention.m3_attn_mega import m3_attn_mega as _m3_attn_mega_op

NUM_HEADS = 16
HEAD_DIM = 128
PAGE_SIZE = 128
MAX_BLOCKS = 16  # capacity: fixed: the MSA top-16 blocks
PACKED_WIDTH = (NUM_HEADS + 4) * HEAD_DIM  # capacity: fixed: the tp4 packed QKV width

_module = None


def _load_module():
    """The compiled ops' namespace (``torch.ops.trtllm``), resolved once; ``_module`` stays None until then."""
    global _module
    if _module is None:
        try:
            torch.ops.trtllm.minimax_m3_attn_mega
        except AttributeError as exc:
            raise RuntimeError(
                "torch.ops.trtllm.minimax_m3_attn_mega is not registered: this TensorRT-LLM build lacks "
                "cpp/tensorrt_llm/kernels/minimaxM3/attnMega.cu"
            ) from exc
        _module = torch.ops.trtllm
    return _module


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"m3_fused_decode_attn: {msg}")


class MegaWorkspace:
    """Per-layer persistent state of the megakernel (allocate before CUDA-graph capture; nothing is ever reset): the
    two call-parity buffers of the packed QKV rows and of the attention's MXFP8 hand-off (marker-filled), and
    ``flags[32 + CTA]``, each CTA's call counter (its low bit selects the parity)."""

    def __init__(self, device, max_tokens: int = 8):
        # Two call-parity buffers of the packed QKV rows, all set to the 0xFFFF "not yet" marker.
        self.qkv_out = torch.full(
            (2, max_tokens, PACKED_WIDTH), -1, dtype=torch.int16, device=device
        ).view(torch.bfloat16)
        self.flags = torch.zeros(256, dtype=torch.int32, device=device)
        # Two call-parity buffers of the attention's MXFP8 hand-off to the o_proj units, all 0xFF ("not yet").
        self.q8 = torch.full(
            (2 * max_tokens * 2048,), 255, dtype=torch.uint8, device=device
        )  # capacity: fixed: a workspace of the kernel's fixed geometry
        self.sf = torch.full(
            (2 * 16 * 512,), 255, dtype=torch.uint8, device=device
        )  # capacity: fixed: a workspace of the kernel's fixed geometry
        self.calls = 0  # host mirror of the call count (eager use only; graph replays advance the device copy)


def m3_attn_mega(
    xq: torch.Tensor,
    xsf: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_weight_scale: torch.Tensor,
    ws: MegaWorkspace,
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
    o_weight: torch.Tensor,
    o_weight_scale: torch.Tensor,
    *,
    eps: float,
    sm_scale: float,
    decode_query_len: int,
    output: torch.Tensor,
    y: torch.Tensor,
    use_pdl: bool = True,
    ar: Optional[tuple] = None,
) -> torch.Tensor:
    """QKV projection (bit-identical to m3_mx_gemv) of the MXFP8 layer input ``xq`` / ``xsf``, the fused decode
    attention (bit-identical to m3_fused_decode_attn) and the o_proj projection (bit-identical to m3_mx_gemv) into
    ``y`` [T, 6144], in one launch.
    ``output`` receives the attention rows (bf16). ``ar`` = m3_mx_gemv's (multicast workspace address, flags address,
    rank, world, bucket tokens, mode 2): also push y into the next m3_mc_allreduce's exchange. Returns the packed QKV
    rows of this call."""
    num_tokens = int(xq.shape[0])
    num_reqs = int(seq_lens.shape[0])
    _check(
        num_tokens == num_reqs * decode_query_len and num_tokens <= ws.qkv_out.shape[0],
        "token count",
    )
    _check(
        num_reqs == 1 and decode_query_len == 1,
        "the megakernel runs one request with one query token (one "
        "attention cluster); use the m3_mx_gemv + m3_fused_decode_attn + m3_mx_gemv path otherwise",
    )
    _check(
        xq.dtype == torch.float8_e4m3fn and xq.shape[1] == 6144 and xq.stride(1) == 1,
        "xq must be e4m3 [T, 6144]",
    )
    _check(qkv_weight.shape == (PACKED_WIDTH, 6144), "QKV weight must be [2560, 6144]")
    out2d = output.view(num_tokens, NUM_HEADS * HEAD_DIM)
    q8, sf = ws.q8, ws.sf
    _check(
        y.dtype == torch.bfloat16 and y.shape == (num_tokens, 6144) and y.is_contiguous(),
        "y must be bf16 [T, 6144]",
    )
    qg, bpc = 1, 1  # one 16-CTA attention cluster, one KV block per CTA
    _load_module()
    _m3_attn_mega_op(
        xq.view(torch.uint8),
        xsf.view(torch.uint8),
        qkv_weight.view(torch.uint8),
        qkv_weight_scale.view(torch.uint8),
        ws.qkv_out,
        ws.flags,
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
        o_weight.view(torch.uint8),
        o_weight_scale.view(torch.uint8),
        y,
        *((ar[0], ar[1], ar[2], ar[3], ar[4], 2) if ar is not None else (0, 0, 0, 1, 8, 0)),
    )
    out_rows = ws.qkv_out[ws.calls & 1, :num_tokens]
    ws.calls += 1
    return out_rows


__all__ = ["MegaWorkspace", "m3_attn_mega"]
