# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL kernel for GLM MLA **context/prefill** RoPE + latent-KV append + FP8-Q.

Replaces, for one context step, the two native kernels that TensorRT-LLM runs
before the sparse-MLA context FMHA (``applyMLARopeAndAssignQKVKernelOptContext``
in absorption mode and ``quantizeCopyInputToFp8Kernel``; the Python op
``torch.ops.trtllm.mla_rope_append_paged_kv_assign_q`` is the same RoPE/append
math with an in-place q layout):

1. GPT-J RoPE of ``q_pe`` (the 64-wide RoPE slice of the projected q, read as a
   strided ``[T, 16, 64]`` view) written into ``fused_q[..., 512:576]`` (BF16).
2. GPT-J RoPE of ``k_pe = latent_cache[:, 512:576]``, written back to
   ``latent_cache`` (BF16) and, as E4M3, into the paged FP8 latent KV cache row
   of the token; ``c_kv = latent_cache[:, :512]`` copied as E4M3 into the same
   row.
3. ``quant_q_buffer = E4M3(fused_q)`` for all 576 columns (FP8 KV scale is 1 in
   TRT-LLM's PyTorch MLA, so the native quantize kernel is a plain cast).

Numerics follow the native kernels exactly: RoPE pairs ``(x_even, x_odd)`` are
rotated in FP32 (``c*x - s*y``, ``c*y + s*x``), rounded to BF16 (RN), and the
E4M3 values are ``cvt.rn.satfinite`` of those BF16 values (``__nv_fp8x2_e4m3(
bf1622float2(float22bf162(v)))`` in ``quantCopy``). Position of token ``t`` of
request ``b`` is ``t - cu_seqlens[b] + cached_lens[b]`` (fresh prefill:
``cached_lens == 0``). The cos/sin table is TRT-LLM's ``rotary_cos_sin``
``[positions, 64, 2]`` FP32 (pair ``p`` at ``[pos, p, :] = (cos, sin)``; pairs
``0..31`` are used by the 64-wide RoPE slice, matching ``cos_sin_cache +
ROPE_DIM * position + head_dim_idx / 2`` in the CUDA kernel).

Paged cache addressing (``KVBlockArray`` for MLA, one K head of 576 E4M3 bytes
per token): byte offset of token ``pos`` of request ``b`` inside the layer's
primary pool is ``block_offsets[b, 0, pos // tokens_per_block] * tokens_per_block
* 576 + (pos % tokens_per_block) * 576``; the layer base pointer is
``primary_pool_ptr + layer_idx_in_pool * tokens_per_block * 576`` (same static
adjustment as ``buildPagedKvCacheBuffers``).

Device code style follows ``self_attn_mla_rope_generation_*.py`` (the decode
RoPE kernel): ``cutlass.experimental.primitives`` only, typed pointer
loads/stores, no CuTe copy/GEMM/pipeline. Work decomposition per 32-token tile:
16 q-RoPE groups (one per head), 1 k-RoPE group, 8 c_kv copy groups, 128 q-base
quant groups (16 heads x 8 dim groups); 256 threads per CTA.
"""

from __future__ import annotations

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir import ir as _mlir_ir
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir.dialects import nvvm as _nvvm
from cutlass.cutlass_dsl import T as _T
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

from ..common.cuda_stream import raw_stream as _raw_stream

NUM_HEADS = 16
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 192
ROPE_DIM = 64
HEAD_SIZE = KV_LORA_RANK + ROPE_DIM  # 576: fused q head / latent row
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + ROPE_DIM  # 256: projected q head (q_pe source)
THREADS = 256
TOKENS_PER_CTA = 32
COPY_TOKENS_PER_CTA = 4
KV_COPY_GROUPS = TOKENS_PER_CTA // COPY_TOKENS_PER_CTA  # 8
Q_BASE_GROUPS_PER_HEAD = TOKENS_PER_CTA // COPY_TOKENS_PER_CTA  # 8 token sub-groups
Q_ROPE_GROUPS = NUM_HEADS
K_ROPE_GROUP = Q_ROPE_GROUPS
KV_COPY_START = K_ROPE_GROUP + 1
Q_BASE_START = KV_COPY_START + KV_COPY_GROUPS
WORK_GROUPS = Q_BASE_START + NUM_HEADS * Q_BASE_GROUPS_PER_HEAD  # 153
# Merged RoPE work group for the q_in_place modes (the non-absorbed context attention, BF16 or
# E4M3 Q). The 16 q-RoPE groups and the k-RoPE group of a token tile run in ONE CTA: each
# thread locates its token's request and loads its cos/sin once, issues all 17 vector loads,
# then ropes and stores. Separate groups launch 17 CTAs per tile that each repeat the 6-probe
# request search for one 16-byte vector per thread; at T ~ 7.3k (q_in_place) that is ~5.7k CTAs
# of latency-chained work, 12.4 us/layer vs a ~6 us DRAM floor. Measured on B300 at T = 7255:
# L2-hot 9.11 -> 7.01 us (BF16 Q), 9.20 -> 6.73 us (E4M3 Q), L2-flushed unchanged (17.4 us);
# the absorbed modes (fused_q / quant_q stores per head) were slower merged (9.97 -> 11.87 us)
# and keep the separate groups. Same per-element arithmetic and destinations, so outputs are
# bitwise identical; only the work decomposition changes (grid y: rope 1 + c_kv copy 8
# [+ q-base 128]).
MERGE_ROPE_GROUPS = True
MERGED_KV_COPY_START = 1
MERGED_Q_BASE_START = MERGED_KV_COPY_START + KV_COPY_GROUPS
TOKENS_PER_BLOCK = 32
MAX_REQUESTS = 64  # context requests per step (B32 wave: <= 32)
DEFAULT_Q_PE_LD = QK_HEAD_DIM
DEFAULT_Q_PE_STRIDE = NUM_HEADS * QK_HEAD_DIM

_COMPILE_CACHE: dict[tuple[object, ...], object] = {}


@dataclass(frozen=True)
class PrefillRopeSpecialization:
    """Compile-time parameters (a different value compiles a different kernel)."""

    tokens_per_block: int = TOKENS_PER_BLOCK
    q_pe_ld: int = DEFAULT_Q_PE_LD
    q_pe_stride: int = DEFAULT_Q_PE_STRIDE
    latent_cache_stride: int = HEAD_SIZE
    # Launch only the rope and c_kv copy groups (grid y = Q_BASE_START): the
    # caller's K-B GEMM already wrote e4m3(q_base) into quant_q_buffer[..., :512],
    # so the 128 q-base quantize groups (and their 129 MB bf16 re-read at
    # T ~ 8k) are skipped. fused_q[..., :512] is then neither read nor needed.
    skip_q_base: bool = False
    # Non-absorbed context route (dense-causal MHA): rope(q_pe) is written back in place
    # into q_pe (the q_b projection output, which is then the MHA's Q), and no E4M3 Q is
    # written at all (no rope tail, no q-base groups). fused_q / quant_q_buffer are not
    # touched; the k_pe rope, the latent write-back and the paged-pool writes are the same
    # code path as the served mode, so the decode cache bytes are unchanged.
    q_in_place: bool = False
    # FP8 context MHA (with q_in_place): q_pe is a view of the E4M3 Q the q_b GEMM's E4M3 epilogue
    # wrote ([T,16,256], a per-tensor power-of-two scale already applied); rope(q_pe) is computed
    # from the widened E4M3 values (E4M3 -> BF16 is exact; rope is linear, so the scale needs no
    # handling) and written back in place as E4M3 (RN, satfinite). Everything else as q_in_place.
    q_e4m3: bool = False
    # One CTA per token tile for all RoPE groups (see MERGE_ROPE_GROUPS; prepare_runner
    # turns it on for the q_in_place modes only).
    merge_rope: bool = False
    # context-step stitch: launch the RoPE groups only (grid y = 1 merged / 17 separate). The caller's
    # q_a/kv_a RMSNorm (``compile_qkv_a_rmsnorm_warp_row_q8_ctx`` with ``kv_append``) already
    # wrote every context row's c_kv E4M3 bytes into the pool rows; the k-RoPE group still
    # writes the k_pe bytes [512:576] and the latent k_pe slot. Needs the copy groups to be
    # the last launched groups, i.e. ``q_in_place`` or ``skip_q_base``.
    skip_kv_copy: bool = False
    # rope_par (0 = off): the served context mode (q_in_place + q_e4m3 + merged RoPE) on the
    # ``..._par_kernel`` build with the tile's RoPE work split over ``rope_par`` CTAs (block
    # comment above ``_cvt_bf16x2_rn_word``); same output bytes.
    rope_par: int = 0

    def validate(self) -> None:
        if self.tokens_per_block <= 0 or (self.tokens_per_block & (self.tokens_per_block - 1)):
            raise ValueError(
                f"tokens_per_block must be a power of two, got {self.tokens_per_block}"
            )
        if self.q_pe_ld < ROPE_DIM or self.q_pe_ld % 8 != 0:
            raise ValueError(
                f"q_pe_ld must be >= {ROPE_DIM} and a multiple of 8, got {self.q_pe_ld}"
            )
        if self.q_pe_stride < NUM_HEADS * self.q_pe_ld or self.q_pe_stride % 8 != 0:
            raise ValueError("q_pe_stride must cover all heads and be a multiple of 8")
        if self.latent_cache_stride < HEAD_SIZE or self.latent_cache_stride % 8 != 0:
            raise ValueError("latent_cache_stride must be >= 576 and a multiple of 8")
        if self.q_e4m3 and not self.q_in_place:
            raise ValueError("q_e4m3 is a variant of q_in_place")
        if self.skip_kv_copy and not (self.q_in_place or self.skip_q_base):
            raise ValueError("skip_kv_copy needs q_in_place or skip_q_base")
        if self.rope_par:
            if self.rope_par not in ROPE_PAR_CHOICES:
                raise ValueError(f"rope_par must be 0 or one of {ROPE_PAR_CHOICES}")
            if not (self.q_in_place and self.q_e4m3 and self.merge_rope) or self.skip_kv_copy:
                raise ValueError("rope_par is a build of the q_in_place + q_e4m3 merged mode")


@dataclass
class PreparedPrefillRopeRunner:
    """Callable launcher with all device pointers and views prebound."""

    compiled: object
    args: tuple[object, ...]
    device_index: int
    stream: cuda.CUstream | None
    specialization: PrefillRopeSpecialization
    owners: tuple[object, ...] = ()

    def __call__(self) -> None:
        stream = self.stream
        if stream is None:
            stream = _current_stream(self.device_index)
        self.compiled(*self.args, stream)


def _device_index(tensor: torch.Tensor) -> int:
    idx = tensor.device.index
    return idx if idx is not None else torch.cuda.current_device()


def _current_stream(device_index: int) -> cuda.CUstream:
    return cuda.CUstream(_raw_stream(device_index))


# ----------------------------------------------------------------------------- device helpers
@cute.jit
def _bf16_rn(value):
    """Round fp32 to bf16 precision (RN-even) without relying on contraction."""
    bits = value.bitcast(cutlass.Int32)
    bits = bits + (((bits >> 16) & 1) + 0x7FFF)
    bits = (bits >> 16) << 16
    return bits.bitcast(cutlass.Float32)


@cute.jit
def _rope_even_bf16_rn(x_even, x_odd, cos, sin) -> cutlass.Float32:
    # nvcc contracts rotary_embedding_transform's ``c*x - s*y`` as
    # mul.f32(s, y) then fma.rn.f32(c, x, -tmp): the product carrying ``v.x`` is fused.
    # Reproduce that exact fp32 rounding; verified bit-exact against the native
    # op on bf16 rounding ties.
    tmp = sin * x_odd
    return _bf16_rn(cute.math.fma(cos, x_even, cutlass.Float32(0.0) - tmp))


@cute.jit
def _rope_odd_bf16_rn(x_even, x_odd, cos, sin) -> cutlass.Float32:
    # ``c*y + s*x`` -> mul.f32(c, y) then fma.rn.f32(s, x, tmp): the ``v.x`` product is fused.
    tmp = cos * x_odd
    return _bf16_rn(cute.math.fma(sin, x_even, tmp))


@cute.jit
def _pack_e4m3x2_word(value0, value1):
    """Two adjacent E4M3 bytes as one little-endian Int16 word (RN, satfinite)."""
    return prims.cvt_f32x2_to_f8x2(
        cutlass.Float32(value1),
        cutlass.Float32(value0),
        cutlass.Float8E4M3FN,
        rnd=prims.FPRoundingMode.RN,
        sat=prims.SaturationMode.SATFINITE,
    )


@cute.jit
def _store_e4m3x8(dst, idx, v0, v1, v2, v3, v4, v5, v6, v7) -> None:
    packed = cutlass.Vector.from_elements(
        (
            _pack_e4m3x2_word(v0, v1),
            _pack_e4m3x2_word(v2, v3),
            _pack_e4m3x2_word(v4, v5),
            _pack_e4m3x2_word(v6, v7),
        ),
        dtype=cutlass.Int16,
    )
    ptr = dst.data_ptr(idx)
    ptr_i16 = cutlass.inttoptr(ptr.toint(), ptr.space, cutlass.Int16)
    ptr_i16.store(packed, alignment=8)


@cute.jit
def _bf16x2_to_f32(packed):
    """The two BF16 halves of one Int32 as exact FP32 values (low half first)."""
    lo = (packed << cutlass.Int32(16)).bitcast(cutlass.Float32)
    hi = (packed & cutlass.Int32(-65536)).bitcast(cutlass.Float32)
    return lo, hi


@cute.jit
def _load_e4m3x8_words(src, idx):
    """Eight E4M3 bytes at ``src[idx]`` (8-byte aligned) as two Int32 words (no widening)."""
    ptr = src.data_ptr(idx)
    ptr_i32 = cutlass.inttoptr(ptr.toint(), ptr.space, cutlass.Int32)
    return ptr_i32.load(count=2, alignment=8)


@cute.jit
def _e4m3x8_words_to_f32(words):
    """The eight E4M3 values of ``_load_e4m3x8_words`` as exact FP32 values, in order."""
    x0, x1 = _bf16x2_to_f32(prims.cvt_e4m3x2_word_to_bf16x2(words[0], 0))
    x2, x3 = _bf16x2_to_f32(prims.cvt_e4m3x2_word_to_bf16x2(words[0], 16))
    x4, x5 = _bf16x2_to_f32(prims.cvt_e4m3x2_word_to_bf16x2(words[1], 0))
    x6, x7 = _bf16x2_to_f32(prims.cvt_e4m3x2_word_to_bf16x2(words[1], 16))
    return (x0, x1, x2, x3, x4, x5, x6, x7)


@cute.jit
def _load_e4m3x8_f32(src, idx):
    """Eight E4M3 bytes at ``src[idx]`` (8-byte aligned) as exact FP32 values, in order."""
    ptr = src.data_ptr(idx)
    ptr_i32 = cutlass.inttoptr(ptr.toint(), ptr.space, cutlass.Int32)
    words = ptr_i32.load(count=2, alignment=8)
    x0, x1 = _bf16x2_to_f32(prims.cvt_e4m3x2_word_to_bf16x2(words[0], 0))
    x2, x3 = _bf16x2_to_f32(prims.cvt_e4m3x2_word_to_bf16x2(words[0], 16))
    x4, x5 = _bf16x2_to_f32(prims.cvt_e4m3x2_word_to_bf16x2(words[1], 0))
    x6, x7 = _bf16x2_to_f32(prims.cvt_e4m3x2_word_to_bf16x2(words[1], 16))
    return (x0, x1, x2, x3, x4, x5, x6, x7)


@cute.jit
def _store_bf16x8(dst, idx, v0, v1, v2, v3, v4, v5, v6, v7) -> None:
    """Store eight bf16-exact fp32 values as one 16-byte bf16 vector."""
    vec = cutlass.Vector.from_elements(
        (
            cutlass.BFloat16(v0),
            cutlass.BFloat16(v1),
            cutlass.BFloat16(v2),
            cutlass.BFloat16(v3),
            cutlass.BFloat16(v4),
            cutlass.BFloat16(v5),
            cutlass.BFloat16(v6),
            cutlass.BFloat16(v7),
        ),
        dtype=cutlass.BFloat16,
    )
    dst.data_ptr(idx).store(vec, alignment=16)


@cute.jit
def _locate_request(cu_seqlens, num_requests, token) -> cutlass.Int32:
    """Request index b: the largest b < num_requests with cu_seqlens[b] <= token.
    Fixed-step binary search (MAX_REQUESTS = 64 -> 6 probes) instead of a 63-load scan;
    same result for non-decreasing cu_seqlens, including zero-length requests."""
    req = cutlass.Int32(0)
    for i in cutlass.range_constexpr(6):
        cand = req + cutlass.Int32(32 >> i)
        if cand < num_requests:
            if token >= cutlass.Int32(cu_seqlens.load(idx=cand)):
                req = cand
    return req


@cute.jit
def _rope8_e4m3(src_vec, cos_sin):
    """Rotate four (even, odd) bf16 pairs; return the eight bf16-rounded fp32 values."""
    x0 = cutlass.Float32(src_vec[0])
    x1 = cutlass.Float32(src_vec[1])
    x2 = cutlass.Float32(src_vec[2])
    x3 = cutlass.Float32(src_vec[3])
    x4 = cutlass.Float32(src_vec[4])
    x5 = cutlass.Float32(src_vec[5])
    x6 = cutlass.Float32(src_vec[6])
    x7 = cutlass.Float32(src_vec[7])
    c0 = cutlass.Float32(cos_sin[0])
    s0 = cutlass.Float32(cos_sin[1])
    c1 = cutlass.Float32(cos_sin[2])
    s1 = cutlass.Float32(cos_sin[3])
    c2 = cutlass.Float32(cos_sin[4])
    s2 = cutlass.Float32(cos_sin[5])
    c3 = cutlass.Float32(cos_sin[6])
    s3 = cutlass.Float32(cos_sin[7])
    y0 = _rope_even_bf16_rn(x0, x1, c0, s0)
    y1 = _rope_odd_bf16_rn(x0, x1, c0, s0)
    y2 = _rope_even_bf16_rn(x2, x3, c1, s1)
    y3 = _rope_odd_bf16_rn(x2, x3, c1, s1)
    y4 = _rope_even_bf16_rn(x4, x5, c2, s2)
    y5 = _rope_odd_bf16_rn(x4, x5, c2, s2)
    y6 = _rope_even_bf16_rn(x6, x7, c3, s3)
    y7 = _rope_odd_bf16_rn(x6, x7, c3, s3)
    return y0, y1, y2, y3, y4, y5, y6, y7


# ----------------------------------------------------------------------------- device kernel
@cute.kernel
def glm52_self_attn_prefill_mla_rope_append_fp8q_kernel(
    fused_q: cutlass.Array,
    q_pe: cutlass.Array,
    latent_cache: cutlass.Array,
    quant_q_buffer: cutlass.Array,
    kv_cache_storage: cutlass.Array,
    kv_cache_block_offsets: cutlass.Array,
    cu_seqlens: cutlass.Array,
    cached_lens: cutlass.Array,
    rotary_cos_sin: cutlass.Array,
    num_tokens: cutlass.Int64,
    num_requests: cutlass.Int64,
    max_blocks_per_seq: cutlass.Int64,
    tokens_per_block: cutlass.Constexpr[int],
    q_pe_ld: cutlass.Constexpr[int],
    q_pe_stride: cutlass.Constexpr[int],
    latent_cache_stride: cutlass.Constexpr[int],
    q_in_place: cutlass.Constexpr[bool],
    q_e4m3: cutlass.Constexpr[bool],
    merge_rope: cutlass.Constexpr[bool],
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    token_tile, work_group, _ = cute.arch.block_idx()
    if cutlass.const_expr(merge_rope):
        # grid y = [rope (all 17 groups)] + [8 c_kv copy groups] + [q-base groups]: map the
        # launch index onto the separate-groups numbering the branches below use.
        if work_group >= cutlass.Int32(MERGED_KV_COPY_START):
            work_group = work_group + cutlass.Int32(KV_COPY_START - MERGED_KV_COPY_START)
    n_tokens = cutlass.Int32(num_tokens)
    n_req = cutlass.Int32(num_requests)
    max_blocks = cutlass.Int32(max_blocks_per_seq)
    tile_base = token_tile * cutlass.Int32(TOKENS_PER_CTA)

    if work_group < cutlass.Int32(KV_COPY_START):
        # ---- RoPE groups: 8 lanes per token, 8 elements per lane (64 RoPE dims) ----
        local_token = tidx // cutlass.Int32(8)
        lane = tidx - local_token * cutlass.Int32(8)
        token = tile_base + local_token
        if token < n_tokens:
            req = _locate_request(cu_seqlens, n_req, token)
            position = (
                token
                - cutlass.Int32(cu_seqlens.load(idx=req))
                + cutlass.Int32(cached_lens.load(idx=req))
            )
            dim_base = lane * cutlass.Int32(8)
            table_base = (position * ROPE_DIM + lane * cutlass.Int32(4)) * 2
            cos_sin = rotary_cos_sin.data_ptr(table_base).load(count=8, alignment=16)
            if cutlass.const_expr(merge_rope):
                # merged: this CTA ropes all 16 q heads and k_pe of its token tile; the
                # request search and the cos/sin load above are shared by all 17. Every
                # load is issued before the first store (an in-place store could alias a
                # later head's load, which would serialize 17 round trips per thread).
                q_srcs = []
                q_raw = []
                for q_head_c in cutlass.range_constexpr(Q_ROPE_GROUPS):
                    src = token * q_pe_stride + cutlass.Int32(q_head_c) * q_pe_ld + dim_base
                    q_srcs.append(src)
                    if cutlass.const_expr(q_e4m3):
                        q_raw.append(_load_e4m3x8_words(q_pe, src))
                    else:
                        q_raw.append(q_pe.data_ptr(src).load(count=8, alignment=16))
                k_src = token * latent_cache_stride + KV_LORA_RANK + dim_base
                k_vec = latent_cache.data_ptr(k_src).load(count=8, alignment=16)
                for q_head_c in cutlass.range_constexpr(Q_ROPE_GROUPS):
                    q_head = cutlass.Int32(q_head_c)
                    src = q_srcs[q_head_c]
                    if cutlass.const_expr(q_e4m3):
                        # FP8 MHA Q: rope the widened E4M3 elements, rewrite them as E4M3
                        q_vec8 = _e4m3x8_words_to_f32(q_raw[q_head_c])
                        y0, y1, y2, y3, y4, y5, y6, y7 = _rope8_e4m3(q_vec8, cos_sin)
                        _store_e4m3x8(q_pe, src, y0, y1, y2, y3, y4, y5, y6, y7)
                    elif cutlass.const_expr(q_in_place):
                        y0, y1, y2, y3, y4, y5, y6, y7 = _rope8_e4m3(q_raw[q_head_c], cos_sin)
                        _store_bf16x8(q_pe, src, y0, y1, y2, y3, y4, y5, y6, y7)
                    else:
                        y0, y1, y2, y3, y4, y5, y6, y7 = _rope8_e4m3(q_raw[q_head_c], cos_sin)
                        dst = (
                            token * NUM_HEADS * HEAD_SIZE
                            + q_head * HEAD_SIZE
                            + KV_LORA_RANK
                            + dim_base
                        )
                        _store_bf16x8(fused_q, dst, y0, y1, y2, y3, y4, y5, y6, y7)
                        _store_e4m3x8(quant_q_buffer, dst, y0, y1, y2, y3, y4, y5, y6, y7)
                y0, y1, y2, y3, y4, y5, y6, y7 = _rope8_e4m3(k_vec, cos_sin)
                _store_bf16x8(latent_cache, k_src, y0, y1, y2, y3, y4, y5, y6, y7)
                block_in_seq = position // cutlass.Int32(tokens_per_block)
                local_cache_token = position & cutlass.Int32(tokens_per_block - 1)
                if block_in_seq < max_blocks:
                    offset_idx = req * 2 * max_blocks + block_in_seq
                    block_offset = cutlass.Int32(kv_cache_block_offsets.load(idx=offset_idx))
                    cache_base = cutlass.Int64(block_offset) * cutlass.Int64(
                        tokens_per_block * HEAD_SIZE
                    ) + cutlass.Int64(local_cache_token * HEAD_SIZE)
                    _store_e4m3x8(
                        kv_cache_storage,
                        cache_base + KV_LORA_RANK + dim_base,
                        y0,
                        y1,
                        y2,
                        y3,
                        y4,
                        y5,
                        y6,
                        y7,
                    )
            else:
                if work_group < cutlass.Int32(Q_ROPE_GROUPS):
                    q_head = work_group
                    src = token * q_pe_stride + q_head * q_pe_ld + dim_base
                    if cutlass.const_expr(q_e4m3):
                        # FP8 MHA Q: rope the widened E4M3 elements, rewrite them as E4M3
                        q_vec8 = _load_e4m3x8_f32(q_pe, src)
                        y0, y1, y2, y3, y4, y5, y6, y7 = _rope8_e4m3(q_vec8, cos_sin)
                        _store_e4m3x8(q_pe, src, y0, y1, y2, y3, y4, y5, y6, y7)
                    elif cutlass.const_expr(q_in_place):
                        q_vec = q_pe.data_ptr(src).load(count=8, alignment=16)
                        y0, y1, y2, y3, y4, y5, y6, y7 = _rope8_e4m3(q_vec, cos_sin)
                        # each lane reads and rewrites its own 8 elements (4 whole pairs)
                        _store_bf16x8(q_pe, src, y0, y1, y2, y3, y4, y5, y6, y7)
                    else:
                        q_vec = q_pe.data_ptr(src).load(count=8, alignment=16)
                        y0, y1, y2, y3, y4, y5, y6, y7 = _rope8_e4m3(q_vec, cos_sin)
                        dst = (
                            token * NUM_HEADS * HEAD_SIZE
                            + q_head * HEAD_SIZE
                            + KV_LORA_RANK
                            + dim_base
                        )
                        _store_bf16x8(fused_q, dst, y0, y1, y2, y3, y4, y5, y6, y7)
                        _store_e4m3x8(quant_q_buffer, dst, y0, y1, y2, y3, y4, y5, y6, y7)
                else:
                    k_src = token * latent_cache_stride + KV_LORA_RANK + dim_base
                    k_vec = latent_cache.data_ptr(k_src).load(count=8, alignment=16)
                    y0, y1, y2, y3, y4, y5, y6, y7 = _rope8_e4m3(k_vec, cos_sin)
                    _store_bf16x8(latent_cache, k_src, y0, y1, y2, y3, y4, y5, y6, y7)
                    block_in_seq = position // cutlass.Int32(tokens_per_block)
                    local_cache_token = position & cutlass.Int32(tokens_per_block - 1)
                    if block_in_seq < max_blocks:
                        offset_idx = req * 2 * max_blocks + block_in_seq
                        block_offset = cutlass.Int32(kv_cache_block_offsets.load(idx=offset_idx))
                        cache_base = cutlass.Int64(block_offset) * cutlass.Int64(
                            tokens_per_block * HEAD_SIZE
                        ) + cutlass.Int64(local_cache_token * HEAD_SIZE)
                        _store_e4m3x8(
                            kv_cache_storage,
                            cache_base + KV_LORA_RANK + dim_base,
                            y0,
                            y1,
                            y2,
                            y3,
                            y4,
                            y5,
                            y6,
                            y7,
                        )
    else:
        # ---- copy groups: 64 lanes x 8 dims = 512 columns, 4 tokens per group ----
        copy_dim_group = tidx & cutlass.Int32(63)
        copy_token_lane = tidx >> cutlass.Int32(6)
        dim = copy_dim_group * cutlass.Int32(8)
        if work_group < cutlass.Int32(Q_BASE_START):
            kv_group = work_group - cutlass.Int32(KV_COPY_START)
            token = tile_base + kv_group * cutlass.Int32(COPY_TOKENS_PER_CTA) + copy_token_lane
            if token < n_tokens:
                req = _locate_request(cu_seqlens, n_req, token)
                position = (
                    token
                    - cutlass.Int32(cu_seqlens.load(idx=req))
                    + cutlass.Int32(cached_lens.load(idx=req))
                )
                block_in_seq = position // cutlass.Int32(tokens_per_block)
                local_cache_token = position & cutlass.Int32(tokens_per_block - 1)
                if block_in_seq < max_blocks:
                    offset_idx = req * 2 * max_blocks + block_in_seq
                    block_offset = cutlass.Int32(kv_cache_block_offsets.load(idx=offset_idx))
                    cache_base = cutlass.Int64(block_offset) * cutlass.Int64(
                        tokens_per_block * HEAD_SIZE
                    ) + cutlass.Int64(local_cache_token * HEAD_SIZE)
                    src = token * latent_cache_stride + dim
                    _store_e4m3x8(
                        kv_cache_storage,
                        cache_base + dim,
                        latent_cache.load(idx=src),
                        latent_cache.load(idx=src + 1),
                        latent_cache.load(idx=src + 2),
                        latent_cache.load(idx=src + 3),
                        latent_cache.load(idx=src + 4),
                        latent_cache.load(idx=src + 5),
                        latent_cache.load(idx=src + 6),
                        latent_cache.load(idx=src + 7),
                    )
        else:
            q_group = work_group - cutlass.Int32(Q_BASE_START)
            q_head = q_group & cutlass.Int32(NUM_HEADS - 1)
            q_token_group = q_group >> cutlass.Int32(4)
            token = tile_base + q_token_group * cutlass.Int32(COPY_TOKENS_PER_CTA) + copy_token_lane
            if token < n_tokens:
                src = token * NUM_HEADS * HEAD_SIZE + q_head * HEAD_SIZE + dim
                _store_e4m3x8(
                    quant_q_buffer,
                    src,
                    fused_q.load(idx=src),
                    fused_q.load(idx=src + 1),
                    fused_q.load(idx=src + 2),
                    fused_q.load(idx=src + 3),
                    fused_q.load(idx=src + 4),
                    fused_q.load(idx=src + 5),
                    fused_q.load(idx=src + 6),
                    fused_q.load(idx=src + 7),
                )


@cute.jit
def _launch(
    fused_q: cute.Tensor,
    q_pe: cute.Tensor,
    latent_cache: cute.Tensor,
    quant_q_buffer: cute.Tensor,
    kv_cache_storage: cute.Pointer,
    kv_cache_block_offsets: cute.Tensor,
    cu_seqlens: cute.Tensor,
    cached_lens: cute.Tensor,
    rotary_cos_sin: cute.Tensor,
    stream: cuda.CUstream,
    tokens_per_block: cutlass.Constexpr[int],
    q_pe_ld: cutlass.Constexpr[int],
    q_pe_stride: cutlass.Constexpr[int],
    latent_cache_stride: cutlass.Constexpr[int],
    skip_q_base: cutlass.Constexpr[bool],
    q_in_place: cutlass.Constexpr[bool],
    q_e4m3: cutlass.Constexpr[bool],
    merge_rope: cutlass.Constexpr[bool],
    skip_kv_copy: cutlass.Constexpr[bool] = False,
) -> None:
    num_tokens = fused_q.shape[0]
    if cutlass.const_expr(q_in_place):
        num_tokens = q_pe.shape[0]  # fused_q / quant_q_buffer are placeholders
    num_requests = cached_lens.shape[0]
    max_blocks_per_seq = kv_cache_block_offsets.shape[2]
    storage = cute.make_tensor(kv_cache_storage, cute.make_layout((1,), stride=(1,)))
    grid_x = (num_tokens + TOKENS_PER_CTA - 1) // TOKENS_PER_CTA
    grid_y = WORK_GROUPS
    if cutlass.const_expr(skip_q_base or q_in_place):
        grid_y = Q_BASE_START
    if cutlass.const_expr(skip_kv_copy):
        grid_y = KV_COPY_START
    if cutlass.const_expr(merge_rope):
        grid_y = grid_y - (KV_COPY_START - MERGED_KV_COPY_START)
    glm52_self_attn_prefill_mla_rope_append_fp8q_kernel(
        fused_q,
        q_pe,
        latent_cache,
        quant_q_buffer,
        storage,
        kv_cache_block_offsets,
        cu_seqlens,
        cached_lens,
        rotary_cos_sin,
        num_tokens,
        num_requests,
        max_blocks_per_seq,
        tokens_per_block,
        q_pe_ld,
        q_pe_stride,
        latent_cache_stride,
        q_in_place,
        q_e4m3,
        merge_rope,
    ).launch(
        grid=(grid_x, grid_y, 1),
        block=(THREADS, 1, 1),
        stream=stream,
    )


# ----------------------------------------------------------------------------- rope_par build
# Build option ``rope_par = RS`` (2 or 4) of the served context mode (``q_in_place`` + ``q_e4m3``
# + merged RoPE, no q-base groups, no ``skip_kv_copy``): a separate kernel
# ``glm52_self_attn_prefill_mla_rope_append_fp8q_par_kernel`` (the served build is untouched).
# The served merged RoPE CTA ropes all 16 q heads and k_pe of a 32-token tile: ~231 such CTAs at
# T ~ 7.4k (< 2 per SM), each thread issuing 17 vector loads and then 17 x 8 rotations in series
# behind a 6-probe request search -- the kernel is latency-bound (1.74 waves, issue
# slots 24 %, 61 % of stall samples long-scoreboard, 20.7 cycles per issued instruction). Here:
#   * the tile's RoPE work is split over RS CTAs (16 / RS q heads each; k_pe in the last), so the
#     per-thread serial rope chain is RS x shorter and RS x more SMs start on it at once;
#   * every RoPE CTA issues its q / k_pe loads before the request search (their addresses depend
#     on the token only), so the search's dependent probes overlap the data loads;
#   * the BF16 rounding of the rotated pairs is one packed ``cvt.rn.bf16x2.f32`` per pair (the
#     served ``_bf16_rn`` integer RN-even rounding: identical value for every finite input);
#   * the c_kv copy groups read their 8 BF16 values as one 16-byte vector (8 scalar loads before).
# Same per-element FP32 arithmetic (``FMUL`` + ``FFMA`` order of ``_rope_even/odd_bf16_rn``), same
# destinations and bytes as the served build.
ROPE_PAR_CHOICES = (2, 4)


@dsl_user_op
def _cvt_bf16x2_rn_word(hi, lo, *, loc=None, ip=None) -> cutlass.Int32:
    """Two FP32 values rounded to BF16 (RN-even) by one ``cvt.rn.bf16x2.f32`` (``hi`` in bits
    31..16, ``lo`` in 15..0)."""
    v = _nvvm.convert_f32x2_to_bf16x2(
        _mlir_ir.VectorType.get([2], cutlass.BFloat16.mlir_type, loc=loc),
        cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
        cutlass.Float32(lo).ir_value(loc=loc, ip=ip),
        rnd=_nvvm.FPRoundingMode.RN,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(_llvm.bitcast(_T.i32(), v, loc=loc, ip=ip))


@cute.jit
def _rope_pair_pack(x_even, x_odd, cos, sin):
    """``(_rope_even_bf16_rn, _rope_odd_bf16_rn)`` of one pair with one packed rounding."""
    tmp_e = sin * x_odd
    pre_e = cute.math.fma(cos, x_even, cutlass.Float32(0.0) - tmp_e)
    tmp_o = cos * x_odd
    pre_o = cute.math.fma(sin, x_even, tmp_o)
    return _bf16x2_to_f32(_cvt_bf16x2_rn_word(pre_o, pre_e))


@cute.jit
def _rope8_pack(src_vec, cos_sin):
    """``_rope8_e4m3`` with the packed BF16 rounding (same eight values)."""
    y0, y1 = _rope_pair_pack(
        cutlass.Float32(src_vec[0]),
        cutlass.Float32(src_vec[1]),
        cutlass.Float32(cos_sin[0]),
        cutlass.Float32(cos_sin[1]),
    )
    y2, y3 = _rope_pair_pack(
        cutlass.Float32(src_vec[2]),
        cutlass.Float32(src_vec[3]),
        cutlass.Float32(cos_sin[2]),
        cutlass.Float32(cos_sin[3]),
    )
    y4, y5 = _rope_pair_pack(
        cutlass.Float32(src_vec[4]),
        cutlass.Float32(src_vec[5]),
        cutlass.Float32(cos_sin[4]),
        cutlass.Float32(cos_sin[5]),
    )
    y6, y7 = _rope_pair_pack(
        cutlass.Float32(src_vec[6]),
        cutlass.Float32(src_vec[7]),
        cutlass.Float32(cos_sin[6]),
        cutlass.Float32(cos_sin[7]),
    )
    return y0, y1, y2, y3, y4, y5, y6, y7


@cute.kernel
def glm52_self_attn_prefill_mla_rope_append_fp8q_par_kernel(
    q_pe: cutlass.Array,
    latent_cache: cutlass.Array,
    kv_cache_storage: cutlass.Array,
    kv_cache_block_offsets: cutlass.Array,
    cu_seqlens: cutlass.Array,
    cached_lens: cutlass.Array,
    rotary_cos_sin: cutlass.Array,
    num_tokens: cutlass.Int64,
    num_requests: cutlass.Int64,
    max_blocks_per_seq: cutlass.Int64,
    tokens_per_block: cutlass.Constexpr[int],
    q_pe_ld: cutlass.Constexpr[int],
    q_pe_stride: cutlass.Constexpr[int],
    latent_cache_stride: cutlass.Constexpr[int],
    rope_par: cutlass.Constexpr[int],
) -> None:
    """The rope_par build (block comment above). Grid ``(ceil(T / 32), rope_par + 8)``: y <
    ``rope_par`` = RoPE part y of the tile, the rest = the served c_kv copy groups."""
    tidx, _, _ = cute.arch.thread_idx()
    token_tile, work_group, _ = cute.arch.block_idx()
    n_tokens = cutlass.Int32(num_tokens)
    n_req = cutlass.Int32(num_requests)
    max_blocks = cutlass.Int32(max_blocks_per_seq)
    tile_base = token_tile * cutlass.Int32(TOKENS_PER_CTA)
    HEADS_PER_PART = Q_ROPE_GROUPS // rope_par

    if work_group < cutlass.Int32(rope_par):
        local_token = tidx // cutlass.Int32(8)
        lane = tidx - local_token * cutlass.Int32(8)
        token = tile_base + local_token
        if token < n_tokens:
            dim_base = lane * cutlass.Int32(8)
            head0 = work_group * cutlass.Int32(HEADS_PER_PART)
            # data loads first: their addresses depend on the token only
            q_srcs = []
            q_raw = []
            for h_c in cutlass.range_constexpr(HEADS_PER_PART):
                src = token * q_pe_stride + (head0 + cutlass.Int32(h_c)) * q_pe_ld + dim_base
                q_srcs.append(src)
                q_raw.append(_load_e4m3x8_words(q_pe, src))
            k_src = token * latent_cache_stride + KV_LORA_RANK + dim_base
            do_k = work_group == cutlass.Int32(rope_par - 1)
            k_vec = latent_cache.data_ptr(k_src).load(count=8, alignment=16)
            req = _locate_request(cu_seqlens, n_req, token)
            position = (
                token
                - cutlass.Int32(cu_seqlens.load(idx=req))
                + cutlass.Int32(cached_lens.load(idx=req))
            )
            table_base = (position * ROPE_DIM + lane * cutlass.Int32(4)) * 2
            cos_sin = rotary_cos_sin.data_ptr(table_base).load(count=8, alignment=16)
            for h_c in cutlass.range_constexpr(HEADS_PER_PART):
                q_vec8 = _e4m3x8_words_to_f32(q_raw[h_c])
                y0, y1, y2, y3, y4, y5, y6, y7 = _rope8_pack(q_vec8, cos_sin)
                _store_e4m3x8(q_pe, q_srcs[h_c], y0, y1, y2, y3, y4, y5, y6, y7)
            if do_k:
                y0, y1, y2, y3, y4, y5, y6, y7 = _rope8_pack(k_vec, cos_sin)
                _store_bf16x8(latent_cache, k_src, y0, y1, y2, y3, y4, y5, y6, y7)
                block_in_seq = position // cutlass.Int32(tokens_per_block)
                local_cache_token = position & cutlass.Int32(tokens_per_block - 1)
                if block_in_seq < max_blocks:
                    offset_idx = req * 2 * max_blocks + block_in_seq
                    block_offset = cutlass.Int32(kv_cache_block_offsets.load(idx=offset_idx))
                    cache_base = cutlass.Int64(block_offset) * cutlass.Int64(
                        tokens_per_block * HEAD_SIZE
                    ) + cutlass.Int64(local_cache_token * HEAD_SIZE)
                    _store_e4m3x8(
                        kv_cache_storage,
                        cache_base + KV_LORA_RANK + dim_base,
                        y0,
                        y1,
                        y2,
                        y3,
                        y4,
                        y5,
                        y6,
                        y7,
                    )
    else:
        # the served c_kv copy groups (64 lanes x 8 dims, 4 tokens per group), 16-byte loads
        copy_dim_group = tidx & cutlass.Int32(63)
        copy_token_lane = tidx >> cutlass.Int32(6)
        dim = copy_dim_group * cutlass.Int32(8)
        kv_group = work_group - cutlass.Int32(rope_par)
        token = tile_base + kv_group * cutlass.Int32(COPY_TOKENS_PER_CTA) + copy_token_lane
        if token < n_tokens:
            src = token * latent_cache_stride + dim
            c_vec = latent_cache.data_ptr(src).load(count=8, alignment=16)
            req = _locate_request(cu_seqlens, n_req, token)
            position = (
                token
                - cutlass.Int32(cu_seqlens.load(idx=req))
                + cutlass.Int32(cached_lens.load(idx=req))
            )
            block_in_seq = position // cutlass.Int32(tokens_per_block)
            local_cache_token = position & cutlass.Int32(tokens_per_block - 1)
            if block_in_seq < max_blocks:
                offset_idx = req * 2 * max_blocks + block_in_seq
                block_offset = cutlass.Int32(kv_cache_block_offsets.load(idx=offset_idx))
                cache_base = cutlass.Int64(block_offset) * cutlass.Int64(
                    tokens_per_block * HEAD_SIZE
                ) + cutlass.Int64(local_cache_token * HEAD_SIZE)
                _store_e4m3x8(
                    kv_cache_storage,
                    cache_base + dim,
                    cutlass.Float32(c_vec[0]),
                    cutlass.Float32(c_vec[1]),
                    cutlass.Float32(c_vec[2]),
                    cutlass.Float32(c_vec[3]),
                    cutlass.Float32(c_vec[4]),
                    cutlass.Float32(c_vec[5]),
                    cutlass.Float32(c_vec[6]),
                    cutlass.Float32(c_vec[7]),
                )


@cute.jit
def _launch_par(
    fused_q: cute.Tensor,
    q_pe: cute.Tensor,
    latent_cache: cute.Tensor,
    quant_q_buffer: cute.Tensor,
    kv_cache_storage: cute.Pointer,
    kv_cache_block_offsets: cute.Tensor,
    cu_seqlens: cute.Tensor,
    cached_lens: cute.Tensor,
    rotary_cos_sin: cute.Tensor,
    stream: cuda.CUstream,
    tokens_per_block: cutlass.Constexpr[int],
    q_pe_ld: cutlass.Constexpr[int],
    q_pe_stride: cutlass.Constexpr[int],
    latent_cache_stride: cutlass.Constexpr[int],
    rope_par: cutlass.Constexpr[int],
) -> None:
    # the ABI of ``_launch`` (fused_q / quant_q_buffer: the q_in_place placeholders, not read)
    num_tokens = q_pe.shape[0]
    num_requests = cached_lens.shape[0]
    max_blocks_per_seq = kv_cache_block_offsets.shape[2]
    storage = cute.make_tensor(kv_cache_storage, cute.make_layout((1,), stride=(1,)))
    grid_x = (num_tokens + TOKENS_PER_CTA - 1) // TOKENS_PER_CTA
    glm52_self_attn_prefill_mla_rope_append_fp8q_par_kernel(
        q_pe,
        latent_cache,
        storage,
        kv_cache_block_offsets,
        cu_seqlens,
        cached_lens,
        rotary_cos_sin,
        num_tokens,
        num_requests,
        max_blocks_per_seq,
        tokens_per_block,
        q_pe_ld,
        q_pe_stride,
        latent_cache_stride,
        rope_par,
    ).launch(
        grid=(grid_x, rope_par + KV_COPY_GROUPS, 1),
        block=(THREADS, 1, 1),
        stream=stream,
    )


# ----------------------------------------------------------------------------- host: compile
def _fake_tensors(spec: PrefillRopeSpecialization):
    make_fake = cute.runtime.make_fake_compact_tensor
    make_fake_strided = cute.runtime.make_fake_tensor
    num_tokens = cute.sym_int64()
    num_requests = cute.sym_int64()
    # q_in_place: fused_q / quant_q_buffer are one-row placeholders, so their row
    # extent is its own symbol (the TVM-FFI entry checks shared symbols for equality)
    q_rows = cute.sym_int64() if spec.q_in_place else num_tokens
    fused_q = make_fake(
        cutlass.BFloat16,
        (q_rows, NUM_HEADS, HEAD_SIZE),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    q_pe = make_fake_strided(
        cutlass.Float8E4M3FN if spec.q_e4m3 else cutlass.BFloat16,
        (num_tokens, NUM_HEADS, ROPE_DIM),
        (spec.q_pe_stride, spec.q_pe_ld, 1),
        assumed_align=16,
    )
    if spec.latent_cache_stride == HEAD_SIZE:
        latent = make_fake(
            cutlass.BFloat16,
            (num_tokens, HEAD_SIZE),
            stride_order=(1, 0),
            assumed_align=16,
        )
    else:
        latent = make_fake_strided(
            cutlass.BFloat16,
            (num_tokens, HEAD_SIZE),
            (spec.latent_cache_stride, 1),
            assumed_align=16,
        )
    quant_q = make_fake(
        cutlass.Uint8,
        (q_rows, NUM_HEADS, HEAD_SIZE),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    storage = cute.runtime.make_ptr(cutlass.Uint8, 0, cutlass.AddressSpace.gmem, assumed_align=16)
    offsets = make_fake(
        cutlass.Int32,
        (num_requests, 2, cute.sym_int64()),
        stride_order=(2, 1, 0),
        assumed_align=4,
    )
    cu_seqlens = make_fake(cutlass.Int32, (cute.sym_int64(),), assumed_align=4)
    cached = make_fake(cutlass.Int32, (num_requests,), assumed_align=4)
    rope = make_fake(
        cutlass.Float32,
        (cute.sym_int64(), ROPE_DIM, 2),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    return fused_q, q_pe, latent, quant_q, storage, offsets, cu_seqlens, cached, rope


def _compile_for(spec: PrefillRopeSpecialization, device_index: int):
    spec.validate()
    key = (
        "core",
        device_index,
        spec.tokens_per_block,
        spec.q_pe_ld,
        spec.q_pe_stride,
        spec.latent_cache_stride,
        spec.skip_q_base,
        spec.q_in_place,
        spec.q_e4m3,
        spec.merge_rope,
        spec.skip_kv_copy,
        spec.rope_par,
    )
    cached = _COMPILE_CACHE.get(key)
    if cached is not None:
        return cached
    cutlass.cuda.initialize_cuda_context(device_index)
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False)
    if spec.rope_par:
        compiled = cute.compile(
            _launch_par,
            *_fake_tensors(spec),
            fake_stream,
            spec.tokens_per_block,
            spec.q_pe_ld,
            spec.q_pe_stride,
            spec.latent_cache_stride,
            spec.rope_par,
            options="--enable-tvm-ffi",
        )
        _COMPILE_CACHE[key] = compiled
        return compiled
    compiled = cute.compile(
        _launch,
        *_fake_tensors(spec),
        fake_stream,
        spec.tokens_per_block,
        spec.q_pe_ld,
        spec.q_pe_stride,
        spec.latent_cache_stride,
        spec.skip_q_base,
        spec.q_in_place,
        spec.q_e4m3,
        spec.merge_rope,
        spec.skip_kv_copy,
        options="--enable-tvm-ffi",
    )
    _COMPILE_CACHE[key] = compiled
    return compiled


def compile(*, tokens_per_block: int = TOKENS_PER_BLOCK, device_index: int = 0):
    """Compile the default GLM specialization (q_pe view of a [T,16,256] q, packed latent)."""
    return _compile_for(PrefillRopeSpecialization(tokens_per_block=tokens_per_block), device_index)


# ----------------------------------------------------------------------------- host: contract + runner
def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _normalize_rotary_cos_sin(rotary_cos_sin: torch.Tensor) -> torch.Tensor:
    _require(rotary_cos_sin.dtype == torch.float32, "rotary_cos_sin must be float32")
    _require(
        rotary_cos_sin.is_cuda and rotary_cos_sin.is_contiguous(),
        "rotary_cos_sin must be contiguous cuda",
    )
    _require(
        rotary_cos_sin.numel() % (ROPE_DIM * 2) == 0 and rotary_cos_sin.numel() > 0,
        "rotary_cos_sin rows",
    )
    return rotary_cos_sin.reshape(-1, ROPE_DIM, 2)


_Q_IN_PLACE_PLACEHOLDERS: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def _q_in_place_placeholders(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """One-row fused_q / quant_q stand-ins for the q_in_place mode (never read or written)."""
    ph = _Q_IN_PLACE_PLACEHOLDERS.get(device)
    if ph is None:
        ph = (
            torch.zeros(1, NUM_HEADS, HEAD_SIZE, dtype=torch.bfloat16, device=device),
            torch.zeros(1, NUM_HEADS, HEAD_SIZE, dtype=torch.uint8, device=device),
        )
        _Q_IN_PLACE_PLACEHOLDERS[device] = ph
    return ph


def prepare_runner(
    fused_q: torch.Tensor | None,
    q_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    quant_q_buffer: torch.Tensor | None,
    kv_cache_storage: torch.Tensor | int,
    kv_cache_block_offsets: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cached_lens: torch.Tensor,
    rotary_cos_sin: torch.Tensor,
    *,
    tokens_per_block: int = TOKENS_PER_BLOCK,
    refresh_stream: bool = False,
    skip_q_base: bool = False,
    q_in_place: bool = False,
    q_e4m3: bool = False,
    merge_rope: bool | None = None,
    skip_kv_copy: bool = False,
    rope_par: int = 0,
) -> PreparedPrefillRopeRunner:
    """Validate the ABI, pick the specialization, compile (cached), bind pointers.

    ``fused_q`` BF16 [T,16,576] contiguous (RoPE slot written); ``q_pe`` BF16 [T,16,64]
    view with strides (q_pe_stride, q_pe_ld, 1) (read-only); ``latent_cache`` BF16
    [T,576] (row stride >= 576; k_pe slot rewritten rotated); ``quant_q_buffer`` uint8
    [T,16,576]; ``kv_cache_storage`` = layer base pointer (int) or the pool tensor;
    ``kv_cache_block_offsets`` int32 [B,2,max_blocks] contiguous (K row used);
    ``cu_seqlens`` int32 [B+1]; ``cached_lens`` int32 [B]; ``rotary_cos_sin`` float32
    [positions,64,2]. ``skip_q_base=True``: the caller has already written
    ``quant_q_buffer[..., :512]`` (e.g. a K-B GEMM with an e4m3 epilogue); only the
    RoPE columns ``[..., 512:]`` of ``quant_q_buffer`` are written and
    ``fused_q[..., :512]`` is not read.
    ``q_in_place=True`` (non-absorbed context route): rope(q_pe) is written back into
    ``q_pe`` and no E4M3 Q is produced; ``fused_q`` / ``quant_q_buffer`` must be None.
    ``q_e4m3=True`` (with ``q_in_place``, FP8 context MHA): ``q_pe`` is an E4M3 view
    (``torch.float8_e4m3fn``) of the q_b GEMM's E4M3 output; rope runs on the widened values
    and the result is written back as E4M3.
    ``merge_rope``: one CTA per token tile for all RoPE groups (None = ``MERGE_ROPE_GROUPS``
    for the q_in_place modes, off otherwise); outputs are bitwise identical either way.
    ``skip_kv_copy=True`` (context-step stitch): the c_kv copy groups are not launched -- the caller's
    norm already wrote the context rows' c_kv E4M3 bytes into the pool; everything else as
    without it.
    """
    _require(not q_e4m3 or q_in_place, "q_e4m3 needs q_in_place")
    rotary_cos_sin = _normalize_rotary_cos_sin(rotary_cos_sin)
    if q_in_place:
        _require(
            fused_q is None and quant_q_buffer is None and not skip_q_base,
            "q_in_place takes no fused_q / quant_q_buffer and no skip_q_base",
        )
        fused_q, quant_q_buffer = _q_in_place_placeholders(q_pe.device)
        t = q_pe.shape[0]
    else:
        t = fused_q.shape[0]
    _require(
        fused_q.dtype == torch.bfloat16 and fused_q.is_cuda and fused_q.is_contiguous(),
        "fused_q",
    )
    _require(
        q_in_place or tuple(fused_q.shape) == (t, NUM_HEADS, HEAD_SIZE),
        "fused_q must be [T,16,576]",
    )
    _require(
        q_pe.dtype == (torch.float8_e4m3fn if q_e4m3 else torch.bfloat16)
        and q_pe.is_cuda
        and tuple(q_pe.shape) == (t, NUM_HEADS, ROPE_DIM),
        "q_pe",
    )
    _require(q_pe.stride(2) == 1, "q_pe last dim must be contiguous")
    _require(latent_cache.dtype == torch.bfloat16 and latent_cache.is_cuda, "latent_cache")
    _require(
        tuple(latent_cache.shape) == (t, HEAD_SIZE) and latent_cache.stride(1) == 1,
        "latent_cache [T,576]",
    )
    _require(
        quant_q_buffer.dtype == torch.uint8
        and quant_q_buffer.is_cuda
        and quant_q_buffer.is_contiguous(),
        "quant_q",
    )
    _require(
        q_in_place or tuple(quant_q_buffer.shape) == (t, NUM_HEADS, HEAD_SIZE),
        "quant_q_buffer must be [T,16,576]",
    )
    _require(
        kv_cache_block_offsets.dtype == torch.int32 and kv_cache_block_offsets.is_cuda,
        "block_offsets",
    )
    _require(
        kv_cache_block_offsets.dim() == 3 and kv_cache_block_offsets.shape[1] == 2,
        "block_offsets [B,2,max_blocks]",
    )
    _require(kv_cache_block_offsets.is_contiguous(), "block_offsets must be contiguous")
    b = kv_cache_block_offsets.shape[0]
    _require(0 < b <= MAX_REQUESTS, f"num_requests must be in [1,{MAX_REQUESTS}]")
    _require(
        cu_seqlens.dtype == torch.int32 and cu_seqlens.is_cuda and cu_seqlens.numel() >= b + 1,
        "cu_seqlens",
    )
    _require(
        cached_lens.dtype == torch.int32
        and cached_lens.is_cuda
        and tuple(cached_lens.shape) == (b,),
        "cached_lens",
    )
    if isinstance(kv_cache_storage, torch.Tensor):
        _require(
            kv_cache_storage.dtype == torch.uint8 and kv_cache_storage.is_cuda,
            "kv_cache_storage tensor",
        )
        pointer = int(kv_cache_storage.data_ptr())
    else:
        pointer = int(kv_cache_storage)
    _require(pointer > 0 and pointer % 16 == 0, "kv_cache_storage pointer alignment")
    # The kernel loads q_pe / latent_cache / rotary_cos_sin and stores fused_q with 16-byte
    # vector accesses and quant_q_buffer with 8-byte packed stores: an unaligned base pointer
    # (e.g. an odd-offset view) must fail here, not as a misaligned-address fault in the kernel.
    for tensor, name, align in (
        (fused_q, "fused_q", 16),
        (q_pe, "q_pe", 16),
        (latent_cache, "latent_cache", 16),
        (quant_q_buffer, "quant_q_buffer", 8),
        (rotary_cos_sin, "rotary_cos_sin", 16),
    ):
        _require(tensor.data_ptr() % align == 0, f"{name} must be {align}-byte aligned")
    # One compiled launcher per device: every operand must live on fused_q's device.
    device = fused_q.device
    for tensor, name in (
        (q_pe, "q_pe"),
        (latent_cache, "latent_cache"),
        (quant_q_buffer, "quant_q_buffer"),
        (kv_cache_block_offsets, "kv_cache_block_offsets"),
        (cu_seqlens, "cu_seqlens"),
        (cached_lens, "cached_lens"),
        (rotary_cos_sin, "rotary_cos_sin"),
    ):
        _require(tensor.device == device, f"{name} must be on {device}")
    if isinstance(kv_cache_storage, torch.Tensor):
        _require(kv_cache_storage.device == device, f"kv_cache_storage must be on {device}")
    spec = PrefillRopeSpecialization(
        tokens_per_block=tokens_per_block,
        q_pe_ld=int(q_pe.stride(1)),
        q_pe_stride=int(q_pe.stride(0)),
        latent_cache_stride=int(latent_cache.stride(0)),
        skip_q_base=bool(skip_q_base),
        q_in_place=bool(q_in_place),
        q_e4m3=bool(q_e4m3),
        merge_rope=(
            MERGE_ROPE_GROUPS and bool(q_in_place) if merge_rope is None else bool(merge_rope)
        ),
        skip_kv_copy=bool(skip_kv_copy),
        rope_par=int(rope_par),
    )
    device_index = _device_index(fused_q)
    compiled = _compile_for(spec, device_index)
    stream = None if refresh_stream else _current_stream(device_index)
    return PreparedPrefillRopeRunner(
        compiled=compiled,
        args=(
            fused_q,
            q_pe,
            latent_cache,
            quant_q_buffer,
            pointer,
            kv_cache_block_offsets,
            cu_seqlens,
            cached_lens,
            rotary_cos_sin,
        ),
        device_index=device_index,
        stream=stream,
        specialization=spec,
        owners=(kv_cache_storage,) if isinstance(kv_cache_storage, torch.Tensor) else (),
    )


# ----------------------------------------------------------------------------- CTX-HOST-DIET: steady state
#: TRTLLM_GLM52_CTX_HOST_DIET=1: the call signatures `prepare_runner` has passed. A signature fixes every
#: input `prepare_runner` checks or keys its specialization on except the row count, the request count
#: and the pointers: the options, every dtype, stride (row-count independent here) and device, the
#: tensors left None, and the rope table's shape. `run_accepted` launches a call whose signature is
#: here with exactly the arguments `prepare_runner` would bind, after the checks the signature leaves
#: open. No torch tensor is held. Bounded: a full table only misses (no eviction).
_ACCEPTED: set[tuple] = set()
ACCEPTED_CAP = 64
#: Calls `run_accepted` sent back to `prepare_runner` (signature not accepted yet).
accepted_misses = 0


def call_signature(
    fused_q, q_pe, latent_cache, quant_q_buffer, kv_cache_block_offsets, cu_seqlens, cached_lens,
    rotary_cos_sin, options: tuple,
) -> tuple:  # fmt: skip
    """`options`: (tokens_per_block, skip_q_base, q_in_place, q_e4m3, merge_rope, skip_kv_copy,
    rope_par), as `prepare_runner` takes them."""
    return (
        options,
        None if fused_q is None else (fused_q.dtype, fused_q.stride(), fused_q.device),
        q_pe.dtype,
        q_pe.stride(),
        q_pe.device,
        latent_cache.dtype,
        latent_cache.stride(),
        latent_cache.device,
        None
        if quant_q_buffer is None
        else (quant_q_buffer.dtype, quant_q_buffer.stride(), quant_q_buffer.device),
        kv_cache_block_offsets.dtype,
        kv_cache_block_offsets.dim(),
        kv_cache_block_offsets.device,
        cu_seqlens.dtype,
        cu_seqlens.device,
        cached_lens.dtype,
        cached_lens.dim(),
        cached_lens.device,
        rotary_cos_sin.dtype,
        rotary_cos_sin.shape,
        rotary_cos_sin.stride(),
        rotary_cos_sin.device,
    )


def run_accepted(
    compiled, fused_q, q_pe, latent_cache, quant_q_buffer, kv_cache_storage: int,
    kv_cache_block_offsets, cu_seqlens, cached_lens, rotary_cos_sin, options: tuple,
) -> tuple | None:  # fmt: skip
    """Launch `compiled` as `prepare_runner(..., **options)` and its runner would, when the call's
    signature was accepted (returns None); else launch nothing and return the signature, for
    `accept` once `prepare_runner` has passed the call. Raises on a call `prepare_runner` would
    reject for its row count, request count or pointers."""
    global accepted_misses
    sig = call_signature(
        fused_q, q_pe, latent_cache, quant_q_buffer, kv_cache_block_offsets, cu_seqlens,
        cached_lens, rotary_cos_sin, options,
    )  # fmt: skip
    if sig not in _ACCEPTED:
        accepted_misses += 1
        return sig
    t = q_pe.shape[0]
    b = kv_cache_block_offsets.shape[0]
    if options[2]:  # q_in_place
        fused_q, quant_q_buffer = _q_in_place_placeholders(q_pe.device)
    else:
        _require(
            fused_q.shape == (t, NUM_HEADS, HEAD_SIZE)
            and quant_q_buffer.shape == (t, NUM_HEADS, HEAD_SIZE),
            "fused_q / quant_q_buffer must be [T,16,576]",
        )
    _require(
        q_pe.shape == (t, NUM_HEADS, ROPE_DIM) and latent_cache.shape == (t, HEAD_SIZE),
        "q_pe must be [T,16,64] and latent_cache [T,576]",
    )
    _require(
        kv_cache_block_offsets.shape[1] == 2 and kv_cache_block_offsets.is_contiguous(),
        "block_offsets must be a contiguous [B,2,max_blocks]",
    )
    _require(0 < b <= MAX_REQUESTS, "num_requests must be in [1,MAX_REQUESTS]")
    _require(cu_seqlens.numel() >= b + 1 and cached_lens.shape == (b,), "cu_seqlens / cached_lens")
    pointer = int(kv_cache_storage)
    _require(pointer > 0 and pointer % 16 == 0, "kv_cache_storage pointer alignment")
    _require(
        not (
            (fused_q.data_ptr() | q_pe.data_ptr() | latent_cache.data_ptr()) & 15
            or quant_q_buffer.data_ptr() & 7
            or rotary_cos_sin.data_ptr() & 15
        ),
        "fused_q / q_pe / latent_cache / rotary_cos_sin 16-byte, quant_q_buffer 8-byte aligned",
    )
    compiled(
        fused_q, q_pe, latent_cache, quant_q_buffer, pointer, kv_cache_block_offsets, cu_seqlens,
        cached_lens, rotary_cos_sin.reshape(-1, ROPE_DIM, 2), _current_stream(_device_index(fused_q)),
    )  # fmt: skip
    return None


def accept(signature: tuple) -> None:
    """Record a signature `prepare_runner` has passed (see `run_accepted`)."""
    if len(_ACCEPTED) < ACCEPTED_CAP:
        _ACCEPTED.add(signature)
