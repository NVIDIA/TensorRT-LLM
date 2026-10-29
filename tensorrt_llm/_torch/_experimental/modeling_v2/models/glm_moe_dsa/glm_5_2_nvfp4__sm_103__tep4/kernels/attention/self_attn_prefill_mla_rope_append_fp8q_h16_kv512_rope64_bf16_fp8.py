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
from cutlass.experimental import primitives as prims

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
    return cuda.CUstream(torch.cuda.current_stream(device_index).cuda_stream)


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
    """Request index b with cu_seqlens[b] <= token < cu_seqlens[b+1] (unrolled scan)."""
    req = cutlass.Int32(0)
    for i in cutlass.range_constexpr(MAX_REQUESTS - 1):
        boundary = cutlass.Int32(i + 1)
        if boundary < num_requests:
            if token >= cutlass.Int32(cu_seqlens.load(idx=boundary)):
                req = req + cutlass.Int32(1)
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
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    token_tile, work_group, _ = cute.arch.block_idx()
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
            if work_group < cutlass.Int32(Q_ROPE_GROUPS):
                q_head = work_group
                src = token * q_pe_stride + q_head * q_pe_ld + dim_base
                q_vec = q_pe.data_ptr(src).load(count=8, alignment=16)
                y0, y1, y2, y3, y4, y5, y6, y7 = _rope8_e4m3(q_vec, cos_sin)
                dst = token * NUM_HEADS * HEAD_SIZE + q_head * HEAD_SIZE + KV_LORA_RANK + dim_base
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
) -> None:
    num_tokens = fused_q.shape[0]
    num_requests = cached_lens.shape[0]
    max_blocks_per_seq = kv_cache_block_offsets.shape[2]
    storage = cute.make_tensor(kv_cache_storage, cute.make_layout((1,), stride=(1,)))
    grid_x = (num_tokens + TOKENS_PER_CTA - 1) // TOKENS_PER_CTA
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
    ).launch(grid=(grid_x, WORK_GROUPS, 1), block=(THREADS, 1, 1), stream=stream)


# ----------------------------------------------------------------------------- host: compile
def _fake_tensors(spec: PrefillRopeSpecialization):
    make_fake = cute.runtime.make_fake_compact_tensor
    make_fake_strided = cute.runtime.make_fake_tensor
    num_tokens = cute.sym_int64()
    num_requests = cute.sym_int64()
    fused_q = make_fake(
        cutlass.BFloat16,
        (num_tokens, NUM_HEADS, HEAD_SIZE),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    q_pe = make_fake_strided(
        cutlass.BFloat16,
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
        (num_tokens, NUM_HEADS, HEAD_SIZE),
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
    )
    cached = _COMPILE_CACHE.get(key)
    if cached is not None:
        return cached
    cutlass.cuda.initialize_cuda_context(device_index)
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False)
    compiled = cute.compile(
        _launch,
        *_fake_tensors(spec),
        fake_stream,
        spec.tokens_per_block,
        spec.q_pe_ld,
        spec.q_pe_stride,
        spec.latent_cache_stride,
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


def prepare_runner(
    fused_q: torch.Tensor,
    q_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    quant_q_buffer: torch.Tensor,
    kv_cache_storage: torch.Tensor | int,
    kv_cache_block_offsets: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cached_lens: torch.Tensor,
    rotary_cos_sin: torch.Tensor,
    *,
    tokens_per_block: int = TOKENS_PER_BLOCK,
    refresh_stream: bool = False,
) -> PreparedPrefillRopeRunner:
    """Validate the ABI, pick the specialization, compile (cached), bind pointers.

    ``fused_q`` BF16 [T,16,576] contiguous (RoPE slot written); ``q_pe`` BF16 [T,16,64]
    view with strides (q_pe_stride, q_pe_ld, 1) (read-only); ``latent_cache`` BF16
    [T,576] (row stride >= 576; k_pe slot rewritten rotated); ``quant_q_buffer`` uint8
    [T,16,576]; ``kv_cache_storage`` = layer base pointer (int) or the pool tensor;
    ``kv_cache_block_offsets`` int32 [B,2,max_blocks] contiguous (K row used);
    ``cu_seqlens`` int32 [B+1]; ``cached_lens`` int32 [B]; ``rotary_cos_sin`` float32
    [positions,64,2].
    """
    rotary_cos_sin = _normalize_rotary_cos_sin(rotary_cos_sin)
    t = fused_q.shape[0]
    _require(
        fused_q.dtype == torch.bfloat16 and fused_q.is_cuda and fused_q.is_contiguous(),
        "fused_q",
    )
    _require(tuple(fused_q.shape) == (t, NUM_HEADS, HEAD_SIZE), "fused_q must be [T,16,576]")
    _require(
        q_pe.dtype == torch.bfloat16
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
        tuple(quant_q_buffer.shape) == (t, NUM_HEADS, HEAD_SIZE),
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
