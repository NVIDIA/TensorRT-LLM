# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL kernel for GLM MLA generation RoPE/QKV assignment.

This is a replacement candidate for ``torch.ops.trtllm.mla_rope_generation``
with BF16 Q/latent inputs, FP8 Q/KV outputs, ``kv_lora_rank=512``, and
``rope_dim=64``. It applies GPT-J RoPE to Q/K, quantizes Q, writes paged KV,
initializes FMHA scales/counters, and builds Q/KV sequence-prefix metadata.

The device implementation uses ``cutlass.experimental.primitives`` and typed
pointer loads/stores; it does not use CuTe copy, GEMM, tiled-MMA, pipeline,
NVGPU helper, or private-IR APIs. The serving adapter resolves pool pointers and
zero-copy views before the timed launch.

Tensor pointers, token/batch counts, sequence lengths, block-table contents,
and the stream remain runtime values. Dtypes, head dimensions, generation-only
RoPE append, MTP width, page geometry, Q-PE strides, and full/tail token tiles
are compile-time specializations. The device kernel supports only primary-pool,
non-Helix, no-op FP8 KV scaling; the host wrapper validates that contract.

Sequence-prefix metadata is produced by one warp after the Q/KV body releases
PDL dependents. The target TRTLLM-Gen FMHA reads ``cache_seq_lens`` rather than
these prefix arrays; any other dependent consumer must wait for this grid before
reading them.

Host options (both default to the packed, q-base-copying kernel; the caller
chooses them):

* ``split_source``: ``latent_cache`` may be the strided ``[T, 576]``
  ``kv_norm|k_pe`` view of the kv_a projection output (row stride >= 576,
  divisible by 8) instead of the packed latent buffer. Same kernel; the option
  changes only the latent row pitch, a compile-time specialization
  (``HEAD_SIZE`` for the packed buffer, the projection row stride for the
  view). Numerics identical: the same elements are loaded and the same math
  runs, only the address computation differs. Off: the latent must be
  contiguous (fail-closed).
* ``emit_q_base``: with ``False`` the kernel does not copy the FP8 q-base into
  ``quant_q_buffer[..., :512]`` and the grid shrinks from (x,153,1) to
  (x,25,1). Same kernel; the option removes the q-base work groups, every other
  output is byte-identical, and the q-base bytes are then produced by the K-B
  GEMM (``emit_fp8_q_base=True``, same FP32->BF16->E4M3 double round). ``True``
  keeps the copy.
"""

import math
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
HEAD_SIZE = KV_LORA_RANK + ROPE_DIM
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + ROPE_DIM
THREADS = 256
TOKENS_PER_CTA = 32
COPY_TOKENS_PER_CTA = 4
DIM_GROUP = 64
Q_BASE_GROUPS_PER_HEAD = KV_LORA_RANK // DIM_GROUP
# The deployed GLM engine uses 32-token KV pages.  Keep this public alias for
# callers, while carrying the value in the specialization config below so a
# different engine page size gets a separately compiled kernel.
TOKENS_PER_BLOCK = 32
FP8_KV_CACHE_QUANT_MODE = 1 << 7
DEFAULT_PREDICTED_TOKENS_PER_SEQ = 2
DEFAULT_MAX_BLOCKS_PER_SEQ = 513
DEFAULT_Q_PE_LD = 256
DEFAULT_Q_PE_STRIDE = NUM_HEADS * DEFAULT_Q_PE_LD
Q_SCALING = 1.0
LOG2_E = 1.4426950408889634
WARP_SIZE = 32
FULL_WARP_MASK = 0xFFFFFFFF
SHFL_UP_MASK_AND_CLAMP = 0x00
SHFL_IDX_MASK_AND_CLAMP = 0x1F

_COMPILE_CACHE: dict[tuple[object, ...], object] = {}


@dataclass(frozen=True)
class MlaRopeRuntimeShape:
    """Launch-varying dimensions read from tensor shapes by the compiled host."""

    num_tokens: int
    num_generations: int

    def validate(self, predicted_tokens_per_seq: int) -> None:
        if self.num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {self.num_tokens}")
        if self.num_generations <= 0:
            raise ValueError(f"num_generations must be positive, got {self.num_generations}")
        expected_tokens = self.num_generations * predicted_tokens_per_seq
        if expected_tokens != self.num_tokens:
            raise ValueError(
                "num_tokens must equal num_generations * "
                "predicted_tokens_per_seq; "
                f"got {self.num_tokens} vs {expected_tokens}"
            )


@dataclass(frozen=True)
class MlaRopeSpecialization:
    """Values compiled as constexpr because they change device code/layout."""

    predicted_tokens_per_seq: int = DEFAULT_PREDICTED_TOKENS_PER_SEQ
    max_blocks_per_seq: int = DEFAULT_MAX_BLOCKS_PER_SEQ
    tokens_per_block: int = TOKENS_PER_BLOCK
    q_pe_ld: int = DEFAULT_Q_PE_LD
    q_pe_stride: int = DEFAULT_Q_PE_STRIDE
    # split_source: row pitch of the latent source; HEAD_SIZE = packed.
    latent_cache_stride: int = HEAD_SIZE
    # False drops the q-base copy groups (the K-B GEMM emits the FP8 q-base).
    emit_q_base: bool = True
    full_token_tiles: bool = True

    def validate(self) -> None:
        if self.predicted_tokens_per_seq <= 0:
            raise ValueError(
                f"predicted_tokens_per_seq must be positive, got {self.predicted_tokens_per_seq}"
            )
        if self.max_blocks_per_seq <= 0:
            raise ValueError(f"max_blocks_per_seq must be positive, got {self.max_blocks_per_seq}")
        if self.tokens_per_block <= 0 or (self.tokens_per_block & (self.tokens_per_block - 1)):
            raise ValueError(
                f"tokens_per_block must be a positive power of two, got {self.tokens_per_block}"
            )
        if self.q_pe_ld < ROPE_DIM or self.q_pe_stride < self.q_pe_ld * NUM_HEADS:
            raise ValueError("q_pe strides do not cover [num_heads, rope_dim]")
        if self.latent_cache_stride < HEAD_SIZE:
            raise ValueError("latent_cache_stride must cover the logical row")
        if self.latent_cache_stride % 8 != 0:
            raise ValueError("latent_cache_stride must be divisible by 8")


def _device_index(tensor: torch.Tensor) -> int:
    device_index = tensor.device.index
    return device_index if device_index is not None else torch.cuda.current_device()


def _current_stream(device_index: int) -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream(device_index).cuda_stream)


@dataclass
class PreparedMlaRopeRunner:
    """Callable launcher with all device pointers and views prebound."""

    compiled: object
    args: tuple[object, ...]
    device_index: int
    stream: cuda.CUstream | None
    runtime_shape: MlaRopeRuntimeShape
    specialization: MlaRopeSpecialization
    owners: tuple[object, ...] = ()

    def __call__(self) -> None:
        stream = self.stream
        if stream is None:
            stream = _current_stream(self.device_index)
        self.compiled(*self.args, stream)


@dataclass(frozen=True)
class PreparedPagedKvCache:
    """Layer-local primary-pool arguments derived from the serving ABI."""

    layer_pool_ptr: int
    block_offsets: torch.Tensor
    sequence_length: torch.Tensor


def _bf16_rn(value):
    """Round fp32 to bf16 precision without relying on contraction behavior."""
    bits = value.bitcast(cutlass.Int32)
    bits = bits + (((bits >> 16) & 1) + 0x7FFF)
    bits = (bits >> 16) << 16
    return bits.bitcast(cutlass.Float32)


@cute.jit
def _rope_even_bf16_rn(x_even, x_odd, cos, sin) -> cutlass.Float32:
    return _bf16_rn(x_even * cos - x_odd * sin)


@cute.jit
def _rope_odd_bf16_rn(x_even, x_odd, cos, sin) -> cutlass.Float32:
    return _bf16_rn(x_odd * cos + x_even * sin)


@cute.jit
def _pack_e4m3x2_word(value0, value1):
    """Pack two adjacent E4M3 bytes into one little-endian Uint16 word."""
    return prims.cvt_f32x2_to_f8x2(
        cutlass.Float32(value1),
        cutlass.Float32(value0),
        cutlass.Float8E4M3FN,
        rnd=prims.FPRoundingMode.RN,
        sat=prims.SaturationMode.SATFINITE,
    )


@cute.jit
def _store_e4m3x8(dst, idx, value0, value1, value2, value3, value4, value5, value6, value7) -> None:
    """Store eight adjacent FP8 bytes through one aligned vector store."""
    packed0 = _pack_e4m3x2_word(value0, value1)
    packed1 = _pack_e4m3x2_word(value2, value3)
    packed2 = _pack_e4m3x2_word(value4, value5)
    packed3 = _pack_e4m3x2_word(value6, value7)
    packed_vec = cutlass.Vector.from_elements(
        (packed0, packed1, packed2, packed3),
        dtype=cutlass.Int16,
    )
    ptr = dst.data_ptr(idx)
    ptr_i16 = cutlass.inttoptr(ptr.toint(), ptr.space, cutlass.Int16)
    ptr_i16.store(packed_vec, alignment=8)


@cute.jit
def _copy_q_base_quant_group(
    fused_q,
    quant_q_buffer,
    token,
    q_head,
    copy_dim_group,
) -> None:
    q_dim = copy_dim_group * cutlass.Int32(8)
    src_idx = token * NUM_HEADS * HEAD_SIZE + q_head * HEAD_SIZE + q_dim
    _store_e4m3x8(
        quant_q_buffer,
        src_idx,
        fused_q.load(idx=src_idx),
        fused_q.load(idx=src_idx + 1),
        fused_q.load(idx=src_idx + 2),
        fused_q.load(idx=src_idx + 3),
        fused_q.load(idx=src_idx + 4),
        fused_q.load(idx=src_idx + 5),
        fused_q.load(idx=src_idx + 6),
        fused_q.load(idx=src_idx + 7),
    )


@cute.jit
def _store_q_rope_quant_head(
    q_pe,
    quant_q_buffer,
    token,
    q_head,
    dim_base,
    q_pe_ld,
    q_pe_stride,
    cos0,
    sin0,
    cos1,
    sin1,
    cos2,
    sin2,
    cos3,
    sin3,
    write_quant,
) -> None:
    q_base = token * q_pe_stride + q_head * q_pe_ld + dim_base
    q_vec = q_pe.data_ptr(q_base).load(count=8, alignment=16)
    q_x0 = cutlass.Float32(q_vec[0])
    q_x1 = cutlass.Float32(q_vec[1])
    q_x2 = cutlass.Float32(q_vec[2])
    q_x3 = cutlass.Float32(q_vec[3])
    q_x4 = cutlass.Float32(q_vec[4])
    q_x5 = cutlass.Float32(q_vec[5])
    q_x6 = cutlass.Float32(q_vec[6])
    q_x7 = cutlass.Float32(q_vec[7])
    q_y0 = _rope_even_bf16_rn(q_x0, q_x1, cos0, sin0)
    q_y1 = _rope_odd_bf16_rn(q_x0, q_x1, cos0, sin0)
    q_y2 = _rope_even_bf16_rn(q_x2, q_x3, cos1, sin1)
    q_y3 = _rope_odd_bf16_rn(q_x2, q_x3, cos1, sin1)
    q_y4 = _rope_even_bf16_rn(q_x4, q_x5, cos2, sin2)
    q_y5 = _rope_odd_bf16_rn(q_x4, q_x5, cos2, sin2)
    q_y6 = _rope_even_bf16_rn(q_x6, q_x7, cos3, sin3)
    q_y7 = _rope_odd_bf16_rn(q_x6, q_x7, cos3, sin3)
    dst_idx = token * NUM_HEADS * HEAD_SIZE + q_head * HEAD_SIZE + KV_LORA_RANK + dim_base
    if cutlass.const_expr(write_quant):
        _store_e4m3x8(
            quant_q_buffer,
            dst_idx,
            q_y0,
            q_y1,
            q_y2,
            q_y3,
            q_y4,
            q_y5,
            q_y6,
            q_y7,
        )


@cute.jit
def _warp_inclusive_scan_i32(value, lane):
    """Return the inclusive sum of ``value`` across the current warp."""
    for shift in cutlass.range_constexpr(5):
        delta = 1 << shift
        lower = prims.shfl_sync(
            FULL_WARP_MASK,
            value,
            delta,
            SHFL_UP_MASK_AND_CLAMP,
            prims.Shfl.UP,
        )
        valid = cute.math.min(lane >> cutlass.Int32(shift), cutlass.Int32(1))
        value = value + lower * valid
    return value


@cute.jit
def _write_sequence_metadata(
    sequence_length: cutlass.Array,
    cu_q_seqlens: cutlass.Array,
    cu_kv_seqlens: cutlass.Array,
    num_generations: cutlass.Int64,
    tidx: cutlass.Int32,
    predicted_tokens_per_seq: cutlass.Constexpr[int],
) -> None:
    """Build Q/KV prefix arrays with one warp and a carried 32-lane scan."""
    lane = tidx & cutlass.Int32(WARP_SIZE - 1)
    q_step = cutlass.Int32(predicted_tokens_per_seq * NUM_HEADS)
    batch_size = cutlass.Int32(num_generations)
    batch_offset = cutlass.Int32(0)
    carry = cutlass.Int32(0)

    while batch_offset < batch_size:
        batch = batch_offset + lane
        seq_len = cutlass.Int32(0)
        if batch < batch_size:
            seq_len = cutlass.Int32(sequence_length.load(idx=batch))

        inclusive = _warp_inclusive_scan_i32(seq_len, lane)
        if batch < batch_size:
            cu_q_seqlens.store(q_step * batch, idx=batch, alignment=4)
            cu_kv_seqlens.store(carry + inclusive - seq_len, idx=batch, alignment=4)

        chunk_sum = prims.shfl_sync(
            FULL_WARP_MASK,
            inclusive,
            WARP_SIZE - 1,
            SHFL_IDX_MASK_AND_CLAMP,
            prims.Shfl.IDX,
        )
        carry = carry + chunk_sum
        batch_offset = batch_offset + cutlass.Int32(WARP_SIZE)

    if lane == 0:
        cu_q_seqlens.store(q_step * batch_size, idx=batch_size, alignment=4)
        cu_kv_seqlens.store(carry, idx=batch_size, alignment=4)


@cute.kernel
def _mla_rope_generation_kernel(
    fused_q: cutlass.Array,
    q_pe: cutlass.Array,
    latent_cache: cutlass.Array,
    quant_q_buffer: cutlass.Array,
    kv_cache_storage: cutlass.Array,
    kv_cache_block_offsets: cutlass.Array,
    sequence_length: cutlass.Array,
    cu_q_seqlens: cutlass.Array,
    cu_kv_seqlens: cutlass.Array,
    fmha_scheduler_counter: cutlass.Array,
    mla_bmm1_scale: cutlass.Array,
    mla_bmm2_scale: cutlass.Array,
    rotary_cos_sin: cutlass.Array,
    num_tokens: cutlass.Int64,
    num_generations: cutlass.Int64,
    predicted_tokens_per_seq: cutlass.Constexpr[int],
    max_blocks_per_seq: cutlass.Constexpr[int],
    tokens_per_block: cutlass.Constexpr[int],
    q_pe_ld: cutlass.Constexpr[int],
    q_pe_stride: cutlass.Constexpr[int],
    latent_cache_stride: cutlass.Constexpr[int],
    host_bmm1_scale: cutlass.Constexpr[float],
    write_quant: cutlass.Constexpr[bool],
    write_kv_cache: cutlass.Constexpr[bool],
    pdl_wait: cutlass.Constexpr[bool],
    full_token_tiles: cutlass.Constexpr[bool],
    emit_q_base: cutlass.Constexpr[bool],
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    token_tile, work_group, _ = cute.arch.block_idx()

    if token_tile == 0 and work_group == 0 and tidx == 0:
        fmha_scheduler_counter.store(cutlass.Uint32(0), idx=0, alignment=4)
        mla_bmm1_scale.store(cutlass.Float32(host_bmm1_scale), idx=0, alignment=4)
        mla_bmm1_scale.store(cutlass.Float32(host_bmm1_scale * LOG2_E), idx=1, alignment=4)
        mla_bmm2_scale.store(cutlass.Float32(1.0), idx=0, alignment=4)

    token_base = token_tile * cutlass.Int32(TOKENS_PER_CTA)

    local_token = tidx // cutlass.Int32(8)
    lane = tidx - local_token * cutlass.Int32(8)
    token = token_base + local_token
    token_valid = token < cutlass.Int32(num_tokens)
    if cutlass.const_expr(full_token_tiles):
        token_valid = cutlass.Boolean(True)
    copy_group_token_span = cutlass.Int32(
        ((num_tokens + TOKENS_PER_CTA - 1) // TOKENS_PER_CTA) * COPY_TOKENS_PER_CTA
    )
    q_rope_groups: cutlass.Constexpr[int] = NUM_HEADS
    k_rope_group: cutlass.Constexpr[int] = q_rope_groups
    kv_copy_start: cutlass.Constexpr[int] = k_rope_group + 1
    q_base_start: cutlass.Constexpr[int] = kv_copy_start + Q_BASE_GROUPS_PER_HEAD

    # Scheduler/scaling initialization and integer launch setup do not consume
    # the K-B GEMM output, so keep them ahead of the PDL wait. Every predecessor-owned
    # q/latent load is below this point.
    if cutlass.const_expr(pdl_wait):
        prims.griddepcontrol(prims.GridDepAction.WAIT)

    # emit_q_base=False: the q-base groups are not launched (grid_y 25)
    # and this branch is dropped at compile time.
    if cutlass.const_expr(emit_q_base):
        is_q_base_group = work_group >= cutlass.Int32(q_base_start) and write_quant
    else:
        is_q_base_group = False
    if is_q_base_group:
        q_group = work_group - cutlass.Int32(q_base_start)
        q_head = q_group & cutlass.Int32(NUM_HEADS - 1)
        q_token_group = q_group >> cutlass.Int32(4)
        copy_dim_group = tidx & cutlass.Int32(63)
        copy_token_lane = tidx >> cutlass.Int32(6)
        token = (
            token_tile * cutlass.Int32(COPY_TOKENS_PER_CTA)
            + q_token_group * cutlass.Int32(copy_group_token_span)
            + copy_token_lane
        )
        if token < cutlass.Int32(num_tokens):
            _copy_q_base_quant_group(
                fused_q,
                quant_q_buffer,
                token,
                q_head,
                copy_dim_group,
            )

    elif work_group < cutlass.Int32(q_rope_groups) and token_valid:
        q_head = work_group
        if cutlass.const_expr(predicted_tokens_per_seq == 2):
            generation = token >> cutlass.Int32(1)
            pred = token & cutlass.Int32(1)
        else:
            generation = token // cutlass.Int32(predicted_tokens_per_seq)
            token_group_base = generation * cutlass.Int32(predicted_tokens_per_seq)
            pred = token - token_group_base
        raw_position = cutlass.Int32(sequence_length.load(idx=generation))
        raw_position = raw_position - cutlass.Int32(predicted_tokens_per_seq)
        raw_position = raw_position + pred
        position = raw_position
        if position < cutlass.Int32(0):
            position = cutlass.Int32(0)

        dim_base = lane * cutlass.Int32(8)
        table_base = (position * ROPE_DIM + lane * cutlass.Int32(4)) * 2
        cos_sin = rotary_cos_sin.data_ptr(table_base).load(count=8, alignment=16)
        cos0 = cutlass.Float32(cos_sin[0])
        sin0 = cutlass.Float32(cos_sin[1])
        cos1 = cutlass.Float32(cos_sin[2])
        sin1 = cutlass.Float32(cos_sin[3])
        cos2 = cutlass.Float32(cos_sin[4])
        sin2 = cutlass.Float32(cos_sin[5])
        cos3 = cutlass.Float32(cos_sin[6])
        sin3 = cutlass.Float32(cos_sin[7])
        _store_q_rope_quant_head(
            q_pe,
            quant_q_buffer,
            token,
            q_head,
            dim_base,
            q_pe_ld,
            q_pe_stride,
            cos0,
            sin0,
            cos1,
            sin1,
            cos2,
            sin2,
            cos3,
            sin3,
            write_quant,
        )
    elif work_group == cutlass.Int32(k_rope_group) and token_valid:
        if cutlass.const_expr(predicted_tokens_per_seq == 2):
            generation = token >> cutlass.Int32(1)
            pred = token & cutlass.Int32(1)
        else:
            generation = token // cutlass.Int32(predicted_tokens_per_seq)
            token_group_base = generation * cutlass.Int32(predicted_tokens_per_seq)
            pred = token - token_group_base
        raw_position = cutlass.Int32(sequence_length.load(idx=generation))
        raw_position = raw_position - cutlass.Int32(predicted_tokens_per_seq)
        raw_position = raw_position + pred
        position = raw_position
        if position < cutlass.Int32(0):
            position = cutlass.Int32(0)

        if cutlass.const_expr(write_kv_cache):
            cache_pos = raw_position
            block_in_seq = cache_pos // cutlass.Int32(tokens_per_block)
            if cache_pos < cutlass.Int32(0):
                block_in_seq = cutlass.Int32(0)
            local_cache_token = cache_pos & cutlass.Int32(tokens_per_block - 1)
            if block_in_seq < cutlass.Int32(max_blocks_per_seq):
                offset_idx = generation * 2 * max_blocks_per_seq + block_in_seq
                block_offset = cutlass.Int32(kv_cache_block_offsets.load(idx=offset_idx))
                # TRT-LLM encodes the physical block and local-layer stride in
                # an int32 table entry.  Promote before converting that entry
                # to a byte offset: production KV pools are much larger than
                # the 2 GiB range of signed int32 address arithmetic.
                block_base = cutlass.Int64(block_offset) * cutlass.Int64(
                    tokens_per_block * HEAD_SIZE
                )
                cache_base = block_base + cutlass.Int64(local_cache_token * HEAD_SIZE)
                dim_base = lane * cutlass.Int32(8)
                k_base = token * latent_cache_stride + KV_LORA_RANK + dim_base
                k_vec = latent_cache.data_ptr(k_base).load(count=8, alignment=16)
                table_base = (position * ROPE_DIM + lane * cutlass.Int32(4)) * 2
                cos_sin = rotary_cos_sin.data_ptr(table_base).load(count=8, alignment=16)
                k_x0 = cutlass.Float32(k_vec[0])
                k_x1 = cutlass.Float32(k_vec[1])
                k_x2 = cutlass.Float32(k_vec[2])
                k_x3 = cutlass.Float32(k_vec[3])
                k_x4 = cutlass.Float32(k_vec[4])
                k_x5 = cutlass.Float32(k_vec[5])
                k_x6 = cutlass.Float32(k_vec[6])
                k_x7 = cutlass.Float32(k_vec[7])
                k_y0 = _rope_even_bf16_rn(
                    k_x0,
                    k_x1,
                    cutlass.Float32(cos_sin[0]),
                    cutlass.Float32(cos_sin[1]),
                )
                k_y1 = _rope_odd_bf16_rn(
                    k_x0,
                    k_x1,
                    cutlass.Float32(cos_sin[0]),
                    cutlass.Float32(cos_sin[1]),
                )
                k_y2 = _rope_even_bf16_rn(
                    k_x2,
                    k_x3,
                    cutlass.Float32(cos_sin[2]),
                    cutlass.Float32(cos_sin[3]),
                )
                k_y3 = _rope_odd_bf16_rn(
                    k_x2,
                    k_x3,
                    cutlass.Float32(cos_sin[2]),
                    cutlass.Float32(cos_sin[3]),
                )
                k_y4 = _rope_even_bf16_rn(
                    k_x4,
                    k_x5,
                    cutlass.Float32(cos_sin[4]),
                    cutlass.Float32(cos_sin[5]),
                )
                k_y5 = _rope_odd_bf16_rn(
                    k_x4,
                    k_x5,
                    cutlass.Float32(cos_sin[4]),
                    cutlass.Float32(cos_sin[5]),
                )
                k_y6 = _rope_even_bf16_rn(
                    k_x6,
                    k_x7,
                    cutlass.Float32(cos_sin[6]),
                    cutlass.Float32(cos_sin[7]),
                )
                k_y7 = _rope_odd_bf16_rn(
                    k_x6,
                    k_x7,
                    cutlass.Float32(cos_sin[6]),
                    cutlass.Float32(cos_sin[7]),
                )
                cache_idx = cache_base + KV_LORA_RANK + dim_base
                _store_e4m3x8(
                    kv_cache_storage,
                    cache_idx,
                    k_y0,
                    k_y1,
                    k_y2,
                    k_y3,
                    k_y4,
                    k_y5,
                    k_y6,
                    k_y7,
                )

    elif work_group >= cutlass.Int32(kv_copy_start) and work_group < cutlass.Int32(q_base_start):
        kv_token_group = work_group - cutlass.Int32(kv_copy_start)
        copy_dim_group = tidx & cutlass.Int32(63)
        copy_token_lane = tidx >> cutlass.Int32(6)
        token = (
            token_tile * cutlass.Int32(COPY_TOKENS_PER_CTA)
            + kv_token_group * cutlass.Int32(copy_group_token_span)
            + copy_token_lane
        )
        if token < cutlass.Int32(num_tokens):
            if cutlass.const_expr(predicted_tokens_per_seq == 2):
                generation = token >> cutlass.Int32(1)
                pred = token & cutlass.Int32(1)
            else:
                generation = token // cutlass.Int32(predicted_tokens_per_seq)
                token_group_base = generation * cutlass.Int32(predicted_tokens_per_seq)
                pred = token - token_group_base
            raw_position = cutlass.Int32(sequence_length.load(idx=generation))
            raw_position = raw_position - cutlass.Int32(predicted_tokens_per_seq)
            raw_position = raw_position + pred
            if cutlass.const_expr(write_kv_cache):
                cache_pos = raw_position
                block_in_seq = cache_pos // cutlass.Int32(tokens_per_block)
                if cache_pos < cutlass.Int32(0):
                    block_in_seq = cutlass.Int32(0)
                local_cache_token = cache_pos & cutlass.Int32(tokens_per_block - 1)
                if block_in_seq < cutlass.Int32(max_blocks_per_seq):
                    offset_idx = generation * 2 * max_blocks_per_seq + block_in_seq
                    block_offset = cutlass.Int32(kv_cache_block_offsets.load(idx=offset_idx))
                    # Keep the byte offset in 64 bits for multi-GiB KV pools.
                    block_base = cutlass.Int64(block_offset) * cutlass.Int64(
                        tokens_per_block * HEAD_SIZE
                    )
                    cache_base = block_base + cutlass.Int64(local_cache_token * HEAD_SIZE)
                    dim = copy_dim_group * cutlass.Int32(8)
                    latent_base = token * latent_cache_stride + dim
                    _store_e4m3x8(
                        kv_cache_storage,
                        cache_base + dim,
                        latent_cache.load(idx=latent_base),
                        latent_cache.load(idx=latent_base + 1),
                        latent_cache.load(idx=latent_base + 2),
                        latent_cache.load(idx=latent_base + 3),
                        latent_cache.load(idx=latent_base + 4),
                        latent_cache.load(idx=latent_base + 5),
                        latent_cache.load(idx=latent_base + 6),
                        latent_cache.load(idx=latent_base + 7),
                    )

    # The sparse MLA attention kernel consumes both the query/cache writes
    # above and these prefix arrays. Publish all of them before releasing the
    # request-index conversion kernel, which then provides the transitive
    # predecessor edge into the attention kernel.
    if token_tile == 0 and work_group == 0 and tidx < cutlass.Int32(WARP_SIZE):
        _write_sequence_metadata(
            sequence_length,
            cu_q_seqlens,
            cu_kv_seqlens,
            num_generations,
            tidx,
            predicted_tokens_per_seq,
        )

    prims.barrier_cta_sync(0)
    if cutlass.const_expr(pdl_wait):
        if tidx == 0:
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)


@cute.jit
def _launch_mla_rope_generation(
    fused_q: cute.Tensor,
    q_pe: cute.Tensor,
    latent_cache: cute.Tensor,
    quant_q_buffer: cute.Tensor,
    kv_cache_storage: cute.Pointer,
    kv_cache_block_offsets: cute.Tensor,
    sequence_length: cute.Tensor,
    cu_q_seqlens: cute.Tensor,
    cu_kv_seqlens: cute.Tensor,
    fmha_scheduler_counter: cute.Tensor,
    mla_bmm1_scale: cute.Tensor,
    mla_bmm2_scale: cute.Tensor,
    rotary_cos_sin: cute.Tensor,
    stream: cuda.CUstream,
    predicted_tokens_per_seq: cutlass.Constexpr[int],
    max_blocks_per_seq: cutlass.Constexpr[int],
    tokens_per_block: cutlass.Constexpr[int],
    q_pe_ld: cutlass.Constexpr[int],
    q_pe_stride: cutlass.Constexpr[int],
    latent_cache_stride: cutlass.Constexpr[int],
    host_bmm1_scale: cutlass.Constexpr[float],
    write_quant: cutlass.Constexpr[bool],
    write_kv_cache: cutlass.Constexpr[bool],
    pdl_wait: cutlass.Constexpr[bool],
    full_token_tiles: cutlass.Constexpr[bool],
    emit_q_base: cutlass.Constexpr[bool],
) -> None:
    num_tokens = fused_q.shape[0]
    num_generations = sequence_length.shape[0]
    kv_cache_storage_tensor = cute.make_tensor(
        kv_cache_storage, cute.make_layout((1,), stride=(1,))
    )
    grid_x = (num_tokens + TOKENS_PER_CTA - 1) // TOKENS_PER_CTA
    q_rope_groups: cutlass.Constexpr[int] = NUM_HEADS
    q_base_start: cutlass.Constexpr[int] = q_rope_groups + 1 + Q_BASE_GROUPS_PER_HEAD
    if cutlass.const_expr(emit_q_base):
        grid_y: cutlass.Constexpr[int] = q_base_start + NUM_HEADS * Q_BASE_GROUPS_PER_HEAD
    else:
        grid_y: cutlass.Constexpr[int] = q_base_start  # 25 non-q-base groups.
    _mla_rope_generation_kernel(
        fused_q,
        q_pe,
        latent_cache,
        quant_q_buffer,
        kv_cache_storage_tensor,
        kv_cache_block_offsets,
        sequence_length,
        cu_q_seqlens,
        cu_kv_seqlens,
        fmha_scheduler_counter,
        mla_bmm1_scale,
        mla_bmm2_scale,
        rotary_cos_sin,
        num_tokens,
        num_generations,
        predicted_tokens_per_seq,
        max_blocks_per_seq,
        tokens_per_block,
        q_pe_ld,
        q_pe_stride,
        latent_cache_stride,
        host_bmm1_scale,
        write_quant,
        write_kv_cache,
        pdl_wait,
        full_token_tiles,
        emit_q_base,
    ).launch(
        grid=(grid_x, grid_y, 1),
        block=(THREADS, 1, 1),
        stream=stream,
        use_pdl=pdl_wait,
    )


def _fake_tensors(specialization: MlaRopeSpecialization):
    make_fake = cute.runtime.make_fake_compact_tensor
    make_fake_strided = cute.runtime.make_fake_tensor
    token_divisibility = TOKENS_PER_CTA if specialization.full_token_tiles else 1
    num_tokens = cute.sym_int64(divisibility=token_divisibility)
    num_generations = cute.sym_int64()
    fused_q = make_fake(
        cutlass.BFloat16,
        (num_tokens, NUM_HEADS, HEAD_SIZE),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    q_pe = make_fake_strided(
        cutlass.BFloat16,
        (num_tokens, NUM_HEADS, ROPE_DIM),
        (specialization.q_pe_stride, specialization.q_pe_ld, 1),
        assumed_align=16,
    )
    if specialization.latent_cache_stride == HEAD_SIZE:
        latent_cache = make_fake(
            cutlass.BFloat16,
            (num_tokens, HEAD_SIZE),
            stride_order=(1, 0),
            assumed_align=16,
        )
    else:
        # split_source: the strided kv_norm|k_pe view of the kv_a projection output.
        latent_cache = make_fake_strided(
            cutlass.BFloat16,
            (num_tokens, HEAD_SIZE),
            (specialization.latent_cache_stride, 1),
            assumed_align=16,
        )
    quant_q = make_fake(
        cutlass.Uint8,
        (num_tokens, NUM_HEADS, HEAD_SIZE),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    kv_cache = cute.runtime.make_ptr(
        cutlass.Uint8,
        0,
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    offsets = make_fake_strided(
        cutlass.Int32,
        (num_generations, 2, specialization.max_blocks_per_seq),
        (
            2 * specialization.max_blocks_per_seq,
            specialization.max_blocks_per_seq,
            1,
        ),
        assumed_align=4,
    )
    seq = make_fake(cutlass.Int32, (num_generations,), assumed_align=4)
    cu_q = make_fake(cutlass.Int32, (cute.sym_int64(),), assumed_align=16)
    cu_kv = make_fake(cutlass.Int32, (cute.sym_int64(),), assumed_align=16)
    counter = make_fake(cutlass.Uint32, (1,), assumed_align=4)
    bmm1 = make_fake(cutlass.Float32, (2,), assumed_align=8)
    bmm2 = make_fake(cutlass.Float32, (1,), assumed_align=4)
    rope = make_fake(
        cutlass.Float32,
        (cute.sym_int64(), ROPE_DIM, 2),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    return (
        fused_q,
        q_pe,
        latent_cache,
        quant_q,
        kv_cache,
        offsets,
        seq,
        cu_q,
        cu_kv,
        counter,
        bmm1,
        bmm2,
        rope,
    )


def _compile_options() -> str:
    return "--enable-tvm-ffi"


def _compile_for(specialization: MlaRopeSpecialization, device_index: int):
    specialization.validate()
    key = (
        "core",
        device_index,
        specialization.full_token_tiles,
        specialization.predicted_tokens_per_seq,
        specialization.max_blocks_per_seq,
        specialization.tokens_per_block,
        specialization.q_pe_ld,
        specialization.q_pe_stride,
        specialization.latent_cache_stride,
        specialization.emit_q_base,
    )
    cached = _COMPILE_CACHE.get(key)
    if cached is not None:
        return cached

    cutlass.cuda.initialize_cuda_context(device_index)
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False)
    host_bmm1_scale = 1.0 / (Q_SCALING * math.sqrt(float(QK_HEAD_DIM)))
    compiled = cute.compile(
        _launch_mla_rope_generation,
        *_fake_tensors(specialization),
        fake_stream,
        specialization.predicted_tokens_per_seq,
        specialization.max_blocks_per_seq,
        specialization.tokens_per_block,
        specialization.q_pe_ld,
        specialization.q_pe_stride,
        specialization.latent_cache_stride,
        host_bmm1_scale,
        True,  # FP8 Q is fixed for the deployed GLM module.
        True,  # rope_append is fixed, so KV cache writes are always enabled.
        True,  # PDL consumers must wait before reading producer outputs.
        specialization.full_token_tiles,
        specialization.emit_q_base,
        options=_compile_options(),
    )
    _COMPILE_CACHE[key] = compiled
    return compiled


def _require_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype,
    shape: tuple[int, ...] | None = None,
    device_type: str | None = None,
    contiguous: bool | None = None,
) -> None:
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must be {dtype}, got {tensor.dtype}")
    if shape is not None and tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if device_type is not None and tensor.device.type != device_type:
        raise ValueError(f"{name} must live on {device_type}, got {tensor.device}")
    if contiguous is not None and bool(tensor.is_contiguous()) != contiguous:
        requirement = "contiguous" if contiguous else "non-contiguous"
        raise ValueError(f"{name} must be {requirement}")


def _validate_latent_cache_layout(latent_cache: torch.Tensor) -> None:
    """Validate one zero-copy latent view before choosing its specialization."""

    if latent_cache.ndim != 2:
        raise ValueError(f"latent_cache must be 2D, got {latent_cache.ndim}D")
    if latent_cache.stride(-1) != 1:
        raise ValueError("latent_cache last dimension must have stride 1")
    row_stride = int(latent_cache.stride(0))
    if row_stride < HEAD_SIZE:
        raise ValueError(f"latent_cache row stride must be at least {HEAD_SIZE}, got {row_stride}")
    if row_stride % 8 != 0:
        raise ValueError(f"latent_cache row stride must be divisible by 8, got {row_stride}")
    if latent_cache.data_ptr() % 16 != 0:
        raise ValueError("latent_cache data pointer must be 16-byte aligned")
    if int(latent_cache.storage_offset()) < 0:
        raise ValueError("latent_cache storage offset must be nonnegative")
    rows = int(latent_cache.shape[0])
    if rows <= 0:
        raise ValueError("latent_cache must have at least one row")
    storage_elements = latent_cache.untyped_storage().nbytes() // latent_cache.element_size()
    last_element = int(latent_cache.storage_offset()) + (rows - 1) * row_stride + HEAD_SIZE - 1
    if last_element >= storage_elements:
        raise ValueError(
            "latent_cache layout exceeds backing storage: "
            f"last={last_element}, storage_elements={storage_elements}"
        )


def _require_noop_scale(
    tensor: torch.Tensor | None,
    *,
    name: str,
    device_type: str = "cuda",
) -> None:
    if tensor is None:
        return
    _require_tensor(
        tensor,
        name=name,
        dtype=torch.float32,
        shape=(1,),
        device_type=device_type,
        contiguous=True,
    )
    if not bool(torch.all(tensor == 1.0).item()):
        raise ValueError(f"{name} must be all-ones for this fixed MLA RoPE target")


def _normalize_rotary_cos_sin(rotary_cos_sin: torch.Tensor) -> torch.Tensor:
    """Return a zero-copy [positions, 64, 2] view of TRT-LLM's table storage."""
    if rotary_cos_sin.dtype != torch.float32:
        raise TypeError(f"rotary_cos_sin must be torch.float32, got {rotary_cos_sin.dtype}")
    if rotary_cos_sin.device.type != "cuda":
        raise ValueError(f"rotary_cos_sin must live on cuda, got {rotary_cos_sin.device}")
    if not rotary_cos_sin.is_contiguous():
        raise ValueError("rotary_cos_sin must be contiguous")
    row_elements = ROPE_DIM * 2
    if rotary_cos_sin.numel() == 0 or rotary_cos_sin.numel() % row_elements != 0:
        raise ValueError(
            "rotary_cos_sin storage must contain a positive whole number of "
            f"[{ROPE_DIM},2] rows, got {rotary_cos_sin.numel()} elements"
        )
    return rotary_cos_sin.reshape(-1, ROPE_DIM, 2)


def _cache_pointer(
    kv_cache_storage: torch.Tensor | int,
    *,
    min_cache_bytes: int,
) -> int:
    """Return a validated 16-byte-aligned cache pointer without copying data."""
    if isinstance(kv_cache_storage, torch.Tensor):
        if (
            kv_cache_storage.dtype != torch.uint8
            or kv_cache_storage.device.type != "cuda"
            or not kv_cache_storage.is_contiguous()
        ):
            raise TypeError("kv_cache_storage must be a contiguous cuda uint8 tensor")
        if kv_cache_storage.numel() < min_cache_bytes:
            raise ValueError(
                f"kv_cache_storage has {kv_cache_storage.numel()} bytes, "
                f"expected at least {min_cache_bytes}"
            )
        pointer = int(kv_cache_storage.data_ptr())
    elif isinstance(kv_cache_storage, int):
        pointer = kv_cache_storage
    else:
        raise TypeError("kv_cache_storage must be a cuda uint8 tensor or raw pointer")
    if pointer <= 0 or pointer % 16 != 0:
        raise ValueError(
            f"kv_cache_storage pointer must be nonzero and 16-byte aligned, got {pointer:#x}"
        )
    return pointer


def _adapt_serving_paged_kv_cache(
    *,
    kv_cache_block_offsets: torch.Tensor,
    host_kv_cache_pool_pointers: torch.Tensor,
    host_kv_cache_pool_mapping: torch.Tensor,
    sequence_length: torch.Tensor,
    num_contexts: int,
    num_generations: int,
    layer_idx: int,
    tokens_per_block: int,
    predicted_tokens_per_seq: int,
    block_ids_per_seq: torch.Tensor | None,
) -> PreparedPagedKvCache:
    """Resolve TRT-LLM's serving metadata to the layer-local primary-pool ABI.

    TRT-LLM stores CPU pool metadata separately from the device block table.
    This routine performs the same static pool/layer pointer adjustment as
    ``buildPagedKvCacheBuffers`` and returns zero-copy generation views. It is
    called during runner preparation, never by the replay hot path.
    """
    if (
        host_kv_cache_pool_pointers.dtype != torch.int64
        or host_kv_cache_pool_pointers.device.type != "cpu"
        or host_kv_cache_pool_pointers.dim() != 2
        or host_kv_cache_pool_pointers.shape[1] < 2
        or not host_kv_cache_pool_pointers.is_contiguous()
    ):
        raise ValueError("host_kv_cache_pool_pointers must be contiguous CPU int64 [num_pools,>=2]")
    if (
        host_kv_cache_pool_mapping.dtype != torch.int32
        or host_kv_cache_pool_mapping.device.type != "cpu"
        or host_kv_cache_pool_mapping.dim() != 2
        or host_kv_cache_pool_mapping.shape[1] < 2
        or not host_kv_cache_pool_mapping.is_contiguous()
    ):
        raise ValueError("host_kv_cache_pool_mapping must be contiguous CPU int32 [num_layers,>=2]")
    if layer_idx < 0 or layer_idx >= host_kv_cache_pool_mapping.shape[0]:
        raise ValueError(
            f"layer_idx {layer_idx} is outside pool mapping with "
            f"{host_kv_cache_pool_mapping.shape[0]} layers"
        )

    pool_index = int(host_kv_cache_pool_mapping[layer_idx, 0].item())
    layer_idx_in_pool = int(host_kv_cache_pool_mapping[layer_idx, 1].item())
    if pool_index < 0 or pool_index >= host_kv_cache_pool_pointers.shape[0]:
        raise ValueError(f"mapped pool_index {pool_index} is out of bounds")
    if layer_idx_in_pool < 0:
        raise ValueError(f"layer_idx_in_pool must be nonnegative, got {layer_idx_in_pool}")

    primary_pool_ptr = int(host_kv_cache_pool_pointers[pool_index, 0].item())
    secondary_pool_ptr = int(host_kv_cache_pool_pointers[pool_index, 1].item())
    if secondary_pool_ptr != 0:
        raise ValueError(
            "secondary KV pool is not supported by the zero-device-overhead "
            "primary-pool specialization"
        )
    bytes_per_block = tokens_per_block * HEAD_SIZE
    layer_pool_ptr = primary_pool_ptr + layer_idx_in_pool * bytes_per_block
    if primary_pool_ptr <= 0 or layer_pool_ptr % 16 != 0:
        raise ValueError(
            "mapped primary KV pool pointer must be nonzero and the layer base "
            f"must be 16-byte aligned, got {layer_pool_ptr:#x}"
        )

    if (
        kv_cache_block_offsets.dtype != torch.int32
        or kv_cache_block_offsets.device.type != "cuda"
        or kv_cache_block_offsets.dim() != 4
        or kv_cache_block_offsets.shape[0] <= pool_index
        or kv_cache_block_offsets.shape[2] != 2
        or not kv_cache_block_offsets.is_contiguous()
    ):
        raise ValueError(
            "kv_cache_block_offsets must be contiguous CUDA int32 [num_pools,num_seqs,2,max_blocks]"
        )
    seq_end = num_contexts + num_generations
    if num_contexts < 0 or seq_end > kv_cache_block_offsets.shape[1]:
        raise ValueError(
            f"generation range [{num_contexts},{seq_end}) exceeds block table "
            f"sequence extent {kv_cache_block_offsets.shape[1]}"
        )
    block_offsets = kv_cache_block_offsets[pool_index, num_contexts:seq_end, :, :]
    if not block_offsets.is_contiguous():
        raise ValueError("selected generation block-offset view must be contiguous")

    if (
        sequence_length.dtype != torch.int32
        or sequence_length.device.type != "cuda"
        or sequence_length.dim() != 1
        or not sequence_length.is_contiguous()
        or seq_end > sequence_length.shape[0]
    ):
        raise ValueError(
            "sequence_length must be contiguous CUDA int32 [num_seqs] and "
            "cover all context and generation sequences"
        )
    generation_sequence_length = sequence_length[num_contexts:seq_end]
    return PreparedPagedKvCache(
        layer_pool_ptr=layer_pool_ptr,
        block_offsets=block_offsets,
        sequence_length=generation_sequence_length,
    )


def _validate_full_public_io_contract(
    *,
    fused_q: torch.Tensor,
    q_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    quant_q_buffer: torch.Tensor,
    kv_cache_storage: torch.Tensor | int,
    kv_cache_block_offsets: torch.Tensor,
    sequence_length: torch.Tensor,
    cu_q_seqlens: torch.Tensor,
    cu_kv_seqlens: torch.Tensor,
    fmha_scheduler_counter: torch.Tensor,
    mla_bmm1_scale: torch.Tensor,
    mla_bmm2_scale: torch.Tensor,
    rotary_cos_sin: torch.Tensor,
    host_past_key_value_lengths: torch.Tensor | None,
    host_context_lengths: torch.Tensor | None,
    num_contexts: int,
    host_kv_cache_pool_pointers: torch.Tensor | None,
    host_kv_cache_pool_mapping: torch.Tensor | None,
    kv_scale_orig_quant: torch.Tensor | None,
    kv_scale_quant_orig: torch.Tensor | None,
    out_scale: torch.Tensor | None,
    block_ids_per_seq: torch.Tensor | None,
    quant_mode: int,
    q_scaling: float,
    predicted_tokens_per_seq: int,
    tokens_per_block: int,
    write_quant: bool,
    split_source: bool,
) -> None:
    """Validate every public-op tensor before launching the subset kernel."""

    if out_scale is not None:
        raise ValueError(
            "output_sf/out_scale must be None because this target writes BF16 "
            "FMHA output; FP8 Q/KV scales are kv_scale_orig_quant and "
            "kv_scale_quant_orig"
        )
    if num_contexts != 0:
        raise ValueError(f"num_contexts must be 0 for generation-only target, got {num_contexts}")
    if q_scaling != Q_SCALING:
        raise ValueError(f"q_scaling must be {Q_SCALING}, got {q_scaling}")
    has_fp8_kv_cache = bool(quant_mode & FP8_KV_CACHE_QUANT_MODE)
    if write_quant != has_fp8_kv_cache:
        raise ValueError(f"write_quant={write_quant} disagrees with quant_mode={quant_mode}")
    if not has_fp8_kv_cache:
        raise ValueError(
            f"quant_mode must contain TRT-LLM's FP8_KV_CACHE bit ({FP8_KV_CACHE_QUANT_MODE})"
        )
    if tokens_per_block <= 0 or tokens_per_block & (tokens_per_block - 1):
        raise ValueError(
            f"tokens_per_block must be a positive power of two, got {tokens_per_block}"
        )

    num_tokens = int(fused_q.shape[0])
    num_generations = int(sequence_length.shape[0])
    max_blocks_per_seq = int(kv_cache_block_offsets.shape[-1])
    expected_tokens = num_generations * predicted_tokens_per_seq
    if num_tokens != expected_tokens:
        raise ValueError(
            "num_tokens must equal num_generations * predicted_tokens_per_seq; "
            f"got {num_tokens} vs {expected_tokens}"
        )

    _require_tensor(
        fused_q,
        name="fused_q",
        dtype=torch.bfloat16,
        shape=(num_tokens, NUM_HEADS, HEAD_SIZE),
        device_type="cuda",
        contiguous=True,
    )
    _require_tensor(
        q_pe,
        name="q_pe",
        dtype=torch.bfloat16,
        shape=(num_tokens, NUM_HEADS, ROPE_DIM),
        device_type="cuda",
    )
    if q_pe.stride(-1) != 1:
        raise ValueError("q_pe last dimension must be contiguous")
    if q_pe.stride(-2) < ROPE_DIM:
        raise ValueError("q_pe head stride must cover rope_dim")
    if q_pe.stride(0) < NUM_HEADS * q_pe.stride(-2):
        raise ValueError("q_pe token stride must cover all local heads")
    _require_tensor(
        latent_cache,
        name="latent_cache",
        dtype=torch.bfloat16,
        shape=(num_tokens, HEAD_SIZE),
        device_type="cuda",
        # split_source: the strided kv_norm|k_pe view of the projection; else packed.
        contiguous=None if split_source else True,
    )
    if split_source:
        _validate_latent_cache_layout(latent_cache)
    _require_tensor(
        quant_q_buffer,
        name="quant_q_buffer",
        dtype=torch.uint8,
        shape=(num_tokens, NUM_HEADS, HEAD_SIZE),
        device_type="cuda",
        contiguous=True,
    )
    _require_tensor(
        kv_cache_block_offsets,
        name="kv_cache_block_offsets[0]",
        dtype=torch.int32,
        shape=(num_generations, 2, max_blocks_per_seq),
        device_type="cuda",
        contiguous=True,
    )
    _require_tensor(
        sequence_length,
        name="sequence_length",
        dtype=torch.int32,
        shape=(num_generations,),
        device_type="cuda",
        contiguous=True,
    )
    _require_tensor(
        cu_q_seqlens,
        name="cu_q_seqlens",
        dtype=torch.int32,
        shape=(num_generations + 1,),
        device_type="cuda",
        contiguous=True,
    )
    _require_tensor(
        cu_kv_seqlens,
        name="cu_kv_seqlens",
        dtype=torch.int32,
        shape=(num_generations + 1,),
        device_type="cuda",
        contiguous=True,
    )
    _require_tensor(
        fmha_scheduler_counter,
        name="fmha_scheduler_counter",
        dtype=torch.uint32,
        shape=(1,),
        device_type="cuda",
        contiguous=True,
    )
    _require_tensor(
        mla_bmm1_scale,
        name="mla_bmm1_scale",
        dtype=torch.float32,
        shape=(2,),
        device_type="cuda",
        contiguous=True,
    )
    _require_tensor(
        mla_bmm2_scale,
        name="mla_bmm2_scale",
        dtype=torch.float32,
        shape=(1,),
        device_type="cuda",
        contiguous=True,
    )
    _require_tensor(
        rotary_cos_sin,
        name="rotary_cos_sin",
        dtype=torch.float32,
        device_type="cuda",
        contiguous=True,
    )
    if rotary_cos_sin.dim() != 3 or tuple(rotary_cos_sin.shape[1:]) != (
        ROPE_DIM,
        2,
    ):
        raise ValueError(
            "normalized rotary_cos_sin must have shape [positions,64,2], got "
            f"{tuple(rotary_cos_sin.shape)}"
        )
    if rotary_cos_sin.shape[0] <= 0:
        raise ValueError("rotary_cos_sin must contain at least one position")
    min_cache_bytes = num_generations * max_blocks_per_seq * tokens_per_block * HEAD_SIZE
    cache_pointer = _cache_pointer(
        kv_cache_storage,
        min_cache_bytes=min_cache_bytes,
    )

    if host_past_key_value_lengths is not None:
        _require_tensor(
            host_past_key_value_lengths,
            name="host_past_key_value_lengths",
            dtype=torch.int32,
            shape=(num_generations,),
            device_type="cpu",
            contiguous=True,
        )
    if host_context_lengths is not None:
        _require_tensor(
            host_context_lengths,
            name="host_context_lengths",
            dtype=torch.int32,
            shape=(num_generations,),
            device_type="cpu",
            contiguous=True,
        )
    if host_kv_cache_pool_pointers is not None:
        _require_tensor(
            host_kv_cache_pool_pointers,
            name="host_kv_cache_pool_pointers",
            dtype=torch.int64,
            shape=(1, 2),
            device_type="cpu",
            contiguous=True,
        )
        expected_pool_pointer = cache_pointer
        actual_pool_pointer = int(host_kv_cache_pool_pointers[0, 0].item())
        if actual_pool_pointer != expected_pool_pointer:
            raise ValueError("host_kv_cache_pool_pointers[0,0] must point at kv_cache_storage")
    if host_kv_cache_pool_mapping is not None:
        _require_tensor(
            host_kv_cache_pool_mapping,
            name="host_kv_cache_pool_mapping",
            dtype=torch.int32,
            shape=(1, 2),
            device_type="cpu",
            contiguous=True,
        )
    if block_ids_per_seq is not None:
        _require_tensor(
            block_ids_per_seq,
            name="block_ids_per_seq",
            dtype=torch.int32,
            shape=(num_generations, max_blocks_per_seq),
            device_type="cuda",
            contiguous=True,
        )
        if not torch.equal(block_ids_per_seq, kv_cache_block_offsets[:, 0, :]):
            raise ValueError("block_ids_per_seq must match K block offsets")
    _require_noop_scale(kv_scale_orig_quant, name="kv_scale_orig_quant")
    _require_noop_scale(kv_scale_quant_orig, name="kv_scale_quant_orig")


def _make_runtime_and_specialization(
    *,
    fused_q: torch.Tensor,
    q_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    kv_cache_block_offsets: torch.Tensor,
    sequence_length: torch.Tensor,
    predicted_tokens_per_seq: int,
    tokens_per_block: int,
    quant_mode: int,
    split_source: bool,
    emit_q_base: bool,
) -> tuple[MlaRopeRuntimeShape, MlaRopeSpecialization]:
    if not quant_mode & FP8_KV_CACHE_QUANT_MODE:
        raise ValueError("the deployed specialization requires FP8 Q/KV")
    runtime_shape = MlaRopeRuntimeShape(
        num_tokens=int(fused_q.shape[0]),
        num_generations=int(sequence_length.shape[0]),
    )
    specialization = MlaRopeSpecialization(
        predicted_tokens_per_seq=predicted_tokens_per_seq,
        max_blocks_per_seq=int(kv_cache_block_offsets.shape[-1]),
        tokens_per_block=tokens_per_block,
        q_pe_ld=int(q_pe.stride(-2)),
        q_pe_stride=int(q_pe.stride(0)),
        latent_cache_stride=(int(latent_cache.stride(0)) if split_source else HEAD_SIZE),
        emit_q_base=emit_q_base,
        full_token_tiles=runtime_shape.num_tokens % TOKENS_PER_CTA == 0,
    )
    runtime_shape.validate(specialization.predicted_tokens_per_seq)
    specialization.validate()
    return runtime_shape, specialization


def prepare_runner(
    fused_q: torch.Tensor,
    q_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    quant_q_buffer: torch.Tensor,
    kv_cache_storage: torch.Tensor | int,
    kv_cache_block_offsets: torch.Tensor,
    sequence_length: torch.Tensor,
    cu_q_seqlens: torch.Tensor,
    cu_kv_seqlens: torch.Tensor,
    fmha_scheduler_counter: torch.Tensor,
    mla_bmm1_scale: torch.Tensor,
    mla_bmm2_scale: torch.Tensor,
    rotary_cos_sin: torch.Tensor,
    *,
    host_past_key_value_lengths: torch.Tensor | None = None,
    host_context_lengths: torch.Tensor | None = None,
    num_contexts: int = 0,
    host_kv_cache_pool_pointers: torch.Tensor | None = None,
    host_kv_cache_pool_mapping: torch.Tensor | None = None,
    kv_scale_orig_quant: torch.Tensor | None = None,
    kv_scale_quant_orig: torch.Tensor | None = None,
    out_scale: torch.Tensor | None = None,
    block_ids_per_seq: torch.Tensor | None = None,
    predicted_tokens_per_seq: int = DEFAULT_PREDICTED_TOKENS_PER_SEQ,
    tokens_per_block: int = TOKENS_PER_BLOCK,
    quant_mode: int = 128,
    q_scaling: float = Q_SCALING,
    refresh_stream: bool = False,
    split_source: bool = False,
    emit_q_base: bool = True,
) -> PreparedMlaRopeRunner:
    """Prepare the zero-adaptation-overhead hot launch path.

    Validation, specialization selection, compilation lookup, pointer/view
    binding, and (by default) stream lookup happen once here.  Replays call
    only the compiled FFI entry.  Set ``refresh_stream=True`` only when a
    prepared runner must follow later changes to PyTorch's current stream.

    ``split_source`` admits the strided kv_a-view latent instead of the
    packed buffer; ``emit_q_base=False`` drops the q-base copy groups.
    See the module docstring.
    """
    rotary_cos_sin = _normalize_rotary_cos_sin(rotary_cos_sin)
    _validate_full_public_io_contract(
        fused_q=fused_q,
        q_pe=q_pe,
        latent_cache=latent_cache,
        quant_q_buffer=quant_q_buffer,
        kv_cache_storage=kv_cache_storage,
        kv_cache_block_offsets=kv_cache_block_offsets,
        sequence_length=sequence_length,
        cu_q_seqlens=cu_q_seqlens,
        cu_kv_seqlens=cu_kv_seqlens,
        fmha_scheduler_counter=fmha_scheduler_counter,
        mla_bmm1_scale=mla_bmm1_scale,
        mla_bmm2_scale=mla_bmm2_scale,
        rotary_cos_sin=rotary_cos_sin,
        host_past_key_value_lengths=host_past_key_value_lengths,
        host_context_lengths=host_context_lengths,
        num_contexts=num_contexts,
        host_kv_cache_pool_pointers=host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping=host_kv_cache_pool_mapping,
        kv_scale_orig_quant=kv_scale_orig_quant,
        kv_scale_quant_orig=kv_scale_quant_orig,
        out_scale=out_scale,
        block_ids_per_seq=block_ids_per_seq,
        quant_mode=quant_mode,
        q_scaling=q_scaling,
        predicted_tokens_per_seq=predicted_tokens_per_seq,
        tokens_per_block=tokens_per_block,
        write_quant=bool(quant_mode & FP8_KV_CACHE_QUANT_MODE),
        split_source=split_source,
    )
    runtime_shape, specialization = _make_runtime_and_specialization(
        fused_q=fused_q,
        q_pe=q_pe,
        latent_cache=latent_cache,
        kv_cache_block_offsets=kv_cache_block_offsets,
        sequence_length=sequence_length,
        predicted_tokens_per_seq=predicted_tokens_per_seq,
        tokens_per_block=tokens_per_block,
        quant_mode=quant_mode,
        split_source=split_source,
        emit_q_base=emit_q_base,
    )
    device_index = _device_index(fused_q)
    compiled = _compile_for(specialization, device_index)
    stream = None if refresh_stream else _current_stream(device_index)
    cache_pointer = _cache_pointer(
        kv_cache_storage,
        min_cache_bytes=(
            runtime_shape.num_generations
            * specialization.max_blocks_per_seq
            * specialization.tokens_per_block
            * HEAD_SIZE
        ),
    )
    return PreparedMlaRopeRunner(
        compiled=compiled,
        args=(
            fused_q,
            q_pe,
            latent_cache,
            quant_q_buffer,
            cache_pointer,
            kv_cache_block_offsets,
            sequence_length,
            cu_q_seqlens,
            cu_kv_seqlens,
            fmha_scheduler_counter,
            mla_bmm1_scale,
            mla_bmm2_scale,
            rotary_cos_sin,
        ),
        device_index=device_index,
        stream=stream,
        runtime_shape=runtime_shape,
        specialization=specialization,
        owners=(kv_cache_storage,) if isinstance(kv_cache_storage, torch.Tensor) else (),
    )


def prepare_runner_from_serving_abi(
    fused_q: torch.Tensor,
    q_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    rotary_cos_sin: torch.Tensor | None,
    cu_q_seqlens: torch.Tensor,
    cu_kv_seqlens: torch.Tensor,
    fmha_scheduler_counter: torch.Tensor,
    mla_bmm1_scale: torch.Tensor | None,
    mla_bmm2_scale: torch.Tensor | None,
    quant_q_buffer: torch.Tensor | None,
    sequence_length: torch.Tensor,
    host_past_key_value_lengths: torch.Tensor,
    host_context_lengths: torch.Tensor,
    num_contexts: int,
    kv_cache_block_offsets: torch.Tensor | None,
    host_kv_cache_pool_pointers: torch.Tensor | None,
    host_kv_cache_pool_mapping: torch.Tensor | None,
    kv_scale_orig_quant: torch.Tensor | None,
    kv_scale_quant_orig: torch.Tensor | None,
    out_scale: torch.Tensor | None,
    block_ids_per_seq: torch.Tensor | None,
    helix_tensor_params: list[torch.Tensor | None],
    predicted_tokens_per_seq: int,
    layer_idx: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    tokens_per_block: int,
    attention_window_size: int,
    beam_width: int,
    quant_mode: int,
    q_scaling: float,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    rope_append: bool,
    *,
    refresh_stream: bool = False,
    split_source: bool = False,
    emit_q_base: bool = True,
) -> PreparedMlaRopeRunner:
    """Prepare the fixed GLM kernel directly from TRT-LLM's serving ABI.

    Pool mapping, layer-base adjustment, and generation slicing are host-only
    preparation work. The returned runner binds the adjusted raw pointer and
    zero-copy views, so eager hot launches and CUDA Graph replay do not repeat
    this adaptation.
    """
    required_optional_tensors = {
        "rotary_cos_sin": rotary_cos_sin,
        "mla_bmm1_scale": mla_bmm1_scale,
        "mla_bmm2_scale": mla_bmm2_scale,
        "quant_q_buffer": quant_q_buffer,
        "kv_cache_block_offsets": kv_cache_block_offsets,
        "host_kv_cache_pool_pointers": host_kv_cache_pool_pointers,
        "host_kv_cache_pool_mapping": host_kv_cache_pool_mapping,
    }
    missing = [name for name, tensor in required_optional_tensors.items() if tensor is None]
    if missing:
        raise ValueError(f"required serving tensors are None: {', '.join(missing)}")
    if len(helix_tensor_params) != 2 or any(tensor is not None for tensor in helix_tensor_params):
        raise ValueError("the fixed GLM target requires two None Helix tensors")
    fixed_parameters = {
        "num_heads": (num_heads, NUM_HEADS),
        "num_kv_heads": (num_kv_heads, 1),
        "head_size": (head_size, HEAD_SIZE),
        "kv_lora_rank": (kv_lora_rank, KV_LORA_RANK),
        "qk_nope_head_dim": (qk_nope_head_dim, QK_NOPE_HEAD_DIM),
        "qk_rope_head_dim": (qk_rope_head_dim, ROPE_DIM),
        "v_head_dim": (v_head_dim, KV_LORA_RANK),
    }
    mismatches = [
        f"{name}={actual}, expected {expected}"
        for name, (actual, expected) in fixed_parameters.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError("unsupported serving specialization: " + "; ".join(mismatches))
    if not rope_append:
        raise ValueError("rope_append must be enabled")
    if beam_width != 1:
        raise ValueError(f"beam_width must be 1, got {beam_width}")
    if attention_window_size <= 0:
        raise ValueError("attention_window_size must be positive")
    if q_lora_rank < 0:
        raise ValueError(f"q_lora_rank must be nonnegative, got {q_lora_rank}")

    for tensor, name in (
        (host_past_key_value_lengths, "host_past_key_value_lengths"),
        (host_context_lengths, "host_context_lengths"),
    ):
        if (
            tensor.dtype != torch.int32
            or tensor.device.type != "cpu"
            or tensor.dim() != 1
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous CPU int32 [num_seqs]")
    if host_past_key_value_lengths.shape != host_context_lengths.shape:
        raise ValueError("host length tensors must have the same shape")
    num_seqs = int(host_context_lengths.shape[0])
    if num_contexts < 0 or num_contexts >= num_seqs:
        raise ValueError(
            f"num_contexts must leave at least one generation; got "
            f"{num_contexts} of {num_seqs} sequences"
        )
    num_generations = num_seqs - num_contexts
    generation_cu_q_seqlens = cu_q_seqlens[: num_generations + 1]
    generation_cu_kv_seqlens = cu_kv_seqlens[: num_generations + 1]

    paged_kv = _adapt_serving_paged_kv_cache(
        kv_cache_block_offsets=kv_cache_block_offsets,
        host_kv_cache_pool_pointers=host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping=host_kv_cache_pool_mapping,
        sequence_length=sequence_length,
        num_contexts=num_contexts,
        num_generations=num_generations,
        layer_idx=layer_idx,
        tokens_per_block=tokens_per_block,
        predicted_tokens_per_seq=predicted_tokens_per_seq,
        block_ids_per_seq=(
            None
            if block_ids_per_seq is None
            else block_ids_per_seq[num_contexts : num_contexts + num_generations]
        ),
    )
    prepared = prepare_runner(
        fused_q,
        q_pe,
        latent_cache,
        quant_q_buffer,
        paged_kv.layer_pool_ptr,
        paged_kv.block_offsets,
        paged_kv.sequence_length,
        generation_cu_q_seqlens,
        generation_cu_kv_seqlens,
        fmha_scheduler_counter,
        mla_bmm1_scale,
        mla_bmm2_scale,
        rotary_cos_sin,
        kv_scale_orig_quant=kv_scale_orig_quant,
        kv_scale_quant_orig=kv_scale_quant_orig,
        out_scale=out_scale,
        block_ids_per_seq=None,
        predicted_tokens_per_seq=predicted_tokens_per_seq,
        tokens_per_block=tokens_per_block,
        quant_mode=quant_mode,
        q_scaling=q_scaling,
        refresh_stream=refresh_stream,
        split_source=split_source,
        emit_q_base=emit_q_base,
    )
    prepared.owners = (
        kv_cache_block_offsets,
        sequence_length,
        host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping,
        block_ids_per_seq,
    )
    return prepared
