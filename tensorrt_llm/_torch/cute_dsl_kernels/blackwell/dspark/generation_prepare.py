# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DSpark generation-step kernels: target-token acceptance and the next step's input tokens, and
the generation-block preparation (bonus tokens, captured rows, positions, slot state and the
draft block inputs)."""

import functools
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import SmemAllocator

from tensorrt_llm.math_utils import ceil_div

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda


Int32, Int64, Uint8 = cutlass.Int32, cutlass.Int64, cutlass.Uint8
BFloat16, Float32 = cutlass.BFloat16, cutlass.Float32

_INDEX_DTYPES = {torch.int32: Int32, torch.int64: Int64}
_LOGIT_DTYPES = {
    torch.float32: Float32,
    torch.bfloat16: BFloat16,
    torch.float16: cutlass.Float16,
}
_INT64_MIN = -(2**63)


@dsl_user_op
def _f32_bits(value, *, loc=None, ip=None):
    return Int32(llvm.bitcast(T.i32(), Float32(value).ir_value(loc=loc, ip=ip)))


@cute.jit
def _warp_max_i64(value: Int64) -> Int64:
    for step in cutlass.range_constexpr(5):
        high = cute.arch.shuffle_sync_bfly(Int32(value >> 32), offset=1 << step)
        low = cute.arch.shuffle_sync_bfly(Int32(value & Int64(0xFFFFFFFF)), offset=1 << step)
        other = (Int64(high) << 32) | (Int64(low) & Int64(0xFFFFFFFF))
        if other > value:
            value = other
    return value


class DSparkArgmaxPartialKernel:
    """``torch.argmax`` partials of logits rows, one block per 4096 columns of a row.

    Block ``(r, c)`` stores ``(key << 32) | (0xFFFFFFFF - column)`` for the first maximum of
    columns ``[4096 c, 4096 c + 4096)`` of row ``r``, so the maximum over a row's blocks names
    its argmax. Maxima compare as ``torch.argmax`` does (the first NaN wins, ``-0.0 == +0.0``),
    and the int32 key orders them the same way: every NaN above everything else and both zeros
    alike. A thread reads 16-byte vectors and visits its columns in increasing order.
    """

    threads = 128
    columns = 4096

    def __init__(self, vec: int):
        self.vec = vec
        self.items = self.columns // (vec * self.threads)

    @cute.jit
    def __call__(self, logits: cute.Tensor, partial: cute.Tensor, stream: cuda.CUstream):
        self.kernel(logits, partial).launch(
            grid=[partial.shape[0], partial.shape[1], 1],
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, logits: cute.Tensor, partial: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        row, chunk, _ = cute.arch.block_idx()
        vocab = logits.shape[1]
        source = cute.zipped_divide(logits[row, None], (self.vec,))
        values = cute.make_rmem_tensor((self.vec, self.items), logits.element_type)
        first = chunk * (self.columns // self.vec) + tid
        for item in cutlass.range_constexpr(self.items):
            vector = first + item * self.threads
            if vector < vocab // self.vec:
                cute.autovec_copy(source[None, vector], values[None, item])
        best = -Float32.inf
        best_column = Int32(-1)
        for item in cutlass.range_constexpr(self.items):
            for element in cutlass.range_constexpr(self.vec):
                column = (first + item * self.threads) * self.vec + element
                value = Float32(values[element, item])
                if (column < vocab) & (
                    (best_column < 0) | ((best == best) & ((value > best) | (value != value)))
                ):
                    best = value
                    best_column = column
        packed = Int64(_INT64_MIN)
        if best_column >= 0:
            bits = _f32_bits(best)
            magnitude = bits & Int32(0x7FFFFFFF)
            key = bits
            if bits < 0:
                key = bits ^ Int32(0x7FFFFFFF)
            if magnitude == 0:
                key = Int32(0)
            if magnitude > Int32(0x7F800000):
                key = Int32(0x7FFFFFFF)
            packed = (Int64(key) << 32) | (Int64(0xFFFFFFFF) - Int64(best_column))
        packed = _warp_max_i64(packed)
        smem = SmemAllocator()
        warps = smem.allocate_tensor(Int64, cute.make_layout(self.threads // 32), 16)
        if tid % 32 == 0:
            warps[tid // 32] = packed
        cute.arch.barrier()
        if tid == 0:
            for warp in cutlass.range_constexpr(1, self.threads // 32):
                if warps[warp] > packed:
                    packed = warps[warp]
            partial[row, chunk] = packed


class DSparkStrictAcceptKernel:
    """Strict acceptance of one generation request per warp.

    Lane ``j <= draft_len`` takes the request's target token ``j`` (given, or decoded from its
    row's ``DSparkArgmaxPartialKernel`` partials) into ``accepted``; the count is one plus the
    leading draft tokens that match, or the forced count (``force`` 1), or the forced base plus
    one pool draw below ``frac`` at ``(counter * counter_stride + request * slot_stride) & mask``,
    capped at ``draft_len + 1`` (``force`` 2).
    """

    def __init__(
        self,
        draft_len: int,
        chunks: int,
        force: int,
        force_base: int,
        counter_stride: int,
        slot_stride: int,
        pool_mask: int,
    ):
        self.draft_len = draft_len
        self.chunks = chunks
        self.force = force
        self.force_base = force_base
        self.counter_stride = counter_stride
        self.slot_stride = slot_stride
        self.pool_mask = pool_mask

    @cute.jit
    def __call__(
        self,
        target: cute.Tensor,
        draft: cute.Tensor,
        accepted: cute.Tensor,
        num_accepted: cute.Tensor,
        pool: Optional[cute.Tensor],
        counter: Optional[cute.Tensor],
        frac: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(target, draft, accepted, num_accepted, pool, counter, frac).launch(
            grid=[draft.shape[0], 1, 1], block=[cute.arch.WARP_SIZE, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        target: cute.Tensor,
        draft: cute.Tensor,
        accepted: cute.Tensor,
        num_accepted: cute.Tensor,
        pool: Optional[cute.Tensor],
        counter: Optional[cute.Tensor],
        frac: Float32,
    ):
        lane, _, _ = cute.arch.thread_idx()
        g, _, _ = cute.arch.block_idx()
        width = self.draft_len + 1
        token = Int64(0)
        if lane < width:
            row = g * width + lane
            if cutlass.const_expr(self.chunks > 0):
                best = Int64(_INT64_MIN)
                for chunk in cutlass.range_constexpr(self.chunks):
                    value = Int64(target[row, chunk])
                    if value > best:
                        best = value
                token = Int64(0xFFFFFFFF) - (best & Int64(0xFFFFFFFF))
            else:
                token = Int64(target[row])
            accepted[g, lane] = Int32(token)
        first = Int32(self.draft_len)
        if lane < self.draft_len:
            if Int64(draft[g, lane]) != token:
                first = lane
        for step in cutlass.range_constexpr(5):
            other = cute.arch.shuffle_sync_bfly(first, offset=1 << step)
            if other < first:
                first = other
        count = first + 1
        if cutlass.const_expr(self.force == 1):
            count = Int32(self.force_base)
        if cutlass.const_expr(self.force == 2):
            index = (
                Int64(counter[0]) * Int64(self.counter_stride) + Int64(g) * Int64(self.slot_stride)
            ) & Int64(self.pool_mask)
            count = Int32(self.force_base)
            if Float32(pool[index]) < frac:
                count = count + 1
            if count > width:
                count = Int32(width)
        if lane == 0:
            num_accepted[g] = count


class DSparkNextNewTokensKernel:
    """Row ``b`` of the next step's input tokens: ``accepted[batch_indices[b], num_accepted[b] - 1]``
    followed by ``draft[b]`` (None without draft tokens), one warp per row; negative indices wrap
    as in indexing."""

    def __init__(self, draft_len: int):
        self.draft_len = draft_len

    @cute.jit
    def __call__(
        self,
        accepted: cute.Tensor,
        batch_indices: cute.Tensor,
        num_accepted: cute.Tensor,
        draft: Optional[cute.Tensor],
        out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(accepted, batch_indices, num_accepted, draft, out).launch(
            grid=[out.shape[0], 1, 1], block=[cute.arch.WARP_SIZE, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        accepted: cute.Tensor,
        batch_indices: cute.Tensor,
        num_accepted: cute.Tensor,
        draft: Optional[cute.Tensor],
        out: cute.Tensor,
    ):
        lane, _, _ = cute.arch.thread_idx()
        b, _, _ = cute.arch.block_idx()
        if lane == 0:
            row = Int64(batch_indices[b])
            if row < 0:
                row = row + Int64(accepted.shape[0])
            column = Int64(num_accepted[b]) - 1
            if column < 0:
                column = column + Int64(accepted.shape[1])
            out[b, 0] = accepted[row, column]
        if cutlass.const_expr(self.draft_len > 0):
            if lane < self.draft_len:
                out[b, 1 + lane] = draft[b, lane]


def _strided_matrix(dtype, rows, columns):
    return cute.runtime.make_fake_tensor(
        dtype,
        (rows, columns),
        stride=(cute.sym_int64(divisibility=1), cute.sym_int64(divisibility=1)),
        assumed_align=4,
    )


def _compact(dtype, shape):
    return cute.runtime.make_fake_compact_tensor(
        dtype, shape, stride_order=tuple(reversed(range(len(shape))))
    )


@functools.cache
def _compile_argmax_partial(logit_dtype: torch.dtype):
    rows = cute.sym_int()
    vec = 16 // logit_dtype.itemsize
    logits = cute.runtime.make_fake_tensor(
        _LOGIT_DTYPES[logit_dtype],
        (rows, cute.sym_int()),
        stride=(cute.sym_int64(divisibility=vec), 1),
        assumed_align=16,
    )
    return cute.compile(
        DSparkArgmaxPartialKernel(vec),
        logits,
        _compact(Int64, (rows, cute.sym_int())),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )


@functools.cache
def _compile_strict_accept(
    draft_len: int,
    chunks: int,
    force: int,
    force_base: int,
    rng: tuple,
    target_dtype: torch.dtype,
    draft_dtype: torch.dtype,
):
    gens = cute.sym_int()
    target = (
        _compact(Int64, (cute.sym_int(), chunks))
        if chunks
        else _compact(_INDEX_DTYPES[target_dtype], (cute.sym_int(),))
    )
    kernel = DSparkStrictAcceptKernel(draft_len, chunks, force, force_base, *rng)
    return cute.compile(
        kernel,
        target,
        _compact(_INDEX_DTYPES[draft_dtype], (gens, draft_len)),
        _compact(Int32, (gens, draft_len + 1)),
        _compact(Int32, (gens,)),
        _compact(Float32, (cute.sym_int(),)) if force == 2 else None,
        _compact(Int64, (1,)) if force == 2 else None,
        Float32(0.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )


@functools.cache
def _compile_next_new_tokens(draft_len: int, index_dtype: torch.dtype, count_dtype: torch.dtype):
    batch = cute.sym_int()
    return cute.compile(
        DSparkNextNewTokensKernel(draft_len),
        _strided_matrix(Int32, cute.sym_int(), cute.sym_int()),
        _compact(_INDEX_DTYPES[index_dtype], (cute.sym_int(),)),
        _compact(_INDEX_DTYPES[count_dtype], (cute.sym_int(),)),
        _strided_matrix(Int32, batch, draft_len) if draft_len else None,
        _compact(Int32, (batch, draft_len + 1)),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )


def dspark_argmax_partials(logits: torch.Tensor) -> torch.Tensor:
    """``DSparkArgmaxPartialKernel`` partials ``[rows, ceil(vocab / 4096)]`` of ``logits``
    (``[rows, vocab]``, unit column stride), or ``torch.argmax(logits, -1)`` for rows that are
    not whole 16-byte vectors."""
    rows, vocab = logits.shape
    vec = 16 // logits.element_size()
    if vocab % vec or logits.stride(0) % vec or logits.data_ptr() % 16:
        return torch.argmax(logits, dim=-1)
    columns = DSparkArgmaxPartialKernel.columns
    partial = torch.empty((rows, ceil_div(vocab, columns)), dtype=torch.int64, device=logits.device)
    _compile_argmax_partial(logits.dtype)(logits, partial)
    return partial


def dspark_strict_accept(
    target: torch.Tensor,
    draft: torch.Tensor,
    *,
    force_base: Optional[int] = None,
    force_frac: float = 0.0,
    rng_pool: Optional[torch.Tensor] = None,
    rng_counter: Optional[torch.Tensor] = None,
    rng: tuple = (0, 0, 0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Strict acceptance of ``G`` generation requests with a uniform verify window
    (``SpecWorkerBase._sample_and_accept_draft_tokens_base``).

    ``target``: ``[G * (K + 1)]`` sampled target tokens, or the ``[G * (K + 1), vocab]`` logits
    (unit column stride) of a greedy batch, whose ``torch.argmax`` is taken; ``draft``: ``[G, K]``.
    ``force_base`` / ``force_frac`` / ``rng_*`` mirror ``_apply_force_accepted_tokens`` (same pool,
    counter and hash, so the same draws; ``rng_counter`` already advanced for this call when
    ``force_frac > 0``, ``rng`` = ``(counter_stride, slot_stride, pool_mask)``).
    Returns ``(accepted [G, K + 1] int32, num_accepted [G] int32)``.
    """
    num_gens, draft_len = draft.shape
    accepted = torch.empty((num_gens, draft_len + 1), dtype=torch.int32, device=draft.device)
    num_accepted = torch.empty((num_gens,), dtype=torch.int32, device=draft.device)
    force = 0 if force_base is None else 2 if force_frac > 0.0 else 1
    if target.dim() == 2:
        target = dspark_argmax_partials(target)
    chunks = target.shape[1] if target.dim() == 2 else 0
    compiled = _compile_strict_accept(
        draft_len, chunks, force, force_base or 0, rng, target.dtype, draft.dtype
    )
    compiled(
        target,
        draft,
        accepted,
        num_accepted,
        rng_pool if force == 2 else None,
        rng_counter if force == 2 else None,
        force_frac,
    )
    return accepted, num_accepted


def dspark_next_new_tokens(
    accepted: torch.Tensor,
    draft: torch.Tensor,
    batch_indices: torch.Tensor,
    num_accepted: torch.Tensor,
) -> torch.Tensor:
    """``SpecWorkerBase._prepare_next_new_tokens`` in one launch.

    Row ``b`` of the int32 ``[B, K + 1]`` result is
    ``accepted[batch_indices[b], num_accepted[b] - 1]`` followed by ``draft[b]`` (``[B, K]``);
    ``batch_indices`` and ``num_accepted`` hold at least ``B`` entries. ``accepted`` and
    ``draft`` are int32.
    """
    batch, draft_len = draft.shape
    out = torch.empty((batch, draft_len + 1), dtype=torch.int32, device=draft.device)
    if batch:
        compiled = _compile_next_new_tokens(draft_len, batch_indices.dtype, num_accepted.dtype)
        compiled(accepted, batch_indices, num_accepted, draft if draft_len else None, out)
    return out


class DSparkGenerationPrepareKernel:
    """One CTA prepares every generation request of a draft block.

    The scratch slot is reset first. Every request then reads its slot state, and
    the advanced state is written back only after a CTA barrier, so the requests
    that share the scratch slot all bootstrap from their own input position.
    """

    threads = 256

    def __init__(self, draft_len: int, window: int, position_cap: Optional[int]):
        self.draft_len = draft_len
        self.window = window
        self.position_cap = position_cap

    @cute.jit
    def __call__(
        self,
        accepted: cute.Tensor,
        num_accepted: cute.Tensor,
        positions: cute.Tensor,
        slots: cute.Tensor,
        ctx_len: cute.Tensor,
        valid_len: cute.Tensor,
        initialized: cute.Tensor,
        bonus: cute.Tensor,
        main_rows: cute.Tensor,
        start_pos: cute.Tensor,
        valid_out: cute.Tensor,
        interim_rows: cute.Tensor,
        interim_pos: cute.Tensor,
        interim_valid: cute.Tensor,
        gen_start: Int32,
        last_row: Int32,
        scratch: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            accepted,
            num_accepted,
            positions,
            slots,
            ctx_len,
            valid_len,
            initialized,
            bonus,
            main_rows,
            start_pos,
            valid_out,
            interim_rows,
            interim_pos,
            interim_valid,
            gen_start,
            last_row,
            scratch,
        ).launch(grid=[1, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        accepted: cute.Tensor,
        num_accepted: cute.Tensor,
        positions: cute.Tensor,
        slots: cute.Tensor,
        ctx_len: cute.Tensor,
        valid_len: cute.Tensor,
        initialized: cute.Tensor,
        bonus: cute.Tensor,
        main_rows: cute.Tensor,
        start_pos: cute.Tensor,
        valid_out: cute.Tensor,
        interim_rows: cute.Tensor,
        interim_pos: cute.Tensor,
        interim_valid: cute.Tensor,
        gen_start: Int32,
        last_row: Int32,
        scratch: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        gens = num_accepted.shape[0]
        width = accepted.shape[1]
        if tid == 0:
            ctx_len[scratch] = Int64(0)
            valid_len[scratch] = Int64(0)
            initialized[scratch] = Uint8(0)
        cute.arch.barrier()
        for g in cutlass.range(tid, gens, self.threads):
            slot = Int32(slots[g])
            count = Int32(num_accepted[g])
            base = gen_start + g * width
            bonus[g] = Int64(accepted[g, count - 1])
            main_rows[g] = Int64(base + count - 1)
            old = Int64(positions[base])
            if initialized[slot] != Uint8(0):
                old = ctx_len[slot]
            start = old + Int64(count)
            if cutlass.const_expr(self.position_cap is not None):
                # Bound both positions by the RoPE table they index: warmup graph
                # replays advance their slots without a request ever completing.
                if old > Int64(self.position_cap):
                    old = Int64(self.position_cap)
                if start > Int64(self.position_cap):
                    start = Int64(self.position_cap)
            start_pos[g] = start
            valid = valid_len[slot] + Int64(count)
            if valid > Int64(self.window):
                valid = Int64(self.window)
            valid_out[g] = valid
            for j in cutlass.range_constexpr(self.draft_len):
                row = base + j
                if row > last_row:
                    row = last_row
                interim_rows[g, j] = Int64(row)
                interim_pos[g, j] = old + Int64(1 + j)
                flag = Uint8(0)
                if j < count - 1:
                    flag = Uint8(1)
                interim_valid[g, j] = flag
        cute.arch.barrier()
        for g in cutlass.range(tid, gens, self.threads):
            slot = Int32(slots[g])
            ctx_len[slot] = start_pos[g]
            valid_len[slot] = valid_out[g]
            initialized[slot] = Uint8(1)
        cute.arch.barrier()
        for g in cutlass.range(tid, gens, self.threads):
            valid_out[g] = valid_len[Int32(slots[g])]


@functools.cache
def _compile(
    draft_len: int,
    window: int,
    position_cap: Optional[int],
    accepted_dtype: torch.dtype,
    count_dtype: torch.dtype,
    positions_dtype: torch.dtype,
):
    gens = cute.sym_int()
    num_slots = cute.sym_int()

    def vector(dtype, size):
        return cute.runtime.make_fake_compact_tensor(dtype, (size,), stride_order=(0,))

    def matrix(dtype, columns):
        return cute.runtime.make_fake_compact_tensor(dtype, (gens, columns), stride_order=(1, 0))

    kernel = DSparkGenerationPrepareKernel(draft_len, window, position_cap)
    return cute.compile(
        kernel,
        matrix(_INDEX_DTYPES[accepted_dtype], cute.sym_int()),
        vector(_INDEX_DTYPES[count_dtype], gens),
        vector(_INDEX_DTYPES[positions_dtype], cute.sym_int()),
        vector(Int64, gens),
        vector(Int64, num_slots),
        vector(Int64, num_slots),
        vector(Uint8, num_slots),
        vector(Int64, gens),
        vector(Int64, gens),
        vector(Int64, gens),
        vector(Int64, gens),
        matrix(Int64, draft_len),
        matrix(Int64, draft_len),
        matrix(Uint8, draft_len),
        Int32(0),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )


def dspark_generation_prepare(
    accepted_tokens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    position_ids: torch.Tensor,
    slots: torch.Tensor,
    ctx_len: torch.Tensor,
    valid_len: torch.Tensor,
    initialized: torch.Tensor,
    *,
    gen_start: int,
    num_rows: int,
    scratch_slot: int,
    draft_len: int,
    window: int,
    position_cap: Optional[int],
) -> tuple[torch.Tensor, ...]:
    """Prepare ``G`` generation requests whose target rows start at ``gen_start``.

    ``accepted_tokens`` is ``[G, W]`` with ``W`` target rows per request and
    ``position_ids`` the flat target positions. A slot without a position starts
    from its first target position. The slot state (``ctx_len``, ``valid_len``,
    ``initialized``) advances by the accepted counts in place.

    Returns the bonus tokens, their captured rows, the draft start positions and
    the window lengths ``[G]``, and the captured rows, window positions and
    validity ``[G, draft_len]`` of the intermediate accepted tokens.
    """
    gens = num_accepted_tokens.shape[0]
    device = accepted_tokens.device
    bonus, main_rows, start_pos, valid_out = torch.empty(4, gens, dtype=torch.int64, device=device)
    interim_rows, interim_pos = torch.empty(2, gens, draft_len, dtype=torch.int64, device=device)
    interim_valid = torch.empty(gens, draft_len, dtype=torch.bool, device=device)
    compiled = _compile(
        draft_len,
        window,
        position_cap,
        accepted_tokens.dtype,
        num_accepted_tokens.dtype,
        position_ids.dtype,
    )
    compiled(
        accepted_tokens,
        num_accepted_tokens,
        position_ids,
        slots,
        ctx_len,
        valid_len,
        initialized.view(torch.uint8),
        bonus,
        main_rows,
        start_pos,
        valid_out,
        interim_rows,
        interim_pos,
        interim_valid.view(torch.uint8),
        gen_start,
        num_rows - 1,
        scratch_slot,
    )
    return bonus, main_rows, start_pos, valid_out, interim_rows, interim_pos, interim_valid


class DSparkDraftBlockInputsKernel:
    """One CTA per draft block row ``(g, j)`` builds the inputs of the first draft stage.

    The row token is the bonus token for ``j == 0`` and the noise token otherwise; its
    embedding row (zero for an id outside the table) is written ``hc_mult`` times. The
    row also copies its RoPE phase at ``start_pos[g] + 1 + j`` (row ``j == 0`` also the
    phase at ``start_pos[g]``) and the one-hot input mix on residual stream 0.
    """

    threads = 128
    vec = 8
    freq_vec = 4

    def __init__(
        self, block_size: int, noise_token_id: int, hc_mult: int, hidden: int, rope_dim: int
    ):
        self.block_size = block_size
        self.noise_token_id = noise_token_id
        self.hc_mult = hc_mult
        self.vectors = hidden // (self.vec * self.threads)
        self.freq_vectors = rope_dim // self.freq_vec

    @cute.jit
    def __call__(
        self,
        bonus: cute.Tensor,
        start_pos: cute.Tensor,
        weight: cute.Tensor,
        freqs: cute.Tensor,
        draft_ids: cute.Tensor,
        x: cute.Tensor,
        main_freqs: cute.Tensor,
        block_freqs: cute.Tensor,
        pre_mix: Optional[cute.Tensor],
        stream: cuda.CUstream,
    ):
        copy_x = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128)
        copy_freqs = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
        )
        self.kernel(
            bonus,
            start_pos,
            weight,
            freqs,
            draft_ids,
            x,
            main_freqs,
            block_freqs,
            pre_mix,
            copy_x,
            copy_freqs,
        ).launch(
            grid=[bonus.shape[0] * self.block_size, 1, 1],
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        bonus: cute.Tensor,
        start_pos: cute.Tensor,
        weight: cute.Tensor,
        freqs: cute.Tensor,
        draft_ids: cute.Tensor,
        x: cute.Tensor,
        main_freqs: cute.Tensor,
        block_freqs: cute.Tensor,
        pre_mix: Optional[cute.Tensor],
        copy_x: cute.CopyAtom,
        copy_freqs: cute.CopyAtom,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        g = row // self.block_size
        j = row - g * self.block_size
        token = Int64(self.noise_token_id)
        if j == 0:
            token = bonus[g]

        weight_v = cute.logical_divide(weight, (None, self.vec))
        values = []
        for k in cutlass.range_constexpr(self.vectors):
            value = cute.make_rmem_tensor(cute.make_layout((self.vec,)), BFloat16)
            value.fill(0.0)
            values.append(value)
        if token >= Int64(0):
            if token < Int64(weight.shape[0]):
                for k in cutlass.range_constexpr(self.vectors):
                    cute.copy(
                        copy_x, weight_v[Int32(token), (None, tid + k * self.threads)], values[k]
                    )

        warp = tid // cute.arch.WARP_SIZE
        lane = tid % cute.arch.WARP_SIZE
        if warp == 0:
            if lane < self.freq_vectors:
                freqs_v = cute.logical_divide(freqs, (None, self.freq_vec))
                block_v = cute.logical_divide(block_freqs, (None, self.freq_vec))
                phase = cute.make_rmem_tensor(cute.make_layout((self.freq_vec,)), Float32)
                cute.copy(copy_freqs, freqs_v[Int32(start_pos[g]) + 1 + j, (None, lane)], phase)
                cute.copy(copy_freqs, phase, block_v[row, (None, lane)])
        if warp == 1:
            if j == 0:
                if lane < self.freq_vectors:
                    freqs_v = cute.logical_divide(freqs, (None, self.freq_vec))
                    main_v = cute.logical_divide(main_freqs, (None, self.freq_vec))
                    phase = cute.make_rmem_tensor(cute.make_layout((self.freq_vec,)), Float32)
                    cute.copy(copy_freqs, freqs_v[Int32(start_pos[g]), (None, lane)], phase)
                    cute.copy(copy_freqs, phase, main_v[g, (None, lane)])
        if warp == 2:
            if lane == 0:
                draft_ids[g, j] = token
            if cutlass.const_expr(pre_mix is not None):
                if lane < self.hc_mult:
                    mix = Float32(0.0)
                    if lane == 0:
                        mix = Float32(1.0)
                    pre_mix[row, lane] = mix

        x_v = cute.logical_divide(x, (None, self.vec))
        for h in cutlass.range_constexpr(self.hc_mult):
            for k in cutlass.range_constexpr(self.vectors):
                cute.copy(
                    copy_x, values[k], x_v[row * self.hc_mult + h, (None, tid + k * self.threads)]
                )


@functools.cache
def _compile_block_inputs(
    block_size: int, noise_token_id: int, hc_mult: int, hidden: int, rope_dim: int, pre_mix: bool
):
    gens = cute.sym_int()
    rows = cute.sym_int()

    def matrix(dtype, num_rows, columns):
        return cute.runtime.make_fake_compact_tensor(
            dtype, (num_rows, columns), stride_order=(1, 0), assumed_align=16
        )

    kernel = DSparkDraftBlockInputsKernel(block_size, noise_token_id, hc_mult, hidden, rope_dim)
    return cute.compile(
        kernel,
        cute.runtime.make_fake_compact_tensor(Int64, (gens,), stride_order=(0,)),
        cute.runtime.make_fake_compact_tensor(Int64, (gens,), stride_order=(0,)),
        matrix(BFloat16, cute.sym_int(), hidden),
        matrix(Float32, cute.sym_int(), rope_dim),
        matrix(Int64, gens, block_size),
        matrix(BFloat16, cute.sym_int(), hidden),
        matrix(Float32, gens, rope_dim),
        matrix(Float32, rows, rope_dim),
        matrix(Float32, rows, hc_mult) if pre_mix else None,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )


def dspark_draft_block_inputs(
    bonus: torch.Tensor,
    start_pos: torch.Tensor,
    embedding: torch.Tensor,
    freqs_cis: torch.Tensor,
    *,
    block_size: int,
    noise_token_id: int,
    hc_mult: int,
    pre_mix: bool,
) -> Optional[tuple]:
    """Build the first-stage inputs of ``G`` draft blocks in one launch.

    ``bonus`` and ``start_pos`` are int64 ``[G]``, ``embedding`` the BF16 ``[vocab, hidden]``
    table and ``freqs_cis`` the complex64 ``[positions, rope_dim // 2]`` RoPE table.

    Returns the hc-replicated block embeddings ``[G, block, hc_mult, hidden]``, the block
    token ids ``[G, block]``, the ``(main [G, 1, rope_dim // 2], block [G, block, rope_dim // 2])``
    RoPE phases and, with ``pre_mix``, the one-hot input mix ``[G, block, hc_mult, 1]``; None
    when the shapes are not supported.
    """
    gens = bonus.shape[0]
    hidden = embedding.shape[1]
    half = freqs_cis.shape[1]
    kernel = DSparkDraftBlockInputsKernel
    if (
        gens == 0
        or bonus.dtype != torch.int64
        or start_pos.dtype != torch.int64
        or embedding.dtype != torch.bfloat16
        or hidden % (kernel.vec * kernel.threads)
        or (2 * half) % kernel.freq_vec
        or 2 * half // kernel.freq_vec > cute.arch.WARP_SIZE
        or hc_mult > cute.arch.WARP_SIZE
    ):
        return None
    device = bonus.device
    x = torch.empty(gens, block_size, hc_mult, hidden, dtype=embedding.dtype, device=device)
    draft_ids = torch.empty(gens, block_size, dtype=torch.int64, device=device)
    main_freqs = torch.empty(gens, 1, half, dtype=freqs_cis.dtype, device=device)
    block_freqs = torch.empty(gens, block_size, half, dtype=freqs_cis.dtype, device=device)
    mix = (
        torch.empty(gens, block_size, hc_mult, 1, dtype=torch.float32, device=device)
        if pre_mix
        else None
    )
    compiled = _compile_block_inputs(block_size, noise_token_id, hc_mult, hidden, 2 * half, pre_mix)
    compiled(
        bonus,
        start_pos,
        embedding,
        torch.view_as_real(freqs_cis).view(-1, 2 * half),
        draft_ids,
        x.view(-1, hidden),
        torch.view_as_real(main_freqs).view(gens, 2 * half),
        torch.view_as_real(block_freqs).view(-1, 2 * half),
        None if mix is None else mix.view(-1, hc_mult),
    )
    return x, draft_ids, (main_freqs, block_freqs), mix
