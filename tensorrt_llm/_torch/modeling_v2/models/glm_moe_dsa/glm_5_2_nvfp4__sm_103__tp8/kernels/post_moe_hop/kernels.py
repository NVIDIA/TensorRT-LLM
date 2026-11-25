# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Triton kernel of the post_moe_hop (the post_moe_hop contract).

One launch per target MoE layer and rank. Program ``(row, chunk)`` owns
``BLOCK`` hidden columns of one token row and:

1. finalizes the raw NVFP4 MoE tuple locally, with the arithmetic of the
   native multi-node chain: fp32 FMA over the top-k expert rows in ``k``
   order (``-1`` entries skipped), one bf16 rounding, then the bf16 add of the
   shared-expert partial;
2. publishes the bf16 local partial through the multicast address of the
   post_moe_hop workspace into slot ``[row][rank]`` of every rank's Lamport buffer;
3. clears the Lamport buffer used by the previous call (all programs share
   the work);
4. polls the local copy of slot ``[row][*]`` until no element holds the
   bf16 ``-0.0`` sentinel, sums the ranks in rank order in fp32 and rounds
   once to bf16, like the native MNNVL one-shot all-reduce;
5. adds the residual once (bf16 add) and writes the new residual;
6. exchanges per-chunk sums of bf16-rounded squares of the new residual
   through a local Lamport buffer, so every program of the row sees the same
   row sum, and writes ``bf16(x * rsqrt(sum / H + eps) * gamma)``.

The last program to arrive advances the epoch (Lamport buffer index) and
records the row count to clear on the next call.

Errors. Every poll is bounded by a ``%globaltimer`` deadline (``timeout_ns``),
read through a program-wide reduction so that all warps of a program leave
the loop at the same iteration; on expiry the program sets a bit in the
per-rank error word, records its poll count, writes NaN residual/normalized
outputs and finishes, so a missing peer is reported instead of hanging the
stream. A permutation entry at or beyond
the FC2 row count is skipped and also sets an error bit. Each Lamport buffer
has its own error word, which collects the bits of the calls that use the
buffer. A call reads the error word of the previous call's buffer, which is
complete at the kernel boundary; when it is nonzero, the call publishes NaN
instead of its partial and carries the bits into its own error word. Errors
are therefore sticky until a collective workspace reset, the group's outputs
become NaN rather than silently wrong (the ranks' epochs are not guaranteed
to agree after a timeout), and a call's own result does not depend on the
order in which its programs record errors.

Workspace layout (see ``workspace.py``):

* ``lamport`` bf16 ``[3, max_rows, world, hidden]`` per rank (multicast
  stores, unicast polls). ``-0.0`` (0x8000) marks an empty element.
* ``partials`` fp32 ``[3, max_rows, MAX_CHUNKS]`` per rank, local only.
  ``-0.0`` marks an empty element.
* ``state`` int32: ``[epoch, prev_rows, arrivals, error[0], calls,
  timeout_polls, error[1], error[2]]``; ``error[b]`` is the error word of
  Lamport buffer ``b`` (the rank's error bits are their OR) and
  ``timeout_polls`` the largest poll count at which a poll of this rank gave
  up (diagnostics).
"""

import triton
import triton.language as tl

STATE_EPOCH = 0
STATE_PREV_ROWS = 1
STATE_ARRIVALS = 2
STATE_CALLS = 4
STATE_TIMEOUT_POLLS = 5
# Error word of Lamport buffer 0, 1, 2.
STATE_ERROR_WORDS = (3, 6, 7)
STATE_WORDS = 8

ERROR_LAMPORT_TIMEOUT = 1
ERROR_PARTIAL_TIMEOUT = 2
ERROR_PERM_RANGE = 4

MAX_CHUNKS = 64

TRACE_POINTS = 8

_BF16_NEG_ZERO_BITS = tl.constexpr(-32768)  # 0x8000 as int16
_FP32_NEG_ZERO_BITS = tl.constexpr(-2147483648)  # 0x80000000 as int32
_BF16_NAN_BITS = tl.constexpr(0x7FC0)
# Polls between two reads of the program clock (each read is a reduction).
_CLOCK_EVERY = tl.constexpr(16)
_FP32_NAN_BITS = tl.constexpr(0x7FC00000)


@triton.jit
def _globaltimer(like):
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l,r", [like], dtype=tl.int64, is_pure=False, pack=1
    )


@triton.jit
def _program_clock(lanes):
    """``%globaltimer`` as one value that every thread of the program agrees on.

    Each thread reads the timer for its elements of ``lanes`` (the program's
    column vector, which spans every warp when ``BLOCK >= 256 * num_warps``)
    and a program-wide max reduction combines them. The poll loops contain
    program-wide reductions, so every quantity in their exit condition must be
    uniform across warps; a per-thread clock, or a pending count reduced only
    within a warp, lets warps leave a loop at different iterations and pair up
    different barriers.
    """
    return tl.max(_globaltimer(lanes), axis=0)


@triton.jit
def _trace(trace_ptr, pid, point: tl.constexpr, TRACE: tl.constexpr):
    if TRACE:
        tl.store(trace_ptr + pid * 8 + point, _globaltimer(pid))


@triton.jit
def _fault_delay(pid, epoch, rank, site: tl.constexpr, FAULT_DELAY_NS: tl.constexpr):
    """Test-only fault injection: about one program in four spins before ``site``.

    The delay (up to ``FAULT_DELAY_NS``) is a hash of program, epoch, rank and
    site, so individual stores of one call land in a scrambled order.
    """
    if FAULT_DELAY_NS > 0:
        h = (pid * 40503 + epoch * 2531 + rank * 977 + site * 7919) % 65521
        if h % 4 == 0:
            delay = (h * 6151) % FAULT_DELAY_NS
            start = _globaltimer(pid)
            now = start
            while now - start < delay:
                now = _globaltimer(pid)


@triton.jit
def _post_moe_hop_kernel(
    fc2_ptr,
    fc2_row_stride,
    scale_ptr,
    perm_ptr,
    shared_ptr,
    residual_ptr,
    gamma_ptr,
    out_ptr,
    residual_out_ptr,
    mc_ptr,
    uc_ptr,
    partials_ptr,
    state_ptr,
    trace_ptr,
    fc2_rows,
    rank,
    eps,
    timeout_ns,
    HIDDEN: tl.constexpr,
    TOP_K: tl.constexpr,
    WORLD: tl.constexpr,
    MAX_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    CHUNKS_P2: tl.constexpr,
    CLEAR_BLOCK: tl.constexpr,
    PARTIAL_STRIDE: tl.constexpr,
    HAS_SHARED: tl.constexpr,
    USE_PDL: tl.constexpr,
    TRACE: tl.constexpr,
    FAULT_DELAY_NS: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    num_rows = tl.num_programs(0)
    pid = row * NUM_CHUNKS + chunk
    num_programs = num_rows * NUM_CHUNKS
    _trace(trace_ptr, pid, 0, TRACE)
    if USE_PDL:
        # Every global access below may depend on the previous kernel,
        # including the state words written by the previous post_moe_hop call.
        tl.extra.cuda.gdc_wait()
    _trace(trace_ptr, pid, 1, TRACE)

    buffer_elems = MAX_ROWS * WORLD * HIDDEN
    epoch = tl.load(state_ptr + 0)
    prev_rows = tl.load(state_ptr + 1)
    cur = epoch % 3
    dirty = (epoch + 2) % 3
    # Error words of this call's buffer and of the previous call's buffer.
    err_ptr = state_ptr + tl.where(cur == 0, 3, 5 + cur)
    sticky_bits = tl.load(state_ptr + tl.where(dirty == 0, 3, 5 + dirty))
    sticky = sticky_bits != 0

    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    col_mask = cols < HIDDEN

    # Inputs that do not depend on the exchange are fetched up front so their
    # latency overlaps the finalize and the cross-rank wait.
    residual = tl.load(residual_ptr + row * HIDDEN + cols, mask=col_mask, other=0.0)
    gamma = tl.load(gamma_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    # 1. Local finalize, native multi-node arithmetic.
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k in tl.static_range(TOP_K):
        permuted = tl.load(perm_ptr + row * TOP_K + k)
        weight = tl.load(scale_ptr + row * TOP_K + k).to(tl.float32)
        valid = (permuted >= 0) & (permuted < fc2_rows)
        if permuted >= fc2_rows:
            tl.atomic_or(err_ptr, 4)
        safe_row = tl.where(valid, permuted, 0).to(tl.int64)
        expert_row = tl.load(
            fc2_ptr + safe_row * fc2_row_stride + cols, mask=col_mask, other=0.0
        ).to(tl.float32)
        acc = tl.where(valid, tl.fma(weight, expert_row, acc), acc)
    local = acc.to(tl.bfloat16)
    if HAS_SHARED:
        shared = tl.load(shared_ptr + row * HIDDEN + cols, mask=col_mask, other=0.0)
        local = (shared.to(tl.float32) + local.to(tl.float32)).to(tl.bfloat16)
    local_bits = local.to(tl.int16, bitcast=True)
    local_bits = tl.where(local_bits == _BF16_NEG_ZERO_BITS, tl.zeros_like(local_bits), local_bits)
    local_bits = tl.where(sticky, tl.full([BLOCK], _BF16_NAN_BITS, tl.int16), local_bits)
    local = local_bits.to(tl.bfloat16, bitcast=True)

    # 2. Publish through multicast into slot [row][rank] of every rank.
    row_base = row * WORLD * HIDDEN
    _fault_delay(pid, epoch, rank, 0, FAULT_DELAY_NS)
    tl.store(mc_ptr + cur * buffer_elems + row_base + rank * HIDDEN + cols, local, mask=col_mask)
    if sticky:
        # After an error recorded by an earlier call: NaN until the reset.
        tl.atomic_or(err_ptr, sticky_bits)
    _trace(trace_ptr, pid, 2, TRACE)

    # 3. Clear the previous call's Lamport rows and partials.
    _fault_delay(pid, epoch, rank, 1, FAULT_DELAY_NS)
    sentinel_bf16 = tl.full([CLEAR_BLOCK], _BF16_NEG_ZERO_BITS, tl.int16).to(
        tl.bfloat16, bitcast=True
    )
    clear_elems = prev_rows * WORLD * HIDDEN
    clear_offsets = tl.arange(0, CLEAR_BLOCK)
    for start in range(pid * CLEAR_BLOCK, clear_elems, num_programs * CLEAR_BLOCK):
        offs = start + clear_offsets
        tl.store(uc_ptr + dirty * buffer_elems + offs, sentinel_bf16, mask=offs < clear_elems)
    # Unconditional: hops with different launch configs share the workspace,
    # so a single-chunk call must still clear what a chunked call left behind.
    if pid == 0:
        sentinel_fp32 = tl.full([PARTIAL_STRIDE], _FP32_NEG_ZERO_BITS, tl.int32).to(
            tl.float32, bitcast=True
        )
        part_offsets = tl.arange(0, PARTIAL_STRIDE)
        for r in range(0, prev_rows):
            tl.store(
                partials_ptr + (dirty * MAX_ROWS + r) * PARTIAL_STRIDE + part_offsets, sentinel_fp32
            )

    _trace(trace_ptr, pid, 3, TRACE)

    # 4. Lamport poll of every rank's slot, then a rank-ordered fp32 sum.
    rank_ids = tl.arange(0, WORLD)
    poll_ptrs = uc_ptr + cur * buffer_elems + row_base + rank_ids[:, None] * HIDDEN + cols[None, :]
    poll_mask = col_mask[None, :] & (rank_ids[:, None] < WORLD)
    values = tl.load(poll_ptrs, mask=poll_mask, other=0.0, volatile=True)
    pending = tl.sum((values.to(tl.int16, bitcast=True) == _BF16_NEG_ZERO_BITS).to(tl.int32))
    spins = 0
    poll_start = _program_clock(cols)
    elapsed = poll_start - poll_start
    while (pending > 0) & (elapsed < timeout_ns):
        values = tl.load(poll_ptrs, mask=poll_mask, other=0.0, volatile=True)
        pending = tl.sum((values.to(tl.int16, bitcast=True) == _BF16_NEG_ZERO_BITS).to(tl.int32))
        spins += 1
        if spins % _CLOCK_EVERY == 0:
            elapsed = _program_clock(cols) - poll_start
    timed_out = pending > 0
    if timed_out:
        tl.atomic_or(err_ptr, 1)
        tl.atomic_max(state_ptr + 5, spins)
    _trace(trace_ptr, pid, 4, TRACE)
    values_f32 = values.to(tl.float32)
    reduced = tl.zeros([BLOCK], dtype=tl.float32)
    for r in tl.static_range(WORLD):
        reduced += tl.sum(tl.where(rank_ids[:, None] == r, values_f32, 0.0), axis=0)
    reduced_bf16 = reduced.to(tl.bfloat16)
    if USE_PDL:
        # Dependents still wait for this grid to finish; only their launch
        # latency overlaps the epilogue.
        tl.extra.cuda.gdc_launch_dependents()

    # 5. Residual added once, after the reduction.
    new_residual = (reduced_bf16.to(tl.float32) + residual.to(tl.float32)).to(tl.bfloat16)
    new_residual = tl.where(
        timed_out,
        tl.full([BLOCK], _BF16_NAN_BITS, tl.int16),
        new_residual.to(tl.int16, bitcast=True),
    ).to(tl.bfloat16, bitcast=True)
    tl.store(residual_out_ptr + row * HIDDEN + cols, new_residual, mask=col_mask)

    # 6. RMSNorm over the full row.
    x = new_residual.to(tl.float32)
    square_sum = tl.sum((x * x).to(tl.bfloat16).to(tl.float32))
    if NUM_CHUNKS > 1:
        part_base = partials_ptr + (cur * MAX_ROWS + row) * PARTIAL_STRIDE
        tl.store(part_base + chunk, square_sum)
        # The poll reads the row's partials through a vector spread over every
        # warp (lane i reads partial i % CHUNKS_P2), so the pending count is a
        # program-wide sum and all warps leave the loop at the same iteration.
        lane_chunk = tl.arange(0, BLOCK) % CHUNKS_P2
        lane_mask = lane_chunk < NUM_CHUNKS
        lane_parts = tl.load(part_base + lane_chunk, mask=lane_mask, other=0.0, volatile=True)
        part_pending = tl.sum(
            (lane_parts.to(tl.int32, bitcast=True) == _FP32_NEG_ZERO_BITS).to(tl.int32)
        )
        part_spins = 0
        part_start = _program_clock(cols)
        part_elapsed = part_start - part_start
        while (part_pending > 0) & (part_elapsed < timeout_ns):
            lane_parts = tl.load(part_base + lane_chunk, mask=lane_mask, other=0.0, volatile=True)
            part_pending = tl.sum(
                (lane_parts.to(tl.int32, bitcast=True) == _FP32_NEG_ZERO_BITS).to(tl.int32)
            )
            part_spins += 1
            if part_spins % _CLOCK_EVERY == 0:
                part_elapsed = _program_clock(cols) - part_start
        if part_pending > 0:
            tl.atomic_or(err_ptr, 2)
            tl.atomic_max(state_ptr + 5, part_spins)
        # Each partial is written once per call, so once none is pending every
        # warp reads the same final values and sums them in the same order.
        chunk_ids = tl.arange(0, CHUNKS_P2)
        parts = tl.load(
            part_base + chunk_ids, mask=chunk_ids < NUM_CHUNKS, other=0.0, volatile=True
        )
        fp32_nan = tl.full([], _FP32_NAN_BITS, tl.int32).to(tl.float32, bitcast=True)
        square_sum = tl.where(part_pending > 0, fp32_nan, tl.sum(parts))
    _trace(trace_ptr, pid, 5, TRACE)
    inv_rms = tl.rsqrt(square_sum / HIDDEN + eps)
    tl.store(out_ptr + row * HIDDEN + cols, ((x * inv_rms) * gamma).to(tl.bfloat16), mask=col_mask)

    _trace(trace_ptr, pid, 6, TRACE)

    # 7. The last program to arrive advances the epoch for the next call.
    _fault_delay(pid, epoch, rank, 2, FAULT_DELAY_NS)
    arrived = tl.atomic_add(state_ptr + 2, 1, sem="acq_rel", scope="gpu")
    if arrived == num_programs - 1:
        tl.store(state_ptr + 1, num_rows)
        tl.store(state_ptr + 2, 0)
        tl.atomic_add(state_ptr + 4, 1, sem="relaxed", scope="gpu")
        tl.store(state_ptr + 0, (epoch + 1) % 3)


def launch_post_moe_hop(
    *,
    fc2_output,
    expert_scale_factor,
    expanded_idx_to_permuted_idx,
    shared_output,
    residual,
    norm_weight,
    out,
    residual_out,
    mc_buffer,
    uc_buffer,
    partials,
    state,
    rank: int,
    world: int,
    max_rows: int,
    eps: float,
    block: int,
    num_warps: int,
    timeout_ns: int,
    use_pdl: bool,
    trace=None,
    clear_block: int = 1024,
    fault_delay_ns: int = 0,
    fc2_rows=None,
) -> None:
    """Launch the fused hop on the current stream.

    Args:
        fc2_output: bf16 ``[N_padded, >=H]`` permuted expert rows, unit column
            stride; permutation entries ``>= N_padded`` are skipped and flagged.
        expert_scale_factor: bf16 or fp32 ``[R, top_k]`` contiguous.
        expanded_idx_to_permuted_idx: int32 ``[R, top_k]`` contiguous; ``-1`` skips a slot.
        shared_output: bf16 ``[R, H]`` contiguous, or ``None``.
        residual: bf16 ``[R, H]`` contiguous.
        norm_weight: bf16 ``[H]``.
        out, residual_out: bf16 ``[R, H]`` outputs.
        mc_buffer, uc_buffer: bf16 views of this rank's Lamport region
            (``[3 * max_rows * world * H]``), multicast and local unicast.
        partials: fp32 ``[3 * max_rows * MAX_CHUNKS]`` local buffer.
        state: int32 ``[STATE_WORDS]`` local state words.
        timeout_ns: deadline of every poll, in ``%globaltimer`` nanoseconds.
        trace: optional int64 ``[R * chunks, TRACE_POINTS]`` buffer that receives
            per-program ``%globaltimer`` stamps at the phase boundaries
            (diagnostics only).
        fault_delay_ns: test-only fault injection (see ``_fault_delay``); 0 in
            production.
        fc2_rows: test-only override of the permutation bound (memcheck
            positive control); defaults to ``fc2_output.shape[0]``.
    """
    if block < 256 * num_warps:
        raise ValueError(
            f"block {block} < 256 * num_warps ({256 * num_warps}): the poll loops "
            "need the column vector to span every warp of the program"
        )
    num_rows, hidden = residual.shape
    top_k = expanded_idx_to_permuted_idx.shape[1]
    num_chunks = triton.cdiv(hidden, block)
    grid = (num_rows, num_chunks)
    _post_moe_hop_kernel[grid](
        fc2_output,
        fc2_output.stride(0),
        expert_scale_factor,
        expanded_idx_to_permuted_idx,
        shared_output if shared_output is not None else residual,
        residual,
        norm_weight,
        out,
        residual_out,
        mc_buffer,
        uc_buffer,
        partials,
        state,
        trace if trace is not None else state,
        fc2_output.shape[0] if fc2_rows is None else fc2_rows,
        rank,
        eps,
        timeout_ns,
        HIDDEN=hidden,
        TOP_K=top_k,
        WORLD=world,
        MAX_ROWS=max_rows,
        BLOCK=block,
        NUM_CHUNKS=num_chunks,
        CHUNKS_P2=triton.next_power_of_2(num_chunks),
        CLEAR_BLOCK=clear_block,
        PARTIAL_STRIDE=MAX_CHUNKS,
        HAS_SHARED=shared_output is not None,
        USE_PDL=use_pdl,
        TRACE=trace is not None,
        FAULT_DELAY_NS=fault_delay_ns,
        num_warps=num_warps,
        launch_pdl=use_pdl,
    )
