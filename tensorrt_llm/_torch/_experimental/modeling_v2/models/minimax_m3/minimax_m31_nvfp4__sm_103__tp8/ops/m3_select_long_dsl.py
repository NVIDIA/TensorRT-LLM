# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Top-16 block select for MiniMax-M3 decode rows of up to 9216 blocks, one row per 384-thread CTA (CuTe DSL).

Bit for bit the selection of this target's ``m3_select_long`` (``torch.ops.trtllm.minimax_m3_select_long``,
one 1024-thread CTA per row), from a CTA of 384 threads:

* Every block's score is forced (the first ``init_blocks`` and the last ``local_blocks`` of the row's valid blocks) and
  mapped to an order key held in registers: thread t owns blocks t, t + 384, ... (4, 11 or 24 of them by the row's
  valid count). Larger scores get smaller keys; NaN sorts above +inf. The key is the 1024-thread kernel's unsigned
  key with its top bit flipped, so signed Int32 order here is that kernel's order. Every thread loads at a clamped
  block and masks the result, so a thread's loads issue back to back.
* 384 is a multiple of 16, so the 16 interleaved groups ``block % 16`` are the lanes' ``lane % 16``. The weakest of
  the 16 group maxima bounds the 16th-largest score, so the top 16 is among the keys at or above it.
* The candidates are compacted by a CTA prefix count, each as one int64 (key << 32 | block), and ranked exactly by
  that value (one signed compare per pair); more than 1024 of them (keys tied at the bound) rank every candidate
  against the whole row instead.
* Warp 0 emits the 16 picked block ids sorted ascending, -1 for an empty or -inf pick.

The kernel waits on its grid dependency first; it releases the dependent sparse decode right after that wait
(``early_trigger``), once every score of its row is loaded (trigger phase 1) or at its exit (phase 0). Kernels are
compiled per (PDL launch, early trigger, trigger phase) with symbolic shapes and strides; ``prebuild`` compiles and
loads them before the executor's warmup, so a decode step never compiles one.
"""

from typing import Dict, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor
from cutlass.utils.smem_allocator import SmemAllocator

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.utils import (
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
)

# The 384-thread select reads its row's valid-block count before its grid wait instead of after it, which takes one
# dependent global load off the post-wait path; the selection is the same either way. Exact: the count
# (MsaMetadata.msa_n_valid_blocks) is written only before the step's first layer, behind a plain launch.
SELECT_N_VALID_PREWAIT = True

SELECT_THREADS = 384
SELECT_WARPS = SELECT_THREADS // 32
# 1M tokens + draft tokens at 128 tokens per block.
MAX_BLOCKS = 9216  # capacity: fixed: the kernel's built blocks per row
MAX_CAND = 1024  # capacity: fixed: the kernel's candidate staging
TOPK = 16  # capacity: fixed: MSA's top-k
NO_KEY = 0x7FFFFFFF  # capacity: fixed: a sentinel, not a size (an unused key slot (the reference's 0xffffffff))
# The key of -inf (the reference's 0xff800000): never emitted.
NEG_INF_KEY = 0x7F800000  # capacity: fixed: a key encoding, not a size
INIT_SCORE = 1.0e30
LOCAL_SCORE = 1.0e29
# (keys per thread, rows up to this many valid blocks)
ITEM_CLASSES = ((4, 4 * SELECT_THREADS), (11, 11 * SELECT_THREADS), (24, MAX_BLOCKS))

# Shared memory of the phase, int32 words: candidates [MAX_CAND] int64 (key << 32 | block, so one signed compare
# orders them by (key, block)), picked keys / blocks [TOPK] each, per-warp group minima [SELECT_WARPS][16], group
# minima [16], per-warp candidate counts [SELECT_WARPS].
_CAND = 0
_PICK_KEY = _CAND + 2 * MAX_CAND
_PICK_BLOCK = _PICK_KEY + TOPK
_WARP_MIN = _PICK_BLOCK + TOPK
_GROUP_MIN = _WARP_MIN + SELECT_WARPS * 16
_WARP_CNT = _GROUP_MIN + 16
SMEM_WORDS = _WARP_CNT + SELECT_WARPS
SMEM_BYTES = 4 * SMEM_WORDS


def _key(score):
    """Order key of a float score: smaller key = larger score, NaN first; signed order = the reference's unsigned."""
    bits = cutlass.Float32(score).bitcast(cutlass.Int32)
    return cutlass.select_(bits < cutlass.Int32(0), bits & cutlass.Int32(0x7FFFFFFF), ~bits)


def _imin(a, b):
    return cutlass.select_(b < a, b, a)


def _imax(a, b):
    return cutlass.select_(b > a, b, a)


def _forced_key(scores, head, block, query, init_blocks, local_start):
    raw = scores[head, block, query]
    s = cutlass.select_(
        block >= local_start,
        cutlass.Float32(LOCAL_SCORE),
        cutlass.select_(block < init_blocks, cutlass.Float32(INIT_SCORE), cutlass.Float32(raw)),
    )
    return _key(s)


@cute.jit
def _select_row(
    row,
    scores: cute.Tensor,  # float32 [num_kv_heads, num_blocks, total_q], any strides
    n_valid: cute.Tensor,  # int32 [>= total_q]
    out: cute.Tensor,  # int32 [total_q, num_kv_heads, 16]
    init_blocks,
    local_blocks,
    num_kv_heads,
    smem: cute.Tensor,  # int32 [SMEM_WORDS], shared memory
    bar_id: cutlass.Constexpr,
    loaded_trigger: cutlass.Constexpr,
    raw_prewait,
    n_valid_prewait: cutlass.Constexpr,
):
    """Select row ``row`` (= query * num_kv_heads + head) with this CTA's SELECT_THREADS threads. Uses named barrier
    ``bar_id`` over SELECT_THREADS threads. ``loaded_trigger`` releases the dependent grid once every score of the
    row is loaded (trigger phase 1). With ``n_valid_prewait`` the row's valid-block count is ``raw_prewait``, read by
    the caller before its grid wait; otherwise it is read here."""
    tid, _, _ = cute.arch.thread_idx()
    lane = tid % cutlass.Int32(32)
    warp = tid // cutlass.Int32(32)
    query = row // num_kv_heads
    head = row % num_kv_heads
    if cutlass.const_expr(n_valid_prewait):
        raw = raw_prewait
    else:
        raw = n_valid[query]
    valid = _imax(_imin(raw, cutlass.Int32(scores.shape[1])), cutlass.Int32(0))
    local_start = _imax(raw - local_blocks, cutlass.Int32(0))
    cand = cute.make_tensor(
        cute.recast_ptr(smem.iterator + _CAND, dtype=cutlass.Int64), cute.make_layout(MAX_CAND)
    )

    if tid < TOPK:
        smem[_PICK_KEY + tid] = cutlass.Int32(NO_KEY)
        smem[_PICK_BLOCK + tid] = cutlass.Int32(0)

    if valid <= TOPK:
        # Every valid block is picked (thread t holds block t).
        if tid < valid:
            smem[_PICK_KEY + tid] = _forced_key(scores, head, tid, query, init_blocks, local_start)
            smem[_PICK_BLOCK + tid] = tid
        if cutlass.const_expr(loaded_trigger):
            griddepcontrol_launch_dependents()
    for ci in cutlass.range_constexpr(len(ITEM_CLASSES)):
        items = ITEM_CLASSES[ci][0]
        lo = TOPK if ci == 0 else ITEM_CLASSES[ci - 1][1]
        hi = ITEM_CLASSES[ci][1]
        if (valid > lo) & (valid <= hi):
            key = cute.make_rmem_tensor(items, cutlass.Int32)
            local_min = cutlass.Int32(NO_KEY)
            for s in cutlass.range_constexpr(items):
                # Every thread loads (a clamped block past the row) so the loads issue back to back, unbranched.
                block = tid + cutlass.Int32(s * SELECT_THREADS)
                k = _forced_key(
                    scores, head, _imin(block, valid - 1), query, init_blocks, local_start
                )
                key[s] = cutlass.select_(block < valid, k, cutlass.Int32(NO_KEY))
                local_min = _imin(local_min, key[s])
            local_min = _imin(local_min, cute.arch.shuffle_sync_bfly(local_min, 16))
            if lane < 16:
                smem[_WARP_MIN + warp * cutlass.Int32(16) + lane] = local_min
            cute.arch.barrier(barrier_id=bar_id, number_of_threads=SELECT_THREADS)
            if cutlass.const_expr(loaded_trigger):
                griddepcontrol_launch_dependents()
            if tid < 16:
                g = cutlass.Int32(NO_KEY)
                for w in cutlass.range_constexpr(SELECT_WARPS):
                    g = _imin(g, smem[_WARP_MIN + w * 16 + tid])
                smem[_GROUP_MIN + tid] = g
            cute.arch.barrier(barrier_id=bar_id, number_of_threads=SELECT_THREADS)
            bound = cutlass.Int32(
                -2147483648
            )  # the smallest key: signed keys, unlike the reference's unsigned ones
            for g in cutlass.range_constexpr(16):
                bound = _imax(bound, smem[_GROUP_MIN + g])

            # Candidates (key <= bound; unused slots hold NO_KEY > any bound): a CTA prefix count gives the slots.
            cnt = cutlass.Int32(0)
            for s in cutlass.range_constexpr(items):
                cnt += cutlass.select_(key[s] <= bound, cutlass.Int32(1), cutlass.Int32(0))
            incl = cnt
            for d in cutlass.range_constexpr(5):
                up = cute.arch.shuffle_sync(
                    incl, _imax(lane - cutlass.Int32(1 << d), cutlass.Int32(0))
                )
                incl += cutlass.select_(lane >= cutlass.Int32(1 << d), up, cutlass.Int32(0))
            if lane == 31:
                smem[_WARP_CNT + warp] = incl
            cute.arch.barrier(barrier_id=bar_id, number_of_threads=SELECT_THREADS)
            before = cutlass.Int32(0)
            total = cutlass.Int32(0)
            for w in cutlass.range_constexpr(SELECT_WARPS):
                c = smem[_WARP_CNT + w]
                before += cutlass.select_(cutlass.Int32(w) < warp, c, cutlass.Int32(0))
                total += c
            slot = before + incl - cnt
            for s in cutlass.range_constexpr(items):
                if key[s] <= bound:
                    if slot < MAX_CAND:
                        cand[slot] = (cutlass.Int64(key[s]) << cutlass.Int64(32)) | cutlass.Int64(
                            tid + cutlass.Int32(s * SELECT_THREADS)
                        )
                    slot += 1
            cute.arch.barrier(barrier_id=bar_id, number_of_threads=SELECT_THREADS)

            if total <= MAX_CAND:
                for i in cutlass.range(tid, total, SELECT_THREADS, unroll=1):
                    mine = cand[i]
                    rank = cutlass.Int32(0)
                    for j in cutlass.range(total, unroll=8):
                        rank += cutlass.select_(cand[j] < mine, cutlass.Int32(1), cutlass.Int32(0))
                    if rank < TOPK:
                        smem[_PICK_KEY + rank] = cutlass.Int32(mine >> cutlass.Int64(32))
                        smem[_PICK_BLOCK + rank] = cutlass.Int32(mine & cutlass.Int64(0xFFFFFFFF))
            else:
                # More than MAX_CAND keys tied at the bound: rank each candidate against the whole row.
                for s in cutlass.range_constexpr(items):
                    if key[s] <= bound:
                        block = tid + cutlass.Int32(s * SELECT_THREADS)
                        rank = cutlass.Int32(0)
                        for j in cutlass.range(valid, unroll=1):
                            kj = _forced_key(scores, head, j, query, init_blocks, local_start)
                            less = (kj < key[s]) | ((kj == key[s]) & (j < block))
                            rank += cutlass.select_(less, cutlass.Int32(1), cutlass.Int32(0))
                        if rank < TOPK:
                            smem[_PICK_KEY + rank] = key[s]
                            smem[_PICK_BLOCK + rank] = block
    cute.arch.barrier(barrier_id=bar_id, number_of_threads=SELECT_THREADS)

    if warp == 0:
        # The 16 picked block ids ascending (NO_KEY = empty or -inf, emitted as -1), bitonic over the warp.
        v = cutlass.Int32(NO_KEY)
        if lane < TOPK:
            pk = smem[_PICK_KEY + lane]
            if (pk != cutlass.Int32(NO_KEY)) & (pk != cutlass.Int32(NEG_INF_KEY)):
                v = smem[_PICK_BLOCK + lane]
        for size_log in cutlass.range_constexpr(1, 6):
            for stride_log in cutlass.range_constexpr(size_log - 1, -1, -1):
                other = cute.arch.shuffle_sync_bfly(v, 1 << stride_log)
                ascending = (lane & cutlass.Int32(1 << size_log)) == cutlass.Int32(0)
                lower = (lane & cutlass.Int32(1 << stride_log)) == cutlass.Int32(0)
                v = cutlass.select_(lower == ascending, _imin(v, other), _imax(v, other))
        if lane < TOPK:
            out[query, head, lane] = cutlass.select_(
                v == cutlass.Int32(NO_KEY), cutlass.Int32(-1), v
            )


class _SelectLongDslKernel:
    """One row per CTA: the grid wait, then ``_select_row``. ``n_valid_prewait`` reads the row's valid-block count
    before the grid wait, so that dependent load leaves the post-wait critical path. That is exact because
    ``n_valid`` is written only by ``on_update_kv_lens`` and the host copy in ``prepare()``, both before the step's
    first layer and behind a plain (non-PDL) launch, so no grid this select can overlap writes it
    (``attention/test_modeling_v2_m3_n_valid_contract.py``)."""

    def __init__(
        self, use_pdl: bool, early_trigger: bool, trigger_phase: int, n_valid_prewait: bool
    ):
        self.use_pdl = use_pdl
        self.early_trigger = early_trigger
        self.loaded_trigger = (not early_trigger) and trigger_phase == 1
        self.n_valid_prewait = n_valid_prewait

    @cute.kernel
    def kernel(self, scores, n_valid, out, init_blocks, local_blocks, num_kv_heads):
        row, _, _ = cute.arch.block_idx()
        smem = SmemAllocator()
        words = smem.allocate_tensor(cutlass.Int32, cute.make_layout(SMEM_WORDS), byte_alignment=16)
        raw_prewait = cutlass.Int32(0)
        if cutlass.const_expr(self.n_valid_prewait):
            raw_prewait = n_valid[row // num_kv_heads]
        griddepcontrol_wait()
        if cutlass.const_expr(self.early_trigger):
            griddepcontrol_launch_dependents()
        _select_row(
            row,
            scores,
            n_valid,
            out,
            init_blocks,
            local_blocks,
            num_kv_heads,
            words,
            bar_id=1,
            loaded_trigger=self.loaded_trigger,
            raw_prewait=raw_prewait,
            n_valid_prewait=self.n_valid_prewait,
        )

    @cute.jit
    def __call__(self, scores, n_valid, out, init_blocks, local_blocks, stream):
        num_kv_heads = scores.shape[0]
        self.kernel(scores, n_valid, out, init_blocks, local_blocks, num_kv_heads).launch(
            grid=(scores.shape[2] * num_kv_heads, 1, 1),
            block=(SELECT_THREADS, 1, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )


#: Compiled kernels by (PDL launch, early trigger, trigger phase, valid count read before the wait).
kernel_cache: Dict[Tuple[bool, bool, int, bool], object] = {}


def kernel_key(
    use_pdl: bool, early_trigger: bool, trigger_phase: int
) -> Tuple[bool, bool, int, bool]:
    """The compile key of a launch (the trigger phase does not matter with the early trigger), with this target's
    ``SELECT_N_VALID_PREWAIT``."""
    return (
        bool(use_pdl),
        bool(early_trigger),
        0 if early_trigger else int(trigger_phase),
        bool(SELECT_N_VALID_PREWAIT),
    )


def _compile(key: Tuple[bool, bool, int, bool]):
    if key not in kernel_cache:
        heads, blocks, rows = cute.sym_int(), cute.sym_int(), cute.sym_int()
        scores = make_fake_tensor(
            cutlass.Float32,
            (heads, blocks, rows),
            stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
        )
        n_valid = make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), stride_order=(0,))
        out = make_fake_tensor(
            cutlass.Int32, (rows, heads, TOPK), stride=(cute.sym_int64(), cute.sym_int64(), 1)
        )
        kernel_cache[key] = cute.compile(
            _SelectLongDslKernel(*key),
            scores,
            n_valid,
            out,
            cutlass.Int32(0),
            cutlass.Int32(0),
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    return kernel_cache[key]


def select_long_dsl(
    scores: torch.Tensor,
    n_valid_blocks: torch.Tensor,
    out: torch.Tensor,
    init_blocks: int,
    local_blocks: int,
    use_pdl: bool,
    early_trigger: bool,
    trigger_phase: int,
) -> None:
    """Launch the select on the current stream (compiling its key first if ``prebuild`` did not)."""
    _compile(kernel_key(use_pdl, early_trigger, trigger_phase))(
        scores, n_valid_blocks, out, init_blocks, local_blocks
    )


def prebuild(keys, device) -> list:
    """Compile every key and launch each once on a small dummy row (loading its module). Returns the keys."""
    scores = torch.zeros(
        1, 2048, 4, dtype=torch.float32, device=device
    )  # capacity: fixed: a dummy prebuild row, not a size
    n_valid = torch.full((4,), 2048, dtype=torch.int32, device=device)
    out = torch.empty(1, 4, TOPK, dtype=torch.int32, device=device).permute(1, 0, 2)
    built = []
    for key in keys:
        key = kernel_key(*key)
        _compile(key)(scores, n_valid, out, 0, 1)
        built.append(key)
    return built
