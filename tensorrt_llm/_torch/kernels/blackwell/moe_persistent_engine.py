# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The persistent unit scheduler of the decode MoE engines (``moe_nvfp4_engine``, ``moe_bf16_engine``); the PTX
helpers they share are in ``ptx_helpers``.

An engine runs one call's routed experts FC1 -> activation -> FC2 in one persistent launch. A unit is one 128-row
weight tile of a segment (at most ``n_max`` rows of one expert): 2I / 128 FC1 and H / 128 FC2 units per segment share
one TMA ring. Each CTA takes its FC1 units round-robin, then claims FC2 units as tickets from a pool, each usable once
its segment's FC1 counter is complete. One scheduler thread per CTA queues the units (``QD`` slots); the weight
loader, activation loader, MMA warp and the two epilogue groups (alternate units) read the queue in order.

Call state (``new_state``, int32): gen (the call's parity), done, the FC2 ticket pools [2] and the segment counters
[2][CNT_CAP], double-buffered by parity: every call zeroes the other parity's pool and counters, and its last CTA
advances gen. Calls on one state must be stream-ordered.
"""

from __future__ import annotations

import enum
from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64
from cutlass.cute.nvgpu import tcgen05

from .ptx_helpers import (
    _atom_add_acq_rel,
    _atom_add_relaxed,
    _bar_sync,
    _ld_acquire_u32,
    _ld_relaxed_u32,
    _red_release_add,
    _st_relaxed_u32,
)

TILE_M = 128
NUM_ACC = 2  # TMEM accumulator buffers (unit k uses buffer k % NUM_ACC)
EPI_GROUPS = 2  # groups of four epilogue warps taking alternate units
EPI_WARP0 = 4
THREADS = 128 + 128 * EPI_GROUPS
CNT_CAP = 512  # segment counters per parity
QD = 8  # unit queue slots
QL = 2  # claimed entries the scheduler may hold ahead of the weight loader
EPI_BAR = 1  # named barrier of epilogue group 0 (group g: EPI_BAR + g)
LOG2E = 1.4426950216293334961
# int32 words of the call state: gen, done, ticket pools [2], then the counters at byte 64
_CNT_OFFSET = 64
# Upper bound of an engine's shared memory outside its operand ring (tables, barriers, routing scratch, alignment)
NON_RING_SMEM = 27 * 1024


# ----------------------------------------------------------------------------------------------------------------
class Release(enum.Enum):
    """Where an engine releases its PDL dependents (a dependent that waits still waits for the grid's completion)."""

    AFTER_WAIT = 0  # right after the engine's own dependency wait
    POOL_SPENT = 1  # once a CTA's scheduler claimed the last FC2 ticket
    END = 2  # as each CTA exits


class PersistentMoeEngine:
    """The scheduling half of a decode MoE engine; a subclass adds its MMA, loaders, prologue and epilogue math.

    ``fc2_wait_in_scheduler``: the scheduler acquires an FC2 unit's segment counter before queueing it (the loaders'
    reads are then ordered through the queue's mbarriers); otherwise the activation loader acquires it per lane
    (``wait_fc1_done``) and the weight loader may stream the unit's weights before FC1 is complete."""

    fc2_wait_in_scheduler: bool = True
    release: Release = Release.END
    # Launched as a programmatic dependent (TRTLLM_ENABLE_PDL); set by launch() before compiling.
    use_pdl: bool = True

    def __init__(self, hidden: int, inter: int, n_max: int, num_stages: int, tile_k: int) -> None:
        assert hidden % tile_k == 0 and inter % tile_k == 0
        assert (2 * inter) % TILE_M == 0 and hidden % TILE_M == 0
        self.hidden, self.inter = hidden, inter
        self.n_max, self.num_stages, self.tile_k = n_max, num_stages, tile_k
        self.k1_tiles, self.k2_tiles = hidden // tile_k, inter // tile_k
        self.m1_tiles, self.m2_tiles = (2 * inter) // TILE_M, hidden // TILE_M
        self.cta_group = tcgen05.CtaGroup.ONE
        self.acc_dtype = cutlass.Float32

    # ------------------------------------------------------------------------------------------------------------
    @cute.jit
    def init_barriers(self, warp_idx, tmem_bar, q_full):
        """The TMEM-address barrier (MMA warp + every epilogue thread) and the queue's full / empty / taken
        barriers [3 QD]."""
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(tmem_bar, 32 + 128 * EPI_GROUPS)
                for i in cutlass.range_constexpr(QD):
                    cute.arch.mbarrier_init(q_full + i, 1)
                    cute.arch.mbarrier_init(q_full + QD + i, 1)
                    cute.arch.mbarrier_init(q_full + 2 * QD + i, 1)
            cute.arch.mbarrier_init_fence()

    @cute.jit
    def wait_dependency(self, state_addr, release_bar=None):
        """The grid dependency wait (and the release, if ``AFTER_WAIT`` without ``release_bar``); returns gen. Read
        after the wait: a PDL launch of the next call can start before this one's last CTA advanced gen, and every CTA
        must see the same parity."""
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.release == Release.AFTER_WAIT and release_bar is None):
            cute.arch.griddepcontrol_launch_dependents()
        return _ld_relaxed_u32(state_addr)

    @cute.jit
    def begin_call(self, state_addr, gen, nseg, bidx, tidx, gdx):
        """Zero the other parity's counters and ticket pool for the next call (no CTA of this call touches them);
        returns (this call's counters, its ticket pool, FC1 units, FC2 units, this CTA's FC1 units)."""
        par = gen & Int32(1)
        cnt_base = state_addr + Int64(_CNT_OFFSET) + Int64(par) * Int64(CNT_CAP * 4)
        nxt_base = state_addr + Int64(_CNT_OFFSET) + Int64(1 - par) * Int64(CNT_CAP * 4)
        for i in cutlass.range(bidx * THREADS + tidx, CNT_CAP, gdx * THREADS, unroll=1):
            _st_relaxed_u32(nxt_base + Int64(i) * 4, Int32(0))
        pool_addr = state_addr + Int64(8) + Int64(par) * 4
        if bidx == 0:
            if tidx == 0:
                _st_relaxed_u32(state_addr + Int64(8) + Int64(1 - par) * 4, Int32(0))
        # FC1 unit u = bidx + j G (segment u / M1, tile u % M1), static; FC2 units as tickets, dynamic
        u1 = nseg * self.m1_tiles
        u2 = nseg * self.m2_tiles
        n1 = Int32(0)
        if bidx < u1:
            n1 = (u1 - 1 - bidx) // gdx + 1
        return cnt_base, pool_addr, u1, u2, n1

    @cute.jit
    def end_call(self, warp_idx, tidx, gdx, tmem_base_smem_ptr, state_addr, gen):
        """After every role returned: free TMEM, count this CTA done (the last one advances gen) and release the
        dependents."""
        cute.arch.barrier()
        if warp_idx == 2:
            tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_smem_ptr)
            cute.arch.dealloc_tmem(tmem_ptr, self.num_tmem_cols)
        if tidx == 0:
            old = _atom_add_acq_rel(state_addr + Int64(4), Int32(1))
            if old == gdx - 1:
                _st_relaxed_u32(state_addr + Int64(4), Int32(0))
                _st_relaxed_u32(state_addr, gen + Int32(1))
        cute.arch.griddepcontrol_launch_dependents()

    # ------------------------------------------------------------------------------------------------------------
    @cute.jit
    def decode(self, g, u1):
        """(is_fc1, segment, weight tile) of global unit g: FC1 units are 0..u1-1, FC2 units u1 + ticket
        (segment-major)."""
        is1 = Int32(0)
        seg = Int32(0)
        mt = Int32(0)
        if g < u1:
            is1 = Int32(1)
            seg = g // self.m1_tiles
            mt = g % self.m1_tiles
        else:
            seg = (g - u1) // self.m2_tiles
            mt = (g - u1) % self.m2_tiles
        return is1, seg, mt

    @cute.jit
    def _push(self, s_q, q_full, k, g):
        """Entry k of the unit queue (one thread): at most QL entries untaken by the weight loader; a slot is reused
        once the epilogue freed it."""
        slot = k % QD
        if k >= QL:
            kk = k - QL
            cute.arch.mbarrier_wait(q_full + 2 * QD + (kk % QD), (kk // QD) & 1)
        if k >= QD:
            cute.arch.mbarrier_wait(q_full + QD + slot, ((k // QD) - 1) & 1)
        s_q[slot] = g
        cute.arch.mbarrier_arrive(q_full + slot)

    @cute.jit
    def wait_fc1_done(self, cnt_base, seg):
        """Acquire segment seg's FC1 counter once all its FC1 units released it."""
        ready = Int32(0)
        while ready == 0:
            if _ld_acquire_u32(cnt_base + Int64(seg) * 4) >= self.m1_tiles:
                ready = Int32(1)

    @cute.jit
    def scheduler(
        self, s_q, q_full, n1, u1, u2, bidx, gdx, pool_addr, cnt_base, lane, release_bar=None
    ):
        """Warp 3, lane 0: this CTA's FC1 units, then FC2 tickets (claimed one ahead), then the end marker.
        ``release_bar``: an mbarrier that completes once this CTA's outputs read by the dependents without their
        dependency wait are written; the dependents are released only after it (also for ``AFTER_WAIT``)."""
        if lane == 0:
            k = Int32(0)
            for j in cutlass.range(n1, unroll=1):
                self._push(s_q, q_full, k, bidx + j * gdx)
                k = k + 1
            if cutlass.const_expr(release_bar is not None and self.release == Release.AFTER_WAIT):
                cute.arch.mbarrier_wait(release_bar, 0)
                cute.arch.griddepcontrol_launch_dependents()
            t = u2
            if u2 > 0:
                t = _atom_add_relaxed(pool_addr, Int32(1))
            while t < u2:
                tn = _atom_add_relaxed(pool_addr, Int32(1))
                if cutlass.const_expr(self.release == Release.POOL_SPENT):
                    if tn >= u2:
                        # the pool is spent and t is this CTA's last ticket: one thread's trigger counts for the CTA
                        if cutlass.const_expr(release_bar is not None):
                            cute.arch.mbarrier_wait(release_bar, 0)
                        cute.arch.griddepcontrol_launch_dependents()
                if cutlass.const_expr(self.fc2_wait_in_scheduler):
                    self.wait_fc1_done(cnt_base, t // self.m2_tiles)
                self._push(s_q, q_full, k, u1 + t)
                k = k + 1
                t = tn
            self._push(s_q, q_full, k, Int32(-1))
            # the group that does not own entry k reads entry k + 1 (the other roles stop at k)
            self._push(s_q, q_full, k + 1, Int32(-1))

    @cute.jit
    def first_unit(self, s_q, q_full, u1, bidx):
        """A loader's first unit: the static first FC1 unit without waiting for the queue (the scheduler queues the
        same entry), else entry 0."""
        g = Int32(-1)
        if bidx < u1:
            g = bidx
        else:
            cute.arch.mbarrier_wait(q_full, 0)
            g = s_q[0]
        return g

    @cute.jit
    def unit(self, s_q, q_full, k):
        """(queue entry k, its slot), once it is queued."""
        slot = k % QD
        cute.arch.mbarrier_wait(q_full + slot, (k // QD) & 1)
        return s_q[slot], slot

    @cute.jit
    def take(self, q_full, slot):
        """The weight loader took the entry in ``slot`` (one arrival per entry)."""
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(q_full + 2 * QD + slot)

    @cute.jit
    def free_unit(self, bar_id, epi_tid, q_full, slot, cnt_base, seg, fc1: cutlass.Constexpr):
        """An epilogue group finished the entry in ``slot`` (every thread of the group stored its rows and read the
        entry); an FC1 unit also releases its segment's counter."""
        _bar_sync(bar_id, 128)
        if epi_tid == 0:
            if cutlass.const_expr(fc1):
                _red_release_add(cnt_base + Int64(seg) * 4, Int32(1))
            cute.arch.mbarrier_arrive(q_full + QD + slot)

    # ------------------------------------------------------------------------------------------------------------
    @cute.jit
    def epilogue_tmem(self, tiled_mma, tmem_bar, tmem_base_smem_ptr, epi_tid):
        """An epilogue thread's TMEM accumulator view: (tmem_ptr, accumulator layout, TMEM -> register copy, its
        slice, the register fragment of one 8-column chunk, this thread's accumulator row)."""
        thr_mma = tiled_mma.get_slice(0)
        acc_shape = tiled_mma.partition_shape_C((TILE_M, self.n_max))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        tCcL = thr_mma.partition_C(cute.make_identity_tensor((TILE_M, self.n_max)))
        cute.arch.mbarrier_arrive(tmem_bar)
        cute.arch.mbarrier_wait(tmem_bar, 0)
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_smem_ptr)
        copy_atom_t2r = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition(8), tcgen05.Pack.NONE), self.acc_dtype
        )
        tAcc0 = cute.flat_divide(
            cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)[((None, None), 0, 0)], (TILE_M, 8)
        )
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc0[(None, None, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tid)
        tTR_cL = thr_copy_t2r.partition_D(cute.flat_divide(tCcL[((None, None), 0, 0)], (TILE_M, 8)))
        rAcc = cute.make_rmem_tensor(tTR_cL[(None, None, None, 0, 0)].shape, self.acc_dtype)
        row0, _c0 = tTR_cL[(None, None, None, 0, 0)][0]
        return tmem_ptr, tCtAcc_fake, tiled_copy_t2r, thr_copy_t2r, rAcc, row0

    @cute.jit
    def load_acc(self, tmem_ptr, tCtAcc_fake, tiled_copy_t2r, thr_copy_t2r, rAcc, buf, ch):
        """Rows 8 ch .. 8 ch + 7 of accumulator buffer buf into rAcc (tcgen05.wait::ld included)."""
        tAcc = cute.flat_divide(
            cute.make_tensor(tmem_ptr + buf * self.n_max, tCtAcc_fake.layout)[((None, None), 0, 0)],
            (TILE_M, 8),
        )
        cute.copy(tiled_copy_t2r, thr_copy_t2r.partition_S(tAcc)[(None, None, None, 0, ch)], rAcc)
        cute.arch.fence_view_async_tmem_load()


# ----------------------------------------------------------------------------------------------------------------
def ring_smem_bytes(device: torch.device) -> int:
    """Shared memory an engine's operand ring may take on ``device``: a block's opt-in maximum but NON_RING_SMEM."""
    from ...cute_dsl_utils import smem_optin_bytes

    return smem_optin_bytes(device) - NON_RING_SMEM


def new_state(device: torch.device) -> torch.Tensor:
    """An engine's call state, zero-initialized; it must exist before CUDA-graph capture."""
    return torch.zeros(_CNT_OFFSET // 4 + 2 * CNT_CAP, dtype=torch.int32, device=device)


def launch(
    make_engine: Callable[[], object],
    key: tuple,
    args: tuple,
    device: torch.device,
    spare_sms: int = 0,
    tail_args: tuple = (),
) -> None:
    """Compile ``make_engine()`` once per ``key`` (the engine class and its compile-time configuration) and the PDL
    setting, and launch it with ``args`` on the current stream, one CTA per SM of ``device`` but ``spare_sms`` and the
    SMs ``cute_dsl_utils.sm_reserve()`` keeps out of every persistent grid (the grid size is a launch argument; the
    units balance over any grid). ``tail_args`` follow the stream (an engine option's parameters). The first call of a
    key must precede CUDA graph capture."""
    from ...cute_dsl_utils import sm_reserve
    from ...flashinfer_utils import get_env_enable_pdl

    use_pdl = get_env_enable_pdl()
    key = (*key, use_pdl)
    grid_ctas = Int32(
        torch.cuda.get_device_properties(device).multi_processor_count - spare_sms - sm_reserve()
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn = _COMPILED.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"compile the MoE engine {key[0].__name__} before CUDA graph capture"
            )
        engine = make_engine()
        engine.use_pdl = use_pdl
        fn = _COMPILED[key] = cute.compile(engine, *args, grid_ctas, stream, *tail_args)
    fn(*args, grid_ctas, stream, *tail_args)


_COMPILED = {}
