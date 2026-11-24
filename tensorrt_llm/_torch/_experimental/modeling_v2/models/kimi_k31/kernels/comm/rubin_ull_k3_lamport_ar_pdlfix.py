# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``rubin_ull_k3_lamport_ar``'s finish kernel with its stores into ``out`` behind
``griddepcontrol.wait``, for ``kimi_k31_pro_nvfp4__sm_107__helix16_wide``.

With PDL and without ``wait_first`` the shared finish stores into ``out`` right after its
Lamport poll, before the wait; ``out`` may be memory the producer still reads after releasing
its dependents (the push GEMMs load their activation after their release), so that store could
overwrite the producer's input. :class:`K3LamportFinishKernelP14` adds a
``griddepcontrol.wait`` before the one-shot store and the two-shot gather's copy; nothing else
changes, so every stored value and the result are bit for bit the shared kernel's.

The shared module stays untouched (the ULL target's path is unchanged); its helix8 twin
``kernels/comm/rubin_lamport_ar.py`` has the same finish, so a workspace of either finishes
here the same way.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

from .rubin_ull_k3_lamport_ar import (
    _FLAG_ARRIVALS,
    NUM_BUFFERS,
    THREADS,
    K3LamportFinishKernel,
    LamportWorkspace,
    _check,
    _stream,
    ag_add_bf16x2,
    ag_row_addr,
    clean4,
    clear_stage,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
    lamport_state,
    ld_acquire_u32,
    ld_volatile_v4,
    multimem_st_v4,
    pack_bf16x2,
    poll_v4,
    red_release_add,
    required_bytes,
    sanitize_bf16x2,
    st_u32,
    st_v4,
    stage_addr,
)


class K3LamportFinishKernelP14(K3LamportFinishKernel):
    """``K3LamportFinishKernel`` whose workers store into ``out`` only after
    ``griddepcontrol.wait`` (the module docstring). Their earlier writes are workspace memory no
    allocation reuses; a worker counts itself before its wait, so the updater (which waits for the
    producer, then every count) never waits for a worker waiting on it: no cycle."""

    @cute.kernel
    def kernel(
        self, uc_local: Int64, mc_base: Int64, flags: Int64, out: Int64, m: Int32, rank: Int32
    ):
        W, H = self.W, self.H
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        workers = m * self.cols
        arrivals = flags + Int64(_FLAG_ARRIVALS * 4)
        if bidx == workers:
            # Updater: the next call's state, once the producer is done with this call's.
            if cutlass.const_expr(self.use_pdl):
                griddepcontrol_launch_dependents()
                griddepcontrol_wait()
            if tidx == 0:
                cur, _, nbytes, _, _, _, _, _ = lamport_state(flags)
                arrived = ld_acquire_u32(arrivals)
                while arrived < workers:
                    arrived = ld_acquire_u32(arrivals)
                row_bytes = H * 2
                clear0 = m * ((W + (1 if self.ag else 0)) * row_bytes)
                clear1 = Int32(0)
                if cutlass.const_expr(not self.one_shot):
                    clear0 = ((m + (W - 1)) // W) * (W * row_bytes)
                    clear1 = m * row_bytes
                st_v4(flags, (cur + 1) % NUM_BUFFERS, cur, nbytes, Int32(1 if self.one_shot else 2))
                st_v4(flags + Int64(16), clear0, clear1, Int32(0), Int32(0))
                st_u32(arrivals, Int32(0))
        else:
            tok = bidx // self.cols
            by = bidx - tok * self.cols
            v = by * THREADS + tidx
            if cutlass.const_expr(self.use_pdl and self.wait_first):
                griddepcontrol_wait()
            cur, dirty, nbytes, dstages, c0, c1, c2, c3 = lamport_state(flags)
            if cutlass.const_expr(self.use_pdl):
                griddepcontrol_launch_dependents()
            # Count this CTA once every thread has read the state.
            cute.arch.barrier()
            if tidx == 0:
                red_release_add(arrivals, Int32(1))
            row_bytes = H * 2
            out_v = out + Int64(tok * H + v * 8) * 2
            if cutlass.const_expr(self.one_shot):
                buf = stage_addr(uc_local, cur, Int32(0), 1, nbytes)
                if v < self.vecs:
                    acc = self.reduce_slots(buf + Int64(tok * W * H + v * 8) * 2, row_bytes)
                    if cutlass.const_expr(self.ag):
                        # bf16(y + bf16(sum)): the sum rounds as the C++ kernel stores it (no -0
                        # arises, see Numerics), then the all-gathered value is added.
                        y = poll_v4(
                            ag_row_addr(uc_local, cur, nbytes, m, tok, W, H) + Int64(v * 16)
                        )
                        o = [ag_add_bf16x2(acc[2 * q], acc[2 * q + 1], y[q]) for q in range(4)]
                    else:
                        o = [pack_bf16x2(acc[2 * q], acc[2 * q + 1]) for q in range(4)]
                        if cutlass.const_expr(self.sanitize_out):
                            o = [sanitize_bf16x2(w) for w in o]
                    # ``out`` may reuse memory the producer still reads after its release.
                    if cutlass.const_expr(self.use_pdl and not self.wait_first):
                        griddepcontrol_wait()
                    st_v4(out_v, *o)
            else:
                if v < self.vecs:
                    if tok % W == rank:
                        scatter = stage_addr(uc_local, cur, Int32(0), 2, nbytes)
                        acc = self.reduce_slots(
                            scatter + Int64((tok // W) * W * H + v * 8) * 2, row_bytes
                        )
                        o = [
                            sanitize_bf16x2(pack_bf16x2(acc[2 * q], acc[2 * q + 1]))
                            for q in range(4)
                        ]
                        bcast_mc = stage_addr(mc_base, cur, Int32(1), 2, nbytes)
                        multimem_st_v4(bcast_mc + Int64(tok * H + v * 8) * 2, *o)
            # Re-arm the previous call's buffer (every stage it used) while the rows travel.
            gtid = bidx * THREADS + tidx
            nthreads = workers * THREADS
            clears = (c0, c1, c2, c3)
            for s in cutlass.range_constexpr(4):
                if s < dstages:
                    clear_stage(
                        uc_local + Int64(dirty * dstages + s) * Int64(nbytes // dstages),
                        clears[s],
                        gtid,
                        nthreads,
                    )
            if cutlass.const_expr(not self.one_shot and self.gather):
                if v < self.vecs:
                    src = (
                        stage_addr(uc_local, cur, Int32(1), 2, nbytes) + Int64(tok * H + v * 8) * 2
                    )
                    w = ld_volatile_v4(src)
                    while clean4(*w) == 0:
                        w = ld_volatile_v4(src)
                    # Behind the wait, as the one-shot store above.
                    if cutlass.const_expr(self.use_pdl and not self.wait_first):
                        griddepcontrol_wait()
                    st_v4(out_v, *w)


@functools.cache
def _compiled_finish(
    world: int,
    hidden: int,
    one_shot: bool,
    use_pdl: bool,
    wait_first: bool = False,
    cpp_one_shot: bool | None = None,
    gather: bool = True,
    ag: bool = False,
):
    """``rubin_ull_k3_lamport_ar._compiled_finish`` of :class:`K3LamportFinishKernelP14`."""
    kernel = K3LamportFinishKernelP14(
        world, hidden, one_shot, use_pdl, wait_first, cpp_one_shot, gather, ag
    )
    a = Int64(1 << 20)
    return cute.compile(kernel, a, a, a, a, Int32(1), Int32(0), _stream())


def run_lamport_finish(
    ws: LamportWorkspace,
    out: torch.Tensor,
    one_shot: bool,
    use_pdl: bool = True,
    wait_first: bool = False,
    cpp_one_shot: bool | None = None,
    ag: bool = False,
) -> torch.Tensor:
    """``rubin_ull_k3_lamport_ar.run_lamport_finish`` on :class:`K3LamportFinishKernelP14`
    (``ws``: a ``LamportWorkspace`` of either Lamport module)."""
    m, hidden = out.shape
    _check(ws, m, hidden, one_shot, ag)
    _compiled_finish(ws.world, hidden, one_shot, use_pdl, wait_first, cpp_one_shot, True, ag)(
        Int64(ws.uc_local),
        Int64(ws.mc_base),
        Int64(ws.flags.data_ptr()),
        Int64(out.data_ptr()),
        Int32(m),
        Int32(ws.rank),
        _stream(),
    )
    return out


__all__ = [
    "K3LamportFinishKernelP14",
    "LamportWorkspace",
    "required_bytes",
    "run_lamport_finish",
]
