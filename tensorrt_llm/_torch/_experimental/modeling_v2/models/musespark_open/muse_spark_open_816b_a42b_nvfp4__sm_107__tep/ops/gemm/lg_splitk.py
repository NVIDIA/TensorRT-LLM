# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``lg_splitk``: the split-K dense GEMM with its activation tile polled into shared memory.

Port of stage S3 / S3b (branch ``mk/s3b``, sm_103__tep, 8-row steps) to this target's 9..32-row
split-K tactics (``kernels/dense_gemms/dispatcher.SM107_TP8_WIDE_KERNELS``). Nothing in the
served path imports this module yet: it is the kernel and the arena the go/no-go edge bench
(``test_modeling_v2_lg_splitk_wide.py``) measures before any wiring.

The kernel subclasses this target's ``splitk_gemm.SplitKDenseGemmKernel`` (no vendored copy):
the same weight TMA warp (warp 0), the same tcgen05 MMA warp (warp 2), the same owner reduction
in the epilogue (the mailbox layout the served call compiles with, ``_launch_options``; never
the reduce-scatter epilogue) and one BF16 rounding per output element, so the bits are the
served kernel's at the same tactic. What changes is how operand B -- the activation -- reaches
shared memory, and when the kernel waits on its predecessor:

* **Polling warps.** The activation is an *arena*: a ``[rows, K]`` buffer the producer stores
  into, holding the sentinel word ``0xFFFEFFFE`` in every 32-bit word until then. ``P`` polling
  warps (``poll_warps`` 1, 2 or 4: warps ``(1,)``, ``(1, 3)`` or ``(1, 3, 4, 5)``; warp 3 is idle
  in the stock kernel and warps 4-5 are epilogue warps that wait for the MMAs anyway) poll their
  share of each k-tile's 16-byte chunks with ``ld.volatile.global.v4`` before any wait, a group of
  k-tiles per statement so the loads are in flight together, and store them with
  ``st.shared.v4`` at the addresses the TMA's 128-byte swizzle fills
  (``lg_splitk_ptx.b_chunk_offsets``), round after round until a warp vote finds no 16-bit half
  equal to ``0xFFFE``. Each lane then executes ``fence.proxy.async.shared::cta`` and arrives on the
  stage's full barrier (``1 + 32 P`` arrivals: the weight warp's ``expect_tx`` and the polling
  lanes). More polling warps keep more of a wide tile's chunks in flight per round: at 32 rows a
  k-tile is 16 chunks per lane of one warp, so one warp polls one k-tile per L2 round trip.
* **Deadline.** A polling warp whose group still holds a sentinel half ``deadline`` ns after it
  started (default 20 us) executes ``griddepcontrol.wait`` and copies its chunks of that group
  and every later one from the fallback source (the producer's output: the arena itself, or the
  tensor the arena mirrors) with plain loads and the same shared stores -- the same bytes at the
  same addresses, and no TMA, so no speculative generic store can race an async-proxy write.
  Each fallback adds one to a fallback count (pinned host memory when available).
* **Trigger at entry (warp 0).** Dependents are released before the first weight load.
* **Wait before publishing.** The epilogue executes ``griddepcontrol.wait`` after its TMEM load
  and the split-K reduction and before its global store. The epilogue warps arrive on the TMEM
  allocation barrier at entry (before polling), so the barrier's phases keep the stock order.
* **Arena retirement (warp 3, after polling), per unit.** A unit is one K slice of one token
  tile (``lg_splitk_ptx``). Once the CTA's MMAs consumed every stage, the warp counts the CTA on
  its unit; the unit's last CTA writes the sentinel back over the unit and publishes the call's
  ``tag`` as the unit's generation (release).
* **Generation check (each polling warp, before its first poll).** A call polls its unit only
  once an ``ld.acquire.gpu`` of the unit's generation returns ``expect`` (the previous call's tag
  on the arena); else, at the deadline, the fallback. Compiled out with ``gen_check=False`` for a
  site whose producer releases the GEMM only after its own wait.

``run_lg_splitk`` is the host entry; ``LgArena`` holds an arena, its control block and the
fallback count. A row count must be a whole number of the tactic's token tiles (``mma_n``).
"""

import os
from typing import Dict, Optional, Tuple

import torch

from . import lg_splitk_ptx as _ptx

#: Default deadline after which a polling warp still finding a sentinel half falls back.
DEFAULT_DEADLINE_NS = 20_000
#: Words of an arena's device control block.
CONTROL_WORDS = _ptx.CONTROL_WORDS
#: The polling warps per ``poll_warps`` setting.
POLL_WARP_IDS = {1: (1,), 2: (1, 3), 4: (1, 3, 4, 5)}
#: ``MUSESPARK_OPEN_LG_SPLITK_POLL_WARPS`` (bench / test override of the default 1).
DEFAULT_POLL_WARPS = int(os.environ.get("MUSESPARK_OPEN_LG_SPLITK_POLL_WARPS", "1"))

# No postponed annotations here: the CuTe DSL evaluates the kernel methods' annotations.
try:
    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    import cutlass.utils.blackwell_helpers as sm100_utils
    from cutlass import Int32, Int64, Uint32
    from cutlass._mlir.dialects import llvm
    from cutlass.cute import experimental as cute_ext
    from cutlass.cutlass_dsl import T, dsl_user_op

    from . import splitk_gemm as _sk
except ImportError:
    HAS_CUTE_DSL = False
else:
    HAS_CUTE_DSL = True


if HAS_CUTE_DSL:

    def _asm(result_type, operands, text, constraints, loc, ip, side_effects=True):
        return llvm.inline_asm(
            result_type,
            operands,
            text,
            constraints,
            has_side_effects=side_effects,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def _globaltimer(*, loc=None, ip=None):
        return Int64(_asm(T.i64(), [], _ptx.GLOBALTIMER, "=l", loc, ip))

    @dsl_user_op
    def _poll_tile(start, deadline, gaddrs, saddrs, backoff_ns=0, *, loc=None, ip=None):
        """``lg_splitk_ptx.poll_tile_ptx``: ARRIVED (0) or FALLBACK (2), the same on every lane."""
        chunks = len(gaddrs)
        operands = [Int64(start).ir_value(loc=loc, ip=ip), Int64(deadline).ir_value(loc=loc, ip=ip)]
        operands += [Int64(g).ir_value(loc=loc, ip=ip) for g in gaddrs]
        operands += [Int32(s).ir_value(loc=loc, ip=ip) for s in saddrs]
        return Uint32(
            _asm(
                T.i32(),
                operands,
                _ptx.poll_tile_ptx(chunks, backoff_ns),
                _ptx.poll_tile_constraints(chunks),
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _copy_chunks(delta, gaddrs, saddrs, *, loc=None, ip=None):
        """``lg_splitk_ptx.copy_chunks_ptx``: the fallback's copy of a group's chunks."""
        chunks = len(gaddrs)
        operands = [Int64(delta).ir_value(loc=loc, ip=ip)]
        operands += [Int64(g).ir_value(loc=loc, ip=ip) for g in gaddrs]
        operands += [Int32(s).ir_value(loc=loc, ip=ip) for s in saddrs]
        _asm(
            None,
            operands,
            _ptx.copy_chunks_ptx(chunks),
            _ptx.copy_chunks_constraints(chunks),
            loc,
            ip,
        )

    @dsl_user_op
    def _fence_proxy_async_shared(*, loc=None, ip=None):
        _asm(None, [], _ptx.FENCE_PROXY_ASYNC_SHARED, "", loc, ip)

    @dsl_user_op
    def _count_fallback(address, *, loc=None, ip=None):
        _asm(None, [Int64(address).ir_value(loc=loc, ip=ip)], _ptx.COUNT_FALLBACK, "l", loc, ip)

    @dsl_user_op
    def _last_cta(counter, total, *, loc=None, ip=None):
        return Uint32(
            _asm(
                T.i32(),
                [Int64(counter).ir_value(loc=loc, ip=ip), Int32(total).ir_value(loc=loc, ip=ip)],
                _ptx.LAST_CTA,
                "=r,l,r",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _store_sentinel_x4(address, *, loc=None, ip=None):
        _asm(None, [Int64(address).ir_value(loc=loc, ip=ip)], _ptx.STORE_SENTINEL_X4, "l", loc, ip)

    @dsl_user_op
    def _await_gen(gen, expect, start, deadline, *, loc=None, ip=None):
        """``lg_splitk_ptx.AWAIT_GEN``: ARRIVED (0) or FALLBACK (2), the same on every lane."""
        return Uint32(
            _asm(
                T.i32(),
                [
                    Int64(gen).ir_value(loc=loc, ip=ip),
                    Int32(expect).ir_value(loc=loc, ip=ip),
                    Int64(start).ir_value(loc=loc, ip=ip),
                    Int64(deadline).ir_value(loc=loc, ip=ip),
                ],
                _ptx.AWAIT_GEN,
                _ptx.AWAIT_GEN_CONSTRAINTS,
                loc,
                ip,
            )
        )

    @dsl_user_op
    def _retire_slice(counter, gen, tag, *, loc=None, ip=None):
        _asm(
            None,
            [
                Int64(counter).ir_value(loc=loc, ip=ip),
                Int64(gen).ir_value(loc=loc, ip=ip),
                Int32(tag).ir_value(loc=loc, ip=ip),
            ],
            _ptx.RETIRE_SLICE,
            _ptx.RETIRE_SLICE_CONSTRAINTS,
            loc,
            ip,
        )

    class LgSplitKPollKernel(_sk.SplitKDenseGemmKernel):
        """The split-K kernel with polling B warps (module docstring). Always PDL, no bias."""

        def __init__(
            self,
            *,
            tactic,
            gen_check: bool = True,
            poll_warps: int = 1,
            mailbox_v4: bool = False,
            backoff_ns: int = 0,
        ) -> None:
            super().__init__(
                tactic=tactic, use_pdl=True, has_bias=False, mailbox_v4=mailbox_v4, rs=False
            )
            if poll_warps not in POLL_WARP_IDS:
                raise ValueError(
                    f"lg_splitk: poll_warps {poll_warps} not in {sorted(POLL_WARP_IDS)}"
                )
            self.poll_warps = int(poll_warps)
            self.chunks_per_lane = _ptx.chunks_per_lane(tactic.mma_n, self.poll_warps)
            self.poll_group = _ptx.poll_group(tactic.mma_n, tactic.ab_stages, self.poll_warps)
            #: Epilogue warps that poll first (warps 4, 5 at ``poll_warps`` 4).
            self.epilog_pollers = sum(1 for w in POLL_WARP_IDS[self.poll_warps] if w >= 4)
            self.gen_check = bool(gen_check)
            #: ``nanosleep`` between polling rounds (``lg_splitk_ptx.poll_tile_ptx``).
            self.backoff_ns = int(backoff_ns)

        @cute.experimental.jit
        def __call__(
            self,
            a: cute.Tensor,
            b: cute.Tensor,
            c: cute.Tensor,
            bias: cute.Tensor,
            arena: Int64,
            delta: Int64,
            row_bytes: Int32,
            control: Int64,
            fallbacks: Int64,
            deadline: Int64,
            unit_ctas: Int32,
            tag: Int32,
            expect: Int32,
            stream: cuda_driver.CUstream,
        ):
            self.kernel(
                a,
                b,
                c,
                bias,
                arena,
                delta,
                row_bytes,
                control,
                fallbacks,
                deadline,
                unit_ctas,
                tag,
                expect,
            ).launch(
                grid=(
                    cute.ceil_div(c.layout.shape[0], self.cta_m),
                    cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                    c.layout.shape[2],
                ),
                block=(self.threads_per_cta, 1, 1),
                cluster=self.cluster_shape,
                smem=cute.Int64(self.launch_smem),
                stream=stream,
                use_pdl=self.use_pdl,
            )

        @cute.experimental.kernel
        def kernel(
            self,
            mA: cute.Tensor,
            mB: cute.Tensor,
            mC: cute.Tensor,
            mBias: cute.Tensor,
            arena: Int64,
            delta: Int64,
            row_bytes: Int32,
            control: Int64,
            fallbacks: Int64,
            deadline: Int64,
            unit_ctas: Int32,
            tag: Int32,
            expect: Int32,
        ):
            """The stock ``kernel`` statement for statement, except: the full barriers count
            1 + 32 P arrivals; warp 0 triggers at entry; the polling warps check their unit's
            generation and poll; warp 3 retires its unit; the epilogue arrives on the TMEM
            barrier at entry and waits before its store."""
            stages = self.num_ab_stage

            ab_dtype = mA.element_type
            tiled_mma = sm100_utils.make_trivial_tiled_mma(
                ab_dtype,
                ab_dtype,
                utils.LayoutEnum.from_tensor(mA).mma_major_mode(),
                utils.LayoutEnum.from_tensor(mB).mma_major_mode(),
                self.acc_dtype,
                self.cta_group,
                self.mma_tiler_mn,
            )

            mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
            block_idx = cute.arch.block_idx()
            bidx = block_idx[0]
            split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            n_idx = block_idx[1] // self.split_k
            l_idx = block_idx[2]
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

            sA = cute_ext.allocate(
                ab_dtype,
                cute.AddressSpace.smem,
                sm100_utils.make_smem_layout_a(tiled_mma, mnk_tiler, ab_dtype, stages),
                alignment=_sk._AB_BUFFER_ALIGN_BYTES,
            )
            sB = cute_ext.allocate(
                ab_dtype,
                cute.AddressSpace.smem,
                sm100_utils.make_smem_layout_b(tiled_mma, mnk_tiler, ab_dtype, stages),
                alignment=_sk._AB_BUFFER_ALIGN_BYTES,
            )

            acc_layout = cute_ext.make_tmem_layout_acc(tiled_mma, self.mma_tiler_mn, acc_stage=1)
            c_tiler_mn = (self.cta_m, self.cta_n)

            bar_full = cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(stages),
                alignment=_sk._MBARRIER_BYTES,
            ).iterator
            bar_empty = cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(stages),
                alignment=_sk._MBARRIER_BYTES,
            ).iterator
            bar_tma_epilog = cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(1),
                alignment=_sk._MBARRIER_BYTES,
            ).iterator
            bar_mma_epilog = cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(1),
                alignment=_sk._MBARRIER_BYTES,
            ).iterator
            bar_tmem_alloc = cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(1),
                alignment=_sk._MBARRIER_BYTES,
            ).iterator
            tmem_base_ptr = cute_ext.allocate(
                cutlass.Int32,
                cute.AddressSpace.smem,
                cute.make_layout(1),
                alignment=_sk._TMEM_POINTER_BYTES,
            ).iterator

            if cutlass.const_expr(self.split_k > 1):
                mailbox = cute_ext.allocate(
                    cutlass.Float32,
                    cute.AddressSpace.smem,
                    cute.make_layout(self.mailbox_elements),
                    alignment=_sk._MAILBOX_ALIGN_BYTES,
                )
                bar_reduce = cute_ext.allocate(
                    cutlass.Int64,
                    cute.AddressSpace.smem,
                    cute.make_layout(1),
                    alignment=_sk._MBARRIER_BYTES,
                ).iterator
            else:
                mailbox = sA
                bar_reduce = bar_mma_epilog

            if warp_idx == 0:
                with cute.arch.elect_one():
                    for i in range(stages):
                        cute.arch.mbarrier_init(bar_full + i, 1 + _ptx.LANES * self.poll_warps)
                        cute.arch.mbarrier_init(bar_empty + i, 1)
                    cute.arch.mbarrier_init(bar_tma_epilog, 32)
                    cute.arch.mbarrier_init(bar_mma_epilog, 1)
                    cute.arch.mbarrier_init(bar_tmem_alloc, 160)

                    if cutlass.const_expr(self.split_k > 1):
                        cute.arch.mbarrier_init(bar_reduce, 1)

            cute.arch.mbarrier_init_fence()
            if cutlass.const_expr(self.split_k > 1):
                cute.arch.cluster_arrive_relaxed()
            else:
                cute.arch.barrier()

            k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
            k_tile_start = split_rank * k_tile_count

            if cutlass.const_expr(self.split_k > 1):
                cute.arch.cluster_wait()

            unit = n_idx * self.split_k + split_rank
            tile_rows = arena + Int64(n_idx * self.cta_n) * Int64(row_bytes)
            gen = control + Int64((_ptx.GEN_WORD + unit) * 4)

            if warp_idx == 0:
                # Trigger at entry; the weight warp triggers again after its last load (a second
                # launch_dependents of a CTA changes nothing).
                cute.arch.griddepcontrol_launch_dependents()
                self.dma_warp(
                    bar_full,
                    bar_empty,
                    bar_tma_epilog,
                    cute.local_tile(mA, (self.cta_m, self.cta_k), (bidx, None, l_idx)),
                    sA,
                    cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A"),
                    k_tile_start,
                    k_tile_count,
                    True,
                )
            elif warp_idx == 1:
                self.poll_warp_b(
                    bar_full,
                    bar_empty,
                    sB,
                    k_tile_start,
                    k_tile_count,
                    tile_rows,
                    delta,
                    row_bytes,
                    fallbacks,
                    deadline,
                    gen,
                    expect,
                    Int32(0),
                )
                # No producer drain: generic stores leave no asynchronous copy in flight.
            elif warp_idx == 2:
                self.mma_warp(
                    bar_full,
                    bar_empty,
                    bar_mma_epilog,
                    bar_tmem_alloc,
                    tiled_mma,
                    sA,
                    sB,
                    tmem_base_ptr,
                    acc_layout,
                    self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2]),
                    k_tile_count,
                )
            elif warp_idx == 3:
                if cutlass.const_expr(self.poll_warps >= 2):
                    self.poll_warp_b(
                        bar_full,
                        bar_empty,
                        sB,
                        k_tile_start,
                        k_tile_count,
                        tile_rows,
                        delta,
                        row_bytes,
                        fallbacks,
                        deadline,
                        gen,
                        expect,
                        Int32(1),
                    )
                self.release_warp(
                    bar_mma_epilog,
                    control + Int64((_ptx.COUNTER_WORD + unit) * 4),
                    gen,
                    tile_rows,
                    row_bytes,
                    k_tile_start,
                    k_tile_count,
                    unit_ctas,
                    tag,
                )
            elif warp_idx >= 4:
                # The epilogue's first TMEM-barrier arrival before any polling: the barrier's
                # first phase then completes as in the stock kernel, before the MMA warp's second
                # arrival.
                cute.arch.mbarrier_arrive(bar_tmem_alloc)
                if cutlass.const_expr(self.epilog_pollers > 0):
                    if warp_idx < 4 + self.epilog_pollers:
                        self.poll_warp_b(
                            bar_full,
                            bar_empty,
                            sB,
                            k_tile_start,
                            k_tile_count,
                            tile_rows,
                            delta,
                            row_bytes,
                            fallbacks,
                            deadline,
                            gen,
                            expect,
                            warp_idx - 2,
                        )
                self.epilog_warp_publish_after_wait(
                    bar_mma_epilog,
                    bar_tmem_alloc,
                    tmem_base_ptr,
                    acc_layout,
                    cute.local_tile(mC, c_tiler_mn, (bidx, n_idx, l_idx)),
                    cute.arch.thread_idx()[0] - 128,
                    mC.element_type,
                    utils.LayoutEnum.from_tensor(mC),
                    mailbox,
                    bar_reduce,
                    split_rank,
                )

        @cute.experimental.jit
        def poll_warp_b(
            self,
            bar_full,
            bar_empty,
            s_tile: cute.Tensor,
            k_tile_start: cutlass.Int32,
            k_tile_count: cutlass.Int32,
            rows: Int64,
            delta: Int64,
            row_bytes: Int32,
            fallbacks: Int64,
            deadline: Int64,
            gen: Int64,
            expect: Int32,
            poll_index: Int32,
        ):
            """Polling warp ``poll_index``: the stock B ``dma_warp`` with its chunks of each
            k-tile polled instead of waited for and TMA-loaded (the same ring and stages).

            First (``gen_check``) the unit's generation must read ``expect``; a deadline there
            takes the fallback for every k-tile. Then ``poll_group`` k-tiles at a time (their
            stages freed first): one ``poll_tile_ptx`` statement keeps all their loads in
            flight. After a deadline: ``griddepcontrol.wait`` and plain copies from the fallback
            source (``rows + delta``)."""
            stages = self.num_ab_stage
            group = self.poll_group
            stage_bytes = self.cta_n * self.cta_k * _ptx.ELEMENT_BYTES
            tile_bytes = self.cta_k * _ptx.ELEMENT_BYTES
            s_base = Int32(s_tile.iterator.toint())
            lane = cute.arch.lane_idx()
            start = _globaltimer()
            mode = Uint32(0)
            if cutlass.const_expr(self.gen_check):
                if _await_gen(gen, expect, start, deadline) != Uint32(_ptx.ARRIVED):
                    mode = Uint32(_ptx.FALLBACK)
                    _count_fallback(fallbacks)
                    cute.arch.sync_warp()
                    cute.arch.griddepcontrol_wait()
            empty_phase = cutlass.Int32(1)
            for g in cutlass.range(k_tile_count // group, unroll=1):
                gaddrs = []
                saddrs = []
                for t in cutlass.range_constexpr(group):
                    stage = (g * group + t) % stages
                    cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
                    if stage == stages - 1:
                        empty_phase = empty_phase ^ 1
                    column_bytes = (k_tile_start + g * group + t) * tile_bytes
                    dst = s_base + stage * stage_bytes
                    for j in cutlass.range_constexpr(self.chunks_per_lane):
                        g_off, s_off = _ptx.b_chunk_offsets(
                            _ptx.chunk_index(lane, poll_index, self.poll_warps, j),
                            row_bytes,
                            self.cta_n,
                        )
                        gaddrs.append(rows + Int64(g_off + column_bytes))
                        saddrs.append(dst + s_off)
                if mode == Uint32(0):
                    code = _poll_tile(start, deadline, gaddrs, saddrs, self.backoff_ns)
                    if code != Uint32(_ptx.ARRIVED):
                        mode = Uint32(_ptx.FALLBACK)
                        _count_fallback(fallbacks)
                        cute.arch.sync_warp()
                        cute.arch.griddepcontrol_wait()
                if mode == Uint32(_ptx.FALLBACK):
                    _copy_chunks(delta, gaddrs, saddrs)
                # Each lane's generic stores before the async proxy's reads (the MMA).
                _fence_proxy_async_shared()
                for t in cutlass.range_constexpr(group):
                    cute.arch.mbarrier_arrive(bar_full + (g * group + t) % stages)

        @cute.experimental.jit
        def release_warp(
            self,
            bar_mma_epilog,
            counter: Int64,
            gen: Int64,
            rows: Int64,
            row_bytes: Int32,
            k_tile_start: cutlass.Int32,
            k_tile_count: cutlass.Int32,
            unit_ctas: Int32,
            tag: Int32,
        ):
            """Once this CTA's MMAs consumed every B stage: count it on its unit; the unit's last
            CTA writes the sentinel back over the unit and publishes ``tag`` as its generation."""
            cute.arch.mbarrier_wait(bar_mma_epilog, 0)
            last = _last_cta(counter, unit_ctas)
            if last != Uint32(0):
                lane = cute.arch.lane_idx()
                for r in cutlass.range_constexpr(self.cta_n):
                    for i in cutlass.range(k_tile_count // 2, unroll=1):
                        _store_sentinel_x4(
                            rows
                            + Int64(
                                _ptx.slice_chunk_offset(
                                    r, i * _ptx.LANES + lane, row_bytes, k_tile_start
                                )
                            )
                        )
                _retire_slice(counter, gen, tag)

        @cute.experimental.jit
        def epilog_warp_publish_after_wait(
            self,
            bar_mma_epilog,
            bar_tmem_alloc,
            tmem_base_ptr,
            acc_layout: cutlass.Constexpr,
            gD_tile: cute.Tensor,
            epi_tid: cutlass.Int32,
            c_dtype: cutlass.Constexpr,
            d_layout: cutlass.Constexpr,
            mailbox,
            bar_reduce,
            split_rank: cutlass.Int32,
        ):
            """The stock ``epilog_warp`` (owner reduction, no bias) without its first TMEM-barrier
            arrival (made at the warp's entry) and with ``griddepcontrol.wait`` before the final
            store: after the TMEM load, the TMEM release and the split-K reduction."""
            cute.arch.mbarrier_wait(bar_tmem_alloc, 0)

            acc_view = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr),
                acc_layout,
            )[((None, None), 0, 0, 0)]

            epi_tile = (self.cta_m, self.cta_n)
            tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(
                sm100_utils.get_tmem_load_op(
                    (self.cta_m, self.cta_n, self.cta_k),
                    d_layout,
                    c_dtype,
                    self.acc_dtype,
                    epi_tile,
                    False,
                ),
                acc_view,
            )
            gD_epi = cute.flat_divide(gD_tile, epi_tile)

            rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
            rAcc = cute_ext.allocate(
                self.acc_dtype,
                cute.AddressSpace.rmem,
                rmem_layout,
                alignment=32,
            )
            rD = cute_ext.allocate(
                c_dtype,
                cute.AddressSpace.rmem,
                rmem_layout,
                alignment=32,
            )
            thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

            cute.arch.mbarrier_wait(bar_mma_epilog, 0)
            cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
            cute.arch.fence_view_async_tmem_load()
            cute.arch.mbarrier_arrive(bar_tmem_alloc)

            if cutlass.const_expr(self.split_k > 1):
                values_per_thread = cutlass.const_expr(cute.size(rmem_layout))
                values_per_peer = cutlass.const_expr(self.epilog_threads * values_per_thread)
                if split_rank != _sk.OWNER_RANK:
                    for value_idx in cutlass.range_constexpr(0, values_per_thread, 4):
                        if cutlass.const_expr(self.mailbox_v4):
                            slot = (value_idx // 4) * (self.epilog_threads * 4) + epi_tid * 4
                        else:
                            slot = epi_tid * values_per_thread + value_idx
                        _sk._store_shared_remote_v4(
                            rAcc[value_idx],
                            rAcc[value_idx + 1],
                            rAcc[value_idx + 2],
                            rAcc[value_idx + 3],
                            mailbox.iterator + (split_rank - Int32(1)) * values_per_peer + slot,
                            bar_reduce,
                            Int32(_sk.OWNER_RANK),
                        )
                else:
                    if epi_tid == 0:
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            bar_reduce, self.expected_transaction_bytes
                        )
                    cute.arch.mbarrier_wait(bar_reduce, 0)
                    for peer in cutlass.range_constexpr(self.split_k - 1):
                        for value_idx in cutlass.range_constexpr(values_per_thread):
                            if cutlass.const_expr(self.mailbox_v4):
                                slot = (
                                    (value_idx // 4) * (self.epilog_threads * 4)
                                    + epi_tid * 4
                                    + value_idx % 4
                                )
                            else:
                                slot = epi_tid * values_per_thread + value_idx
                            # Same additions in the same order (own, then peers in rank order).
                            rAcc[value_idx] = (
                                rAcc[value_idx] + mailbox[peer * values_per_peer + slot]
                            )

            # Publish after wait: the predecessor grid has completed before any output store.
            cute.arch.griddepcontrol_wait()
            if split_rank == _sk.OWNER_RANK:
                rD.store(rAcc.load().to(c_dtype))
                cute_ext.partition_and_copy(thr_t2r, rD, gD_epi[None, None, 0, 0])

    @cute.experimental.jit
    def bmm(
        gemm_op: cutlass.Constexpr,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        arena: Int64,
        delta: Int64,
        row_bytes: Int32,
        control: Int64,
        fallbacks: Int64,
        deadline: Int64,
        unit_ctas: Int32,
        tag: Int32,
        expect: Int32,
        stream: cuda_driver.CUstream,
    ):
        """The stock ``_bmm_no_bias`` with the arena's arguments passed through."""
        c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
        gemm_op(
            cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
            cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
            c,
            cute.make_tensor(c.iterator, cute.select(c.layout, mode=[0, 1, 2])),
            arena,
            delta,
            row_bytes,
            control,
            fallbacks,
            deadline,
            unit_ctas,
            tag,
            expect,
            stream,
        )


class LgArena:
    """One site's arena: the ``[rows, k]`` BF16 buffer its producer publishes into, filled with
    ``SENTINEL_WORD``; the device control block (per unit a release counter and a generation
    tag, ``lg_splitk_ptx``); and the fallback count.

    ``slices`` is the split-K of the site's tactic and ``tiles`` its token tiles
    (``rows / mma_n``); ``rest_tag`` the generation every unit holds when the arena is at rest --
    the tag of the last call of the site's cycle, which the cycle's first call expects. A one-call
    cycle (``rest_tag`` 1, every call ``tag = expect = 1``) suits calls a host synchronization
    separates (the op tests).
    """

    def __init__(
        self,
        rows: int,
        k: int,
        device: torch.device,
        slices: int = 1,
        tiles: int = 1,
        rest_tag: int = 1,
    ) -> None:
        if (rows * k * _ptx.ELEMENT_BYTES) % (_ptx.CHUNK_BYTES * _ptx.LANES):
            raise ValueError(f"an arena of {rows} x {k} BF16 is not whole 512-byte rounds")
        if not 1 <= int(slices) or k % (int(slices) * _ptx.CTA_K):
            raise ValueError(f"{slices} slices of a K of {k} are not whole k-tiles")
        if not 1 <= int(tiles) or rows % int(tiles):
            raise ValueError(f"{tiles} token tiles do not split {rows} rows")
        if int(slices) * int(tiles) > _ptx.MAX_UNITS:
            raise ValueError(
                f"{slices} slices x {tiles} tiles exceed the control block's {_ptx.MAX_UNITS} units"
            )
        if not 1 <= int(rest_tag) < 2**31:
            raise ValueError(f"rest tag {rest_tag} is not a positive int32")
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.rows, self.k, self.device = int(rows), int(k), device
        self.slices, self.tiles, self.rest_tag = int(slices), int(tiles), int(rest_tag)
        self.units = self.slices * self.tiles
        self.tensor = torch.empty((self.rows, self.k), dtype=torch.bfloat16, device=self.device)
        self.control = torch.zeros(CONTROL_WORDS, dtype=torch.int32, device=self.device)
        self.host_fallbacks: Optional[torch.Tensor] = None
        try:
            self.host_fallbacks = torch.zeros(4, dtype=torch.int32, pin_memory=True)
        except Exception:  # noqa: BLE001 - the device word serves instead
            self.host_fallbacks = None
        self.refill()
        torch.cuda.synchronize(self.device)

    @property
    def fallback_address(self) -> int:
        if self.host_fallbacks is not None:
            return int(self.host_fallbacks.data_ptr())
        return int(self.control.data_ptr()) + 4 * _ptx.FALLBACK_WORD

    def fallbacks(self) -> int:
        """Fallbacks so far (a polling warp counts once per call)."""
        torch.cuda.synchronize(self.device)
        if self.host_fallbacks is not None:
            return int(self.host_fallbacks[0])
        return int(self.control[_ptx.FALLBACK_WORD])

    def holds(self, x: torch.Tensor) -> bool:
        """Whether ``x`` is this arena (same start; the GEMM reads it as ``[rows, k]``)."""
        return (
            x.is_cuda
            and x.data_ptr() == self.tensor.data_ptr()
            and x.numel() == self.tensor.numel()
            and x.dtype == torch.bfloat16
        )

    def mirrors(self, x: torch.Tensor) -> bool:
        """Whether ``x`` has the arena's layout without being the arena (a mirror's source)."""
        return (
            x.is_cuda
            and x.device == self.device
            and x.dtype == torch.bfloat16
            and x.numel() == self.tensor.numel()
            and int(x.shape[-1]) == self.k
            and x.is_contiguous()
            and x.data_ptr() % 16 == 0
            and x.data_ptr() != self.tensor.data_ptr()
        )

    def refill(self, tag: Optional[int] = None) -> None:
        """Every word back to the sentinel, the counters to 0, then every unit's generation to
        ``tag`` (default ``rest_tag``), on the current stream."""
        self.tensor.view(torch.int32).fill_(_ptx.SENTINEL_I32)
        self.control[_ptx.COUNTER_WORD : _ptx.COUNTER_WORD + _ptx.MAX_UNITS].zero_()
        self.control[_ptx.GEN_WORD : _ptx.GEN_WORD + self.units].fill_(
            self.rest_tag if tag is None else int(tag)
        )

    def generations(self) -> list:
        """Every unit's generation (synchronizes; tests and checks)."""
        torch.cuda.synchronize(self.device)
        return [int(v) for v in self.control[_ptx.GEN_WORD : _ptx.GEN_WORD + self.units]]

    def is_clean(self, tag: Optional[int] = None) -> bool:
        """Every word the sentinel, every counter 0 and every generation ``tag`` (default
        ``rest_tag``). Synchronizes; tests and checks."""
        torch.cuda.synchronize(self.device)
        words = self.tensor.view(torch.int32)
        counters = self.control[_ptx.COUNTER_WORD : _ptx.COUNTER_WORD + _ptx.MAX_UNITS]
        want = self.rest_tag if tag is None else int(tag)
        return (
            bool((words == _ptx.SENTINEL_I32).all())
            and not bool(counters.any())
            and self.generations() == [want] * self.units
        )


#: ``key -> compiled kernel``.
_COMPILED: Dict[tuple, object] = {}
#: Launches made by ``run_lg_splitk`` (host side; tests read it).
LAUNCHES = 0


def _mailbox_v4(rows: int, split_k: int) -> bool:
    """The mailbox layout the served call of ``rows`` at ``split_k`` compiles with."""
    return bool(_sk._launch_options(rows, split_k).get("mailbox_v4", False))


def _key(dtype, tactic, leading_dims, gen_check, poll_warps, mailbox_v4, backoff_ns=0) -> tuple:
    fields = (tactic.mma_m, tactic.mma_n, tactic.split_k, tactic.ab_stages)
    return (
        dtype,
        *fields,
        tuple(leading_dims),
        bool(gen_check),
        int(poll_warps),
        bool(mailbox_v4),
        int(backoff_ns),
    )


def compiled(
    dtype, tactic, leading_dims, gen_check=True, poll_warps=1, mailbox_v4=False, backoff_ns=0
):
    """The compiled kernel (compiles outside a capture; raises inside one)."""
    key = _key(dtype, tactic, leading_dims, gen_check, poll_warps, mailbox_v4, backoff_ns)
    found = _COMPILED.get(key)
    if found is not None:
        return found
    if not HAS_CUTE_DSL:
        raise ImportError("lg_splitk needs the CuTe DSL")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("lg_splitk: compile the kernel before the capture (an eager call)")
    kernel = LgSplitKPollKernel(
        tactic=_sk.SplitKTactic(*key[1:5]),
        gen_check=bool(gen_check),
        poll_warps=int(poll_warps),
        mailbox_v4=bool(mailbox_v4),
        backoff_ns=int(backoff_ns),
    )
    a, b, c, _ = _sk._make_compile_repr_tensors(dtype, False, *leading_dims)
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    found = cute_ext.compile(
        bmm,
        kernel,
        a,
        b,
        c,
        Int64(0),
        Int64(0),
        Int32(0),
        Int64(0),
        Int64(0),
        Int64(0),
        Int32(0),
        Int32(0),
        Int32(0),
        stream,
    )
    _COMPILED[key] = found
    return found


def serves(
    x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, tactic, poll_warps: int = 1
) -> Optional[str]:
    """Why the kernel cannot serve ``x [M, K] @ w [K, N] -> out``, or None.

    The split-K kernel's own validation, plus: BF16, ``M`` a whole number of the tactic's token
    tiles (the polled tiles have no out-of-bounds rows), ``x`` contiguous with 16-byte rows, the
    owner epilogue (not the reduce-scatter one), whole polling groups and retirement rounds, and
    at most ``MAX_UNITS`` units."""
    if not HAS_CUTE_DSL:
        return "the CuTe DSL is not importable"
    try:
        _sk.validate_tactic(tactic, *_sk._validate_runtime_tensors(x, w, None, out))
    except ValueError as error:
        return str(error)
    if x.dtype != torch.bfloat16:
        return f"x is {x.dtype}, not BF16"
    rows = int(x.shape[0])
    if rows % int(tactic.mma_n):
        return f"{rows} rows are not whole token tiles of the tactic's mma_n {tactic.mma_n}"
    if not x.is_contiguous() or (x.stride(0) * _ptx.ELEMENT_BYTES) % 16:
        return "x is not contiguous with 16-byte rows"
    if _sk._rs_for(int(tactic.split_k)):
        return f"split_k {tactic.split_k} takes the reduce-scatter epilogue (not ported)"
    try:
        group = _ptx.poll_group(int(tactic.mma_n), int(tactic.ab_stages), int(poll_warps))
    except ValueError as error:
        return str(error)
    tiles = int(x.shape[1]) // _ptx.CTA_K // int(tactic.split_k)
    if tiles % group:
        return f"{tiles} k-tiles per CTA are not whole polling groups of {group}"
    if tiles % 2:
        return f"{tiles} k-tiles per CTA: the unit retirement stores whole 32-chunk rounds"
    if int(tactic.split_k) * (rows // int(tactic.mma_n)) > _ptx.MAX_UNITS:
        return f"{tactic.split_k} slices x {rows // int(tactic.mma_n)} tiles exceed {_ptx.MAX_UNITS} units"
    return None


def run_lg_splitk(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
    tactic,
    arena: LgArena,
    deadline_ns: int = DEFAULT_DEADLINE_NS,
    tag: Optional[int] = None,
    expect: Optional[int] = None,
    mirror: bool = False,
    gen_check: bool = True,
    poll_warps: Optional[int] = None,
    backoff_ns: int = 0,
) -> torch.Tensor:
    """``x [M, K] @ w [K, N] -> out [M, N]`` (``w`` the weight's transpose view, as
    ``splitk_gemm.run_splitk_dense`` takes it) with ``x``'s k-tiles polled from ``arena``.

    The bits are ``splitk_gemm.run_splitk_dense(x, w, None, out, True, tactic)``'s. Without
    ``mirror`` ``x`` must be the arena's tensor; with ``mirror`` ``x`` is the producer's own output
    of the arena's layout, and the producer also stores the same values into the arena (the
    fallback reads ``x``). ``tag`` is this call's generation, ``expect`` the previous call's on the
    arena (both default to ``arena.rest_tag``). ``deadline_ns <= 0`` never falls back; refused
    with the generation check unless ``expect == tag``.

    Raises:
        ValueError: when ``serves`` refuses, ``x`` is not the arena (or, with ``mirror``, not of
            its layout), the arena's slices / tiles are not the call's, a tag is out of range, or
            the deadline is 0 with a generation check that may wait.
    """
    global LAUNCHES
    poll_warps = DEFAULT_POLL_WARPS if poll_warps is None else int(poll_warps)
    why = serves(x, w, out, tactic, poll_warps)
    if why is not None:
        raise ValueError(f"lg_splitk: {why}")
    if mirror:
        if not arena.mirrors(x):
            raise ValueError("lg_splitk: a mirrored x must have the arena's layout (not be it)")
    elif not arena.holds(x):
        raise ValueError("lg_splitk: x must be the arena's tensor")
    tiles = int(x.shape[0]) // int(tactic.mma_n)
    if int(tactic.split_k) != arena.slices or tiles != arena.tiles:
        raise ValueError(
            f"lg_splitk: the call reads {tactic.split_k} slices x {tiles} tiles, the arena has "
            f"{arena.slices} x {arena.tiles}"
        )
    tag = arena.rest_tag if tag is None else int(tag)
    expect = arena.rest_tag if expect is None else int(expect)
    if not (1 <= tag < 2**31 and 1 <= expect < 2**31):
        raise ValueError(f"lg_splitk: tags must be positive int32, got {tag}, {expect}")
    if int(deadline_ns) <= 0 and gen_check and expect != tag:
        raise ValueError(
            "lg_splitk: a deadline of 0 never falls back, and a generation check waiting for "
            f"tag {expect} (this call's is {tag}) would spin forever if it never came"
        )
    a_c, b_c, c_c, _, leading_dims = _sk._to_cute_swap(x, w, out, None)
    mailbox_v4 = _mailbox_v4(int(x.shape[0]), int(tactic.split_k))
    kernel = compiled(
        x.dtype, tactic, leading_dims, gen_check, poll_warps, mailbox_v4, int(backoff_ns)
    )
    n = int(out.shape[1])
    unit_ctas = -(-n // tactic.mma_m)
    row_bytes = int(x.stride(0)) * _ptx.ELEMENT_BYTES
    delta = int(x.data_ptr()) - int(arena.tensor.data_ptr())
    deadline = int(deadline_ns) if int(deadline_ns) > 0 else (1 << 62)
    stream = cuda_driver.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    kernel(
        a_c,
        b_c,
        c_c,
        int(arena.tensor.data_ptr()),
        delta,
        row_bytes,
        int(arena.control.data_ptr()),
        arena.fallback_address,
        deadline,
        unit_ctas,
        tag,
        expect,
        stream,
    )
    LAUNCHES += 1
    return out


def stock(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, tactic) -> torch.Tensor:
    """The served split-K call this kernel's bits are checked against (PDL launch)."""
    return _sk.run_splitk_dense(x, w, None, out, True, tactic)


def tactic_of(kernel: str):
    """``"splitk:m,n,s,stages"`` -> ``SplitKTactic``."""
    family, _, fields = kernel.partition(":")
    if family != "splitk":
        raise ValueError(f"lg_splitk: {kernel!r} is not a split-K kernel")
    return _sk.SplitKTactic(*(int(v) for v in fields.split(",")))


def site_tactics(tp_size: int = 8) -> Dict[str, Tuple[Tuple[int, str], ...]]:
    """The served 9..32-row ``(last row count, kernel)`` steps of each site."""
    from ...kernels.dense_gemms import dispatcher as _dispatcher

    return _dispatcher.wide_steps(tp_size)
