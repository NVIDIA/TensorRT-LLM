# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Task definitions for the throughput 2CTA MLA decode TS kernel.

Based on
``tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_2cta/tasks.py``; unused branches removed
(the dense softmax schedule: the K3.1 decode is sparse).

The selected graph depends on dtype and scheduler policy. BF16 uses a 12-warp
combined TMA/MMA graph, with an additional scheduler task under CLC. FP8 uses a
16-warp split K/V and QK/PV graph with two softmax groups.

Domain semantics: domain = k_tile_count (total k-tiles to process).
Tasks with domain_start=1 handle the first k-tile in HEAD, then LOOP
covers k-tiles 1..N-1, and TAIL handles cleanup.

Stagger pattern (K loads 1 ahead of V, QK MMA 1 ahead of PV MMA):
  k-tile 0: load K[0], QK MMA -> S[0]
  k-tile 1: load K[1]+V[0], PV MMA(P[0],V[0])->O[0], QK MMA->S[1]
  ...
  k-tile N-1: load K[N-1]+V[N-2], PV MMA(P[N-2],V[N-2])->O[N-2], QK MMA->S[N-1]
  tail: load V[N-1], PV MMA(P[N-1],V[N-1])->O[N-1]

The BF16 register roles are:
  LoadTmaTask     (warp 9,    1 warp,   96 regs; owns page-ID window)
  MmaTask         (warp 8,    1 warp,   96 regs)
  SoftmaxTask     (warps 0-3, 4 warps, 192 regs)
  CorrectionTask  (warps 4-7, 4 warps, 208 regs)
  PaddingTask     (warps 10-11,          96 regs; non-CLC alignment)
  SchedulerTask   (warp 11,              96 regs; CLC only)

It also holds the compatibility helper for the CuTe DSL 4.7/4.8 task execution APIs.
"""

import inspect
from collections.abc import Callable

import cutlass
from cutlass import cute
from cutlass.experimental.task_scheduling.memory import ResourceContext
from cutlass.experimental.task_scheduling.schedule_builder import domain_loop, schedule
from cutlass.experimental.task_scheduling.task import Task

from .mla_decode_2cta_resources import (
    GmemOResource,
    MlaWorkQueue,
    SmemKResource,
    SmemPResource,
    SmemQResource,
    SmemVResource,
    TmemCorrResource,
    TmemOResource,
    TmemSResource,
)
from .mla_decode_helpers_schedule import (
    captured_loop_bounds,
    staged_pv_mma_v_tile_per_n,
    staged_qk_mma_k_tile,
    work_queue_tail,
)

_TASK_ACCEPTS_CONTEXT = "context" in inspect.signature(Task._run_task_body_impl).parameters


def task_context_kwargs(
    context: ResourceContext | None,
) -> dict[str, ResourceContext | None]:
    """Supply the explicit execution context only on the 4.7 task API.

    DSL 4.8 removed this argument from task execution and stage-info methods:
    ``init_variables`` now stores it on the task. Keep the 4.7 propagation,
    including ``None``, while allowing both APIs to call PrimTS overrides.
    Signature detection happens once at import; this helper runs at trace time.
    """
    return {"context": context} if _TASK_ACCEPTS_CONTEXT else {}


def _capture_clc_work_tile_body(
    work_queue,
    body: Callable[..., None],
    non_skippable_prelude: Callable[[], object] | None = None,
    *,
    use_clc_dynamic: bool = False,
) -> None:
    """Capture one MLA tile with data work skippable and WQ progress mandatory.

    Pure register-state initializers may run in ``non_skippable_prelude`` so
    values they create dominate the separately guarded HEAD, LOOP, and TAIL
    regions emitted by the stock skipped-tile executor.  The prelude must not
    issue memory operations or advance pipeline state.
    """

    def run_body():
        prelude_state = non_skippable_prelude() if non_skippable_prelude is not None else None
        if non_skippable_prelude is None:
            body()
        else:
            body(prelude_state)

    run_body()
    work_queue_tail(work_queue, advance_label="advance_tile")


class MlaTask(Task):
    """Task subclass that recomputes MLA k-domain per persistent work tile."""

    @cute.jit
    def _run_one_mla_work_tile(
        self,
        work_tile,
        context: ResourceContext | None = None,
    ) -> None:
        """Run a persistent work tile using the cached split-KV domain."""

        # WorkQueue decomposes the MLA persistent tile and caches the K-domain.
        # Keep task bodies on that cached value so page-offset/TMA/MMA/softmax
        # paths do not each rebuild the same split-KV arithmetic.
        self.domain = work_tile.k_tile_count
        self._run_task_body_impl(work_tile, **task_context_kwargs(context))

    @cute.jit
    def _drain_mla_work_tile_tails(self) -> None:
        """Drain producer tails after a persistent work-tile body completes."""

        for resource in self.dst_resources:
            if cutlass.const_expr(
                resource.pipeline_config is not None
                and resource is not self.work_queue
                and not self._is_fork_secondary(resource)
            ):
                self._producer_tail(resource)
        if cutlass.const_expr(
            self.work_queue is not None
            and self.work_queue in self.dst_resources
            and self.work_queue.pipeline_config is not None
        ):
            self.work_queue.producer_tail()

    @cute.jit
    def _run_task_body_persistent(
        self,
        context: ResourceContext | None = None,
    ) -> None:
        """Schedule one or more persistent work tiles for this task instance."""

        params = self.work_queue.tile_sched_params

        if cutlass.const_expr(not params.is_persistent):
            work_tile = self.work_queue._work_tile_from_block_idx(cute.arch.block_idx())
            self.work_queue._set_consumer_var_from_ts("work_tile", work_tile)
            self._run_pre_work_loop_entries(work_tile, **task_context_kwargs(context))
            # Runtime K/Q metadata can make a statically launched split empty.
            # Keep the CTA on the ordinary initialized-pipeline path, but skip
            # its captured HEAD/LOOP/TAIL data work when the domain is zero.
            if work_tile.k_tile_count > cutlass.Int32(0):
                self._run_one_mla_work_tile(work_tile, context)
            self._run_post_work_loop_entries(work_tile, **task_context_kwargs(context))
            self._drain_mla_work_tile_tails()
            return

        current_work_linear_idx = cute.arch.block_idx()[0]
        num_blocks = (
            params.cluster_shape_mnk[0]
            * params.problem_shape_s
            * params.problem_shape_b
            * params.split_kv
        )
        work_tile = self.work_queue._work_tile_from_linear_idx(current_work_linear_idx)
        self.work_queue._set_consumer_var_from_ts("work_tile", work_tile)

        self._run_pre_work_loop_entries(work_tile, **task_context_kwargs(context))
        while current_work_linear_idx < num_blocks:
            work_tile.update_from(
                self.work_queue._work_tile_from_linear_idx(current_work_linear_idx)
            )
            self.work_queue._set_consumer_var_from_ts("work_tile", work_tile)

            # Variable K and causal Q visibility can leave individual logical
            # splits empty. Skip them and continue grid-striding rather than
            # running a captured HEAD/TAIL sequence with domain zero.
            if work_tile.k_tile_count > cutlass.Int32(0):
                self._run_one_mla_work_tile(work_tile, context)

            # Each warp branch advances from the same scalar tile id, keeping
            # the persistent loop state compact across task bodies.
            current_work_linear_idx += cute.size(cute.arch.grid_dim())
            # self.dummy keeps the captured persistent loop body live even when
            # a specialized task instance has no visible local result.
            self.dummy = True
        work_tile.update_from(self.work_queue._work_tile_from_linear_idx(current_work_linear_idx))
        self.work_queue._set_consumer_var_from_ts("work_tile", work_tile)
        self._run_post_work_loop_entries(work_tile, **task_context_kwargs(context))
        self._drain_mla_work_tile_tails()


def create_sparse_offsets_task(page_offsets, work_queue=None, task_class=MlaTask, **task_kwargs):
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 0)

    @schedule
    def offsets_schedule(page_offsets, work_queue=None):
        page_offsets.init_sparse_state()
        page_offsets.bind_sparse_request()
        with domain_loop(loop_start, loop_end, loop_step):
            page_offsets.acquire()
            page_offsets.load_sparse_offsets()
            page_offsets.commit()
        work_queue_tail(work_queue, advance_label="advance_tile")

    result = (
        offsets_schedule(page_offsets)
        if work_queue is None
        else offsets_schedule(page_offsets, work_queue)
    )
    return task_class(
        src_resources=[work_queue] if work_queue is not None else [],
        dst_resources=[page_offsets],
        warp_idx=9,
        num_warps=1,
        schedule=result,
        name="SparseOffsetsTask",
        **task_kwargs,
    )


def create_load_k_task(
    smem_q: SmemQResource,
    smem_k: SmemKResource,
    page_offsets=None,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    **task_kwargs,
) -> Task:
    """Create the FP8 Q/K TMA task on its configured loader warps.

    Q is loaded once before the loop. K uses one whole-tile pipeline stage per
    logical K tile. Sparse indices are prefetched through the shared offset
    ring (``page_offsets``).
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 0)

    def load_k_body(smem_q, smem_k, page_offsets=None, work_queue=None):
        """Load Q once and then publish one K stage per loop tile."""
        smem_q.init_load_state()
        smem_k.init_load_state()
        smem_q.acquire()
        smem_q.tma_load()
        smem_q.commit()

        with domain_loop(loop_start, loop_end, loop_step):
            if page_offsets is not None:
                page_offsets.wait()
                page_stage = page_offsets.sparse_stage()
                smem_k.acquire()
                smem_k.tma_load_shared(page_offset_stage=page_stage)
                smem_k.commit()
                # The last quad has been copied into TMA operands. Only now
                # may the metadata producer overwrite this shared ring slot.
                page_offsets.release()
        work_queue_tail(work_queue, advance_label="advance_tile")

    @schedule
    def load_k_schedule(smem_q, smem_k, page_offsets, work_queue=None):
        load_k_body(smem_q, smem_k, page_offsets, work_queue)

    schedule_result = (
        load_k_schedule(smem_q, smem_k, page_offsets)
        if work_queue is None
        else load_k_schedule(smem_q, smem_k, page_offsets, work_queue)
    )

    src = [page_offsets] + ([work_queue] if work_queue is not None else [])
    return task_class(
        src_resources=src,
        dst_resources=[smem_q, smem_k],
        warp_idx=smem_k.cfg.load_k_warp_id,
        num_warps=smem_k.cfg.load_num_warps,
        schedule=schedule_result,
        name="LoadKTask",
        **task_kwargs,
    )


def create_load_v_task(
    smem_v: SmemVResource,
    page_offsets=None,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    **task_kwargs,
) -> Task:
    """Create the FP8 V TMA task on its configured loader warps.

    Sparse indices come from the shared offset ring (``page_offsets``).
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 0)

    def load_v_body(smem_v, page_offsets=None, work_queue=None):
        """Publish one V stage per logical K tile."""
        smem_v.init_load_state()
        with domain_loop(loop_start, loop_end, loop_step):
            if page_offsets is not None:
                page_offsets.wait()
                page_stage = page_offsets.sparse_stage()
                smem_v.acquire()
                smem_v.tma_load_shared(page_offset_stage=page_stage)
                smem_v.commit()
                # Keep the ring slot protected through every shared read.
                page_offsets.release()
        work_queue_tail(work_queue, advance_label="advance_tile")

    @schedule
    def load_v_schedule(smem_v, page_offsets, work_queue=None):
        load_v_body(smem_v, page_offsets, work_queue)

    schedule_result = (
        load_v_schedule(smem_v, page_offsets)
        if work_queue is None
        else load_v_schedule(smem_v, page_offsets, work_queue)
    )

    src = [page_offsets] + ([work_queue] if work_queue is not None else [])
    return task_class(
        src_resources=src,
        dst_resources=[smem_v],
        warp_idx=smem_v.cfg.load_v_warp_id,
        num_warps=smem_v.cfg.load_num_warps,
        schedule=schedule_result,
        name="LoadVTask",
        **task_kwargs,
    )


def create_mma_qk_direct_task(
    smem_q: SmemQResource,
    smem_k: SmemKResource,
    tmem_s: TmemSResource,
    iterations_qk: int = 5,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    **task_kwargs,
) -> Task:
    """Create the FP8 QK task with one K stage per domain-loop iteration."""
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 0)

    @schedule
    def mma_qk_direct_schedule(smem_q, smem_k, tmem_s, work_queue=None):
        """Consume Q once and publish one S stage per K tile."""
        smem_q.wait()
        smem_q.q_desc()
        with domain_loop(loop_start, loop_end, loop_step):
            staged_qk_mma_k_tile(smem_k, tmem_s, iterations_qk)
        smem_q.release()
        work_queue_tail(work_queue, advance_label="advance_tile")

    schedule_result = (
        mma_qk_direct_schedule(smem_q, smem_k, tmem_s)
        if work_queue is None
        else mma_qk_direct_schedule(smem_q, smem_k, tmem_s, work_queue)
    )

    src = [smem_q, smem_k]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[tmem_s],
        warp_idx=8,
        num_warps=1,
        schedule=schedule_result,
        name="MmaQkTask",
        run_only_on_cta_id=0,
        **task_kwargs,
    )


def create_mma_pv_direct_task(
    smem_v: SmemVResource,
    smem_p: SmemPResource,
    tmem_o: TmemOResource,
    iterations_pv: int = 4,
    iterations_pv_k: int = 2,
    iterations_pv_n: int = 2,
    per_n_o_pipeline: bool = False,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    **task_kwargs,
) -> Task:
    """Create the FP8 PV task with one V stage per domain-loop iteration."""
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 0)

    @schedule
    def mma_pv_direct_schedule(smem_v, smem_p, tmem_o, work_queue=None):
        """Consume one P/V pair and publish one O stage per K tile."""
        with domain_loop(loop_start, loop_end, loop_step):
            if per_n_o_pipeline:
                staged_pv_mma_v_tile_per_n(
                    smem_v,
                    smem_p,
                    tmem_o,
                    iterations_pv_k=iterations_pv_k,
                    iterations_pv_n=iterations_pv_n,
                )
        work_queue_tail(work_queue, advance_label="advance_tile")

    schedule_result = (
        mma_pv_direct_schedule(smem_v, smem_p, tmem_o)
        if work_queue is None
        else mma_pv_direct_schedule(smem_v, smem_p, tmem_o, work_queue)
    )

    src = [smem_v, smem_p]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[tmem_o],
        warp_idx=11,
        num_warps=1,
        schedule=schedule_result,
        name="MmaPvTask",
        run_only_on_cta_id=0,
        **task_kwargs,
    )


def create_softmax_task(
    tmem_s: TmemSResource,
    tmem_corr: TmemCorrResource,
    smem_p: SmemPResource,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    page_offsets=None,
    warp_idx: int = 0,
    name: str = "SoftmaxTask",
    softmax_group_id: int = 0,
    **task_kwargs,
) -> Task:
    """Create the Softmax task (warps 0-3, 4 warps, 192 regs).

    domain_start=0: processes all k_tile_count S tiles.

    LOOP: consume S, compute softmax, produce correction factors + P.
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 0)
    use_clc_dynamic = bool(work_queue is not None and work_queue.use_clc_dynamic)

    def softmax_prelude(tmem_s):
        """Create softmax register arrays before the dynamic skip guard."""
        init_softmax_state = (
            tmem_s.init_softmax_state_odd if softmax_group_id == 1 else tmem_s.init_softmax_state
        )
        return init_softmax_state()

    def softmax_body(tmem_s, tmem_corr, smem_p, softmax_state, page_offsets=None):
        """Consume S tiles, materialize P, and publish correction factors."""
        load_s = (
            tmem_s.load_s_odd
            if softmax_group_id == 1
            else tmem_s.load_s_cached
            if page_offsets is not None
            else tmem_s.load_s
        )
        finish_softmax = (
            tmem_s.finish_softmax_odd if softmax_group_id == 1 else tmem_s.finish_softmax
        )
        finish_row_sum = (
            tmem_s.finish_row_sum_odd if softmax_group_id == 1 else tmem_s.finish_row_sum
        )
        store_corr = tmem_corr.store_corr_odd if softmax_group_id == 1 else tmem_corr.store_corr
        store_p = smem_p.store_p_odd if softmax_group_id == 1 else smem_p.store_p
        (
            qk_acc_regs,
            row_max,
            row_sum,
            row_sum_out,
            row_max_new,
            correction_factor_out,
            no_correction_out,
        ) = softmax_state

        tmem_s.bind_sparse_request()
        with domain_loop(loop_start, loop_end, loop_step):
            mask_kwargs = {}
            if page_offsets is not None:
                page_offsets.wait()
                mask_low, mask_high = page_offsets.read_sparse_mask()
                page_offsets.release()
                mask_kwargs = {
                    "cached_mask_low": mask_low,
                    "cached_mask_high": mask_high,
                }
            tmem_s.wait()
            (
                qk_acc_regs,
                row_max,
                row_sum,
                row_sum_out,
                row_max_new,
                correction_factor_out,
                no_correction_out,
            ) = load_s(
                qk_acc_regs=qk_acc_regs,
                row_max=row_max,
                row_sum=row_sum,
                row_sum_out=row_sum_out,
                row_max_new=row_max_new,
                correction_factor_out=correction_factor_out,
                no_correction_out=no_correction_out,
                **mask_kwargs,
            )
            tmem_s.release()
            (
                qk_acc_regs,
                row_max,
                row_sum,
                row_sum_out,
                row_max_new,
                correction_factor_out,
                no_correction_out,
            ) = finish_softmax(
                qk_acc_regs=qk_acc_regs,
                row_max=row_max,
                row_sum=row_sum,
                row_sum_out=row_sum_out,
                row_max_new=row_max_new,
                correction_factor_out=correction_factor_out,
                no_correction_out=no_correction_out,
            )
            smem_p.acquire()
            store_p(qk_acc_regs=qk_acc_regs)
            smem_p.commit()
            row_sum, row_sum_out = finish_row_sum(
                qk_acc_regs=qk_acc_regs,
                row_sum=row_sum,
                correction_factor_out=correction_factor_out,
            )
            tmem_corr.acquire()
            store_corr(
                row_sum_out=row_sum_out,
                row_max_new=row_max_new,
                correction_factor_out=correction_factor_out,
                no_correction_out=no_correction_out,
            )
            tmem_corr.commit()

    def capture_softmax(tmem_s, tmem_corr, smem_p, work_queue, page_offsets):
        _capture_clc_work_tile_body(
            work_queue,
            lambda softmax_state: softmax_body(
                tmem_s, tmem_corr, smem_p, softmax_state, page_offsets
            ),
            lambda: softmax_prelude(tmem_s),
            use_clc_dynamic=use_clc_dynamic,
        )

    @schedule
    def sparse_softmax_schedule(tmem_s, tmem_corr, smem_p, page_offsets, work_queue=None):
        capture_softmax(tmem_s, tmem_corr, smem_p, work_queue, page_offsets)

    schedule_result = (
        sparse_softmax_schedule(tmem_s, tmem_corr, smem_p, page_offsets)
        if work_queue is None
        else sparse_softmax_schedule(tmem_s, tmem_corr, smem_p, page_offsets, work_queue)
    )

    src = [tmem_s]
    if page_offsets is not None:
        src.append(page_offsets)
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[tmem_corr, smem_p],
        warp_idx=warp_idx,
        num_warps=4,
        schedule=schedule_result,
        name=name,
        **task_kwargs,
    )


def create_correction_task(
    tmem_corr: TmemCorrResource,
    tmem_o: TmemOResource,
    gmem_o: GmemOResource,
    iterations_pv_n: int = 1,
    per_n_o_pipeline: bool = False,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    **task_kwargs,
) -> Task:
    """Create the Correction task (warps 4-7, 4 warps, 208 regs).

    domain_start=1: HEAD handles initial correction (no O yet), LOOP
    handles correction+O pairs, TAIL handles final O + epilogue.

    TMEM visibility is ensured by kernel-level named barrier sync before
    task_manager.run(), so no o_init pipeline is needed here.

    HEAD: consume Corr[0] (initial max/sum, no O rescaling).
    LOOP: consume Corr[n] + O[n-1], rescale accumulated O.
    TAIL: consume O[last], final epilogue store O + LSE to GMEM.
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 1)
    use_clc_dynamic = bool(work_queue is not None and work_queue.use_clc_dynamic)
    bool(gmem_o.cfg.use_tma_output)

    def correction_prelude(tmem_corr):
        """Create correction and epilogue register state before skip guards."""
        return tmem_corr.init_load_state()

    def correction_body(tmem_corr, tmem_o, gmem_o, correction_state):
        """Apply online-softmax correction and store the final O/LSE result."""
        row_sum, row_max, correction_factor, no_correction = correction_state

        # HEAD: consume the first correction factors. There is no O tile yet.
        tmem_corr.wait()
        row_sum, row_max, correction_factor, no_correction = tmem_corr.load_corr()
        tmem_corr.release()

        with domain_loop(loop_start, loop_end, loop_step):
            # LOOP: consume Corr[n] and rescale O[n-1].
            tmem_corr.wait()
            _row_sum, _row_max, correction_factor, no_correction = tmem_corr.load_corr()
            tmem_corr.release()
            if per_n_o_pipeline:
                for iter_n in range(iterations_pv_n):
                    tmem_o.wait()
                    tmem_o.rescale_o_slice(
                        correction_factor=correction_factor,
                        no_correction=no_correction,
                        iter_n=iter_n,
                    )
                    tmem_o.release()

        # TAIL: consume final O. Do not call rescale_o here; the correction was
        # already applied in LOOP and the last loop correction value would be
        # stale for a second application. epilogue_store also writes LSE.
        if per_n_o_pipeline:
            epilogue_row_sum, epilogue_row_max = tmem_corr.prepare_epilogue_slice_store()
            for iter_n in range(iterations_pv_n):
                tmem_o.wait()
                gmem_o.epilogue_store_slice(
                    row_sum=epilogue_row_sum,
                    row_max=epilogue_row_max,
                    iter_n=iter_n,
                )
                tmem_o.release()

    @schedule
    def correction_schedule(tmem_corr, tmem_o, gmem_o, work_queue=None):
        """Capture one active correction tile and unconditional queue progress."""

        _capture_clc_work_tile_body(
            work_queue,
            lambda correction_state: correction_body(
                tmem_corr,
                tmem_o,
                gmem_o,
                correction_state,
            ),
            lambda: correction_prelude(tmem_corr),
            use_clc_dynamic=use_clc_dynamic,
        )

    captured_schedule = (
        correction_schedule(tmem_corr, tmem_o, gmem_o)
        if work_queue is None
        else correction_schedule(tmem_corr, tmem_o, gmem_o, work_queue)
    )

    src = [tmem_corr, tmem_o]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[gmem_o],
        warp_idx=4,
        num_warps=4,
        schedule=captured_schedule,
        name="CorrectionTask",
        **task_kwargs,
    )


def create_padding_task(
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    warp_idx: int = 11,
    num_warps: int = 1,
    **task_kwargs,
) -> Task:
    """Create a padding task for unused warps in producer warpgroup 2.

    Empty task for warp-group alignment.
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 0)
    use_clc_dynamic = bool(work_queue is not None and work_queue.use_clc_dynamic)

    @schedule
    def padding_schedule(work_queue=None):
        """Reserve unused producer warps without issuing kernel work."""

        def padding_body():
            with domain_loop(loop_start, loop_end, loop_step):
                pass

        _capture_clc_work_tile_body(
            work_queue,
            padding_body,
            use_clc_dynamic=use_clc_dynamic,
        )

    captured_schedule = padding_schedule() if work_queue is None else padding_schedule(work_queue)
    src = [work_queue] if work_queue is not None else []
    return task_class(
        src_resources=src,
        dst_resources=[],
        warp_idx=warp_idx,
        num_warps=num_warps,
        schedule=captured_schedule,
        name="PaddingTask",
        **task_kwargs,
    )
