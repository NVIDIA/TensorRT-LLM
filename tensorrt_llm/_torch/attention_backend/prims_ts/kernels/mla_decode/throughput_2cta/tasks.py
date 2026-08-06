# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 by FlashInfer team.
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
"""

from collections.abc import Callable

import cutlass
import cutlass.cute as cute

from cutlass.experimental.task_scheduling.memory import ResourceContext
from cutlass.experimental.task_scheduling.schedule_builder import (
    domain_loop,
    schedule,
    work_tile_loop,
)
from cutlass.experimental.task_scheduling.resources import StageInfo
from cutlass.experimental.task_scheduling.task import Task

from .resources import (
    MlaWorkQueue,
    WorkThrottleBarrierResource,
    PageOffsetWindowResource,
    SmemQResource,
    SmemKResource,
    SmemKVResource,
    SmemVResource,
    SmemPResource,
    TmemSResource,
    TmemCorrResource,
    TmemOResource,
    GmemOResource,
)
from ..helpers.schedule import (
    captured_loop_bounds,
    staged_kv_tma_load,
    staged_pv_mma,
    staged_pv_mma_v_tile,
    staged_pv_mma_v_tile_per_n,
    staged_qk_mma,
    staged_qk_mma_k_tile,
    work_queue_tail,
)


def _fixed_lane_work_tile_bounds(fixed_lane, cumulative_k_parity, actual_domain):
    """Model fixed-lane work-tile bounds for CPU contract tests.

    Generated code keeps this arithmetic inline because returning staged DSL
    values through a plain-Python tuple breaks staged-frontend state threading.
    """

    mapped_start = (fixed_lane + cumulative_k_parity) % 2
    lane_iterations = (actual_domain - mapped_start + 1) // 2
    fixed_loop_end = fixed_lane + lane_iterations * 2
    return mapped_start, fixed_loop_end


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
        prelude_state = (
            non_skippable_prelude() if non_skippable_prelude is not None else None
        )
        if non_skippable_prelude is None:
            body()
        else:
            body(prelude_state)

    if work_queue is not None and use_clc_dynamic:
        with work_tile_loop(
            work_queue,
            skip_if=MlaWorkQueue.skip_work_tile_if,
        ) as work_tiles:
            prelude_state = (
                non_skippable_prelude() if non_skippable_prelude is not None else None
            )
            with work_tiles.skippable():
                if non_skippable_prelude is None:
                    body()
                else:
                    body(prelude_state)
            work_queue_tail(work_queue, advance_label="advance_tile")
        return

    run_body()
    work_queue_tail(work_queue, advance_label="advance_tile")


class MlaClcTask(Task):
    """Stock Task persistent loop with an MLA-specific dynamic K domain."""

    @cute.jit
    def get_domain(self, tile_coord):
        """Recompute the loop bound required by the stock Task public API."""

        assert isinstance(self.work_queue, MlaWorkQueue)
        return self.work_queue.k_tile_count_for_tile(tile_coord)


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
        self._run_task_body_impl(work_tile, context=context)

    @cute.jit
    def _drain_mla_work_tile_tails(self) -> None:
        """Drain producer tails after a persistent work-tile body completes."""

        for resource in self.dst_resources:
            if cutlass.const_expr(
                resource.pipeline_config is not None
                and resource is not self.work_queue
                and not self._is_fork_secondary(resource)
            ):
                if cutlass.const_expr(
                    resource.pipeline_config.producer_acquire_interleave_stride > 1
                    or resource.pipeline_config.producer_commit_interleave_stride > 1
                ):
                    # Interleaved producers own lane-specific pipeline states.
                    # The generic producer tail still drains at resource
                    # granularity, so calling it here can wait on a peer lane's
                    # physical stages.  Consumer wait/release drains the live
                    # lane for these score/P resources.
                    pass
                else:
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
            self._run_pre_work_loop_entries(work_tile, context)
            self._run_one_mla_work_tile(work_tile, context)
            self._run_post_work_loop_entries(work_tile, context)
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

        self._run_pre_work_loop_entries(work_tile, context)
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
        work_tile.update_from(
            self.work_queue._work_tile_from_linear_idx(current_work_linear_idx)
        )
        self.work_queue._set_consumer_var_from_ts("work_tile", work_tile)
        self._run_post_work_loop_entries(work_tile, context)
        self._drain_mla_work_tile_tails()


class MlaInterleavedTask(MlaTask):
    """Persistent task that carries its interleave lane across work tiles."""

    def __init__(self, *args, **kwargs) -> None:
        """Create staged parity and index-remapping state for this task."""

        super().__init__(*args, **kwargs)
        self._cumulative_k_parity = cutlass.Int32(0)
        self._mapped_domain_start = cutlass.Int32(self.domain_start)
        self._actual_domain = cutlass.Int32(0)
        self._fixed_loop_end = cutlass.Int32(self.domain_start)

    @cute.jit
    def _run_one_mla_work_tile(
        self,
        work_tile,
        context: ResourceContext | None = None,
    ) -> None:
        """Run one tile by mapping its local offsets onto this fixed lane."""

        fixed_lane = cutlass.Int32(self.domain_start)
        self._actual_domain = work_tile.k_tile_count
        self._mapped_domain_start = (
            fixed_lane + self._cumulative_k_parity
        ) % cutlass.Int32(2)
        lane_iterations = (
            self._actual_domain - self._mapped_domain_start + cutlass.Int32(1)
        ) // cutlass.Int32(2)
        self._fixed_loop_end = fixed_lane + lane_iterations * cutlass.Int32(2)
        self.domain = self._fixed_loop_end
        self._run_task_body_impl(work_tile, context=context)
        self._cumulative_k_parity = (
            self._cumulative_k_parity + self._actual_domain
        ) % cutlass.Int32(2)

    @cute.jit
    def _create_stage_info(
        self,
        resource,
        idx,
        work_tile=None,
        is_producer=None,
        resolved_domain=None,
        label=None,
        schedule_stage=None,
        routing_slot=None,
        context: ResourceContext | None = None,
    ) -> StageInfo:
        """Return base pipeline state with the work tile's actual K offset."""

        base_info = Task._create_stage_info(
            self,
            resource,
            idx,
            work_tile,
            is_producer,
            resolved_domain,
            label,
            schedule_stage,
            routing_slot,
            context=context,
        )
        actual_loop_offset = self._mapped_domain_start + (
            cutlass.Int32(base_info.loop_offset) - cutlass.Int32(self.domain_start)
        )
        return StageInfo(
            loop_offset=actual_loop_offset,
            loop_start=self._mapped_domain_start,
            loop_end=self._actual_domain,
            loop_step=base_info.loop_step,
            stage_idx=base_info.stage_idx,
            label=base_info.label,
            barrier=base_info.barrier,
            work_tile=base_info.work_tile,
            num_active_stages=base_info.num_active_stages,
            context=base_info.context,
            task_cache=base_info.task_cache,
        )


def create_load_tma_task(
    page_offset_window: PageOffsetWindowResource,
    smem_q: SmemQResource,
    smem_kv: SmemKVResource,
    iterations_qk: int = 9,
    iterations_pv: int = 8,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    **task_kwargs,
) -> Task:
    """Create the TMA load task (warp 9, 1 warp, 96 regs).

    domain_start=1: HEAD handles k-tile[0], LOOP handles k-tiles 1..N-1.

    HEAD: consume page offsets[0], load Q, load K[0].
    LOOP: consume page offsets[n], load K[n], load V[n-1].
    TAIL: load V[last].
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 1)
    use_clc_dynamic = bool(work_queue is not None and work_queue.use_clc_dynamic)

    def load_tma_prelude(page_offset_window, smem_q, smem_kv):
        """Create register and descriptor state before any dynamic skip guard."""
        cached_page_state = page_offset_window.init_read_state()
        smem_q.init_load_state()
        smem_kv.init_load_state()
        return cached_page_state

    def load_tma_body(page_offset_window, smem_q, smem_kv, cached_page_state):
        """Load Q once and load K/V tiles with the K-before-V cadence."""
        cached_k_pages, cached_v_pages, cached_next_v_pages, cached_window_page = (
            cached_page_state
        )

        # HEAD: Q is independent of page IDs.  Enqueue it before the TMA warp
        # refreshes its first coalesced 32-entry page-table window.
        smem_q.acquire()
        smem_q.tma_load()
        smem_q.commit()
        cached_k_pages, cached_v_pages, cached_next_v_pages, cached_window_page = (
            page_offset_window.read_page_offset_window(
                cached_k_pages=cached_k_pages,
                cached_v_pages=cached_v_pages,
                cached_next_v_pages=cached_next_v_pages,
                cached_window_page=cached_window_page,
                init_v_cache=True,
            )
        )
        staged_kv_tma_load(
            smem_kv,
            iterations_qk,
            cached_k_pages,
            cached_v_pages,
            cached_next_v_pages,
            is_v=False,
        )

        with domain_loop(loop_start, loop_end, loop_step):
            # LOOP: cache K[n]/V[n] offsets, load K[n], then deferred V[n-1].
            cached_k_pages, cached_v_pages, cached_next_v_pages, cached_window_page = (
                page_offset_window.read_page_offset_window(
                    cached_k_pages=cached_k_pages,
                    cached_v_pages=cached_v_pages,
                    cached_next_v_pages=cached_next_v_pages,
                    cached_window_page=cached_window_page,
                )
            )
            # K sub-tiles then deferred V sub-tiles, each with its own local
            # sub-tile index; ``is_v`` tells the loader which path to take.
            staged_kv_tma_load(
                smem_kv,
                iterations_qk,
                cached_k_pages,
                cached_v_pages,
                cached_next_v_pages,
                is_v=False,
            )
            staged_kv_tma_load(
                smem_kv,
                iterations_pv,
                cached_k_pages,
                cached_v_pages,
                cached_next_v_pages,
                is_v=True,
            )

        # TAIL: V[last] uses the next-V offsets cached by the final wait.
        cached_k_pages, cached_v_pages, cached_next_v_pages = (
            page_offset_window.forward_page_ids(
                cached_k_pages=cached_k_pages,
                cached_v_pages=cached_v_pages,
                cached_next_v_pages=cached_next_v_pages,
            )
        )
        staged_kv_tma_load(
            smem_kv,
            iterations_pv,
            cached_k_pages,
            cached_v_pages,
            cached_next_v_pages,
            is_v=True,
            use_next_v_pages=True,
        )

    @schedule
    def load_tma_schedule(page_offset_window, smem_q, smem_kv, work_queue=None):
        """Capture one active TMA tile and unconditional queue progress."""

        _capture_clc_work_tile_body(
            work_queue,
            lambda cached_page_state: load_tma_body(
                page_offset_window,
                smem_q,
                smem_kv,
                cached_page_state,
            ),
            lambda: load_tma_prelude(page_offset_window, smem_q, smem_kv),
            use_clc_dynamic=use_clc_dynamic,
        )

    if work_queue is None:
        captured_schedule = load_tma_schedule(page_offset_window, smem_q, smem_kv)
    else:
        captured_schedule = load_tma_schedule(
            page_offset_window,
            smem_q,
            smem_kv,
            work_queue,
        )

    src = [page_offset_window]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[smem_q, smem_kv],
        warp_idx=9,
        num_warps=1,
        schedule=captured_schedule,
        name="LoadTmaTask",
        **task_kwargs,
    )


def create_load_k_task(
    smem_q: SmemQResource,
    smem_k: SmemKResource,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    **task_kwargs,
) -> Task:
    """Create the FP8 Q/K TMA task (warp 9).

    Q is loaded once before the loop. K uses one whole-tile pipeline stage per
    logical K tile and reads page offsets directly from GMEM.
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 0)

    @schedule
    def load_k_schedule(smem_q, smem_k, work_queue=None):
        """Load Q once and then publish one K stage per loop tile."""
        smem_q.init_load_state()
        smem_k.init_load_state()
        smem_q.acquire()
        smem_q.tma_load()
        smem_q.commit()

        with domain_loop(loop_start, loop_end, loop_step):
            smem_k.acquire()
            smem_k.tma_load_direct()
            smem_k.commit()
        work_queue_tail(work_queue, advance_label="advance_tile")

    schedule_result = (
        load_k_schedule(smem_q, smem_k)
        if work_queue is None
        else load_k_schedule(smem_q, smem_k, work_queue)
    )

    src = [work_queue] if work_queue is not None else []
    return task_class(
        src_resources=src,
        dst_resources=[smem_q, smem_k],
        warp_idx=9,
        num_warps=1,
        schedule=schedule_result,
        name="LoadKTask",
        **task_kwargs,
    )


def create_load_v_task(
    smem_v: SmemVResource,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    **task_kwargs,
) -> Task:
    """Create the FP8 V TMA task (warp 10)."""
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 0)

    @schedule
    def load_v_schedule(smem_v, work_queue=None):
        """Publish one V stage per logical K tile."""
        smem_v.init_load_state()
        with domain_loop(loop_start, loop_end, loop_step):
            smem_v.acquire()
            smem_v.tma_load_direct()
            smem_v.commit()
        work_queue_tail(work_queue, advance_label="advance_tile")

    schedule_result = (
        load_v_schedule(smem_v)
        if work_queue is None
        else load_v_schedule(smem_v, work_queue)
    )

    src = [work_queue] if work_queue is not None else []
    return task_class(
        src_resources=src,
        dst_resources=[smem_v],
        warp_idx=10,
        num_warps=1,
        schedule=schedule_result,
        name="LoadVTask",
        **task_kwargs,
    )


def create_mma_task(
    smem_q: SmemQResource,
    smem_kv: SmemKVResource,
    smem_p: SmemPResource,
    tmem_s: TmemSResource,
    tmem_o: TmemOResource,
    iterations_qk: int = 9,
    iterations_pv: int = 8,
    work_queue: MlaWorkQueue = None,
    work_throttle: WorkThrottleBarrierResource = None,
    task_class: type = MlaTask,
    **task_kwargs,
) -> Task:
    """Create the MMA task (warp 8, 1 warp, 96 regs).

    domain_start=1: HEAD handles first QK MMA, LOOP handles PV+QK pairs.
    run_only_on_cta_id=0 keeps the schedule on the CTA-pair leader, matching
    the 2CTA UMMA contract.

    HEAD: consume Q, QK MMA for k-tile[0] -> S[0].
    LOOP: PV MMA for k-tile[n-1] -> O[n-1], then QK MMA for k-tile[n] -> S[n].
    TAIL: PV MMA for last k-tile -> O[last], release Q.
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 1)
    use_clc_dynamic = bool(work_queue is not None and work_queue.use_clc_dynamic)

    def mma_body(
        smem_q,
        smem_kv,
        smem_p,
        tmem_s,
        tmem_o,
        work_throttle=None,
    ):
        """Run QK one tile ahead of PV and keep UMMA on the leader CTA."""
        # HEAD: wait Q once and compute S[0] from K[0].
        smem_q.wait()
        if work_throttle is not None:
            # The leader MMA reaching Q proves that this cluster has started
            # the current tile.  Permit the scheduler to prepare one more work
            # ID without relying on CTA-private resource-ownership fields.
            work_throttle.try_acquire()
            work_throttle.acquire()
            work_throttle.commit()
        smem_q.q_desc()
        staged_qk_mma(smem_kv, tmem_s, iterations_qk)

        with domain_loop(loop_start, loop_end, loop_step):
            # LOOP: compute S[n] before PV[n-1]. This gives softmax the QK
            # instruction window to produce P[n-1] before PV consumes it.
            staged_qk_mma(smem_kv, tmem_s, iterations_qk)
            staged_pv_mma(
                smem_kv,
                smem_p,
                tmem_o,
                iterations_pv,
            )

        # TAIL: finish PV[last], then release the Q descriptor stage.
        staged_pv_mma(
            smem_kv,
            smem_p,
            tmem_o,
            iterations_pv,
            is_tail=True,
        )
        smem_q.release()

    @schedule
    def mma_schedule(
        smem_q,
        smem_kv,
        smem_p,
        tmem_s,
        tmem_o,
        work_queue=None,
        work_throttle=None,
    ):
        """Capture one active MMA tile and unconditional queue progress."""

        _capture_clc_work_tile_body(
            work_queue,
            lambda: mma_body(
                smem_q,
                smem_kv,
                smem_p,
                tmem_s,
                tmem_o,
                work_throttle,
            ),
            use_clc_dynamic=use_clc_dynamic,
        )

    if work_queue is None:
        captured_schedule = mma_schedule(smem_q, smem_kv, smem_p, tmem_s, tmem_o)
    elif work_throttle is None:
        captured_schedule = mma_schedule(
            smem_q,
            smem_kv,
            smem_p,
            tmem_s,
            tmem_o,
            work_queue,
        )
    else:
        captured_schedule = mma_schedule(
            smem_q,
            smem_kv,
            smem_p,
            tmem_s,
            tmem_o,
            work_queue,
            work_throttle,
        )

    src = [smem_q, smem_kv, smem_p]
    if work_queue is not None:
        src.append(work_queue)
    dst = [tmem_s, tmem_o]
    if work_throttle is not None:
        dst.append(work_throttle)
    return task_class(
        src_resources=src,
        dst_resources=dst,
        warp_idx=8,
        num_warps=1,
        schedule=captured_schedule,
        name="MmaTask",
        run_only_on_cta_id=0,
        **task_kwargs,
    )


def create_mma_qk_task(
    smem_q: SmemQResource,
    smem_kv: SmemKVResource,
    tmem_s: TmemSResource,
    iterations_qk: int = 9,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    **task_kwargs,
) -> Task:
    """Create the FP8 QK-only MMA task (warp 8).

    QK stays one k-tile ahead of PV. Keeping QK and PV on separate warps avoids
    serializing the two UMMA issue streams while preserving the existing K-before-V
    TMA cadence and one-softmax schedule.
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 1)

    @schedule
    def mma_qk_schedule(smem_q, smem_kv, tmem_s, work_queue=None):
        """Consume Q/K stages and publish S for every k-tile."""
        smem_q.wait()
        smem_q.q_desc()
        staged_qk_mma(smem_kv, tmem_s, iterations_qk)

        with domain_loop(loop_start, loop_end, loop_step):
            staged_qk_mma(smem_kv, tmem_s, iterations_qk)

        smem_q.release()
        work_queue_tail(work_queue, advance_label="advance_tile")

    schedule_result = (
        mma_qk_schedule(smem_q, smem_kv, tmem_s)
        if work_queue is None
        else mma_qk_schedule(smem_q, smem_kv, tmem_s, work_queue)
    )

    src = [smem_q, smem_kv]
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


def create_mma_pv_task(
    smem_kv: SmemKVResource,
    smem_p: SmemPResource,
    tmem_o: TmemOResource,
    iterations_pv: int = 8,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    **task_kwargs,
) -> Task:
    """Create the FP8 PV-only MMA task (warp 11).

    The PV task consumes P[n-1]/V[n-1] while QK produces S[n], then handles the
    final P/V tile in TAIL. This matches the existing correction schedule.
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 1)

    @schedule
    def mma_pv_schedule(smem_kv, smem_p, tmem_o, work_queue=None):
        """Consume delayed V and P stages and accumulate O."""
        with domain_loop(loop_start, loop_end, loop_step):
            staged_pv_mma(smem_kv, smem_p, tmem_o, iterations_pv)

        staged_pv_mma(smem_kv, smem_p, tmem_o, iterations_pv)
        work_queue_tail(work_queue, advance_label="advance_tile")

    schedule_result = (
        mma_pv_schedule(smem_kv, smem_p, tmem_o)
        if work_queue is None
        else mma_pv_schedule(smem_kv, smem_p, tmem_o, work_queue)
    )

    src = [smem_kv, smem_p]
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
            else:
                staged_pv_mma_v_tile(smem_v, smem_p, tmem_o, iterations_pv)
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
            tmem_s.init_softmax_state_odd
            if softmax_group_id == 1
            else tmem_s.init_softmax_state
        )
        return init_softmax_state()

    def softmax_body(tmem_s, tmem_corr, smem_p, softmax_state):
        """Consume S tiles, materialize P, and publish correction factors."""
        load_s = tmem_s.load_s_odd if softmax_group_id == 1 else tmem_s.load_s
        finish_softmax = (
            tmem_s.finish_softmax_odd
            if softmax_group_id == 1
            else tmem_s.finish_softmax
        )
        finish_row_sum = (
            tmem_s.finish_row_sum_odd
            if softmax_group_id == 1
            else tmem_s.finish_row_sum
        )
        store_corr = (
            tmem_corr.store_corr_odd if softmax_group_id == 1 else tmem_corr.store_corr
        )
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

        with domain_loop(loop_start, loop_end, loop_step):
            tmem_s.wait()
            if softmax_group_id == 1:
                (
                    qk_acc_regs,
                    row_max,
                    row_sum,
                    row_sum_out,
                    row_max_new,
                    correction_factor_out,
                    no_correction_out,
                ) = load_s(
                    qk_acc_regs_odd=qk_acc_regs,
                    row_max_odd=row_max,
                    row_sum_odd=row_sum,
                    row_sum_out_odd=row_sum_out,
                    row_max_new_odd=row_max_new,
                    correction_factor_out_odd=correction_factor_out,
                    no_correction_out_odd=no_correction_out,
                )
            else:
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
                )
            tmem_s.release()
            if softmax_group_id == 1:
                (
                    qk_acc_regs,
                    row_max,
                    row_sum,
                    row_sum_out,
                    row_max_new,
                    correction_factor_out,
                    no_correction_out,
                ) = finish_softmax(
                    qk_acc_regs_odd=qk_acc_regs,
                    row_max_odd=row_max,
                    row_sum_odd=row_sum,
                    row_sum_out_odd=row_sum_out,
                    row_max_new_odd=row_max_new,
                    correction_factor_out_odd=correction_factor_out,
                    no_correction_out_odd=no_correction_out,
                )
            else:
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
            if softmax_group_id == 1:
                store_p(qk_acc_regs_odd=qk_acc_regs)
            else:
                store_p(qk_acc_regs=qk_acc_regs)
            smem_p.commit()
            if softmax_group_id == 1:
                row_sum, row_sum_out = finish_row_sum(
                    qk_acc_regs_odd=qk_acc_regs,
                    row_sum_odd=row_sum,
                    correction_factor_out_odd=correction_factor_out,
                )
            else:
                row_sum, row_sum_out = finish_row_sum(
                    qk_acc_regs=qk_acc_regs,
                    row_sum=row_sum,
                    correction_factor_out=correction_factor_out,
                )
            tmem_corr.acquire()
            if softmax_group_id == 1:
                store_corr(
                    row_sum_out_odd=row_sum_out,
                    row_max_new_odd=row_max_new,
                    correction_factor_out_odd=correction_factor_out,
                    no_correction_out_odd=no_correction_out,
                )
            else:
                store_corr(
                    row_sum_out=row_sum_out,
                    row_max_new=row_max_new,
                    correction_factor_out=correction_factor_out,
                    no_correction_out=no_correction_out,
                )
            tmem_corr.commit()

    @schedule
    def softmax_schedule(tmem_s, tmem_corr, smem_p, work_queue=None):
        """Capture one active softmax tile and unconditional queue progress."""

        _capture_clc_work_tile_body(
            work_queue,
            lambda softmax_state: softmax_body(
                tmem_s,
                tmem_corr,
                smem_p,
                softmax_state,
            ),
            lambda: softmax_prelude(tmem_s),
            use_clc_dynamic=use_clc_dynamic,
        )

    schedule_result = (
        softmax_schedule(tmem_s, tmem_corr, smem_p)
        if work_queue is None
        else softmax_schedule(tmem_s, tmem_corr, smem_p, work_queue)
    )

    src = [tmem_s]
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
    task_manager.run(), so o_init pipeline is no longer needed here.

    HEAD: consume Corr[0] (initial max/sum, no O rescaling).
    LOOP: consume Corr[n] + O[n-1], rescale accumulated O.
    TAIL: consume O[last], final epilogue store O + LSE to GMEM.
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 1)
    use_clc_dynamic = bool(work_queue is not None and work_queue.use_clc_dynamic)

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
            row_sum, row_max, correction_factor, no_correction = tmem_corr.load_corr()
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
            else:
                tmem_o.wait()
                tmem_o.rescale_o(
                    correction_factor=correction_factor,
                    no_correction=no_correction,
                )
                tmem_o.release()

        # TAIL: consume final O. Do not call rescale_o here; the correction was
        # already applied in LOOP and the last loop correction value would be
        # stale for a second application. epilogue_store also writes LSE.
        if per_n_o_pipeline:
            epilogue_row_sum, epilogue_row_max = (
                tmem_corr.prepare_epilogue_slice_store()
            )
            for iter_n in range(iterations_pv_n):
                tmem_o.wait()
                gmem_o.epilogue_store_slice(
                    row_sum=epilogue_row_sum,
                    row_max=epilogue_row_max,
                    iter_n=iter_n,
                )
                tmem_o.release()
        else:
            tmem_o.wait()
            gmem_o.epilogue_store()
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

    captured_schedule = (
        padding_schedule() if work_queue is None else padding_schedule(work_queue)
    )
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


def create_scheduler_task(
    work_queue: MlaWorkQueue,
    work_throttle: WorkThrottleBarrierResource = None,
    task_class: type = MlaTask,
    **task_kwargs,
) -> Task:
    """Create the BF16 cluster-wide CLC scheduler on warp 11."""

    @schedule
    def scheduler_schedule(work_queue, work_throttle=None):
        """Fetch and distribute the next logical MLA cluster tile."""

        with work_tile_loop(
            work_queue,
            skip_if=MlaWorkQueue.skip_work_tile_if,
        ) as work_tiles:
            with work_tiles.skippable():
                with domain_loop(0, 0, 1):
                    pass
                if work_throttle is not None:
                    work_throttle.wait()
                    work_throttle.release()
            work_queue.acquire()
            work_queue.fetch_work_tile()
            work_queue.commit()
            work_queue_tail(work_queue, advance_label="advance_tile")

    captured_schedule = (
        scheduler_schedule(work_queue)
        if work_throttle is None
        else scheduler_schedule(work_queue, work_throttle)
    )
    src = [work_queue]
    if work_throttle is not None:
        src.append(work_throttle)
    return task_class(
        src_resources=src,
        dst_resources=[work_queue],
        warp_idx=11,
        num_warps=1,
        schedule=captured_schedule,
        name="SchedulerTask",
        run_only_on_cta_id=0,
        **task_kwargs,
    )
