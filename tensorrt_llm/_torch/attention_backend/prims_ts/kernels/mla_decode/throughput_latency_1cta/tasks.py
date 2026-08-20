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

"""Captured task schedules for the throughput-latency 1CTA MLA TS path."""

import cutlass.cute as cute
from cutlass import Int32
from cutlass.experimental import primitives as prims

from cutlass.experimental.task_scheduling.resources import WorkQueue
from cutlass.experimental.task_scheduling.schedule_builder import domain_loop, schedule
from cutlass.experimental.task_scheduling.task import Task

from ..helpers.constants import (
    WARP_LANES,
    WARP_LANE_MASK,
    WARP_LANE_SHIFT,
    WARPGROUP_WARPS,
)
from ..helpers.schedule import (
    page_offsets_produce,
    staged_kv_load,
    staged_pv_mma,
    staged_pv_mma_tmem_p,
    staged_qk_mma,
    runtime_work_tile_skip_if,
    schedule_token_throttle_head,
    schedule_token_throttle_tail,
    work_tile_schedule_loop,
)
from ..helpers.stage import MlaStage
from ..helpers.tile import (
    runtime_base_seq_len_kv,
    runtime_local_kv_tiles,
    runtime_seq_len_kv_from_task_cache,
)


class MlaDecodeTask(Task):
    """MLA decode task with cached thread state and Q-aware KV domain logic."""

    def __init__(self, **kwargs):
        """Initialize MLA-specific task parameters and cached thread indices."""
        self.cfg = kwargs.pop("cfg", None)
        self.seqlens_kv = kwargs.pop("seqlens_kv", None)
        self.cu_seqlens_q = kwargs.pop("cu_seqlens_q", None)
        self.domain_bias = kwargs.pop("domain_bias", 0)
        super().__init__(**kwargs)
        self._tmem_base_offset = Int32(0)
        self._warp_grp_thread_idx = Int32(0)
        self._local_warp_idx = Int32(0)
        self._lane_idx = Int32(0)
        self._seq_len_kv = Int32(0)

    def init_variables(self, context=None):
        """Initialize per-task thread cache and TMEM base state."""
        super().init_variables(context)
        tidx, _, _ = cute.arch.thread_idx()
        warp_grp_start = Int32(
            (self.warp_idx // WARPGROUP_WARPS) * WARPGROUP_WARPS * WARP_LANES
        )
        self._warp_grp_thread_idx = tidx - warp_grp_start
        self._local_warp_idx = self._warp_grp_thread_idx >> Int32(WARP_LANE_SHIFT)
        self._lane_idx = self._warp_grp_thread_idx & Int32(WARP_LANE_MASK)
        if context is not None and context.tmem_ptr_i32 is not None:
            loaded = Int32(context.tmem_ptr_i32.load())
            self._tmem_base_offset = cute.arch.make_warp_uniform(
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=loaded,
                    offset=0,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                )
            )

    @cute.jit
    def make_task_cache(self):
        """Return the compact tuple of cached per-task thread values."""
        return (
            self._tmem_base_offset,
            self._warp_grp_thread_idx,
            self._local_warp_idx,
            self._lane_idx,
            self._seq_len_kv,
            Int32(0),
            Int32(0),
            Int32(0),
        )

    def get_domain(self, tile_coord):
        """Return the runtime loop domain for the current 1CTA work tile."""
        if self.cfg is None:
            return self.domain

        # Persistent 1CTA coordinates combine batch and head tile in z;
        # non-persistent grids keep z as batch. cache_seqs remains batch-owned.
        if isinstance(tile_coord[2], int):
            if self.cfg.use_persistent_scheduler == 1:
                batch_idx = tile_coord[2] // self.cfg.num_ctas_for_all_heads
            else:
                batch_idx = tile_coord[2]
        else:
            batch_idx = Int32(tile_coord[2])
            if self.cfg.use_persistent_scheduler == 1:
                batch_idx = batch_idx // Int32(self.cfg.num_ctas_for_all_heads)

        # Task-domain ownership must match K/V, softmax, and reduction: dense
        # uses the full runtime K length, while causal uses the largest
        # logical-Q-visible length in this groups_tokens_heads_q CTA. Using the
        # raw batch length for causal can move an all-masked tile into tail and
        # replace a real loop accumulator at a tile boundary.
        cta_idx_q = tile_coord[0]
        if self.cfg.use_persistent_scheduler != 1:
            # Nonpersistent grid X combines (head tile, Q tile, KV split), with
            # KV split innermost. Persistent work queues already expose Q as
            # tile coordinate 0 and must not be decoded a second time.
            cta_idx_q = (tile_coord[0] // Int32(self.cfg.num_ctas_per_seq_kv)) % Int32(
                self.cfg.num_ctas_per_seq_q
            )
        # Cache the raw batch length once per task/work tile. Resource work
        # methods derive their CTA-visible length from this task cache instead
        # of independently reloading cache_seqs throughout the hot loop.
        self._seq_len_kv = runtime_base_seq_len_kv(
            self.cfg,
            self.seqlens_kv,
            batch_idx,
        )
        seq_len_kv = runtime_seq_len_kv_from_task_cache(
            self.cfg,
            self.make_task_cache(),
            cta_idx_q,
            self.cu_seqlens_q,
            batch_idx,
        )
        total_kv_tiles = runtime_local_kv_tiles(self.cfg, seq_len_kv)
        remaining_kv_tiles = cute.math.max(
            total_kv_tiles - Int32(self.cfg.num_insts_kv), Int32(0)
        )
        num_insts_kv = Int32(self.cfg.num_insts_kv)
        loop_domain = (remaining_kv_tiles + num_insts_kv - Int32(1)) // num_insts_kv
        return loop_domain + Int32(self.domain_bias)


def create_throughput_latency_softmax_task_impl(
    tmem_s,
    tmem_softmax_local,
    p_resource,
    tmem_softmax_global,
    work_queue: WorkQueue | None,
    cfg,
    *,
    inst_id: int,
    domain,
    store_stats_before_p: bool = True,
    task_name: str | None = None,
    task_class=MlaDecodeTask,
    **kw,
) -> Task:
    """Create one softmax task for a throughput-latency 1CTA score pipe.

    The task consumes TMEM scores, updates online-softmax state, materializes
    P for PV MMA, and stores per-iteration statistics for correction.
    """

    work_tile_skip_if = runtime_work_tile_skip_if(work_queue)

    @schedule
    def softmax_schedule(
        tmem_s,
        tmem_softmax_local,
        p_resource,
        tmem_softmax_global,
        work_queue=None,
    ):
        """Captured softmax schedule for one of the two interleaved instances."""
        (
            old_max_arr,
            sum_arr,
            new_max_arr,
            local_sum_arr,
            s_arr,
        ) = tmem_s.init_softmax_state()
        if store_stats_before_p:
            p_resource.init_materialize_state()

        def store_stats(inst_idx: int):
            """Store local online-softmax stats for correction."""
            tmem_softmax_local.acquire()
            tmem_softmax_local.store_loop_stats(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
                inst_idx=inst_idx,
            )
            tmem_softmax_local.commit()

        def init_work_tile_state():
            return tmem_s.init_softmax_work_tile_state()

        with work_tile_schedule_loop(
            work_queue,
            skip_if=work_tile_skip_if,
            non_skippable_prelude=(
                init_work_tile_state if work_queue is not None else None
            ),
        ) as (_, work_tile_state):
            if work_queue is not None:
                (
                    old_max_arr,
                    sum_arr,
                    new_max_arr,
                    local_sum_arr,
                    s_arr,
                ) = work_tile_state
            with domain_loop(0, domain, 1) as d:
                tmem_s.wait()
                (
                    old_max_arr,
                    sum_arr,
                    new_max_arr,
                    local_sum_arr,
                    s_arr,
                ) = tmem_s.update_softmax(
                    old_max_arr=old_max_arr,
                    sum_arr=sum_arr,
                    new_max_arr=new_max_arr,
                    local_sum_arr=local_sum_arr,
                    s_arr=s_arr,
                    section=MlaStage.Loop,
                )
                tmem_s.release()
                if store_stats_before_p:
                    store_stats(0)
                # AsyncUmma protects P until PV MMA releases this stage.
                p_resource.acquire()
                p_resource.materialize_p(
                    new_max_arr=new_max_arr,
                    s_arr=s_arr,
                    local_sum_arr=local_sum_arr,
                )
                p_resource.commit()
                tmem_softmax_global.track_global()
                (
                    old_max_arr,
                    sum_arr,
                    new_max_arr,
                    local_sum_arr,
                    s_arr,
                ) = tmem_s.update_softmax_sum(
                    old_max_arr=old_max_arr,
                    sum_arr=sum_arr,
                    new_max_arr=new_max_arr,
                    local_sum_arr=local_sum_arr,
                    s_arr=s_arr,
                )
                if not store_stats_before_p:
                    store_stats(0)
                with d.last_iter():
                    store_stats(1)

    captured_schedule = (
        softmax_schedule(
            tmem_s,
            tmem_softmax_local,
            p_resource,
            tmem_softmax_global,
        )
        if work_queue is None
        else softmax_schedule(
            tmem_s,
            tmem_softmax_local,
            p_resource,
            tmem_softmax_global,
            work_queue,
        )
    )
    src = [tmem_s]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[tmem_softmax_local, p_resource, tmem_softmax_global],
        cfg=cfg,
        warp_idx=cfg.softmax0_warp_idx if inst_id == 0 else cfg.softmax1_warp_idx,
        num_warps=cfg.softmax0_num_warps if inst_id == 0 else cfg.softmax1_num_warps,
        schedule=captured_schedule,
        num_registers=cfg.softmax_regs,
        name=task_name or f"Softmax{inst_id}Task",
        **kw,
    )


def create_keeps_mma_ab_softmax_task(
    tmem_s,
    tmem_softmax_local,
    tmem_p,
    tmem_softmax_global,
    work_queue: WorkQueue | None,
    cfg,
    *,
    domain,
    task_class=MlaDecodeTask,
    **kw,
) -> Task:
    """Create the single-pipe softmax/P task for the keeps-MMA-AB path."""
    return create_throughput_latency_softmax_task_impl(
        tmem_s,
        tmem_softmax_local,
        tmem_p,
        tmem_softmax_global,
        work_queue,
        cfg,
        inst_id=0,
        domain=domain,
        store_stats_before_p=False,
        task_name="SoftmaxTask",
        task_class=task_class,
        **kw,
    )


def create_keeps_mma_ab_correction_task(
    tmem_softmax_local,
    tmem_o,
    tmem_corr,
    work_queue: WorkQueue | None,
    cfg,
    *,
    domain,
    task_class=MlaDecodeTask,
    **kw,
) -> Task:
    """Create the single-pipe correction/store task for keeps-MMA-AB."""

    work_tile_skip_if = runtime_work_tile_skip_if(work_queue)

    @schedule
    def correction_schedule(
        tmem_softmax_local,
        tmem_o,
        tmem_corr,
        work_queue=None,
    ):
        """Captured keeps-MMA-AB correction schedule for loop and tail."""

        def load_initial_local_stats(local_state):
            """Wait for the keeps-MMA-AB pipe and return its initial stats."""
            (
                old_max_arr,
                new_max_arr,
                sum_arr,
                inst0_old_max_arr,
                inst0_new_max_arr,
                inst0_sum_arr,
                inst1_old_max_arr,
                inst1_new_max_arr,
                inst1_sum_arr,
            ) = local_state
            tmem_softmax_local.wait()
            local_state = tmem_softmax_local.load_initial_stats(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
                inst0_old_max_arr=inst0_old_max_arr,
                inst0_new_max_arr=inst0_new_max_arr,
                inst0_sum_arr=inst0_sum_arr,
                inst1_old_max_arr=inst1_old_max_arr,
                inst1_new_max_arr=inst1_new_max_arr,
                inst1_sum_arr=inst1_sum_arr,
            )
            tmem_softmax_local.release()
            return local_state

        def load_loop_local_stats(local_state):
            """Wait for the keeps-MMA-AB pipe and load loop stats."""
            (
                old_max_arr,
                new_max_arr,
                sum_arr,
                inst0_old_max_arr,
                inst0_new_max_arr,
                inst0_sum_arr,
                inst1_old_max_arr,
                inst1_new_max_arr,
                inst1_sum_arr,
            ) = local_state
            tmem_softmax_local.wait()
            local_state = tmem_softmax_local.load_loop_stats(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
                inst0_old_max_arr=inst0_old_max_arr,
                inst0_new_max_arr=inst0_new_max_arr,
                inst0_sum_arr=inst0_sum_arr,
                inst1_old_max_arr=inst1_old_max_arr,
                inst1_new_max_arr=inst1_new_max_arr,
                inst1_sum_arr=inst1_sum_arr,
            )
            tmem_softmax_local.release()
            return local_state

        def load_tail_local_stats(local_state):
            """Wait for the keeps-MMA-AB pipe and load tail stats."""
            (
                old_max_arr,
                new_max_arr,
                sum_arr,
                inst0_old_max_arr,
                inst0_new_max_arr,
                inst0_sum_arr,
                inst1_old_max_arr,
                inst1_new_max_arr,
                inst1_sum_arr,
            ) = local_state
            tmem_softmax_local.wait()
            local_state = tmem_softmax_local.load_tail_stats(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
                inst0_old_max_arr=inst0_old_max_arr,
                inst0_new_max_arr=inst0_new_max_arr,
                inst0_sum_arr=inst0_sum_arr,
                inst1_old_max_arr=inst1_old_max_arr,
                inst1_new_max_arr=inst1_new_max_arr,
                inst1_sum_arr=inst1_sum_arr,
            )
            tmem_softmax_local.release()
            return local_state

        def correct_loop_o(local_state, tail_o_stage_idx_0, tail_o_stage_idx_1):
            """Rescale one keeps-MMA-AB loop O stage."""
            local_state = load_loop_local_stats(local_state)
            (
                old_max_arr,
                new_max_arr,
                sum_arr,
                _inst0_old_max_arr,
                inst0_new_max_arr,
                inst0_sum_arr,
                _inst1_old_max_arr,
                inst1_new_max_arr,
                inst1_sum_arr,
            ) = local_state
            tmem_o.wait()
            (
                o_stage_idx,
                tail_o_stage_idx_0,
                tail_o_stage_idx_1,
            ) = tmem_o.o_stage(
                tail_o_stage_idx_0=tail_o_stage_idx_0,
                tail_o_stage_idx_1=tail_o_stage_idx_1,
                inst_idx=0,
            )
            tmem_corr.correct_loop_and_store(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
                inst0_new_max_arr=inst0_new_max_arr,
                inst0_sum_arr=inst0_sum_arr,
                inst1_new_max_arr=inst1_new_max_arr,
                inst1_sum_arr=inst1_sum_arr,
                o_stage_idx=o_stage_idx,
                tail_o_stage_idx_0=tail_o_stage_idx_0,
                tail_o_stage_idx_1=tail_o_stage_idx_1,
            )
            tmem_o.release()
            return local_state, tail_o_stage_idx_0, tail_o_stage_idx_1

        def correct_tail_o(local_state, tail_o_stage_idx_0, tail_o_stage_idx_1):
            """Normalize and store the final keeps-MMA-AB O stage."""
            local_state = load_tail_local_stats(local_state)
            (
                old_max_arr,
                new_max_arr,
                sum_arr,
                _inst0_old_max_arr,
                inst0_new_max_arr,
                inst0_sum_arr,
                _inst1_old_max_arr,
                inst1_new_max_arr,
                inst1_sum_arr,
            ) = local_state
            tmem_o.wait()
            (
                o_stage_idx,
                tail_o_stage_idx_0,
                tail_o_stage_idx_1,
            ) = tmem_o.o_stage(
                tail_o_stage_idx_0=tail_o_stage_idx_0,
                tail_o_stage_idx_1=tail_o_stage_idx_1,
                inst_idx=0,
                is_tail=True,
            )
            tmem_corr.correct_tail_and_store(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
                inst0_new_max_arr=inst0_new_max_arr,
                inst0_sum_arr=inst0_sum_arr,
                inst1_new_max_arr=inst1_new_max_arr,
                inst1_sum_arr=inst1_sum_arr,
                o_stage_idx=o_stage_idx,
                tail_o_stage_idx_0=tail_o_stage_idx_0,
                tail_o_stage_idx_1=tail_o_stage_idx_1,
            )
            tmem_o.release()
            return local_state, tail_o_stage_idx_0, tail_o_stage_idx_1

        local_state = tmem_softmax_local.init_stats_state()
        _, tail_o_stage_idx_0, tail_o_stage_idx_1 = tmem_o.init_stage_state()
        tmem_corr.init_store_state()

        def init_work_tile_state():
            local_state = tmem_softmax_local.init_stats_work_tile_state()
            _, tail_stage_0, tail_stage_1 = tmem_o.init_stage_work_tile_state()
            return local_state, tail_stage_0, tail_stage_1

        with work_tile_schedule_loop(
            work_queue,
            skip_if=work_tile_skip_if,
            non_skippable_prelude=(
                init_work_tile_state if work_queue is not None else None
            ),
        ) as (_, work_tile_state):
            if work_queue is not None:
                (
                    local_state,
                    tail_o_stage_idx_0,
                    tail_o_stage_idx_1,
                ) = work_tile_state
            local_state = load_initial_local_stats(local_state)
            with domain_loop(0, domain, 1):
                (
                    local_state,
                    tail_o_stage_idx_0,
                    tail_o_stage_idx_1,
                ) = correct_loop_o(local_state, tail_o_stage_idx_0, tail_o_stage_idx_1)

            (
                local_state,
                tail_o_stage_idx_0,
                tail_o_stage_idx_1,
            ) = correct_tail_o(local_state, tail_o_stage_idx_0, tail_o_stage_idx_1)

    schedule_result = (
        correction_schedule(tmem_softmax_local, tmem_o, tmem_corr)
        if work_queue is None
        else correction_schedule(
            tmem_softmax_local,
            tmem_o,
            tmem_corr,
            work_queue,
        )
    )
    src = [tmem_softmax_local, tmem_o]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[tmem_corr],
        cfg=cfg,
        warp_idx=cfg.correction_warp_idx,
        num_warps=cfg.correction_num_warps,
        schedule=schedule_result,
        num_registers=cfg.correction_regs,
        name="CorrectionTask",
        **kw,
    )


def create_throughput_latency_softmax0_task(
    tmem_s0,
    tmem_softmax_local0,
    smem_p0,
    tmem_softmax_global0,
    work_queue: WorkQueue | None,
    cfg,
    *,
    domain,
    task_class=MlaDecodeTask,
    **kw,
) -> Task:
    """Create the softmax/P producer task for score instance 0."""
    return create_throughput_latency_softmax_task_impl(
        tmem_s0,
        tmem_softmax_local0,
        smem_p0,
        tmem_softmax_global0,
        work_queue,
        cfg,
        inst_id=0,
        domain=domain,
        task_class=task_class,
        **kw,
    )


def create_throughput_latency_softmax1_task(
    tmem_s1,
    tmem_softmax_local1,
    smem_p1,
    tmem_softmax_global1,
    work_queue: WorkQueue | None,
    cfg,
    *,
    domain,
    task_class=MlaDecodeTask,
    **kw,
) -> Task:
    """Create the softmax/P producer task for score instance 1."""
    return create_throughput_latency_softmax_task_impl(
        tmem_s1,
        tmem_softmax_local1,
        smem_p1,
        tmem_softmax_global1,
        work_queue,
        cfg,
        inst_id=1,
        domain=domain,
        task_class=task_class,
        **kw,
    )


def create_throughput_latency_correction_task(
    tmem_softmax_local0,
    tmem_softmax_local1,
    tmem_o,
    tmem_corr0,
    tmem_corr1,
    work_queue: WorkQueue | None,
    cfg,
    *,
    domain,
    task_class=MlaDecodeTask,
    **kw,
) -> Task:
    """Create the correction task that normalizes and stores 1CTA output.

    The task consumes local softmax statistics and TMEM O stages from both
    interleaved MMA pipes, applies online-softmax correction, and stores O/LSE
    or split-KV partials.
    """

    work_tile_skip_if = runtime_work_tile_skip_if(work_queue)

    @schedule
    def correction_schedule(
        tmem_softmax_local0,
        tmem_softmax_local1,
        tmem_o,
        tmem_corr0,
        tmem_corr1,
        work_queue=None,
    ):
        """Captured correction schedule for loop and tail O-stage draining."""

        def load_initial_local_stats(tmem_softmax_local, local_state):
            """Wait for one softmax pipe and return its initial stat state."""
            (
                old_max_arr,
                new_max_arr,
                sum_arr,
                inst0_old_max_arr,
                inst0_new_max_arr,
                inst0_sum_arr,
                inst1_old_max_arr,
                inst1_new_max_arr,
                inst1_sum_arr,
            ) = local_state
            tmem_softmax_local.wait()
            local_state = tmem_softmax_local.load_initial_stats(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
                inst0_old_max_arr=inst0_old_max_arr,
                inst0_new_max_arr=inst0_new_max_arr,
                inst0_sum_arr=inst0_sum_arr,
                inst1_old_max_arr=inst1_old_max_arr,
                inst1_new_max_arr=inst1_new_max_arr,
                inst1_sum_arr=inst1_sum_arr,
            )
            tmem_softmax_local.release()
            return local_state

        def load_loop_local_stats(tmem_softmax_local, local_state):
            """Wait for one softmax pipe and load loop old/new stats."""
            (
                old_max_arr,
                new_max_arr,
                sum_arr,
                inst0_old_max_arr,
                inst0_new_max_arr,
                inst0_sum_arr,
                inst1_old_max_arr,
                inst1_new_max_arr,
                inst1_sum_arr,
            ) = local_state
            tmem_softmax_local.wait()
            local_state = tmem_softmax_local.load_loop_stats(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
                inst0_old_max_arr=inst0_old_max_arr,
                inst0_new_max_arr=inst0_new_max_arr,
                inst0_sum_arr=inst0_sum_arr,
                inst1_old_max_arr=inst1_old_max_arr,
                inst1_new_max_arr=inst1_new_max_arr,
                inst1_sum_arr=inst1_sum_arr,
            )
            tmem_softmax_local.release()
            return local_state

        def load_tail_local_stats(tmem_softmax_local, local_state):
            """Wait for one softmax pipe and load tail sum/new-max stats."""
            (
                old_max_arr,
                new_max_arr,
                sum_arr,
                inst0_old_max_arr,
                inst0_new_max_arr,
                inst0_sum_arr,
                inst1_old_max_arr,
                inst1_new_max_arr,
                inst1_sum_arr,
            ) = local_state
            tmem_softmax_local.wait()
            local_state = tmem_softmax_local.load_tail_stats(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
                inst0_old_max_arr=inst0_old_max_arr,
                inst0_new_max_arr=inst0_new_max_arr,
                inst0_sum_arr=inst0_sum_arr,
                inst1_old_max_arr=inst1_old_max_arr,
                inst1_new_max_arr=inst1_new_max_arr,
                inst1_sum_arr=inst1_sum_arr,
            )
            tmem_softmax_local.release()
            return local_state

        def correct_loop_o(
            tmem_softmax_local,
            tmem_corr,
            local_state,
            tail_o_stage_idx_0,
            tail_o_stage_idx_1,
            o_stage_inst_idx,
        ):
            """Rescale one loop O accumulator using local softmax stats."""
            local_state = load_loop_local_stats(tmem_softmax_local, local_state)
            (
                old_max_arr,
                new_max_arr,
                sum_arr,
                _inst0_old_max_arr,
                inst0_new_max_arr,
                inst0_sum_arr,
                _inst1_old_max_arr,
                inst1_new_max_arr,
                inst1_sum_arr,
            ) = local_state
            tmem_o.wait()
            (
                o_stage_idx,
                tail_o_stage_idx_0,
                tail_o_stage_idx_1,
            ) = tmem_o.o_stage(
                tail_o_stage_idx_0=tail_o_stage_idx_0,
                tail_o_stage_idx_1=tail_o_stage_idx_1,
                inst_idx=o_stage_inst_idx,
            )
            tmem_corr.correct_loop_and_store(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
                inst0_new_max_arr=inst0_new_max_arr,
                inst0_sum_arr=inst0_sum_arr,
                inst1_new_max_arr=inst1_new_max_arr,
                inst1_sum_arr=inst1_sum_arr,
                o_stage_idx=o_stage_idx,
                tail_o_stage_idx_0=tail_o_stage_idx_0,
                tail_o_stage_idx_1=tail_o_stage_idx_1,
            )
            tmem_o.release()
            return local_state, tail_o_stage_idx_0, tail_o_stage_idx_1

        local0_state = tmem_softmax_local0.init_stats_state()
        local1_state = tmem_softmax_local1.init_stats_state()
        _, tail_o_stage_idx_0, tail_o_stage_idx_1 = tmem_o.init_stage_state()
        tmem_corr0.init_store_state()
        tmem_corr1.init_store_state()

        def init_work_tile_state():
            local0_state = tmem_softmax_local0.init_stats_work_tile_state()
            local1_state = tmem_softmax_local1.init_stats_work_tile_state()
            _, tail_stage_0, tail_stage_1 = tmem_o.init_stage_work_tile_state()
            return local0_state, local1_state, tail_stage_0, tail_stage_1

        with work_tile_schedule_loop(
            work_queue,
            skip_if=work_tile_skip_if,
            non_skippable_prelude=(
                init_work_tile_state if work_queue is not None else None
            ),
        ) as (_, work_tile_state):
            if work_queue is not None:
                (
                    local0_state,
                    local1_state,
                    tail_o_stage_idx_0,
                    tail_o_stage_idx_1,
                ) = work_tile_state
            local0_state = load_initial_local_stats(tmem_softmax_local0, local0_state)
            local1_state = load_initial_local_stats(tmem_softmax_local1, local1_state)

            with domain_loop(0, domain, 1):
                (
                    local0_state,
                    tail_o_stage_idx_0,
                    tail_o_stage_idx_1,
                ) = correct_loop_o(
                    tmem_softmax_local0,
                    tmem_corr0,
                    local0_state,
                    tail_o_stage_idx_0,
                    tail_o_stage_idx_1,
                    0,
                )
                (
                    local1_state,
                    tail_o_stage_idx_0,
                    tail_o_stage_idx_1,
                ) = correct_loop_o(
                    tmem_softmax_local1,
                    tmem_corr1,
                    local1_state,
                    tail_o_stage_idx_0,
                    tail_o_stage_idx_1,
                    1,
                )

            stats0 = load_tail_local_stats(tmem_softmax_local0, local0_state)
            (
                _old_max_arr0,
                _new_max_arr0,
                _sum_arr0,
                _inst0_old_max_arr0,
                inst0_new_max_arr0,
                inst0_sum_arr0,
                _inst1_old_max_arr0,
                _inst1_new_max_arr0,
                _inst1_sum_arr0,
            ) = stats0
            tmem_o.wait()
            (
                o_stage_idx,
                tail_o_stage_idx_0,
                tail_o_stage_idx_1,
            ) = tmem_o.o_stage(
                tail_o_stage_idx_0=tail_o_stage_idx_0,
                tail_o_stage_idx_1=tail_o_stage_idx_1,
                inst_idx=0,
                is_tail=True,
            )
            # Loop-body correction already rescaled stream 0 to the final
            # stream-0 max before the tail PV0 accumulation. Keep the tail
            # stage index for final stream combination, but avoid a second
            # rescale of the same O0 data.
            del o_stage_idx
            stats1 = load_tail_local_stats(tmem_softmax_local1, local1_state)
            (
                old_max_arr1,
                new_max_arr1,
                sum_arr1,
                _inst0_old_max_arr1,
                _inst0_new_max_arr1,
                _inst0_sum_arr1,
                _inst1_old_max_arr1,
                inst1_new_max_arr1,
                inst1_sum_arr1,
            ) = stats1
            tmem_o.wait()
            (
                o_stage_idx,
                tail_o_stage_idx_0,
                tail_o_stage_idx_1,
            ) = tmem_o.o_stage(
                tail_o_stage_idx_0=tail_o_stage_idx_0,
                tail_o_stage_idx_1=tail_o_stage_idx_1,
                inst_idx=1,
                is_tail=True,
            )
            tmem_corr1.correct_tail_and_store(
                old_max_arr=old_max_arr1,
                new_max_arr=new_max_arr1,
                sum_arr=sum_arr1,
                inst0_new_max_arr=inst0_new_max_arr0,
                inst0_sum_arr=inst0_sum_arr0,
                inst1_new_max_arr=inst1_new_max_arr1,
                inst1_sum_arr=inst1_sum_arr1,
                o_stage_idx=o_stage_idx,
                tail_o_stage_idx_0=tail_o_stage_idx_0,
                tail_o_stage_idx_1=tail_o_stage_idx_1,
            )
            tmem_o.release()
            tmem_o.release()

    captured_schedule = (
        correction_schedule(
            tmem_softmax_local0,
            tmem_softmax_local1,
            tmem_o,
            tmem_corr0,
            tmem_corr1,
        )
        if work_queue is None
        else correction_schedule(
            tmem_softmax_local0,
            tmem_softmax_local1,
            tmem_o,
            tmem_corr0,
            tmem_corr1,
            work_queue,
        )
    )
    src = [tmem_softmax_local0, tmem_softmax_local1, tmem_o]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[tmem_corr0, tmem_corr1],
        cfg=cfg,
        warp_idx=cfg.correction_warp_idx,
        num_warps=cfg.correction_num_warps,
        schedule=captured_schedule,
        num_registers=cfg.correction_regs,
        name="CorrectionTask",
        **kw,
    )


def _make_page_offsets_task(
    smem_page_offsets,
    work_queue: WorkQueue | None,
    cfg,
    *,
    schedule_result,
    warp_idx=None,
    num_warps=None,
    task_class=MlaDecodeTask,
    **kw,
) -> Task:
    """Create a page-offset staging task from a captured schedule result."""
    src = []
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[smem_page_offsets],
        cfg=cfg,
        warp_idx=cfg.page_offsets_warp_idx if warp_idx is None else warp_idx,
        num_warps=cfg.page_offsets_num_warps if num_warps is None else num_warps,
        schedule=schedule_result,
        num_registers=cfg.mma_load_regs,
        name="LoadPageTableTask",
        **kw,
    )


def create_load_page_offsets_task(
    smem_page_offsets,
    work_queue: WorkQueue | None,
    cfg,
    *,
    domain,
    warp_idx=None,
    num_warps=None,
    task_class=MlaDecodeTask,
    **kw,
) -> Task:
    """Create the page-offset staging task for the selected 1CTA cadence.

    Swaps-MMA-AB stages only K0/K1 offsets because each delayed V tile shares
    its K tile's page map and reuses the register-cached IDs. Keeps-MMA-AB
    retains its independent K/V offset cadence. The produced offset stages are
    consumed by the generic load task.
    """

    work_tile_skip_if = runtime_work_tile_skip_if(work_queue)

    @schedule
    def page_offsets_schedule(smem_page_offsets, work_queue=None):
        """Stage page offsets in the selected K-before-V cadence."""
        smem_page_offsets.init_load_state()

        with work_tile_schedule_loop(work_queue, skip_if=work_tile_skip_if):
            if cfg.kernel_variant == "keeps_mma_ab":
                page_offsets_produce(
                    smem_page_offsets, "load_k0", section=MlaStage.Head
                )
                with domain_loop(0, domain, 1):
                    page_offsets_produce(
                        smem_page_offsets, "load_k0", section=MlaStage.Loop
                    )
                    page_offsets_produce(
                        smem_page_offsets, "load_v0", section=MlaStage.Loop
                    )
                page_offsets_produce(
                    smem_page_offsets, "load_v0", section=MlaStage.Tail
                )
            else:
                page_offsets_produce(
                    smem_page_offsets, "load_k0", section=MlaStage.Head
                )
                page_offsets_produce(
                    smem_page_offsets, "load_k1", section=MlaStage.Head
                )

                with domain_loop(0, domain, 1):
                    page_offsets_produce(
                        smem_page_offsets, "load_k0", section=MlaStage.Loop
                    )
                    page_offsets_produce(
                        smem_page_offsets, "load_k1", section=MlaStage.Loop
                    )

    schedule_result = (
        page_offsets_schedule(smem_page_offsets)
        if work_queue is None
        else page_offsets_schedule(smem_page_offsets, work_queue)
    )
    return _make_page_offsets_task(
        smem_page_offsets,
        work_queue,
        cfg,
        schedule_result=schedule_result,
        warp_idx=warp_idx,
        num_warps=num_warps,
        task_class=task_class,
        **kw,
    )


def _staged_kv_load_with_reused_page_ids(
    smem_kv,
    *,
    head_dim_stages,
    producer_label,
    section: MlaStage,
    smem_page_offsets,
    cached_page_ids,
    page_id_slot: int,
    consume_offsets: bool,
):
    """Stage one swaps-MMA K/V tile using its K-owned page-ID cache slot."""
    if consume_offsets:
        smem_page_offsets.wait()
        cached_page_ids = smem_page_offsets.read_offsets(
            cached_page_ids=cached_page_ids,
            cache_slot=page_id_slot,
        )
    for stage_idx in range(head_dim_stages):
        smem_kv.acquire()
        getattr(smem_kv, producer_label)(
            stage_idx=stage_idx,
            section=section,
            cached_page_ids=cached_page_ids,
            page_id_slot=page_id_slot,
        )
        smem_kv.commit()
    if consume_offsets:
        smem_page_offsets.release()
    return cached_page_ids


def create_throughput_latency_load_task(
    smem_q,
    smem_kv,
    work_queue: WorkQueue | None,
    schedule_token_throttle,
    cfg,
    *,
    domain,
    smem_page_offsets=None,
    use_page_offsets=False,
    warp_idx=None,
    num_warps=None,
    task_class=MlaDecodeTask,
    **kw,
) -> Task:
    """Create the Q/K/V load task.

    The load warp stages Q once, then follows the selected K-before-V cadence.
    Swaps-MMA-AB uses two interleaved K/V streams, while keeps-MMA-AB uses a
    single K stream followed by the deferred V stream.
    """
    page_offsets = smem_page_offsets if use_page_offsets else None
    work_tile_skip_if = runtime_work_tile_skip_if(work_queue)

    def load_schedule_body(
        smem_q,
        smem_kv,
        smem_page_offsets,
        work_queue,
        schedule_token_throttle,
    ):
        """Shared captured load schedule body for optional page offsets/WQ paths."""
        smem_q.init_load_state()
        smem_kv.init_load_state()
        if smem_page_offsets is not None:
            cached_page_ids = smem_page_offsets.init_read_state()
        else:
            cached_page_ids = None
        reuse_delayed_v_page_ids = (
            smem_page_offsets is not None and cfg.kernel_variant != "keeps_mma_ab"
        )

        def load_kv(
            *,
            head_dim_stages,
            producer_label,
            section,
            page_id_slot=0,
            reuse_cached_page_ids=False,
        ):
            """Select the native paged swaps cache or the existing load path."""
            nonlocal cached_page_ids
            if reuse_delayed_v_page_ids:
                cached_page_ids = _staged_kv_load_with_reused_page_ids(
                    smem_kv,
                    head_dim_stages=head_dim_stages,
                    producer_label=producer_label,
                    section=section,
                    smem_page_offsets=smem_page_offsets,
                    cached_page_ids=cached_page_ids,
                    page_id_slot=page_id_slot,
                    consume_offsets=not reuse_cached_page_ids,
                )
            else:
                cached_page_ids = staged_kv_load(
                    smem_kv,
                    head_dim_stages=head_dim_stages,
                    producer_label=producer_label,
                    section=section,
                    smem_page_offsets=smem_page_offsets,
                    cached_page_ids=cached_page_ids,
                )

        with work_tile_schedule_loop(work_queue, skip_if=work_tile_skip_if):
            schedule_token_throttle_head(schedule_token_throttle)
            smem_q.acquire()
            smem_q.load_q()
            smem_q.commit()
            load_kv(
                head_dim_stages=cfg.qk_head_dim_stages,
                producer_label="load_k0",
                section=MlaStage.Head,
                page_id_slot=0,
            )
            if cfg.kernel_variant != "keeps_mma_ab":
                load_kv(
                    head_dim_stages=cfg.qk_head_dim_stages,
                    producer_label="load_k1",
                    section=MlaStage.Head,
                    page_id_slot=1,
                )

            with domain_loop(0, domain, 1):
                if cfg.kernel_variant == "keeps_mma_ab":
                    load_kv(
                        head_dim_stages=cfg.qk_head_dim_stages,
                        producer_label="load_k0",
                        section=MlaStage.Loop,
                    )
                    load_kv(
                        head_dim_stages=cfg.v_head_dim_stages,
                        producer_label="load_v0",
                        section=MlaStage.Loop,
                    )
                else:
                    load_kv(
                        head_dim_stages=cfg.v_head_dim_stages,
                        producer_label="load_v0",
                        section=MlaStage.Loop,
                        page_id_slot=0,
                        reuse_cached_page_ids=True,
                    )
                    load_kv(
                        head_dim_stages=cfg.qk_head_dim_stages,
                        producer_label="load_k0",
                        section=MlaStage.Loop,
                        page_id_slot=0,
                    )
                    load_kv(
                        head_dim_stages=cfg.v_head_dim_stages,
                        producer_label="load_v1",
                        section=MlaStage.Loop,
                        page_id_slot=1,
                        reuse_cached_page_ids=True,
                    )
                    load_kv(
                        head_dim_stages=cfg.qk_head_dim_stages,
                        producer_label="load_k1",
                        section=MlaStage.Loop,
                        page_id_slot=1,
                    )

            load_kv(
                head_dim_stages=cfg.v_head_dim_stages,
                producer_label="load_v0",
                section=MlaStage.Tail,
                page_id_slot=0,
                reuse_cached_page_ids=True,
            )
            if cfg.kernel_variant != "keeps_mma_ab":
                load_kv(
                    head_dim_stages=cfg.v_head_dim_stages,
                    producer_label="load_v1",
                    section=MlaStage.Tail,
                    page_id_slot=1,
                    reuse_cached_page_ids=True,
                )

    @schedule
    def load_schedule(smem_q, smem_kv):
        """Captured load schedule without persistent work queue."""
        load_schedule_body(smem_q, smem_kv, None, None, None)

    @schedule
    def load_wq_schedule(smem_q, smem_kv, work_queue):
        """Captured load schedule with a persistent work queue."""
        load_schedule_body(smem_q, smem_kv, None, work_queue, None)

    @schedule
    def load_wq_throttle_schedule(
        smem_q,
        smem_kv,
        work_queue,
        schedule_token_throttle,
    ):
        """Captured load schedule with persistent work queue throttling."""
        load_schedule_body(
            smem_q,
            smem_kv,
            None,
            work_queue,
            schedule_token_throttle,
        )

    @schedule
    def load_page_offsets_schedule(
        smem_q,
        smem_kv,
        smem_page_offsets,
    ):
        """Captured load schedule with precomputed page offsets."""
        load_schedule_body(
            smem_q,
            smem_kv,
            smem_page_offsets,
            None,
            None,
        )

    @schedule
    def load_page_offsets_wq_schedule(
        smem_q,
        smem_kv,
        smem_page_offsets,
        work_queue,
    ):
        """Captured load schedule with page offsets and persistent work queue."""
        load_schedule_body(
            smem_q,
            smem_kv,
            smem_page_offsets,
            work_queue,
            None,
        )

    @schedule
    def load_page_offsets_wq_throttle_schedule(
        smem_q,
        smem_kv,
        smem_page_offsets,
        work_queue,
        schedule_token_throttle,
    ):
        """Captured load schedule with page offsets, work queue, and throttling."""
        load_schedule_body(
            smem_q,
            smem_kv,
            smem_page_offsets,
            work_queue,
            schedule_token_throttle,
        )

    if page_offsets is None:
        if work_queue is None:
            captured_schedule = load_schedule(smem_q, smem_kv)
        elif schedule_token_throttle is None:
            captured_schedule = load_wq_schedule(smem_q, smem_kv, work_queue)
        else:
            captured_schedule = load_wq_throttle_schedule(
                smem_q,
                smem_kv,
                work_queue,
                schedule_token_throttle,
            )
    elif work_queue is None:
        captured_schedule = load_page_offsets_schedule(
            smem_q,
            smem_kv,
            page_offsets,
        )
    elif schedule_token_throttle is None:
        captured_schedule = load_page_offsets_wq_schedule(
            smem_q,
            smem_kv,
            page_offsets,
            work_queue,
        )
    else:
        captured_schedule = load_page_offsets_wq_throttle_schedule(
            smem_q,
            smem_kv,
            page_offsets,
            work_queue,
            schedule_token_throttle,
        )
    src = []
    if use_page_offsets:
        src.append(smem_page_offsets)
    if work_queue is not None:
        src.append(work_queue)
    dst = [smem_q, smem_kv]
    if schedule_token_throttle is not None:
        dst.append(schedule_token_throttle)
    return task_class(
        src_resources=src,
        dst_resources=dst,
        cfg=cfg,
        warp_idx=cfg.load_warp_idx if warp_idx is None else warp_idx,
        num_warps=cfg.load_num_warps if num_warps is None else num_warps,
        schedule=captured_schedule,
        num_registers=cfg.mma_load_regs,
        name="LoadTmaTask",
        **kw,
    )


def create_throughput_latency_scheduler_task(
    work_queue: WorkQueue,
    schedule_token_throttle,
    cfg,
    *,
    task_class=MlaDecodeTask,
    **kw,
) -> Task:
    """Create the CLC dynamic persistent scheduler task."""

    @schedule
    def scheduler_schedule(work_queue, schedule_token_throttle=None):
        """Fetch and publish the next dynamic persistent work tile."""
        _work_tile, _ = work_queue.init_work_tile()
        with domain_loop(0, 0, 1):
            pass
        schedule_token_throttle_tail(schedule_token_throttle)
        work_queue.acquire()
        work_queue.fetch_work_tile()
        work_queue.commit()
        work_queue.wait()
        work_queue.get_and_advance_work_tile()
        work_queue.release()

    captured_schedule = (
        scheduler_schedule(work_queue)
        if schedule_token_throttle is None
        else scheduler_schedule(work_queue, schedule_token_throttle)
    )
    src = [work_queue]
    if schedule_token_throttle is not None:
        src.append(schedule_token_throttle)
    return task_class(
        src_resources=src,
        dst_resources=[work_queue],
        cfg=cfg,
        warp_idx=cfg.scheduler_warp_idx,
        num_warps=cfg.scheduler_num_warps,
        schedule=captured_schedule,
        num_registers=cfg.mma_load_regs,
        name="SchedulerTask",
        **kw,
    )


def create_padding_task(
    cfg,
    work_queue: WorkQueue | None = None,
    *,
    warp_idx=None,
    num_warps=None,
    task_class=MlaDecodeTask,
    **kw,
) -> Task:
    """Create the padding task used to reserve otherwise idle warps."""

    work_tile_skip_if = runtime_work_tile_skip_if(work_queue)

    @schedule
    def padding_schedule(work_queue=None):
        """Captured no-op schedule with optional persistent work-queue tail."""
        # Keep the task-scheduling and domain-loop scopes distinct for CuTe DSL.
        with work_tile_schedule_loop(  # noqa: SIM117
            work_queue, skip_if=work_tile_skip_if
        ):
            with domain_loop(0, 1, 1):
                pass

    captured_schedule = (
        padding_schedule() if work_queue is None else padding_schedule(work_queue)
    )
    src = [work_queue] if work_queue is not None else []
    return task_class(
        src_resources=src,
        dst_resources=[],
        cfg=cfg,
        warp_idx=cfg.padding_warp_idx if warp_idx is None else warp_idx,
        num_warps=cfg.padding_num_warps if num_warps is None else num_warps,
        schedule=captured_schedule,
        num_registers=cfg.mma_load_regs,
        name="PaddingTask",
        **kw,
    )


def create_throughput_latency_mma_task(
    smem_q,
    smem_kv,
    tmem_s0,
    tmem_s1,
    smem_p0,
    smem_p1,
    tmem_o,
    work_queue: WorkQueue | None,
    cfg,
    *,
    domain,
    task_class=MlaDecodeTask,
    **kw,
) -> Task:
    """Create the MMA task for interleaved QK and PV stages.

    The task consumes Q/K/V SMEM descriptors, produces two TMEM score pipes,
    materializes PV output into TMEM O, and drains the final V tiles in tail.
    """

    work_tile_skip_if = runtime_work_tile_skip_if(work_queue)

    @schedule
    def mma_schedule(
        smem_q,
        smem_kv,
        tmem_s0,
        tmem_s1,
        smem_p0,
        smem_p1,
        tmem_o,
        work_queue=None,
    ):
        """Captured MMA schedule with K-prefetch and V-tail drain."""
        smem_q.init_descriptor_state()
        smem_kv.init_descriptor_state()
        smem_p0.init_descriptor_state()
        smem_p1.init_descriptor_state()
        tmem_o.init_mma_state()

        with work_tile_schedule_loop(work_queue, skip_if=work_tile_skip_if):
            if work_queue is not None:
                tmem_s0.reset_softmax_work_tile_state()
                tmem_s1.reset_softmax_work_tile_state()
            smem_q.wait()
            q_desc, q_desc_rope = smem_q.q_desc()
            tmem_s0.set_q_desc(q_desc=q_desc, q_desc_rope=q_desc_rope)
            tmem_s1.set_q_desc(q_desc=q_desc, q_desc_rope=q_desc_rope)
            staged_qk_mma(
                smem_kv,
                tmem_s0,
                head_dim_stages=cfg.qk_head_dim_stages,
                consumer_label="k_desc_0",
            )
            staged_qk_mma(
                smem_kv,
                tmem_s1,
                head_dim_stages=cfg.qk_head_dim_stages,
                consumer_label="k_desc_1",
            )
            with domain_loop(0, domain, 1):
                tmem_s0.acquire()
                staged_pv_mma(
                    smem_kv,
                    smem_p0,
                    tmem_o,
                    head_dim_stages=cfg.v_head_dim_stages,
                    consumer_label="v_desc_0",
                    producer_label="pv_mma_loop_0",
                )
                staged_qk_mma(
                    smem_kv,
                    tmem_s0,
                    head_dim_stages=cfg.qk_head_dim_stages,
                    consumer_label="k_desc_0",
                    include_acquire=False,
                )
                tmem_s1.acquire()
                staged_pv_mma(
                    smem_kv,
                    smem_p1,
                    tmem_o,
                    head_dim_stages=cfg.v_head_dim_stages,
                    consumer_label="v_desc_1",
                    producer_label="pv_mma_loop_1",
                )
                staged_qk_mma(
                    smem_kv,
                    tmem_s1,
                    head_dim_stages=cfg.qk_head_dim_stages,
                    consumer_label="k_desc_1",
                    include_acquire=False,
                )
            staged_pv_mma(
                smem_kv,
                smem_p0,
                tmem_o,
                head_dim_stages=cfg.v_head_dim_stages,
                consumer_label="v_desc_0",
                producer_label="pv_mma_tail_0",
            )
            staged_pv_mma(
                smem_kv,
                smem_p1,
                tmem_o,
                head_dim_stages=cfg.v_head_dim_stages,
                consumer_label="v_desc_1",
                producer_label="pv_mma_tail_1",
            )
            smem_q.release()

    captured_schedule = (
        mma_schedule(
            smem_q,
            smem_kv,
            tmem_s0,
            tmem_s1,
            smem_p0,
            smem_p1,
            tmem_o,
        )
        if work_queue is None
        else mma_schedule(
            smem_q,
            smem_kv,
            tmem_s0,
            tmem_s1,
            smem_p0,
            smem_p1,
            tmem_o,
            work_queue,
        )
    )
    src = [smem_q, smem_kv, smem_p0, smem_p1]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[tmem_s0, tmem_s1, tmem_o],
        cfg=cfg,
        warp_idx=cfg.mma_warp_idx,
        num_warps=cfg.mma_num_warps,
        schedule=captured_schedule,
        num_registers=cfg.mma_load_regs,
        name="MmaTask",
        **kw,
    )


def create_keeps_mma_ab_mma_task(
    smem_q,
    smem_kv,
    tmem_s,
    tmem_p,
    tmem_o,
    work_queue: WorkQueue | None,
    cfg,
    *,
    domain,
    task_class=MlaDecodeTask,
    **kw,
) -> Task:
    """Create the keeps-MMA-AB single-pipe MMA task.

    HEAD computes QK for K[0]. LOOP computes QK for K[n] before PV for
    K[n-1], and TAIL drains PV for K[last].
    """

    work_tile_skip_if = runtime_work_tile_skip_if(work_queue)

    @schedule
    def mma_schedule(
        smem_q,
        smem_kv,
        tmem_s,
        tmem_p,
        tmem_o,
        work_queue=None,
    ):
        """Captured keeps-MMA-AB schedule with TMEM P."""
        smem_q.init_descriptor_state()
        smem_kv.init_descriptor_state()
        tmem_o.init_mma_state()
        tmem_p.init_stage_state()

        with work_tile_schedule_loop(work_queue, skip_if=work_tile_skip_if):
            if work_queue is not None:
                tmem_s.reset_softmax_work_tile_state()
                tmem_p.init_stage_work_tile_state()
            smem_q.wait()
            q_desc, q_desc_rope = smem_q.q_desc()
            tmem_s.set_q_desc(q_desc=q_desc, q_desc_rope=q_desc_rope)
            staged_qk_mma(
                smem_kv,
                tmem_s,
                head_dim_stages=cfg.qk_head_dim_stages,
                consumer_label="k_desc_0",
            )
            with domain_loop(0, domain, 1):
                staged_qk_mma(
                    smem_kv,
                    tmem_s,
                    head_dim_stages=cfg.qk_head_dim_stages,
                    consumer_label="k_desc_0",
                )
                staged_pv_mma_tmem_p(
                    smem_kv,
                    tmem_p,
                    tmem_o,
                    head_dim_stages=cfg.v_head_dim_stages,
                    consumer_label="v_desc_0",
                    producer_label="pv_mma_loop_tmem_p",
                )
            staged_pv_mma_tmem_p(
                smem_kv,
                tmem_p,
                tmem_o,
                head_dim_stages=cfg.v_head_dim_stages,
                consumer_label="v_desc_0",
                producer_label="pv_mma_tail_tmem_p",
            )
            smem_q.release()

    schedule_result = (
        mma_schedule(smem_q, smem_kv, tmem_s, tmem_p, tmem_o)
        if work_queue is None
        else mma_schedule(smem_q, smem_kv, tmem_s, tmem_p, tmem_o, work_queue)
    )
    src = [smem_q, smem_kv, tmem_p]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[tmem_s, tmem_o],
        cfg=cfg,
        warp_idx=cfg.mma_warp_idx,
        num_warps=cfg.mma_num_warps,
        schedule=schedule_result,
        num_registers=cfg.mma_load_regs,
        name="MmaTask",
        **kw,
    )
