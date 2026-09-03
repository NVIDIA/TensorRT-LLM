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

"""Task definitions for the TS FMHA kernel.

Tasks own ordering, not data movement bodies. Each schedule below sequences
resource waits, acquires, work calls, commits, and releases for one warp role.
The resource methods contain the actual TMA, MMA, softmax, correction, and
epilogue work.

Schedule phase terms follow TS schedule-builder naming. HEAD is the one-time
schedule before the repeated K/V tile loop, LOOP is the repeated K/V tile body,
and TAIL is the one-time cleanup and drain after LOOP exits.
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass import Int32

from ..stage import FmhaStage
from cutlass.experimental.task_scheduling.schedule_builder import (
    domain_loop,
    schedule,
    work_tile_loop,
)
from cutlass.experimental.task_scheduling.resources import MemoryResource, WorkQueue
from cutlass.experimental.task_scheduling.task import Task

from .fmha_resources import (
    FmhaConfig,
    GmemOResource,
    GmemQKVResource,
    S0S1SequenceResource,
    SmemKVResource,
    SmemOResource,
    SmemPageOffsetsKvResource,
    SmemQResource,
    TmemOResource,
    TmemPResource,
    TmemSPResource,
    TmemStatsResource,
    TmemStatsDoneResource,
)


@dataclass(kw_only=True)
class PackedContextWorkQueue(WorkQueue):
    """Persistent queue that skips Q tiles outside a runtime packed request."""

    cfg: cutlass.Constexpr[FmhaConfig] = field(init=False, default=None)
    cum_seqlen_q: Any = field(init=False, default=None)

    def __init__(
        self,
        cfg: FmhaConfig,
        cum_seqlen_q: cute.Tensor,
        **kwargs: Any,
    ) -> None:
        """Attach the per-run packed-Q metadata used by the skip predicate."""
        super().__init__(**kwargs)
        self.cfg = cfg
        self.cum_seqlen_q = cum_seqlen_q

    @cute.jit
    def skip_work_tile_if(self, work_tile: Any) -> cutlass.Boolean:
        """Skip a scheduler tile whose first Q row is outside its request."""
        seq_idx, _, batch_idx = self.cfg.work_tile_coord_indices
        seq_coord = Int32(work_tile.tile_idx[seq_idx])
        if cutlass.const_expr(self.cfg.uses_causal_reversed_head_batch_seq_tile_order):
            seq_coord = Int32(self.cfg.num_seq_tiles) - seq_coord - Int32(1)
        batch_coord = Int32(work_tile.tile_idx[batch_idx])
        q_begin = Int32(self.cum_seqlen_q[batch_coord])
        q_end = Int32(self.cum_seqlen_q[batch_coord + Int32(1)])
        seqlen_q = q_end - q_begin
        return seq_coord * Int32(self.cfg.cta_tiler[0]) >= seqlen_q


def _persistent_tail(work_queue: WorkQueue) -> None:
    """Advance and release the persistent work tile after one task body."""
    work_queue.wait()
    work_queue.get_and_advance_work_tile()
    work_queue.release()


def _src_resources(
    *resources: MemoryResource,
    work_queue: WorkQueue | None,
) -> list[MemoryResource]:
    """Build a task source-resource list, including WorkQueue when present."""
    src = list(resources)
    if work_queue is not None:
        src.append(work_queue)
    return src


def _schedule_with_work_queue(
    schedule: Callable[..., object],
    *resources: MemoryResource,
    work_queue: WorkQueue | None,
) -> object:
    """Invoke a captured schedule with the optional WorkQueue argument."""
    if work_queue is None:
        return schedule(*resources)
    return schedule(*resources, work_queue)


def _packed_context_skip_predicate(
    work_queue: WorkQueue | None,
) -> Callable[..., object] | None:
    """Select the runtime-Q skip predicate before schedule capture creates proxies."""
    if isinstance(work_queue, PackedContextWorkQueue):
        return PackedContextWorkQueue.skip_work_tile_if
    return None


@contextmanager
def _work_tile_schedule_loop(
    work_queue: WorkQueue | None,
    *,
    skip_if: Callable[..., object] | None = None,
) -> Generator[object | None, None, None]:
    """Wrap a task body once per persistent work tile, or once for static schedules."""
    if skip_if is not None:
        assert work_queue is not None
        with work_tile_loop(
            work_queue,
            skip_if=skip_if,
        ) as work_tiles:
            with work_tiles.skippable():
                yield work_tiles
            # Every fetched tile, including a skipped one, must advance and
            # release the queue exactly once so persistent workers converge.
            _persistent_tail(work_queue)
    elif work_queue is not None:
        with work_tile_loop(work_queue) as work_tile:
            yield work_tile
            _persistent_tail(work_queue)
    else:
        yield None


def _captured_loop_bounds(
    task_class: type[Task],
    task_kwargs: dict[str, object],
) -> tuple[object, object, object]:
    """Infer ``(start, end, step)`` loop bounds for a captured schedule.

    Dense schedules pass a static ``domain``; causal schedules pass
    ``num_kv_tiles`` and use the task class's ``get_domain`` as a dynamic end.
    """
    loop_start = task_kwargs.pop("domain_start", 0)
    loop_step = task_kwargs.pop("step", 1)
    loop_end = task_kwargs.pop("domain", None)
    if loop_end is None:
        if "num_kv_tiles" not in task_kwargs:
            raise ValueError(
                "create_*_task requires a 'domain' or 'num_kv_tiles' kwarg to "
                "determine the loop end."
            )
        loop_end = task_class.get_domain
    return loop_start, loop_end, loop_step


def create_load_task(
    gmem_qkv: GmemQKVResource,
    smem_q: SmemQResource,
    smem_kv: SmemKVResource,
    work_queue: WorkQueue | None,
    task_class: type[Task] = Task,
    smem_page_offsets_kv: SmemPageOffsetsKvResource | None = None,
    smem_page_offsets_v: SmemPageOffsetsKvResource | None = None,
    **task_kwargs: Any,
) -> Task:
    """Create the one-warp TMA load task.

    When ``smem_page_offsets_kv`` is provided, each K/V TMA load consumes page
    IDs prefetched by the auxiliary warp through the ordinary asynchronous
    page-offset pipeline.
    """
    loop_start, loop_end, loop_step = _captured_loop_bounds(task_class, task_kwargs)
    skip_work_tile_if = _packed_context_skip_predicate(work_queue)
    src = _src_resources(gmem_qkv, work_queue=work_queue)
    dst = [smem_q, smem_kv]
    if smem_page_offsets_kv is not None:
        src.append(smem_page_offsets_kv)
    if smem_page_offsets_v is not None:
        src.append(smem_page_offsets_v)
    if smem_q.cfg.single_qkv_instance and smem_q.cfg.has_tmem_p_pipeline:
        num_head_dim_stages_k = smem_kv.cfg.num_head_dim_stages_k
        num_head_dim_stages_v = smem_kv.cfg.num_head_dim_stages_v

        if smem_page_offsets_v is not None:
            if smem_page_offsets_kv is None:
                raise ValueError("a V page window requires a matching K page window")
            pages_per_tile = smem_kv.cfg.kv_tile_n // smem_kv.cfg.num_tokens_per_page
            page_window_period = smem_kv.cfg.page_table_window_entries // pages_per_tile
            if (
                not isinstance(loop_start, int)
                or not isinstance(loop_end, int)
                or not isinstance(loop_step, int)
                or loop_start != 0
                or loop_step != 1
                or loop_end < page_window_period
                or loop_end % page_window_period != 0
            ):
                raise ValueError(
                    "reused page windows require a compile-time K/V domain "
                    "divisible by the topology-derived page-window period"
                )

            def load_reused_page_windows_schedule_body(
                gqkv: GmemQKVResource,
                sq: SmemQResource,
                skv: SmemKVResource,
                spok: SmemPageOffsetsKvResource,
                spov: SmemPageOffsetsKvResource,
                wq: WorkQueue | None,
            ) -> None:
                """Load staged K/V while retaining each page-ID window."""
                sq.init_load_state()
                skv.init_load_state()
                spok.init_read_state()
                cached_v_page_ids = spov.init_cached_read_state()

                with _work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
                    (
                        _seq_coord,
                        head_coord,
                        kv_head_coord,
                        _head_coord_kv,
                        batch_coord,
                        seq_coord_q,
                        cuseqlen_q,
                        cuseqlen_k,
                        seqlen_q,
                        seqlen_k,
                        kv_tile_start,
                        kv_request_begin,
                        kv_page_idx_ub,
                    ) = gqkv.compute_coords()
                    sq.acquire()
                    sq.tma_load(
                        seq_coord_q=seq_coord_q,
                        head_coord=head_coord,
                        batch_coord=batch_coord,
                        cuseqlen_q=cuseqlen_q,
                        seqlen_q=seqlen_q,
                        inst_idx=0,
                    )
                    sq.commit()

                    def load_k_tile(*, tile_offset: int) -> None:
                        for head_dim_stage_idx in range(num_head_dim_stages_k):
                            skv.try_acquire()
                            skv.acquire()
                            skv.k_load_stage(
                                stage_id=head_dim_stage_idx,
                                tile_offset=tile_offset,
                                kv_head_coord=kv_head_coord,
                                batch_coord=batch_coord,
                                cuseqlen_k=cuseqlen_k,
                                seqlen_k=seqlen_k,
                                kv_tile_start=kv_tile_start,
                                kv_request_begin=kv_request_begin,
                                kv_page_idx_ub=kv_page_idx_ub,
                            )
                            skv.commit()

                    def cache_v_tile(*, tile_offset: int) -> None:
                        nonlocal cached_v_page_ids
                        cached_v_page_ids = spov.cache_tile_page_ids(
                            cached_page_ids=cached_v_page_ids,
                            kv_tile_start=kv_tile_start,
                            tile_offset=tile_offset,
                        )

                    def load_v_tile(
                        *, tile_offset: int, reuse_cached_page_ids: bool = False
                    ) -> None:
                        for head_dim_stage_idx in range(num_head_dim_stages_v):
                            skv.try_acquire()
                            skv.acquire()
                            if reuse_cached_page_ids:
                                skv.v_load_stage_cached(
                                    cached_v_page_ids=cached_v_page_ids,
                                    stage_id=head_dim_stage_idx,
                                    tile_offset=tile_offset,
                                    kv_head_coord=kv_head_coord,
                                    batch_coord=batch_coord,
                                    cuseqlen_k=cuseqlen_k,
                                    seqlen_k=seqlen_k,
                                    kv_tile_start=kv_tile_start,
                                    kv_request_begin=kv_request_begin,
                                    kv_page_idx_ub=kv_page_idx_ub,
                                )
                            else:
                                skv.v_load_stage(
                                    stage_id=head_dim_stage_idx,
                                    tile_offset=tile_offset,
                                    kv_head_coord=kv_head_coord,
                                    batch_coord=batch_coord,
                                    cuseqlen_k=cuseqlen_k,
                                    seqlen_k=seqlen_k,
                                    kv_tile_start=kv_tile_start,
                                    kv_request_begin=kv_request_begin,
                                    kv_page_idx_ub=kv_page_idx_ub,
                                )
                            skv.commit()

                    # Window zero: K stays one tile ahead of V. Cache the final
                    # V IDs before releasing the window because its last V
                    # tile is delayed across the boundary.
                    spok.wait()
                    spok.read_offsets()
                    load_k_tile(tile_offset=0)
                    load_k_tile(tile_offset=1)
                    spov.wait()
                    load_v_tile(tile_offset=0)
                    for tile_delta in range(2, page_window_period - 1):
                        load_k_tile(tile_offset=tile_delta)
                        load_v_tile(tile_offset=tile_delta - 1)
                    load_k_tile(tile_offset=page_window_period - 1)
                    spok.release()
                    load_v_tile(tile_offset=page_window_period - 2)
                    cache_v_tile(tile_offset=page_window_period - 1)
                    spov.release()

                    # Each structural iteration consumes one complete K/V page
                    # window. Only register page IDs cross the loop boundary.
                    with domain_loop(
                        page_window_period,
                        loop_end,
                        page_window_period,
                    ):
                        spok.wait()
                        spok.read_offsets()
                        load_k_tile(tile_offset=0)
                        load_v_tile(tile_offset=-1, reuse_cached_page_ids=True)
                        load_k_tile(tile_offset=1)
                        spov.wait()
                        load_v_tile(tile_offset=0)
                        for tile_delta in range(2, page_window_period - 1):
                            load_k_tile(tile_offset=tile_delta)
                            load_v_tile(tile_offset=tile_delta - 1)
                        load_k_tile(tile_offset=page_window_period - 1)
                        spok.release()
                        load_v_tile(tile_offset=page_window_period - 2)
                        cache_v_tile(tile_offset=page_window_period - 1)
                        spov.release()

                    load_v_tile(
                        tile_offset=page_window_period - 1,
                        reuse_cached_page_ids=True,
                    )

            @schedule
            def load_reused_page_windows_schedule(
                gqkv: GmemQKVResource,
                sq: SmemQResource,
                skv: SmemKVResource,
                spok: SmemPageOffsetsKvResource,
                spov: SmemPageOffsetsKvResource,
                wq: WorkQueue | None = None,
            ) -> None:
                load_reused_page_windows_schedule_body(gqkv, sq, skv, spok, spov, wq)

            captured_schedule = _schedule_with_work_queue(
                load_reused_page_windows_schedule,
                gmem_qkv,
                smem_q,
                smem_kv,
                smem_page_offsets_kv,
                smem_page_offsets_v,
                work_queue=work_queue,
            )
            return task_class(
                src_resources=src,
                dst_resources=dst,
                warp_idx=smem_kv.cfg.load_warp_id,
                num_warps=1,
                schedule=captured_schedule,
                num_registers=smem_kv.cfg.num_regs_other,
                name="LoadTask",
                **task_kwargs,
            )

        def load_schedule_body(
            gqkv: GmemQKVResource,
            sq: SmemQResource,
            skv: SmemKVResource,
            spo: SmemPageOffsetsKvResource | None,
            wq: WorkQueue | None,
        ) -> None:
            sq.init_load_state()
            skv.init_load_state()
            if spo is not None:
                spo.init_read_state()
            with _work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
                coords = gqkv.compute_coords()
                (
                    _seq_coord,
                    head_coord,
                    kv_head_coord,
                    _head_coord_kv,
                    batch_coord,
                    seq_coord_q,
                    cuseqlen_q,
                    cuseqlen_k,
                    seqlen_q,
                    seqlen_k,
                    kv_tile_start,
                    kv_request_begin,
                    kv_page_idx_ub,
                ) = coords
                sq.acquire()
                sq.tma_load(
                    seq_coord_q=seq_coord_q,
                    head_coord=head_coord,
                    batch_coord=batch_coord,
                    cuseqlen_q=cuseqlen_q,
                    seqlen_q=seqlen_q,
                    inst_idx=0,
                )
                sq.commit()

                for head_dim_stage_idx in range(num_head_dim_stages_k):
                    skv.try_acquire()
                    if spo is not None and head_dim_stage_idx == 0:
                        spo.wait()
                        spo.read_offsets()
                    skv.acquire()
                    skv.k_load_stage(
                        stage_id=head_dim_stage_idx,
                        kv_head_coord=kv_head_coord,
                        batch_coord=batch_coord,
                        cuseqlen_k=cuseqlen_k,
                        seqlen_k=seqlen_k,
                        kv_tile_start=kv_tile_start,
                        kv_request_begin=kv_request_begin,
                        kv_page_idx_ub=kv_page_idx_ub,
                    )
                    skv.commit()
                if spo is not None:
                    spo.release()

                with domain_loop(loop_start + 1, loop_end, loop_step):
                    for head_dim_stage_idx in range(num_head_dim_stages_k):
                        skv.try_acquire()
                        if spo is not None and head_dim_stage_idx == 0:
                            spo.wait()
                            spo.read_offsets()
                        skv.acquire()
                        skv.k_load_stage(
                            stage_id=head_dim_stage_idx,
                            kv_head_coord=kv_head_coord,
                            batch_coord=batch_coord,
                            cuseqlen_k=cuseqlen_k,
                            seqlen_k=seqlen_k,
                            kv_tile_start=kv_tile_start,
                            kv_request_begin=kv_request_begin,
                            kv_page_idx_ub=kv_page_idx_ub,
                        )
                        skv.commit()
                    if spo is not None:
                        spo.release()

                    for head_dim_stage_idx in range(num_head_dim_stages_v):
                        skv.try_acquire()
                        if spo is not None and head_dim_stage_idx == 0:
                            spo.wait()
                            spo.read_offsets()
                        skv.acquire()
                        skv.v_load_stage(
                            stage_id=head_dim_stage_idx,
                            previous=True,
                            kv_head_coord=kv_head_coord,
                            batch_coord=batch_coord,
                            cuseqlen_k=cuseqlen_k,
                            seqlen_k=seqlen_k,
                            kv_tile_start=kv_tile_start,
                            kv_request_begin=kv_request_begin,
                            kv_page_idx_ub=kv_page_idx_ub,
                        )
                        skv.commit()
                    if spo is not None:
                        spo.release()

                for head_dim_stage_idx in range(num_head_dim_stages_v):
                    skv.try_acquire()
                    if spo is not None and head_dim_stage_idx == 0:
                        spo.wait()
                        spo.read_offsets()
                    skv.acquire()
                    skv.v_load_stage(
                        stage_id=head_dim_stage_idx,
                        previous=False,
                        kv_head_coord=kv_head_coord,
                        batch_coord=batch_coord,
                        cuseqlen_k=cuseqlen_k,
                        seqlen_k=seqlen_k,
                        kv_tile_start=kv_tile_start,
                        kv_request_begin=kv_request_begin,
                        kv_page_idx_ub=kv_page_idx_ub,
                    )
                    skv.commit()
                if spo is not None:
                    spo.release()

        @schedule
        def load_schedule(
            gqkv: GmemQKVResource,
            sq: SmemQResource,
            skv: SmemKVResource,
            wq: WorkQueue | None = None,
        ) -> None:
            load_schedule_body(gqkv, sq, skv, None, wq)

        @schedule
        def load_page_offsets_schedule(
            gqkv: GmemQKVResource,
            sq: SmemQResource,
            skv: SmemKVResource,
            spo: SmemPageOffsetsKvResource,
            wq: WorkQueue | None = None,
        ) -> None:
            load_schedule_body(gqkv, sq, skv, spo, wq)

        if smem_page_offsets_kv is None:
            captured_schedule = _schedule_with_work_queue(
                load_schedule, gmem_qkv, smem_q, smem_kv, work_queue=work_queue
            )
        else:
            captured_schedule = _schedule_with_work_queue(
                load_page_offsets_schedule,
                gmem_qkv,
                smem_q,
                smem_kv,
                smem_page_offsets_kv,
                work_queue=work_queue,
            )
        return task_class(
            src_resources=src,
            dst_resources=dst,
            warp_idx=smem_kv.cfg.load_warp_id,
            num_warps=1,
            schedule=captured_schedule,
            num_registers=smem_kv.cfg.num_regs_other,
            name="LoadTask",
            **task_kwargs,
        )

    if smem_q.cfg.single_qkv_instance:
        raise ValueError("single-instance context requires the staged TMEM-P topology")
    if smem_page_offsets_kv is not None or smem_page_offsets_v is not None:
        raise ValueError("paired context resolves paged K/V IDs directly")

    def load_schedule_body(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        skv: SmemKVResource,
        wq: WorkQueue | None,
    ) -> None:
        """Load paired Q instances and their directly addressed K/V tiles."""
        sq.init_load_state()
        skv.init_load_state()
        with _work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):  # noqa: SIM117
            # The first K-loop iteration also loads Q0/Q1. Later iterations
            # only stream the next K/V tiles through the SmemKV pipeline.
            with domain_loop(loop_start, loop_end, loop_step) as d:
                with d.first_iter():
                    (
                        _seq_coord,
                        head_coord,
                        kv_head_coord,
                        _head_coord_kv,
                        batch_coord,
                        seq_coord_q,
                        cuseqlen_q,
                        cuseqlen_k,
                        seqlen_q,
                        seqlen_k,
                        kv_tile_start,
                        kv_request_begin,
                        kv_page_idx_ub,
                    ) = gqkv.compute_coords()
                    # Load Q0 for the first Q tile in this work tile.
                    sq.acquire()
                    sq.tma_load(
                        seq_coord_q=seq_coord_q,
                        head_coord=head_coord,
                        batch_coord=batch_coord,
                        cuseqlen_q=cuseqlen_q,
                        seqlen_q=seqlen_q,
                        inst_idx=0,
                    )
                    sq.commit()
                # Throttle TMA before reserving a KV stage.
                skv.try_acquire()
                # Load Ki, with K0 handled by the first iteration.
                skv.acquire()
                skv.k_load(
                    kv_head_coord=kv_head_coord,
                    batch_coord=batch_coord,
                    cuseqlen_k=cuseqlen_k,
                    seqlen_k=seqlen_k,
                    kv_tile_start=kv_tile_start,
                    kv_request_begin=kv_request_begin,
                    kv_page_idx_ub=kv_page_idx_ub,
                )
                skv.commit()
                with d.first_iter():
                    # Load Q1 for the second Q tile in this work tile.
                    sq.acquire()
                    sq.tma_load(
                        seq_coord_q=seq_coord_q,
                        head_coord=head_coord,
                        batch_coord=batch_coord,
                        cuseqlen_q=cuseqlen_q,
                        seqlen_q=seqlen_q,
                        inst_idx=1,
                    )
                    sq.commit()
                # Throttle TMA before reserving a KV stage.
                skv.try_acquire()
                # Load Vi, with V0 handled by the first iteration.
                skv.acquire()
                skv.v_load(
                    kv_head_coord=kv_head_coord,
                    batch_coord=batch_coord,
                    cuseqlen_k=cuseqlen_k,
                    seqlen_k=seqlen_k,
                    kv_tile_start=kv_tile_start,
                    kv_request_begin=kv_request_begin,
                    kv_page_idx_ub=kv_page_idx_ub,
                )
                skv.commit()

    @schedule
    def load_schedule(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        skv: SmemKVResource,
        wq: WorkQueue | None = None,
    ) -> None:
        """Contiguous-KV captured schedule."""
        # Mypy retains the earlier branch's five-argument closure signature.
        load_schedule_body(gqkv, sq, skv, wq)  # type: ignore[call-arg]

    captured_schedule = _schedule_with_work_queue(
        load_schedule, gmem_qkv, smem_q, smem_kv, work_queue=work_queue
    )
    return task_class(
        src_resources=src,
        dst_resources=dst,
        warp_idx=smem_kv.cfg.load_warp_id,
        num_warps=1,
        schedule=captured_schedule,
        num_registers=smem_kv.cfg.num_regs_other,
        name="LoadTask",
        **task_kwargs,
    )


def create_mma_task(
    gmem_qkv: GmemQKVResource,
    smem_q: SmemQResource,
    smem_kv: SmemKVResource,
    tmem_sp0: TmemSPResource,
    tmem_sp1: TmemSPResource | None,
    tmem_p0: TmemPResource | None,
    tmem_o: TmemOResource,
    tmem_vec_done_0: TmemStatsDoneResource,
    tmem_vec_done_1: TmemStatsDoneResource | None,
    work_queue: WorkQueue | None,
    task_class: type[Task] = Task,
    **task_kwargs: Any,
) -> Task:
    """Create the one-warp MMA compute task."""
    loop_start, loop_end, loop_step = _captured_loop_bounds(task_class, task_kwargs)
    skip_work_tile_if = _packed_context_skip_predicate(work_queue)
    src = _src_resources(gmem_qkv, smem_q, smem_kv, work_queue=work_queue)

    if (
        smem_q.cfg.single_qkv_instance
        and smem_q.cfg.has_tmem_p_pipeline
        and tmem_p0 is not None
    ):
        split_src = _src_resources(
            gmem_qkv, smem_q, smem_kv, tmem_p0, work_queue=work_queue
        )
        num_head_dim_stages_k = smem_kv.cfg.num_head_dim_stages_k
        num_head_dim_stages_v = smem_kv.cfg.num_head_dim_stages_v
        loop_carried_head_dim_stages = 2
        if (
            num_head_dim_stages_k != loop_carried_head_dim_stages
            or num_head_dim_stages_v != loop_carried_head_dim_stages
        ):
            raise ValueError("loop-carried split S/P scheduling expects two K/V stages")

        @schedule
        def mma_schedule(
            gqkv: GmemQKVResource,
            sq: SmemQResource,
            skv: SmemKVResource,
            sp0: TmemSPResource,
            tp0: TmemPResource,
            to: TmemOResource,
            vd0: TmemStatsDoneResource,
            wq: WorkQueue | None = None,
        ) -> None:
            sq.init_descriptor_state()
            skv.init_descriptor_state()
            sp0.init_mma_state()
            to.init_mma_state()
            with _work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
                sp0.init_mma_work_tile_state()
                to.init_mma_work_tile_state()
                v_seqlen_k = Int32(0)
                v_kv_tile_start = Int32(0)
                if cutlass.const_expr(smem_q.cfg.use_paged_kv):
                    (
                        _seq_coord,
                        _head_coord,
                        _kv_head_coord,
                        _head_coord_kv,
                        _batch_coord,
                        _seq_coord_q,
                        _cuseqlen_q,
                        _cuseqlen_k,
                        _seqlen_q,
                        v_seqlen_k,
                        v_kv_tile_start,
                        _kv_request_begin,
                        _kv_page_idx_ub,
                    ) = gqkv.compute_coords()

                sq.wait()
                desc_q0_base = sq.q0_desc(inst_idx=0)
                if not smem_q.cfg.stats_via_smem:
                    vd0.acquire()
                sp0.acquire()
                for head_dim_stage_idx in range(num_head_dim_stages_k):
                    skv.wait()
                    desc_k_base = skv.k_desc()
                    sp0.qk_mma(
                        desc_q_base=desc_q0_base,
                        desc_k_base=desc_k_base,
                        section=FmhaStage.Head,
                        head_dim_stage_idx=head_dim_stage_idx,
                    )
                    skv.release()
                sp0.commit()
                if not smem_q.cfg.stats_via_smem:
                    vd0.commit()
                sp0.acquire()

                # Loop offset i is local to the steady state. LoadTask has
                # already advanced K by one tile, so these waits consume K(i+1)
                # for QK and V(i) for PV.
                with domain_loop(loop_start, loop_end, loop_step):
                    if not smem_q.cfg.stats_via_smem:
                        vd0.acquire()
                    skv.wait()
                    desc_k_base = skv.k_desc()
                    sp0.qk_mma(
                        desc_q_base=desc_q0_base,
                        desc_k_base=desc_k_base,
                        section=FmhaStage.Loop,
                        head_dim_stage_idx=0,
                    )
                    skv.release()

                    skv.wait()
                    desc_k_base = skv.k_desc()
                    sp0.qk_mma(
                        desc_q_base=desc_q0_base,
                        desc_k_base=desc_k_base,
                        section=FmhaStage.Loop,
                        head_dim_stage_idx=1,
                    )
                    skv.release()
                    sp0.commit()
                    if not smem_q.cfg.stats_via_smem:
                        vd0.commit()
                    sp0.acquire()

                    to.acquire()
                    tp0.wait()
                    tmem_p_base = tp0.p_base()
                    to.set_p_base(tmem_p_base=tmem_p_base)

                    skv.wait()
                    if cutlass.const_expr(smem_q.cfg.use_paged_kv):
                        desc_v_base = skv.v_desc_paged(
                            section=FmhaStage.Loop,
                            seqlen_k=v_seqlen_k,
                            kv_tile_start=v_kv_tile_start,
                        )
                    else:
                        desc_v_base = skv.v_desc()
                    to.pv_mma(
                        desc_v_base=desc_v_base,
                        section=FmhaStage.Loop,
                        head_dim_stage_idx=0,
                    )
                    skv.release()

                    skv.wait()
                    if cutlass.const_expr(smem_q.cfg.use_paged_kv):
                        desc_v_base = skv.v_desc_paged(
                            section=FmhaStage.Loop,
                            seqlen_k=v_seqlen_k,
                            kv_tile_start=v_kv_tile_start,
                        )
                    else:
                        desc_v_base = skv.v_desc()
                    to.pv_mma(
                        desc_v_base=desc_v_base,
                        section=FmhaStage.Loop,
                        head_dim_stage_idx=1,
                    )
                    skv.release()
                    to.commit()
                    tp0.release()

                sq.release()
                to.acquire()
                tp0.wait()
                tmem_p_base = tp0.p_base()
                to.set_p_base(tmem_p_base=tmem_p_base)
                for head_dim_stage_idx in range(num_head_dim_stages_v):
                    skv.wait()
                    if cutlass.const_expr(smem_q.cfg.use_paged_kv):
                        desc_v_base = skv.v_desc_paged(
                            section=FmhaStage.Tail,
                            seqlen_k=v_seqlen_k,
                            kv_tile_start=v_kv_tile_start,
                        )
                    else:
                        desc_v_base = skv.v_desc()
                    to.pv_mma(
                        desc_v_base=desc_v_base,
                        section=FmhaStage.Tail,
                        head_dim_stage_idx=head_dim_stage_idx,
                        is_tail=True,
                    )
                    skv.release()
                to.commit()
                tp0.release()
                if not smem_q.cfg.stats_via_smem:
                    vd0.acquire()
                sp0.commit()
                if not smem_q.cfg.stats_via_smem:
                    vd0.commit()
                tp0.wait()
                tp0.release()

        captured_schedule = _schedule_with_work_queue(
            mma_schedule,
            gmem_qkv,
            smem_q,
            smem_kv,
            tmem_sp0,
            tmem_p0,
            tmem_o,
            tmem_vec_done_0,
            work_queue=work_queue,
        )
        return task_class(
            src_resources=split_src,
            dst_resources=[tmem_sp0, tmem_o]
            + ([] if smem_q.cfg.stats_via_smem else [tmem_vec_done_0]),
            warp_idx=smem_q.cfg.mma_warp_id,
            num_warps=1,
            schedule=captured_schedule,
            name="MmaTask",
            num_registers=smem_q.cfg.num_regs_other,
            **task_kwargs,
        )

    if smem_q.cfg.single_qkv_instance:
        num_head_dim_stages_k = smem_kv.cfg.num_head_dim_stages_k
        num_head_dim_stages_v = smem_kv.cfg.num_head_dim_stages_v

        @schedule
        def mma_schedule(
            gqkv: GmemQKVResource,
            sq: SmemQResource,
            skv: SmemKVResource,
            sp0: TmemSPResource,
            to: TmemOResource,
            vd0: TmemStatsDoneResource,
            wq: WorkQueue | None = None,
        ) -> None:
            desc_q0_base, _desc_q1_base = sq.create_function_variables()
            desc_k_base, desc_v_base = skv.create_function_variables()
            sp0.create_function_variables()
            to.create_function_variables()
            vd0.create_function_variables()
            with _work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
                v_seqlen_k = Int32(0)
                v_kv_tile_start = Int32(0)
                if cutlass.const_expr(smem_q.cfg.use_paged_kv):
                    (
                        _seq_coord,
                        _head_coord,
                        _kv_head_coord,
                        _head_coord_kv,
                        _batch_coord,
                        _seq_coord_q,
                        _cuseqlen_q,
                        _cuseqlen_k,
                        _seqlen_q,
                        v_seqlen_k,
                        v_kv_tile_start,
                        _kv_request_begin,
                        _kv_page_idx_ub,
                    ) = gqkv.compute_coords()
                if wq is not None:
                    sp0.create_work_tile_variables()
                    to.create_work_tile_variables()

                with domain_loop(loop_start, loop_end, loop_step) as d:
                    with d.first_iter():
                        sq.wait()
                        desc_q0_base = sq.q0_desc(inst_idx=0)
                        vd0.acquire()
                        sp0.acquire()
                    for head_dim_stage_idx in range(num_head_dim_stages_k):
                        skv.wait()
                        desc_k_base = skv.k_desc()
                        sp0.qk_mma(
                            desc_q_base=desc_q0_base,
                            desc_k_base=desc_k_base,
                            section=FmhaStage.Loop,
                            head_dim_stage_idx=head_dim_stage_idx,
                        )
                        skv.release()
                    sp0.commit()
                    with d.first_iter():
                        vd0.commit()
                    to.acquire()
                    sp0.acquire()
                    sp0.p_read()
                    for head_dim_stage_idx in range(num_head_dim_stages_v):
                        skv.wait()
                        if cutlass.const_expr(smem_q.cfg.use_paged_kv):
                            desc_v_base = skv.v_desc_paged(
                                section=FmhaStage.Loop,
                                seqlen_k=v_seqlen_k,
                                kv_tile_start=v_kv_tile_start,
                            )
                        else:
                            desc_v_base = skv.v_desc()
                        to.pv_mma(
                            desc_v_base=desc_v_base,
                            section=FmhaStage.Loop,
                            head_dim_stage_idx=head_dim_stage_idx,
                        )
                        skv.release()
                    to.commit()

                sq.release()
                sp0.commit()

        captured_schedule = _schedule_with_work_queue(
            mma_schedule,
            gmem_qkv,
            smem_q,
            smem_kv,
            tmem_sp0,
            tmem_o,
            tmem_vec_done_0,
            work_queue=work_queue,
        )
        return task_class(
            src_resources=src,
            dst_resources=[tmem_sp0, tmem_o, tmem_vec_done_0],
            warp_idx=smem_q.cfg.mma_warp_id,
            num_warps=1,
            schedule=captured_schedule,
            name="MmaTask",
            num_registers=smem_q.cfg.num_regs_other,
            **task_kwargs,
        )

    if tmem_sp1 is None or tmem_vec_done_1 is None:
        raise ValueError("paired MMA scheduling requires peer-1 resources")

    @schedule
    def mma_schedule(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        skv: SmemKVResource,
        sp0: TmemSPResource,
        sp1: TmemSPResource,
        to: TmemOResource,
        vd0: TmemStatsDoneResource,
        vd1: TmemStatsDoneResource,
        wq: WorkQueue | None = None,
    ) -> None:
        """Captured schedule for interleaved QK and PV MMA work."""
        sq.init_descriptor_state()
        skv.init_descriptor_state()
        sp0.init_mma_state()
        sp1.init_mma_state()
        to.init_mma_state()
        with _work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
            v_seqlen_k = Int32(0)
            v_kv_tile_start = Int32(0)
            if cutlass.const_expr(smem_q.cfg.use_paged_kv):
                (
                    _seq_coord,
                    _head_coord,
                    _kv_head_coord,
                    _head_coord_kv,
                    _batch_coord,
                    _seq_coord_q,
                    _cuseqlen_q,
                    _cuseqlen_k,
                    _seqlen_q,
                    v_seqlen_k,
                    v_kv_tile_start,
                    _kv_request_begin,
                    _kv_page_idx_ub,
                ) = gqkv.compute_coords()
            # HEAD: consume Q0, K0, Q1, and V0. TmemStatsDone starts empty, so
            # the first acquire succeeds without priming. On later work tiles,
            # correction has released the previous stats slot.
            #
            # Consume Q0, K0, then QK(Q0,K0)→S0.
            sq.wait()
            desc_q0_base = sq.q0_desc(inst_idx=0)
            skv.wait()
            desc_k_base = skv.k_desc()
            if not smem_q.cfg.stats_via_smem:
                vd0.acquire()
            sp0.acquire()
            sp0.qk_mma(
                desc_q_base=desc_q0_base,
                desc_k_base=desc_k_base,
                section=FmhaStage.Head,
            )
            sp0.commit()
            if not smem_q.cfg.stats_via_smem:
                vd0.commit()
            # Consume Q1, then QK(Q1,K0)→S1.
            sq.wait()
            desc_q1_base = sq.q1_desc(inst_idx=1)
            if not smem_q.cfg.stats_via_smem:
                vd1.acquire()
            sp1.acquire()
            sp1.qk_mma(
                desc_q_base=desc_q1_base,
                desc_k_base=desc_k_base,
                section=FmhaStage.Head,
            )
            sp1.commit()
            if not smem_q.cfg.stats_via_smem:
                vd1.commit()
            # Q0/Q1 stay live because UMMA reads Q throughout the K-loop.
            # Release K0 (done with QK→S0 and QK→S1), then consume V0.
            skv.release()
            skv.wait()
            if cutlass.const_expr(smem_q.cfg.use_paged_kv):
                desc_v_base = skv.v_desc_paged(
                    section=FmhaStage.Head,
                    seqlen_k=v_seqlen_k,
                    kv_tile_start=v_kv_tile_start,
                )
            else:
                desc_v_base = skv.v_desc()
            # Acquire O first (off critical path), then acquire SP0 and run PV→O0.
            to.acquire()
            sp0.acquire()
            sp0.p_read()
            to.pv_mma(desc_v_base=desc_v_base, section=FmhaStage.Head)
            to.commit()

            # LOOP: interleave QK and PV work while preserving the previous V
            # tile until its PV MMA has consumed it:
            #   QK0(deferred commit) -> PV1(V_prev, release V_prev) ->
            #   QK1(commit) -> release Ki+1 -> wait Vi+1 -> PV0(no commit)
            with domain_loop(loop_start, loop_end, loop_step):
                skv.wait()
                desc_k_base = skv.k_desc()
                # QK0: QK(Q0,Ki+1) → S0 (no acquire; handle held from PV0).
                sp0.qk_mma(
                    desc_q_base=desc_q0_base,
                    desc_k_base=desc_k_base,
                    section=FmhaStage.Loop,
                )
                sp0.commit()
                # PV1(V_prev): P1 * V_prev → O1.
                to.acquire()
                sp1.acquire()
                sp1.p_read()
                to.pv_mma(desc_v_base=desc_v_base, section=FmhaStage.Loop)
                to.commit()
                # Release V_prev after PV1 UMMA consumed SMEM data.
                skv.release()
                # QK1: QK(Q1,Ki+1) → S1 (no acquire; handle held from PV1).
                sp1.qk_mma(
                    desc_q_base=desc_q1_base,
                    desc_k_base=desc_k_base,
                    section=FmhaStage.Loop,
                )
                sp1.commit()
                # Release Ki+1, then wait Vi+1.
                skv.release()
                skv.wait()
                if cutlass.const_expr(smem_q.cfg.use_paged_kv):
                    desc_v_base = skv.v_desc_paged(
                        section=FmhaStage.Loop,
                        tile_offset=1,
                        seqlen_k=v_seqlen_k,
                        kv_tile_start=v_kv_tile_start,
                    )
                else:
                    desc_v_base = skv.v_desc()
                # PV0: P0 * Vi+1 → O0.
                to.acquire()
                sp0.acquire()
                sp0.p_read()
                to.pv_mma(
                    desc_v_base=desc_v_base,
                    section=FmhaStage.Loop,
                    inst_idx=1,
                )
                to.commit()

            # TAIL: release Qs, close the deferred SP state, and run the final
            # PV→O1 MMA.
            sq.release()
            sq.release()
            sp0.commit()
            to.acquire()
            sp1.acquire()
            sp1.p_read()
            to.pv_mma(
                desc_v_base=desc_v_base,
                section=FmhaStage.Tail,
                is_tail=True,
            )
            to.commit()
            skv.release()
            sp1.commit()

    captured_schedule = _schedule_with_work_queue(
        mma_schedule,
        gmem_qkv,
        smem_q,
        smem_kv,
        tmem_sp0,
        tmem_sp1,
        tmem_o,
        tmem_vec_done_0,
        tmem_vec_done_1,
        work_queue=work_queue,
    )
    return task_class(
        src_resources=src,
        dst_resources=[tmem_sp0, tmem_sp1, tmem_o]
        + ([] if smem_q.cfg.stats_via_smem else [tmem_vec_done_0, tmem_vec_done_1]),
        warp_idx=12,
        num_warps=1,
        schedule=captured_schedule,
        name="MmaTask",
        num_registers=smem_q.cfg.num_regs_other,
        **task_kwargs,
    )


def create_softmax_task(
    index: int,
    tmem_sp: TmemSPResource,
    tmem_vec: TmemStatsResource,
    tmem_p: TmemPResource | None,
    s0s1_seq: S0S1SequenceResource | None,
    work_queue: WorkQueue | None,
    task_class: type[Task] = Task,
    **task_kwargs: Any,
) -> Task:
    """Create a four-warp Softmax task.

    index=0: warps 0-3 (Softmax0Task) — S0-S1 producer (acquire/commit)
    index=1: warps 4-7 (Softmax1Task) — S0-S1 consumer (wait/release)

    Args:
        task_class: Task subclass used to instantiate the softmax schedule.
    """
    loop_start, loop_end, loop_step = _captured_loop_bounds(task_class, task_kwargs)
    skip_work_tile_if = _packed_context_skip_predicate(work_queue)

    # A missing S0-S1 sequence resource selects the single-QKV-instance path.
    # For D>128, the TMEM P pipeline gives S and P independent readiness
    # handoffs. MMA can issue next-tile QK on one stage while previous-tile PV
    # uses the other.
    if s0s1_seq is None:
        if tmem_p is not None and tmem_sp.cfg.has_tmem_p_pipeline:
            src = _src_resources(tmem_sp, work_queue=work_queue)
            dst = [tmem_vec, tmem_p]

            @schedule
            def softmax_schedule(
                sp: TmemSPResource,
                vec: TmemStatsResource,
                tp: TmemPResource,
                wq: WorkQueue | None = None,
            ) -> None:
                p_chunk = sp.init_softmax_state()
                scale_softmax_log2 = sp.load_scale_softmax_log2()
                vec.init_store_state()
                with _work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
                    old_row_max, row_max, row_sum, q_offset = (
                        sp.init_softmax_work_tile_state()
                    )
                    vec.init_store_work_tile_state()
                    if tmem_sp.uses_varlen_q_offset_cache:
                        q_offset = sp.cache_q_offset()
                    if tmem_sp.uses_packed_dense_k_mask:
                        seqlen_k = sp.cache_seqlen_k()
                    window_start = Int32(0)
                    window_end = Int32(0)
                    if tmem_sp.uses_variable_window:
                        window_start, window_end = sp.cache_variable_window_bounds()
                    vec.acquire()
                    with domain_loop(loop_start, loop_end, loop_step):
                        # Softmax(i): wait for QK(Q,Ki) -> S(i).
                        sp.wait()
                        if tmem_sp.uses_variable_window:
                            old_row_max, row_max = sp.variable_window_row_max(
                                row_max=row_max,
                                window_start=window_start,
                                window_end=window_end,
                            )
                        elif tmem_sp.uses_left_window_loop_mask:
                            old_row_max, row_max = sp.left_masked_row_max(
                                row_max=row_max,
                                q_offset=q_offset,
                            )
                        elif tmem_sp.uses_varlen_loop_right_mask:
                            old_row_max, row_max = sp.right_masked_row_max(
                                row_max=row_max,
                                q_offset=q_offset,
                                section=FmhaStage.Loop,
                            )
                        elif tmem_sp.uses_query_paired_q_offset_loop_mask:
                            old_row_max, row_max = sp.loop_masked_row_max(
                                row_max=row_max,
                                q_offset=q_offset,
                            )
                        elif tmem_sp.uses_fixed_dense_k_tail_mask:
                            old_row_max, row_max = sp.fixed_dense_k_tail_masked_row_max(
                                row_max=row_max,
                            )
                        elif tmem_sp.uses_packed_dense_k_mask:
                            old_row_max, row_max = sp.packed_dense_k_masked_row_max(
                                row_max=row_max,
                                seqlen_k=seqlen_k,
                                section=FmhaStage.Loop,
                            )
                        else:
                            old_row_max, row_max = sp.compute_row_max(row_max=row_max)
                        # Stats(i): S(i) -> row max/sum for correction.
                        vec.store_vec(
                            old_row_max=old_row_max,
                            row_max=row_max,
                            row_sum=row_sum,
                        )
                        vec.commit()
                        # P(i): acquire the matching P-ready handoff stage.
                        tp.acquire()
                        # P(i): exp2(S(i)) -> P(i) in the same TMEM stage.
                        p_chunk = sp.exp2_p(
                            row_max=row_max,
                            scale_softmax_log2=scale_softmax_log2,
                        )
                        # P(i) ready: P(i) -> PV(Pi,Vi).
                        tp.commit()
                        # S/P(i): release softmax ownership for the next QK stage.
                        sp.release()
                        # Aux(i): finish the row-sum reduction after releasing SP.
                        row_sum = sp.softmax_aux_reduce(
                            old_row_max=old_row_max,
                            row_max=row_max,
                            row_sum=row_sum,
                            p_chunk=p_chunk,
                            scale_softmax_log2=scale_softmax_log2,
                        )
                        vec.acquire()

                    if tmem_sp.uses_head_paired_causal_tail_mask:
                        # Tail S: consume and mask the final head-paired score tile.
                        sp.wait()
                        old_row_max, row_max = sp.right_masked_row_max(
                            row_max=row_max,
                            q_offset=q_offset,
                            section=FmhaStage.Tail,
                        )
                        vec.store_vec(
                            old_row_max=old_row_max,
                            row_max=row_max,
                            row_sum=row_sum,
                        )
                        vec.commit()
                        # Tail P: publish the final probability tile to PV.
                        tp.acquire()
                        p_chunk = sp.exp2_p(
                            row_max=row_max,
                            scale_softmax_log2=scale_softmax_log2,
                        )
                        tp.commit()
                        sp.release()
                        row_sum = sp.softmax_aux_reduce(
                            old_row_max=old_row_max,
                            row_max=row_max,
                            row_sum=row_sum,
                            p_chunk=p_chunk,
                            scale_softmax_log2=scale_softmax_log2,
                        )
                        vec.acquire()
                        # Drain the final SP/P-ready slots and publish identity stats.
                        sp.wait()
                        sp.release()
                        tp.acquire()
                        tp.commit()
                        old_row_max = sp.softmax_aux_identity(row_max=row_max)
                        vec.store_vec(
                            old_row_max=old_row_max,
                            row_max=row_max,
                            row_sum=row_sum,
                            final_stats=True,
                        )
                        vec.commit()
                    elif tmem_sp.uses_query_paired_causal_tail_mask:
                        # Tail S: consume and mask the final query-paired score tile.
                        sp.wait()
                        old_row_max, row_max = sp.masked_row_max(
                            row_max=row_max,
                            q_offset=q_offset,
                        )
                        vec.store_vec(
                            old_row_max=old_row_max,
                            row_max=row_max,
                            row_sum=row_sum,
                        )
                        vec.commit()
                        # Tail P: publish the final probability tile to PV.
                        tp.acquire()
                        p_chunk = sp.masked_exp2_p(
                            row_max=row_max,
                            scale_softmax_log2=scale_softmax_log2,
                        )
                        tp.commit()
                        sp.release()
                        row_sum = sp.softmax_aux_reduce(
                            old_row_max=old_row_max,
                            row_max=row_max,
                            row_sum=row_sum,
                            p_chunk=p_chunk,
                            scale_softmax_log2=scale_softmax_log2,
                        )
                        if tmem_sp.uses_query_paired_invalid_tail:
                            # Invalid peer tail: consume its padded SP slot without PV.
                            sp.wait()
                            old_row_max, row_max = sp.invalid_row_max(row_max=row_max)
                            vec.acquire()
                            vec.store_vec(
                                old_row_max=old_row_max,
                                row_max=row_max,
                                row_sum=row_sum,
                            )
                            vec.commit()
                            sp.invalid_exp2_p(row_max=row_max)
                            sp.release()
                        # Drain the final SP/P-ready slots and publish identity stats.
                        sp.wait()
                        sp.release()
                        tp.acquire()
                        tp.commit()
                        old_row_max = sp.softmax_aux_identity(row_max=row_max)
                        vec.acquire()
                        vec.store_vec(
                            old_row_max=old_row_max,
                            row_max=row_max,
                            row_sum=row_sum,
                            final_stats=True,
                        )
                        vec.commit()
                    elif tmem_sp.cfg.is_causal:
                        # Tail S: consume and causally mask the final score tile.
                        sp.wait()
                        old_row_max, row_max = sp.masked_row_max(
                            row_max=row_max,
                            q_offset=q_offset,
                        )
                        vec.store_vec(
                            old_row_max=old_row_max,
                            row_max=row_max,
                            row_sum=row_sum,
                        )
                        vec.commit()
                        # Tail P: publish the final probability tile to PV.
                        tp.acquire()
                        p_chunk = sp.masked_exp2_p(
                            row_max=row_max,
                            scale_softmax_log2=scale_softmax_log2,
                        )
                        tp.commit()
                        sp.release()
                        row_sum = sp.softmax_aux_reduce(
                            old_row_max=old_row_max,
                            row_max=row_max,
                            row_sum=row_sum,
                            p_chunk=p_chunk,
                            scale_softmax_log2=scale_softmax_log2,
                        )
                        # Drain the final SP/P-ready slots and publish identity stats.
                        sp.wait()
                        sp.release()
                        tp.acquire()
                        tp.commit()
                        old_row_max = sp.softmax_aux_identity(row_max=row_max)
                        vec.acquire()
                        vec.store_vec(
                            old_row_max=old_row_max,
                            row_max=row_max,
                            row_sum=row_sum,
                            final_stats=True,
                        )
                        vec.commit()
                    else:
                        # Non-causal cleanup: drain SP and the matching P-ready slot.
                        sp.wait()
                        sp.release()
                        tp.acquire()
                        tp.commit()
                        old_row_max = sp.softmax_aux_identity(row_max=row_max)
                        vec.store_vec(
                            old_row_max=old_row_max,
                            row_max=row_max,
                            row_sum=row_sum,
                            final_stats=True,
                        )
                        vec.commit()
                    if tmem_sp.cfg.stats_via_smem:
                        # Balance the two-stage stats cursor before the next
                        # captured persistent work tile.  The context task
                        # runtime carries pipeline state across work tiles,
                        # while each tile's static call layout begins at the
                        # same stage; the empty record keeps both in phase.
                        vec.acquire()
                        vec.store_vec(
                            old_row_max=old_row_max,
                            row_max=row_max,
                            row_sum=row_sum,
                        )
                        vec.commit()

            captured_schedule = _schedule_with_work_queue(
                softmax_schedule, tmem_sp, tmem_vec, tmem_p, work_queue=work_queue
            )
            return task_class(
                src_resources=src,
                dst_resources=dst,
                warp_idx=index * 4,
                num_warps=4,
                schedule=captured_schedule,
                num_registers=tmem_sp.cfg.num_regs_softmax,
                name=f"Softmax{index}Task",
                **task_kwargs,
            )

        # Non-split single-instance fallback: softmax writes P into the current
        # SP stage and releases that same resource for MMA to consume directly.
        src = _src_resources(tmem_sp, work_queue=work_queue)
        dst = [tmem_vec]

        @schedule
        def softmax_schedule(
            sp: TmemSPResource,
            vec: TmemStatsResource,
            wq: WorkQueue | None = None,
        ) -> None:
            old_row_max, row_max, row_sum, p_chunk, q_offset = (
                sp.create_function_variables()
            )
            vec.create_function_variables()
            with _work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
                if wq is not None:
                    old_row_max, row_max, row_sum, q_offset = (
                        sp.create_work_tile_variables(
                            old_row_max=old_row_max,
                            row_max=row_max,
                            row_sum=row_sum,
                            q_offset=q_offset,
                        )
                    )
                    vec.create_work_tile_variables()
                if tmem_sp.uses_varlen_q_offset_cache:
                    q_offset = sp.cache_q_offset()
                if tmem_sp.uses_packed_dense_k_mask:
                    seqlen_k = sp.cache_seqlen_k()
                window_start = Int32(0)
                window_end = Int32(0)
                if tmem_sp.uses_variable_window:
                    window_start, window_end = sp.cache_variable_window_bounds()
                vec.acquire()
                with domain_loop(loop_start, loop_end, loop_step):
                    sp.wait()
                    if tmem_sp.uses_variable_window:
                        old_row_max, row_max = sp.variable_window_row_max(
                            row_max=row_max,
                            window_start=window_start,
                            window_end=window_end,
                        )
                    elif tmem_sp.uses_left_window_loop_mask:
                        old_row_max, row_max = sp.left_masked_row_max(
                            row_max=row_max,
                            q_offset=q_offset,
                        )
                    elif tmem_sp.uses_varlen_loop_right_mask:
                        old_row_max, row_max = sp.right_masked_row_max(
                            row_max=row_max,
                            q_offset=q_offset,
                            section=FmhaStage.Loop,
                        )
                    elif tmem_sp.uses_query_paired_q_offset_loop_mask:
                        old_row_max, row_max = sp.loop_masked_row_max(
                            row_max=row_max,
                            q_offset=q_offset,
                        )
                    elif tmem_sp.uses_fixed_dense_k_tail_mask:
                        old_row_max, row_max = sp.fixed_dense_k_tail_masked_row_max(
                            row_max=row_max
                        )
                    elif tmem_sp.uses_packed_dense_k_mask:
                        old_row_max, row_max = sp.packed_dense_k_masked_row_max(
                            row_max=row_max,
                            seqlen_k=seqlen_k,
                            section=FmhaStage.Loop,
                        )
                    else:
                        old_row_max, row_max = sp.row_max(row_max)
                    vec.store_vec(
                        old_row_max,
                        row_max,
                        row_sum,
                    )
                    vec.commit()
                    p_chunk = sp.exp2_p(row_max)
                    sp.release()
                    row_sum = sp.softmax_post_release_reduce(
                        old_row_max, row_max, row_sum, p_chunk
                    )
                    vec.acquire()

                if tmem_sp.uses_head_paired_causal_tail_mask:
                    sp.wait()
                    old_row_max, row_max = sp.right_masked_row_max(
                        row_max=row_max,
                        q_offset=q_offset,
                        section=FmhaStage.Tail,
                    )
                    vec.store_vec(
                        old_row_max,
                        row_max,
                        row_sum,
                    )
                    vec.commit()
                    p_chunk = sp.exp2_p(row_max)
                    sp.release()
                    row_sum = sp.softmax_post_release_reduce(
                        old_row_max, row_max, row_sum, p_chunk
                    )
                    vec.acquire()
                    sp.wait()
                    sp.release()
                    old_row_max = sp.softmax_post_release_identity(row_max)
                    vec.store_vec(
                        old_row_max,
                        row_max,
                        row_sum,
                        final_stats=True,
                    )
                    vec.commit()
                elif tmem_sp.uses_query_paired_causal_tail_mask:
                    sp.wait()
                    old_row_max, row_max = sp.masked_row_max(
                        row_max=row_max,
                        q_offset=q_offset,
                    )
                    vec.store_vec(old_row_max, row_max, row_sum)
                    vec.commit()
                    p_chunk = sp.masked_exp2_p(
                        row_max=row_max,
                    )
                    sp.release()
                    row_sum = sp.softmax_post_release_reduce(
                        old_row_max, row_max, row_sum, p_chunk
                    )
                    if tmem_sp.uses_query_paired_invalid_tail:
                        sp.wait()
                        old_row_max, row_max = sp.invalid_row_max(row_max)
                        vec.acquire()
                        vec.store_vec(old_row_max, row_max, row_sum)
                        vec.commit()
                        sp.invalid_exp2_p(row_max=row_max)
                        sp.release()
                    sp.wait()
                    sp.release()
                    old_row_max = sp.softmax_post_release_identity(row_max)
                    vec.acquire()
                    vec.store_vec(
                        old_row_max,
                        row_max,
                        row_sum,
                        final_stats=True,
                    )
                    vec.commit()
                elif tmem_sp.cfg.is_causal:
                    sp.wait()
                    old_row_max, row_max = sp.masked_row_max(
                        row_max=row_max,
                        q_offset=q_offset,
                    )
                    vec.store_vec(old_row_max, row_max, row_sum)
                    vec.commit()
                    p_chunk = sp.masked_exp2_p(
                        row_max=row_max,
                    )
                    sp.release()
                    row_sum = sp.softmax_post_release_reduce(
                        old_row_max, row_max, row_sum, p_chunk
                    )
                    sp.wait()
                    sp.release()
                    old_row_max = sp.softmax_post_release_identity(row_max)
                    vec.acquire()
                    vec.store_vec(
                        old_row_max,
                        row_max,
                        row_sum,
                        final_stats=True,
                    )
                    vec.commit()
                else:
                    sp.wait()
                    sp.release()
                    old_row_max = sp.softmax_post_release_identity(row_max)
                    vec.store_vec(old_row_max, row_max, row_sum)
                    vec.commit()

        captured_schedule = _schedule_with_work_queue(
            softmax_schedule, tmem_sp, tmem_vec, work_queue=work_queue
        )
        return task_class(
            src_resources=src,
            dst_resources=dst,
            warp_idx=index * 4,
            num_warps=4,
            schedule=captured_schedule,
            num_registers=tmem_sp.cfg.num_regs_softmax,
            name=f"Softmax{index}Task",
            **task_kwargs,
        )

    # Paired QKV instances use separate SP resources. S0S1SequenceResource
    # orders their P stores, so this path does not need the TMEM P handoff.
    if s0s1_seq is not None and index == 1:
        src = _src_resources(tmem_sp, s0s1_seq, work_queue=work_queue)
    else:
        src = _src_resources(tmem_sp, work_queue=work_queue)
    dst = [tmem_vec]
    if s0s1_seq is not None and index == 0:
        dst.append(s0s1_seq)

    @schedule
    def softmax_schedule(
        sp: TmemSPResource,
        vec: TmemStatsResource,
        seq: S0S1SequenceResource,
        wq: WorkQueue | None = None,
    ) -> None:
        """Captured schedule for one softmax warp group."""
        if tmem_sp.enable_early_tile_sum:
            # The contribution is produced and consumed inside each iteration;
            # do not carry even the scalar tile sum through the persistent loop.
            sp.init_softmax_state_early()
        else:
            p_chunk = sp.init_softmax_state()
        scale_softmax_log2 = sp.load_scale_softmax_log2()
        vec.init_store_state()
        with _work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
            # Recompute per-tile SP/Vec TMEM state.
            old_row_max, row_max, row_sum, q_offset = sp.init_softmax_work_tile_state()
            vec.init_store_work_tile_state()
            if tmem_sp.uses_varlen_q_offset_cache:
                q_offset = sp.cache_q_offset()
            if tmem_sp.uses_packed_dense_k_mask:
                seqlen_k = sp.cache_seqlen_k()
            window_start = Int32(0)
            window_end = Int32(0)
            if tmem_sp.uses_variable_window:
                window_start, window_end = sp.cache_variable_window_bounds()
            # Reserve a stats slot before the first softmax result is published.
            vec.acquire()
            with domain_loop(loop_start, loop_end, loop_step):
                sp.wait()
                # Compute row max and publish vec.
                if tmem_sp.uses_variable_window:
                    old_row_max, row_max = sp.variable_window_row_max(
                        row_max=row_max,
                        window_start=window_start,
                        window_end=window_end,
                    )
                elif tmem_sp.uses_left_window_loop_mask:
                    old_row_max, row_max = sp.left_masked_row_max(
                        row_max=row_max,
                        q_offset=q_offset,
                    )
                elif tmem_sp.uses_varlen_loop_right_mask:
                    old_row_max, row_max = sp.right_masked_row_max(
                        row_max=row_max,
                        q_offset=q_offset,
                        section=FmhaStage.Loop,
                    )
                elif tmem_sp.uses_query_paired_q_offset_loop_mask:
                    old_row_max, row_max = sp.loop_masked_row_max(
                        row_max=row_max,
                        q_offset=q_offset,
                    )
                elif tmem_sp.uses_fixed_dense_k_tail_mask:
                    old_row_max, row_max = sp.fixed_dense_k_tail_masked_row_max(
                        row_max=row_max,
                    )
                elif tmem_sp.uses_packed_dense_k_mask:
                    old_row_max, row_max = sp.packed_dense_k_masked_row_max(
                        row_max=row_max,
                        seqlen_k=seqlen_k,
                        section=FmhaStage.Loop,
                    )
                else:
                    old_row_max, row_max = sp.compute_row_max(row_max=row_max)
                vec.store_vec(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                )
                vec.commit()
                if s0s1_seq is None:
                    pass
                elif index == 0:
                    # Softmax0 is the S0-S1 producer: acquire/commit sequence.
                    seq.acquire()
                else:
                    # Softmax1 is the S0-S1 consumer: wait/release sequence.
                    seq.wait()
                # Apply softmax and write P.
                p_chunk = sp.exp2_p(
                    row_max=row_max,
                    scale_softmax_log2=scale_softmax_log2,
                )
                if s0s1_seq is None:
                    pass
                elif index == 0:
                    seq.commit()
                else:
                    seq.release()
                sp.release()
                # Reduction.
                row_sum = sp.softmax_aux_reduce(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                    p_chunk=p_chunk,
                    scale_softmax_log2=scale_softmax_log2,
                )
                # Acquire vec for next iter.
                vec.acquire()

            if tmem_sp.uses_head_paired_causal_tail_mask:
                # Head-paired maps Q0/Q1 to adjacent Hq slices at the same S
                # tile. Its tail mask uses right_masked_row_max(), which does
                # not add the query-paired q_half * q_tile_m sequence advance.
                sp.wait()
                old_row_max, row_max = sp.right_masked_row_max(
                    row_max=row_max,
                    q_offset=q_offset,
                    section=FmhaStage.Tail,
                )
                vec.store_vec(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                )
                vec.commit()
                if s0s1_seq is None:
                    pass
                elif index == 0:
                    seq.acquire()
                else:
                    seq.wait()
                p_chunk = sp.exp2_p(
                    row_max=row_max,
                    scale_softmax_log2=scale_softmax_log2,
                )
                if s0s1_seq is None:
                    pass
                elif index == 0:
                    seq.commit()
                else:
                    seq.release()
                sp.release()
                row_sum = sp.softmax_aux_reduce(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                    p_chunk=p_chunk,
                    scale_softmax_log2=scale_softmax_log2,
                )
                vec.acquire()
                sp.wait()
                sp.release()
                old_row_max = sp.softmax_aux_identity(row_max=row_max)
                vec.store_vec(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                    final_stats=True,
                )
                vec.commit()
            elif tmem_sp.uses_query_paired_causal_tail_mask:
                # Query-paired maps Q1 to the next S tile. Its generic causal
                # tail uses masked_row_max(), which includes q_half * q_tile_m
                # so each peer tile is masked at the right sequence boundary.
                sp.wait()
                old_row_max, row_max = sp.masked_row_max(
                    row_max=row_max,
                    q_offset=q_offset,
                )
                vec.store_vec(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                )
                vec.commit()
                if s0s1_seq is not None:
                    seq.acquire()
                p_chunk = sp.masked_exp2_p(
                    row_max=row_max,
                    scale_softmax_log2=scale_softmax_log2,
                )
                if s0s1_seq is not None:
                    seq.commit()
                sp.release()
                row_sum = sp.softmax_aux_reduce(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                    p_chunk=p_chunk,
                    scale_softmax_log2=scale_softmax_log2,
                )
                if tmem_sp.uses_query_paired_invalid_tail:
                    sp.wait()
                    old_row_max, row_max = sp.invalid_row_max(row_max=row_max)
                    vec.acquire()
                    vec.store_vec(
                        old_row_max=old_row_max,
                        row_max=row_max,
                        row_sum=row_sum,
                    )
                    vec.commit()
                    if s0s1_seq is not None:
                        seq.acquire()
                    sp.invalid_exp2_p(row_max=row_max)
                    if s0s1_seq is not None:
                        seq.commit()
                    sp.release()
                sp.wait()
                sp.release()
                old_row_max = sp.softmax_aux_identity(row_max=row_max)
                vec.acquire()
                vec.store_vec(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                    final_stats=True,
                )
                vec.commit()
            elif tmem_sp.cfg.is_causal:
                # Causal softmax1 TAIL handles masked rows and cleanup.
                sp.wait()
                old_row_max, row_max = sp.masked_row_max(
                    row_max=row_max,
                    q_offset=q_offset,
                )
                vec.store_vec(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                )
                vec.commit()
                if s0s1_seq is not None:
                    seq.wait()
                p_chunk = sp.masked_exp2_p(
                    row_max=row_max,
                    scale_softmax_log2=scale_softmax_log2,
                )
                if s0s1_seq is not None:
                    seq.release()
                sp.release()
                row_sum = sp.softmax_aux_reduce(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                    p_chunk=p_chunk,
                    scale_softmax_log2=scale_softmax_log2,
                )
                sp.wait()
                sp.release()
                old_row_max = sp.softmax_aux_identity(row_max=row_max)
                vec.acquire()
                vec.store_vec(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                    final_stats=True,
                )
                vec.commit()
            else:
                # Non-causal TAIL commits the reserved stats slot and lets MMA
                # complete its cleanup path.
                sp.wait()
                sp.release()
                old_row_max = sp.softmax_aux_identity(row_max=row_max)
                vec.store_vec(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                    final_stats=True,
                )
                vec.commit()

    captured_schedule = _schedule_with_work_queue(
        softmax_schedule, tmem_sp, tmem_vec, s0s1_seq, work_queue=work_queue
    )
    return task_class(
        src_resources=src,
        dst_resources=dst,
        warp_idx=index * 4,
        num_warps=4,
        schedule=captured_schedule,
        num_registers=tmem_sp.cfg.num_regs_softmax,
        name=f"Softmax{index}Task",
        **task_kwargs,
    )


def create_correction_task(
    tmem_vec0: TmemStatsResource,
    tmem_vec1: TmemStatsResource | None,
    tmem_o: TmemOResource,
    smem_o_0: SmemOResource,
    smem_o_1: SmemOResource | None,
    gmem_o_0: GmemOResource,
    gmem_o_1: GmemOResource | None,
    tmem_vec_done_0: TmemStatsDoneResource,
    tmem_vec_done_1: TmemStatsDoneResource | None,
    work_queue: WorkQueue | None,
    task_class: type[Task] = Task,
    **task_kwargs: Any,
) -> Task:
    """Create the four-warp Correction task (warps 8-11)."""
    loop_start, loop_end, loop_step = _captured_loop_bounds(task_class, task_kwargs)
    skip_work_tile_if = _packed_context_skip_predicate(work_queue)

    def _create_single_instance_task() -> Task:
        fuse_epilogue = smem_o_0.cfg.fuse_epilogue_into_correction
        num_o_head_dim_stages = smem_o_0.cfg.num_o_head_dim_stages

        src = _src_resources(
            tmem_vec0,
            tmem_o,
            *([] if tmem_vec0.cfg.stats_via_smem else [tmem_vec_done_0]),
            *([smem_o_0] if fuse_epilogue else []),
            work_queue=work_queue,
        )

        @schedule
        def correction_schedule(
            v0: TmemStatsResource,
            to: TmemOResource,
            so0: SmemOResource,
            go0: GmemOResource,
            vd0: TmemStatsDoneResource,
            wq: WorkQueue | None = None,
        ) -> None:
            v0.init_read_state()
            scale_softmax_log2 = v0.load_scale_softmax_log2()
            output_scale = v0.load_output_scale()
            to.init_correction_state()
            so0.init_store_state()
            if fuse_epilogue:
                go0.init_store_state()
            with _work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
                v0.init_read_work_tile_state()
                to.init_correction_work_tile_state()
                so0.init_store_work_tile_state()

                v0.wait()
                v0.release()
                if not tmem_vec0.cfg.stats_via_smem:
                    vd0.wait()
                    vd0.release()
                with domain_loop(loop_start, loop_end, loop_step):
                    v0.wait()
                    vec_old_max, vec_new_max, _, vec_scale = v0.read_vec(
                        scale_softmax_log2=scale_softmax_log2,
                    )
                    if not tmem_vec0.cfg.stats_via_smem:
                        vd0.wait()
                        vd0.release()
                    to.wait()
                    to.correct(
                        vec_old_max=vec_old_max,
                        vec_new_max=vec_new_max,
                        vec_scale=vec_scale,
                        inst_idx=0,
                    )
                    v0.release()
                    to.release()

                v0.wait()
                _, _, vec_row_sum, vec_scale = v0.read_vec(
                    scale_softmax_log2=scale_softmax_log2,
                    final_stats=True,
                )
                if not tmem_vec0.cfg.stats_via_smem:
                    vd0.wait()
                    vd0.release()
                v0.release()
                to.wait()
                for head_dim_stage_idx in range(num_o_head_dim_stages):
                    so0.acquire()
                    so0.store_o(
                        vec_row_sum=vec_row_sum,
                        vec_scale=vec_scale,
                        output_scale=output_scale,
                        head_dim_stage_idx=head_dim_stage_idx,
                    )
                    so0.commit()
                    if fuse_epilogue:
                        # The same four-warp group consumes the completed SMEM
                        # stage.  Only its first warp issues TMA; all four wait
                        # and release the pipeline stage together.
                        so0.wait()
                        head_coord, batch_coord, seq_coord_q = (
                            so0.compute_output_coords()
                        )
                        go0.tma_store(
                            head_coord=head_coord,
                            batch_coord=batch_coord,
                            seq_coord_q=seq_coord_q,
                            head_dim_stage_idx=head_dim_stage_idx,
                            correction_fused=True,
                        )
                        so0.release()
                to.release()
                if tmem_vec0.cfg.stats_via_smem:
                    # Consume the cursor-balancing record emitted by Softmax.
                    v0.wait()
                    v0.release()

        captured_schedule = _schedule_with_work_queue(
            correction_schedule,
            tmem_vec0,
            tmem_o,
            smem_o_0,
            gmem_o_0,
            tmem_vec_done_0,
            work_queue=work_queue,
        )
        dst = [smem_o_0]
        if fuse_epilogue:
            dst.append(gmem_o_0)
        return task_class(
            src_resources=src,
            dst_resources=dst,
            warp_idx=smem_o_0.cfg.correction_warp_ids[0],
            num_warps=4,
            schedule=captured_schedule,
            num_registers=smem_o_0.cfg.num_regs_correction,
            name="CorrectionTask",
            **task_kwargs,
        )

    def _create_paired_task() -> Task:
        if tmem_vec1 is None or smem_o_1 is None or tmem_vec_done_1 is None:
            raise ValueError("paired correction scheduling requires peer-1 resources")

        src = _src_resources(
            tmem_vec0,
            tmem_vec1,
            tmem_o,
            *(
                []
                if tmem_vec0.cfg.stats_via_smem
                else [tmem_vec_done_0, tmem_vec_done_1]
            ),
            work_queue=work_queue,
        )

        @schedule
        def correction_schedule(
            v0: TmemStatsResource,
            v1: TmemStatsResource,
            to: TmemOResource,
            so0: SmemOResource,
            so1: SmemOResource,
            vd0: TmemStatsDoneResource,
            vd1: TmemStatsDoneResource,
            wq: WorkQueue | None = None,
        ) -> None:
            """Captured schedule for O rescale and SMEM staging."""
            v0.init_read_state()
            v1.init_read_state()
            scale_softmax_log2_v0 = v0.load_scale_softmax_log2()
            scale_softmax_log2_v1 = v1.load_scale_softmax_log2()
            output_scale0 = v0.load_output_scale()
            output_scale1 = v1.load_output_scale()
            to.init_correction_state()
            so0.init_store_state()
            so1.init_store_state()
            with _work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
                # Per-tile TMEM/SMEM cached addresses are computed here.
                v0.init_read_work_tile_state()
                v1.init_read_work_tile_state()
                to.init_correction_work_tile_state()
                so0.init_store_work_tile_state()
                so1.init_store_work_tile_state()
                # The empty stats pipeline needs no priming. Discard its first
                # slot and retain TmemStats1 for the first loop cross-release.
                v0.wait()
                v0.release()
                v1.wait()
                # The correction loop consumes vec/O pairs in alternating order so
                # each half can unblock the other half's next producer.
                with domain_loop(loop_start, loop_end, loop_step):
                    # Part 1: consume TmemStats0 + O0, release TmemStats1.
                    v0.wait()
                    vec_old_max, vec_new_max, _, vec_scale = v0.read_vec(
                        scale_softmax_log2=scale_softmax_log2_v0,
                    )
                    to.wait()
                    to.correct(
                        vec_old_max=vec_old_max,
                        vec_new_max=vec_new_max,
                        vec_scale=vec_scale,
                        inst_idx=0,
                    )
                    v1.release()
                    to.release()
                    # Part 2: consume TmemStats1 + O1, release TmemStats0.
                    v1.wait()
                    vec_old_max, vec_new_max, _, vec_scale = v1.read_vec(
                        scale_softmax_log2=scale_softmax_log2_v1,
                    )
                    to.wait()
                    to.correct(
                        vec_old_max=vec_old_max,
                        vec_new_max=vec_new_max,
                        vec_scale=vec_scale,
                        inst_idx=1,
                    )
                    v0.release()
                    to.release()
                # TAIL: consume remaining stats, release tmem-stats-done gates, and
                # stage corrected O0/O1 into SMEM for the epilogue task.
                v1.release()
                v0.wait()
                _, _, vec_row_sum, vec_scale = v0.read_vec(
                    scale_softmax_log2=scale_softmax_log2_v0,
                    final_stats=True,
                )
                if not tmem_vec0.cfg.stats_via_smem:
                    vd0.wait()
                    vd0.release()
                v0.release()
                to.wait()
                so0.acquire()
                so0.store_o(
                    vec_row_sum=vec_row_sum,
                    vec_scale=vec_scale,
                    output_scale=output_scale0,
                )
                so0.commit()
                to.release()
                v1.wait()
                _, _, vec_row_sum, vec_scale = v1.read_vec(
                    scale_softmax_log2=scale_softmax_log2_v1,
                    final_stats=True,
                )
                if not tmem_vec0.cfg.stats_via_smem:
                    vd1.wait()
                    vd1.release()
                v1.release()
                to.wait()
                so1.acquire()
                so1.store_o(
                    vec_row_sum=vec_row_sum,
                    vec_scale=vec_scale,
                    output_scale=output_scale1,
                )
                so1.commit()
                to.release()

        captured_schedule = _schedule_with_work_queue(
            correction_schedule,
            tmem_vec0,
            tmem_vec1,
            tmem_o,
            smem_o_0,
            smem_o_1,
            tmem_vec_done_0,
            tmem_vec_done_1,
            work_queue=work_queue,
        )
        return task_class(
            src_resources=src,
            dst_resources=[smem_o_0, smem_o_1],
            warp_idx=8,
            num_warps=4,
            schedule=captured_schedule,
            num_registers=smem_o_0.cfg.num_regs_correction,
            name="CorrectionTask",
            **task_kwargs,
        )

    if smem_o_0.cfg.single_qkv_instance:
        return _create_single_instance_task()
    return _create_paired_task()


def create_epilogue_task(
    smem_o_0: SmemOResource,
    smem_o_1: SmemOResource | None,
    gmem_o_0: GmemOResource,
    gmem_o_1: GmemOResource | None,
    work_queue: WorkQueue | None,
    task_class: type[Task] = Task,
    **task_kwargs: Any,
) -> Task:
    """Create the one-warp Epilogue store task (warp 14)."""
    loop_start, loop_end, loop_step = _captured_loop_bounds(task_class, task_kwargs)
    skip_work_tile_if = _packed_context_skip_predicate(work_queue)

    def _create_single_instance_task() -> Task:
        src = _src_resources(smem_o_0, work_queue=work_queue)
        num_o_head_dim_stages = smem_o_0.cfg.num_o_head_dim_stages

        @schedule
        def epilogue_schedule(
            so0: SmemOResource,
            go0: GmemOResource,
            wq: WorkQueue | None = None,
        ) -> None:
            so0.init_output_state()
            go0.init_store_state()
            with _work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
                so0.init_output_work_tile_state()
                with domain_loop(loop_start, loop_end, loop_step):
                    pass
                for head_dim_stage_idx in range(num_o_head_dim_stages):
                    so0.wait()
                    head_coord, batch_coord, seq_coord_q = so0.compute_output_coords()
                    go0.tma_store(
                        head_coord=head_coord,
                        batch_coord=batch_coord,
                        seq_coord_q=seq_coord_q,
                        head_dim_stage_idx=head_dim_stage_idx,
                    )
                    so0.release()

        captured_schedule = _schedule_with_work_queue(
            epilogue_schedule,
            smem_o_0,
            gmem_o_0,
            work_queue=work_queue,
        )
        return task_class(
            src_resources=src,
            dst_resources=[gmem_o_0],
            warp_idx=gmem_o_0.cfg.epilogue_warp_id,
            num_warps=1,
            schedule=captured_schedule,
            num_registers=gmem_o_0.cfg.num_regs_other,
            name="EpilogueTask",
            **task_kwargs,
        )

    def _create_paired_task() -> Task:
        if smem_o_1 is None or gmem_o_1 is None:
            raise ValueError("paired epilogue scheduling requires peer-1 resources")

        src = _src_resources(smem_o_0, smem_o_1, work_queue=work_queue)

        @schedule
        def epilogue_schedule(
            so0: SmemOResource,
            so1: SmemOResource,
            go0: GmemOResource,
            go1: GmemOResource,
            wq: WorkQueue | None = None,
        ) -> None:
            """Captured schedule for GMEM O stores."""
            so0.init_output_state()
            so1.init_output_state()
            go0.init_store_state()
            go1.init_store_state()
            with _work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
                # Per-tile SMEM O address base is computed in work vars.
                so0.init_output_work_tile_state()
                so1.init_output_work_tile_state()
                with domain_loop(loop_start, loop_end, loop_step):
                    pass
                # Store the first corrected O tile through gmem_o_0.
                so0.wait()
                head_coord, batch_coord, seq_coord_q = so0.compute_output_coords()
                go0.tma_store(
                    head_coord=head_coord,
                    batch_coord=batch_coord,
                    seq_coord_q=seq_coord_q,
                )
                so0.release()
                # Store the second corrected O tile through gmem_o_1.
                so1.wait()
                head_coord, batch_coord, seq_coord_q = so1.compute_output_coords()
                go1.tma_store(
                    head_coord=head_coord,
                    batch_coord=batch_coord,
                    seq_coord_q=seq_coord_q,
                )
                so1.release()

        captured_schedule = _schedule_with_work_queue(
            epilogue_schedule,
            smem_o_0,
            smem_o_1,
            gmem_o_0,
            gmem_o_1,
            work_queue=work_queue,
        )
        return task_class(
            src_resources=src,
            dst_resources=[gmem_o_0, gmem_o_1],
            warp_idx=14,
            num_warps=1,
            schedule=captured_schedule,
            num_registers=gmem_o_0.cfg.num_regs_other,
            name="EpilogueTask",
            **task_kwargs,
        )

    if smem_o_0.cfg.single_qkv_instance:
        return _create_single_instance_task()
    return _create_paired_task()


def create_padding_task(
    work_queue: WorkQueue | None,
    warp_idx: int = 15,
    num_warps: int = 1,
    num_registers: int = 32,
    name: str = "PaddingTask",
    task_class: type[Task] = Task,
    **task_kwargs: Any,
) -> Task:
    """Create the one-warp padding task (warp 15 in the D128 schedule).

    Required in ALL modes (persistent and non-persistent) because
    ``setmaxnreg.sync`` requires every warp in the warp group to
    participate. In D128, warps 12-15 form warp group 3; without the padding
    task its final warp never calls ``setmaxregister``, deadlocking the group.

    In persistent mode the task also consumes work_queue tiles so that
    the auxiliary warp participates in the persistent outer loop.

    In CLC dynamic mode, the padding task is replaced by a scheduler task
    (see ``create_scheduler_task``).
    """
    loop_start, loop_end, loop_step = _captured_loop_bounds(task_class, task_kwargs)
    skip_work_tile_if = _packed_context_skip_predicate(work_queue)
    src = _src_resources(work_queue=work_queue)

    @schedule
    def padding_schedule(wq: WorkQueue | None = None) -> None:
        """Captured schedule for warp-group register participation."""
        with (
            _work_tile_schedule_loop(wq, skip_if=skip_work_tile_if),
            domain_loop(loop_start, loop_end, loop_step),
        ):
            pass

    captured_schedule = _schedule_with_work_queue(
        padding_schedule, work_queue=work_queue
    )
    return task_class(
        src_resources=src,
        dst_resources=[],
        warp_idx=warp_idx,
        num_warps=num_warps,
        schedule=captured_schedule,
        num_registers=num_registers,
        name=name,
        **task_kwargs,
    )


def _prefetch_page_offsets_for_work_tile(
    spo: SmemPageOffsetsKvResource,
    *,
    kv_tile_start: Int32,
    kv_request_begin: Int32,
    kv_page_idx_ub: Int32,
    loop_start: int,
    loop_end: int,
    loop_step: int,
    staged_single_instance: bool,
) -> None:
    """Produce page-ID stages in the same logical order as the load task."""
    if staged_single_instance:
        # D>128 overlaps QK(i) with PV(i-1): K0, then K_i/V_{i-1}, then V_last.
        # One page-ID stage is shared by all head-dimension slices of a logical
        # K or V tile, so this producer fires once per tile rather than per slice.
        spo.acquire()
        spo.load_k(
            kv_tile_start=kv_tile_start,
            kv_request_begin=kv_request_begin,
            kv_page_idx_ub=kv_page_idx_ub,
        )
        spo.commit()
        with domain_loop(loop_start + 1, loop_end, loop_step):
            spo.acquire()
            spo.load_k(
                kv_tile_start=kv_tile_start,
                kv_request_begin=kv_request_begin,
                kv_page_idx_ub=kv_page_idx_ub,
            )
            spo.commit()
            spo.acquire()
            spo.load_v(
                previous=True,
                kv_tile_start=kv_tile_start,
                kv_request_begin=kv_request_begin,
                kv_page_idx_ub=kv_page_idx_ub,
            )
            spo.commit()
        spo.acquire()
        spo.load_v(
            previous=False,
            kv_tile_start=kv_tile_start,
            kv_request_begin=kv_request_begin,
            kv_page_idx_ub=kv_page_idx_ub,
        )
        spo.commit()
        return

    with domain_loop(loop_start, loop_end, loop_step):
        spo.acquire()
        spo.load_k(
            kv_tile_start=kv_tile_start,
            kv_request_begin=kv_request_begin,
            kv_page_idx_ub=kv_page_idx_ub,
        )
        spo.commit()
        spo.acquire()
        spo.load_v(
            kv_tile_start=kv_tile_start,
            kv_request_begin=kv_request_begin,
            kv_page_idx_ub=kv_page_idx_ub,
        )
        spo.commit()


def _prefetch_reused_page_windows_for_work_tile(
    spok: SmemPageOffsetsKvResource,
    spov: SmemPageOffsetsKvResource,
    *,
    kv_tile_start: Int32,
    kv_request_begin: Int32,
    kv_page_idx_ub: Int32,
    loop_start: object,
    loop_end: object,
    loop_step: object,
    page_window_period: int,
) -> None:
    """Publish independent K/V page windows at their structural cadence."""
    if (
        not isinstance(loop_start, int)
        or not isinstance(loop_end, int)
        or not isinstance(loop_step, int)
        or loop_start != 0
        or loop_step != 1
        or loop_end < page_window_period
        or loop_end % page_window_period != 0
    ):
        raise ValueError(
            "reused page windows require a compile-time K/V domain "
            "divisible by the topology-derived page-window period"
        )

    spok.acquire()
    spok.load_k(
        tile_offset=0,
        kv_tile_start=kv_tile_start,
        kv_request_begin=kv_request_begin,
        kv_page_idx_ub=kv_page_idx_ub,
    )
    spok.commit()
    spov.acquire()
    spov.load_v(
        tile_offset=0,
        kv_tile_start=kv_tile_start,
        kv_request_begin=kv_request_begin,
        kv_page_idx_ub=kv_page_idx_ub,
    )
    spov.commit()

    with domain_loop(page_window_period, loop_end, page_window_period):
        spok.acquire()
        spok.load_k(
            tile_offset=0,
            kv_tile_start=kv_tile_start,
            kv_request_begin=kv_request_begin,
            kv_page_idx_ub=kv_page_idx_ub,
        )
        spok.commit()
        spov.acquire()
        spov.load_v(
            tile_offset=0,
            kv_tile_start=kv_tile_start,
            kv_request_begin=kv_request_begin,
            kv_page_idx_ub=kv_page_idx_ub,
        )
        spov.commit()


def create_page_offsets_task(
    gmem_qkv: GmemQKVResource,
    smem_page_offsets_kv: SmemPageOffsetsKvResource,
    work_queue: WorkQueue | None,
    task_class: type[Task] = Task,
    num_registers: int = 32,
    smem_page_offsets_v: SmemPageOffsetsKvResource | None = None,
    **task_kwargs: Any,
) -> Task:
    """Create the one-warp paged-KV page-offsets prefetch task.

    Replaces ``create_padding_task`` when ``cfg.use_paged_kv`` is True. The
    configuration's empty/scheduler warp prefetches page-table entries into
    SMEM so the load warp can read cached page IDs when issuing paged TMA
    copies. This also preserves that warp's ``setmaxnreg.sync`` participation.

    Paired D128 CLC does not instantiate this task: its load warp reads page
    IDs directly, leaving warp 15 exclusively responsible for CLC. Staged
    D256 can use this task with CLC because page offsets run on the empty warp
    while the freed epilogue warp owns scheduling. No task therefore combines
    dynamic work-queue and page-offset production through the public DSL API.
    """
    loop_start, loop_end, loop_step = _captured_loop_bounds(task_class, task_kwargs)
    skip_work_tile_if = _packed_context_skip_predicate(work_queue)
    src = _src_resources(gmem_qkv, work_queue=work_queue)
    dst = [smem_page_offsets_kv]
    if smem_page_offsets_v is not None:
        dst.append(smem_page_offsets_v)
    staged_single_instance = (
        smem_page_offsets_kv.cfg.single_qkv_instance
        and smem_page_offsets_kv.cfg.has_tmem_p_pipeline
    )
    page_window_period = smem_page_offsets_kv.cfg.page_table_window_entries // (
        smem_page_offsets_kv.cfg.kv_tile_n
        // smem_page_offsets_kv.cfg.num_tokens_per_page
    )

    def page_offsets_schedule_body(
        gqkv: GmemQKVResource,
        spo: SmemPageOffsetsKvResource,
        spov: SmemPageOffsetsKvResource | None,
        wq: WorkQueue | None = None,
    ) -> None:
        """Captured schedule for K/V page-table prefetch."""
        spo.init_load_state()
        if spov is not None:
            spov.init_load_state()
        with _work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
            (
                kv_tile_start,
                kv_request_begin,
                kv_page_idx_ub,
            ) = gqkv.compute_page_coords()
            if spov is None:
                _prefetch_page_offsets_for_work_tile(
                    spo,
                    kv_tile_start=kv_tile_start,
                    kv_request_begin=kv_request_begin,
                    kv_page_idx_ub=kv_page_idx_ub,
                    loop_start=loop_start,
                    loop_end=loop_end,
                    loop_step=loop_step,
                    staged_single_instance=staged_single_instance,
                )
            else:
                _prefetch_reused_page_windows_for_work_tile(
                    spo,
                    spov,
                    kv_tile_start=kv_tile_start,
                    kv_request_begin=kv_request_begin,
                    kv_page_idx_ub=kv_page_idx_ub,
                    loop_start=loop_start,
                    loop_end=loop_end,
                    loop_step=loop_step,
                    page_window_period=page_window_period,
                )

    @schedule
    def page_offsets_schedule(
        gqkv: GmemQKVResource,
        spo: SmemPageOffsetsKvResource,
        wq: WorkQueue | None = None,
    ) -> None:
        page_offsets_schedule_body(gqkv, spo, None, wq)

    @schedule
    def reused_page_windows_schedule(
        gqkv: GmemQKVResource,
        spok: SmemPageOffsetsKvResource,
        spov: SmemPageOffsetsKvResource,
        wq: WorkQueue | None = None,
    ) -> None:
        page_offsets_schedule_body(gqkv, spok, spov, wq)

    if smem_page_offsets_v is None:
        captured_schedule = _schedule_with_work_queue(
            page_offsets_schedule,
            gmem_qkv,
            smem_page_offsets_kv,
            work_queue=work_queue,
        )
    else:
        captured_schedule = _schedule_with_work_queue(
            reused_page_windows_schedule,
            gmem_qkv,
            smem_page_offsets_kv,
            smem_page_offsets_v,
            work_queue=work_queue,
        )
    return task_class(
        src_resources=src,
        dst_resources=dst,
        warp_idx=smem_page_offsets_kv.cfg.empty_warp_id,
        num_warps=1,
        schedule=captured_schedule,
        num_registers=num_registers,
        name="PageTableTask",
        **task_kwargs,
    )


def create_scheduler_task(
    work_queue: WorkQueue,
    warp_idx: int = 15,
    num_registers: int = 32,
    task_class: type[Task] = Task,
    **task_kwargs: Any,
) -> Task:
    """Create the one-warp CLC scheduler task (warp 15 in D128).

    Replaces the padding task in CLC dynamic persistent mode.
    Issues CLC tile-fetch queries (producer side) and participates in
    the persistent outer loop.  Still satisfies the ``setmaxnreg.sync``
    requirement for the final warp group.
    """
    loop_start, loop_end, loop_step = _captured_loop_bounds(task_class, task_kwargs)

    @schedule
    def scheduler_schedule(wq: WorkQueue) -> None:
        """Captured schedule for CLC work-tile fetches."""
        with _work_tile_schedule_loop(wq):
            with domain_loop(loop_start, loop_end, loop_step):
                pass
            # Producer side: issue CLC tile-fetch query.
            wq.acquire()
            wq.fetch_work_tile()
            wq.commit()

    captured_schedule = scheduler_schedule(work_queue)
    return task_class(
        src_resources=[work_queue],
        dst_resources=[work_queue],
        warp_idx=warp_idx,
        num_warps=1,
        schedule=captured_schedule,
        num_registers=num_registers,
        name="SchedulerTask",
        **task_kwargs,
    )
