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
    SmemPResource,
    TmemSPResource,
    TmemStatsResource,
    TmemStatsDoneResource,
    TmemPPrefixReadyResource,
)
from .vc_resources import SmemMuResource


@dataclass(kw_only=True)
class PackedContextWorkQueue(WorkQueue):
    """Persistent queue that skips Q tiles outside a live packed request."""

    cfg: cutlass.Constexpr[FmhaConfig] = field(init=False, default=None)
    cum_seqlen_q: Any = field(init=False, default=None)

    def __init__(
        self,
        cfg: FmhaConfig,
        cum_seqlen_q: cute.Tensor,
        **kwargs: Any,
    ) -> None:
        """Attach the live packed-Q metadata used by the skip predicate."""
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
    """Select the live-Q skip predicate before schedule capture creates proxies."""
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
    smem_k_or_kv: SmemKVResource,
    smem_v: SmemKVResource | None,
    work_queue: WorkQueue | None,
    task_class: type[Task] = Task,
    smem_page_offsets_kv: SmemPageOffsetsKvResource | None = None,
    smem_page_offsets_v: SmemPageOffsetsKvResource | None = None,
    smem_mu: SmemMuResource | None = None,
    **task_kwargs: Any,
) -> Task:
    """Create the one-warp TMA load task.

    ``smem_mu`` (VC-Attention-QK16) stages one bf16 tile-mean operand per K/V tile
    beside V.

    When ``smem_page_offsets_kv`` is provided, each K/V TMA load consumes page
    IDs prefetched by the auxiliary warp through the ordinary asynchronous
    page-offset pipeline.
    """
    loop_start, loop_end, loop_step = _captured_loop_bounds(task_class, task_kwargs)
    skip_work_tile_if = _packed_context_skip_predicate(work_queue)
    src = _src_resources(gmem_qkv, work_queue=work_queue)
    # One K/V resource when they share a buffer, two when their dtypes split it.
    split_kv = smem_k_or_kv.cfg.split_kv_pipelines
    if split_kv and smem_v is None:
        raise ValueError("split K/V staging requires a separate V buffer")
    kv_resources = (smem_k_or_kv, smem_v) if split_kv else (smem_k_or_kv,)
    dst = [smem_q, *kv_resources]
    if smem_mu is not None:
        dst.append(smem_mu)
    if smem_page_offsets_kv is not None:
        src.append(smem_page_offsets_kv)
    if smem_page_offsets_v is not None:
        src.append(smem_page_offsets_v)
    if smem_q.cfg.single_qkv_instance and smem_q.cfg.has_tmem_p_pipeline:
        num_head_dim_stages_k = smem_k_or_kv.cfg.num_head_dim_stages_k
        num_head_dim_stages_v = smem_k_or_kv.cfg.num_head_dim_stages_v

        if smem_page_offsets_v is not None:
            if smem_page_offsets_kv is None:
                raise ValueError("a V page window requires a matching K page window")
            pages_per_tile = (
                smem_k_or_kv.cfg.kv_tile_n // smem_k_or_kv.cfg.num_tokens_per_page
            )
            page_window_period = (
                smem_k_or_kv.cfg.page_table_window_entries // pages_per_tile
            )
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
                sk: SmemKVResource,
                sv: SmemKVResource,
                spok: SmemPageOffsetsKvResource,
                spov: SmemPageOffsetsKvResource,
                wq: WorkQueue | None,
            ) -> None:
                """Load staged K/V while retaining each page-ID window."""
                sq.init_load_state()
                sk.init_load_state()
                sv.init_load_state()
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
                        """Issue the K TMA loads for one tile across the K head-dim stages."""
                        for head_dim_stage_idx in range(num_head_dim_stages_k):
                            sk.try_acquire()
                            sk.acquire()
                            sk.k_load_stage(
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
                            sk.commit()

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
                        """Issue the V TMA loads for one tile across the V head-dim stages.

                        With ``reuse_cached_page_ids`` the loads reuse page IDs staged by
                        ``cache_v_tile`` instead of re-reading the page table.
                        """
                        for head_dim_stage_idx in range(num_head_dim_stages_v):
                            sv.try_acquire()
                            sv.acquire()
                            if reuse_cached_page_ids:
                                sv.v_load_stage_cached(
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
                                sv.v_load_stage(
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
                            sv.commit()

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
                """Shared-buffer captured schedule."""
                load_reused_page_windows_schedule_body(
                    gqkv, sq, skv, skv, spok, spov, wq
                )

            @schedule
            def load_reused_page_windows_split_schedule(
                gqkv: GmemQKVResource,
                sq: SmemQResource,
                sk: SmemKVResource,
                sv: SmemKVResource,
                spok: SmemPageOffsetsKvResource,
                spov: SmemPageOffsetsKvResource,
                wq: WorkQueue | None = None,
            ) -> None:
                """Split K/V captured schedule."""
                load_reused_page_windows_schedule_body(gqkv, sq, sk, sv, spok, spov, wq)

            captured_schedule = _schedule_with_work_queue(
                load_reused_page_windows_split_schedule
                if split_kv
                else load_reused_page_windows_schedule,
                gmem_qkv,
                smem_q,
                *kv_resources,
                smem_page_offsets_kv,
                smem_page_offsets_v,
                work_queue=work_queue,
            )
            return task_class(
                src_resources=src,
                dst_resources=dst,
                warp_idx=smem_k_or_kv.cfg.load_warp_id,
                num_warps=1,
                schedule=captured_schedule,
                num_registers=smem_k_or_kv.cfg.num_regs_other,
                name="LoadTask",
                **task_kwargs,
            )

        def load_schedule_body(
            gqkv: GmemQKVResource,
            sq: SmemQResource,
            sk: SmemKVResource,
            sv: SmemKVResource,
            spo: SmemPageOffsetsKvResource | None,
            wq: WorkQueue | None,
        ) -> None:
            """Load-warp schedule: stage Q, then stream K and V tiles through their rings.

            K runs one tile ahead of V so QK(i+1) can start while PV(i) waits for V.
            Page offsets are read once per tile when paged.
            """
            sq.init_load_state()
            sk.init_load_state()
            sv.init_load_state()
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
                    sk.try_acquire()
                    if spo is not None and head_dim_stage_idx == 0:
                        spo.wait()
                        spo.read_offsets()
                    sk.acquire()
                    sk.k_load_stage(
                        stage_id=head_dim_stage_idx,
                        kv_head_coord=kv_head_coord,
                        batch_coord=batch_coord,
                        cuseqlen_k=cuseqlen_k,
                        seqlen_k=seqlen_k,
                        kv_tile_start=kv_tile_start,
                        kv_request_begin=kv_request_begin,
                        kv_page_idx_ub=kv_page_idx_ub,
                    )
                    sk.commit()
                if spo is not None:
                    spo.release()

                with domain_loop(loop_start + 1, loop_end, loop_step):
                    for head_dim_stage_idx in range(num_head_dim_stages_k):
                        sk.try_acquire()
                        if spo is not None and head_dim_stage_idx == 0:
                            spo.wait()
                            spo.read_offsets()
                        sk.acquire()
                        sk.k_load_stage(
                            stage_id=head_dim_stage_idx,
                            kv_head_coord=kv_head_coord,
                            batch_coord=batch_coord,
                            cuseqlen_k=cuseqlen_k,
                            seqlen_k=seqlen_k,
                            kv_tile_start=kv_tile_start,
                            kv_request_begin=kv_request_begin,
                            kv_page_idx_ub=kv_page_idx_ub,
                        )
                        sk.commit()
                    if spo is not None:
                        spo.release()

                    for head_dim_stage_idx in range(num_head_dim_stages_v):
                        sv.try_acquire()
                        if spo is not None and head_dim_stage_idx == 0:
                            spo.wait()
                            spo.read_offsets()
                        sv.acquire()
                        sv.v_load_stage(
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
                        sv.commit()
                    if spo is not None:
                        spo.release()

                for head_dim_stage_idx in range(num_head_dim_stages_v):
                    sv.try_acquire()
                    if spo is not None and head_dim_stage_idx == 0:
                        spo.wait()
                        spo.read_offsets()
                    sv.acquire()
                    sv.v_load_stage(
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
                    sv.commit()
                if spo is not None:
                    spo.release()

        @schedule
        def load_schedule(
            gqkv: GmemQKVResource,
            sq: SmemQResource,
            skv: SmemKVResource,
            wq: WorkQueue | None = None,
        ) -> None:
            """Shared-buffer captured schedule."""
            load_schedule_body(gqkv, sq, skv, skv, None, wq)

        @schedule
        def load_split_schedule(
            gqkv: GmemQKVResource,
            sq: SmemQResource,
            sk: SmemKVResource,
            sv: SmemKVResource,
            wq: WorkQueue | None = None,
        ) -> None:
            """Split K/V captured schedule."""
            load_schedule_body(gqkv, sq, sk, sv, None, wq)

        @schedule
        def load_page_offsets_schedule(
            gqkv: GmemQKVResource,
            sq: SmemQResource,
            skv: SmemKVResource,
            spo: SmemPageOffsetsKvResource,
            wq: WorkQueue | None = None,
        ) -> None:
            """Shared-buffer captured schedule with a page-ID ring."""
            load_schedule_body(gqkv, sq, skv, skv, spo, wq)

        @schedule
        def load_page_offsets_split_schedule(
            gqkv: GmemQKVResource,
            sq: SmemQResource,
            sk: SmemKVResource,
            sv: SmemKVResource,
            spo: SmemPageOffsetsKvResource,
            wq: WorkQueue | None = None,
        ) -> None:
            """Split K/V captured schedule with a page-ID ring."""
            load_schedule_body(gqkv, sq, sk, sv, spo, wq)

        if smem_page_offsets_kv is None:
            captured_schedule = _schedule_with_work_queue(
                load_split_schedule if split_kv else load_schedule,
                gmem_qkv,
                smem_q,
                *kv_resources,
                work_queue=work_queue,
            )
        else:
            captured_schedule = _schedule_with_work_queue(
                load_page_offsets_split_schedule
                if split_kv
                else load_page_offsets_schedule,
                gmem_qkv,
                smem_q,
                *kv_resources,
                smem_page_offsets_kv,
                work_queue=work_queue,
            )
        return task_class(
            src_resources=src,
            dst_resources=dst,
            warp_idx=smem_k_or_kv.cfg.load_warp_id,
            num_warps=1,
            schedule=captured_schedule,
            num_registers=smem_k_or_kv.cfg.num_regs_other,
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
        sk: SmemKVResource,
        sv: SmemKVResource,
        wq: WorkQueue | None,
        smu: SmemMuResource | None = None,
    ) -> None:
        """Load paired Q instances and their directly addressed K/V tiles."""
        sq.init_load_state()
        sk.init_load_state()
        sv.init_load_state()
        if smu is not None:
            smu.init_load_state()
        with _work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):  # noqa: SIM117
            # The first K-loop iteration also loads Q0/Q1. Later iterations
            # only stream the next K/V tiles through their respective pipelines.
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
                if smem_k_or_kv.cfg.stage_kv_by_head_dim:
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
                    for head_dim_stage_idx in range(
                        smem_k_or_kv.cfg.num_head_dim_stages_k
                    ):
                        sk.try_acquire()
                        sk.acquire()
                        sk.k_load_stage(
                            stage_id=head_dim_stage_idx,
                            kv_head_coord=kv_head_coord,
                            batch_coord=batch_coord,
                            cuseqlen_k=cuseqlen_k,
                            kv_tile_start=kv_tile_start,
                            seqlen_k=seqlen_k,
                            kv_request_begin=kv_request_begin,
                            kv_page_idx_ub=kv_page_idx_ub,
                        )
                        sk.commit()
                else:
                    # Throttle TMA before reserving a KV stage.
                    sk.try_acquire()
                    # Load Ki, with K0 handled by the first iteration.
                    sk.acquire()
                    sk.k_load(
                        kv_head_coord=kv_head_coord,
                        batch_coord=batch_coord,
                        cuseqlen_k=cuseqlen_k,
                        kv_tile_start=kv_tile_start,
                        seqlen_k=seqlen_k,
                        kv_request_begin=kv_request_begin,
                        kv_page_idx_ub=kv_page_idx_ub,
                    )
                    sk.commit()
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
                sv.try_acquire()
                # Load Vi, with V0 handled by the first iteration.
                sv.acquire()
                sv.v_load(
                    kv_head_coord=kv_head_coord,
                    batch_coord=batch_coord,
                    cuseqlen_k=cuseqlen_k,
                    seqlen_k=seqlen_k,
                    kv_tile_start=kv_tile_start,
                    kv_request_begin=kv_request_begin,
                    kv_page_idx_ub=kv_page_idx_ub,
                )
                sv.commit()
                if smu is not None:
                    # VC-Attention-QK16: the mean operand of tile i-1 rides beside V_i.
                    smu.try_acquire()
                    smu.acquire()
                    smu.mu_load(
                        kv_head_coord=kv_head_coord,
                        batch_coord=batch_coord,
                        kv_tile_start=kv_tile_start,
                    )
                    smu.commit()
            if smu is not None:
                # The last tile's mean operand feeds the tail mean step.
                (
                    _seq_coord,
                    _head_coord,
                    kv_head_coord,
                    _head_coord_kv,
                    batch_coord,
                    _seq_coord_q,
                    _cuseqlen_q,
                    _cuseqlen_k,
                    _seqlen_q,
                    _seqlen_k,
                    kv_tile_start,
                    _kv_request_begin,
                    _kv_page_idx_ub,
                ) = gqkv.compute_coords()
                smu.try_acquire()
                smu.acquire()
                smu.mu_load_last(
                    kv_head_coord=kv_head_coord,
                    batch_coord=batch_coord,
                    kv_tile_start=kv_tile_start,
                )
                smu.commit()

    @schedule
    def load_schedule_mu(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        skv: SmemKVResource,
        smu: SmemMuResource,
        wq: WorkQueue | None = None,
    ) -> None:
        """Shared K/V buffer plus the VC-Attention-QK16 tile-mean ring."""
        load_schedule_body(gqkv, sq, skv, skv, wq, smu)  # type: ignore[call-arg]

    @schedule
    def load_split_schedule_mu(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        sk: SmemKVResource,
        sv: SmemKVResource,
        smu: SmemMuResource,
        wq: WorkQueue | None = None,
    ) -> None:
        """Split K and V buffers plus the VC-Attention-QK16 tile-mean ring."""
        load_schedule_body(gqkv, sq, sk, sv, wq, smu)  # type: ignore[call-arg]

    @schedule
    def load_schedule(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        skv: SmemKVResource,
        wq: WorkQueue | None = None,
    ) -> None:
        """Contiguous-KV captured schedule over a shared K/V buffer."""
        # Mypy retains the earlier branch's five-argument closure signature.
        load_schedule_body(gqkv, sq, skv, skv, wq)  # type: ignore[call-arg]

    @schedule
    def load_split_schedule(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        sk: SmemKVResource,
        sv: SmemKVResource,
        wq: WorkQueue | None = None,
    ) -> None:
        """Contiguous-KV captured schedule over split K and V buffers."""
        load_schedule_body(gqkv, sq, sk, sv, wq)  # type: ignore[call-arg]

    if smem_mu is not None:
        captured_schedule = _schedule_with_work_queue(
            load_split_schedule_mu if split_kv else load_schedule_mu,
            gmem_qkv,
            smem_q,
            *kv_resources,
            smem_mu,
            work_queue=work_queue,
        )
    else:
        captured_schedule = _schedule_with_work_queue(
            load_split_schedule if split_kv else load_schedule,
            gmem_qkv,
            smem_q,
            *kv_resources,
            work_queue=work_queue,
        )
    return task_class(
        src_resources=src,
        dst_resources=dst,
        warp_idx=smem_k_or_kv.cfg.load_warp_id,
        num_warps=1,
        schedule=captured_schedule,
        num_registers=smem_k_or_kv.cfg.num_regs_other,
        name="LoadTask",
        **task_kwargs,
    )


def create_mma_task(
    gmem_qkv: GmemQKVResource,
    smem_q: SmemQResource,
    smem_k_or_kv: SmemKVResource,
    smem_v: SmemKVResource | None,
    tmem_sp0: TmemSPResource,
    tmem_sp1: TmemSPResource | None,
    tmem_p0: TmemPResource | None,
    tmem_o: TmemOResource,
    tmem_vec_done_0: TmemStatsDoneResource,
    tmem_vec_done_1: TmemStatsDoneResource | None,
    work_queue: WorkQueue | None,
    task_class: type[Task] = Task,
    tmem_p_prefix_ready_0: TmemPPrefixReadyResource | None = None,
    tmem_p_prefix_ready_1: TmemPPrefixReadyResource | None = None,
    smem_p0: SmemPResource | None = None,
    smem_p1: SmemPResource | None = None,
    smem_mu: SmemMuResource | None = None,
    **task_kwargs: Any,
) -> Task:
    """Create the one-warp MMA compute task.

    ``smem_mu`` (VC-Attention-QK16) supplies the bf16 tile-mean operand consumed by
    one extra ``kind::f16`` step after each tile's PV steps.
    """
    loop_start, loop_end, loop_step = _captured_loop_bounds(task_class, task_kwargs)
    skip_work_tile_if = _packed_context_skip_predicate(work_queue)
    # Only paged MMA schedules read request coordinates from global metadata.
    qkv_resources = [gmem_qkv] if smem_q.cfg.use_paged_kv else []
    split_kv = smem_k_or_kv.cfg.split_kv_pipelines
    if split_kv and smem_v is None:
        raise ValueError("split K/V staging requires a separate V buffer")
    kv_resources = (smem_k_or_kv, smem_v) if split_kv else (smem_k_or_kv,)
    pv_half_overlap = smem_q.cfg.pv_half_overlap
    if pv_half_overlap and (
        tmem_p_prefix_ready_0 is None or tmem_p_prefix_ready_1 is None
    ):
        raise ValueError("pv_half_overlap requires both P-prefix barriers")
    p_prefix_resources = (
        (tmem_p_prefix_ready_0, tmem_p_prefix_ready_1) if pv_half_overlap else ()
    )
    p_in_smem = smem_q.cfg.p_in_smem
    if p_in_smem and (smem_p0 is None or smem_p1 is None):
        raise ValueError("p_in_smem requires both SMEM P resources")
    smem_p_resources = (smem_p0, smem_p1) if p_in_smem else ()
    mu_resources = (smem_mu,) if smem_mu is not None else ()
    src = _src_resources(
        *qkv_resources,
        smem_q,
        *kv_resources,
        *p_prefix_resources,
        *smem_p_resources,
        *mu_resources,
        work_queue=work_queue,
    )
    num_head_dim_stages_k = smem_k_or_kv.cfg.num_head_dim_stages_k
    num_head_dim_stages_v = smem_k_or_kv.cfg.num_head_dim_stages_v

    if (
        smem_q.cfg.single_qkv_instance
        and smem_q.cfg.has_tmem_p_pipeline
        and tmem_p0 is not None
    ):
        split_src = _src_resources(
            *qkv_resources, smem_q, *kv_resources, tmem_p0, work_queue=work_queue
        )
        num_head_dim_stages_k = smem_k_or_kv.cfg.num_head_dim_stages_k
        num_head_dim_stages_v = smem_k_or_kv.cfg.num_head_dim_stages_v
        loop_carried_head_dim_stages = 2
        if (
            num_head_dim_stages_k != loop_carried_head_dim_stages
            or num_head_dim_stages_v != loop_carried_head_dim_stages
        ):
            raise ValueError("loop-carried split S/P scheduling expects two K/V stages")

        def mma_schedule_body(
            gqkv: GmemQKVResource,
            sq: SmemQResource,
            sk: SmemKVResource,
            sv: SmemKVResource,
            sp0: TmemSPResource,
            tp0: TmemPResource,
            to: TmemOResource,
            vd0: TmemStatsDoneResource,
            wq: WorkQueue | None = None,
        ) -> None:
            """MMA-warp schedule with separate S and P TMEM pipelines.

            Prologue issues QK for the first tile; the steady-state loop then alternates
            QK(i+1) into S with PV(i) from P, consuming K and V from independent rings.
            """
            sq.init_descriptor_state()
            sk.init_descriptor_state()
            sv.init_descriptor_state()
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
                    sk.wait()
                    desc_k_base = sk.k_desc()
                    sp0.qk_mma(
                        desc_q_base=desc_q0_base,
                        desc_k_base=desc_k_base,
                        section=FmhaStage.Head,
                        head_dim_stage_idx=head_dim_stage_idx,
                    )
                    sk.release()
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
                    sk.wait()
                    desc_k_base = sk.k_desc()
                    sp0.qk_mma(
                        desc_q_base=desc_q0_base,
                        desc_k_base=desc_k_base,
                        section=FmhaStage.Loop,
                        head_dim_stage_idx=0,
                    )
                    sk.release()

                    sk.wait()
                    desc_k_base = sk.k_desc()
                    sp0.qk_mma(
                        desc_q_base=desc_q0_base,
                        desc_k_base=desc_k_base,
                        section=FmhaStage.Loop,
                        head_dim_stage_idx=1,
                    )
                    sk.release()
                    sp0.commit()
                    if not smem_q.cfg.stats_via_smem:
                        vd0.commit()
                    sp0.acquire()

                    to.acquire()
                    tp0.wait()
                    tmem_p_base = tp0.p_base()
                    to.set_p_base(tmem_p_base=tmem_p_base)

                    # TODO: sv.wait() only depends on V's own TMA-load barrier,
                    # not on tp0 (softmax P). Now that K/V have independent
                    # pipelines, it could be issued earlier to overlap with the
                    # K/QK/softmax work above instead of waiting until here.
                    sv.wait()
                    if cutlass.const_expr(smem_q.cfg.needs_paged_v_tail_clear):
                        desc_v_base = sv.v_desc_paged(
                            section=FmhaStage.Loop,
                            seqlen_k=v_seqlen_k,
                            kv_tile_start=v_kv_tile_start,
                        )
                    else:
                        desc_v_base = sv.v_desc()
                    to.pv_mma(
                        desc_v_base=desc_v_base,
                        section=FmhaStage.Loop,
                        head_dim_stage_idx=0,
                    )
                    sv.release()

                    sv.wait()
                    if cutlass.const_expr(smem_q.cfg.needs_paged_v_tail_clear):
                        desc_v_base = sv.v_desc_paged(
                            section=FmhaStage.Loop,
                            seqlen_k=v_seqlen_k,
                            kv_tile_start=v_kv_tile_start,
                        )
                    else:
                        desc_v_base = sv.v_desc()
                    to.pv_mma(
                        desc_v_base=desc_v_base,
                        section=FmhaStage.Loop,
                        head_dim_stage_idx=1,
                    )
                    sv.release()
                    to.commit()
                    tp0.release()

                sq.release()
                to.acquire()
                tp0.wait()
                tmem_p_base = tp0.p_base()
                to.set_p_base(tmem_p_base=tmem_p_base)
                for head_dim_stage_idx in range(num_head_dim_stages_v):
                    sv.wait()
                    if cutlass.const_expr(smem_q.cfg.needs_paged_v_tail_clear):
                        desc_v_base = sv.v_desc_paged(
                            section=FmhaStage.Tail,
                            seqlen_k=v_seqlen_k,
                            kv_tile_start=v_kv_tile_start,
                        )
                    else:
                        desc_v_base = sv.v_desc()
                    to.pv_mma(
                        desc_v_base=desc_v_base,
                        section=FmhaStage.Tail,
                        head_dim_stage_idx=head_dim_stage_idx,
                        is_tail=True,
                    )
                    sv.release()
                to.commit()
                tp0.release()
                if not smem_q.cfg.stats_via_smem:
                    vd0.acquire()
                sp0.commit()
                if not smem_q.cfg.stats_via_smem:
                    vd0.commit()
                tp0.wait()
                tp0.release()

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
            """Shared-buffer captured schedule."""
            mma_schedule_body(gqkv, sq, skv, skv, sp0, tp0, to, vd0, wq)

        @schedule
        def mma_split_schedule(
            gqkv: GmemQKVResource,
            sq: SmemQResource,
            sk: SmemKVResource,
            sv: SmemKVResource,
            sp0: TmemSPResource,
            tp0: TmemPResource,
            to: TmemOResource,
            vd0: TmemStatsDoneResource,
            wq: WorkQueue | None = None,
        ) -> None:
            """Split K/V captured schedule."""
            mma_schedule_body(gqkv, sq, sk, sv, sp0, tp0, to, vd0, wq)

        captured_schedule = _schedule_with_work_queue(
            mma_split_schedule if split_kv else mma_schedule,
            gmem_qkv,
            smem_q,
            *kv_resources,
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
        num_head_dim_stages_k = smem_k_or_kv.cfg.num_head_dim_stages_k
        num_head_dim_stages_v = smem_k_or_kv.cfg.num_head_dim_stages_v

        def mma_schedule_body(
            gqkv: GmemQKVResource,
            sq: SmemQResource,
            sk: SmemKVResource,
            sv: SmemKVResource,
            sp0: TmemSPResource,
            to: TmemOResource,
            vd0: TmemStatsDoneResource,
            wq: WorkQueue | None = None,
        ) -> None:
            desc_q0_base, _desc_q1_base = sq.create_function_variables()
            # Every instance declares both descriptor slots; a split buffer
            # only ever fills the half matching its role.
            desc_k_base, _desc_v_base = sk.create_function_variables()
            _desc_k_base, desc_v_base = sv.create_function_variables()
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
                        sk.wait()
                        desc_k_base = sk.k_desc()
                        sp0.qk_mma(
                            desc_q_base=desc_q0_base,
                            desc_k_base=desc_k_base,
                            section=FmhaStage.Loop,
                            head_dim_stage_idx=head_dim_stage_idx,
                        )
                        sk.release()
                    sp0.commit()
                    with d.first_iter():
                        vd0.commit()
                    to.acquire()
                    sp0.acquire()
                    sp0.p_read()
                    for head_dim_stage_idx in range(num_head_dim_stages_v):
                        sv.wait()
                        if cutlass.const_expr(smem_q.cfg.needs_paged_v_tail_clear):
                            desc_v_base = sv.v_desc_paged(
                                section=FmhaStage.Loop,
                                seqlen_k=v_seqlen_k,
                                kv_tile_start=v_kv_tile_start,
                            )
                        else:
                            desc_v_base = sv.v_desc()
                        to.pv_mma(
                            desc_v_base=desc_v_base,
                            section=FmhaStage.Loop,
                            head_dim_stage_idx=head_dim_stage_idx,
                        )
                        sv.release()
                    to.commit()

                sq.release()
                sp0.commit()

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
            """Shared-buffer captured schedule."""
            mma_schedule_body(gqkv, sq, skv, skv, sp0, to, vd0, wq)

        @schedule
        def mma_split_schedule(
            gqkv: GmemQKVResource,
            sq: SmemQResource,
            sk: SmemKVResource,
            sv: SmemKVResource,
            sp0: TmemSPResource,
            to: TmemOResource,
            vd0: TmemStatsDoneResource,
            wq: WorkQueue | None = None,
        ) -> None:
            """Split K/V captured schedule."""
            mma_schedule_body(gqkv, sq, sk, sv, sp0, to, vd0, wq)

        captured_schedule = _schedule_with_work_queue(
            mma_split_schedule if split_kv else mma_schedule,
            gmem_qkv,
            smem_q,
            *kv_resources,
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

    # Padded QK is at most 256 for the admitted paired geometries. Keep each
    # 128-wide slice in its own TS binding so later slices cannot replace it.
    # The stage loops below handle both the full and partial final slice.
    if num_head_dim_stages_k > 2:
        raise ValueError("paired QK staging supports at most two 128-wide slices")

    def qk_mma_slice(sp, desc_q_base, desc_k, section, head_dim_stage_idx):
        """Route a slice through its own descriptor binding to the shared QK body."""
        if head_dim_stage_idx == 0:
            sp.qk_mma(
                desc_q_base=desc_q_base,
                desc_k_base=desc_k,
                section=section,
                head_dim_stage_idx=head_dim_stage_idx,
            )
        else:
            sp.qk_mma_stage(
                desc_q_base=desc_q_base,
                desc_k_stage=desc_k,
                section=section,
                head_dim_stage_idx=head_dim_stage_idx,
            )

    def load_k_stage(sk, head_dim_stage_idx):
        """Retain the descriptor of each waited K slice for both query tiles."""
        sk.wait()
        if head_dim_stage_idx == 0:
            return sk.k_desc()
        return sk.k_stage_desc()

    def qk_mma_stages(sk, sp, desc_q_base, section, *, descriptors=None):
        """Issue all K slices, loading descriptors once and reusing them for Q1."""
        stages = []
        for head_dim_stage_idx in range(num_head_dim_stages_k):
            desc_k = (
                load_k_stage(sk, head_dim_stage_idx)
                if descriptors is None
                else descriptors[head_dim_stage_idx]
            )
            qk_mma_slice(sp, desc_q_base, desc_k, section, head_dim_stage_idx)
            stages.append(desc_k)
        return stages

    def mma_schedule_body(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        sk: SmemKVResource,
        sv: SmemKVResource,
        sp0: TmemSPResource,
        sp1: TmemSPResource,
        to: TmemOResource,
        vd0: TmemStatsDoneResource,
        vd1: TmemStatsDoneResource,
        wq: WorkQueue | None = None,
        pr0: TmemPPrefixReadyResource | None = None,
        pr1: TmemPPrefixReadyResource | None = None,
        pb0: SmemPResource | None = None,
        pb1: SmemPResource | None = None,
        smu: SmemMuResource | None = None,
    ) -> None:
        """Interleave paired QK/PV while retaining each K slice for both Qs."""
        # VC-Attention-QK16: the current tile's mean-operand descriptor, waited and
        # released in lockstep with V.
        mu_state: dict[str, Any] = {"desc": None}

        def pv(
            sp: TmemSPResource,
            pr: TmemPPrefixReadyResource | None,
            pb: SmemPResource | None,
            *,
            group: int,
            **kw: Any,
        ) -> None:
            """PV for query group ``group``. P-ready comes from the SMEM P tile with
            P in SMEM, from the P-prefix barrier with the half overlap, else from the SP acquire."""
            if p_in_smem:
                pb.wait()
                to.pv_mma(inst_idx=group, **kw)
                if smu is not None:
                    # VC-Attention-QK16: a group's row sums travel with the P of the
                    # tile after it, so its mean step follows that tile's PV steps.
                    to.vc_mean_mma(
                        inst_idx=group,
                        desc_mu_base=mu_state["desc"],
                        is_tail=kw.get("is_tail", False),
                    )
                pb.release()
                return
            if pv_half_overlap:
                pr.wait()
                to.pv_mma(k_half=0, **kw)
                pr.release()
            sp.acquire()
            sp.p_read()
            to.pv_mma(k_half=1 if pv_half_overlap else None, **kw)

        sq.init_descriptor_state()
        sk.init_descriptor_state()
        sv.init_descriptor_state()
        if smu is not None:
            smu.init_descriptor_state()
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
            for head_dim_stage_idx in range(num_head_dim_stages_k):
                desc_k = load_k_stage(sk, head_dim_stage_idx)
                if head_dim_stage_idx == 0:
                    if not smem_q.cfg.stats_via_smem:
                        vd0.acquire()
                    sp0.acquire()
                qk_mma_slice(
                    sp0, desc_q0_base, desc_k, FmhaStage.Head, head_dim_stage_idx
                )
                if head_dim_stage_idx == num_head_dim_stages_k - 1:
                    sp0.commit()
                    if not smem_q.cfg.stats_via_smem:
                        vd0.commit()
                if head_dim_stage_idx == 0:
                    sq.wait()
                    desc_q1_base = sq.q1_desc(inst_idx=1)
                    if not smem_q.cfg.stats_via_smem:
                        vd1.acquire()
                    sp1.acquire()
                qk_mma_slice(
                    sp1, desc_q1_base, desc_k, FmhaStage.Head, head_dim_stage_idx
                )
                if head_dim_stage_idx > 0:
                    sk.release()
            sp1.commit()
            if not smem_q.cfg.stats_via_smem:
                vd1.commit()
            # Q0/Q1 stay live because UMMA reads Q throughout the K-loop.
            # Release K0 (done with QK→S0 and QK→S1), then consume V0.
            sk.release()
            # TODO: sv.wait() (V0) only depends on V's own TMA-load barrier,
            # not on sk/sp0/sp1, so it could move earlier to overlap with the
            # QK0/QK1 work above.
            sv.wait()
            if cutlass.const_expr(smem_q.cfg.needs_paged_v_tail_clear):
                desc_v_base = sv.v_desc_paged(
                    section=FmhaStage.Head,
                    seqlen_k=v_seqlen_k,
                    kv_tile_start=v_kv_tile_start,
                )
            else:
                desc_v_base = sv.v_desc()
            # Acquire O first (off critical path), then acquire SP0 and run PV→O0.
            if not p_in_smem:
                to.acquire()
                pv(
                    sp0,
                    pr0,
                    pb0,
                    group=0,
                    desc_v_base=desc_v_base,
                    section=FmhaStage.Head,
                )
                to.commit()

            def wait_next_v() -> Any:
                """Release Ki+1, then wait Vi+1 and build its descriptor."""
                for _ in range(num_head_dim_stages_k):
                    sk.release()
                # TODO: sv.wait() (Vi+1) could move earlier in this iteration
                # to overlap with QK0/PV1/QK1.
                sv.wait()
                if cutlass.const_expr(smem_q.cfg.needs_paged_v_tail_clear):
                    return sv.v_desc_paged(
                        section=FmhaStage.Loop,
                        tile_offset=1,
                        seqlen_k=v_seqlen_k,
                        kv_tile_start=v_kv_tile_start,
                    )
                return sv.v_desc()

            def release_mu() -> None:
                """Release the mean operand of the tile whose PV steps were issued."""
                if smu is not None:
                    smu.release()

            def wait_next_mu() -> None:
                """Wait for the next tile's mean operand and cache its descriptor."""
                if smu is not None:
                    smu.wait()
                    mu_state["desc"] = smu.mu_desc()

            if p_in_smem:
                # With P in SMEM, softmax releases the S stage right after loading
                # S, so QK(i+1) overlaps exp2(i). The MMA issues QK0(i+1), PV0(i),
                # QK1(i+1), PV1(i). Each QK acquires its SP slot here and each PV
                # waits on its group's P tile instead. The tail drains no SP slot.
                with domain_loop(loop_start, loop_end, loop_step):
                    sp0.acquire()
                    k_descriptors = qk_mma_stages(sk, sp0, desc_q0_base, FmhaStage.Loop)
                    sp0.commit()
                    # Wait for the mean operand here, off the QK issue path.
                    wait_next_mu()
                    to.acquire()
                    pv(
                        sp0,
                        pr0,
                        pb0,
                        group=0,
                        desc_v_base=desc_v_base,
                        section=FmhaStage.Loop,
                    )
                    to.commit()
                    sp1.acquire()
                    qk_mma_stages(
                        sk, sp1, desc_q1_base, FmhaStage.Loop, descriptors=k_descriptors
                    )
                    sp1.commit()
                    to.acquire()
                    pv(
                        sp1,
                        pr1,
                        pb1,
                        group=1,
                        desc_v_base=desc_v_base,
                        section=FmhaStage.Loop,
                    )
                    to.commit()
                    sv.release()
                    release_mu()
                    desc_v_base = wait_next_v()
                sq.release()
                sq.release()
                if smu is not None:
                    # Each group's tail O window issues mean(N-2) and mean(N-1),
                    # so both operands are held.
                    wait_next_mu()
                    smu.wait()
                    desc_mu_last = smu.mu_desc_last()

                def tail_mean(group: int, pb: Any) -> None:
                    """Issue the final mean step of ``group`` inside its O window."""
                    if smu is not None:
                        pb.wait()
                        to.vc_mean_mma_last(inst_idx=group, desc_mu_last=desc_mu_last)
                        pb.release()

                to.acquire()
                pv(
                    sp0,
                    pr0,
                    pb0,
                    group=0,
                    desc_v_base=desc_v_base,
                    section=FmhaStage.Tail,
                    is_tail=True,
                )
                tail_mean(0, pb0)
                to.commit()
                to.acquire()
                pv(
                    sp1,
                    pr1,
                    pb1,
                    group=1,
                    desc_v_base=desc_v_base,
                    section=FmhaStage.Tail,
                    is_tail=True,
                )
                tail_mean(1, pb1)
                to.commit()
                sv.release()
                release_mu()
                release_mu()
            else:
                # QK0 -> PV1 -> QK1 -> PV0: retain each K slice and the previous V
                # until both peer query tiles have consumed the corresponding data.
                with domain_loop(loop_start, loop_end, loop_step):
                    k_descriptors = qk_mma_stages(sk, sp0, desc_q0_base, FmhaStage.Loop)
                    sp0.commit()
                    to.acquire()
                    pv(
                        sp1,
                        pr1,
                        pb1,
                        group=1,
                        desc_v_base=desc_v_base,
                        section=FmhaStage.Loop,
                    )
                    to.commit()
                    # Release V_prev after PV1 UMMA consumed SMEM data.
                    sv.release()
                    qk_mma_stages(
                        sk, sp1, desc_q1_base, FmhaStage.Loop, descriptors=k_descriptors
                    )
                    sp1.commit()
                    desc_v_base = wait_next_v()
                    # PV0: P0 * Vi+1 → O0.
                    to.acquire()
                    pv(
                        sp0,
                        pr0,
                        pb0,
                        group=0,
                        desc_v_base=desc_v_base,
                        section=FmhaStage.Loop,
                        inst_idx=1,
                    )
                    to.commit()

                sq.release()
                sq.release()
                sp0.commit()
                to.acquire()
                pv(
                    sp1,
                    pr1,
                    pb1,
                    group=1,
                    desc_v_base=desc_v_base,
                    section=FmhaStage.Tail,
                    is_tail=True,
                )
                to.commit()
                sv.release()
                sp1.commit()

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
        """Shared-buffer captured schedule."""
        mma_schedule_body(gqkv, sq, skv, skv, sp0, sp1, to, vd0, vd1, wq)

    @schedule
    def mma_split_schedule(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        sk: SmemKVResource,
        sv: SmemKVResource,
        sp0: TmemSPResource,
        sp1: TmemSPResource,
        to: TmemOResource,
        vd0: TmemStatsDoneResource,
        vd1: TmemStatsDoneResource,
        wq: WorkQueue | None = None,
    ) -> None:
        """Split K/V captured schedule."""
        mma_schedule_body(gqkv, sq, sk, sv, sp0, sp1, to, vd0, vd1, wq)

    # Overlap variants carry the two P-prefix barriers.
    @schedule
    def mma_schedule_pr(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        skv: SmemKVResource,
        sp0: TmemSPResource,
        sp1: TmemSPResource,
        to: TmemOResource,
        vd0: TmemStatsDoneResource,
        vd1: TmemStatsDoneResource,
        pr0: TmemPPrefixReadyResource,
        pr1: TmemPPrefixReadyResource,
        wq: WorkQueue | None = None,
    ) -> None:
        """Shared-buffer captured schedule with P-prefix overlap."""
        mma_schedule_body(gqkv, sq, skv, skv, sp0, sp1, to, vd0, vd1, wq, pr0, pr1)

    @schedule
    def mma_split_schedule_pr(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        sk: SmemKVResource,
        sv: SmemKVResource,
        sp0: TmemSPResource,
        sp1: TmemSPResource,
        to: TmemOResource,
        vd0: TmemStatsDoneResource,
        vd1: TmemStatsDoneResource,
        pr0: TmemPPrefixReadyResource,
        pr1: TmemPPrefixReadyResource,
        wq: WorkQueue | None = None,
    ) -> None:
        """Split K/V captured schedule with P-prefix overlap."""
        mma_schedule_body(gqkv, sq, sk, sv, sp0, sp1, to, vd0, vd1, wq, pr0, pr1)

    # P-in-SMEM variants carry the two SMEM P tiles.
    @schedule
    def mma_split_schedule_pb(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        sk: SmemKVResource,
        sv: SmemKVResource,
        sp0: TmemSPResource,
        sp1: TmemSPResource,
        to: TmemOResource,
        vd0: TmemStatsDoneResource,
        vd1: TmemStatsDoneResource,
        pb0: SmemPResource,
        pb1: SmemPResource,
        wq: WorkQueue | None = None,
    ) -> None:
        """Split K/V captured schedule with P staged in SMEM."""
        mma_schedule_body(
            gqkv, sq, sk, sv, sp0, sp1, to, vd0, vd1, wq, None, None, pb0, pb1
        )

    @schedule
    def mma_schedule_pb(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        skv: SmemKVResource,
        sp0: TmemSPResource,
        sp1: TmemSPResource,
        to: TmemOResource,
        vd0: TmemStatsDoneResource,
        vd1: TmemStatsDoneResource,
        pb0: SmemPResource,
        pb1: SmemPResource,
        wq: WorkQueue | None = None,
    ) -> None:
        """Shared-buffer captured schedule with P staged in SMEM."""
        mma_schedule_body(
            gqkv, sq, skv, skv, sp0, sp1, to, vd0, vd1, wq, None, None, pb0, pb1
        )

    @schedule
    def mma_split_schedule_pb_mu(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        sk: SmemKVResource,
        sv: SmemKVResource,
        sp0: TmemSPResource,
        sp1: TmemSPResource,
        to: TmemOResource,
        vd0: TmemStatsDoneResource,
        vd1: TmemStatsDoneResource,
        pb0: SmemPResource,
        pb1: SmemPResource,
        smu: SmemMuResource,
        wq: WorkQueue | None = None,
    ) -> None:
        """Split K/V captured schedule with P in SMEM and VC tile means."""
        mma_schedule_body(
            gqkv, sq, sk, sv, sp0, sp1, to, vd0, vd1, wq, None, None, pb0, pb1, smu
        )

    @schedule
    def mma_schedule_pb_mu(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        skv: SmemKVResource,
        sp0: TmemSPResource,
        sp1: TmemSPResource,
        to: TmemOResource,
        vd0: TmemStatsDoneResource,
        vd1: TmemStatsDoneResource,
        pb0: SmemPResource,
        pb1: SmemPResource,
        smu: SmemMuResource,
        wq: WorkQueue | None = None,
    ) -> None:
        """Shared-buffer captured schedule with P in SMEM and VC tile means."""
        mma_schedule_body(
            gqkv, sq, skv, skv, sp0, sp1, to, vd0, vd1, wq, None, None, pb0, pb1, smu
        )

    if smem_mu is not None:
        selected_mma_schedule = (
            mma_split_schedule_pb_mu if split_kv else mma_schedule_pb_mu
        )
    elif p_in_smem:
        selected_mma_schedule = mma_split_schedule_pb if split_kv else mma_schedule_pb
    elif pv_half_overlap:
        selected_mma_schedule = mma_split_schedule_pr if split_kv else mma_schedule_pr
    else:
        selected_mma_schedule = mma_split_schedule if split_kv else mma_schedule
    captured_schedule = _schedule_with_work_queue(
        selected_mma_schedule,
        gmem_qkv,
        smem_q,
        *kv_resources,
        tmem_sp0,
        tmem_sp1,
        tmem_o,
        tmem_vec_done_0,
        tmem_vec_done_1,
        *p_prefix_resources,
        *smem_p_resources,
        *mu_resources,
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
    tmem_p_prefix_ready: TmemPPrefixReadyResource | None = None,
    smem_p: SmemPResource | None = None,
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
                        if tmem_sp.cfg.corr_skip_threshold_log2 > 0:
                            row_max = sp.freeze_row_max(
                                old_row_max=old_row_max,
                                row_max=row_max,
                                scale_softmax_log2=scale_softmax_log2,
                            )
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

    # Paired QKV instances use separate SP resources to protect P readiness.
    # The sequence token paces peer progress; FP8 overlaps P computation.
    if s0s1_seq is not None and index == 1:
        src = _src_resources(tmem_sp, s0s1_seq, work_queue=work_queue)
    else:
        src = _src_resources(tmem_sp, work_queue=work_queue)
    dst = [tmem_vec]
    if s0s1_seq is not None and index == 0:
        dst.append(s0s1_seq)
    pv_half_overlap = tmem_sp.cfg.pv_half_overlap
    if pv_half_overlap:
        if tmem_p_prefix_ready is None:
            raise ValueError("pv_half_overlap requires the P-prefix barrier")
        dst.append(tmem_p_prefix_ready)
    p_in_smem = tmem_sp.cfg.p_in_smem
    if p_in_smem:
        if smem_p is None:
            raise ValueError("p_in_smem requires the group's SMEM P resource")
        dst.append(smem_p)
    vc_attention = tmem_sp.cfg.vc_attention
    # VC-Attention-QK16 V treatment: tile means restored in-kernel, or V repair tiles.
    vc_restores_means = tmem_sp.cfg.vc_restores_means
    vc_repairs_v = vc_attention and not vc_restores_means

    def softmax_schedule_body(
        sp: TmemSPResource,
        vec: TmemStatsResource,
        seq: S0S1SequenceResource,
        wq: WorkQueue | None = None,
        pr: TmemPPrefixReadyResource | None = None,
        pb: SmemPResource | None = None,
    ) -> None:
        """Captured schedule for one softmax warp group."""
        if tmem_sp.enable_early_tile_sum:
            # The contribution is produced and consumed inside each iteration;
            # do not carry even the scalar tile sum through the persistent loop.
            sp.init_softmax_state_early()
        else:
            p_chunk = sp.init_softmax_state()
        scale_softmax_log2 = sp.load_scale_softmax_log2()

        vc_state: dict[str, Any] = {
            "prev_sum": None,
            "row_sum": None,
            "sums": None,
            "k_next": None,
            "kept": None,
        }

        def exp2_p(
            sp: TmemSPResource,
            *,
            row_max: Any,
            scale_softmax_log2: Any,
            old_row_max: Any = None,
            is_tail: bool = False,
        ) -> Any:
            """Softmax and P store. With P in SMEM the store goes to the SMEM P tile,
            with the half overlap the leading half is published behind ``pr``, else P goes
            to the TMEM S/P stage. VC-Attention-QK16 then carries the group row sums
            (rescaled to this tile's max) and stores the completed group's operand
            for the deferred mean step."""
            if p_in_smem:
                # The S stage is released already, so exp2 and the P store are
                # auxiliary work on the loaded S and the SMEM P tile.
                p_chunk = sp.exp2_p_smem(
                    row_max=row_max, scale_softmax_log2=scale_softmax_log2
                )
                pb.acquire()
                if vc_repairs_v:
                    out = sp.vc_store_p_repair(
                        old_row_max=old_row_max,
                        row_max=row_max,
                        row_sum=vc_state["row_sum"],
                        vc_kept_row_sum=vc_state["kept"],
                        p_chunk=p_chunk,
                        is_tail=is_tail,
                    )
                    vc_state["kept"], vc_state["row_sum"] = out[0], out[1]
                elif vc_restores_means:
                    out = sp.vc_store_p(
                        old_row_max=old_row_max,
                        row_max=row_max,
                        row_sum=vc_state["row_sum"],
                        vc_prev_tile_sum=vc_state["prev_sum"],
                        p_chunk=p_chunk,
                        is_tail=is_tail,
                        **{
                            f"vc_pend{i}": value
                            for i, value in enumerate(vc_state["sums"])
                        },
                    )
                    vc_state["prev_sum"], vc_state["row_sum"] = out[0], out[1]
                    vc_state["sums"] = out[2:]
                else:
                    sp.store_p()
                pb.commit()
                return p_chunk
            if not pv_half_overlap:
                return sp.exp2_p(row_max=row_max, scale_softmax_log2=scale_softmax_log2)
            pr.acquire()
            p_lo = sp.exp2_p_lo(row_max=row_max, scale_softmax_log2=scale_softmax_log2)
            pr.commit()
            return sp.exp2_p_hi(
                row_max=row_max, scale_softmax_log2=scale_softmax_log2, p_lo=p_lo
            )

        vec.init_store_state()
        with _work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
            # Recompute per-tile SP/Vec TMEM state.
            old_row_max, row_max, row_sum, q_offset = sp.init_softmax_work_tile_state()
            vec.init_store_work_tile_state()
            vc_state["row_sum"] = row_sum
            if vc_attention:
                init = sp.vc_init_row_scale()
                vc_row_scale, vc_state["prev_sum"] = init[0], init[1]
                vc_state["sums"] = init[2:18]
                vc_state["kept"] = init[18]
                vc_state["k_next"] = init[19]
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
                elif tmem_sp.uses_packed_dense_k_mask:
                    old_row_max, row_max = sp.packed_dense_k_masked_row_max(
                        row_max=row_max,
                        seqlen_k=seqlen_k,
                        section=FmhaStage.Loop,
                    )
                elif vc_attention:
                    old_row_max, row_max, vc_state["k_next"] = sp.vc_compute_row_max(
                        row_max=row_max,
                        vc_row_scale=vc_row_scale,
                        vc_k_scale_next=vc_state["k_next"],
                    )
                else:
                    old_row_max, row_max = sp.compute_row_max(row_max=row_max)
                if tmem_sp.cfg.corr_skip_threshold_log2 > 0:
                    row_max = sp.freeze_row_max(
                        old_row_max=old_row_max,
                        row_max=row_max,
                        scale_softmax_log2=scale_softmax_log2,
                    )
                vec.store_vec(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                )
                vec.commit()
                if p_in_smem:
                    # S is in registers. Free the stage so QK(i+1) overlaps exp2.
                    sp.release()
                if s0s1_seq is None:
                    pass
                elif index == 0:
                    # Softmax0 is the S0-S1 producer: acquire/commit sequence.
                    seq.acquire()
                else:
                    # Softmax1 is the S0-S1 consumer: wait/release sequence.
                    seq.wait()
                # FP8 returns the pacing token before P work. Its SP release
                # still follows every P store's completion.
                early_token = tmem_sp.cfg.uses_d128_fp8_softmax_cadence or (
                    p_in_smem and tmem_sp.cfg.fp8_psmem_early_token
                )
                if early_token:
                    if index == 0:
                        seq.commit()
                    else:
                        seq.release()
                # Apply softmax and write P.
                p_chunk = exp2_p(
                    sp,
                    row_max=row_max,
                    scale_softmax_log2=scale_softmax_log2,
                    old_row_max=old_row_max,
                )
                if s0s1_seq is None or early_token:
                    pass
                elif index == 0:
                    seq.commit()
                else:
                    seq.release()
                if not p_in_smem:
                    sp.release()
                # Reduction.
                if vc_attention:
                    row_sum = vc_state["row_sum"]
                else:
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
                if vc_attention:
                    row_sum = vc_state["row_sum"]
                else:
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
                if p_in_smem:
                    sp.release()
                if s0s1_seq is not None:
                    seq.acquire()
                if p_in_smem:
                    p_chunk = sp.masked_exp2_p_smem(
                        row_max=row_max,
                        scale_softmax_log2=scale_softmax_log2,
                    )
                    pb.acquire()
                    sp.store_p()
                    pb.commit()
                else:
                    p_chunk = sp.masked_exp2_p(
                        row_max=row_max,
                        scale_softmax_log2=scale_softmax_log2,
                    )
                if s0s1_seq is not None:
                    seq.commit()
                if not p_in_smem:
                    sp.release()
                if vc_attention:
                    row_sum = vc_state["row_sum"]
                else:
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
                    if p_in_smem:
                        # MMA skips this PV but still waits on the P tile.
                        pb.acquire()
                        pb.commit()
                    if s0s1_seq is not None:
                        seq.commit()
                    sp.release()
                if not p_in_smem:
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
                # Causal tail: mask the last tile, then publish the final stats.
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
                if p_in_smem:
                    sp.release()
                if s0s1_seq is not None:
                    seq.wait()
                if p_in_smem:
                    p_chunk = sp.masked_exp2_p_smem(
                        row_max=row_max,
                        scale_softmax_log2=scale_softmax_log2,
                    )
                    pb.acquire()
                    sp.store_p()
                    pb.commit()
                else:
                    p_chunk = sp.masked_exp2_p(
                        row_max=row_max,
                        scale_softmax_log2=scale_softmax_log2,
                    )
                if s0s1_seq is not None:
                    seq.release()
                if not p_in_smem:
                    sp.release()
                if vc_attention:
                    row_sum = vc_state["row_sum"]
                else:
                    row_sum = sp.softmax_aux_reduce(
                        old_row_max=old_row_max,
                        row_max=row_max,
                        row_sum=row_sum,
                        p_chunk=p_chunk,
                        scale_softmax_log2=scale_softmax_log2,
                    )
                if not p_in_smem:
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
            elif tmem_sp.uses_fixed_dense_k_tail_mask:
                # Dense tail: one more loop step with the zero-filled lanes masked out.
                sp.wait()
                if vc_attention:
                    old_row_max, row_max = sp.vc_fixed_dense_k_tail_masked_row_max(
                        row_max=row_max,
                        vc_row_scale=vc_row_scale,
                        vc_k_scale_next=vc_state["k_next"],
                    )
                else:
                    old_row_max, row_max = sp.fixed_dense_k_tail_masked_row_max(
                        row_max=row_max,
                        section=FmhaStage.Tail,
                    )
                vec.store_vec(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                )
                vec.commit()
                if p_in_smem:
                    sp.release()
                if s0s1_seq is None:
                    pass
                elif index == 0:
                    seq.acquire()
                else:
                    seq.wait()
                p_chunk = exp2_p(
                    sp,
                    row_max=row_max,
                    scale_softmax_log2=scale_softmax_log2,
                    old_row_max=old_row_max,
                    is_tail=True,
                )
                if s0s1_seq is None:
                    pass
                elif index == 0:
                    seq.commit()
                else:
                    seq.release()
                if not p_in_smem:
                    sp.release()
                if vc_attention:
                    row_sum = vc_state["row_sum"]
                else:
                    row_sum = sp.softmax_aux_reduce(
                        old_row_max=old_row_max,
                        row_max=row_max,
                        row_sum=row_sum,
                        p_chunk=p_chunk,
                        scale_softmax_log2=scale_softmax_log2,
                    )
                vec.acquire()
                if vc_restores_means:
                    # The last row sums ride a second P handoff to the tail step.
                    pb.acquire()
                    sp.vc_store_rowsum_final(
                        vc_prev_tile_sum=vc_state["prev_sum"],
                        **{
                            f"vc_pend{i}": value
                            for i, value in enumerate(vc_state["sums"])
                        },
                    )
                    pb.commit()
                elif vc_repairs_v:
                    # The repair tiles stay out of the denominator.
                    row_sum = vc_state["kept"]
                # Cleanup: drain the final SP slot and publish identity stats.
                if not p_in_smem:
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
            else:
                # Non-causal tail: no more tiles, just publish the final stats.
                if vc_restores_means:
                    # The last row sums ride a second P handoff to the tail step.
                    pb.acquire()
                    sp.vc_store_rowsum_final(
                        vc_prev_tile_sum=vc_state["prev_sum"],
                        **{
                            f"vc_pend{i}": value
                            for i, value in enumerate(vc_state["sums"])
                        },
                    )
                    pb.commit()
                elif vc_repairs_v:
                    # The repair tiles stay out of the denominator.
                    row_sum = vc_state["kept"]
                if not p_in_smem:
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

    @schedule
    def softmax_schedule(
        sp: TmemSPResource,
        vec: TmemStatsResource,
        seq: S0S1SequenceResource,
        wq: WorkQueue | None = None,
    ) -> None:
        softmax_schedule_body(sp, vec, seq, wq)

    @schedule
    def softmax_schedule_pr(
        sp: TmemSPResource,
        vec: TmemStatsResource,
        seq: S0S1SequenceResource,
        pr: TmemPPrefixReadyResource,
        wq: WorkQueue | None = None,
    ) -> None:
        softmax_schedule_body(sp, vec, seq, wq, pr)

    @schedule
    def softmax_schedule_pb(
        sp: TmemSPResource,
        vec: TmemStatsResource,
        seq: S0S1SequenceResource,
        pb: SmemPResource,
        wq: WorkQueue | None = None,
    ) -> None:
        softmax_schedule_body(sp, vec, seq, wq, None, pb)

    if p_in_smem:
        captured_schedule = _schedule_with_work_queue(
            softmax_schedule_pb,
            tmem_sp,
            tmem_vec,
            s0s1_seq,
            smem_p,
            work_queue=work_queue,
        )
    elif pv_half_overlap:
        captured_schedule = _schedule_with_work_queue(
            softmax_schedule_pr,
            tmem_sp,
            tmem_vec,
            s0s1_seq,
            tmem_p_prefix_ready,
            work_queue=work_queue,
        )
    else:
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
        release_stats_after_read = tmem_vec0.cfg.p_in_smem
        if release_stats_after_read and not tmem_vec0.cfg.stats_via_smem:
            raise ValueError("early stats release requires SMEM-staged stats")

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
                if release_stats_after_read:
                    v1.release()
                # The correction loop consumes vec/O pairs in alternating order so
                # each half can unblock the other half's next producer. With P in
                # SMEM each stats slot is released right after read_vec instead.
                with domain_loop(loop_start, loop_end, loop_step):
                    # Part 1: consume TmemStats0 + O0, release TmemStats1.
                    v0.wait()
                    vec_old_max, vec_new_max, _, vec_scale = v0.read_vec(
                        scale_softmax_log2=scale_softmax_log2_v0,
                    )
                    if release_stats_after_read:
                        v0.release()
                    to.wait()
                    to.correct(
                        vec_old_max=vec_old_max,
                        vec_new_max=vec_new_max,
                        vec_scale=vec_scale,
                        inst_idx=0,
                    )
                    if not release_stats_after_read:
                        v1.release()
                    to.release()
                    # Part 2: consume TmemStats1 + O1, release TmemStats0.
                    v1.wait()
                    vec_old_max, vec_new_max, _, vec_scale = v1.read_vec(
                        scale_softmax_log2=scale_softmax_log2_v1,
                    )
                    if release_stats_after_read:
                        v1.release()
                    to.wait()
                    to.correct(
                        vec_old_max=vec_old_max,
                        vec_new_max=vec_new_max,
                        vec_scale=vec_scale,
                        inst_idx=1,
                    )
                    if not release_stats_after_read:
                        v0.release()
                    to.release()
                # Tail: read the final stats, then write corrected O0/O1 to smem.
                if not release_stats_after_read:
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
                for head_dim_stage_idx in range(smem_o_0.cfg.num_o_head_dim_stages):
                    so0.acquire()
                    so0.store_o(
                        vec_row_sum=vec_row_sum,
                        vec_scale=vec_scale,
                        output_scale=output_scale0,
                        head_dim_stage_idx=head_dim_stage_idx,
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
                for head_dim_stage_idx in range(smem_o_0.cfg.num_o_head_dim_stages):
                    so1.acquire()
                    so1.store_o(
                        vec_row_sum=vec_row_sum,
                        vec_scale=vec_scale,
                        output_scale=output_scale1,
                        head_dim_stage_idx=head_dim_stage_idx,
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
                for head_dim_stage_idx in range(smem_o_0.cfg.num_o_head_dim_stages):
                    so0.wait()
                    head_coord, batch_coord, seq_coord_q = so0.compute_output_coords()
                    go0.tma_store(
                        head_coord=head_coord,
                        batch_coord=batch_coord,
                        seq_coord_q=seq_coord_q,
                        head_dim_stage_idx=head_dim_stage_idx,
                    )
                    so0.release()
                # Store the second corrected O tile through gmem_o_1.
                for head_dim_stage_idx in range(smem_o_0.cfg.num_o_head_dim_stages):
                    so1.wait()
                    head_coord, batch_coord, seq_coord_q = so1.compute_output_coords()
                    go1.tma_store(
                        head_coord=head_coord,
                        batch_coord=batch_coord,
                        seq_coord_q=seq_coord_q,
                        head_dim_stage_idx=head_dim_stage_idx,
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
