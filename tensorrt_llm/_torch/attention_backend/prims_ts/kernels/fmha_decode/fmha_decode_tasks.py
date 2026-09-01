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

"""Task definitions for the FMHA decode TS kernel.

The schedule follows the SwapsMmaAb decode pipeline shape:

- HEAD: initial Q / K setup and first BMM1 wave
- LOOP: staggered K/V loads with Qk0 -> Pv0 -> Qk1 -> Pv1
- TAIL: final V loads, final BMM2, final correction/output

Each schedule is written as explicit TS resource transitions. Producer
resources use acquire/work/commit; consumer resources use wait/work/release.
The comments below name the logical pipeline step so the ordering can be read
without expanding the decorators on each resource method.
"""

import functools
from dataclasses import dataclass, field
from typing import Any, Callable

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.memory import (
    ResourceContext,
    SmemAllocation,
)
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    StageInfo,
    TaskLocalVariable,
    WorkQueue,
    consumer_work,
    producer_work,
)
from cutlass.experimental.task_scheduling.schedule_builder import (
    Schedule,
    domain_loop,
    schedule,
    work_tile_loop,
)
from cutlass.experimental.task_scheduling.task import Task

from ..stage import FmhaStage
from .fmha_decode_config import FmhaDecodeConfig
from .fmha_decode_constants import (
    KV_INST0,
    KV_INST1,
    KV_KIND_K,
    KV_KIND_V,
    KV_TILE_256_SHARED_FIFO_STAGES,
)
from .fmha_decode_resources.helpers_common import (
    ResourceVars,
    _q_group_token_base,
    _q_seq_bounds,
    _warp_broadcast_i32,
)
from .fmha_decode_resources.helpers_kv_tile_idx import (
    _runtime_last_valid_page_idx,
    _runtime_total_kv_tiles,
    _sliding_window_start_idx,
)

QBoundBinding = tuple[MemoryResource, bool]
TaskKwarg = (
    int
    | bool
    | cutlass.Int32
    | cute.Pointer
    | FmhaDecodeConfig
    | tuple[QBoundBinding, ...]
    | None
)


def _schedule_with_optional_resources(
    fn: Callable[..., None],
) -> Callable[..., Schedule]:
    """Capture one schedule while omitting absent resources from its graph.

    CUTLASS ``@schedule`` intentionally accepts only concrete resources.  Some
    FMHA task variants have orthogonal optional resources, such as block-sparse
    metadata and a persistent work queue.  Filter absent slots before tracing,
    then restore their named positions for the schedule body.  This adapter is
    entirely host-side: each captured schedule contains only present resources
    and the ``None`` branches disappear while tracing.
    """

    @functools.wraps(fn)
    def traced(*resource_slots: object) -> Schedule:
        present_slots = tuple(slot is not None for slot in resource_slots)
        resources = tuple(
            slot
            for slot, is_present in zip(resource_slots, present_slots, strict=True)
            if is_present
        )

        @functools.wraps(fn)
        def restore_slots(*resource_proxies: object) -> None:
            proxy_iter = iter(resource_proxies)
            restored_slots = tuple(
                next(proxy_iter) if is_present else None for is_present in present_slots
            )
            fn(*restored_slots)

        return schedule(restore_slots)(*resources)

    return traced


def _block_sparse_route_loop_domain(
    route_count: cutlass.Int32,
    *,
    num_insts_kv: int,
) -> cutlass.Int32:
    """Return LOOP iterations after HEAD reserves one candidate per instance."""

    remaining = route_count - cutlass.Int32(num_insts_kv)
    remaining = cute.math.max(remaining, cutlass.Int32(0))
    insts = cutlass.Int32(num_insts_kv)
    return (remaining + insts - cutlass.Int32(1)) // insts


@dataclass(kw_only=True)
class ScheduleTokenThrottleResource(MemoryResource):
    """Order persistent load consumption before the scheduler reuses a slot."""

    @producer_work
    @cute.jit
    def publish_schedule_token(self, stage_info: StageInfo) -> None:
        """Publish that the load task has consumed the current schedule token."""
        del stage_info

    @consumer_work
    @cute.jit
    def consume_schedule_token(self, stage_info: StageInfo) -> None:
        """Wait until the load task no longer needs the current schedule token."""
        del stage_info


@dataclass(kw_only=True)
class SmemKvReuseCreditResource(MemoryResource):
    """One-slot credit carrying the rotating KV256 exchange-stage index.

    Load publishes which drained 64-KiB physical K/V stage Correction may use
    as tail scratch. The one-stage pipeline couples that payload to the same
    ownership epoch: the following Load may use the other two physical stages,
    but cannot publish a new alias until Correction releases this credit after
    all output work completes.
    """

    cfg: cutlass.Constexpr[FmhaDecodeConfig] = None
    _alloc: cutlass.Constexpr[SmemAllocation | None] = None
    scratch_stage_slot: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )

    def __post_init__(self) -> None:
        """Create the routed consumer slot for one physical K/V stage."""
        assert KV_TILE_256_SHARED_FIFO_STAGES == 3, (
            "KV256 reuse-credit rotation requires exactly three shared FIFO stages"
        )
        if not self.cfg.uses_rotating_kv256_exchange:
            raise ValueError(
                "rotating KV scratch requires persistent direct Q64/KV256 "
                "with two KV instructions, one head-dimension stage, and "
                "one load warp"
            )
        self.scratch_stage_slot = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Physical shared-K/V stage reserved for KV256 tail exchange.",
        )

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Allocate the one-word stage payload guarded by this pipeline."""
        if self._alloc is None:
            self._alloc = SmemAllocation(
                name=f"{self.name}_scratchStage",
                size_bytes=4,
                alignment=4,
            )
        return [self._alloc]

    @cute.jit
    def _payload(self, stage_info: StageInfo) -> cutlass.Array:
        """Return the natural next-stage cursor owned by this credit."""
        return cutlass.Array(
            stage_info.context.smem_base.data_ptr() + self._alloc.offset,
            dtype=Int32,
            shape=(1,),
            addrspace=3,
        )

    @cute.jit
    def create_function_variables(
        self,
        context: ResourceContext | None = None,
    ) -> ResourceVars:
        """Initialize the persistent ring cursor before TS tasks start."""
        if cutlass.const_expr(context is not None and context.smem_base is not None):
            payload = cutlass.Array(
                context.smem_base.data_ptr() + self._alloc.offset,
                dtype=Int32,
                shape=(1,),
                addrspace=3,
            )
            thread_idx, _, _ = cute.arch.thread_idx()
            if thread_idx == Int32(0):
                payload[0] = Int32(0)
        return {}

    @producer_work
    @cute.jit
    def publish_scratch_stage(self, stage_info: StageInfo) -> None:
        """Advance the persistent ring cursor and publish the drained stage."""
        num_stages = Int32(KV_TILE_256_SHARED_FIFO_STAGES)
        if prims.elect_sync():
            payload = self._payload(stage_info)
            # Each work commits T = 4 * (loop_end + 1) K/V transactions.
            # Since 4 == 1 (mod 3), the cursor advances by loop_end + 1.
            # loop_end is the resolved per-work domain, so heterogeneous
            # runtime sequence lengths do not inherit a captured host bound.
            payload[0] = (
                Int32(payload[0]) + stage_info.loop_end + Int32(1)
            ) % num_stages

    @consumer_work(returns=scratch_stage_slot)
    @cute.jit
    def read_scratch_stage(self, stage_info: StageInfo) -> Int32:
        """Read the alias only after the matching credit wait completes."""
        num_stages = Int32(KV_TILE_256_SHARED_FIFO_STAGES)
        next_stage = Int32(self._payload(stage_info)[0])
        return (next_stage + num_stages - Int32(1)) % num_stages


@dataclass(kw_only=True)
class PackedDecodeWorkQueue(WorkQueue):
    """CLC work queue that drops packed-Q tiles beyond a batch's Q length."""

    cfg: cutlass.Constexpr[FmhaDecodeConfig] = field(init=False, default=None)
    cu_seqlens_q: Any = field(init=False, default=None)

    def __init__(
        self,
        cfg: FmhaDecodeConfig,
        cu_seqlens_q: cute.Pointer,
        **kwargs: Any,
    ) -> None:
        """Attach packed-Q metadata to the shared persistent work queue."""
        super().__init__(**kwargs)
        self.cfg = cfg
        self.cu_seqlens_q = cu_seqlens_q

    @cute.jit
    def skip_work_tile_if(self, work_tile: Any) -> cutlass.Boolean:
        """Skip a fetched Q group when its first token is outside this batch."""
        q_group_cta_idx, _, b_idx = work_tile.tile_idx
        q_group_idx = cutlass.Int32(q_group_cta_idx)
        if cutlass.const_expr(self.cfg.use_split_kv):
            q_group_idx = q_group_idx // cutlass.Int32(self.cfg.splits_kv)
        _, seq_len_q = _q_seq_bounds(
            self.cfg,
            self.cu_seqlens_q,
            cutlass.Int32(b_idx),
        )
        return _q_group_token_base(self.cfg, q_group_idx) >= seq_len_q


def _page_offsets_consume(
    smem_page_offsets: MemoryResource | None, label: str = "read_offsets"
) -> None:
    """Consume the paged-KV offsets that match the next K/V TMA load."""
    if smem_page_offsets is None:
        return
    # ConsWait: wait until the page-offset producer staged the page IDs.
    smem_page_offsets.wait()
    # ConsWork: expose the page IDs to the K/V load resource.
    getattr(smem_page_offsets, label)()


def _page_offsets_release(smem_page_offsets: MemoryResource | None) -> None:
    """Release the page-offset stage after the matching K/V load is issued."""
    if smem_page_offsets is None:
        return
    # ConsRelease: the load resource no longer needs these page IDs.
    smem_page_offsets.release()


def _page_offsets_produce(
    smem_page_offsets: MemoryResource, label: str, section: FmhaStage
) -> None:
    """Produce page IDs for one K/V load label in a schedule section."""
    # ProdAcquire: reserve a page-offset SMEM pipeline stage.
    smem_page_offsets.acquire()
    # ProdWork: load the page-table entries for K0/K1/V0/V1.
    getattr(smem_page_offsets, label)(section=section)
    # ProdCommit: publish the staged offsets to LoadTask.
    smem_page_offsets.commit()


def _can_hold_native_split_page_window(
    cfg: FmhaDecodeConfig,
    smem_page_offsets: MemoryResource | None,
) -> bool:
    """Return whether one native page-ID stage covers every runtime KV range."""
    if (
        smem_page_offsets is None
        or not smem_page_offsets.use_native_paged_kv
        or not cfg.use_split_kv
        or cfg.use_sliding_window_causal
    ):
        return False
    pages_per_tile = cfg.tile_size_kv // cfg.num_tokens_per_page
    # Runtime ragged lengths can reduce the split-local span in
    # ``num_insts_kv`` increments and therefore change every split rank's
    # aligned starting page. Hold one stage only when every possible span
    # both fits and evenly partitions a 32-ID window; otherwise a shorter row
    # could straddle the next window even when the static maximum does not.
    for local_tiles in range(
        cfg.num_insts_kv,
        cfg.static_local_kv_tiles + 1,
        cfg.num_insts_kv,
    ):
        window_pages = local_tiles * pages_per_tile
        if window_pages <= 0 or window_pages > 32 or 32 % window_pages != 0:
            return False
    return True


def _staged_kv_load(
    resource: MemoryResource,
    label: str,
    section: FmhaStage,
    cfg: FmhaDecodeConfig,
    *,
    smem_page_offsets: MemoryResource | None = None,
    page_offsets_label: str = "read_offsets",
    manage_page_offsets: bool = True,
    cached_page_ids: Any = None,
) -> Any:
    """Issue all head-dim stages for one logical K/V load.

    H256 SwapsMmaAb uses two 128-wide K/V stages, but both slices address the
    same token tile and therefore the same page IDs. Keep one page-offset
    consumer stage live across the complete logical K/V load.
    """
    reuse_page_ids = smem_page_offsets is not None and cfg.num_head_dim_stages_kv > 1
    # Optional ConsWait/ConsWork: fetch page IDs for this logical K/V tile.
    # The cached D256 path performs its ConsumerWork below while materializing
    # the register array; single-stage kernels keep the original no-cache path.
    if manage_page_offsets:
        if reuse_page_ids:
            smem_page_offsets.wait()
        else:
            _page_offsets_consume(smem_page_offsets, page_offsets_label)
    load_label = label
    if reuse_page_ids:
        inst_id = KV_INST1 if label.endswith("1") else KV_INST0
        kv_kind = KV_KIND_V if "_v" in label else KV_KIND_K
        cached_page_ids = smem_page_offsets.cache_page_ids(
            cached_page_ids=cached_page_ids,
            inst_id=inst_id,
            kv_kind=kv_kind,
            section=section,
        )
        load_label = f"{label}_cached"
    for head_dim_stage_idx in range(cfg.num_head_dim_stages_kv):
        # ProdAcquire/ProdWork/ProdCommit: issue one K or V slice into the
        # matching SMEM stage.
        resource.acquire()
        if cached_page_ids is None:
            getattr(resource, load_label)(
                section=section, head_dim_stage_idx=head_dim_stage_idx
            )
        else:
            getattr(resource, load_label)(
                cached_page_ids=cached_page_ids,
                section=section,
                head_dim_stage_idx=head_dim_stage_idx,
            )
        resource.commit()
    # ConsRelease: every head-dimension slice consumed the staged page IDs.
    if manage_page_offsets:
        _page_offsets_release(smem_page_offsets)
    return cached_page_ids


def _produce_staged_page_offsets(
    smem_page_offsets: MemoryResource,
    label: str,
    section: FmhaStage,
    cfg: FmhaDecodeConfig,
) -> None:
    """Produce the page-offset stage shared by one logical K/V load."""
    _ = cfg
    _page_offsets_produce(smem_page_offsets, label, section)


def _consume_staged_qk_mma(
    smem_kv: MemoryResource,
    tmem_s: MemoryResource,
    aliased_p: MemoryResource,
    q_desc: Any,
    k_desc_label: str,
    qk_mma_label: str,
    section: FmhaStage,
    cfg: FmhaDecodeConfig,
) -> None:
    """Consume all K head-dim stages for one QK MMA wave."""
    tmem_s.acquire()
    for head_dim_stage_idx in range(cfg.num_head_dim_stages_kv):
        smem_kv.wait()
        kv_desc = getattr(smem_kv, k_desc_label)()
        if cutlass.const_expr(
            cfg.streams_tmem_p_fragments
            and head_dim_stage_idx == 0
            and (section == FmhaStage.Loop or cfg.use_persistent_scheduler)
        ):
            # Wait as late as possible: K staging overlaps the previous PV,
            # but QK cannot overwrite the matching S/P alias until PV is done.
            # Static HEAD has no previous tile; persistent HEAD may follow the
            # same CTA's tail from another logical work tile and must wait.
            aliased_p.wait_until_reusable_before_qk()
        if cfg.uses_q_desc_ref:
            getattr(tmem_s, f"{qk_mma_label}_from_q_ref")(
                kv_desc=kv_desc,
                head_dim_stage_idx=head_dim_stage_idx,
            )
        else:
            getattr(tmem_s, qk_mma_label)(
                q_desc=q_desc,
                kv_desc=kv_desc,
                head_dim_stage_idx=head_dim_stage_idx,
            )
        smem_kv.release()
    tmem_s.commit()


def _consume_staged_pv_mma(
    smem_kv: MemoryResource,
    tmem_p: MemoryResource,
    tmem_o: MemoryResource,
    v_desc_label: str,
    vp_mma_label: str,
    p_desc_idx: int,
    section: FmhaStage,
    cfg: FmhaDecodeConfig,
) -> None:
    """Consume all V head-dim stages for one PV MMA wave."""
    _ = section
    if cutlass.const_expr(cfg.streams_tmem_p_fragments):
        assert cfg.num_head_dim_stages_kv == 1
        fragment_label = f"{vp_mma_label}_fragment"

        # P fragment 0 is the earliest dependency. Wait for it and for the
        # correction credit before holding the shared V FIFO stage.
        p_tmem_addr = tmem_p.wait_p_fragment(fragment_idx=0)
        tmem_o.acquire()
        smem_kv.wait()
        v_desc = getattr(smem_kv, v_desc_label)()
        getattr(tmem_o, fragment_label)(
            v_desc=v_desc,
            p_tmem_addr=p_tmem_addr,
            fragment_idx=0,
        )

        # Later P fragments may become ready while the previous PV fragment is
        # already executing. Keep every slot live through the complete async
        # UMMA wave so the producer cannot overwrite an operand prematurely.
        for fragment_idx in range(1, cfg.num_softmax_score_fragments):
            p_tmem_addr = tmem_p.wait_p_fragment(fragment_idx=fragment_idx)
            getattr(tmem_o, fragment_label)(
                v_desc=v_desc,
                p_tmem_addr=p_tmem_addr,
                fragment_idx=fragment_idx,
            )
        smem_kv.release()
        tmem_o.commit()
        return

    tmem_p.wait()
    p_desc_0, p_desc_1, p_tmem_addr_0, p_tmem_addr_1 = tmem_p.p_operands()
    p_desc = p_desc_0 if p_desc_idx == KV_INST0 else p_desc_1
    p_tmem_addr = p_tmem_addr_0 if p_desc_idx == KV_INST0 else p_tmem_addr_1
    tmem_o.acquire()
    for head_dim_stage_idx in range(cfg.num_head_dim_stages_kv):
        smem_kv.wait()
        v_desc = getattr(smem_kv, v_desc_label)()
        getattr(tmem_o, vp_mma_label)(
            v_desc_0=v_desc,
            v_desc_1=v_desc,
            p_desc_0=p_desc,
            p_desc_1=p_desc,
            p_tmem_addr_0=p_tmem_addr,
            p_tmem_addr_1=p_tmem_addr,
            inst_idx=p_desc_idx,
            head_dim_stage_idx=head_dim_stage_idx,
        )
        smem_kv.release()
    tmem_o.commit()
    tmem_p.release()


def _work_queue_tail(work_queue: WorkQueue | None, work_tile=None):
    """Finish one persistent-scheduler tile after task-local work drains."""
    _ = work_tile
    if work_queue is None:
        return None
    # ConsWait: wait for the scheduler token for this task.
    work_queue.wait()
    # ConsWork: advance the task-local tile cursor.
    work_queue.get_and_advance_work_tile()
    # ConsRelease: allow the scheduler to hand out the next tile.
    work_queue.release()
    return None


def _schedule_token_throttle_head(
    schedule_token_throttle: MemoryResource | None,
) -> None:
    """Let the scheduler recycle a schedule token slot once Load owns its tile state."""
    if schedule_token_throttle is None:
        return
    schedule_token_throttle.acquire()
    schedule_token_throttle.publish_schedule_token()
    schedule_token_throttle.commit()


def _schedule_token_throttle_tail(
    schedule_token_throttle: MemoryResource | None,
) -> None:
    """Pace scheduler schedule token reuse against the persistent Load task."""
    if schedule_token_throttle is None:
        return
    schedule_token_throttle.wait()
    schedule_token_throttle.consume_schedule_token()
    schedule_token_throttle.release()


def _decode_work_tile_schedule(
    cfg: FmhaDecodeConfig,
    work_queue: WorkQueue | None,
    body: Callable[[], None],
    non_skippable_head: Callable[[], None] | None = None,
) -> None:
    """Trace one worker schedule, making only packed persistent data skippable."""
    if cfg.use_persistent_scheduler:
        assert work_queue is not None
        if not cfg.use_variable_seqlens_q:
            with work_tile_loop(work_queue):
                if non_skippable_head is not None:
                    non_skippable_head()
                body()
                _work_queue_tail(work_queue)
            return
        with work_tile_loop(
            work_queue,
            skip_if=PackedDecodeWorkQueue.skip_work_tile_if,
        ) as work_tiles:
            if non_skippable_head is not None:
                non_skippable_head()
            with work_tiles.skippable():
                body()
            # WorkQueue operations stay inside the persistent loop but outside
            # the skippable data region. DecodeGenTask therefore advances the
            # queue exactly once for both active and inactive packed tiles.
            _work_queue_tail(work_queue)
        return

    # Non-persistent kernels execute one unguarded data region.
    body()
    _work_queue_tail(work_queue)


def _decode_work_tile_schedule_with_invariant_bridge(
    cfg: FmhaDecodeConfig,
    work_queue: WorkQueue | None,
    invariant_setup: Callable[[], None],
    bridge: Callable[[], Any],
    active_prelude: Callable[[], None],
    body: Callable[[Any], None],
) -> None:
    """Trace data work around an invariant value needed across schedule phases.

    The bridge is reserved for descriptor construction that reads only
    task-local descriptor metadata. It must not issue a memory operation,
    barrier, or pipeline-state transition.
    """
    if cfg.use_persistent_scheduler:
        assert work_queue is not None
        # Descriptor backing storage depends only on the task's SMEM
        # allocation, so initialize it once before the persistent loop.
        invariant_setup()
        if not cfg.use_variable_seqlens_q:
            with work_tile_loop(work_queue):
                active_prelude()
                invariant = bridge()
                body(invariant)
                _work_queue_tail(work_queue)
            return
        with work_tile_loop(
            work_queue,
            skip_if=PackedDecodeWorkQueue.skip_work_tile_if,
        ) as work_tiles:
            with work_tiles.skippable():
                active_prelude()
                # Packed persistent QK reads Q's consumer-work stage directly,
                # so it needs no descriptor task local crossing this guard.
                body(None)
            _work_queue_tail(work_queue)
        return

    # Fixed paths initialize descriptor state, wait for Q, construct the
    # descriptor, and then issue the remaining MMA work.
    invariant_setup()
    active_prelude()
    invariant = bridge()
    body(invariant)
    _work_queue_tail(work_queue)


@cute.jit
def _load_prepared_sparse_row_warp(
    row_route_offsets: cute.Pointer,
    row_route_counts: cute.Pointer,
    row_address: cutlass.Int32,
    lane_idx: cutlass.Int32,
) -> tuple[cutlass.Int32, cutlass.Int32]:
    """Load one prepared row header once per warp and broadcast it.

    DecodeGenTask.get_domain is a regular Task override, so staged control flow
    lives in this JIT helper rather than in the Python method itself.
    """

    loaded_row_route_begin = cutlass.Int32(0)
    loaded_route_count = cutlass.Int32(0)
    if lane_idx == cutlass.Int32(0):
        loaded_row_route_begin = cutlass.Int32(row_route_offsets[row_address])
        loaded_route_count = cutlass.Int32(row_route_counts[row_address])
    row_route_begin = _warp_broadcast_i32(loaded_row_route_begin, 0)
    route_count = _warp_broadcast_i32(loaded_route_count, 0)
    return row_route_begin, route_count


class DecodeGenTask(Task):
    """Decode-gen task with task-cached values used by hot resource paths."""

    def __init__(self, **kwargs: TaskKwarg) -> None:
        """Capture decode-specific task config and initialize cache slots."""
        self.cfg = kwargs.pop("cfg", None)
        self.seqlens_kv = kwargs.pop("seqlens_kv", None)
        self.paged_kv_indptr = kwargs.pop("paged_kv_indptr", None)
        self.sparse_row_route_offsets = kwargs.pop("sparse_row_route_offsets", None)
        self.sparse_row_route_counts = kwargs.pop("sparse_row_route_counts", None)
        self.num_heads_kv = kwargs.pop("num_heads_kv", None)
        self.max_seq_len_kv = kwargs.pop("max_seq_len_kv", cutlass.Int32(0))
        self.seq_len_q = kwargs.pop("seq_len_q", None)
        self.domain_bias = kwargs.pop("domain_bias", 0)
        self.q_bound_resources = kwargs.pop("q_bound_resources", ())
        super().__init__(**kwargs)
        self._tmem_base_offset = cutlass.Int32(0)
        self._warp_grp_thread_idx = cutlass.Int32(0)
        self._local_warp_idx = cutlass.Int32(0)
        self._lane_idx = cutlass.Int32(0)
        self._seq_len_kv = cutlass.Int32(0)
        self._kv_request_begin = cutlass.Int32(0)
        self._kv_page_idx_ub = cutlass.Int32(0)
        self._kv_raw_tile_base = cutlass.Int32(0)
        self._kv_valid_tile_end = cutlass.Int32(0)
        self._kv_window_start = cutlass.Int32(0)
        # Keep the DSL loop-carried task structure stable. Persistent loops
        # assign this liveness marker on every path, so it must exist before
        # the first dynamic ``while`` is lowered by the stock compiler.
        self.dummy = cutlass.Boolean(False)

    def init_variables(self, context: cute.Pointer | None = None) -> None:
        """Initialize per-task thread and TMEM cached values."""
        super().init_variables(context)
        # Cache thread identity inside the 4-warp task group for TMEM and SMEM
        # resource operations.
        tidx, _, _ = cute.arch.thread_idx()
        warp_grp_start = cutlass.Int32((self.warp_idx // 4) * 4 * 32)
        self._warp_grp_thread_idx = tidx - warp_grp_start
        self._local_warp_idx = self._warp_grp_thread_idx >> cutlass.Int32(5)
        self._lane_idx = self._warp_grp_thread_idx & cutlass.Int32(0x1F)

        if context is not None and context.tmem_ptr_i32 is not None:
            # tcgen05_alloc provides the CTA's TMEM base through shared
            # context. Broadcast it so all lanes issue TMEM operations from the
            # same base column.
            loaded = cutlass.Int32(context.tmem_ptr_i32.load())
            self._tmem_base_offset = _warp_broadcast_i32(loaded, 0)

    @cute.jit
    def make_task_cache(
        self,
    ) -> tuple[
        cutlass.Int32,
        cutlass.Int32,
        cutlass.Int32,
        cutlass.Int32,
        cutlass.Int32,
        cutlass.Int32,
        cutlass.Int32,
        cutlass.Int32,
        cutlass.Int32,
        cutlass.Int32,
    ]:
        """Return cached task-local values passed through StageInfo."""
        # The cache is threaded through StageInfo so resource methods can read
        # task-local constants without recomputing them or adding more
        # resource variables to every schedule edge.
        return (
            self._tmem_base_offset,
            self._warp_grp_thread_idx,
            self._local_warp_idx,
            self._lane_idx,
            self._seq_len_kv,
            self._kv_request_begin,
            self._kv_page_idx_ub,
            self._kv_raw_tile_base,
            self._kv_valid_tile_end,
            self._kv_window_start,
        )

    @cute.jit
    def _refresh_packed_q_bounds(self, work_tile: Any) -> None:
        """Attach one active CLC tile's packed Q bounds to its data resources."""
        assert self.cfg is not None
        assert isinstance(self.work_queue, PackedDecodeWorkQueue)
        _, _, b_idx = work_tile.tile_idx
        q_token_offset, seq_len_q = _q_seq_bounds(
            self.cfg,
            self.work_queue.cu_seqlens_q,
            cutlass.Int32(b_idx),
        )
        if cutlass.const_expr(self.seq_len_q is not None):
            self.seq_len_q = seq_len_q
        for resource, updates_q_token_offset in self.q_bound_resources:
            if cutlass.const_expr(updates_q_token_offset):
                resource.q_token_offset = q_token_offset
            resource.seq_len_q = seq_len_q

    @cute.jit
    def _run_packed_skip_iteration(
        self,
        work_tile: Any,
        context: ResourceContext | None = None,
    ) -> None:
        """Advance one inactive tile through WorkQueue bookkeeping only."""
        # Packed schedules place every data-path entry inside ``skippable()``;
        # only the WorkQueue wait/advance/release tail remains outside it. Use
        # a unit domain solely to populate that tail's StageInfo. In particular,
        # do not call get_domain(), which would read per-batch KV metadata for a
        # tile whose Q sequence is empty or overlaunched.
        bookkeeping_domain = cutlass.Int32(1)
        for is_skippable_head, head_entries in self._head_exec_groups:
            if cutlass.const_expr(not is_skippable_head):
                self._run_head_entry_group(
                    head_entries,
                    work_tile,
                    bookkeeping_domain,
                    context,
                )
        for is_skippable_tail, tail_entries in self._tail_exec_groups:
            if cutlass.const_expr(not is_skippable_tail):
                self._run_tail_entry_group(
                    tail_entries,
                    work_tile,
                    bookkeeping_domain,
                    context,
                )

    @cute.jit
    def _run_task_body_impl(
        self,
        work_tile: cute.Coord,
        skip_work_tile: Any = None,
        context: ResourceContext | None = None,
    ) -> None:
        """Run one ordinary task tile and synchronize attention-sink tails."""
        Task._run_task_body_impl(
            self,
            work_tile,
            skip_work_tile,
            context=context,
        )
        if cutlass.const_expr(
            self.cfg is not None
            and self.cfg.use_persistent_scheduler
            and self.cfg.use_attention_sinks
        ):
            # Attention sinks extend correction's tail beyond the ordinary
            # task graph.  Keep all persistent tasks on the same logical tile
            # until that tail has drained.  KV256's shared-KV alias instead
            # uses a narrow Load/Correction credit in the captured schedule.
            if cutlass.const_expr(
                self.cfg.use_variable_seqlens_q and self.cfg.use_persistent_scheduler
            ):
                assert isinstance(self.work_queue, PackedDecodeWorkQueue)
                q_group_cta_idx, _, b_idx = work_tile.tile_idx
                _, seq_len_q = _q_seq_bounds(
                    self.cfg,
                    self.work_queue.cu_seqlens_q,
                    cutlass.Int32(b_idx),
                )
                q_group_idx = cutlass.Int32(q_group_cta_idx)
                if _q_group_token_base(self.cfg, q_group_idx) < seq_len_q:
                    prims.barrier_cta_sync(12, thread_count=16 * 32)
            else:
                prims.barrier_cta_sync(12, thread_count=16 * 32)

    @cute.jit
    def _run_task_body_persistent(
        self,
        context: ResourceContext | None = None,
    ) -> None:
        """Drain inactive packed tiles before each unconditional active body."""
        use_packed_early_stop = (
            self.cfg is not None
            and self.cfg.use_variable_seqlens_q
            and self.cfg.use_persistent_scheduler
            and self._has_skip_if
        )
        if cutlass.const_expr(not use_packed_early_stop):
            Task._run_task_body_persistent(self, context)
            return

        assert self.work_queue is not None
        work_tile = self.work_queue.initial_work_tile_info()
        self.work_queue._set_consumer_var_from_ts("work_tile", work_tile)

        self._run_pre_work_loop_entries(work_tile, context)
        work_tile = self.work_queue._get_consumer_var_from_ts("work_tile")
        for resource in self.dst_resources:
            if cutlass.const_expr(
                resource.pipeline_config is not None
                and resource.pipeline_config.advance_on_acquire
                and not self._is_fork_secondary(resource)
            ):
                self._thread_advance_on_acquire_state(resource)

        # Consume overlaunched Q groups before entering the active loop. The
        # inner loop executes only the non-skippable WorkQueue tail, so no TMA,
        # descriptor, pipeline, task data, or sink barrier is issued.
        while work_tile.is_valid_tile and self._should_skip_work_tile(work_tile):
            self._run_packed_skip_iteration(work_tile, context)
            work_tile = self.work_queue._get_consumer_var_from_ts("work_tile")
            self.dummy = cutlass.Boolean(True)

        while work_tile.is_valid_tile:
            self._refresh_packed_q_bounds(work_tile)
            # The tile is known active here. Running the complete schedule
            # without a dynamic skip guard keeps HEAD-produced pipeline state
            # in scope for LOOP and TAIL.
            Task._run_task_body_impl(self, work_tile, None, context=context)
            if cutlass.const_expr(self.cfg.use_attention_sinks):
                prims.barrier_cta_sync(12, thread_count=16 * 32)
            work_tile = self.work_queue._get_consumer_var_from_ts("work_tile")
            self.dummy = cutlass.Boolean(True)

            while work_tile.is_valid_tile and self._should_skip_work_tile(work_tile):
                self._run_packed_skip_iteration(work_tile, context)
                work_tile = self.work_queue._get_consumer_var_from_ts("work_tile")
                self.dummy = cutlass.Boolean(True)

        self._run_post_work_loop_entries(work_tile, context)
        for resource in self.dst_resources:
            if cutlass.const_expr(
                resource.pipeline_config is not None
                and resource is not self.work_queue
                and not self._is_fork_secondary(resource)
            ):
                pipeline_config = resource.pipeline_config
                assert pipeline_config is not None
                if cutlass.const_expr(pipeline_config.advance_on_acquire):
                    self._thread_advance_on_acquire_state(resource)
                self._producer_tail(resource)
        if cutlass.const_expr(
            self.work_queue in self.dst_resources
            and self.work_queue.pipeline_config is not None
        ):
            self.work_queue.producer_tail()
        self.dummy = cutlass.Boolean(True)

    def get_domain(self, tile_coord: cute.Coord) -> cutlass.Int32 | int:
        """Return this task's loop domain for one static or persistent tile."""
        if self.cfg is None:
            return self.domain

        # Sparse rows have a runtime-dependent number of prepared KV routes
        # even when sequence lengths are static. Load their compact header
        # before the fixed-dense early return below. Persistent workers pass
        # the logical WorkQueue tile here, so static and CLC schedules share
        # the same (q_group, head, batch) mapping.
        if self.cfg.use_block_sparse:
            # Validation-only TaskManagers intentionally omit prepared GMEM
            # pointers and retain their configured static graph domain.
            row_route_offsets = self.sparse_row_route_offsets
            row_route_counts = self.sparse_row_route_counts
            if row_route_offsets is None or row_route_counts is None:
                return self.domain
            if self.num_heads_kv is None:
                raise ValueError(
                    "num_heads_kv is required to resolve block-sparse rows"
                )

            q_group_idx = cutlass.Int32(tile_coord[0])
            h_idx = cutlass.Int32(tile_coord[1])
            b_idx = cutlass.Int32(tile_coord[2])
            q_token_base = _q_group_token_base(self.cfg, q_group_idx)

            q_block = q_token_base // self.cfg.q_block_size
            num_q_blocks = (
                self.cfg.max_seq_len_q + self.cfg.q_block_size - 1
            ) // self.cfg.q_block_size
            row_address = (b_idx * self.num_heads_kv + h_idx) * num_q_blocks + q_block

            row_route_begin, route_count = _load_prepared_sparse_row_warp(
                row_route_offsets,
                row_route_counts,
                cutlass.Int32(row_address),
                self._lane_idx,
            )

            # Sparse route-span accessors share two underlying cache words
            # with paged KV. Clear dense/paged-only coordinates on every
            # logical tile because persistent tasks reuse the same task object.
            if self.seqlens_kv is None:
                self._seq_len_kv = self.max_seq_len_kv
            else:
                self._seq_len_kv = cutlass.Int32(self.seqlens_kv[b_idx])
            self._kv_request_begin = row_route_begin
            self._kv_page_idx_ub = route_count
            self._kv_raw_tile_base = cutlass.Int32(0)
            self._kv_valid_tile_end = route_count
            self._kv_window_start = cutlass.Int32(0)

            loop_domain = _block_sparse_route_loop_domain(
                route_count,
                num_insts_kv=self.cfg.num_insts_kv,
            )
            return loop_domain + cutlass.Int32(self.domain_bias)

        # Resolve the sequence length for this work tile. Static-seqlen kernels
        # can use the configured max length; variable-seqlen kernels read the
        # batch-specific length from GMEM.
        b_idx = cutlass.Int32(tile_coord[2])
        if cutlass.const_expr(self.paged_kv_indptr is not None):
            request_begin = cutlass.Int32(self.paged_kv_indptr[b_idx])
            request_end = cutlass.Int32(self.paged_kv_indptr[b_idx + cutlass.Int32(1)])
            self._kv_request_begin = request_begin
            self._kv_page_idx_ub = request_end - request_begin - cutlass.Int32(1)
        if self.seqlens_kv is None:
            seq_len_kv = cutlass.Int32(self.max_seq_len_kv)
        else:
            seq_len_kv = cutlass.Int32(self.seqlens_kv[b_idx])
        self._seq_len_kv = seq_len_kv
        if cutlass.const_expr(self.paged_kv_indptr is not None):
            self._kv_page_idx_ub = cute.math.min(
                self._kv_page_idx_ub,
                _runtime_last_valid_page_idx(self.cfg, seq_len_kv),
            )
        if (
            self.seqlens_kv is None
            and not self.cfg.use_split_kv
            and not self.cfg.uses_runtime_q_kv_union
        ):
            return self.domain
        tile_size_kv = cutlass.Int32(self.cfg.tile_size_kv)

        # Q-independent full-K nonsplit decode has no leading window skip.
        # Resolve its runtime loop span directly and avoid repeating the general
        # causal/window/split coordinate construction in every task warp.
        if cutlass.const_expr(
            not self.cfg.use_split_kv
            and not self.cfg.uses_runtime_q_kv_union
            and not self.cfg.use_sliding_window_causal
        ):
            total_kv_tiles = (
                seq_len_kv + tile_size_kv - cutlass.Int32(1)
            ) // tile_size_kv
            self._kv_window_start = cutlass.Int32(0)
            self._kv_valid_tile_end = total_kv_tiles
            self._kv_raw_tile_base = cutlass.Int32(0)
            remaining_kv_tiles = cute.math.max(
                total_kv_tiles - cutlass.Int32(self.cfg.num_insts_kv),
                cutlass.Int32(0),
            )
            num_insts_kv = cutlass.Int32(self.cfg.num_insts_kv)
            loop_domain = (
                remaining_kv_tiles + num_insts_kv - cutlass.Int32(1)
            ) // num_insts_kv
            return loop_domain + cutlass.Int32(self.domain_bias)

        # Decode the logical Q tile with the configured physical split fanout,
        # then derive its causal/window K union and useful runtime split prefix.
        q_group_cta_idx = cutlass.Int32(tile_coord[0])
        q_group_idx = q_group_cta_idx
        if self.cfg.use_split_kv:
            q_group_idx = q_group_cta_idx // cutlass.Int32(self.cfg.splits_kv)
        q_token_base = _q_group_token_base(self.cfg, q_group_idx)
        seq_len_q = (
            cutlass.Int32(self.cfg.max_seq_len_q)
            if self.seq_len_q is None
            else cutlass.Int32(self.seq_len_q)
        )
        self._kv_window_start = _sliding_window_start_idx(
            self.cfg,
            seq_len_kv,
            seq_len_q,
            q_token_base,
        )
        skipped_tiles = self._kv_window_start // tile_size_kv
        total_kv_tiles = _runtime_total_kv_tiles(
            self.cfg,
            seq_len_kv,
            seq_len_q,
            q_token_base,
        )
        self._kv_valid_tile_end = skipped_tiles + total_kv_tiles

        # Split-KV groups CTAs by K/V split. The loop domain is rounded so
        # each CTA in the group executes the same number of instruction pairs.
        if self.cfg.use_split_kv:
            # The useful active prefix is derived from this configured-fanout
            # local span at kernel entry. Recomputing it in every scheduled
            # task produces the same span but adds a runtime integer division
            # to each task warp.
            splits_kv = cutlass.Int32(self.cfg.splits_kv)
            num_insts_kv = cutlass.Int32(self.cfg.num_insts_kv)
            tiles_per_cta_group = splits_kv * num_insts_kv
            num_groups = (
                total_kv_tiles + tiles_per_cta_group - cutlass.Int32(1)
            ) // tiles_per_cta_group
            total_kv_tiles = cute.math.max(
                num_groups * num_insts_kv,
                num_insts_kv,
            )
            # Physical split coordinates and page-cache strides are laid out with
            # the configured launch fanout.  Runtime pruning only shortens the
            # useful prefix; it must not renumber the remaining CTAs.
            split_idx = cutlass.Int32(tile_coord[0]) % cutlass.Int32(self.cfg.splits_kv)
            self._kv_raw_tile_base = skipped_tiles + split_idx * total_kv_tiles
        else:
            self._kv_raw_tile_base = skipped_tiles
        remaining_kv_tiles = cute.math.max(
            total_kv_tiles - cutlass.Int32(self.cfg.num_insts_kv), cutlass.Int32(0)
        )
        num_insts_kv = cutlass.Int32(self.cfg.num_insts_kv)
        loop_domain = (
            remaining_kv_tiles + num_insts_kv - cutlass.Int32(1)
        ) // num_insts_kv
        # All tasks share the MMA-loop domain; tail-only tasks add a bias.
        return loop_domain + cutlass.Int32(self.domain_bias)


# ======================================================================
# LoadTask — warp 13 (or warp 15 under CLC persistent), 1 warp
# K and V share a single SmemKv ring; loads alternate K and V tiles.
#   HEAD:    Q + K0 + K1
#   LOOP[i]: K(i+2) + V(i)
#   TAIL:    V(last-1) + V(last)
# Dense paged-KV consumes page offsets produced by PageTableTask. Sparse
# paged-KV instead consumes physical page IDs retained with its prepared route.
# ======================================================================
def _resolve_and_store_sparse_route(
    sparse_kv_metadata: MemoryResource | None,
    section: FmhaStage,
) -> tuple[Any, Any, Any, Any] | None:
    """Resolve one prepared route and retain it for the matching K/V pair."""

    if sparse_kv_metadata is None:
        return None
    (
        resolved_origin0,
        resolved_origin1,
        resolved_atom_validity,
        route_record_word_offset,
    ) = sparse_kv_metadata.resolve_route(section=section)
    sparse_kv_metadata.store_route(
        resolved_origin0=resolved_origin0,
        resolved_origin1=resolved_origin1,
        resolved_atom_validity=resolved_atom_validity,
        route_record_word_offset=route_record_word_offset,
    )
    return (
        resolved_origin0,
        resolved_origin1,
        resolved_atom_validity,
        route_record_word_offset,
    )


def _publish_sparse_softmax_route(
    sparse_softmax_metadata: MemoryResource | None,
    route: tuple[Any, Any, Any, Any] | None,
) -> None:
    """Stage one resolved route for its paired Softmax consumer."""

    if sparse_softmax_metadata is None:
        return
    assert route is not None
    (
        resolved_origin0,
        resolved_origin1,
        resolved_atom_validity,
        route_record_word_offset,
    ) = route
    sparse_softmax_metadata.acquire()
    sparse_softmax_metadata.store_route(
        resolved_origin0=resolved_origin0,
        resolved_origin1=resolved_origin1,
        resolved_atom_validity=resolved_atom_validity,
        route_record_word_offset=route_record_word_offset,
    )
    sparse_softmax_metadata.commit()


def create_load_task(
    smem_q: MemoryResource,
    smem_kv: MemoryResource,
    work_queue: WorkQueue | None,
    schedule_token_throttle: MemoryResource | None,
    smem_kv_reuse_credit: MemoryResource | None,
    cfg: FmhaDecodeConfig,
    *,
    domain: int | cutlass.Int32,
    smem_page_offsets: MemoryResource | None = None,
    sparse_kv_metadata0: MemoryResource | None = None,
    sparse_kv_metadata1: MemoryResource | None = None,
    sparse_softmax_metadata0: MemoryResource | None = None,
    sparse_softmax_metadata1: MemoryResource | None = None,
    warp_idx: int | None = None,
    num_warps: int | None = None,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> Task:
    """Create the shared-KV load task and optional page-offset dependency."""
    hold_page_window = _can_hold_native_split_page_window(cfg, smem_page_offsets)

    def load_schedule_body(
        smem_q: MemoryResource,
        smem_kv: MemoryResource,
        smem_page_offsets: MemoryResource | None,
        schedule_token_throttle: MemoryResource | None,
        smem_kv_reuse_credit: MemoryResource | None,
        sparse_kv_metadata0: MemoryResource | None = None,
        sparse_kv_metadata1: MemoryResource | None = None,
        sparse_softmax_metadata0: MemoryResource | None = None,
        sparse_softmax_metadata1: MemoryResource | None = None,
    ) -> None:
        """Build the shared-KV load cadence for HEAD, LOOP, and TAIL."""
        smem_q.init_load_state()
        smem_kv.init_load_state()
        for sparse_resource in (
            sparse_kv_metadata0,
            sparse_kv_metadata1,
            sparse_softmax_metadata0,
            sparse_softmax_metadata1,
        ):
            if sparse_resource is not None:
                sparse_resource.init_load_state()
        cached_page_ids = None
        if smem_page_offsets is not None:
            if cfg.num_head_dim_stages_kv > 1:
                cached_page_ids = smem_page_offsets.init_cached_read_state()
            else:
                smem_page_offsets.init_read_state()

        def _kv_load(label: str, section: FmhaStage) -> None:
            """Run one staged shared-KV load with optional page-offset handoff."""
            nonlocal cached_page_ids
            cached_page_ids = _staged_kv_load(
                smem_kv,
                label,
                section,
                cfg,
                smem_page_offsets=smem_page_offsets,
                manage_page_offsets=not hold_page_window,
                cached_page_ids=cached_page_ids,
            )

        # HEAD: load Q once, then prefetch the first two K tiles.
        smem_q.acquire()
        smem_q.tma_load()
        smem_q.commit()
        if hold_page_window:
            # The native K/V caches share one page table. Keep its single
            # 32-ID consumer stage live across every K/V and head-dimension
            # load owned by this split CTA, matching the reference page-window
            # lifetime and avoiding redundant pipeline handoffs.
            if cfg.num_head_dim_stages_kv > 1:
                smem_page_offsets.wait()
            else:
                _page_offsets_consume(smem_page_offsets)
        if sparse_kv_metadata0 is None:
            for label in ("load_k0", "load_k1"):
                _kv_load(label, FmhaStage.Head)
        else:
            route0 = _resolve_and_store_sparse_route(
                sparse_kv_metadata0, FmhaStage.Head
            )
            _kv_load("load_k0", FmhaStage.Head)
            route1 = _resolve_and_store_sparse_route(
                sparse_kv_metadata1, FmhaStage.Head
            )
            _kv_load("load_k1", FmhaStage.Head)
            # Issue both K tiles before either metadata FIFO can backpressure
            # the load warp, matching the split-resource sparse cadence.
            _publish_sparse_softmax_route(sparse_softmax_metadata0, route0)
            _publish_sparse_softmax_route(sparse_softmax_metadata1, route1)
        if smem_kv_reuse_credit is not None:
            # K0/K1 occupy the two stages disjoint from the previous work's
            # scratch. Acquire only before issuing the third K/V transaction.
            smem_kv_reuse_credit.acquire()

        # LOOP: each iter prefetches the full ``num_insts_kv`` K/V pair set.
        # When P aliases the consumed S columns, MMA must consume each V/P pair
        # before the following same-instance QK overwrites S. Keep the
        # shared-ring producer order identical to that consumer order.
        loop_labels = (
            ("load_v0", "load_k0", "load_v1", "load_k1")
            if cfg.uses_two_inst_tmem_p
            else ("load_k0", "load_v0", "load_k1", "load_v1")
        )
        with domain_loop(0, domain, 1, unroll=1):
            if sparse_kv_metadata0 is None:
                for label in loop_labels:
                    _kv_load(label, FmhaStage.Loop)
            else:
                # Follow the dense KV256 stage order exactly. Each V consumes
                # its retained route before the matching K label replaces it.
                loop_routes = []
                for label in loop_labels:
                    route = None
                    sparse_softmax_metadata = None
                    if label == "load_k0":
                        route = _resolve_and_store_sparse_route(
                            sparse_kv_metadata0, FmhaStage.Loop
                        )
                        sparse_softmax_metadata = sparse_softmax_metadata0
                    elif label == "load_k1":
                        route = _resolve_and_store_sparse_route(
                            sparse_kv_metadata1, FmhaStage.Loop
                        )
                        sparse_softmax_metadata = sparse_softmax_metadata1
                    _kv_load(label, FmhaStage.Loop)
                    if route is not None:
                        loop_routes.append((sparse_softmax_metadata, route))
                for sparse_softmax_metadata, route in loop_routes:
                    _publish_sparse_softmax_route(sparse_softmax_metadata, route)

        # TAIL: after no more future K tiles are needed, load the final two V
        # tiles consumed by the final BMM2 calls.
        for label in ("load_v0", "load_v1"):
            _kv_load(label, FmhaStage.Tail)
        if smem_kv_reuse_credit is not None:
            # Publish the physical stage drained by this work together with
            # the ownership token consumed by the correction tail.
            smem_kv_reuse_credit.publish_scratch_stage()
            smem_kv_reuse_credit.commit()
        if hold_page_window:
            _page_offsets_release(smem_page_offsets)

    @_schedule_with_optional_resources
    def load_schedule(
        smem_q: MemoryResource,
        smem_kv: MemoryResource,
        smem_page_offsets: MemoryResource | None,
        sparse_kv_metadata0: MemoryResource | None,
        sparse_kv_metadata1: MemoryResource | None,
        sparse_softmax_metadata0: MemoryResource | None,
        sparse_softmax_metadata1: MemoryResource | None,
        work_queue: WorkQueue | None,
        schedule_token_throttle: MemoryResource | None,
        smem_kv_reuse_credit: MemoryResource | None,
    ) -> None:
        """Schedule shared-KV loads with only the resources in this profile."""

        _decode_work_tile_schedule(
            cfg,
            work_queue,
            lambda: load_schedule_body(
                smem_q,
                smem_kv,
                smem_page_offsets,
                schedule_token_throttle,
                smem_kv_reuse_credit,
                sparse_kv_metadata0,
                sparse_kv_metadata1,
                sparse_softmax_metadata0,
                sparse_softmax_metadata1,
            ),
            lambda: _schedule_token_throttle_head(schedule_token_throttle),
        )

    sparse_resources = (
        sparse_kv_metadata0,
        sparse_kv_metadata1,
        sparse_softmax_metadata0,
        sparse_softmax_metadata1,
    )
    sparse_resources_present = tuple(
        resource is not None for resource in sparse_resources
    )
    has_sparse_metadata = any(sparse_resources_present)
    if has_sparse_metadata and not all(sparse_resources_present):
        raise ValueError("shared sparse K/V requires both route/Softmax pairs")
    if has_sparse_metadata and smem_page_offsets is not None:
        raise ValueError("block-sparse and paged-KV cannot share a load task")

    captured_schedule = load_schedule(
        smem_q,
        smem_kv,
        smem_page_offsets,
        sparse_kv_metadata0,
        sparse_kv_metadata1,
        sparse_softmax_metadata0,
        sparse_softmax_metadata1,
        work_queue,
        schedule_token_throttle,
        smem_kv_reuse_credit,
    )
    src = []
    for sparse_kv_metadata in (sparse_kv_metadata0, sparse_kv_metadata1):
        if sparse_kv_metadata is not None:
            src.append(sparse_kv_metadata)
    if smem_page_offsets is not None:
        src.append(smem_page_offsets)
    if work_queue is not None:
        src.append(work_queue)
    dst = [smem_q, smem_kv]
    for sparse_resource in sparse_resources:
        if sparse_resource is not None and sparse_resource not in dst:
            dst.append(sparse_resource)
    if schedule_token_throttle is not None:
        dst.append(schedule_token_throttle)
    if smem_kv_reuse_credit is not None:
        dst.append(smem_kv_reuse_credit)
    return task_class(
        src_resources=src,
        dst_resources=dst,
        q_bound_resources=((smem_q, True), (smem_kv, False)),
        cfg=cfg,
        warp_idx=cfg.load_warp_idx if warp_idx is None else warp_idx,
        num_warps=cfg.load_num_warps if num_warps is None else num_warps,
        schedule=captured_schedule,
        num_registers=cfg.mma_load_task_num_registers,
        name="LoadTask",
        **kw,
    )


def create_page_offsets_task(
    smem_page_offsets: MemoryResource,
    work_queue: WorkQueue | None,
    cfg: FmhaDecodeConfig,
    *,
    domain: int | cutlass.Int32,
    warp_idx: int | None = None,
    num_warps: int | None = None,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> Task:
    """Prefetch page-table entries that LoadTask consumes for paged KV.

    The schedule matches LoadTask's K/V cadence exactly so the page-offsets
    ring and the SmemKv ring stay aligned.
    """
    hold_page_window = _can_hold_native_split_page_window(cfg, smem_page_offsets)

    def page_offsets_schedule_body(
        smem_page_offsets: MemoryResource,
    ) -> None:
        """Schedule page-offset prefetches for the shared-KV load cadence."""
        smem_page_offsets.init_load_state()
        if hold_page_window:
            # K0's aligned 32-ID window covers every contiguous tile assigned
            # to this split CTA, and native CSR uses those same IDs for V.
            _page_offsets_produce(smem_page_offsets, "load_k0", FmhaStage.Head)
            # Preserve the runtime domain contract even though this fast path
            # needs no per-iteration page-window work.
            with domain_loop(0, domain, 1, unroll=1):
                pass
            return
        # Page offsets are staged by a separate producer so the load warp can
        # issue K/V TMA copies without reading page tables itself.
        # HEAD: produce page IDs for the two prefetched K tiles.
        _produce_staged_page_offsets(smem_page_offsets, "load_k0", FmhaStage.Head, cfg)
        _produce_staged_page_offsets(smem_page_offsets, "load_k1", FmhaStage.Head, cfg)

        # LOOP: mirror LoadTask's K/V production cadence exactly.
        loop_labels = (
            ("load_v0", "load_k0", "load_v1", "load_k1")
            if cfg.uses_two_inst_tmem_p
            else ("load_k0", "load_v0", "load_k1", "load_v1")
        )
        with domain_loop(0, domain, 1, unroll=1):
            for label in loop_labels:
                _produce_staged_page_offsets(
                    smem_page_offsets, label, FmhaStage.Loop, cfg
                )

        # TAIL: produce page IDs for the final two V loads.
        for label in ("load_v0", "load_v1"):
            _produce_staged_page_offsets(smem_page_offsets, label, FmhaStage.Tail, cfg)

    @schedule
    def page_offsets_schedule(
        smem_page_offsets: MemoryResource,
        work_queue: WorkQueue | None = None,
    ) -> None:
        """Wrap page-offset data work in packed persistent skip handling."""
        _decode_work_tile_schedule(
            cfg,
            work_queue,
            lambda: page_offsets_schedule_body(smem_page_offsets),
        )

    captured_schedule = (
        page_offsets_schedule(smem_page_offsets)
        if work_queue is None
        else page_offsets_schedule(smem_page_offsets, work_queue)
    )
    src = []
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[smem_page_offsets],
        q_bound_resources=((smem_page_offsets, False),),
        cfg=cfg,
        warp_idx=cfg.page_offsets_warp_idx if warp_idx is None else warp_idx,
        num_warps=cfg.page_offsets_num_warps if num_warps is None else num_warps,
        schedule=captured_schedule,
        num_registers=cfg.mma_load_task_num_registers,
        name="PageTableTask",
        **kw,
    )


def create_page_offsets_task_split_kv(
    smem_page_offsets_k: MemoryResource,
    smem_page_offsets_v: MemoryResource,
    work_queue: WorkQueue | None,
    cfg: FmhaDecodeConfig,
    *,
    domain: int | cutlass.Int32,
    warp_idx: int | None = None,
    num_warps: int | None = None,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> Task:
    """Prefetch split-KV K/V page windows with independent consumer state."""

    def page_offsets_schedule_body(
        smem_page_offsets_k: MemoryResource,
        smem_page_offsets_v: MemoryResource,
    ) -> None:
        """Publish one page-offset stage for every paired K/V tile."""
        smem_page_offsets_k.init_load_state()
        smem_page_offsets_v.init_load_state()

        # HEAD: publish the initial K0/K1 pair as two independent stages.
        _page_offsets_produce(smem_page_offsets_k, "load_k0", FmhaStage.Head)
        _page_offsets_produce(smem_page_offsets_k, "load_k1", FmhaStage.Head)

        # LOOP: mirror LoadTask's cross-resource consumption order exactly.
        with domain_loop(0, domain, 1, unroll=1):
            _page_offsets_produce(smem_page_offsets_v, "load_v0", FmhaStage.Loop)
            _page_offsets_produce(smem_page_offsets_k, "load_k0", FmhaStage.Loop)
            _page_offsets_produce(smem_page_offsets_v, "load_v1", FmhaStage.Loop)
            _page_offsets_produce(smem_page_offsets_k, "load_k1", FmhaStage.Loop)

        # TAIL: drain V0/V1 through distinct page-offset stages.
        _page_offsets_produce(smem_page_offsets_v, "load_v0", FmhaStage.Tail)
        _page_offsets_produce(smem_page_offsets_v, "load_v1", FmhaStage.Tail)

    @schedule
    def page_offsets_schedule(
        smem_page_offsets_k: MemoryResource,
        smem_page_offsets_v: MemoryResource,
        work_queue: WorkQueue | None = None,
    ) -> None:
        """Wrap split page-offset work in packed persistent skip handling."""
        _decode_work_tile_schedule(
            cfg,
            work_queue,
            lambda: page_offsets_schedule_body(
                smem_page_offsets_k,
                smem_page_offsets_v,
            ),
        )

    schedule_result = (
        page_offsets_schedule(smem_page_offsets_k, smem_page_offsets_v)
        if work_queue is None
        else page_offsets_schedule(smem_page_offsets_k, smem_page_offsets_v, work_queue)
    )
    src = []
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[smem_page_offsets_k, smem_page_offsets_v],
        q_bound_resources=(
            (smem_page_offsets_k, False),
            (smem_page_offsets_v, False),
        ),
        cfg=cfg,
        warp_idx=cfg.page_offsets_warp_idx if warp_idx is None else warp_idx,
        num_warps=cfg.page_offsets_num_warps if num_warps is None else num_warps,
        schedule=schedule_result,
        num_registers=cfg.mma_load_task_num_registers,
        name="PageTableTask",
        **kw,
    )


def create_page_offsets_task_one_inst_qkv(
    smem_page_offsets: MemoryResource,
    work_queue: WorkQueue | None,
    cfg: FmhaDecodeConfig,
    *,
    domain: int | cutlass.Int32,
    warp_idx: int | None = None,
    num_warps: int | None = None,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> Task:
    """Prefetch page-table entries for the one-inst keepsMmaAb QKV path."""

    def page_offsets_schedule_body(
        smem_page_offsets: MemoryResource,
    ) -> None:
        """Schedule page-offset prefetches for the one-inst QKV load cadence."""
        smem_page_offsets.init_load_state()

        _page_offsets_produce(smem_page_offsets, "load_k0", FmhaStage.Head)
        with domain_loop(0, domain, 1, unroll=1):
            for label in ("load_k0", "load_v0"):
                _page_offsets_produce(smem_page_offsets, label, FmhaStage.Loop)
        _page_offsets_produce(smem_page_offsets, "load_v0", FmhaStage.Tail)

    @schedule
    def page_offsets_schedule(
        smem_page_offsets: MemoryResource,
        work_queue: WorkQueue | None = None,
    ) -> None:
        """Wrap one-inst page-offset work in packed persistent skip handling."""
        _decode_work_tile_schedule(
            cfg,
            work_queue,
            lambda: page_offsets_schedule_body(smem_page_offsets),
        )

    schedule_result = (
        page_offsets_schedule(smem_page_offsets)
        if work_queue is None
        else page_offsets_schedule(smem_page_offsets, work_queue)
    )
    src = []
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[smem_page_offsets],
        q_bound_resources=((smem_page_offsets, False),),
        cfg=cfg,
        warp_idx=cfg.page_offsets_warp_idx if warp_idx is None else warp_idx,
        num_warps=cfg.page_offsets_num_warps if num_warps is None else num_warps,
        schedule=schedule_result,
        num_registers=cfg.mma_load_task_num_registers,
        name="PageTableTask",
        **kw,
    )


def create_load_task_split_kv(
    smem_q: MemoryResource | None,
    smem_k0: MemoryResource | None,
    smem_k1: MemoryResource | None,
    smem_v0: MemoryResource | None,
    smem_v1: MemoryResource | None,
    work_queue: WorkQueue | None,
    schedule_token_throttle: MemoryResource | None,
    cfg: FmhaDecodeConfig,
    *,
    domain: int | cutlass.Int32,
    smem_page_offsets: MemoryResource | None = None,
    smem_page_offsets_v: MemoryResource | None = None,
    sparse_kv_metadata0: MemoryResource | None = None,
    sparse_kv_metadata1: MemoryResource | None = None,
    sparse_softmax_metadata0: MemoryResource | None = None,
    sparse_softmax_metadata1: MemoryResource | None = None,
    warp_idx: int | None = None,
    num_warps: int | None = None,
    task_name: str = "LoadTask",
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> Task:
    """Create loads for independent K0/K1/V0/V1 resources.

    Here ``split`` describes those resources, not split-KV reduction.
    """
    if schedule_token_throttle is not None and work_queue is None:
        raise ValueError("schedule-token throttle requires a work queue")
    if smem_page_offsets_v is not None and smem_page_offsets is None:
        raise ValueError("V page offsets require K page offsets")
    if (smem_k0 is None) != (smem_v0 is None) or (smem_k1 is None) != (smem_v1 is None):
        raise ValueError("each split K resource requires its matching V resource")
    if smem_k0 is None and smem_k1 is None:
        raise ValueError("at least one split K/V instance is required")

    def load_schedule_body(
        smem_q: MemoryResource | None,
        smem_k0: MemoryResource | None,
        smem_k1: MemoryResource | None,
        smem_v0: MemoryResource | None,
        smem_v1: MemoryResource | None,
        sparse_kv_metadata0: MemoryResource | None,
        sparse_kv_metadata1: MemoryResource | None,
        sparse_softmax_metadata0: MemoryResource | None,
        sparse_softmax_metadata1: MemoryResource | None,
        smem_page_offsets: MemoryResource | None,
        smem_page_offsets_v: MemoryResource | None = None,
        schedule_token_throttle: MemoryResource | None = None,
    ) -> None:
        """Build the split-resource K/V load cadence for all schedule phases."""
        if smem_q is not None:
            smem_q.init_load_state()
        active_instances = (
            (
                smem_k0,
                smem_v0,
                sparse_kv_metadata0,
                sparse_softmax_metadata0,
                "load_k0",
                "load_v0",
            ),
            (
                smem_k1,
                smem_v1,
                sparse_kv_metadata1,
                sparse_softmax_metadata1,
                "load_k1",
                "load_v1",
            ),
        )
        # Preserve the original full-resource lowering order while allowing a
        # per-instance task to omit the other stream's resources.
        for smem_k, _, _, _, _, _ in active_instances:
            if smem_k is not None:
                smem_k.init_load_state()
        for _, smem_v, _, _, _, _ in active_instances:
            if smem_v is not None:
                smem_v.init_load_state()
        for _, _, sparse_kv_metadata, _, _, _ in active_instances:
            if sparse_kv_metadata is not None:
                sparse_kv_metadata.init_load_state()
        for _, _, _, sparse_softmax_metadata, _, _ in active_instances:
            if sparse_softmax_metadata is not None:
                sparse_softmax_metadata.init_load_state()
        if smem_page_offsets is not None:
            smem_page_offsets.init_read_state()
        if smem_page_offsets_v is not None:
            smem_page_offsets_v.init_read_state()

        smem_page_offsets_k = smem_page_offsets
        smem_page_offsets_v_local = (
            smem_page_offsets_v
            if smem_page_offsets_v is not None
            else smem_page_offsets
        )

        def load_tile(
            resource: MemoryResource,
            label: str,
            offsets: MemoryResource | None,
            section: FmhaStage,
        ) -> None:
            """Acquire, load all head-dim stages, and release page offsets."""
            _page_offsets_consume(offsets, label.replace("load", "read_offsets"))
            for head_dim_stage_idx in range(cfg.num_head_dim_stages_kv):
                resource.acquire()
                getattr(resource, label)(
                    section=section, head_dim_stage_idx=head_dim_stage_idx
                )
                resource.commit()
            _page_offsets_release(offsets)

        if smem_q is not None:
            smem_q.acquire()
            smem_q.tma_load()
            smem_q.commit()

        head_routes = []
        for (
            smem_k,
            _,
            sparse_kv_metadata,
            sparse_softmax_metadata,
            load_k,
            _,
        ) in active_instances:
            if smem_k is None:
                continue
            route = _resolve_and_store_sparse_route(sparse_kv_metadata, FmhaStage.Head)
            load_tile(smem_k, load_k, smem_page_offsets_k, FmhaStage.Head)
            head_routes.append((sparse_softmax_metadata, route))
        # In the combined task, preserve both K issues ahead of Softmax
        # backpressure. A per-instance task naturally stages its sole route.
        for sparse_softmax_metadata, route in head_routes:
            _publish_sparse_softmax_route(sparse_softmax_metadata, route)

        with domain_loop(0, domain, 1, unroll=1):
            # V consumes the retained route before the next K overwrites it.
            loop_routes = []
            for (
                smem_k,
                smem_v,
                sparse_kv_metadata,
                sparse_softmax_metadata,
                load_k,
                load_v,
            ) in active_instances:
                if smem_k is None:
                    continue
                assert smem_v is not None
                load_tile(
                    smem_v,
                    load_v,
                    smem_page_offsets_v_local,
                    FmhaStage.Loop,
                )
                route = _resolve_and_store_sparse_route(
                    sparse_kv_metadata, FmhaStage.Loop
                )
                load_tile(smem_k, load_k, smem_page_offsets_k, FmhaStage.Loop)
                loop_routes.append((sparse_softmax_metadata, route))
            for sparse_softmax_metadata, route in loop_routes:
                _publish_sparse_softmax_route(sparse_softmax_metadata, route)

        for _, smem_v, _, _, _, load_v in active_instances:
            if smem_v is not None:
                load_tile(
                    smem_v,
                    load_v,
                    smem_page_offsets_v_local,
                    FmhaStage.Tail,
                )

    @_schedule_with_optional_resources
    def load_schedule(
        smem_q: MemoryResource | None,
        smem_k0: MemoryResource | None,
        smem_k1: MemoryResource | None,
        smem_v0: MemoryResource | None,
        smem_v1: MemoryResource | None,
        sparse_kv_metadata0: MemoryResource | None,
        sparse_kv_metadata1: MemoryResource | None,
        sparse_softmax_metadata0: MemoryResource | None,
        sparse_softmax_metadata1: MemoryResource | None,
        smem_page_offsets: MemoryResource | None,
        smem_page_offsets_v: MemoryResource | None,
        work_queue: WorkQueue | None,
        schedule_token_throttle: MemoryResource | None,
    ) -> None:
        """Schedule split K/V loads with only the supplied optional resources."""
        _decode_work_tile_schedule(
            cfg,
            work_queue,
            lambda: load_schedule_body(
                smem_q,
                smem_k0,
                smem_k1,
                smem_v0,
                smem_v1,
                sparse_kv_metadata0,
                sparse_kv_metadata1,
                sparse_softmax_metadata0,
                sparse_softmax_metadata1,
                smem_page_offsets,
                smem_page_offsets_v,
                schedule_token_throttle,
            ),
            lambda: _schedule_token_throttle_head(schedule_token_throttle),
        )

    for smem_k, sparse_kv_metadata, sparse_softmax_metadata in (
        (smem_k0, sparse_kv_metadata0, sparse_softmax_metadata0),
        (smem_k1, sparse_kv_metadata1, sparse_softmax_metadata1),
    ):
        if smem_k is None and (
            sparse_kv_metadata is not None or sparse_softmax_metadata is not None
        ):
            raise ValueError("inactive K/V instances cannot own sparse metadata")
        if sparse_softmax_metadata is not None and sparse_kv_metadata is None:
            raise ValueError("Softmax sparse metadata requires retained KV metadata")
    if sparse_kv_metadata0 is not None or sparse_kv_metadata1 is not None:
        if smem_page_offsets is not None or smem_page_offsets_v is not None:
            raise ValueError(
                "block-sparse and paged-KV load resources cannot be combined"
            )

    schedule_result = load_schedule(
        smem_q,
        smem_k0,
        smem_k1,
        smem_v0,
        smem_v1,
        sparse_kv_metadata0,
        sparse_kv_metadata1,
        sparse_softmax_metadata0,
        sparse_softmax_metadata1,
        smem_page_offsets,
        smem_page_offsets_v,
        work_queue,
        schedule_token_throttle,
    )
    src = []
    # Route resolution is ConsumerWork and K/V retention is ProducerWork on
    # the same pipeline-free resource, so register both sides of that route.
    for sparse_kv_metadata in (sparse_kv_metadata0, sparse_kv_metadata1):
        if sparse_kv_metadata is not None:
            src.append(sparse_kv_metadata)
    if smem_page_offsets is not None:
        src.append(smem_page_offsets)
    if smem_page_offsets_v is not None:
        src.append(smem_page_offsets_v)
    if work_queue is not None:
        src.append(work_queue)
    dst = [
        resource
        for resource in (smem_q, smem_k0, smem_k1, smem_v0, smem_v1)
        if resource is not None
    ]
    for sparse_resource in (
        sparse_kv_metadata0,
        sparse_kv_metadata1,
        sparse_softmax_metadata0,
        sparse_softmax_metadata1,
    ):
        if sparse_resource is not None:
            dst.append(sparse_resource)
    if schedule_token_throttle is not None:
        dst.append(schedule_token_throttle)
    q_bound_resources = tuple(
        (resource, resource is smem_q)
        for resource in (smem_q, smem_k0, smem_k1, smem_v0, smem_v1)
        if resource is not None
    )
    return task_class(
        src_resources=src,
        dst_resources=dst,
        q_bound_resources=q_bound_resources,
        cfg=cfg,
        warp_idx=cfg.load_warp_idx if warp_idx is None else warp_idx,
        num_warps=cfg.load_num_warps if num_warps is None else num_warps,
        schedule=schedule_result,
        num_registers=cfg.mma_load_task_num_registers,
        name=task_name,
        **kw,
    )


def create_block_sparse_load_tasks_per_inst(
    smem_q: MemoryResource,
    smem_k0: MemoryResource,
    smem_k1: MemoryResource,
    smem_v0: MemoryResource,
    smem_v1: MemoryResource,
    work_queue: WorkQueue | None,
    schedule_token_throttle: MemoryResource | None,
    cfg: FmhaDecodeConfig,
    *,
    domain: int | cutlass.Int32,
    sparse_kv_metadata0: MemoryResource,
    sparse_kv_metadata1: MemoryResource,
    sparse_softmax_metadata0: MemoryResource,
    sparse_softmax_metadata1: MemoryResource,
    warp_indices: tuple[int, int],
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> tuple[Task, Task]:
    """Assign each sparse K/V instruction stream to an independent load warp.

    Load0 alone owns Q and the persistent schedule-token throttle. Both tasks
    consume the same logical work tile, while their K/V and sparse-metadata
    pipelines remain disjoint.
    """

    if not cfg.use_block_sparse:
        raise ValueError("per-instance load tasks require block-sparse metadata")

    load0 = create_load_task_split_kv(
        smem_q,
        smem_k0,
        None,
        smem_v0,
        None,
        work_queue,
        schedule_token_throttle,
        cfg,
        domain=domain,
        sparse_kv_metadata0=sparse_kv_metadata0,
        sparse_softmax_metadata0=sparse_softmax_metadata0,
        warp_idx=warp_indices[0],
        task_name="LoadTask0",
        task_class=task_class,
        **kw,
    )
    load1 = create_load_task_split_kv(
        None,
        None,
        smem_k1,
        None,
        smem_v1,
        work_queue,
        None,
        cfg,
        domain=domain,
        sparse_kv_metadata1=sparse_kv_metadata1,
        sparse_softmax_metadata1=sparse_softmax_metadata1,
        warp_idx=warp_indices[1],
        task_name="LoadTask1",
        task_class=task_class,
        **kw,
    )
    return load0, load1


def create_load_task_one_inst_qkv(
    smem_q: MemoryResource,
    smem_k: MemoryResource,
    smem_v: MemoryResource,
    work_queue: WorkQueue | None,
    schedule_token_throttle: MemoryResource | None,
    cfg: FmhaDecodeConfig,
    *,
    domain: int | cutlass.Int32,
    smem_page_offsets: MemoryResource | None = None,
    warp_idx: int | None = None,
    num_warps: int | None = None,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> Task:
    """Load schedule for the one-inst keepsMmaAb QKV path."""

    def load_schedule_body(
        smem_q: MemoryResource,
        smem_k: MemoryResource,
        smem_v: MemoryResource,
        smem_page_offsets: MemoryResource | None,
        schedule_token_throttle: MemoryResource | None,
    ) -> None:
        """Build the one-inst Q/K/V load cadence."""
        smem_q.init_load_state()
        smem_k.init_load_state()
        smem_v.init_load_state()
        if smem_page_offsets is not None:
            smem_page_offsets.init_read_state()

        def load_tile(resource: MemoryResource, label: str, section: FmhaStage) -> None:
            """Acquire, load all head-dim stages, and release one page window."""
            _page_offsets_consume(
                smem_page_offsets, label.replace("load", "read_offsets")
            )
            for head_dim_stage_idx in range(cfg.num_head_dim_stages_kv):
                resource.acquire()
                getattr(resource, label)(
                    section=section, head_dim_stage_idx=head_dim_stage_idx
                )
                resource.commit()
            _page_offsets_release(smem_page_offsets)

        smem_q.acquire()
        smem_q.tma_load()
        smem_q.commit()
        load_tile(smem_k, "load_k0", FmhaStage.Head)

        with domain_loop(0, domain, 1, unroll=1):
            load_tile(smem_k, "load_k0", FmhaStage.Loop)
            load_tile(smem_v, "load_v0", FmhaStage.Loop)

        load_tile(smem_v, "load_v0", FmhaStage.Tail)

    @schedule
    def load_schedule(
        smem_q: MemoryResource,
        smem_k: MemoryResource,
        smem_v: MemoryResource,
        work_queue: WorkQueue | None = None,
        schedule_token_throttle: MemoryResource | None = None,
    ) -> None:
        """Schedule one-inst Q/K/V loads without page-offset resources."""
        _decode_work_tile_schedule(
            cfg,
            work_queue,
            lambda: load_schedule_body(
                smem_q, smem_k, smem_v, None, schedule_token_throttle
            ),
            lambda: _schedule_token_throttle_head(schedule_token_throttle),
        )

    @schedule
    def load_page_offsets_schedule(
        smem_q: MemoryResource,
        smem_k: MemoryResource,
        smem_v: MemoryResource,
        smem_page_offsets: MemoryResource,
        work_queue: WorkQueue | None = None,
        schedule_token_throttle: MemoryResource | None = None,
    ) -> None:
        """Schedule one-inst Q/K/V loads with paged-KV offsets."""
        _decode_work_tile_schedule(
            cfg,
            work_queue,
            lambda: load_schedule_body(
                smem_q,
                smem_k,
                smem_v,
                smem_page_offsets,
                schedule_token_throttle,
            ),
            lambda: _schedule_token_throttle_head(schedule_token_throttle),
        )

    if smem_page_offsets is None:
        if work_queue is None:
            schedule_result = load_schedule(smem_q, smem_k, smem_v)
        elif schedule_token_throttle is None:
            schedule_result = load_schedule(smem_q, smem_k, smem_v, work_queue)
        else:
            schedule_result = load_schedule(
                smem_q, smem_k, smem_v, work_queue, schedule_token_throttle
            )
    else:
        if work_queue is None:
            schedule_result = load_page_offsets_schedule(
                smem_q, smem_k, smem_v, smem_page_offsets
            )
        elif schedule_token_throttle is None:
            schedule_result = load_page_offsets_schedule(
                smem_q,
                smem_k,
                smem_v,
                smem_page_offsets,
                work_queue,
            )
        else:
            schedule_result = load_page_offsets_schedule(
                smem_q,
                smem_k,
                smem_v,
                smem_page_offsets,
                work_queue,
                schedule_token_throttle,
            )
    src = []
    if smem_page_offsets is not None:
        src.append(smem_page_offsets)
    if work_queue is not None:
        src.append(work_queue)
    dst = [smem_q, smem_k, smem_v]
    if schedule_token_throttle is not None:
        dst.append(schedule_token_throttle)
    return task_class(
        src_resources=src,
        dst_resources=dst,
        q_bound_resources=((smem_q, True), (smem_k, False), (smem_v, False)),
        cfg=cfg,
        warp_idx=cfg.load_warp_idx if warp_idx is None else warp_idx,
        num_warps=cfg.load_num_warps if num_warps is None else num_warps,
        schedule=schedule_result,
        num_registers=cfg.mma_load_task_num_registers,
        name="LoadTask",
        **kw,
    )


def create_mma_task_split_kv(
    smem_q: MemoryResource,
    smem_k0: MemoryResource,
    smem_k1: MemoryResource,
    smem_v0: MemoryResource,
    smem_v1: MemoryResource,
    tmem_s0: MemoryResource,
    tmem_s1: MemoryResource,
    smem_p0: MemoryResource,
    smem_p1: MemoryResource,
    tmem_o: MemoryResource,
    work_queue: WorkQueue | None,
    cfg: FmhaDecodeConfig,
    *,
    domain: int | cutlass.Int32,
    tmem_stats_done0: MemoryResource | None = None,
    tmem_stats_done1: MemoryResource | None = None,
    warp_idx: int | None = None,
    num_warps: int | None = None,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> Task:
    """Create the MMA task for split K/V resources."""

    def mma_schedule_body(
        smem_q: MemoryResource,
        smem_k0: MemoryResource,
        smem_k1: MemoryResource,
        smem_v0: MemoryResource,
        smem_v1: MemoryResource,
        tmem_s0: MemoryResource,
        tmem_s1: MemoryResource,
        smem_p0: MemoryResource,
        smem_p1: MemoryResource,
        tmem_o: MemoryResource,
        tmem_stats_done0: MemoryResource | None,
        tmem_stats_done1: MemoryResource | None,
        q_desc: Any,
    ) -> None:
        """Schedule split-resource QK and PV waves across HEAD/LOOP/TAIL."""

        def qk_mma(
            smem_kv: MemoryResource,
            tmem_s: MemoryResource,
            tmem_stats_done: MemoryResource | None,
            q_desc,
            qk_mma_label: str,
            section: FmhaStage,
        ) -> None:
            """Issue one scheduled QK wave using the selected phase work."""
            _ = section
            if tmem_stats_done is not None:
                tmem_stats_done.acquire()
            tmem_s.acquire()
            for head_dim_stage_idx in range(cfg.num_head_dim_stages_kv):
                smem_kv.wait()
                kv_desc = smem_kv.kv_desc()
                if cfg.uses_q_desc_ref:
                    getattr(tmem_s, f"{qk_mma_label}_from_q_ref")(
                        kv_desc=kv_desc,
                        head_dim_stage_idx=head_dim_stage_idx,
                    )
                else:
                    getattr(tmem_s, qk_mma_label)(
                        q_desc=q_desc,
                        kv_desc=kv_desc,
                        head_dim_stage_idx=head_dim_stage_idx,
                    )
                smem_kv.release()
            tmem_s.commit()
            if tmem_stats_done is not None:
                tmem_stats_done.commit()

        def pv_mma(
            smem_kv: MemoryResource,
            tmem_p: MemoryResource,
            vp_mma_label: str,
            inst_idx: int,
            section: FmhaStage,
        ) -> None:
            """Issue one scheduled PV wave using the selected phase work."""
            _ = section
            tmem_p.wait()
            p_desc_0, p_desc_1, p_tmem_addr_0, p_tmem_addr_1 = tmem_p.p_operands()
            tmem_o.acquire()
            for head_dim_stage_idx in range(cfg.num_head_dim_stages_kv):
                smem_kv.wait()
                v_desc = smem_kv.v_desc()
                getattr(tmem_o, vp_mma_label)(
                    v_desc_0=v_desc,
                    v_desc_1=v_desc,
                    p_desc_0=p_desc_0,
                    p_desc_1=p_desc_1,
                    p_tmem_addr_0=p_tmem_addr_0,
                    p_tmem_addr_1=p_tmem_addr_1,
                    inst_idx=inst_idx,
                    head_dim_stage_idx=head_dim_stage_idx,
                )
                smem_kv.release()
            tmem_o.commit()
            tmem_p.release()

        qk_mma(
            smem_k0,
            tmem_s0,
            tmem_stats_done0,
            q_desc,
            "qk_mma_head",
            FmhaStage.Head,
        )
        qk_mma(
            smem_k1,
            tmem_s1,
            tmem_stats_done1,
            q_desc,
            "qk_mma_head",
            FmhaStage.Head,
        )

        with domain_loop(0, domain, 1, unroll=1):
            pv_mma(smem_v0, smem_p0, "vp_mma_loop", KV_INST0, FmhaStage.Loop)
            qk_mma(
                smem_k0,
                tmem_s0,
                tmem_stats_done0,
                q_desc,
                "qk_mma_loop",
                FmhaStage.Loop,
            )
            pv_mma(smem_v1, smem_p1, "vp_mma_loop", KV_INST1, FmhaStage.Loop)
            qk_mma(
                smem_k1,
                tmem_s1,
                tmem_stats_done1,
                q_desc,
                "qk_mma_loop",
                FmhaStage.Loop,
            )

        pv_mma(smem_v0, smem_p0, "vp_mma_tail", KV_INST0, FmhaStage.Tail)
        pv_mma(smem_v1, smem_p1, "vp_mma_tail", KV_INST1, FmhaStage.Tail)
        smem_q.release()

    def mma_schedule_prelude(
        smem_q: MemoryResource,
        smem_k0: MemoryResource,
        smem_k1: MemoryResource,
        smem_v0: MemoryResource,
        smem_v1: MemoryResource,
        smem_p0: MemoryResource,
        smem_p1: MemoryResource,
    ) -> None:
        """Initialize invariant split-resource descriptor slots."""
        smem_q.init_descriptor_state()
        smem_k0.init_descriptor_state()
        smem_k1.init_descriptor_state()
        smem_v0.init_descriptor_state()
        smem_v1.init_descriptor_state()
        smem_p0.init_descriptor_state()
        smem_p1.init_descriptor_state()

    def run_mma_schedule(
        smem_q: MemoryResource,
        smem_k0: MemoryResource,
        smem_k1: MemoryResource,
        smem_v0: MemoryResource,
        smem_v1: MemoryResource,
        tmem_s0: MemoryResource,
        tmem_s1: MemoryResource,
        smem_p0: MemoryResource,
        smem_p1: MemoryResource,
        tmem_o: MemoryResource,
        tmem_stats_done0: MemoryResource | None,
        tmem_stats_done1: MemoryResource | None,
        work_queue: WorkQueue | None,
    ) -> None:
        """Wrap split-resource MMA work with optional stats lifetime gates."""
        _decode_work_tile_schedule_with_invariant_bridge(
            cfg,
            work_queue,
            lambda: mma_schedule_prelude(
                smem_q, smem_k0, smem_k1, smem_v0, smem_v1, smem_p0, smem_p1
            ),
            lambda: smem_q.q_desc(),
            lambda: smem_q.wait(),
            lambda q_desc: mma_schedule_body(
                smem_q,
                smem_k0,
                smem_k1,
                smem_v0,
                smem_v1,
                tmem_s0,
                tmem_s1,
                smem_p0,
                smem_p1,
                tmem_o,
                tmem_stats_done0,
                tmem_stats_done1,
                q_desc,
            ),
        )

    @schedule
    def mma_schedule(
        smem_q: MemoryResource,
        smem_k0: MemoryResource,
        smem_k1: MemoryResource,
        smem_v0: MemoryResource,
        smem_v1: MemoryResource,
        tmem_s0: MemoryResource,
        tmem_s1: MemoryResource,
        smem_p0: MemoryResource,
        smem_p1: MemoryResource,
        tmem_o: MemoryResource,
        work_queue: WorkQueue | None = None,
    ) -> None:
        """Capture the Swaps split-resource MMA schedule."""
        run_mma_schedule(
            smem_q,
            smem_k0,
            smem_k1,
            smem_v0,
            smem_v1,
            tmem_s0,
            tmem_s1,
            smem_p0,
            smem_p1,
            tmem_o,
            None,
            None,
            work_queue,
        )

    @schedule
    def mma_keeps_schedule(
        smem_q: MemoryResource,
        smem_k0: MemoryResource,
        smem_k1: MemoryResource,
        smem_v0: MemoryResource,
        smem_v1: MemoryResource,
        tmem_s0: MemoryResource,
        tmem_s1: MemoryResource,
        smem_p0: MemoryResource,
        smem_p1: MemoryResource,
        tmem_o: MemoryResource,
        tmem_stats_done0: MemoryResource,
        tmem_stats_done1: MemoryResource,
        work_queue: WorkQueue | None = None,
    ) -> None:
        """Capture Keeps MMA with explicit stats lifetime gates."""
        run_mma_schedule(
            smem_q,
            smem_k0,
            smem_k1,
            smem_v0,
            smem_v1,
            tmem_s0,
            tmem_s1,
            smem_p0,
            smem_p1,
            tmem_o,
            tmem_stats_done0,
            tmem_stats_done1,
            work_queue,
        )

    if tmem_stats_done0 is None or tmem_stats_done1 is None:
        schedule_result = (
            mma_schedule(
                smem_q,
                smem_k0,
                smem_k1,
                smem_v0,
                smem_v1,
                tmem_s0,
                tmem_s1,
                smem_p0,
                smem_p1,
                tmem_o,
            )
            if work_queue is None
            else mma_schedule(
                smem_q,
                smem_k0,
                smem_k1,
                smem_v0,
                smem_v1,
                tmem_s0,
                tmem_s1,
                smem_p0,
                smem_p1,
                tmem_o,
                work_queue,
            )
        )
        dst = [tmem_s0, tmem_s1, tmem_o]
    else:
        schedule_result = (
            mma_keeps_schedule(
                smem_q,
                smem_k0,
                smem_k1,
                smem_v0,
                smem_v1,
                tmem_s0,
                tmem_s1,
                smem_p0,
                smem_p1,
                tmem_o,
                tmem_stats_done0,
                tmem_stats_done1,
            )
            if work_queue is None
            else mma_keeps_schedule(
                smem_q,
                smem_k0,
                smem_k1,
                smem_v0,
                smem_v1,
                tmem_s0,
                tmem_s1,
                smem_p0,
                smem_p1,
                tmem_o,
                tmem_stats_done0,
                tmem_stats_done1,
                work_queue,
            )
        )
        dst = [
            tmem_s0,
            tmem_s1,
            tmem_o,
            tmem_stats_done0,
            tmem_stats_done1,
        ]
    src = [smem_q, smem_k0, smem_k1, smem_v0, smem_v1, smem_p0, smem_p1]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=dst,
        cfg=cfg,
        warp_idx=cfg.mma_warp_idx if warp_idx is None else warp_idx,
        num_warps=cfg.mma_num_warps if num_warps is None else num_warps,
        schedule=schedule_result,
        num_registers=cfg.mma_load_task_num_registers,
        name="MmaTask",
        **kw,
    )


def create_mma_task_one_inst_qkv(
    smem_q: MemoryResource,
    smem_k: MemoryResource,
    smem_v: MemoryResource,
    tmem_s: MemoryResource,
    smem_p: MemoryResource,
    tmem_o: MemoryResource,
    work_queue: WorkQueue | None,
    cfg: FmhaDecodeConfig,
    *,
    tmem_stats_done: MemoryResource,
    domain: int | cutlass.Int32,
    warp_idx: int | None = None,
    num_warps: int | None = None,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> Task:
    """Create the one-inst QKV MMA task."""

    def mma_schedule_body(
        smem_q: MemoryResource,
        smem_k: MemoryResource,
        smem_v: MemoryResource,
        tmem_s: MemoryResource,
        smem_p: MemoryResource,
        tmem_o: MemoryResource,
        tmem_stats_done: MemoryResource,
        q_desc: Any,
    ) -> None:
        """Schedule single-instance QK and PV waves across HEAD/LOOP/TAIL."""

        def qk_mma(q_desc, qk_mma_label: str, section: FmhaStage) -> None:
            """Issue one scheduled single-instance QK wave."""
            _ = section
            tmem_stats_done.acquire()
            tmem_s.acquire()
            for head_dim_stage_idx in range(cfg.num_head_dim_stages_kv):
                smem_k.wait()
                kv_desc = smem_k.kv_desc()
                if cfg.uses_q_desc_ref:
                    getattr(tmem_s, f"{qk_mma_label}_from_q_ref")(
                        kv_desc=kv_desc,
                        head_dim_stage_idx=head_dim_stage_idx,
                    )
                else:
                    getattr(tmem_s, qk_mma_label)(
                        q_desc=q_desc,
                        kv_desc=kv_desc,
                        head_dim_stage_idx=head_dim_stage_idx,
                    )
                smem_k.release()
            tmem_s.commit()
            tmem_stats_done.commit()

        def pv_mma(vp_mma_label: str, section: FmhaStage) -> None:
            """Issue one scheduled single-instance PV wave."""
            _ = section
            smem_p.wait()
            p_desc_0, p_desc_1, p_tmem_addr_0, p_tmem_addr_1 = smem_p.p_operands()
            tmem_o.acquire()
            for head_dim_stage_idx in range(cfg.num_head_dim_stages_kv):
                smem_v.wait()
                v_desc = smem_v.v_desc()
                getattr(tmem_o, vp_mma_label)(
                    v_desc_0=v_desc,
                    v_desc_1=v_desc,
                    p_desc_0=p_desc_0,
                    p_desc_1=p_desc_1,
                    p_tmem_addr_0=p_tmem_addr_0,
                    p_tmem_addr_1=p_tmem_addr_1,
                    inst_idx=KV_INST0,
                    head_dim_stage_idx=head_dim_stage_idx,
                )
                smem_v.release()
            tmem_o.commit()
            smem_p.release()

        qk_mma(q_desc, "qk_mma_head", FmhaStage.Head)

        with domain_loop(0, domain, 1, unroll=1):
            qk_mma(q_desc, "qk_mma_loop", FmhaStage.Loop)
            pv_mma("vp_mma_loop", FmhaStage.Loop)

        pv_mma("vp_mma_tail", FmhaStage.Tail)
        smem_q.release()

    def mma_schedule_prelude(
        smem_q: MemoryResource,
        smem_k: MemoryResource,
        smem_v: MemoryResource,
        smem_p: MemoryResource,
    ) -> None:
        """Initialize invariant one-inst descriptor slots."""
        smem_q.init_descriptor_state()
        smem_k.init_descriptor_state()
        smem_v.init_descriptor_state()
        smem_p.init_descriptor_state()

    @schedule
    def mma_schedule(
        smem_q: MemoryResource,
        smem_k: MemoryResource,
        smem_v: MemoryResource,
        tmem_s: MemoryResource,
        smem_p: MemoryResource,
        tmem_o: MemoryResource,
        tmem_stats_done: MemoryResource,
        work_queue: WorkQueue | None = None,
    ) -> None:
        """Wrap one-inst MMA work in packed persistent skip handling."""
        _decode_work_tile_schedule_with_invariant_bridge(
            cfg,
            work_queue,
            lambda: mma_schedule_prelude(smem_q, smem_k, smem_v, smem_p),
            lambda: smem_q.q_desc(),
            lambda: smem_q.wait(),
            lambda q_desc: mma_schedule_body(
                smem_q,
                smem_k,
                smem_v,
                tmem_s,
                smem_p,
                tmem_o,
                tmem_stats_done,
                q_desc,
            ),
        )

    schedule_result = (
        mma_schedule(
            smem_q,
            smem_k,
            smem_v,
            tmem_s,
            smem_p,
            tmem_o,
            tmem_stats_done,
        )
        if work_queue is None
        else mma_schedule(
            smem_q,
            smem_k,
            smem_v,
            tmem_s,
            smem_p,
            tmem_o,
            tmem_stats_done,
            work_queue,
        )
    )
    src = [smem_q, smem_k, smem_v, smem_p]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[tmem_s, tmem_o, tmem_stats_done],
        cfg=cfg,
        warp_idx=cfg.mma_warp_idx if warp_idx is None else warp_idx,
        num_warps=cfg.mma_num_warps if num_warps is None else num_warps,
        schedule=schedule_result,
        num_registers=cfg.mma_load_task_num_registers,
        name="MmaTask",
        **kw,
    )


# ======================================================================
# MmaTask — warp 12, 1 warp
# K and V share a single SmemKv ring; each MMA loop iter consumes 4 stages
# (K0, V0, K1, V1) of the shared buffer.
#   HEAD: wait Q, BMM1(K0), BMM1(K1)
#   LOOP[i]: BMM1(nextK0), BMM2(currV0), BMM1(nextK1), BMM2(currV1)
#   TAIL: final BMM2(lastV0), final BMM2(lastV1), release Q
# ======================================================================
def create_mma_task(
    smem_q: MemoryResource,
    smem_kv: MemoryResource,
    tmem_s0: MemoryResource,
    tmem_s1: MemoryResource,
    smem_p0: MemoryResource,
    smem_p1: MemoryResource,
    tmem_o: MemoryResource,
    work_queue: WorkQueue | None,
    cfg: FmhaDecodeConfig,
    *,
    domain: int | cutlass.Int32,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> Task:
    """Create the MMA task for the shared K/V ring path."""

    def mma_schedule_body(
        smem_q: MemoryResource,
        smem_kv: MemoryResource,
        tmem_s0: MemoryResource,
        tmem_s1: MemoryResource,
        smem_p0: MemoryResource,
        smem_p1: MemoryResource,
        tmem_o: MemoryResource,
        q_desc: Any,
    ) -> None:
        """Schedule shared-KV QK and PV waves across HEAD/LOOP/TAIL."""
        # HEAD: consume Q once and launch the two initial BMM1 waves.
        _consume_staged_qk_mma(
            smem_kv,
            tmem_s0,
            smem_p0,
            q_desc,
            "k_desc_0",
            "qk_mma_head",
            FmhaStage.Head,
            cfg,
        )
        _consume_staged_qk_mma(
            smem_kv,
            tmem_s1,
            smem_p1,
            q_desc,
            "k_desc_1",
            "qk_mma_head",
            FmhaStage.Head,
            cfg,
        )

        # LOOP: consume aliased TMEM P before the next same-instance QK
        # overwrites its S columns. SMEM-P profiles retain their established
        # K-before-V cadence because P no longer depends on S lifetime.
        with domain_loop(0, domain, 1, unroll=1):
            if cfg.uses_two_inst_tmem_p:
                _consume_staged_pv_mma(
                    smem_kv,
                    smem_p0,
                    tmem_o,
                    "v_desc_0",
                    "vp_mma_loop",
                    KV_INST0,
                    FmhaStage.Loop,
                    cfg,
                )
            _consume_staged_qk_mma(
                smem_kv,
                tmem_s0,
                smem_p0,
                q_desc,
                "k_desc_0",
                "qk_mma_loop",
                FmhaStage.Loop,
                cfg,
            )
            if not cfg.uses_two_inst_tmem_p:
                _consume_staged_pv_mma(
                    smem_kv,
                    smem_p0,
                    tmem_o,
                    "v_desc_0",
                    "vp_mma_loop",
                    KV_INST0,
                    FmhaStage.Loop,
                    cfg,
                )
            if cfg.uses_two_inst_tmem_p:
                _consume_staged_pv_mma(
                    smem_kv,
                    smem_p1,
                    tmem_o,
                    "v_desc_1",
                    "vp_mma_loop",
                    KV_INST1,
                    FmhaStage.Loop,
                    cfg,
                )
            _consume_staged_qk_mma(
                smem_kv,
                tmem_s1,
                smem_p1,
                q_desc,
                "k_desc_1",
                "qk_mma_loop",
                FmhaStage.Loop,
                cfg,
            )
            if not cfg.uses_two_inst_tmem_p:
                _consume_staged_pv_mma(
                    smem_kv,
                    smem_p1,
                    tmem_o,
                    "v_desc_1",
                    "vp_mma_loop",
                    KV_INST1,
                    FmhaStage.Loop,
                    cfg,
                )

        # TAIL: no future K tiles remain, so only the final two BMM2 waves run.
        _consume_staged_pv_mma(
            smem_kv,
            smem_p0,
            tmem_o,
            "v_desc_0",
            "vp_mma_tail",
            KV_INST0,
            FmhaStage.Tail,
            cfg,
        )
        _consume_staged_pv_mma(
            smem_kv,
            smem_p1,
            tmem_o,
            "v_desc_1",
            "vp_mma_tail",
            KV_INST1,
            FmhaStage.Tail,
            cfg,
        )
        # Q is live for every BMM1 call and can be released only after the loop.
        smem_q.release()

    def mma_schedule_prelude(
        smem_q: MemoryResource,
        smem_kv: MemoryResource,
        smem_p0: MemoryResource,
        smem_p1: MemoryResource,
    ) -> None:
        """Initialize invariant shared-ring descriptor slots."""
        smem_q.init_descriptor_state()
        smem_kv.init_descriptor_state()
        smem_p0.init_descriptor_state()
        smem_p1.init_descriptor_state()

    @schedule
    def mma_schedule(
        smem_q: MemoryResource,
        smem_kv: MemoryResource,
        tmem_s0: MemoryResource,
        tmem_s1: MemoryResource,
        smem_p0: MemoryResource,
        smem_p1: MemoryResource,
        tmem_o: MemoryResource,
        work_queue: WorkQueue | None = None,
    ) -> None:
        """Wrap shared-ring MMA work in packed persistent skip handling."""
        _decode_work_tile_schedule_with_invariant_bridge(
            cfg,
            work_queue,
            lambda: mma_schedule_prelude(smem_q, smem_kv, smem_p0, smem_p1),
            lambda: smem_q.q_desc(),
            lambda: smem_q.wait(),
            lambda q_desc: mma_schedule_body(
                smem_q,
                smem_kv,
                tmem_s0,
                tmem_s1,
                smem_p0,
                smem_p1,
                tmem_o,
                q_desc,
            ),
        )

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
        num_registers=cfg.mma_load_task_num_registers,
        name="MmaTask",
        **kw,
    )


# ======================================================================
# Softmax0Task — warps 0-3, 4 warps, profile-selected register budget
# LOOP: consume S, produce stats + P + running sum
# LoopLastIter: emit final sum/max for correction tail
# ======================================================================
def create_softmax0_task(
    tmem_s0: MemoryResource,
    tmem_softmax_local0: MemoryResource,
    smem_p0: MemoryResource,
    tmem_softmax_global0: MemoryResource,
    tmem_softmax_order: MemoryResource | None,
    sparse_softmax_metadata: MemoryResource | None,
    work_queue: WorkQueue | None,
    cfg: FmhaDecodeConfig,
    *,
    domain: int | cutlass.Int32,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> Task:
    """Create the first softmax task, including optional ordered publication."""

    def softmax0_schedule_body(
        tmem_s0: MemoryResource,
        tmem_softmax_local0: MemoryResource,
        smem_p0: MemoryResource,
        tmem_softmax_global0: MemoryResource,
        tmem_softmax_order: MemoryResource | None,
        sparse_softmax_metadata: MemoryResource | None,
    ) -> None:
        """Build the softmax0 loop, P publication, and final stats handoff."""
        (
            old_max_arr,
            sum_arr,
            new_max_arr,
            s_arr,
        ) = tmem_s0.init_softmax_state()
        smem_p0.init_compute_state()
        if sparse_softmax_metadata is not None:
            sparse_softmax_metadata.init_read_state()

        with domain_loop(0, domain, 1, unroll=1) as d:
            # ConsWait/ConsWork: load S from TMEM and compute the tile max.
            tmem_s0.wait()
            if sparse_softmax_metadata is not None:
                sparse_softmax_metadata.wait()
                # Copy the complete payload to registers before release, so
                # masking cannot race the producer's next SMEM-stage reuse.
                (
                    sparse_origin0,
                    sparse_origin1,
                    sparse_route_flags,
                    sparse_token_word0,
                    sparse_token_word1,
                    sparse_token_word2,
                    sparse_token_word3,
                ) = sparse_softmax_metadata.load_route()
                sparse_softmax_metadata.release()
                old_max_arr, sum_arr, new_max_arr, s_arr = (
                    tmem_s0.compute_block_sparse_softmax_loop(
                        old_max_arr=old_max_arr,
                        sum_arr=sum_arr,
                        new_max_arr=new_max_arr,
                        s_arr=s_arr,
                        sparse_origin0=sparse_origin0,
                        sparse_origin1=sparse_origin1,
                        sparse_route_flags=sparse_route_flags,
                        sparse_token_word0=sparse_token_word0,
                        sparse_token_word1=sparse_token_word1,
                        sparse_token_word2=sparse_token_word2,
                        sparse_token_word3=sparse_token_word3,
                    )
                )
            else:
                old_max_arr, sum_arr, new_max_arr, s_arr = tmem_s0.compute_softmax_loop(
                    old_max_arr=old_max_arr,
                    sum_arr=sum_arr,
                    new_max_arr=new_max_arr,
                    s_arr=s_arr,
                )
            if cutlass.const_expr(not cfg.use_keeps_mma_ab or not cfg.uses_tmem_p):
                # ConsRelease: free S once the scores are in registers unless
                # a Keeps TMEM-P operand still aliases the consumed columns.
                tmem_s0.release()
            # Publish old/new max before the P path so correction can observe
            # the same stats order as the decode pipeline.
            # ProdWork: store old/new max for correction's in-loop O update.
            tmem_softmax_local0.acquire()
            tmem_softmax_local0.store_loop_old_new_stats(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
            )
            tmem_softmax_local0.commit()
            if cutlass.const_expr(cfg.streams_tmem_p_fragments):
                # Publish one K32 probability fragment at a time so PV can
                # consume early fragments while later scores are processed.
                for fragment_idx in range(cfg.num_softmax_score_fragments):
                    s_arr = tmem_s0.load_softmax_p_fragment(
                        fragment_idx=fragment_idx,
                        s_arr=s_arr,
                    )
                    smem_p0.compute_p_fragment(
                        fragment_idx=fragment_idx,
                        new_max_arr=new_max_arr,
                        s_arr=s_arr,
                    )
            else:
                # Wait for a free P stage before entering the ordered window so
                # BMM2 backpressure on this group's P pipeline cannot extend the
                # baton hold and stall the partner softmax group.
                smem_p0.acquire()
                if tmem_softmax_order is not None:
                    tmem_softmax_order.wait_softmax0()
                # ProdWork: compute P=exp(S-new_max), store it in the profile's
                # SMEM or staged-TMEM operand, and record local sums for the
                # running softmax sum update.
                smem_p0.compute_p(
                    new_max_arr=new_max_arr,
                    s_arr=s_arr,
                )  # publishes the local denominator through tmem_s0
                smem_p0.commit()
                if tmem_softmax_order is not None:
                    tmem_softmax_order.release_softmax1()
            if cutlass.const_expr(cfg.use_keeps_mma_ab and cfg.uses_tmem_p):
                # The TMEM-P store has consumed the aliased S columns, so the
                # next QK wave can now overwrite them.
                tmem_s0.release()
            # ProdWork: FP8 path applies the cross-resource sum correction
            # before TmemS.reduce_sums publishes the new running sums.
            tmem_softmax_global0.global_correction(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
            )  # publishes the corrected denominator through tmem_s0
            # ConsTailWork: update the running sum after P is available.
            sum_arr = tmem_s0.reduce_sums(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
            )
            if cutlass.const_expr(not cfg.use_keeps_mma_ab):
                with d.last_iter():
                    # LastIter ProdWork: store the final sums for correction's
                    # normalization/output tail.
                    tmem_softmax_local0.acquire()
                    tmem_softmax_local0.store_loop_sum_new_stats(
                        old_max_arr=old_max_arr,
                        new_max_arr=new_max_arr,
                        sum_arr=sum_arr,
                    )
                    tmem_softmax_local0.commit()
        if cutlass.const_expr(cfg.use_keeps_mma_ab):
            # The final sum payload has no matching QK/StatsDone token.
            tmem_softmax_local0.acquire()
            tmem_softmax_local0.store_tail_sum_new_stats(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
            )
            tmem_softmax_local0.commit()
            if cutlass.const_expr(
                cfg.use_persistent_scheduler and cfg.uses_staged_one_inst_tmem_p
            ):
                # Tail stats add one stage beyond the S cadence. Advance the
                # second stats slot so both pipelines start the next work tile
                # on the same physical TMEM stage.
                tmem_softmax_local0.acquire()
                tmem_softmax_local0.commit()

    @_schedule_with_optional_resources
    def softmax0_schedule(
        tmem_s0: MemoryResource,
        tmem_softmax_local0: MemoryResource,
        smem_p0: MemoryResource,
        tmem_softmax_global0: MemoryResource,
        tmem_softmax_order: MemoryResource | None,
        sparse_softmax_metadata: MemoryResource | None,
        work_queue: WorkQueue | None,
    ) -> None:
        """Schedule softmax0 with the supplied order and sparse resources."""
        _decode_work_tile_schedule(
            cfg,
            work_queue,
            lambda: softmax0_schedule_body(
                tmem_s0,
                tmem_softmax_local0,
                smem_p0,
                tmem_softmax_global0,
                tmem_softmax_order,
                sparse_softmax_metadata,
            ),
        )

    schedule_result = softmax0_schedule(
        tmem_s0,
        tmem_softmax_local0,
        smem_p0,
        tmem_softmax_global0,
        tmem_softmax_order,
        sparse_softmax_metadata,
        work_queue,
    )
    src = [tmem_s0]
    if sparse_softmax_metadata is not None:
        src.append(sparse_softmax_metadata)
    if work_queue is not None:
        src.append(work_queue)
    dst = [tmem_softmax_local0, smem_p0, tmem_softmax_global0]
    if tmem_softmax_order is not None:
        dst.append(tmem_softmax_order)
    return task_class(
        src_resources=src,
        dst_resources=dst,
        q_bound_resources=((tmem_s0, False),),
        cfg=cfg,
        warp_idx=cfg.softmax0_warp_idx,
        num_warps=cfg.softmax0_num_warps,
        schedule=schedule_result,
        num_registers=cfg.softmax_task_num_registers,
        name="Softmax0Task",
        **kw,
    )


# ======================================================================
# Softmax1Task — warps 4-7, 4 warps, profile-selected register budget
# ======================================================================
def create_softmax1_task(
    tmem_s1: MemoryResource,
    tmem_softmax_local1: MemoryResource,
    smem_p1: MemoryResource,
    tmem_softmax_global1: MemoryResource,
    tmem_softmax_order: MemoryResource | None,
    sparse_softmax_metadata: MemoryResource | None,
    work_queue: WorkQueue | None,
    cfg: FmhaDecodeConfig,
    *,
    domain: int | cutlass.Int32,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> Task:
    """Create the second softmax task, including optional ordered publication."""

    def softmax1_schedule_body(
        tmem_s1: MemoryResource,
        tmem_softmax_local1: MemoryResource,
        smem_p1: MemoryResource,
        tmem_softmax_global1: MemoryResource,
        tmem_softmax_order: MemoryResource | None,
        sparse_softmax_metadata: MemoryResource | None,
    ) -> None:
        """Build the softmax1 loop, P publication, and final stats handoff."""
        (
            old_max_arr,
            sum_arr,
            new_max_arr,
            s_arr,
        ) = tmem_s1.init_softmax_state()
        smem_p1.init_compute_state()
        if sparse_softmax_metadata is not None:
            sparse_softmax_metadata.init_read_state()

        with domain_loop(0, domain, 1, unroll=1) as d:
            # ConsWait/ConsWork: load the second S instance and compute max.
            tmem_s1.wait()
            if sparse_softmax_metadata is not None:
                sparse_softmax_metadata.wait()
                # Copy to registers before release so the producer can reuse
                # the SMEM stage while this warp group applies the masks.
                (
                    sparse_origin0,
                    sparse_origin1,
                    sparse_route_flags,
                    sparse_token_word0,
                    sparse_token_word1,
                    sparse_token_word2,
                    sparse_token_word3,
                ) = sparse_softmax_metadata.load_route()
                sparse_softmax_metadata.release()
                old_max_arr, sum_arr, new_max_arr, s_arr = (
                    tmem_s1.compute_block_sparse_softmax_loop(
                        old_max_arr=old_max_arr,
                        sum_arr=sum_arr,
                        new_max_arr=new_max_arr,
                        s_arr=s_arr,
                        sparse_origin0=sparse_origin0,
                        sparse_origin1=sparse_origin1,
                        sparse_route_flags=sparse_route_flags,
                        sparse_token_word0=sparse_token_word0,
                        sparse_token_word1=sparse_token_word1,
                        sparse_token_word2=sparse_token_word2,
                        sparse_token_word3=sparse_token_word3,
                    )
                )
            else:
                old_max_arr, sum_arr, new_max_arr, s_arr = tmem_s1.compute_softmax_loop(
                    old_max_arr=old_max_arr,
                    sum_arr=sum_arr,
                    new_max_arr=new_max_arr,
                    s_arr=s_arr,
                )
            if cutlass.const_expr(not cfg.use_keeps_mma_ab or not cfg.uses_tmem_p):
                # ConsRelease: SMEM-P Keeps and Swaps no longer need S after
                # the score fragment has been loaded into registers.
                tmem_s1.release()
            # ProdWork: store old/new max for the correction task.
            tmem_softmax_local1.acquire()
            tmem_softmax_local1.store_loop_old_new_stats(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
            )
            tmem_softmax_local1.commit()
            if cutlass.const_expr(cfg.streams_tmem_p_fragments):
                for fragment_idx in range(cfg.num_softmax_score_fragments):
                    s_arr = tmem_s1.load_softmax_p_fragment(
                        fragment_idx=fragment_idx,
                        s_arr=s_arr,
                    )
                    smem_p1.compute_p_fragment(
                        fragment_idx=fragment_idx,
                        new_max_arr=new_max_arr,
                        s_arr=s_arr,
                    )
            else:
                # Wait for a free P stage before entering the ordered window so
                # BMM2 backpressure on this group's P pipeline cannot extend the
                # baton hold and stall the partner softmax group.
                smem_p1.acquire()
                if tmem_softmax_order is not None:
                    tmem_softmax_order.wait_softmax1()
                # ProdWork: compute and publish P1 for BMM2.
                smem_p1.compute_p(new_max_arr=new_max_arr, s_arr=s_arr)
                smem_p1.commit()
                if tmem_softmax_order is not None:
                    tmem_softmax_order.release_softmax0()
            if cutlass.const_expr(cfg.use_keeps_mma_ab and cfg.uses_tmem_p):
                # The TMEM-P store has consumed the aliased S columns, so the
                # next QK wave can now overwrite them.
                tmem_s1.release()
            # ProdWork: update FP8 global sums if the configuration needs the
            # split softmax-sum correction path.
            tmem_softmax_global1.global_correction(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
            )
            # ConsTailWork: fold local sums into the running sums.
            sum_arr = tmem_s1.reduce_sums(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
            )
            if cutlass.const_expr(not cfg.use_keeps_mma_ab):
                with d.last_iter():
                    # LastIter ProdWork: publish final sums for output
                    # normalization.
                    tmem_softmax_local1.acquire()
                    tmem_softmax_local1.store_loop_sum_new_stats(
                        old_max_arr=old_max_arr,
                        new_max_arr=new_max_arr,
                        sum_arr=sum_arr,
                    )
                    tmem_softmax_local1.commit()
        if cutlass.const_expr(cfg.use_keeps_mma_ab):
            # The final sum payload has no matching QK/StatsDone token.
            tmem_softmax_local1.acquire()
            tmem_softmax_local1.store_tail_sum_new_stats(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
            )
            tmem_softmax_local1.commit()

    @_schedule_with_optional_resources
    def softmax1_schedule(
        tmem_s1: MemoryResource,
        tmem_softmax_local1: MemoryResource,
        smem_p1: MemoryResource,
        tmem_softmax_global1: MemoryResource,
        tmem_softmax_order: MemoryResource | None,
        sparse_softmax_metadata: MemoryResource | None,
        work_queue: WorkQueue | None,
    ) -> None:
        """Schedule softmax1 with the supplied order and sparse resources."""
        # Prime the first P0 -> P1 baton once per CTA.  Each completed P1
        # publication leaves the same barrier half-arrived for the next
        # persistent work tile, so re-priming inside the work-tile loop would
        # toggle its phase early and deadlock on the second tile.
        if tmem_softmax_order is not None:
            tmem_softmax_order.prime_softmax1()
        _decode_work_tile_schedule(
            cfg,
            work_queue,
            lambda: softmax1_schedule_body(
                tmem_s1,
                tmem_softmax_local1,
                smem_p1,
                tmem_softmax_global1,
                tmem_softmax_order,
                sparse_softmax_metadata,
            ),
        )

    schedule_result = softmax1_schedule(
        tmem_s1,
        tmem_softmax_local1,
        smem_p1,
        tmem_softmax_global1,
        tmem_softmax_order,
        sparse_softmax_metadata,
        work_queue,
    )
    src = [tmem_s1]
    if sparse_softmax_metadata is not None:
        src.append(sparse_softmax_metadata)
    if tmem_softmax_order is not None:
        src.append(tmem_softmax_order)
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[tmem_softmax_local1, smem_p1, tmem_softmax_global1],
        q_bound_resources=((tmem_s1, False),),
        cfg=cfg,
        warp_idx=cfg.softmax1_warp_idx,
        num_warps=cfg.softmax1_num_warps,
        schedule=schedule_result,
        num_registers=cfg.softmax_task_num_registers,
        name="Softmax1Task",
        **kw,
    )


# ======================================================================
# CorrectionTask — warps 8-11, 4 warps, profile-selected register budget
# HEAD: drain initial softmax-local stats
# LOOP: correct O0 / O1 in-place before the next BMM2 accumulation
# TAIL: combine final O0/O1, normalize, store to GMEM
# ======================================================================
def create_correction_task(
    tmem_softmax_local0: MemoryResource,
    tmem_softmax_local1: MemoryResource,
    tmem_o: MemoryResource,
    tmem_corr0: MemoryResource,
    tmem_corr1: MemoryResource,
    work_queue: WorkQueue | None,
    smem_kv_reuse_credit: MemoryResource | None,
    cfg: FmhaDecodeConfig,
    *,
    domain: int | cutlass.Int32,
    tmem_stats_done0: MemoryResource | None = None,
    tmem_stats_done1: MemoryResource | None = None,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> Task:
    """Create the two-instance correction and output task."""

    if smem_kv_reuse_credit is not None and work_queue is None:
        raise ValueError("KV reuse credit requires a work queue")

    def correction_schedule_body(
        tmem_softmax_local0: MemoryResource,
        tmem_softmax_local1: MemoryResource,
        tmem_o: MemoryResource,
        tmem_corr0: MemoryResource,
        tmem_corr1: MemoryResource,
        tmem_stats_done0: MemoryResource | None,
        tmem_stats_done1: MemoryResource | None,
        smem_kv_reuse_credit: MemoryResource | None,
    ) -> None:
        """Schedule two-instance O correction and final output normalization."""

        def consume_local_with_load(
            tmem_softmax_local: MemoryResource,
            tmem_stats_done: MemoryResource | None,
            local_state: tuple[
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
            ],
            load_stats_work,
        ) -> tuple[
            cutlass.Array,
            cutlass.Array,
            cutlass.Array,
            cutlass.Array,
            cutlass.Array,
            cutlass.Array,
        ]:
            """Wait, load, and release one phase-specific stats payload."""
            (
                old_max_arr,
                new_max_arr,
                sum_arr,
                inst_old_max_arr,
                inst_new_max_arr,
                inst_sum_arr,
            ) = local_state
            # ConsWait/ConsWork: load the phase-specific softmax stats payload
            # from TMEM-local storage.
            tmem_softmax_local.wait()
            local_state = load_stats_work(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
                inst_old_max_arr=inst_old_max_arr,
                inst_new_max_arr=inst_new_max_arr,
                inst_sum_arr=inst_sum_arr,
            )
            if tmem_stats_done is not None:
                # Return the matching S-overwrite credit only after the stats
                # payload is resident in Correction registers.
                tmem_stats_done.wait()
                tmem_stats_done.release()
            # ConsRelease: the softmax-local payload can now be reused.
            tmem_softmax_local.release()
            return local_state

        def correct_o(
            tmem_softmax_local: MemoryResource,
            tmem_stats_done: MemoryResource | None,
            tmem_corr: MemoryResource,
            local_state: tuple[
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
            ],
            tail_0: cutlass.Int32,
            tail_1: cutlass.Int32,
        ) -> tuple[
            tuple[
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
            ],
            cutlass.Int32,
            cutlass.Int32,
        ]:
            """Consume loop stats and rescale the matching O stage."""
            # Load old/new max for the O stage that is about to be corrected.
            local_state = consume_local_with_load(
                tmem_softmax_local,
                tmem_stats_done,
                local_state,
                tmem_softmax_local.load_loop_stats,
            )
            old_max_arr = local_state[0]
            new_max_arr = local_state[1]
            # ConsWait/ConsWork: wait for the matching TMEM O stage and record
            # which O buffer is being consumed.
            tmem_o.wait()
            o_stage_idx, tail_0, tail_1 = tmem_o.update_o_stage_loop(
                tail_o_stage_idx_0=tail_0,
                tail_o_stage_idx_1=tail_1,
            )
            # ProdWork: rescale O in place before the next BMM2 accumulation.
            tmem_corr.correction_loop_epilogue(
                o_stage_idx=o_stage_idx,
                tail_o_stage_idx_0=tail_0,
                tail_o_stage_idx_1=tail_1,
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
            )
            tmem_o.release()
            return local_state, tail_0, tail_1

        local0_state = tmem_softmax_local0.init_stats_state()
        local1_state = tmem_softmax_local1.init_stats_state()
        _, tail_0, tail_1 = tmem_o.init_stage_state()
        tmem_corr0.init_epilogue_state()
        tmem_corr1.init_epilogue_state()

        # HEAD: drain the first softmax-local handoff. No O is corrected here;
        # this aligns the stats pipeline before loop work starts.
        local0_state = consume_local_with_load(
            tmem_softmax_local0,
            tmem_stats_done0,
            local0_state,
            tmem_softmax_local0.load_head_stats,
        )
        local1_state = consume_local_with_load(
            tmem_softmax_local1,
            tmem_stats_done1,
            local1_state,
            tmem_softmax_local1.load_head_stats,
        )

        # LOOP: each iteration corrects O0 and O1 before later BMM2 waves
        # accumulate into the same TMEM columns.
        with domain_loop(0, domain, 1, unroll=1):
            local0_state, tail_0, tail_1 = correct_o(
                tmem_softmax_local0,
                tmem_stats_done0,
                tmem_corr0,
                local0_state,
                tail_0,
                tail_1,
            )
            local1_state, tail_0, tail_1 = correct_o(
                tmem_softmax_local1,
                tmem_stats_done1,
                tmem_corr1,
                local1_state,
                tail_0,
                tail_1,
            )

        # TAIL inst0: consume the final stats and mark the first tail O stage.
        local0_state = consume_local_with_load(
            tmem_softmax_local0,
            None,
            local0_state,
            tmem_softmax_local0.load_tail_stats,
        )
        old_max_arr = local0_state[0]
        new_max_arr = local0_state[1]
        inst0_new_max_arr = local0_state[4]
        inst0_sum_arr = local0_state[5]
        tmem_o.wait()
        o_stage_idx, tail_0, tail_1 = tmem_o.update_o_stage_tail(
            tail_o_stage_idx_0=tail_0,
            tail_o_stage_idx_1=tail_1,
            inst_idx=KV_INST0,
        )
        tmem_corr0.correction_tail_epilogue(
            o_stage_idx=o_stage_idx,
            tail_o_stage_idx_0=tail_0,
            tail_o_stage_idx_1=tail_1,
            old_max_arr=old_max_arr,
            new_max_arr=new_max_arr,
            inst0_new_max_arr=inst0_new_max_arr,
            inst0_sum_arr=inst0_sum_arr,
            inst1_new_max_arr=inst0_new_max_arr,
            inst1_sum_arr=inst0_sum_arr,
        )
        # TAIL inst1: consume final stats for the second instance. This call
        # performs the final two-instance normalization and output store.
        local1_state = consume_local_with_load(
            tmem_softmax_local1,
            None,
            local1_state,
            tmem_softmax_local1.load_tail_stats,
        )
        old_max_arr = local1_state[0]
        new_max_arr = local1_state[1]
        inst1_new_max_arr = local1_state[4]
        inst1_sum_arr = local1_state[5]
        tmem_o.wait()
        o_stage_idx, tail_0, tail_1 = tmem_o.update_o_stage_tail(
            tail_o_stage_idx_0=tail_0,
            tail_o_stage_idx_1=tail_1,
            inst_idx=KV_INST1,
        )
        if smem_kv_reuse_credit is None:
            tmem_corr1.correction_tail_epilogue(
                o_stage_idx=o_stage_idx,
                tail_o_stage_idx_0=tail_0,
                tail_o_stage_idx_1=tail_1,
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                inst0_new_max_arr=inst0_new_max_arr,
                inst0_sum_arr=inst0_sum_arr,
                inst1_new_max_arr=inst1_new_max_arr,
                inst1_sum_arr=inst1_sum_arr,
            )
        else:
            # The stage selector and ownership token share one pipeline epoch.
            # Wait before the first aliased access and release immediately
            # after correction stops touching the selected KV-ring stage.
            smem_kv_reuse_credit.wait()
            scratch_stage = smem_kv_reuse_credit.read_scratch_stage()
            tmem_corr1.correction_tail_epilogue_rotating_exchange(
                scratch_stage=scratch_stage,
                o_stage_idx=o_stage_idx,
                tail_o_stage_idx_0=tail_0,
                tail_o_stage_idx_1=tail_1,
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                inst0_new_max_arr=inst0_new_max_arr,
                inst0_sum_arr=inst0_sum_arr,
                inst1_new_max_arr=inst1_new_max_arr,
                inst1_sum_arr=inst1_sum_arr,
            )
            smem_kv_reuse_credit.release()
        # Inst1 final reduction consumes both O0 and O1, so defer O0 release
        # until after inst1 has finished reading it.
        tmem_o.release()
        tmem_o.release()

    def run_correction_schedule(
        tmem_softmax_local0: MemoryResource,
        tmem_softmax_local1: MemoryResource,
        tmem_o: MemoryResource,
        tmem_corr0: MemoryResource,
        tmem_corr1: MemoryResource,
        tmem_stats_done0: MemoryResource | None,
        tmem_stats_done1: MemoryResource | None,
        smem_kv_reuse_credit: MemoryResource | None,
        work_queue: WorkQueue | None,
    ) -> None:
        """Wrap correction with optional stats lifetime gates."""
        _decode_work_tile_schedule(
            cfg,
            work_queue,
            lambda: correction_schedule_body(
                tmem_softmax_local0,
                tmem_softmax_local1,
                tmem_o,
                tmem_corr0,
                tmem_corr1,
                tmem_stats_done0,
                tmem_stats_done1,
                smem_kv_reuse_credit,
            ),
        )

    @schedule
    def correction_schedule(
        tmem_softmax_local0: MemoryResource,
        tmem_softmax_local1: MemoryResource,
        tmem_o: MemoryResource,
        tmem_corr0: MemoryResource,
        tmem_corr1: MemoryResource,
        work_queue: WorkQueue | None = None,
        smem_kv_reuse_credit: MemoryResource | None = None,
    ) -> None:
        """Capture the Swaps correction schedule."""
        run_correction_schedule(
            tmem_softmax_local0,
            tmem_softmax_local1,
            tmem_o,
            tmem_corr0,
            tmem_corr1,
            None,
            None,
            smem_kv_reuse_credit,
            work_queue,
        )

    @schedule
    def correction_keeps_schedule(
        tmem_softmax_local0: MemoryResource,
        tmem_softmax_local1: MemoryResource,
        tmem_o: MemoryResource,
        tmem_corr0: MemoryResource,
        tmem_corr1: MemoryResource,
        tmem_stats_done0: MemoryResource,
        tmem_stats_done1: MemoryResource,
        work_queue: WorkQueue | None = None,
        smem_kv_reuse_credit: MemoryResource | None = None,
    ) -> None:
        """Capture Keeps correction with explicit stats lifetime gates."""
        run_correction_schedule(
            tmem_softmax_local0,
            tmem_softmax_local1,
            tmem_o,
            tmem_corr0,
            tmem_corr1,
            tmem_stats_done0,
            tmem_stats_done1,
            smem_kv_reuse_credit,
            work_queue,
        )

    if tmem_stats_done0 is None or tmem_stats_done1 is None:
        if work_queue is None:
            captured_schedule = correction_schedule(
                tmem_softmax_local0,
                tmem_softmax_local1,
                tmem_o,
                tmem_corr0,
                tmem_corr1,
            )
        elif smem_kv_reuse_credit is None:
            captured_schedule = correction_schedule(
                tmem_softmax_local0,
                tmem_softmax_local1,
                tmem_o,
                tmem_corr0,
                tmem_corr1,
                work_queue,
            )
        else:
            captured_schedule = correction_schedule(
                tmem_softmax_local0,
                tmem_softmax_local1,
                tmem_o,
                tmem_corr0,
                tmem_corr1,
                work_queue,
                smem_kv_reuse_credit,
            )
        src = [tmem_softmax_local0, tmem_softmax_local1, tmem_o]
    else:
        if work_queue is None:
            captured_schedule = correction_keeps_schedule(
                tmem_softmax_local0,
                tmem_softmax_local1,
                tmem_o,
                tmem_corr0,
                tmem_corr1,
                tmem_stats_done0,
                tmem_stats_done1,
            )
        elif smem_kv_reuse_credit is None:
            captured_schedule = correction_keeps_schedule(
                tmem_softmax_local0,
                tmem_softmax_local1,
                tmem_o,
                tmem_corr0,
                tmem_corr1,
                tmem_stats_done0,
                tmem_stats_done1,
                work_queue,
            )
        else:
            captured_schedule = correction_keeps_schedule(
                tmem_softmax_local0,
                tmem_softmax_local1,
                tmem_o,
                tmem_corr0,
                tmem_corr1,
                tmem_stats_done0,
                tmem_stats_done1,
                work_queue,
                smem_kv_reuse_credit,
            )
        src = [
            tmem_softmax_local0,
            tmem_softmax_local1,
            tmem_o,
            tmem_stats_done0,
            tmem_stats_done1,
        ]
    if work_queue is not None:
        src.append(work_queue)
    if smem_kv_reuse_credit is not None:
        src.append(smem_kv_reuse_credit)
    return task_class(
        src_resources=src,
        dst_resources=[tmem_corr0, tmem_corr1],
        q_bound_resources=((tmem_corr0, True), (tmem_corr1, True)),
        cfg=cfg,
        warp_idx=cfg.correction_warp_idx,
        num_warps=cfg.correction_num_warps,
        schedule=captured_schedule,
        num_registers=cfg.correction_task_num_registers,
        name="CorrectionTask",
        **kw,
    )


def create_correction_task_one_inst_qkv(
    tmem_softmax_local: MemoryResource,
    tmem_o: MemoryResource,
    tmem_corr: MemoryResource,
    work_queue: WorkQueue | None,
    cfg: FmhaDecodeConfig,
    *,
    tmem_stats_done: MemoryResource,
    domain: int | cutlass.Int32,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> Task:
    """Create the correction task for the one-inst QKV path."""

    def correction_schedule_body(
        tmem_softmax_local: MemoryResource,
        tmem_o: MemoryResource,
        tmem_corr: MemoryResource,
        tmem_stats_done: MemoryResource,
    ) -> None:
        """Schedule one-inst O correction and final output normalization."""

        def consume_local_with_load(
            local_state: tuple[
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
            ],
            load_stats_work,
            release_stats_done: bool,
        ) -> tuple[
            cutlass.Array,
            cutlass.Array,
            cutlass.Array,
            cutlass.Array,
            cutlass.Array,
            cutlass.Array,
        ]:
            """Wait, load, and release one one-inst stats payload."""
            (
                old_max_arr,
                new_max_arr,
                sum_arr,
                inst_old_max_arr,
                inst_new_max_arr,
                inst_sum_arr,
            ) = local_state
            # ConsWait/ConsWork: load the stage-specific softmax-local payload
            # produced by the one-inst softmax task.
            tmem_softmax_local.wait()
            local_state = load_stats_work(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
                inst_old_max_arr=inst_old_max_arr,
                inst_new_max_arr=inst_new_max_arr,
                inst_sum_arr=inst_sum_arr,
            )
            if release_stats_done:
                # Return one S-overwrite credit for this QK-derived payload.
                tmem_stats_done.wait()
                tmem_stats_done.release()
            # ConsRelease: the softmax-local payload can now be reused.
            tmem_softmax_local.release()
            return local_state

        def correct_o(
            local_state: tuple[
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
            ],
            tail_0: cutlass.Int32,
            tail_1: cutlass.Int32,
        ) -> tuple[
            tuple[
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
            ],
            cutlass.Int32,
            cutlass.Int32,
        ]:
            """Consume loop stats and rescale the one-inst O stage."""
            local_state = consume_local_with_load(
                local_state,
                tmem_softmax_local.load_loop_stats,
                True,
            )
            old_max_arr = local_state[0]
            new_max_arr = local_state[1]
            # ConsWait/ConsWork: consume the O stage produced by BMM2. The
            # stage state tracks which TMEM O slot is live and which two slots
            # become the final tail pair.
            tmem_o.wait()
            o_stage_idx, tail_0, tail_1 = tmem_o.update_o_stage_loop(
                tail_o_stage_idx_0=tail_0,
                tail_o_stage_idx_1=tail_1,
            )
            # ProdWork: loop phase rescales O in place before the next BMM2
            # accumulation.
            tmem_corr.correction_loop_epilogue(
                o_stage_idx=o_stage_idx,
                tail_o_stage_idx_0=tail_0,
                tail_o_stage_idx_1=tail_1,
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
            )
            # ConsRelease: the consumed O stage can be reused by later BMM2.
            tmem_o.release()
            return local_state, tail_0, tail_1

        local_state = tmem_softmax_local.init_stats_state()
        _, tail_0, tail_1 = tmem_o.init_stage_state()
        tmem_corr.init_epilogue_state()

        # HEAD: drain the first stats handoff. The first payload only seeds
        # the correction state; there is no prior O stage to rescale.
        local_state = consume_local_with_load(
            local_state,
            tmem_softmax_local.load_head_stats,
            True,
        )
        # LOOP: each payload/O pair corresponds to a completed BMM2 wave whose
        # accumulator must be max-corrected before more V work accumulates.
        with domain_loop(0, domain, 1, unroll=1):
            local_state, tail_0, tail_1 = correct_o(local_state, tail_0, tail_1)
        # TAIL: the final payload carries sum/new max and the final O stage is
        # normalized/stored instead of being staged for another accumulation.
        local_state = consume_local_with_load(
            local_state,
            tmem_softmax_local.load_tail_stats,
            False,
        )
        if cutlass.const_expr(
            cfg.use_persistent_scheduler and cfg.uses_staged_one_inst_tmem_p
        ):
            # Match the producer's empty tail stage before the persistent
            # work queue advances to the next tile.
            tmem_softmax_local.wait()
            tmem_softmax_local.release()
        old_max_arr = local_state[0]
        new_max_arr = local_state[1]
        inst_new_max_arr = local_state[4]
        inst_sum_arr = local_state[5]
        # ConsWait/ConsWork: consume the final O stage produced by BMM2.
        tmem_o.wait()
        o_stage_idx, tail_0, tail_1 = tmem_o.update_o_stage_tail(
            tail_o_stage_idx_0=tail_0,
            tail_o_stage_idx_1=tail_1,
            inst_idx=KV_INST0,
        )
        # ProdWork: tail phase normalizes and stores final O.
        tmem_corr.correction_tail_epilogue(
            o_stage_idx=o_stage_idx,
            tail_o_stage_idx_0=tail_0,
            tail_o_stage_idx_1=tail_1,
            old_max_arr=old_max_arr,
            new_max_arr=new_max_arr,
            inst0_new_max_arr=inst_new_max_arr,
            inst0_sum_arr=inst_sum_arr,
            inst1_new_max_arr=inst_new_max_arr,
            inst1_sum_arr=inst_sum_arr,
        )
        # ConsRelease: the final O stage is no longer needed.
        tmem_o.release()

    @schedule
    def correction_schedule(
        tmem_softmax_local: MemoryResource,
        tmem_o: MemoryResource,
        tmem_corr: MemoryResource,
        tmem_stats_done: MemoryResource,
        work_queue: WorkQueue | None = None,
    ) -> None:
        """Wrap one-inst correction in packed persistent skip handling."""
        _decode_work_tile_schedule(
            cfg,
            work_queue,
            lambda: correction_schedule_body(
                tmem_softmax_local,
                tmem_o,
                tmem_corr,
                tmem_stats_done,
            ),
        )

    schedule_result = (
        correction_schedule(
            tmem_softmax_local,
            tmem_o,
            tmem_corr,
            tmem_stats_done,
        )
        if work_queue is None
        else correction_schedule(
            tmem_softmax_local,
            tmem_o,
            tmem_corr,
            tmem_stats_done,
            work_queue,
        )
    )
    src = [tmem_softmax_local, tmem_o, tmem_stats_done]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[tmem_corr],
        q_bound_resources=((tmem_corr, True),),
        cfg=cfg,
        warp_idx=cfg.correction_warp_idx,
        num_warps=cfg.correction_num_warps,
        schedule=schedule_result,
        num_registers=cfg.correction_task_num_registers,
        name="CorrectionTask",
        **kw,
    )


# ======================================================================
# PaddingTask — fills the unused tail warps of one warp group
# ======================================================================
def create_padding_task(
    cfg: FmhaDecodeConfig,
    work_queue: WorkQueue | None = None,
    *,
    warp_idx: int,
    num_warps: int,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> Task:
    """Create the padding task used to balance persistent warp groups."""

    @schedule
    def padding_schedule(work_queue: WorkQueue | None = None) -> None:
        """Run the padding schedule and advance the work queue if present."""

        def padding_body() -> None:
            # Empty loop body: this warp participates in task/register
            # scheduling so the persistent warpgroup layout remains balanced.
            with domain_loop(0, 1, 1, unroll=1):
                pass

        _decode_work_tile_schedule(cfg, work_queue, padding_body)

    captured_schedule = (
        padding_schedule() if work_queue is None else padding_schedule(work_queue)
    )
    src = []
    if work_queue is not None:
        src = [work_queue]
    return task_class(
        src_resources=src,
        dst_resources=[],
        cfg=cfg,
        warp_idx=warp_idx,
        num_warps=num_warps,
        schedule=captured_schedule,
        num_registers=cfg.mma_load_task_num_registers,
        name="PaddingTask",
        **kw,
    )


def create_scheduler_task(
    work_queue: WorkQueue,
    schedule_token_throttle: MemoryResource | None,
    cfg: FmhaDecodeConfig,
    *,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> Task:
    """Create the CLC scheduler task.

    Two-inst profiles place scheduler/load/padding roles in WG3. One-inst Keeps
    profiles place scheduler at warp 9 beside MMA, page-or-padding, and load in
    WG2; a separate padding task keeps WG3 and the 512-thread CLC contract full.
    """

    @schedule
    def scheduler_schedule(
        work_queue: WorkQueue,
        schedule_token_throttle: MemoryResource | None = None,
    ) -> None:
        """Fetch and publish the next persistent-scheduler work tile."""
        with work_tile_loop(work_queue):
            # The scheduler owns work-tile discovery for persistent kernels.
            # Empty domain keeps the generated schedule shape consistent with
            # other TS tasks while all real work happens through WorkQueue.
            with domain_loop(0, 0, 1, unroll=1):
                pass
            _schedule_token_throttle_tail(schedule_token_throttle)
            # ProdAcquire/ProdWork/ProdCommit: fetch and publish the next work
            # tile to all persistent tasks.
            work_queue.acquire()
            work_queue.fetch_work_tile()
            work_queue.commit()
            _work_queue_tail(work_queue)

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
        num_registers=cfg.mma_load_task_num_registers,
        name="SchedulerTask",
        **kw,
    )
