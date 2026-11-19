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

"""BatchedGemm-local Task Scheduling extensions for physical CGAs.

The installed TS framework intentionally remains untouched.  These subclasses
add only the runtime CTA policy and TMA-UMMA mask behavior needed by this
kernel; legacy resources continue to use the wheel implementations unchanged.

The extensions cover two independent synchronization layers:

* :class:`CgaTask` decides which physical CTAs execute a task. It does not
  decide where a TMA instruction sends its payload.
* ``CgaPipeline*`` classes replace static 2x1 peer masks with masks relative to
  the current 2-CTA atom. :class:`CgaPipelineTmaUmma` additionally separates
  payload destinations, full-barrier owners, and empty-barrier releases as
  documented in :mod:`batched_gemm_cga`.
"""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Boolean, Int32
from cutlass.cutlass_dsl import if_generate
from cutlass.experimental.task_scheduling.pipeline import (
    TSPipelineAsyncUmma,
    TSPipelineUmmaUmma,
)
from cutlass.experimental.task_scheduling.task import Task
from cutlass.pipeline.sm100 import PipelineTmaUmma, PipelineUmmaAsync

from .batched_gemm_cga import (
    CgaPolicy,
    cga_policy_selected,
    cga_tma_masks,
)


# Newer TS exposes ``_run_selected_body`` as the run_body entry point (and
# ``_leader_cta_gate`` as the producer-gate hook); older TS does not. CgaTask
# overrides both, so it picks the matching variant at import time.
_TS_HAS_SELECTED_BODY = hasattr(Task, "_run_selected_body")


class CgaTask(Task):
    """Task whose CTA ownership can follow the active physical CGA.

    Selection is the intersection of three independent predicates:

    1. the task's assigned warp interval;
    2. the stock ``run_only_on_cta_id`` constraint, when present;
    3. the :class:`CgaPolicy` physical-topology predicate.

    Most tasks remain stock :class:`Task` objects. This subclass is constructed
    for CTA-rank-gated register redistribution, adaptive multicast/per-atom
    ownership, or a resource whose producer must be physical-CGA-global CTA0
    rather than the static 2x1 CTA leader used by the Task Scheduling wheel.
    """

    def __init__(self, *args, cfg, cga_policy: int = int(CgaPolicy.ALL), **kwargs):
        self.cfg = cfg
        dst_resources = kwargs.get("dst_resources", ())
        self._global_cta0_producer_pipeline_configs = tuple(
            resource.pipeline_config
            for resource in dst_resources
            if getattr(resource, "global_cga_producer_cta0", False)
            and resource.pipeline_config is not None
        )
        super().__init__(*args, **kwargs)
        self.cga_policy = int(cga_policy)

    @cute.jit
    def is_selected(self) -> bool:
        """Evaluate warp, legacy CTA, and physical-CGA policy ownership."""
        warp_id = cute.arch.warp_idx()
        selected = (warp_id >= self.warp_start) & (warp_id < self.warp_end)
        if cutlass.const_expr(self.run_only_on_cta_id is not None):
            cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            selected = selected & (cta_rank == self.run_only_on_cta_id)
        if cutlass.const_expr(self.cga_policy != int(CgaPolicy.ALL)):
            selected = selected & cga_policy_selected(self.cfg, self.cga_policy)
        return selected

    def _is_global_cta0_producer_config(self, pipeline_config) -> bool:
        """True for a pipeline whose producer is the physical-CGA-global CTA0."""
        return any(
            pipeline_config is candidate
            for candidate in self._global_cta0_producer_pipeline_configs
        )

    def _leader_cta_gate(self, cfg, signaling):
        """Use a physical-CGA-global CTA0 leader for selected local resources.

        The stock gate selects ``rank % v_size == 0`` from the static 2-CTA
        atom, which would also select ranks 2, 4, ... in a wider CGA.  Used by TS
        releases that call this hook; older TS uses ``_is_producer_signaling_cta``.
        """
        if cutlass.const_expr(
            signaling.has_cta_leader()
            and not self._runs_on_single_cta()
            and self._is_global_cta0_producer_config(cfg)
        ):
            cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            return cta_rank == Int32(0)
        return super()._leader_cta_gate(cfg, signaling)

    def _is_producer_signaling_cta(self, pipeline_config) -> bool:
        """Use a physical-CGA-global producer for selected local resources.

        Used by older TS only; newer TS uses ``_leader_cta_gate``.
        """
        if cutlass.const_expr(self._is_global_cta0_producer_config(pipeline_config)):
            cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            return cta_rank == Int32(0)
        return super()._is_producer_signaling_cta(pipeline_config)

    @cute.jit
    def run_body(self, context=None) -> None:
        """Run selected tasks and redistribute registers on skipped CTA ranks.

        ``Task.run_body()`` places ``setmaxregister`` inside ``is_selected()``.
        A CTA ownership constraint or runtime CGA policy can therefore skip a
        register-adjusted task on only part of the physical CGA.  Those warps
        would retain the kernel's default register allocation, violating the
        warp-group-wide ``setmaxnreg`` contract and potentially starving a
        later high-register task in the same CTA.

        Keep the task body runtime-gated, while making every warp assigned to
        this task perform the same configured register request, including an
        increase.  All warps in a warp group must execute the same request even
        when a task has no work on the current CTA rank.
        """
        warp_id = cute.arch.warp_idx()
        is_assigned_warp = (warp_id >= self.warp_start) & (warp_id < self.warp_end)
        is_selected = self.is_selected()
        if cutlass.const_expr(_TS_HAS_SELECTED_BODY):
            self._seed_route_slots()

        if is_selected:
            if cutlass.const_expr(_TS_HAS_SELECTED_BODY):
                self._run_selected_body()
                # Mirrors ``Task.run_body``.
                self.dummy = cutlass.Boolean(True)
            else:
                self._set_max_register()
                # Only TS 4.7 passes ``context``; its body helpers accept it,
                # 4.8's take no argument.
                if cutlass.const_expr(context is None):
                    if cutlass.const_expr(self.is_persistent):
                        self._run_task_body_persistent()
                    else:
                        self._run_task_body_non_persistent()
                else:
                    if cutlass.const_expr(self.is_persistent):
                        self._run_task_body_persistent(context)
                    else:
                        self._run_task_body_non_persistent(context)
        else:
            if is_assigned_warp:
                self._set_max_register()


@dataclass(frozen=True)
class CgaPipelineTmaUmma(PipelineTmaUmma):
    """TMA-UMMA pipeline with active-shape full/release masks.

    CUTLASS 4.7's Python pipeline stores ``producer_mask`` but arms only the
    local full barrier.  Flexible CGA multicast needs every projected 2-CTA
    barrier owner armed, so the elected even-X CTA performs one remote
    arrive-and-expect-tx per owner bit. Here ``producer_mask`` is specifically
    the full-owner mask and ``consumer_mask`` is specifically the empty-release
    mask; the TMALDG payload mask is passed independently by each resource.

    Lanes 0..15 parallelize the remote full-barrier arrivals::

        lane/rank      0 1 2 3 ... 15
        producer bit  O . O . ...  .
        action         arrive rank 0 and rank 2 in parallel

    The elected producer warp performs this once per stage. Allowing multiple
    producer warps to enter would over-arrive the same full barriers.
    """

    def producer_acquire(
        self,
        state,
        try_acquire_token: Boolean | None = None,
        *,
        expected_tx: Int32 | None = None,
        loc=None,
        ip=None,
    ) -> None:
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(
                state.index, state.phase, loc=loc, ip=ip
            ),
            loc=loc,
            ip=ip,
        )
        tx = self.sync_object_full.tx_count if expected_tx is None else expected_tx

        def arrive_owners() -> None:
            dst_rank = cute.arch.lane_idx(loc=loc, ip=ip)
            lane_mask = Int32(1) << dst_rank

            def arrive_owner() -> None:
                self.sync_object_full.arrive_and_expect_tx_with_dst(
                    state.index,
                    tx,
                    dst_rank,
                    loc=loc,
                    ip=ip,
                )

            # Match TRTLLM-gen's temporary custom TMA-mbarrier path: lanes
            # 0..15 map directly to cluster ranks and arm their selected
            # remote owner barriers in parallel.
            if_generate(
                (dst_rank < Int32(16)) & ((self.producer_mask & lane_mask) != Int32(0)),
                arrive_owner,
                loc=loc,
                ip=ip,
            )

        if_generate(self.is_leader_cta, arrive_owners, loc=loc, ip=ip)


def override_tma_umma_pipeline(
    cfg: cutlass.Constexpr[object],
    base,
    mask_kind: cutlass.Constexpr[str],
):
    """Return ``base`` with active-shape masks for one TMA-to-UMMA operand.

    Callers gate this override on physical multi-atom topology, independently
    of whether the operand uses extended payload multicast.  A local payload
    in the second or later UMMA atom still needs atom-relative full/empty
    barrier ownership. Fixed wider launches use compile-time dimensions;
    flexible launches query the active preferred/fallback shape.

    ``cga_tma_masks`` also maps the operand to the TRT-LLM-equivalent multicast
    direction: A/SFA use Row/physical-Y releases and B/SFB use
    Col/physical-X-atom releases. FlashInfer has no RowCol case because these
    resources own separate pipelines.
    """
    _, producer_mask, consumer_mask, consumer_arrival_count = cga_tma_masks(
        cfg, mask_kind
    )

    # The stock pipeline was materialized with the static 2x1 UMMA atom layout,
    # so its empty barrier expects one release. Wider physical CGAs route
    # releases from multiple 2-CTA atoms back to one producer-owner barrier:
    # dimY releases for A/SfA and dimX/2 releases for B/SfB. Reinitialize only
    # the empty array with that runtime count. This second init is ordered after
    # the stock init and before the kernel's common mbarrier fence/cluster sync.
    sync_object_empty = PipelineTmaUmma._make_sync_object(
        base.sync_object_empty.barrier_storage,
        base.num_stages,
        (
            pipeline.PipelineOp.TCGen05Mma,
            pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                consumer_arrival_count,
            ),
        ),
    )
    # ``MemoryResource.create_pipeline`` has already narrowed the base
    # pipeline's static even-X CTA predicate with TaskWarpLeader when the
    # resource aggregates multiple producer warps.  Physical mixed-CGA X is
    # required to be even, so the static 2x1 atom predicate remains exactly
    # the runtime even-X predicate.  Preserve it here instead of replacing it
    # and accidentally letting every producer warp arm the same barriers.
    is_leader_cta = base.is_leader_cta
    return CgaPipelineTmaUmma(
        base.sync_object_full,
        sync_object_empty,
        base.num_stages,
        producer_mask,
        consumer_mask,
        is_leader_cta,
        base.cta_group,
    )


@cute.jit
def umma_atom_mask() -> Int32:
    """Return the rank mask of the current adjacent 2-CTA UMMA atom.

    Supported wider/flexible shapes always have even X, so clearing bit zero
    never crosses a Y-row boundary and does not require active dimensions.
    """
    rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    atom_leader = rank & Int32(~1)
    return (Int32(1) << atom_leader) | (Int32(1) << (atom_leader + Int32(1)))


@cute.jit
def umma_atom_leader_rank() -> Int32:
    """Return the even physical rank that owns the current atom's barriers."""
    rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    return rank & Int32(~1)


@dataclass(frozen=True)
class CgaPipelineUmmaUmma(TSPipelineUmmaUmma):
    """UMMA-to-UMMA pipeline whose two sides signal the current atom."""

    def producer_commit(self, state, *, loc=None, ip=None) -> None:
        self.sync_object_full.arrive(
            state.index,
            umma_atom_mask(),
            self.cta_group,
            loc=loc,
            ip=ip,
        )

    def consumer_release(self, state, *, loc=None, ip=None) -> None:
        self.sync_object_empty.arrive(
            state.index,
            umma_atom_mask(),
            self.cta_group,
            loc=loc,
            ip=ip,
        )


@dataclass(frozen=True)
class CgaPipelineAsyncUmma(TSPipelineAsyncUmma):
    """Async-to-UMMA pipeline with an atom-relative UMMA release mask."""

    def producer_commit(self, state, *, loc=None, ip=None) -> None:
        if (
            self.producer_op == pipeline.PipelineOp.AsyncThread
            and self.cta_group is cute.nvgpu.tcgen05.CtaGroup.TWO
        ):
            # The stock TS shortcut clears SM100's peer-address bit.  That is
            # sufficient for a physical 2x1 cluster, but it does not encode
            # which later 2-CTA atom owns the destination in a wider CGA.
            # Route every async-thread arrival explicitly to this atom leader.
            self.sync_object_full.arrive(
                state.index,
                umma_atom_leader_rank(),
                loc=loc,
                ip=ip,
            )
        else:
            # AsyncLoad uses cp.async.mbarrier.arrive on each CTA's local full
            # barrier and must remain local.
            self.sync_object_full.arrive(
                state.index, self.producer_mask, loc=loc, ip=ip
            )

    def consumer_release(self, state, *, loc=None, ip=None) -> None:
        self.sync_object_empty.arrive(
            state.index,
            umma_atom_mask(),
            self.cta_group,
            loc=loc,
            ip=ip,
        )


@dataclass(frozen=True)
class CgaPipelineUmmaAsync(PipelineUmmaAsync):
    """UMMA-to-async pipeline with an atom-relative UMMA commit mask."""

    def producer_commit(self, state, *, loc=None, ip=None) -> None:
        self.sync_object_full.arrive(
            state.index,
            umma_atom_mask(),
            self.cta_group,
            loc=loc,
            ip=ip,
        )


def override_umma_atom_pipeline(base):
    """Replace only static peer operations that must follow the current atom.

    The stock wheel pipeline describes one logical 2x1 CTA layout, so its peer
    bits name ranks 0/1. In a wider physical CGA the same logical operation in
    ranks 2/3, 4/5, ... must add that atom's rank base. Each wrapper below
    changes only the UMMA-side commit/release operation; async-local operations
    and all pipeline storage/state are preserved.
    """
    if isinstance(base, TSPipelineUmmaUmma):
        return CgaPipelineUmmaUmma(
            base.sync_object_full,
            base.sync_object_empty,
            base.num_stages,
            base.producer_mask,
            base.consumer_mask,
            base.cta_group,
        )
    if isinstance(base, TSPipelineAsyncUmma):
        return CgaPipelineAsyncUmma(
            base.sync_object_full,
            base.sync_object_empty,
            base.num_stages,
            base.producer_mask,
            base.consumer_mask,
            base.cta_group,
            base.producer_op,
        )
    if isinstance(base, PipelineUmmaAsync):
        return CgaPipelineUmmaAsync(
            base.sync_object_full,
            base.sync_object_empty,
            base.num_stages,
            base.producer_mask,
            base.consumer_mask,
            base.cta_group,
        )
    return base
