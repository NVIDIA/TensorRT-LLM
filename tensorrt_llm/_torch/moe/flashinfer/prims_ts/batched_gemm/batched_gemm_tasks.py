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

"""Task factories for the BatchedGemm TS example.

Each factory captures the task schedule with ``@schedule`` and
returns a Task. Work bodies are in the resources; this file only wires
the schedule.
"""

from contextlib import contextmanager, nullcontext

import cutlass

from cutlass.experimental.task_scheduling.schedule_builder import (
    domain_loop,
    schedule,
    work_tile_loop,
)
from cutlass.experimental.task_scheduling.task import Task
from cutlass.experimental.task_scheduling.resources import MemoryResource, WorkQueue

from .batched_gemm_config import (
    BatchedGemmConfig,
    SfSmemToTmemCopy,
)
from .smem_misc_resources import BatchedGemmWorkQueue

Constexpr = cutlass.Constexpr


def _is_persistent(cfg: BatchedGemmConfig) -> bool:
    return cfg.is_persistent


def _call_named_aux_if_present(resource, method_name: str, use_named: bool) -> None:
    if use_named:
        getattr(resource, method_name)()


def _producer_commit_prefetch_depth(resource) -> int:
    depth = getattr(resource, "producer_commit_prefetch_depth", 0)
    if depth <= 0:
        return 0
    pipeline_cfg = resource.pipeline_config
    if pipeline_cfg is None or not pipeline_cfg.advance_on_acquire:
        raise ValueError(
            f"{resource.name} sets producer_commit_prefetch_depth={depth}, "
            "but its pipeline does not use advance_on_acquire."
        )
    return depth


def _persistent_tail(work_queue) -> None:
    """Persistent WorkQueue consumer tail entries required on every task.

    ClcFetchAsync has no useful non-blocking consumer try-wait path.
    Emits only fetch_next_work(), which performs wait, response decode, and
    release, so keep the TS schedule to those semantic operations.
    """
    work_queue.wait()
    work_queue.get_and_advance_work_tile()
    work_queue.release()


def _persistent_work_tile_loop(cfg, work_queue):
    if cutlass.const_expr(cfg.use_early_exit):
        # CUDA-graph early exit over-launches token CTAs.  Use skip_if to
        # skip inactive persistent work-tile iterations while preserving the
        # WorkQueue tail that advances CLC to the next tile.
        return work_tile_loop(
            work_queue, skip_if=BatchedGemmWorkQueue.should_skip_work_tile
        )
    return work_tile_loop(work_queue)


def _persistent_skippable(cfg, wtwl):
    if cutlass.const_expr(cfg.use_early_exit):
        return wtwl.skippable()
    return nullcontext()


@contextmanager
def _work_tile_schedule_loop(cfg, work_queue, work_tile_preheader=None):
    """Wrap setup once per persistent work tile, or once for static schedules."""
    if _is_persistent(cfg):
        with _persistent_work_tile_loop(cfg, work_queue) as wtwl:
            if work_tile_preheader is not None:
                work_tile_preheader()
            with _persistent_skippable(cfg, wtwl):
                yield
            _persistent_tail(work_queue)
    else:
        if work_tile_preheader is not None:
            work_tile_preheader()
        yield


@contextmanager
def _k_tile_schedule_loop(cfg, work_queue, num_k_tiles: int, domain_start: int = 0):
    """Wrap a task body in either the persistent work-tile loop or static K loop."""
    # Keep these scopes nested: the task-scheduling DSL captures their structure.
    with _work_tile_schedule_loop(cfg, work_queue):  # noqa: SIM117
        with domain_loop(domain_start, num_k_tiles, 1):
            yield


def _call_schedule_with_optional_work_queue(
    schedule_fn, is_persistent: bool, work_queue, *resources
):
    if is_persistent:
        return schedule_fn(*resources, work_queue)
    return schedule_fn(*resources)


def _pdl_wait_resources(pdl_wait_resource):
    return [pdl_wait_resource] if pdl_wait_resource is not None else []


def _pdl_launch_resources(pdl_launch_resource):
    return [pdl_launch_resource] if pdl_launch_resource is not None else []


# ---------------------------------------------------------------------------
# Load tasks
# ---------------------------------------------------------------------------


def create_load_a_task(
    cfg,
    gmem_a,
    smem_a,
    work_queue,
    num_k_tiles: int,
    work_throttle=None,
    pdl_wait_resource=None,
    pdl_launch_resource=None,
) -> Task:
    is_persistent = _is_persistent(cfg)
    has_work_throttle = is_persistent and work_throttle is not None
    has_pdl_wait = pdl_wait_resource is not None
    has_pdl_launch = pdl_launch_resource is not None
    has_prepare_gather_tile = hasattr(smem_a, "prepare_gather_tile")

    @schedule
    def load_a_schedule(gmem: MemoryResource, smem: MemoryResource, *extra) -> None:
        idx = 0
        throttle = extra[idx] if has_work_throttle else None
        idx += 1 if has_work_throttle else 0
        pdl_wait = extra[idx] if has_pdl_wait else None
        idx += 1 if has_pdl_wait else 0
        pdl_launch = extra[idx] if has_pdl_launch else None
        idx += 1 if has_pdl_launch else 0
        wq = extra[idx] if is_persistent else None
        if pdl_wait is not None:
            pdl_wait.wait_griddep()
        _ = gmem.init_coords_state()
        smem.init_load_state()
        with _work_tile_schedule_loop(cfg, wq):
            # The DSL captures these named outputs even though Python does not.
            coords = gmem.compute_a_coords_head()  # noqa: F841
            _call_named_aux_if_present(
                smem, "prepare_gather_tile", has_prepare_gather_tile
            )
            if throttle is not None:
                throttle.try_acquire()
                throttle.acquire()
                throttle.commit()
            with domain_loop(0, num_k_tiles, 1):
                coord_a_k, coord_a_mn, coord_a_l, expert_idx, mn_limit = (
                    gmem.compute_a_coords_loop()
                )
                smem.try_acquire()
                smem.acquire()
                smem.load_a_tile(
                    coord_a_k=coord_a_k,
                    coord_a_mn=coord_a_mn,
                    coord_a_l=coord_a_l,
                    expert_idx=expert_idx,
                    mn_limit=mn_limit,
                )
                smem.commit()
        if pdl_launch is not None:
            pdl_launch.launch_griddep()

    extra_resources = []
    if has_work_throttle:
        extra_resources.append(work_throttle)
    if pdl_wait_resource is not None:
        extra_resources.append(pdl_wait_resource)
    if pdl_launch_resource is not None:
        extra_resources.append(pdl_launch_resource)
    if is_persistent:
        extra_resources.append(work_queue)
    captured_schedule = load_a_schedule(gmem_a, smem_a, *extra_resources)
    return Task(
        src_resources=(
            [gmem_a]
            + _pdl_wait_resources(pdl_wait_resource)
            + ([work_queue] if cfg.is_persistent else [])
        ),
        dst_resources=(
            [smem_a]
            + ([work_throttle] if has_work_throttle else [])
            + _pdl_launch_resources(pdl_launch_resource)
        ),
        warp_idx=cfg.load_a_warp_idx,
        num_warps=cfg.num_load_a_warps,
        schedule=captured_schedule,
        num_registers=cfg.load_a_task_regs,
        name="LoadATask",
    )


def create_load_b_task(
    cfg,
    gmem_b,
    smem_b,
    work_queue,
    num_k_tiles: int,
    work_throttle=None,
    pdl_wait_resource=None,
    pdl_launch_resource=None,
) -> Task:
    is_persistent = _is_persistent(cfg)
    has_work_throttle = is_persistent and work_throttle is not None
    has_pdl_wait = pdl_wait_resource is not None
    has_pdl_launch = pdl_launch_resource is not None
    has_prepare_gather_tile = hasattr(smem_b, "prepare_gather_tile")

    @schedule
    def load_b_schedule(gmem: MemoryResource, smem: MemoryResource, *extra) -> None:
        idx = 0
        throttle = extra[idx] if has_work_throttle else None
        idx += 1 if has_work_throttle else 0
        pdl_wait = extra[idx] if has_pdl_wait else None
        idx += 1 if has_pdl_wait else 0
        pdl_launch = extra[idx] if has_pdl_launch else None
        idx += 1 if has_pdl_launch else 0
        wq = extra[idx] if is_persistent else None
        if pdl_wait is not None:
            pdl_wait.wait_griddep()
        _ = gmem.init_coords_state()
        smem.init_load_state()
        with _work_tile_schedule_loop(cfg, wq):
            # The DSL captures these named outputs even though Python does not.
            coords = gmem.compute_b_coords_head()  # noqa: F841
            _call_named_aux_if_present(
                smem, "prepare_gather_tile", has_prepare_gather_tile
            )
            if throttle is not None:
                throttle.try_acquire()
                throttle.acquire()
                throttle.commit()
            with domain_loop(0, num_k_tiles, 1):
                coord_b_k, coord_b_mn, coord_b_l, mn_limit = (
                    gmem.compute_b_coords_loop()
                )
                smem.try_acquire()
                smem.acquire()
                smem.load_b_tile(
                    coord_b_k=coord_b_k,
                    coord_b_mn=coord_b_mn,
                    coord_b_l=coord_b_l,
                    mn_limit=mn_limit,
                )
                smem.commit()
        if pdl_launch is not None:
            pdl_launch.launch_griddep()

    extra_resources = []
    if has_work_throttle:
        extra_resources.append(work_throttle)
    if pdl_wait_resource is not None:
        extra_resources.append(pdl_wait_resource)
    if pdl_launch_resource is not None:
        extra_resources.append(pdl_launch_resource)
    if is_persistent:
        extra_resources.append(work_queue)
    captured_schedule = load_b_schedule(gmem_b, smem_b, *extra_resources)
    return Task(
        src_resources=(
            [gmem_b]
            + _pdl_wait_resources(pdl_wait_resource)
            + ([work_queue] if cfg.is_persistent else [])
        ),
        dst_resources=(
            [smem_b]
            + ([work_throttle] if has_work_throttle else [])
            + _pdl_launch_resources(pdl_launch_resource)
        ),
        warp_idx=cfg.load_b_warp_idx,
        num_warps=cfg.num_load_b_warps,
        schedule=captured_schedule,
        num_registers=cfg.load_b_task_regs,
        name="LoadBTask",
    )


def create_gather_task(
    cfg,
    gmem_act,
    smem_gather,
    work_queue,
    num_k_tiles: int,
    pdl_wait_resource=None,
    pdl_launch_resource=None,
) -> Task:
    """Gather task: LDGSTS load of routed activations into SMEM.

    Replaces LoadA (non-swapAB) or LoadB (swapAB) when activations need
    routing via route map.
    """
    is_persistent = _is_persistent(cfg)
    has_pdl_wait = pdl_wait_resource is not None
    has_pdl_launch = pdl_launch_resource is not None
    has_prepare_gather_tile = hasattr(smem_gather, "prepare_gather_tile")

    def _gather_gmem_consumer(gmem, *, head: bool):
        if head:
            if cfg.is_swap_ab:
                return gmem.compute_b_coords_head()
            return gmem.compute_a_coords_head()
        if cfg.is_swap_ab:
            return gmem.compute_b_coords_loop()
        return gmem.compute_a_coords_loop()

    def _gather_smem_producer(smem, coords):
        if cfg.is_swap_ab:
            coord_b_k, coord_b_mn, coord_b_l, mn_limit = coords
            smem.load_b_tile(
                coord_b_k=coord_b_k,
                coord_b_mn=coord_b_mn,
                coord_b_l=coord_b_l,
                mn_limit=mn_limit,
            )
        else:
            coord_a_k, coord_a_mn, coord_a_l, expert_idx, mn_limit = coords
            smem.load_a_tile(
                coord_a_k=coord_a_k,
                coord_a_mn=coord_a_mn,
                coord_a_l=coord_a_l,
                expert_idx=expert_idx,
                mn_limit=mn_limit,
            )

    @schedule
    def gather_schedule(gmem: MemoryResource, smem: MemoryResource, *extra) -> None:
        idx = 0
        pdl_wait = extra[idx] if has_pdl_wait else None
        idx += 1 if has_pdl_wait else 0
        pdl_launch = extra[idx] if has_pdl_launch else None
        idx += 1 if has_pdl_launch else 0
        wq = extra[idx] if is_persistent else None
        if pdl_wait is not None:
            pdl_wait.wait_griddep()
        _ = gmem.init_coords_state()
        smem.init_load_state()
        with _work_tile_schedule_loop(cfg, wq):
            coords = _gather_gmem_consumer(gmem, head=True)
            _call_named_aux_if_present(
                smem, "prepare_gather_tile", has_prepare_gather_tile
            )
            with domain_loop(0, num_k_tiles, 1):
                coords = _gather_gmem_consumer(gmem, head=False)
                smem.try_acquire()
                smem.acquire()
                _gather_smem_producer(smem, coords)
                smem.commit()
        if pdl_launch is not None:
            pdl_launch.launch_griddep()

    extra_resources = []
    if pdl_wait_resource is not None:
        extra_resources.append(pdl_wait_resource)
    if pdl_launch_resource is not None:
        extra_resources.append(pdl_launch_resource)
    if is_persistent:
        extra_resources.append(work_queue)
    captured_schedule = gather_schedule(gmem_act, smem_gather, *extra_resources)
    return Task(
        src_resources=(
            [gmem_act]
            + _pdl_wait_resources(pdl_wait_resource)
            + ([work_queue] if cfg.is_persistent else [])
        ),
        dst_resources=[smem_gather] + _pdl_launch_resources(pdl_launch_resource),
        warp_idx=cfg.gather_warp_idx,
        num_warps=cfg.num_gather_warps,
        schedule=captured_schedule,
        num_registers=cfg.gather_regs,
        name="GatherTask",
    )


def create_fused_load_a_sfa_task(
    cfg,
    gmem_a,
    smem_a,
    gmem_sfa,
    smem_sfa,
    work_queue,
    num_k_tiles: int,
    work_throttle=None,
) -> Task:
    """Load the weight operand and its scale factors on the same warp.

    This is the swap-AB high-throughput schedule used by TRT-LLM Gen: one warp
    issues both TMA streams before committing either pipeline.
    """
    is_persistent = _is_persistent(cfg)
    has_work_throttle = is_persistent and work_throttle is not None
    has_prepare_a_tile = hasattr(smem_a, "prepare_gather_tile")
    has_prepare_sfa_tile = hasattr(smem_sfa, "prepare_sfa_tile")

    @schedule
    def fused_load_a_sfa_schedule(
        gmem_a_res: MemoryResource,
        smem_a_res: MemoryResource,
        gmem_sfa_res: MemoryResource,
        smem_sfa_res: MemoryResource,
        *extra,
    ) -> None:
        idx = 0
        throttle = extra[idx] if has_work_throttle else None
        idx += 1 if has_work_throttle else 0
        wq = extra[idx] if is_persistent else None

        _ = gmem_a_res.init_coords_state()
        _ = gmem_sfa_res.init_coords_state()
        smem_a_res.init_load_state()
        smem_sfa_res.init_load_state()
        with _work_tile_schedule_loop(cfg, wq):
            _ = gmem_a_res.compute_a_coords_head()
            _call_named_aux_if_present(
                smem_a_res, "prepare_gather_tile", has_prepare_a_tile
            )
            _call_named_aux_if_present(
                smem_sfa_res, "prepare_sfa_tile", has_prepare_sfa_tile
            )
            if throttle is not None:
                throttle.try_acquire()
                throttle.acquire()
                throttle.commit()
            with domain_loop(0, num_k_tiles, 1):
                coord_a_k, coord_a_mn, coord_a_l, expert_idx, mn_limit = (
                    gmem_a_res.compute_a_coords_loop()
                )
                coord_sfa_k, coord_sfa_mn = gmem_sfa_res.compute_sfa_coords_loop()
                smem_a_res.try_acquire()
                smem_sfa_res.try_acquire()
                smem_a_res.acquire()
                smem_sfa_res.acquire()
                smem_a_res.load_a_tile(
                    coord_a_k=coord_a_k,
                    coord_a_mn=coord_a_mn,
                    coord_a_l=coord_a_l,
                    expert_idx=expert_idx,
                    mn_limit=mn_limit,
                )
                smem_sfa_res.load_sfa_tile(
                    coord_sfa_k=coord_sfa_k,
                    coord_sfa_mn=coord_sfa_mn,
                )
                smem_a_res.commit()
                smem_sfa_res.commit()

    extra_resources = []
    if has_work_throttle:
        extra_resources.append(work_throttle)
    if is_persistent:
        extra_resources.append(work_queue)
    captured_schedule = fused_load_a_sfa_schedule(
        gmem_a,
        smem_a,
        gmem_sfa,
        smem_sfa,
        *extra_resources,
    )
    return Task(
        src_resources=[gmem_a, gmem_sfa] + ([work_queue] if cfg.is_persistent else []),
        dst_resources=[smem_a, smem_sfa]
        + ([work_throttle] if has_work_throttle else []),
        warp_idx=cfg.load_a_warp_idx,
        num_warps=cfg.num_load_a_warps,
        schedule=captured_schedule,
        num_registers=max(cfg.load_a_task_regs, cfg.load_sfa_task_regs),
        name="FusedLoadASfATask",
    )


def create_fused_gather_sfb_task(
    cfg,
    gmem_b,
    smem_b,
    gmem_sfb,
    smem_sfb,
    work_queue,
    num_k_tiles: int,
    pdl_wait_resource=None,
    pdl_launch_resource=None,
) -> Task:
    """Gather routed activations and their SFs with one LDGSTS warpgroup.

    Both resources advance their acquire cursor ahead of the commit cursor.
    The SFB copy closes the shared cp.async group, then its drain publishes
    both barriers together after the same two-stage prefetch window as Gen.
    """
    is_persistent = _is_persistent(cfg)
    has_pdl_wait = pdl_wait_resource is not None
    has_pdl_launch = pdl_launch_resource is not None
    has_prepare_gather_tile = hasattr(smem_b, "prepare_gather_tile")
    has_prepare_sfb_tile = hasattr(smem_sfb, "prepare_sfb_tile")
    prefetch_depth = _producer_commit_prefetch_depth(smem_sfb)

    @schedule
    def fused_gather_sfb_schedule(
        gmem_b_res: MemoryResource,
        smem_b_res: MemoryResource,
        gmem_sfb_res: MemoryResource,
        smem_sfb_res: MemoryResource,
        *extra,
    ) -> None:
        idx = 0
        pdl_wait = extra[idx] if has_pdl_wait else None
        idx += 1 if has_pdl_wait else 0
        pdl_launch = extra[idx] if has_pdl_launch else None
        idx += 1 if has_pdl_launch else 0
        wq = extra[idx] if is_persistent else None
        if pdl_wait is not None:
            pdl_wait.wait_griddep()

        _ = gmem_b_res.init_coords_state()
        _ = gmem_sfb_res.init_coords_state()
        smem_b_res.init_load_state()
        smem_sfb_res.init_load_state()
        with _work_tile_schedule_loop(cfg, wq):
            _ = gmem_b_res.compute_b_coords_head()
            _ = gmem_sfb_res.init_tile_state()
            _call_named_aux_if_present(
                smem_b_res, "prepare_gather_tile", has_prepare_gather_tile
            )
            _call_named_aux_if_present(
                smem_sfb_res, "prepare_sfb_tile", has_prepare_sfb_tile
            )
            if prefetch_depth > 0:
                for prefetch_idx in range(prefetch_depth):
                    coord_b_k, coord_b_mn, coord_b_l, mn_limit = (
                        gmem_b_res.compute_b_coords_prefetch(prefetch_idx=prefetch_idx)
                    )
                    coord_sfb_k, coord_sfb_mn = gmem_sfb_res.compute_sfb_coords_head(
                        prefetch_idx=prefetch_idx + 1
                    )
                    smem_b_res.try_acquire()
                    smem_sfb_res.try_acquire()
                    smem_b_res.acquire()
                    smem_sfb_res.acquire()
                    smem_b_res.load_b_tile(
                        coord_b_k=coord_b_k,
                        coord_b_mn=coord_b_mn,
                        coord_b_l=coord_b_l,
                        mn_limit=mn_limit,
                    )
                    smem_sfb_res.load_sfb_tile(
                        coord_sfb_k=coord_sfb_k,
                        coord_sfb_mn=coord_sfb_mn,
                    )
                with domain_loop(prefetch_depth, num_k_tiles, 1):
                    smem_sfb_res.drain_loop()
                    smem_b_res.commit()
                    smem_sfb_res.commit()
                    coord_b_k, coord_b_mn, coord_b_l, mn_limit = (
                        gmem_b_res.compute_b_coords_loop()
                    )
                    coord_sfb_k, coord_sfb_mn = gmem_sfb_res.compute_sfb_coords_loop()
                    smem_b_res.try_acquire()
                    smem_sfb_res.try_acquire()
                    smem_b_res.acquire()
                    smem_sfb_res.acquire()
                    smem_b_res.load_b_tile(
                        coord_b_k=coord_b_k,
                        coord_b_mn=coord_b_mn,
                        coord_b_l=coord_b_l,
                        mn_limit=mn_limit,
                    )
                    smem_sfb_res.load_sfb_tile(
                        coord_sfb_k=coord_sfb_k,
                        coord_sfb_mn=coord_sfb_mn,
                    )
                for prefetch_idx in range(prefetch_depth):
                    smem_sfb_res.drain_tail(prefetch_idx=prefetch_idx)
                    smem_b_res.commit()
                    smem_sfb_res.commit()
            else:
                with domain_loop(0, num_k_tiles, 1):
                    coord_b_k, coord_b_mn, coord_b_l, mn_limit = (
                        gmem_b_res.compute_b_coords_loop()
                    )
                    coord_sfb_k, coord_sfb_mn = gmem_sfb_res.compute_sfb_coords_loop()
                    smem_b_res.try_acquire()
                    smem_sfb_res.try_acquire()
                    smem_b_res.acquire()
                    smem_sfb_res.acquire()
                    smem_b_res.load_b_tile(
                        coord_b_k=coord_b_k,
                        coord_b_mn=coord_b_mn,
                        coord_b_l=coord_b_l,
                        mn_limit=mn_limit,
                    )
                    smem_sfb_res.load_sfb_tile(
                        coord_sfb_k=coord_sfb_k,
                        coord_sfb_mn=coord_sfb_mn,
                    )
                    smem_sfb_res.drain_loop()
                    smem_b_res.commit()
                    smem_sfb_res.commit()
        if pdl_launch is not None:
            pdl_launch.launch_griddep()

    extra_resources = []
    if pdl_wait_resource is not None:
        extra_resources.append(pdl_wait_resource)
    if pdl_launch_resource is not None:
        extra_resources.append(pdl_launch_resource)
    if is_persistent:
        extra_resources.append(work_queue)
    captured_schedule = fused_gather_sfb_schedule(
        gmem_b,
        smem_b,
        gmem_sfb,
        smem_sfb,
        *extra_resources,
    )
    return Task(
        src_resources=(
            [gmem_b, gmem_sfb]
            + _pdl_wait_resources(pdl_wait_resource)
            + ([work_queue] if cfg.is_persistent else [])
        ),
        dst_resources=[smem_b, smem_sfb] + _pdl_launch_resources(pdl_launch_resource),
        warp_idx=cfg.gather_warp_idx,
        num_warps=cfg.num_gather_warps,
        schedule=captured_schedule,
        num_registers=max(cfg.gather_regs, cfg.load_sfb_task_regs),
        name="FusedGatherSfBTask",
    )


def create_sync_task(
    cfg,
    proxy_cluster,
    gather_smem,
    tma_smem,
    work_queue,
    num_k_tiles: int,
    sync_warp_idx: int = 7,
) -> Task:
    """Cross-CTA sync task (2-CTA only).

    Observes data readiness on gather and TMA SMEM resources, then
    cross-CTA arrives on the proxy barrier. MMA waits on the proxy
    as a single synchronization point for all SMEM data.
    """
    is_persistent = _is_persistent(cfg)

    @schedule
    def sync_schedule(
        proxy: MemoryResource,
        gather: MemoryResource,
        tma: MemoryResource,
        wq: WorkQueue = None,
    ) -> None:
        with _k_tile_schedule_loop(cfg, wq, num_k_tiles):
            gather.try_wait()
            gather.wait()
            tma.try_wait()
            tma.wait()
            proxy.try_acquire()
            proxy.acquire()
            proxy.producer_work()
            proxy.commit()

    captured_schedule = _call_schedule_with_optional_work_queue(
        sync_schedule,
        is_persistent,
        work_queue,
        proxy_cluster,
        gather_smem,
        tma_smem,
    )
    return Task(
        src_resources=[gather_smem, tma_smem]
        + ([work_queue] if cfg.is_persistent else []),
        dst_resources=[proxy_cluster],
        warp_idx=sync_warp_idx,
        num_warps=1,
        schedule=captured_schedule,
        num_registers=cfg.sync_regs,
        name="SyncTask",
    )


def create_load_sfa_task(
    cfg,
    gmem_sfa,
    smem_sfa,
    work_queue,
    num_k_tiles: int,
    pdl_wait_resource=None,
    pdl_launch_resource=None,
) -> Task:
    is_persistent = _is_persistent(cfg)
    has_pdl_wait = pdl_wait_resource is not None
    has_pdl_launch = pdl_launch_resource is not None
    has_prepare_sfa_tile = hasattr(smem_sfa, "prepare_sfa_tile")
    needs_drain = (
        cfg.has_routed_sfs
        and cfg.uses_ldgsts_routed_sfs
        and hasattr(smem_sfa, "drain_loop")
    )
    prefetch_depth = _producer_commit_prefetch_depth(smem_sfa)

    @schedule
    def load_sfa_schedule(gmem: MemoryResource, smem: MemoryResource, *extra) -> None:
        idx = 0
        pdl_wait = extra[idx] if has_pdl_wait else None
        idx += 1 if has_pdl_wait else 0
        pdl_launch = extra[idx] if has_pdl_launch else None
        idx += 1 if has_pdl_launch else 0
        wq = extra[idx] if is_persistent else None
        if pdl_wait is not None:
            pdl_wait.wait_griddep()
        _ = gmem.init_coords_state()
        smem.init_load_state()
        with _work_tile_schedule_loop(cfg, wq):
            _call_named_aux_if_present(smem, "prepare_sfa_tile", has_prepare_sfa_tile)
            if prefetch_depth > 0:
                for prefetch_idx in range(prefetch_depth):
                    coord_sfa_k, coord_sfa_mn = gmem.compute_sfa_coords_head(
                        prefetch_idx=prefetch_idx
                    )
                    smem.try_acquire()
                    smem.acquire()
                    smem.load_sfa_tile(
                        coord_sfa_k=coord_sfa_k,
                        coord_sfa_mn=coord_sfa_mn,
                    )
                with domain_loop(prefetch_depth, num_k_tiles, 1):
                    if needs_drain:
                        smem.drain_loop()
                    smem.commit()
                    coord_sfa_k, coord_sfa_mn = gmem.compute_sfa_coords_loop()
                    smem.try_acquire()
                    smem.acquire()
                    smem.load_sfa_tile(
                        coord_sfa_k=coord_sfa_k,
                        coord_sfa_mn=coord_sfa_mn,
                    )
                for prefetch_idx in range(prefetch_depth):
                    if needs_drain:
                        smem.drain_tail(prefetch_idx=prefetch_idx)
                    smem.commit()
            else:
                with domain_loop(0, num_k_tiles, 1):
                    coord_sfa_k, coord_sfa_mn = gmem.compute_sfa_coords_loop()
                    smem.try_acquire()
                    smem.acquire()
                    smem.load_sfa_tile(
                        coord_sfa_k=coord_sfa_k,
                        coord_sfa_mn=coord_sfa_mn,
                    )
                    if needs_drain:
                        smem.drain_loop()
                    smem.commit()
        if pdl_launch is not None:
            pdl_launch.launch_griddep()

    extra_resources = []
    if pdl_wait_resource is not None:
        extra_resources.append(pdl_wait_resource)
    if pdl_launch_resource is not None:
        extra_resources.append(pdl_launch_resource)
    if is_persistent:
        extra_resources.append(work_queue)
    captured_schedule = load_sfa_schedule(gmem_sfa, smem_sfa, *extra_resources)
    return Task(
        src_resources=(
            [gmem_sfa]
            + _pdl_wait_resources(pdl_wait_resource)
            + ([work_queue] if cfg.is_persistent else [])
        ),
        dst_resources=[smem_sfa] + _pdl_launch_resources(pdl_launch_resource),
        warp_idx=cfg.load_sfa_warp_idx,
        num_warps=cfg.num_load_sfa_warps,
        schedule=captured_schedule,
        num_registers=cfg.load_sfa_task_regs,
        name="LoadSfATask",
    )


def create_load_sfb_task(
    cfg,
    gmem_sfb,
    smem_sfb,
    work_queue,
    num_k_tiles: int,
    pdl_wait_resource=None,
    pdl_launch_resource=None,
) -> Task:
    is_persistent = _is_persistent(cfg)
    has_pdl_wait = pdl_wait_resource is not None
    has_pdl_launch = pdl_launch_resource is not None
    has_prepare_sfb_tile = hasattr(smem_sfb, "prepare_sfb_tile")
    needs_drain = (
        cfg.has_routed_sfs
        and cfg.uses_ldgsts_routed_sfs
        and hasattr(smem_sfb, "drain_loop")
    )
    prefetch_depth = _producer_commit_prefetch_depth(smem_sfb)

    @schedule
    def load_sfb_schedule(gmem: MemoryResource, smem: MemoryResource, *extra) -> None:
        idx = 0
        pdl_wait = extra[idx] if has_pdl_wait else None
        idx += 1 if has_pdl_wait else 0
        pdl_launch = extra[idx] if has_pdl_launch else None
        idx += 1 if has_pdl_launch else 0
        wq = extra[idx] if is_persistent else None
        if pdl_wait is not None:
            pdl_wait.wait_griddep()
        _ = gmem.init_coords_state()
        smem.init_load_state()
        with _work_tile_schedule_loop(cfg, wq):
            _ = gmem.init_tile_state()
            _call_named_aux_if_present(smem, "prepare_sfb_tile", has_prepare_sfb_tile)
            if prefetch_depth > 0:
                for prefetch_idx in range(prefetch_depth):
                    coord_sfb_k, coord_sfb_mn = gmem.compute_sfb_coords_head(
                        prefetch_idx=prefetch_idx + 1
                    )
                    smem.try_acquire()
                    smem.acquire()
                    smem.load_sfb_tile(
                        coord_sfb_k=coord_sfb_k,
                        coord_sfb_mn=coord_sfb_mn,
                    )
                with domain_loop(prefetch_depth, num_k_tiles, 1):
                    if needs_drain:
                        smem.drain_loop()
                    smem.commit()
                    coord_sfb_k, coord_sfb_mn = gmem.compute_sfb_coords_loop()
                    smem.try_acquire()
                    smem.acquire()
                    smem.load_sfb_tile(
                        coord_sfb_k=coord_sfb_k,
                        coord_sfb_mn=coord_sfb_mn,
                    )
                for prefetch_idx in range(prefetch_depth):
                    if needs_drain:
                        smem.drain_tail(prefetch_idx=prefetch_idx)
                    smem.commit()
            else:
                with domain_loop(0, num_k_tiles, 1):
                    coord_sfb_k, coord_sfb_mn = gmem.compute_sfb_coords_loop()
                    smem.try_acquire()
                    smem.acquire()
                    smem.load_sfb_tile(
                        coord_sfb_k=coord_sfb_k,
                        coord_sfb_mn=coord_sfb_mn,
                    )
                    if needs_drain:
                        smem.drain_loop()
                    smem.commit()
        if pdl_launch is not None:
            pdl_launch.launch_griddep()

    extra_resources = []
    if pdl_wait_resource is not None:
        extra_resources.append(pdl_wait_resource)
    if pdl_launch_resource is not None:
        extra_resources.append(pdl_launch_resource)
    if is_persistent:
        extra_resources.append(work_queue)
    captured_schedule = load_sfb_schedule(gmem_sfb, smem_sfb, *extra_resources)
    return Task(
        src_resources=(
            [gmem_sfb]
            + _pdl_wait_resources(pdl_wait_resource)
            + ([work_queue] if cfg.is_persistent else [])
        ),
        dst_resources=[smem_sfb] + _pdl_launch_resources(pdl_launch_resource),
        warp_idx=cfg.load_sfb_warp_idx,
        num_warps=cfg.num_load_sfb_warps,
        schedule=captured_schedule,
        num_registers=cfg.load_sfb_task_regs,
        name="LoadSfBTask",
    )


# ---------------------------------------------------------------------------
# Copy SF tasks (LDS+STTM, only when not UTCCP-fused)
# ---------------------------------------------------------------------------


def create_copy_sfa_task(
    cfg,
    smem_sfa,
    tmem_sfa,
    work_queue,
    num_k_tiles: int,
) -> Task:
    is_persistent = _is_persistent(cfg)

    @schedule
    def copy_sfa_schedule(
        smem: MemoryResource, tmem: MemoryResource, wq: WorkQueue = None
    ) -> None:
        _ = smem.init_s2t_state()
        tmem.init_copy_state()
        with _k_tile_schedule_loop(cfg, wq, num_k_tiles):
            smem.try_wait()
            smem.wait()
            desc_a_s2t_base, smem_sfa_stage_ptr = smem.build_sfa_s2t_desc()
            tmem.acquire()
            tmem.copy_sfa(
                desc_a_s2t_base=desc_a_s2t_base,
                smem_sfa_stage_ptr=smem_sfa_stage_ptr,
            )
            tmem.commit()
            smem.release()

    captured_schedule = _call_schedule_with_optional_work_queue(
        copy_sfa_schedule, is_persistent, work_queue, smem_sfa, tmem_sfa
    )
    return Task(
        src_resources=[smem_sfa] + ([work_queue] if cfg.is_persistent else []),
        dst_resources=[tmem_sfa],
        warp_idx=cfg.copy_sfa_warp_idx,
        num_warps=cfg.num_copy_sfa_warps,
        schedule=captured_schedule,
        num_registers=cfg.copy_sfa_task_regs,
        name="CopySfATask",
        run_only_on_cta_id=0 if cfg.has_cluster else None,
    )


def create_copy_sfb_task(
    cfg,
    smem_sfb,
    tmem_sfb,
    work_queue,
    num_k_tiles: int,
) -> Task:
    is_persistent = _is_persistent(cfg)

    @schedule
    def copy_sfb_schedule(
        smem: MemoryResource, tmem: MemoryResource, wq: WorkQueue = None
    ) -> None:
        _ = smem.init_s2t_state()
        tmem.init_copy_state()

        def issue_sfb_copy() -> None:
            smem.try_wait()
            smem.wait()
            desc_b_s2t_base = smem.build_sfb_s2t_desc()
            tmem.acquire()
            tmem.copy_sfb(desc_b_s2t_base=desc_b_s2t_base)

        with _k_tile_schedule_loop(cfg, wq, num_k_tiles):
            issue_sfb_copy()
            tmem.commit()
            smem.release()

    captured_schedule = _call_schedule_with_optional_work_queue(
        copy_sfb_schedule, is_persistent, work_queue, smem_sfb, tmem_sfb
    )
    return Task(
        src_resources=[smem_sfb] + ([work_queue] if cfg.is_persistent else []),
        dst_resources=[tmem_sfb],
        warp_idx=cfg.copy_sfb_warp_idx,
        num_warps=cfg.num_copy_sfb_warps,
        schedule=captured_schedule,
        num_registers=cfg.copy_sfb_task_regs,
        name="CopySfBTask",
        run_only_on_cta_id=0
        if (
            cfg.has_cluster
            and cfg.sfb_smem_to_tmem_copy != int(SfSmemToTmemCopy.LDS_STTM)
            and not (cfg.has_routed_sfs and cfg.uses_tma_routed_sfs and cfg.is_swap_ab)
        )
        else None,
    )


def create_copy_sfab_task(
    cfg,
    smem_sfa,
    smem_sfb,
    tmem_sfab,
    work_queue,
    num_k_tiles: int,
) -> Task:
    is_persistent = _is_persistent(cfg)

    @schedule
    def copy_sfab_schedule(
        smem_a: MemoryResource,
        smem_b: MemoryResource,
        tmem: MemoryResource,
        wq: WorkQueue = None,
    ) -> None:
        _ = smem_a.init_s2t_state()
        _ = smem_b.init_s2t_state()
        tmem.init_copy_state()

        def issue_sfab_copy():
            smem_a.try_wait()
            smem_b.try_wait()
            smem_a.wait()
            smem_b.wait()
            desc_a_s2t_base, smem_sfa_stage_ptr = smem_a.build_sfa_s2t_desc()
            desc_b_s2t_base = smem_b.build_sfb_s2t_desc()
            tmem.acquire()
            tmem.copy_sfab(
                desc_a_s2t_base=desc_a_s2t_base,
                smem_sfa_stage_ptr=smem_sfa_stage_ptr,
                desc_b_s2t_base=desc_b_s2t_base,
            )

        def commit_previous_sfab_copy():
            tmem.commit()
            smem_a.release()
            smem_b.release()

        with _work_tile_schedule_loop(cfg, wq):
            issue_sfab_copy()
            with domain_loop(1, num_k_tiles, 1):
                commit_previous_sfab_copy()
                issue_sfab_copy()
            commit_previous_sfab_copy()

    captured_schedule = _call_schedule_with_optional_work_queue(
        copy_sfab_schedule,
        is_persistent,
        work_queue,
        smem_sfa,
        smem_sfb,
        tmem_sfab,
    )
    return Task(
        src_resources=[smem_sfa, smem_sfb]
        + ([work_queue] if cfg.is_persistent else []),
        dst_resources=[tmem_sfab],
        warp_idx=cfg.copy_sfa_warp_idx,
        num_warps=cfg.num_copy_sfa_warps,
        schedule=captured_schedule,
        num_registers=cfg.copy_sfa_task_regs,
        name="CopySfAbTask",
        run_only_on_cta_id=0 if cfg.has_cluster else None,
    )


# ---------------------------------------------------------------------------
# CastA task (MXFP4 A -> BF16 TMEM)
# ---------------------------------------------------------------------------


def create_cast_a_task(
    cfg,
    smem_a,
    smem_sfa,
    tmem_cast_a,
    work_queue,
    num_k_tiles: int,
) -> Task:
    is_persistent = _is_persistent(cfg)

    @schedule
    def cast_a_schedule(
        smem: MemoryResource,
        smem_sf: MemoryResource,
        tmem: MemoryResource,
        wq: WorkQueue = None,
    ) -> None:
        _ = smem.init_mma_state()
        _ = smem_sf.init_s2t_state()
        tmem.init_copy_state()
        with _k_tile_schedule_loop(cfg, wq, num_k_tiles):
            smem.try_wait()
            smem_sf.try_wait()
            smem.wait()
            smem_sf.wait()
            desc_a = smem.build_mma_desc_a()
            desc_sf = smem_sf.build_sfa_s2t_desc()
            tmem.try_acquire()
            tmem.acquire()
            tmem.cast_a(
                smem_a_stage_ptr=desc_a[1],
                smem_sfa_stage_ptr=desc_sf[1],
            )
            tmem.commit()
            tmem.sync_cast_a_warps()
            smem.release()
            smem_sf.release()

    captured_schedule = _call_schedule_with_optional_work_queue(
        cast_a_schedule,
        is_persistent,
        work_queue,
        smem_a,
        smem_sfa,
        tmem_cast_a,
    )
    return Task(
        src_resources=[smem_a, smem_sfa] + ([work_queue] if cfg.is_persistent else []),
        dst_resources=[tmem_cast_a],
        warp_idx=cfg.cast_a_warp_idx_first,
        num_warps=cfg.num_cast_a_warps,
        schedule=captured_schedule,
        num_registers=cfg.cast_a_regs,
        name="CastATask",
    )


# ---------------------------------------------------------------------------
# MMA task
# ---------------------------------------------------------------------------


def create_mma_task(
    cfg,
    smem_a,
    smem_b,
    smem_sfa,
    smem_sfb,
    tmem_c,
    work_queue,
    num_k_tiles: int,
    proxy_cluster=None,
    tmem_sfa=None,
    tmem_sfb=None,
    tmem_sfab=None,
    tmem_cast_a=None,
) -> Task:
    """MMA task.

    smem_sfa/smem_sfb: when not None, MMA does fused S2T+MMA (consumes SmemSf).
    tmem_sfa/tmem_sfb: when not None, MMA consumes TmemSf (SF already in TMEM
      from separate CopySf tasks — used when SF is routed).

    Only one of (smem_sfa, tmem_sfa) should be set for each operand.
    """
    if (smem_sfa is None) != (smem_sfb is None):
        raise ValueError("create_mma_task requires smem_sfa and smem_sfb together")
    if (tmem_sfa is None) != (tmem_sfb is None):
        raise ValueError("create_mma_task requires tmem_sfa and tmem_sfb together")

    has_fused_sf = smem_sfa is not None
    has_separate_sf = tmem_sfa is not None
    has_combined_sf = tmem_sfab is not None
    has_cast_a = tmem_cast_a is not None
    active_sf_modes = [
        name
        for name, enabled in (
            ("fused", has_fused_sf),
            ("separate", has_separate_sf),
            ("combined", has_combined_sf),
        )
        if enabled
    ]
    if len(active_sf_modes) > 1:
        raise ValueError(
            "create_mma_task accepts at most one SF mode; got "
            + ", ".join(active_sf_modes)
        )
    if has_cast_a and active_sf_modes:
        raise ValueError("create_mma_task forbids tmem_cast_a with SF modes")
    has_proxy = proxy_cluster is not None
    is_persistent = _is_persistent(cfg)

    @schedule
    def mma_schedule(
        smem_a_res: MemoryResource,
        smem_b_res: MemoryResource,
        tmem_c_res: MemoryResource,
        *extra,
    ) -> None:
        idx = 0
        proxy = extra[idx] if has_proxy else None
        idx += 1 if has_proxy else 0
        smem_sfa_res = extra[idx] if has_fused_sf else None
        idx += 1 if has_fused_sf else 0
        smem_sfb_res = extra[idx] if has_fused_sf else None
        idx += 1 if has_fused_sf else 0
        tmem_sfa_res = extra[idx] if has_separate_sf else None
        idx += 1 if has_separate_sf else 0
        tmem_sfb_res = extra[idx] if has_separate_sf else None
        idx += 1 if has_separate_sf else 0
        tmem_sfab_res = extra[idx] if has_combined_sf else None
        idx += 1 if has_combined_sf else 0
        tmem_cast_a_res = extra[idx] if has_cast_a else None
        idx += 1 if has_cast_a else 0
        wq = extra[idx] if is_persistent else None

        if not has_cast_a:
            _ = smem_a_res.init_mma_state()
        _ = smem_b_res.init_mma_state()
        tmem_c_res.init_accumulator_state()
        if proxy is not None:
            pass
        if smem_sfa_res is not None:
            _ = smem_sfa_res.init_s2t_state()
        if smem_sfb_res is not None:
            _ = smem_sfb_res.init_s2t_state()
        if tmem_sfa_res is not None:
            _ = tmem_sfa_res.init_mma_state()
        if tmem_sfb_res is not None:
            _ = tmem_sfb_res.init_mma_state()
        if tmem_sfab_res is not None:
            _ = tmem_sfab_res.init_mma_state()
        if tmem_cast_a_res is not None:
            _ = tmem_cast_a_res.init_mma_state()

        with _work_tile_schedule_loop(cfg, wq):
            if cfg.has_deepseek_fp8:
                with domain_loop(0, num_k_tiles, 1):
                    tmem_c_res.try_acquire()
                    tmem_c_res.acquire()
                    _mma_loop_body(
                        smem_a_res,
                        smem_b_res,
                        tmem_c_res,
                        proxy,
                        smem_sfa_res,
                        smem_sfb_res,
                        tmem_sfa_res,
                        tmem_sfb_res,
                        tmem_sfab_res,
                        tmem_cast_a_res,
                        release_sources=False,
                    )
                    tmem_c_res.commit()
                    _release_mma_sources(
                        smem_a_res,
                        smem_b_res,
                        proxy,
                        smem_sfa_res,
                        smem_sfb_res,
                        tmem_sfa_res,
                        tmem_sfb_res,
                        tmem_sfab_res,
                        tmem_cast_a_res,
                    )
            else:
                tmem_c_res.try_acquire()
                tmem_c_res.acquire()
                if getattr(cfg, "use_unroll_loop_2x_for_mma", 0):
                    with domain_loop(0, num_k_tiles, 1, unroll=2):
                        _mma_loop_body(
                            smem_a_res,
                            smem_b_res,
                            tmem_c_res,
                            proxy,
                            smem_sfa_res,
                            smem_sfb_res,
                            tmem_sfa_res,
                            tmem_sfb_res,
                            tmem_sfab_res,
                            tmem_cast_a_res,
                        )
                else:
                    with domain_loop(0, num_k_tiles, 1):
                        _mma_loop_body(
                            smem_a_res,
                            smem_b_res,
                            tmem_c_res,
                            proxy,
                            smem_sfa_res,
                            smem_sfb_res,
                            tmem_sfa_res,
                            tmem_sfb_res,
                            tmem_sfab_res,
                            tmem_cast_a_res,
                        )
                tmem_c_res.commit()
                if cfg.use_tile256_tmem_overlap and cfg.num_epilogue_warps == 4:
                    tmem_c_res.advance_mma_overlap_window()

    def _mma_loop_body(
        smem_a_res,
        smem_b_res,
        tmem_c_res,
        proxy,
        smem_sfa_res,
        smem_sfb_res,
        tmem_sfa_res,
        tmem_sfb_res,
        tmem_sfab_res,
        tmem_cast_a_res,
        *,
        release_sources: bool = True,
    ) -> None:
        if has_cast_a:
            tmem_cast_a_res.try_wait()
        if has_proxy:
            proxy.try_wait()
            proxy.wait()
            proxy_stage_idx = proxy.consumer_work()
        elif has_cast_a:
            smem_b_res.try_wait()
        else:
            smem_a_res.try_wait()
            smem_b_res.try_wait()
        if has_fused_sf:
            smem_sfa_res.try_wait()
            smem_sfb_res.try_wait()
        if has_separate_sf:
            tmem_sfa_res.try_wait()
            tmem_sfb_res.try_wait()
        if has_combined_sf:
            tmem_sfab_res.try_wait()
        if has_cast_a:
            tmem_cast_a_res.wait()
        if not has_proxy:
            if has_cast_a:
                smem_b_res.wait()
            else:
                smem_a_res.wait()
                smem_b_res.wait()
        if has_fused_sf:
            smem_sfa_res.wait()
            smem_sfb_res.wait()
        if has_separate_sf:
            tmem_sfa_res.wait()
            tmem_sfb_res.wait()
        if has_combined_sf:
            tmem_sfab_res.wait()
        if has_cast_a:
            tmem_cast_a_addr = tmem_cast_a_res.publish_cast_a_addr()
            desc_b_mma_base, smem_b_stage_ptr = smem_b_res.build_mma_desc_b()
            tmem_c_res.mma_cast_a(
                desc_b_mma_base=desc_b_mma_base,
                smem_b_stage_ptr=smem_b_stage_ptr,
                tmem_cast_a_addr=tmem_cast_a_addr,
            )
        else:
            if has_proxy:
                desc_a_mma_base, smem_a_stage_ptr = (
                    smem_a_res.build_mma_desc_a_at_stage(
                        pipeline_stage_idx=proxy_stage_idx
                    )
                )
                desc_b_mma_base, smem_b_stage_ptr = (
                    smem_b_res.build_mma_desc_b_at_stage(
                        pipeline_stage_idx=proxy_stage_idx
                    )
                )
            else:
                desc_a_mma_base, smem_a_stage_ptr = smem_a_res.build_mma_desc_a()
                desc_b_mma_base, smem_b_stage_ptr = smem_b_res.build_mma_desc_b()
            if has_fused_sf:
                desc_a_s2t_base, smem_sfa_stage_ptr = smem_sfa_res.build_sfa_s2t_desc()
                desc_b_s2t_base = smem_sfb_res.build_sfb_s2t_desc()
                tmem_c_res.mma_fused_sf(
                    desc_a_mma_base=desc_a_mma_base,
                    smem_a_stage_ptr=smem_a_stage_ptr,
                    desc_b_mma_base=desc_b_mma_base,
                    smem_b_stage_ptr=smem_b_stage_ptr,
                    desc_a_s2t_base=desc_a_s2t_base,
                    smem_sfa_stage_ptr=smem_sfa_stage_ptr,
                    desc_b_s2t_base=desc_b_s2t_base,
                )
            elif has_separate_sf:
                sfa_stage_col_offset = tmem_sfa_res.publish_sfa_offset()
                sfb_stage_col_offset = tmem_sfb_res.publish_sfb_offset()
                tmem_c_res.mma_separate_sf(
                    desc_a_mma_base=desc_a_mma_base,
                    smem_a_stage_ptr=smem_a_stage_ptr,
                    desc_b_mma_base=desc_b_mma_base,
                    smem_b_stage_ptr=smem_b_stage_ptr,
                    sfa_stage_col_offset=sfa_stage_col_offset,
                    sfb_stage_col_offset=sfb_stage_col_offset,
                )
            elif has_combined_sf:
                sfa_stage_col_offset, sfb_stage_col_offset = (
                    tmem_sfab_res.publish_sfab_offset()
                )
                tmem_c_res.mma_separate_sf(
                    desc_a_mma_base=desc_a_mma_base,
                    smem_a_stage_ptr=smem_a_stage_ptr,
                    desc_b_mma_base=desc_b_mma_base,
                    smem_b_stage_ptr=smem_b_stage_ptr,
                    sfa_stage_col_offset=sfa_stage_col_offset,
                    sfb_stage_col_offset=sfb_stage_col_offset,
                )
            else:
                tmem_c_res.mma(
                    desc_a_mma_base=desc_a_mma_base,
                    smem_a_stage_ptr=smem_a_stage_ptr,
                    desc_b_mma_base=desc_b_mma_base,
                    smem_b_stage_ptr=smem_b_stage_ptr,
                )
        if release_sources:
            _release_mma_sources(
                smem_a_res,
                smem_b_res,
                proxy,
                smem_sfa_res,
                smem_sfb_res,
                tmem_sfa_res,
                tmem_sfb_res,
                tmem_sfab_res,
                tmem_cast_a_res,
            )

    def _release_mma_sources(
        smem_a_res,
        smem_b_res,
        proxy,
        smem_sfa_res,
        smem_sfb_res,
        tmem_sfa_res,
        tmem_sfb_res,
        tmem_sfab_res,
        tmem_cast_a_res,
    ) -> None:
        if has_cast_a:
            tmem_cast_a_res.release()
            smem_b_res.release()
        else:
            smem_a_res.release()
            smem_b_res.release()
        if has_proxy:
            proxy.release()
        if has_fused_sf:
            smem_sfa_res.release()
            smem_sfb_res.release()
        if has_separate_sf:
            tmem_sfa_res.release()
            tmem_sfb_res.release()
        if has_combined_sf:
            tmem_sfab_res.release()

    src_resources = [tmem_cast_a, smem_b] if has_cast_a else [smem_a, smem_b]
    if has_proxy:
        src_resources.append(proxy_cluster)
    if has_fused_sf:
        src_resources += [smem_sfa, smem_sfb]
    if has_separate_sf:
        src_resources += [tmem_sfa, tmem_sfb]
    if has_combined_sf:
        src_resources.append(tmem_sfab)
    if cfg.is_persistent:
        src_resources.append(work_queue)
    extra_resources = []
    if has_proxy:
        extra_resources.append(proxy_cluster)
    if has_fused_sf:
        extra_resources += [smem_sfa, smem_sfb]
    if has_separate_sf:
        extra_resources += [tmem_sfa, tmem_sfb]
    if has_combined_sf:
        extra_resources.append(tmem_sfab)
    if has_cast_a:
        extra_resources.append(tmem_cast_a)
    if is_persistent:
        extra_resources.append(work_queue)
    captured_schedule = mma_schedule(smem_a, smem_b, tmem_c, *extra_resources)

    return Task(
        src_resources=src_resources,
        dst_resources=[tmem_c],
        warp_idx=cfg.mma_warp_idx,
        num_warps=cfg.num_mma_warps,
        schedule=captured_schedule,
        num_registers=cfg.mma_regs,
        name="MmaTask0",
        # For 2-CTA: only leader CTA runs the MMA task (including pipeline ops).
        run_only_on_cta_id=0 if cfg.has_cluster else None,
    )


# ---------------------------------------------------------------------------
# Epilogue task
# ---------------------------------------------------------------------------


def create_epilogue_task(
    cfg,
    tmem_c,
    gmem_c,
    work_queue,
    num_k_tiles: int,
) -> Task:
    if cfg.is_swap_ab:
        epi_warpgroup_count = max(1, cfg.num_epilogue_warps // 4)
        epi_cols_per_call = cfg.epi_tile_n * epi_warpgroup_count
        epi_subtile_cnt = max(1, cfg.tile_n // max(1, epi_cols_per_call))
    else:
        epi_t2r_repx = cfg.epi_tile_n // 4
        epi_subtile_cnt = max(1, cfg.tile_n // max(1, epi_t2r_repx))
    is_persistent = _is_persistent(cfg)

    @schedule
    def epilogue_schedule(
        tmem: MemoryResource, gmem: MemoryResource, wq: WorkQueue = None
    ) -> None:
        _ = tmem.init_epilogue_state()
        gmem.init_store_state()
        with _work_tile_schedule_loop(cfg, wq):
            gmem.init_epilogue_tile_state()
            with domain_loop(0, num_k_tiles, 1):
                pass
            _epilogue_tail(tmem, gmem)

    def _epilogue_tail(tmem, gmem) -> None:
        tmem.try_wait()
        tmem.wait()
        if cfg.use_tile256_tmem_overlap and cfg.num_epilogue_warps == 4:
            # Max-TMEM-overlap schedule: load the shared
            # middle D tile first, release TmemC so the next MMA can begin,
            # then consume/store that preloaded fragment and the remaining D
            # tiles. The corresponding T2R index remap must be SSA-safe; see
            # the tmem_c_resources.py comment around t2r_output_call_idx.
            t2r_rmem, t2r_rmem_1, t2r_output_call_idx = tmem.consumer_work(
                subtile_idx=0
            )
            tmem.release()
            gmem.store_epilogue(
                t2r_rmem=t2r_rmem,
                t2r_rmem_1=t2r_rmem_1,
                t2r_output_call_idx=t2r_output_call_idx,
                subtile_idx=0,
            )
            for subtile_idx in range(max(0, epi_subtile_cnt - 1)):
                t2r_rmem, t2r_rmem_1, t2r_output_call_idx = tmem.load_overlap_subtile(
                    subtile_idx=subtile_idx
                )
                gmem.store_epilogue(
                    t2r_rmem=t2r_rmem,
                    t2r_rmem_1=t2r_rmem_1,
                    t2r_output_call_idx=t2r_output_call_idx,
                    subtile_idx=subtile_idx + 1,
                )
        else:
            for subtile_idx in range(epi_subtile_cnt):
                t2r_rmem, t2r_rmem_1, t2r_output_call_idx = tmem.consumer_work(
                    subtile_idx=subtile_idx
                )
                gmem.store_epilogue(
                    t2r_rmem=t2r_rmem,
                    t2r_rmem_1=t2r_rmem_1,
                    t2r_output_call_idx=t2r_output_call_idx,
                    subtile_idx=subtile_idx,
                )
            tmem.release()

    captured_schedule = _call_schedule_with_optional_work_queue(
        epilogue_schedule,
        is_persistent,
        work_queue,
        tmem_c,
        gmem_c,
    )
    return Task(
        src_resources=[tmem_c] + ([work_queue] if cfg.is_persistent else []),
        dst_resources=[gmem_c],
        warp_idx=cfg.epilogue_warp_idx,
        num_warps=cfg.num_epilogue_warps,
        schedule=captured_schedule,
        num_registers=cfg.epilogue_regs,
        name="EpilogueTask0",
    )


# ---------------------------------------------------------------------------
# DeepSeek FP8 tasks (use_deepseek_fp8=1)
# ---------------------------------------------------------------------------


def create_load_sfab_task(
    cfg,
    smem_dsfp8_sfab,
    work_queue,
    num_k_tiles: int,
    pdl_wait_resource=None,
    pdl_launch_resource=None,
) -> Task:
    """Producer task for DeepSeek FP8 per-K-chunk dequant SFs.

    Single warp issues cp.async groups for one activation-SF tile + 1 weight
    float per K-tile into ``SmemDeepSeekSfAb``. No paired Gmem resource — the GMEM
    pointers (``params.ptrSfA``, ``params.ptrSfB``) are read directly by the
    producer body.
    """
    is_persistent = _is_persistent(cfg)
    has_pdl_wait = pdl_wait_resource is not None
    has_pdl_launch = pdl_launch_resource is not None

    @schedule
    def load_sfab_schedule(smem: MemoryResource, *extra) -> None:
        idx = 0
        pdl_wait = extra[idx] if has_pdl_wait else None
        idx += 1 if has_pdl_wait else 0
        pdl_launch = extra[idx] if has_pdl_launch else None
        idx += 1 if has_pdl_launch else 0
        wq = extra[idx] if is_persistent else None
        if pdl_wait is not None:
            pdl_wait.wait_griddep()
        smem.init_load_state()
        with _work_tile_schedule_loop(cfg, wq):
            smem.prepare_sfab_tile()
            with domain_loop(0, num_k_tiles, 1):
                smem.try_acquire()
                smem.acquire()
                smem.load_sfab_tile()
                smem.commit()
        if pdl_launch is not None:
            pdl_launch.launch_griddep()

    extra_resources = []
    if pdl_wait_resource is not None:
        extra_resources.append(pdl_wait_resource)
    if pdl_launch_resource is not None:
        extra_resources.append(pdl_launch_resource)
    if is_persistent:
        extra_resources.append(work_queue)
    captured_schedule = load_sfab_schedule(smem_dsfp8_sfab, *extra_resources)
    return Task(
        src_resources=(
            _pdl_wait_resources(pdl_wait_resource)
            + ([work_queue] if cfg.is_persistent else [])
        ),
        dst_resources=[smem_dsfp8_sfab] + _pdl_launch_resources(pdl_launch_resource),
        warp_idx=cfg.load_sfab_warp_idx,
        num_warps=cfg.num_load_sfab_warps,
        schedule=captured_schedule,
        num_registers=cfg.load_sfab_regs,
        name="LoadSfAbTask",
    )


def create_epilogue_task_dsfp8(
    cfg,
    tmem_c,
    smem_dsfp8_sfab,
    gmem_c,
    work_queue,
    num_k_tiles: int,
) -> Task:
    """DeepSeek FP8 epilogue: per-K-tile SF drain + tail TmemC drain.

    Per K-tile (inside the loop):
      - Wait on ``SmemDeepSeekSfAb`` (the matching dequant SF stage).
      - Read ``act_lane * weight`` from SMEM and ffma2 into a FP32 register
        accumulator partial.
      - Release ``SmemDeepSeekSfAb``.
    Tail (after the K-loop):
      - Wait on ``TmemC`` (single MMA accumulator), LDTM to registers,
        release.
      - Cast the register accumulator to ``dtype_c`` and TMA/STG store.

    This TS path uses a single TmemC accumulator drained once in the tail,
    matching the existing ``create_epilogue_task`` shape.
    """
    is_persistent = _is_persistent(cfg)

    @schedule
    def epi_schedule(
        tmem: MemoryResource,
        sf: MemoryResource,
        gmem: MemoryResource,
        wq: WorkQueue = None,
    ) -> None:
        _ = tmem.init_epilogue_state()
        _ = sf.init_epilogue_state()
        gmem.init_store_state()

        def reset_work_tile_state():
            sf.reset_dequant_accumulator()
            if cutlass.const_expr(
                cfg.has_deepseek_fp8_c_scale and cfg.epi_tile_n == 64
            ):
                gmem.reset_dsfp8_absmax()

        with _work_tile_schedule_loop(
            cfg,
            wq,
            work_tile_preheader=reset_work_tile_state,
        ):
            gmem.init_epilogue_tile_state()
            with domain_loop(0, num_k_tiles, 1) as d:
                tmem.try_wait()
                sf.try_wait()
                tmem.wait()
                tok = tmem.consumer_work(subtile_idx=0)
                tmem.release()
                sf.wait()
                sf.consume_sfab_tile()
                scaled_tok = sf.apply_dequant_to_t2r(
                    t2r_rmem=tok[0],
                    t2r_rmem_1=tok[1],
                    t2r_output_call_idx=tok[2],
                )
                acc_tok = sf.accumulate_scaled_t2r(
                    t2r_rmem=scaled_tok[0],
                    t2r_rmem_1=scaled_tok[1],
                    t2r_output_call_idx=scaled_tok[2],
                    t2r_dequant_scale=scaled_tok[3],
                )
                sf.release()
                with d.last_iter():
                    gmem.store_epilogue(
                        t2r_rmem=acc_tok[0],
                        t2r_rmem_1=acc_tok[1],
                        t2r_output_call_idx=acc_tok[2],
                        subtile_idx=0,
                    )

    captured_schedule = _call_schedule_with_optional_work_queue(
        epi_schedule,
        is_persistent,
        work_queue,
        tmem_c,
        smem_dsfp8_sfab,
        gmem_c,
    )
    return Task(
        src_resources=[tmem_c, smem_dsfp8_sfab]
        + ([work_queue] if cfg.is_persistent else []),
        dst_resources=[gmem_c],
        warp_idx=cfg.epilogue_warp_idx,
        num_warps=cfg.num_epilogue_warps,
        schedule=captured_schedule,
        num_registers=cfg.epilogue_regs,
        name="EpilogueTask0DsFp8",
    )


# ---------------------------------------------------------------------------
# WorkId task (CLC dynamic persistent only)
# ---------------------------------------------------------------------------


def create_workid_task(
    cfg,
    work_queue,
    num_k_tiles: int,
    work_throttle=None,
) -> Task:
    """CLC dynamic persistent scheduler task.

    The producer side issues CLC queries; the consumer side reads responses.
    domain=0 because this task only runs in the persistent tail loop
    (all schedule entries are Tail-tagged).
    """
    has_work_throttle = work_throttle is not None

    @schedule
    def workid_schedule(wq: WorkQueue, throttle: MemoryResource = None) -> None:
        with _persistent_work_tile_loop(cfg, wq) as work_tile:
            with _persistent_skippable(cfg, work_tile):
                with domain_loop(0, 0, 1):
                    pass
                if throttle is not None:
                    throttle.wait()
                    throttle.release()
            wq.acquire()
            wq.fetch_work_tile()
            wq.commit()
            wq.wait()
            wq.get_and_advance_work_tile()
            wq.release()

    if has_work_throttle:
        captured_schedule = workid_schedule(work_queue, work_throttle)
    else:
        captured_schedule = workid_schedule(work_queue)
    return Task(
        src_resources=[work_queue]
        + ([work_throttle] if work_throttle is not None else []),
        dst_resources=[work_queue],
        warp_idx=cfg.workid_warp_idx,
        num_warps=cfg.num_workid_warps,
        schedule=captured_schedule,
        num_registers=cfg.workid_regs,
        name="WorkScheduleTask",
        run_only_on_cta_id=0 if cfg.has_cluster else None,
    )


# ---------------------------------------------------------------------------
# Padding task
# ---------------------------------------------------------------------------


def create_padding_task(
    cfg,
    work_queue,
    num_k_tiles: int,
) -> Task:
    is_persistent = _is_persistent(cfg)

    @schedule
    def padding_schedule(wq: WorkQueue = None) -> None:
        with _k_tile_schedule_loop(cfg, wq, num_k_tiles):
            pass

    captured_schedule = (
        padding_schedule(work_queue) if is_persistent else padding_schedule()
    )
    return Task(
        src_resources=[work_queue] if is_persistent else [],
        dst_resources=[],
        warp_idx=cfg.padding_warp_idx,
        num_warps=cfg.num_padding_warps,
        schedule=captured_schedule,
        num_registers=cfg.padding_regs,
        name="PaddingTask",
    )
