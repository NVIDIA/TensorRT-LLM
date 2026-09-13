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

"""BatchedGemm TS kernel assembly.

Entry points:
  - build_batched_gemm_task_manager()  — pure Python, validation only (no GPU)
  - batched_gemm_kernel()             — @cute.kernel for GPU execution
  - gemm()                            — host-side launcher (creates TMA descs, compiles, launches)
"""

import os
from typing import Any

from ..cutlass_dsl import require_cutlass_dsl_experimental, task_scheduling_scope

require_cutlass_dsl_experimental()

import cutlass
import cutlass.experimental.cuda as cuda
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import Int32

if not hasattr(cuda, "create_tensor_map_tiled_from_tensor") and hasattr(
    cuda, "create_tensor_map_tiled_from_view"
):
    cuda.create_tensor_map_tiled_from_tensor = cuda.create_tensor_map_tiled_from_view

from cutlass.experimental.task_scheduling.enums import (
    SignalingThreads,
    TileSchedulerType,
)
from cutlass.experimental.task_scheduling.memory import (
    SmemAllocation,
    SmemAllocator,
    TmemAllocator,
)
from cutlass.experimental.task_scheduling.resources import (
    PdlLaunchBarrier,
    PdlWaitBarrier,
    PipelineConfig,
    TileSchedulerConfig,
    WorkQueue,
)
from cutlass.experimental.task_scheduling.task_manager import TaskManager

from .batched_gemm_config import (
    BatchedGemmConfig,
    DType,
    RouteImpl,
    SfLayout,
    SfSmemToTmemCopy,
    _ldgsts_sfb_producer_commit_prefetch_depth,
    compute_num_k_tiles,
    compute_warp_layout,
    make_config,
    resolve_early_exit_max_token_ctas,
    uses_routed_sfa_tma_desc,
    uses_routed_sfb_tma_desc,
    validate_config,
)
from .batched_gemm_resources import (
    GmemAResource,
    GmemBResource,
    GmemSfAResource,
    GmemSfBResource,
    SmemAResource,
    SmemBResource,
    SmemGatherResource,
    SmemTmaGatherResource,
    ProxyClusterBarrierResource,
    BatchedGemmWorkQueue,
    WorkThrottleBarrierResource,
    SmemSfAResource,
    SmemSfBResource,
    SmemSfGatherAResource,
    SmemSfGatherBResource,
    SmemSfLdgstsAResource,
    SmemSfLdgstsBResource,
    TmemCastAResource,
    TmemSfAResource,
    TmemSfABResource,
    TmemSfBResource,
    TmemSfRouteAResource,
    TmemSfRouteBResource,
    TmemCResource,
    GmemCResource,
    SmemDeepSeekSfAbResource,
)
from .batched_gemm_tasks import (
    create_load_a_task,
    create_load_b_task,
    create_gather_task,
    create_fused_load_a_sfa_task,
    create_fused_gather_sfb_task,
    create_sync_task,
    create_load_sfa_task,
    create_load_sfb_task,
    create_cast_a_task,
    create_copy_sfa_task,
    create_copy_sfab_task,
    create_copy_sfb_task,
    create_mma_task,
    create_epilogue_task,
    create_workid_task,
    create_padding_task,
    create_load_sfab_task,
    create_epilogue_task_dsfp8,
)
from cutlass.experimental import primitives as prims

TMA_DIM_MAX = 1 << 31
TMA_XLARGE_N = 1 << 35


def _exhaustive_deadlock_race_check_enabled() -> bool:
    value = os.environ.get("FLASHINFER_PRIMS_TS_DEBUG_CHECKS", "0").lower()
    return value not in {"0", "false", "no", "off"}


def _env_flag_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _task_manager_verify_enabled() -> bool:
    return _env_flag_enabled("FLASHINFER_PRIMS_TS_DEBUG_CHECKS")


def _pdl_wait_completed_before_tasks(cfg: BatchedGemmConfig) -> bool:
    return bool(cfg.do_pdl_wait_for_num_non_exiting_ctas)


class _ProductionTaskManager(TaskManager):
    def print_and_verify(self) -> None:
        return None


def _round_up_tmem_columns(num_columns: int) -> int:
    """tcgen05_alloc requires a power-of-two column count in [32, 512]."""
    return max(32, 1 << (num_columns - 1).bit_length())


def _require_even_divide(value: int, divisor: int, name: str) -> int:
    """Return ``value // divisor`` or fail if the split would be fractional."""
    if divisor <= 0:
        raise ValueError(f"{name}: divisor must be positive, got {divisor}.")
    if value % divisor != 0:
        raise ValueError(f"{name}: expected {value} to divide evenly by {divisor}.")
    return value // divisor


# ---------------------------------------------------------------------------
# Pipeline config builder (shared between validation + GPU paths)
# ---------------------------------------------------------------------------


def _make_pipeline_configs(cfg):
    """Build all PipelineConfig objects for the given config."""
    one_thread = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    cluster_vmnk = (cfg.cluster_m, 1, 1, 1)

    smem_a_num_bytes = cfg.num_bytes_a_tma_per_stage * cfg.cluster_m
    smem_b_num_bytes = cfg.num_bytes_b_tma_per_stage * cfg.cluster_m
    smem_a_producer = one_thread
    smem_b_producer = one_thread
    smem_a_producer_signaling = SignalingThreads.All
    smem_b_producer_signaling = SignalingThreads.All
    smem_a_num_bytes_per_warp_per_cta = None
    smem_b_num_bytes_per_warp_per_cta = None
    routed_tma_producer_signaling = (
        SignalingThreads.CtaLeader | SignalingThreads.TaskWarpLeader
    )
    if cfg.has_tma_route:
        if cfg.is_swap_ab:
            smem_b_num_bytes = cfg.num_bytes_b_tma_per_stage * (
                cfg.cluster_m if cfg.split_b_across_ctas else 1
            )
            smem_b_producer_signaling = routed_tma_producer_signaling
            smem_b_num_bytes_per_warp_per_cta = _require_even_divide(
                smem_b_num_bytes,
                cfg.cluster_m * max(1, cfg.num_load_b_warps),
                "smem_b_num_bytes_per_warp_per_cta",
            )
        else:
            smem_a_num_bytes = cfg.num_bytes_a_tma_per_stage * cfg.cluster_m
            smem_a_producer_signaling = routed_tma_producer_signaling
            smem_a_num_bytes_per_warp_per_cta = _require_even_divide(
                smem_a_num_bytes,
                cfg.cluster_m * max(1, cfg.num_load_a_warps),
                "smem_a_num_bytes_per_warp_per_cta",
            )
    if cfg.has_cluster:
        if smem_a_num_bytes_per_warp_per_cta is None:
            smem_a_num_bytes_per_warp_per_cta = _require_even_divide(
                smem_a_num_bytes,
                cfg.cluster_m * max(1, cfg.num_load_a_warps),
                "smem_a_num_bytes_per_warp_per_cta",
            )
        if smem_b_num_bytes_per_warp_per_cta is None:
            smem_b_num_bytes_per_warp_per_cta = _require_even_divide(
                smem_b_num_bytes,
                cfg.cluster_m * max(1, cfg.num_load_b_warps),
                "smem_b_num_bytes_per_warp_per_cta",
            )

    if cfg.has_cast_a:
        smem_a_cfg = PipelineConfig.create_tma_async_pipeline_cfg(
            num_stages=cfg.num_stages_a,
            num_bytes=cfg.num_bytes_a_tma_per_stage,
            producer_group=smem_a_producer,
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, cfg.num_cast_a_warps * 32
            ),
            cta_layout_vmnk=cluster_vmnk,
            producer_signaling_threads=smem_a_producer_signaling,
            consumer_signaling_threads=SignalingThreads.All,
        )
    else:
        smem_a_cfg = PipelineConfig.create_tma_umma_pipeline_cfg(
            num_stages=cfg.num_stages_a,
            num_bytes=smem_a_num_bytes,
            producer_group=smem_a_producer,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            cta_layout_vmnk=cluster_vmnk,
            producer_signaling_threads=smem_a_producer_signaling,
            consumer_signaling_threads=SignalingThreads.CtaLeader,
            mcast_mode_mn=(1, 0),
            num_bytes_per_warp_per_cta=smem_a_num_bytes_per_warp_per_cta,
        )
    smem_b_cfg = PipelineConfig.create_tma_umma_pipeline_cfg(
        num_stages=cfg.num_stages_b,
        num_bytes=smem_b_num_bytes,
        producer_group=smem_b_producer,
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        cta_layout_vmnk=cluster_vmnk,
        producer_signaling_threads=smem_b_producer_signaling,
        consumer_signaling_threads=SignalingThreads.CtaLeader,
        mcast_mode_mn=(0, 1),
        num_bytes_per_warp_per_cta=smem_b_num_bytes_per_warp_per_cta,
    )
    smem_sfa_cfg = None
    smem_sfb_cfg = None
    if cfg.has_scale_factor_a:
        smem_sfa_num_bytes = cfg.num_bytes_sfa_per_stage * cfg.cluster_m
        smem_sfa_num_bytes_per_warp_per_cta = (
            _require_even_divide(
                smem_sfa_num_bytes,
                cfg.cluster_m * max(1, cfg.num_load_sfa_warps),
                "smem_sfa_num_bytes_per_warp_per_cta",
            )
            if cfg.has_cluster
            else None
        )
        smem_sfa_cfg = PipelineConfig.create_tma_umma_pipeline_cfg(
            num_stages=cfg.num_stages_smem_sfa,
            num_bytes=smem_sfa_num_bytes,
            producer_group=one_thread,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            cta_layout_vmnk=cluster_vmnk,
            consumer_signaling_threads=SignalingThreads.CtaLeader,
            advance_on_wait=cfg.has_cluster and cfg.use_combined_sfab_copy,
            num_bytes_per_warp_per_cta=smem_sfa_num_bytes_per_warp_per_cta,
        )
        if cfg.has_cast_a:
            smem_sfa_cfg = PipelineConfig.create_tma_async_pipeline_cfg(
                num_stages=cfg.num_stages_smem_sfa,
                num_bytes=cfg.num_bytes_sfa_per_stage,
                producer_group=one_thread,
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, cfg.num_cast_a_warps * 32
                ),
                cta_layout_vmnk=cluster_vmnk,
                consumer_signaling_threads=SignalingThreads.All,
            )
        elif cfg.sfa_smem_to_tmem_copy == int(SfSmemToTmemCopy.LDS_STTM):
            # Linear SFA SMEM layout is consumed by the four-warp LDS+STTM
            # CopySfA task.
            sfa_tma_num_bytes_per_warp = (
                cfg.num_bytes_sfa_per_stage // cfg.num_load_sfa_warps
            )
            smem_sfa_cfg = PipelineConfig.create_tma_async_pipeline_cfg(
                num_stages=cfg.num_stages_smem_sfa,
                num_bytes=cfg.num_bytes_sfa_per_stage,
                producer_group=one_thread,
                # With no separate CTA barrier, wait for every LDS+STTM copy
                # thread before allowing the producer to reuse this stage.
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, cfg.num_copy_sfa_warps * 32
                ),
                cta_layout_vmnk=None,
                producer_signaling_threads=SignalingThreads.TaskWarpLeader,
                consumer_signaling_threads=SignalingThreads.All,
                num_bytes_per_warp_per_cta=sfa_tma_num_bytes_per_warp,
            )
    if cfg.has_scale_factors:
        smem_sfb_num_bytes = (
            cfg.num_bytes_sfb_per_stage * cfg.cluster_m
            if cfg.has_cluster
            and not (
                cfg.sfb_smem_to_tmem_copy == int(SfSmemToTmemCopy.LDS_STTM)
                and cfg.smem_sfb_layout == int(SfLayout.R8c4)
            )
            else cfg.num_bytes_sfb_per_stage
        )
        smem_sfb_num_bytes_per_warp_per_cta = (
            _require_even_divide(
                smem_sfb_num_bytes,
                cfg.cluster_m * max(1, cfg.num_load_sfb_warps),
                "smem_sfb_num_bytes_per_warp_per_cta",
            )
            if cfg.has_cluster
            and not (
                cfg.sfb_smem_to_tmem_copy == int(SfSmemToTmemCopy.LDS_STTM)
                and cfg.smem_sfb_layout == int(SfLayout.R8c4)
            )
            else None
        )
        smem_sfb_cfg = PipelineConfig.create_tma_umma_pipeline_cfg(
            num_stages=cfg.num_stages_smem_sfb,
            num_bytes=smem_sfb_num_bytes,
            producer_group=one_thread,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            cta_layout_vmnk=cluster_vmnk,
            consumer_signaling_threads=SignalingThreads.CtaLeader,
            num_bytes_per_warp_per_cta=smem_sfb_num_bytes_per_warp_per_cta,
        )
        if cfg.sfb_smem_to_tmem_copy == int(
            SfSmemToTmemCopy.LDS_STTM
        ) and cfg.smem_sfb_layout == int(SfLayout.R8c4):
            # Low-N SFB is consumed by all four CopySfB warps through LDS+STTM,
            # not by a single UMMA warp through tcgen05_cp.
            sfb_tma_num_bytes = cfg.num_bytes_sfb_tma_per_stage
            if cfg.has_routed_sfs and cfg.uses_tma_routed_sfs:
                # PipelineTmaAsync arms one transaction barrier per
                # producer warp. TMA-routed compact SFB distributes the
                # generated gather4 row work over num_load_sfb_warps, so each
                # arrival must expect only that warp's slice of the stage.
                sfb_tma_num_bytes //= cfg.num_load_sfb_warps
            smem_sfb_cfg = PipelineConfig.create_tma_async_pipeline_cfg(
                num_stages=cfg.num_stages_smem_sfb,
                num_bytes=sfb_tma_num_bytes,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, cfg.num_load_sfb_warps
                ),
                # With no separate CTA barrier, wait for every LDS+STTM copy
                # thread before allowing the producer to reuse this stage.
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, cfg.num_copy_sfb_warps * 32
                ),
                cta_layout_vmnk=None,
                consumer_signaling_threads=SignalingThreads.All,
            )
        elif cfg.has_routed_sfs and cfg.uses_tma_routed_sfs and cfg.is_swap_ab:
            # Linear SFB SMEM layout (routed via TMA gather4) is consumed by the
            # multi-warp LDS+STTM CopySfB task, mirroring the routed-TMA SFA
            # path above. The LoadSfB task spreads gather4 row work over
            # num_load_sfb_warps warps, each electing one thread to issue its
            # gather4 loads (see SmemSfGatherResource._producer_work_impl). The
            # PipelineTmaAsync arms one transaction barrier per producer warp, so
            # signal with TaskWarpLeader and have each arrival expect only that
            # warp's slice of the stage.
            #
            # The Linear layout cannot use UTCCP, so there is no 2-CTA-scope
            # multicast: even under a 2-CTA cluster each CTA issues its own
            # single-CTA gather4 redundantly and runs its own CopySfB. The
            # pipeline is therefore purely per-CTA -- no cluster byte multiplier
            # and no leader-routed (CtaLeader) signaling -- in contrast to the
            # R128c4/UTCCP path, which does multiply.
            sfb_tma_num_bytes_per_warp = (
                cfg.num_bytes_sfb_per_stage // cfg.num_load_sfb_warps
            )
            smem_sfb_cfg = PipelineConfig.create_tma_async_pipeline_cfg(
                num_stages=cfg.num_stages_smem_sfb,
                num_bytes=cfg.num_bytes_sfb_per_stage,
                producer_group=one_thread,
                # With no separate CTA barrier, wait for every LDS+STTM copy
                # thread before allowing the producer to reuse this stage.
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, cfg.num_copy_sfb_warps * 32
                ),
                cta_layout_vmnk=None,
                producer_signaling_threads=SignalingThreads.TaskWarpLeader,
                consumer_signaling_threads=SignalingThreads.All,
                num_bytes_per_warp_per_cta=sfb_tma_num_bytes_per_warp,
            )
    tmem_c_cfg = PipelineConfig.create_umma_async_pipeline_cfg(
        num_stages=cfg.num_stages_tmem_acc,
        producer_group=one_thread,
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, cfg.num_epilogue_warps * 32 * cfg.cluster_m
        ),
        cta_layout_vmnk=cluster_vmnk,
        producer_signaling_threads=SignalingThreads.CtaLeader,
    )
    result = {
        "smem_a": smem_a_cfg,
        "smem_b": smem_b_cfg,
        "tmem_c": tmem_c_cfg,
    }
    if cfg.has_cast_a:
        result["tmem_cast_a"] = PipelineConfig.create_async_umma_pipeline_cfg(
            num_stages=cfg.num_stages_cast_a,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, cfg.num_cast_a_warps * 32 * cfg.cluster_m
            ),
            consumer_group=one_thread,
            cta_layout_vmnk=cluster_vmnk,
            producer_signaling_threads=SignalingThreads.All,
            consumer_signaling_threads=SignalingThreads.CtaLeader,
        )

    # Gather (LDGSTS) pipeline for routed activations
    if cfg.has_gather:
        gather_consumer_wait_signaling = (
            SignalingThreads.All if cfg.has_cluster and cfg.num_sync_warps > 0 else None
        )
        gather_producer_threads = cfg.num_gather_warps * 32
        gather_producer_op = pipeline.PipelineOp.AsyncLoad
        gather_advance_on_acquire = False
        if cfg.fuse_operand_sf_loads:
            gather_producer_threads *= cfg.cluster_m
            gather_producer_op = pipeline.PipelineOp.AsyncThread
            gather_advance_on_acquire = True
        gather_cfg = PipelineConfig.create_async_umma_pipeline_cfg(
            num_stages=cfg.num_stages_b if cfg.is_swap_ab else cfg.num_stages_a,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, gather_producer_threads
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            cta_layout_vmnk=cluster_vmnk,
            producer_signaling_threads=SignalingThreads.All,
            consumer_signaling_threads=SignalingThreads.CtaLeader,
            consumer_wait_signaling_threads=gather_consumer_wait_signaling,
            producer_op=gather_producer_op,
            advance_on_acquire=gather_advance_on_acquire,
        )
        result["smem_gather"] = gather_cfg

    # 2-CTA + gather: proxy barrier for cross-CTA sync
    if cfg.has_gather and cfg.has_cluster:
        proxy_cfg = PipelineConfig.create_async_umma_pipeline_cfg(
            # validate_config() requires equal A/B depths for this path.
            num_stages=cfg.num_stages_a,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * cfg.cluster_m,  # SyncTask warp × CTAs
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            cta_layout_vmnk=cluster_vmnk,
            producer_signaling_threads=SignalingThreads.All,
            consumer_signaling_threads=SignalingThreads.CtaLeader,
        )
        result["proxy"] = proxy_cfg

    if cfg.has_scale_factor_a:
        result["smem_sfa"] = smem_sfa_cfg
    if cfg.has_scale_factors:
        result["smem_sfb"] = smem_sfb_cfg
        # LDGSTS SF uses cp.async producer work.  The exact
        # clustered R128c4 SFB path drains cp.async groups, then uses a
        # normal AsyncThread producer_commit into the 2-CTA UMMA full barrier.
        # Compact low-N SFB keeps generic Async because CopySfB performs
        # LDS+STTM directly.
        if cfg.has_routed_sfs and cfg.uses_ldgsts_routed_sfs:
            ldgsts_consumer_signaling = (
                SignalingThreads.CtaLeader if cfg.has_cluster else SignalingThreads.All
            )
            ldgsts_sfa_producer_warps = max(1, cfg.num_load_sfa_warps)
            ldgsts_sfb_producer_warps = max(1, cfg.num_load_sfb_warps)
            ldgsts_sfa_cfg = PipelineConfig.create_async_umma_pipeline_cfg(
                num_stages=cfg.num_stages_smem_sfa,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    ldgsts_sfa_producer_warps * 32,
                ),
                consumer_group=one_thread,
                cta_layout_vmnk=cluster_vmnk,
                consumer_signaling_threads=ldgsts_consumer_signaling,
                producer_op=pipeline.PipelineOp.AsyncLoad,
            )
            if cfg.is_swap_ab and cfg.sfb_smem_to_tmem_copy == int(
                SfSmemToTmemCopy.LDS_STTM
            ):
                # Compact routed SFB uses the LDS+STTM path where all
                # CopySfB warps participate; keep generic async semantics.
                # Generated clustered tile64/N192 kernels use a per-CTA
                # CutlassCpAsyncPipeline here: one LoadSfB producer warp feeds
                # the four local CopySfB warps, so the producer group is not
                # scaled by cluster_m.
                ldgsts_sfb_cfg = PipelineConfig.create_async_async_pipeline_cfg(
                    num_stages=cfg.num_stages_smem_sfb,
                    producer_group=pipeline.CooperativeGroup(
                        pipeline.Agent.Thread,
                        ldgsts_sfb_producer_warps * 32,
                    ),
                    consumer_group=pipeline.CooperativeGroup(
                        pipeline.Agent.Thread, cfg.num_copy_sfb_warps * 32
                    ),
                    cta_layout_vmnk=cluster_vmnk,
                    producer_op=pipeline.PipelineOp.AsyncLoad,
                )
            else:
                sfb_producer_threads = ldgsts_sfb_producer_warps * 32
                sfb_producer_op = pipeline.PipelineOp.AsyncLoad
                sfb_advance_on_acquire = False
                if cfg.has_cluster and cfg.is_swap_ab and cfg.tile_n >= 128:
                    sfb_producer_threads *= cfg.cluster_m
                    sfb_producer_op = pipeline.PipelineOp.AsyncThread
                    sfb_advance_on_acquire = True
                ldgsts_sfb_cfg = PipelineConfig.create_async_umma_pipeline_cfg(
                    num_stages=cfg.num_stages_smem_sfb,
                    producer_group=pipeline.CooperativeGroup(
                        pipeline.Agent.Thread,
                        sfb_producer_threads,
                    ),
                    consumer_group=one_thread,
                    cta_layout_vmnk=cluster_vmnk,
                    consumer_signaling_threads=ldgsts_consumer_signaling,
                    producer_op=sfb_producer_op,
                    advance_on_wait=cfg.has_cluster and cfg.use_combined_sfab_copy,
                    advance_on_acquire=sfb_advance_on_acquire,
                )
            if cfg.is_swap_ab:
                result["smem_sfb"] = ldgsts_sfb_cfg
            else:
                result["smem_sfa"] = ldgsts_sfa_cfg

    # TmemSf pipeline: the combined tcgen05_cp path uses, so
    # the full-barrier commit is a UMMA/TCGen05 producer arrival.  Compact SFB
    # uses LDS+STTM and keeps async-producer + UMMA-consumer semantics.
    if cfg.has_scale_factors and cfg.uses_unfused_tmem_sf_copy:
        if cfg.use_combined_sfab_copy:
            result["tmem_sfab"] = PipelineConfig.create_umma_umma_pipeline_cfg(
                num_stages=max(cfg.num_stages_tmem_sfa, cfg.num_stages_tmem_sfb),
                producer_group=one_thread,
                consumer_group=one_thread,
                cta_layout_vmnk=cluster_vmnk,
                producer_signaling_threads=SignalingThreads.CtaLeader,
                consumer_signaling_threads=SignalingThreads.CtaLeader,
            )
        else:
            tmem_sfb_producer_threads = cfg.num_copy_sfb_warps * 32
            tmem_sfb_producer_signaling = SignalingThreads.CtaLeader
            if cfg.has_cluster and cfg.sfb_smem_to_tmem_copy == int(
                SfSmemToTmemCopy.LDS_STTM
            ):
                # Generated compact SFB STTM runs the CopySfB warpgroup in every
                # CTA and uses cluster-scoped AsyncUmma-style signaling.
                tmem_sfb_producer_threads *= cfg.cluster_m
                tmem_sfb_producer_signaling = SignalingThreads.All
            tmem_sfa_cfg = PipelineConfig.create_umma_umma_pipeline_cfg(
                num_stages=cfg.num_stages_tmem_sfa,
                producer_group=one_thread,
                consumer_group=one_thread,
                cta_layout_vmnk=cluster_vmnk,
                producer_signaling_threads=SignalingThreads.CtaLeader,
                consumer_signaling_threads=SignalingThreads.CtaLeader,
            )
            if cfg.sfb_smem_to_tmem_copy == int(SfSmemToTmemCopy.LDS_STTM):
                tmem_sfb_cfg = PipelineConfig.create_async_umma_pipeline_cfg(
                    num_stages=cfg.num_stages_tmem_sfb,
                    producer_group=pipeline.CooperativeGroup(
                        pipeline.Agent.Thread, tmem_sfb_producer_threads
                    ),
                    consumer_group=one_thread,
                    cta_layout_vmnk=cluster_vmnk,
                    producer_signaling_threads=tmem_sfb_producer_signaling,
                    consumer_signaling_threads=SignalingThreads.CtaLeader,
                )
            else:
                tmem_sfb_cfg = PipelineConfig.create_umma_umma_pipeline_cfg(
                    num_stages=cfg.num_stages_tmem_sfb,
                    producer_group=one_thread,
                    consumer_group=one_thread,
                    cta_layout_vmnk=cluster_vmnk,
                    producer_signaling_threads=SignalingThreads.CtaLeader,
                    consumer_signaling_threads=SignalingThreads.CtaLeader,
                )
            if cfg.sfa_smem_to_tmem_copy == int(SfSmemToTmemCopy.LDS_STTM):
                tmem_sfa_cfg = PipelineConfig.create_async_umma_pipeline_cfg(
                    num_stages=cfg.num_stages_tmem_sfa,
                    producer_group=pipeline.CooperativeGroup(
                        pipeline.Agent.Thread, cfg.num_copy_sfa_warps * 32
                    ),
                    consumer_group=one_thread,
                    cta_layout_vmnk=cluster_vmnk,
                    producer_signaling_threads=SignalingThreads.CtaLeader,
                    consumer_signaling_threads=SignalingThreads.CtaLeader,
                )
            result["tmem_sfa"] = tmem_sfa_cfg
            result["tmem_sfb"] = tmem_sfb_cfg

    # DeepSeek FP8 SF: cp.async producer (single warp) → epilogue consumer.
    # Uses
    # options.mNumStagesSmemSfA; in TS this is the shared SF stage knob. One
    # warp issues cp.async for 32 activation floats + 1 weight float per
    # K-tile, the epilogue waits per K-tile and FMAs the dequant into the
    # partial accumulator before the configured FP32 output cast.
    if cfg.has_deepseek_fp8:
        result["smem_dsfp8_sfab"] = PipelineConfig.create_async_async_pipeline_cfg(
            num_stages=cfg.num_stages_smem_sfa,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                max(1, cfg.num_load_sfab_warps) * 32,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, cfg.num_epilogue_warps * 32 * cfg.cluster_m
            ),
            cta_layout_vmnk=cluster_vmnk,
            producer_op=pipeline.PipelineOp.AsyncLoad,
        )

    # CLC dynamic persistent WorkQueue pipeline
    if cfg.is_persistent:
        clc_cluster_scale = cfg.cluster_m if cfg.has_cluster else 1
        num_clc_consumer_threads = 0

        def add_clc_consumer_warps(
            num_warps: int, *, leader_only: bool = False
        ) -> None:
            nonlocal num_clc_consumer_threads
            cta_scale = 1 if leader_only else clc_cluster_scale
            num_clc_consumer_threads += 32 * cta_scale * num_warps

        # ClcFetchAsync routes all consumer arrives to CTA0's empty barrier.
        # Generated clustered kernels mix all-CTA consumers (load/epilogue)
        # with leader-only consumers (MMA/WorkId), so count each task by its
        # actual CTA ownership.
        add_clc_consumer_warps(cfg.num_load_a_warps)
        add_clc_consumer_warps(cfg.num_load_b_warps)
        add_clc_consumer_warps(cfg.num_epilogue_warps)
        add_clc_consumer_warps(cfg.num_padding_warps)
        add_clc_consumer_warps(cfg.num_gather_warps)
        add_clc_consumer_warps(cfg.num_sync_warps)
        add_clc_consumer_warps(cfg.num_cast_a_warps)
        add_clc_consumer_warps(cfg.num_mma_warps, leader_only=cfg.has_cluster)
        add_clc_consumer_warps(cfg.num_workid_warps, leader_only=cfg.has_cluster)
        if cfg.has_scale_factor_a and not cfg.fuse_operand_sf_loads:
            add_clc_consumer_warps(cfg.num_load_sfa_warps)
        if cfg.has_deepseek_fp8:
            add_clc_consumer_warps(cfg.num_load_sfab_warps)
        if cfg.has_scale_factors:
            if not cfg.fuse_operand_sf_loads:
                add_clc_consumer_warps(cfg.num_load_sfb_warps)
            if cfg.use_combined_sfab_copy:
                add_clc_consumer_warps(
                    cfg.num_copy_sfa_warps,
                    leader_only=cfg.has_cluster,
                )
            else:
                add_clc_consumer_warps(
                    cfg.num_copy_sfa_warps,
                    leader_only=cfg.has_cluster,
                )
                add_clc_consumer_warps(
                    cfg.num_copy_sfb_warps,
                    # Keep in sync with CopySfBTask.run_only_on_cta_id: CopySfB
                    # runs on every CTA for the per-CTA LDS+STTM paths (compact
                    # R8c4 and Linear routed-TMA SFB), and leader-only otherwise.
                    leader_only=cfg.has_cluster
                    and cfg.sfb_smem_to_tmem_copy != int(SfSmemToTmemCopy.LDS_STTM)
                    and not (
                        cfg.has_routed_sfs
                        and cfg.uses_tma_routed_sfs
                        and cfg.is_swap_ab
                    ),
                )
        workid_cfg = PipelineConfig.create_clc_fetch_async_pipeline_cfg(
            num_stages=cfg.num_stages_workid,
            num_bytes=16,  # CLC response is 16 bytes
            producer_group=one_thread,
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_clc_consumer_threads
            ),
            cta_layout_vmnk=cluster_vmnk,
            producer_signaling_threads=SignalingThreads.CtaLeader,
            consumer_signaling_threads=SignalingThreads.All,
        )
        result["workid"] = workid_cfg
        if cfg.use_work_throttle_barrier:
            result["work_throttle"] = PipelineConfig.create_async_async_pipeline_cfg(
                num_stages=cfg.num_stages_workid,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    cfg.num_work_throttle_producer_warps * 32,
                ),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
                cta_layout_vmnk=cluster_vmnk,
                producer_signaling_threads=SignalingThreads.CtaLeader,
                consumer_signaling_threads=SignalingThreads.CtaLeader,
            )

    return result


# ---------------------------------------------------------------------------
# Schedule builder (validation mode — no GPU, no TMA descs)
# ---------------------------------------------------------------------------


def _build_schedule_validate(cfg, num_k_tiles=4):
    """Build schedule for validation only (no MLIR context needed for WorkQueue)."""
    compute_warp_layout(cfg)
    validate_config(cfg)
    pcfgs = _make_pipeline_configs(cfg)

    gmem_a = GmemAResource(cfg=cfg, name="GmemA")
    gmem_b = GmemBResource(cfg=cfg, name="GmemB")

    smem_a = SmemAResource(cfg=cfg, pipeline_config=pcfgs["smem_a"], name="SmemA")
    smem_b = SmemBResource(cfg=cfg, pipeline_config=pcfgs["smem_b"], name="SmemB")

    tmem_c = TmemCResource(cfg=cfg, pipeline_config=pcfgs["tmem_c"], name="TmemC")
    gmem_c = GmemCResource(cfg=cfg, name="GmemC")

    # WorkQueue — CLC persistent or static (non-persistent)
    if cfg.is_persistent:
        tile_sched_cfg = TileSchedulerConfig(
            tile_scheduler_type=TileSchedulerType.ClcDynamicPersistent,
            tile_scheduler_params=None,
        )
        work_queue = BatchedGemmWorkQueue(
            tile_scheduler_config=tile_sched_cfg,
            cfg=cfg,
            pipeline_config=pcfgs.get("workid"),
            name="WorkQueue",
        )
    else:
        tile_sched_cfg = TileSchedulerConfig(
            tile_scheduler_type=TileSchedulerType.StaticPersistent,
            tile_scheduler_params=None,
        )
        work_queue = WorkQueue(tile_scheduler_config=tile_sched_cfg, name="WorkQueue")

    work_throttle = None
    if cfg.use_work_throttle_barrier:
        work_throttle = WorkThrottleBarrierResource(
            pipeline_config=pcfgs["work_throttle"],
            name="WorkThrottle",
        )
    pdl_wait_completed_before_tasks = _pdl_wait_completed_before_tasks(cfg)
    pdl_wait_resource = (
        PdlWaitBarrier(name="PdlWait")
        if cfg.use_pdl and not pdl_wait_completed_before_tasks
        else None
    )
    pdl_launch_resource = PdlLaunchBarrier(name="PdlLaunch") if cfg.use_pdl else None

    # Tasks: LoadA/LoadB (or Gather replacing one of them), MMA, Epilogue
    proxy_cluster = None
    if cfg.has_gather:
        smem_gather = SmemGatherResource(
            cfg=cfg,
            pipeline_config=pcfgs.get("smem_gather"),
            _operand="b" if cfg.is_swap_ab else "a",
            name="SmemGather",
        )
        if cfg.is_swap_ab:
            load_a = create_load_a_task(
                cfg,
                gmem_a,
                smem_a,
                work_queue,
                num_k_tiles,
                work_throttle=work_throttle,
            )
            gather = create_gather_task(
                cfg,
                gmem_b,
                smem_gather,
                work_queue,
                num_k_tiles,
                pdl_wait_resource=pdl_wait_resource,
                pdl_launch_resource=pdl_launch_resource,
            )
            tma_smem, gather_smem = smem_a, smem_gather
            smem_b = smem_gather
            task_list = [load_a, gather]
        else:
            gather = create_gather_task(
                cfg,
                gmem_a,
                smem_gather,
                work_queue,
                num_k_tiles,
                pdl_wait_resource=pdl_wait_resource,
                pdl_launch_resource=pdl_launch_resource,
            )
            load_b = create_load_b_task(
                cfg,
                gmem_b,
                smem_b,
                work_queue,
                num_k_tiles,
                work_throttle=work_throttle,
            )
            gather_smem, tma_smem = smem_gather, smem_b
            smem_a = smem_gather
            task_list = [gather, load_b]

        if cfg.has_cluster and cfg.num_sync_warps > 0:
            proxy_cluster = ProxyClusterBarrierResource(
                cfg=cfg,
                pipeline_config=pcfgs.get("proxy"),
                name="ProxyCluster",
            )
            sync = create_sync_task(
                cfg,
                proxy_cluster,
                gather_smem,
                tma_smem,
                work_queue,
                num_k_tiles,
                sync_warp_idx=cfg.sync_warp_idx,
            )
            task_list.append(sync)
    elif cfg.has_tma_route:
        # TMA gather4: SmemTmaGatherResource replaces the activation SMEM
        smem_tma_gather = SmemTmaGatherResource(
            cfg=cfg,
            pipeline_config=pcfgs["smem_b" if cfg.is_swap_ab else "smem_a"],
            _operand="b" if cfg.is_swap_ab else "a",
            name="SmemTmaGather",
        )
        if cfg.is_swap_ab:
            load_a = create_load_a_task(
                cfg,
                gmem_a,
                smem_a,
                work_queue,
                num_k_tiles,
                work_throttle=work_throttle,
            )
            load_b = create_load_b_task(
                cfg,
                gmem_b,
                smem_tma_gather,
                work_queue,
                num_k_tiles,
                pdl_wait_resource=pdl_wait_resource,
                pdl_launch_resource=pdl_launch_resource,
            )
            smem_b = smem_tma_gather
            task_list = [load_a, load_b]
        else:
            load_a = create_load_a_task(
                cfg,
                gmem_a,
                smem_tma_gather,
                work_queue,
                num_k_tiles,
                work_throttle=work_throttle,
                pdl_wait_resource=pdl_wait_resource,
                pdl_launch_resource=pdl_launch_resource,
            )
            load_b = create_load_b_task(cfg, gmem_b, smem_b, work_queue, num_k_tiles)
            smem_a = smem_tma_gather
            task_list = [load_a, load_b]
    else:
        load_a = create_load_a_task(
            cfg,
            gmem_a,
            smem_a,
            work_queue,
            num_k_tiles,
            work_throttle=work_throttle,
            pdl_wait_resource=None if cfg.is_swap_ab else pdl_wait_resource,
            pdl_launch_resource=None if cfg.is_swap_ab else pdl_launch_resource,
        )
        load_b = create_load_b_task(
            cfg,
            gmem_b,
            smem_b,
            work_queue,
            num_k_tiles,
            pdl_wait_resource=pdl_wait_resource if cfg.is_swap_ab else None,
            pdl_launch_resource=pdl_launch_resource if cfg.is_swap_ab else None,
        )
        task_list = [load_a, load_b]

    smem_sfa = None
    smem_sfb = None
    tmem_sfa = None
    tmem_sfb = None
    tmem_sfab = None
    tmem_cast_a = None
    if cfg.has_cast_a:
        gmem_sfa = GmemSfAResource(cfg=cfg, name="GmemSfA")
        smem_sfa = SmemSfAResource(
            cfg=cfg, pipeline_config=pcfgs["smem_sfa"], name="SmemSfA"
        )
        tmem_cast_a = TmemCastAResource(
            cfg=cfg, pipeline_config=pcfgs["tmem_cast_a"], name="TmemCastA"
        )
        load_sfa = create_load_sfa_task(
            cfg,
            gmem_sfa,
            smem_sfa,
            work_queue,
            num_k_tiles,
            pdl_wait_resource=None if cfg.is_swap_ab else pdl_wait_resource,
            pdl_launch_resource=None if cfg.is_swap_ab else pdl_launch_resource,
        )
        cast_a = create_cast_a_task(
            cfg,
            smem_a,
            smem_sfa,
            tmem_cast_a,
            work_queue,
            num_k_tiles,
        )
        task_list += [load_sfa, cast_a]
    if cfg.has_scale_factors:
        gmem_sfa = GmemSfAResource(cfg=cfg, name="GmemSfA")
        gmem_sfb = GmemSfBResource(cfg=cfg, name="GmemSfB")
        if cfg.has_routed_sfs and cfg.uses_ldgsts_routed_sfs:
            if cfg.is_swap_ab:
                smem_sfa = SmemSfAResource(
                    cfg=cfg, pipeline_config=pcfgs["smem_sfa"], name="SmemSfA"
                )
                smem_sfb = SmemSfLdgstsBResource(
                    cfg=cfg,
                    pipeline_config=pcfgs["smem_sfb"],
                    producer_commit_prefetch_depth=(
                        _ldgsts_sfb_producer_commit_prefetch_depth(cfg)
                    ),
                    name="SmemSfB",
                )
            else:
                smem_sfa = SmemSfLdgstsAResource(
                    cfg=cfg,
                    pipeline_config=pcfgs["smem_sfa"],
                    name="SmemSfA",
                )
                smem_sfb = SmemSfBResource(
                    cfg=cfg, pipeline_config=pcfgs["smem_sfb"], name="SmemSfB"
                )
        else:
            smem_sfa = SmemSfAResource(
                cfg=cfg, pipeline_config=pcfgs["smem_sfa"], name="SmemSfA"
            )
            smem_sfb = SmemSfBResource(
                cfg=cfg, pipeline_config=pcfgs["smem_sfb"], name="SmemSfB"
            )

        if cfg.use_combined_sfab_copy:
            tmem_sfab = TmemSfABResource(
                cfg=cfg, pipeline_config=pcfgs["tmem_sfab"], name="TmemSfAb"
            )
        elif cfg.uses_unfused_tmem_sf_copy:
            # Separate CopySf tasks: TmemSf with pipelines
            if cfg.is_swap_ab:
                tmem_sfa = TmemSfAResource(
                    cfg=cfg, pipeline_config=pcfgs["tmem_sfa"], name="TmemSfA"
                )
                if cfg.has_routed_sfs or cfg.sfb_smem_to_tmem_copy == int(
                    SfSmemToTmemCopy.LDS_STTM
                ):
                    tmem_sfb = TmemSfRouteBResource(
                        cfg=cfg,
                        smem_sf_resource=smem_sfb,
                        pipeline_config=pcfgs["tmem_sfb"],
                        name="TmemSfB",
                    )
                else:
                    tmem_sfb = TmemSfBResource(
                        cfg=cfg, pipeline_config=pcfgs["tmem_sfb"], name="TmemSfB"
                    )
            else:
                if cfg.has_routed_sfs:
                    tmem_sfa = TmemSfRouteAResource(
                        cfg=cfg,
                        smem_sf_resource=smem_sfa,
                        pipeline_config=pcfgs["tmem_sfa"],
                        name="TmemSfA",
                    )
                else:
                    tmem_sfa = TmemSfAResource(
                        cfg=cfg, pipeline_config=pcfgs["tmem_sfa"], name="TmemSfA"
                    )
                if cfg.sfb_smem_to_tmem_copy == int(SfSmemToTmemCopy.LDS_STTM):
                    tmem_sfb = TmemSfRouteBResource(
                        cfg=cfg,
                        smem_sf_resource=smem_sfb,
                        pipeline_config=pcfgs["tmem_sfb"],
                        name="TmemSfB",
                    )
                else:
                    tmem_sfb = TmemSfBResource(
                        cfg=cfg, pipeline_config=pcfgs["tmem_sfb"], name="TmemSfB"
                    )
        else:
            # Fused S2T in MMA: no pipeline on TmemSf
            tmem_sfa = TmemSfAResource(cfg=cfg, pipeline_config=None, name="TmemSfA")
            tmem_sfb = TmemSfBResource(cfg=cfg, pipeline_config=None, name="TmemSfB")

        if cfg.fuse_operand_sf_loads:
            task_list = [
                create_fused_load_a_sfa_task(
                    cfg,
                    gmem_a,
                    smem_a,
                    gmem_sfa,
                    smem_sfa,
                    work_queue,
                    num_k_tiles,
                    work_throttle=work_throttle,
                ),
                create_fused_gather_sfb_task(
                    cfg,
                    gmem_b,
                    smem_b,
                    gmem_sfb,
                    smem_sfb,
                    work_queue,
                    num_k_tiles,
                    pdl_wait_resource=pdl_wait_resource,
                    pdl_launch_resource=pdl_launch_resource,
                ),
            ]
        else:
            load_sfa = create_load_sfa_task(
                cfg,
                gmem_sfa,
                smem_sfa,
                work_queue,
                num_k_tiles,
                pdl_wait_resource=None if cfg.is_swap_ab else pdl_wait_resource,
                pdl_launch_resource=None if cfg.is_swap_ab else pdl_launch_resource,
            )
            load_sfb = create_load_sfb_task(
                cfg,
                gmem_sfb,
                smem_sfb,
                work_queue,
                num_k_tiles,
                pdl_wait_resource=pdl_wait_resource if cfg.is_swap_ab else None,
                pdl_launch_resource=pdl_launch_resource if cfg.is_swap_ab else None,
            )
            task_list += [load_sfa, load_sfb]

        if cfg.uses_unfused_tmem_sf_copy:
            if cfg.use_combined_sfab_copy:
                copy_sfab = create_copy_sfab_task(
                    cfg, smem_sfa, smem_sfb, tmem_sfab, work_queue, num_k_tiles
                )
                task_list.append(copy_sfab)
            else:
                copy_sfa = create_copy_sfa_task(
                    cfg, smem_sfa, tmem_sfa, work_queue, num_k_tiles
                )
                copy_sfb = create_copy_sfb_task(
                    cfg, smem_sfb, tmem_sfb, work_queue, num_k_tiles
                )
                task_list += [copy_sfa, copy_sfb]

    # MMA task
    if cfg.has_cast_a:
        mma = create_mma_task(
            cfg,
            smem_a,
            smem_b,
            None,
            None,
            tmem_c,
            work_queue,
            num_k_tiles,
            proxy_cluster=proxy_cluster,
            tmem_cast_a=tmem_cast_a,
        )
    elif cfg.has_scale_factors and cfg.uses_unfused_tmem_sf_copy:
        if cfg.use_combined_sfab_copy:
            mma = create_mma_task(
                cfg,
                smem_a,
                smem_b,
                None,
                None,
                tmem_c,
                work_queue,
                num_k_tiles,
                proxy_cluster=proxy_cluster,
                tmem_sfab=tmem_sfab,
            )
        else:
            mma = create_mma_task(
                cfg,
                smem_a,
                smem_b,
                None,
                None,
                tmem_c,
                work_queue,
                num_k_tiles,
                proxy_cluster=proxy_cluster,
                tmem_sfa=tmem_sfa,
                tmem_sfb=tmem_sfb,
            )
    elif cfg.has_scale_factors:
        mma = create_mma_task(
            cfg,
            smem_a,
            smem_b,
            smem_sfa,
            smem_sfb,
            tmem_c,
            work_queue,
            num_k_tiles,
            proxy_cluster=proxy_cluster,
        )
    else:
        mma = create_mma_task(
            cfg,
            smem_a,
            smem_b,
            None,
            None,
            tmem_c,
            work_queue,
            num_k_tiles,
            proxy_cluster=proxy_cluster,
        )
    smem_dsfp8_sfab = None
    if cfg.has_deepseek_fp8:
        # DeepSeek FP8 uses the normal MMA producer and a per-K-tile epilogue
        # drain that also consumes SmemDeepSeekSfAb dequant scales.
        smem_dsfp8_sfab = SmemDeepSeekSfAbResource(
            cfg=cfg,
            pipeline_config=pcfgs["smem_dsfp8_sfab"],
            name="SmemDeepSeekSfAb",
        )
        load_sfab = create_load_sfab_task(
            cfg,
            smem_dsfp8_sfab,
            work_queue,
            num_k_tiles,
            pdl_wait_resource=pdl_wait_resource,
            pdl_launch_resource=pdl_launch_resource,
        )
        epi = create_epilogue_task_dsfp8(
            cfg, tmem_c, smem_dsfp8_sfab, gmem_c, work_queue, num_k_tiles
        )
        task_list += [mma, load_sfab, epi]
    else:
        epi = create_epilogue_task(cfg, tmem_c, gmem_c, work_queue, num_k_tiles)
        task_list += [mma, epi]

    if cfg.num_padding_warps > 0:
        pad = create_padding_task(cfg, work_queue, num_k_tiles)
        task_list.append(pad)
    # WorkScheduleTask (CLC persistent only)
    if cfg.is_persistent:
        workid = create_workid_task(
            cfg,
            work_queue,
            num_k_tiles,
            work_throttle=work_throttle,
        )
        task_list.append(workid)

    # Dependency graph
    resource_dependency_graph = {
        smem_a: [gmem_a],
        smem_b: [gmem_b],
        gmem_c: [tmem_c],
    }
    if cfg.has_scale_factors and cfg.uses_unfused_tmem_sf_copy:
        resource_dependency_graph[smem_sfa] = [gmem_sfa]
        resource_dependency_graph[smem_sfb] = [gmem_sfb]
        ab_dependencies = [smem_a, smem_b]
        if proxy_cluster is not None:
            resource_dependency_graph[proxy_cluster] = ab_dependencies
            ab_dependencies = [proxy_cluster]
        if cfg.use_combined_sfab_copy:
            resource_dependency_graph[tmem_sfab] = [smem_sfa, smem_sfb]
            resource_dependency_graph[tmem_c] = ab_dependencies + [tmem_sfab]
        else:
            resource_dependency_graph[tmem_sfa] = [smem_sfa]
            resource_dependency_graph[tmem_sfb] = [smem_sfb]
            resource_dependency_graph[tmem_c] = ab_dependencies + [
                tmem_sfa,
                tmem_sfb,
            ]
    elif cfg.has_scale_factors:
        resource_dependency_graph[smem_sfa] = [gmem_sfa]
        resource_dependency_graph[smem_sfb] = [gmem_sfb]
        ab_dependencies = [smem_a, smem_b]
        if proxy_cluster is not None:
            resource_dependency_graph[proxy_cluster] = ab_dependencies
            ab_dependencies = [proxy_cluster]
        resource_dependency_graph[tmem_c] = ab_dependencies + [smem_sfa, smem_sfb]
    elif cfg.has_cast_a:
        resource_dependency_graph[smem_sfa] = [gmem_sfa]
        resource_dependency_graph[tmem_cast_a] = [smem_a, smem_sfa]
        if proxy_cluster is not None:
            resource_dependency_graph[proxy_cluster] = [smem_b]
            resource_dependency_graph[tmem_c] = [tmem_cast_a, proxy_cluster]
        else:
            resource_dependency_graph[tmem_c] = [tmem_cast_a, smem_b]
    else:
        if proxy_cluster is not None:
            resource_dependency_graph[proxy_cluster] = [smem_a, smem_b]
            resource_dependency_graph[tmem_c] = [proxy_cluster]
        else:
            resource_dependency_graph[tmem_c] = [smem_a, smem_b]

    if cfg.has_deepseek_fp8 and smem_dsfp8_sfab is not None:
        # SmemDeepSeekSfAb's GMEM source is raw kernel-param pointers
        # (params.ptrSfA/ptrSfB), not an TS Gmem resource. PDL still orders
        # those raw GMEM reads when a wait resource is present.
        resource_dependency_graph[smem_dsfp8_sfab] = (
            [pdl_wait_resource] if pdl_wait_resource is not None else []
        )
        resource_dependency_graph[gmem_c] = [tmem_c, smem_dsfp8_sfab]

    if cfg.is_persistent:
        for k in list(resource_dependency_graph.keys()):
            resource_dependency_graph[k].append(work_queue)
        if work_throttle is not None:
            resource_dependency_graph[work_throttle] = [
                smem_a if cfg.is_swap_ab or not cfg.has_gather else smem_b
            ]
            resource_dependency_graph[work_queue] = [work_queue, work_throttle]
    if pdl_launch_resource is not None:
        resource_dependency_graph[pdl_launch_resource] = []
    if pdl_wait_resource is not None:
        resource_dependency_graph[pdl_wait_resource] = []
        if cfg.is_swap_ab:
            resource_dependency_graph[gmem_b] = [pdl_wait_resource]
            if cfg.has_scale_factors:
                resource_dependency_graph[gmem_sfb] = [pdl_wait_resource]
        else:
            resource_dependency_graph[gmem_a] = [pdl_wait_resource]
            if cfg.has_cast_a or cfg.has_scale_factors:
                resource_dependency_graph[gmem_sfa] = [pdl_wait_resource]

    # SMEM allocator
    smem_allocator = SmemAllocator()
    smem_resources = (smem_a, smem_b)
    if cfg.has_cast_a:
        smem_resources = smem_resources + (smem_sfa,)
    if cfg.has_scale_factors:
        smem_resources = smem_resources + (smem_sfa, smem_sfb)
    if cfg.has_deepseek_fp8 and smem_dsfp8_sfab is not None:
        smem_resources = smem_resources + (smem_dsfp8_sfab,)
    if cfg.is_persistent:
        smem_resources = smem_resources + (work_queue,)
    smem_resources = smem_resources + (gmem_c,)
    for r in smem_resources:
        smem_allocator.add_resource(r)
    if cutlass.const_expr(cfg.aliases_c_scratch_with_ab):
        # Alias SmemA/B with GmemC scratch to save SMEM when the generated
        # schedule does not require a disjoint epilogue staging window.
        alloc_a = smem_a._alloc if hasattr(smem_a, "_alloc") else smem_a._alloc_a
        alloc_b = smem_b._alloc if hasattr(smem_b, "_alloc") else smem_b._alloc_b
        smem_allocator.add_alias_group(
            [
                [alloc_a, alloc_b],
                [gmem_c._alloc_sc],
            ]
        )
    smem_allocator.compute_layout()

    # TMEM allocator
    tmem_allocator = TmemAllocator()
    # C first (accumulator at offset 0, matching nvfp4_gemm reference)
    tmem_allocator.add_resource(tmem_c)
    if cfg.has_cast_a:
        tmem_allocator.add_resource(tmem_cast_a)
    if cfg.has_scale_factors and cfg.uses_unfused_tmem_sf_copy:
        if cfg.use_combined_sfab_copy:
            tmem_allocator.add_resource(tmem_sfab)
        else:
            tmem_allocator.add_resource(tmem_sfa)
            tmem_allocator.add_resource(tmem_sfb)
    tmem_allocator.compute_layout()

    return task_list, resource_dependency_graph, smem_allocator, tmem_allocator


def build_batched_gemm_task_manager(
    *,
    num_experts=2,
    num_tokens=128,
    top_k=1,
    early_exit_max_token_ctas=0,
    verbose=True,
    **cfg_overrides,
) -> TaskManager:
    """Build and validate the TaskManager (no GPU needed)."""
    with task_scheduling_scope():
        return _build_batched_gemm_task_manager(
            num_experts=num_experts,
            num_tokens=num_tokens,
            top_k=top_k,
            early_exit_max_token_ctas=early_exit_max_token_ctas,
            verbose=verbose,
            **cfg_overrides,
        )


def _build_batched_gemm_task_manager(
    *,
    num_experts=2,
    num_tokens=128,
    top_k=1,
    early_exit_max_token_ctas=0,
    verbose=True,
    **cfg_overrides,
) -> TaskManager:
    _ = resolve_early_exit_max_token_ctas(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        cfg_overrides=cfg_overrides,
        explicit_early_exit_max_token_ctas=early_exit_max_token_ctas,
    )
    cfg = make_config(**cfg_overrides)
    # The validation-only config has no concrete problem K, so
    # compute_num_k_tiles() sees a single tile.  Validate at four tiles to
    # cover the maximum three-stage LDGSTS producer prefetch and the 2x MMA
    # unroll without reporting a false unmatched producer commit.
    num_k_tiles = max(4, compute_num_k_tiles(cfg))
    task_list, dep_graph, smem_alloc, tmem_alloc = _build_schedule_validate(
        cfg,
        num_k_tiles=num_k_tiles,
    )
    return TaskManager(
        tasks=task_list,
        resource_dependency_graph=dep_graph,
        smem_allocator=smem_alloc,
        tmem_allocator=tmem_alloc,
        assume_pdl_wait_completed=_pdl_wait_completed_before_tasks(cfg),
        exhaustive_deadlock_race_check=_exhaustive_deadlock_race_check_enabled(),
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# GPU kernel body and wrapper (@cute.kernel)
# ---------------------------------------------------------------------------


@cute.jit
def _batched_gemm_kernel_bf16_body(
    tma_a_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_b_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_c_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_sfa_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_sfb_desc: cutlass.GridConstant[cuda.TensorMap],
    c_tensor: cute.Tensor,
    sf_c_tensor: cute.Tensor,
    bias_tensor: cute.Tensor,
    scale_c_tensor: cute.Tensor,
    scale_gate_tensor: cute.Tensor,
    gemm1_alpha_tensor: cute.Tensor,
    gemm1_beta_tensor: cute.Tensor,
    gemm1_clamp_limit_tensor: cute.Tensor,
    per_token_sf_a_tensor: cute.Tensor,
    per_token_sf_b_tensor: cute.Tensor,
    tile_idx_tensor: cute.Tensor,
    route_map_tensor: cute.Tensor,
    mn_limit_tensor: cute.Tensor,
    num_non_exiting_ctas_tensor: cute.Tensor,
    total_num_padded_tokens_tensor: cute.Tensor,
    act_tensor: cute.Tensor,
    sfa_gmem_tensor: cute.Tensor,  # SF GMEM for LDGSTS SF (dummy when not LDGSTS)
    sfb_gmem_tensor: cute.Tensor,  # SF GMEM for LDGSTS SF (dummy when not LDGSTS)
    problem_m: cutlass.Int32,
    problem_n: cutlass.Int32,
    problem_k: cutlass.Int32,
    num_tokens: cutlass.Int32,
    num_experts: cutlass.Int32,
    tile_sched_params: object,
    cfg: cutlass.Constexpr[BatchedGemmConfig],
    early_exit_max_token_ctas: cutlass.Int32,
) -> None:
    """GPU kernel for BF16/FP4 BatchedGemm.

    tma_sfa_desc, tma_sfb_desc: SF TMA descriptors (FP4 only; dummy when BF16).
    route_map_tensor, mn_limit_tensor, act_tensor: gather only; dummy when off.
    """
    k_tile_cnt = (problem_k + Int32(cfg.tile_k - 1)) // Int32(cfg.tile_k)
    problem_m_tiles = (problem_m + Int32(cfg.tile_m - 1)) // Int32(cfg.tile_m)
    problem_n_tiles = (problem_n + Int32(cfg.tile_n - 1)) // Int32(cfg.tile_n)

    warp_idx = cute.arch.warp_idx()

    pcfgs = _make_pipeline_configs(cfg)

    num_non_exiting_ctas_value = None
    if cutlass.const_expr(
        cfg.is_persistent
        and cfg.use_early_exit
        and cfg.do_pdl_wait_for_num_non_exiting_ctas
        and not _pdl_wait_completed_before_tasks(cfg)
    ):
        num_non_exiting_ctas_view = cutlass.make_array_view(num_non_exiting_ctas_tensor)
        num_non_exiting_ctas_value = num_non_exiting_ctas_view.load(
            idx=Int32(0), vector_size=1
        )[0]

    tma_a_ptr = tma_a_desc.get_ptr()
    tma_b_ptr = tma_b_desc.get_ptr()
    tma_c_ptr = tma_c_desc.get_ptr()
    prims.prefetch_tensormap(tma_a_ptr)
    prims.prefetch_tensormap(tma_b_ptr)
    if cutlass.const_expr(cfg.has_scale_factors):
        prims.prefetch_tensormap(tma_sfa_desc.get_ptr())
        prims.prefetch_tensormap(tma_sfb_desc.get_ptr())
    prims.prefetch_tensormap(tma_c_ptr)

    # tile_idx_tensor: maps token tiles to expert indices.
    # mn_limit_tensor: maps token tiles to TRT-LLM Gen absolute end-row limits.
    tile_idx_view = cutlass.make_array_view(tile_idx_tensor)
    mn_limit_view = cutlass.make_array_view(mn_limit_tensor)
    rpe = (
        problem_m // num_experts
        if cutlass.const_expr(cfg.route_act == int(RouteImpl.TMA))
        else None
    )
    gmem_a = GmemAResource(
        cfg=cfg,
        tile_idx_view=tile_idx_view,
        mn_limit_view=mn_limit_view,
        rows_per_expert=rpe,
        name="GmemA",
    )
    gmem_b = GmemBResource(
        cfg=cfg,
        tile_idx_view=tile_idx_view,
        mn_limit_view=mn_limit_view,
        name="GmemB",
    )

    # Activation routing: LDGSTS gather or TMA gather4. Do not create a
    # default SMEM resource and then replace it inside a staged branch: PyIR
    # treats that as mutating a Python object from staged control flow.
    if cutlass.const_expr(cfg.has_gather and cfg.is_swap_ab):
        route_map_view = cutlass.make_array_view(route_map_tensor)
        mn_limit_view = cutlass.make_array_view(mn_limit_tensor)
        act_byte_ptr = cutlass.make_array_view(act_tensor).data_ptr()
        act_stride = problem_k * Int32(cfg.dtype_b_bits) // Int32(8)
        smem_a = SmemAResource(
            cfg=cfg,
            tma_a_desc=tma_a_ptr,
            pipeline_config=pcfgs["smem_a"],
            name="SmemA",
        )
        smem_b = SmemGatherResource(
            cfg=cfg,
            act_gmem_ptr=act_byte_ptr,
            act_stride_bytes=act_stride,
            route_map=route_map_view,
            mn_limit=mn_limit_view,
            pipeline_config=pcfgs["smem_gather"],
            _operand="b",
            name="SmemGather",
        )
    elif cutlass.const_expr(cfg.has_gather):
        route_map_view = cutlass.make_array_view(route_map_tensor)
        mn_limit_view = cutlass.make_array_view(mn_limit_tensor)
        act_byte_ptr = cutlass.make_array_view(act_tensor).data_ptr()
        act_stride = problem_k * Int32(cfg.dtype_a_bits) // Int32(8)
        smem_a = SmemGatherResource(
            cfg=cfg,
            act_gmem_ptr=act_byte_ptr,
            act_stride_bytes=act_stride,
            route_map=route_map_view,
            mn_limit=mn_limit_view,
            pipeline_config=pcfgs["smem_gather"],
            _operand="a",
            name="SmemGather",
        )
        smem_b = SmemBResource(
            cfg=cfg,
            tma_b_desc=tma_b_ptr,
            pipeline_config=pcfgs["smem_b"],
            name="SmemB",
        )
    elif cutlass.const_expr(cfg.has_tma_route and cfg.is_swap_ab):
        route_map_view = cutlass.make_array_view(route_map_tensor)
        mn_limit_view = cutlass.make_array_view(mn_limit_tensor)
        smem_a = SmemAResource(
            cfg=cfg,
            tma_a_desc=tma_a_ptr,
            pipeline_config=pcfgs["smem_a"],
            name="SmemA",
        )
        smem_b = SmemTmaGatherResource(
            cfg=cfg,
            tma_desc=tma_b_ptr,
            route_map=route_map_view,
            mn_limit=mn_limit_view,
            pipeline_config=pcfgs["smem_b"],
            _operand="b",
            name="SmemTmaGather",
        )
    elif cutlass.const_expr(cfg.has_tma_route):
        route_map_view = cutlass.make_array_view(route_map_tensor)
        mn_limit_view = cutlass.make_array_view(mn_limit_tensor)
        smem_a = SmemTmaGatherResource(
            cfg=cfg,
            tma_desc=tma_a_ptr,
            route_map=route_map_view,
            mn_limit=mn_limit_view,
            pipeline_config=pcfgs["smem_a"],
            _operand="a",
            name="SmemTmaGather",
        )
        smem_b = SmemBResource(
            cfg=cfg,
            tma_b_desc=tma_b_ptr,
            pipeline_config=pcfgs["smem_b"],
            name="SmemB",
        )
    else:
        smem_a = SmemAResource(
            cfg=cfg,
            tma_a_desc=tma_a_ptr,
            pipeline_config=pcfgs["smem_a"],
            name="SmemA",
        )
        smem_b = SmemBResource(
            cfg=cfg,
            tma_b_desc=tma_b_ptr,
            pipeline_config=pcfgs["smem_b"],
            name="SmemB",
        )

    tmem_c = TmemCResource(cfg=cfg, pipeline_config=pcfgs["tmem_c"], name="TmemC")
    tmem_sfa = None
    tmem_sfb = None
    tmem_sfab = None
    tmem_cast_a = None

    if cutlass.const_expr(cfg.has_cast_a):
        tma_sfa_ptr = tma_sfa_desc.get_ptr()
        gmem_sfa = GmemSfAResource(
            cfg=cfg,
            tile_idx_view=tile_idx_view,
            problem_m_tiles=problem_m_tiles,
            name="GmemSfA",
        )
        smem_sfa = SmemSfAResource(
            cfg=cfg,
            tma_sfa_desc=tma_sfa_ptr,
            pipeline_config=pcfgs["smem_sfa"],
            name="SmemSfA",
        )
        tmem_cast_a = TmemCastAResource(
            cfg=cfg,
            pipeline_config=pcfgs["tmem_cast_a"],
            name="TmemCastA",
        )

    # Scale factor resources (FP4/FP8 only)
    if cutlass.const_expr(cfg.has_scale_factors):
        tma_sfa_ptr = tma_sfa_desc.get_ptr()
        tma_sfb_ptr = tma_sfb_desc.get_ptr()
        gmem_sfa = GmemSfAResource(
            cfg=cfg,
            tile_idx_view=tile_idx_view,
            problem_m_tiles=problem_m_tiles,
            name="GmemSfA",
        )
        gmem_sfb = GmemSfBResource(
            cfg=cfg,
            tile_idx_view=tile_idx_view,
            problem_n_tiles=problem_n_tiles,
            name="GmemSfB",
        )

        # Routed operand's SF uses gather4 when route_sfs_act == TMA.
        # non-swapAB: A=activations (routed) → SFA gathered.
        # swapAB: B=activations (routed) → SFB gathered.
        if cutlass.const_expr(cfg.has_routed_sfs and cfg.uses_tma_routed_sfs):
            route_map_sf = cutlass.make_array_view(route_map_tensor)
            mn_limit_sf = cutlass.make_array_view(mn_limit_tensor)
            if cutlass.const_expr(cfg.is_swap_ab):
                smem_sfa = SmemSfAResource(
                    cfg=cfg,
                    tma_sfa_desc=tma_sfa_ptr,
                    pipeline_config=pcfgs["smem_sfa"],
                    name="SmemSfA",
                )
                smem_sfb = SmemSfGatherBResource(
                    cfg=cfg,
                    tma_sf_desc=tma_sfb_ptr,
                    route_map=route_map_sf,
                    mn_limit=mn_limit_sf,
                    pipeline_config=pcfgs["smem_sfb"],
                    name="SmemSfB",
                )
            else:
                smem_sfa = SmemSfGatherAResource(
                    cfg=cfg,
                    tma_sf_desc=tma_sfa_ptr,
                    route_map=route_map_sf,
                    mn_limit=mn_limit_sf,
                    pipeline_config=pcfgs["smem_sfa"],
                    name="SmemSfA",
                )
                smem_sfb = SmemSfBResource(
                    cfg=cfg,
                    tma_sfb_desc=tma_sfb_ptr,
                    pipeline_config=pcfgs["smem_sfb"],
                    name="SmemSfB",
                )
        elif cutlass.const_expr(cfg.has_routed_sfs and cfg.uses_ldgsts_routed_sfs):
            # LDGSTS/LDG+STS SF: per-thread loading with route map lookup.
            route_map_sf = cutlass.make_array_view(route_map_tensor)
            mn_limit_sf = cutlass.make_array_view(mn_limit_tensor)
            # SF GMEM stride is the number of scale-factor elements per row,
            # padded to 16-byte alignment. NVFP4 uses K/16; MX uses K/32.
            sf_stride_unpadded = problem_k // Int32(cfg.sf_vec_size)
            sf_stride = (sf_stride_unpadded + Int32(15)) // Int32(16) * Int32(16)
            if cutlass.const_expr(cfg.is_swap_ab):
                # SFB is routed (activations)
                sf_gmem_view = cutlass.make_array_view(sfb_gmem_tensor)
                smem_sfa = SmemSfAResource(
                    cfg=cfg,
                    tma_sfa_desc=tma_sfa_ptr,
                    pipeline_config=pcfgs["smem_sfa"],
                    name="SmemSfA",
                )
                smem_sfb = SmemSfLdgstsBResource(
                    cfg=cfg,
                    sf_gmem_ptr=sf_gmem_view,
                    sf_gmem_stride=sf_stride,
                    route_map=route_map_sf,
                    mn_limit=mn_limit_sf,
                    pipeline_config=pcfgs["smem_sfb"],
                    producer_commit_prefetch_depth=(
                        _ldgsts_sfb_producer_commit_prefetch_depth(cfg)
                    ),
                    name="SmemSfB",
                )
            else:
                # SFA is routed (activations)
                sf_gmem_view = cutlass.make_array_view(sfa_gmem_tensor)
                smem_sfa = SmemSfLdgstsAResource(
                    cfg=cfg,
                    sf_gmem_ptr=sf_gmem_view,
                    sf_gmem_stride=sf_stride,
                    route_map=route_map_sf,
                    mn_limit=mn_limit_sf,
                    pipeline_config=pcfgs["smem_sfa"],
                    name="SmemSfA",
                )
                smem_sfb = SmemSfBResource(
                    cfg=cfg,
                    tma_sfb_desc=tma_sfb_ptr,
                    pipeline_config=pcfgs["smem_sfb"],
                    name="SmemSfB",
                )
        else:
            smem_sfa = SmemSfAResource(
                cfg=cfg,
                tma_sfa_desc=tma_sfa_ptr,
                pipeline_config=pcfgs["smem_sfa"],
                name="SmemSfA",
            )
            smem_sfb = SmemSfBResource(
                cfg=cfg,
                tma_sfb_desc=tma_sfb_ptr,
                pipeline_config=pcfgs["smem_sfb"],
                name="SmemSfB",
            )
        # TmemSf resources: when SF is routed, separate CopySf tasks need pipelines.
        # When not routed, S2T is fused in MMA (pipeline_config=None).
        if cutlass.const_expr(cfg.use_combined_sfab_copy):
            tmem_sfab = TmemSfABResource(
                cfg=cfg,
                pipeline_config=pcfgs["tmem_sfab"],
                name="TmemSfAb",
            )
        elif cutlass.const_expr(cfg.uses_unfused_tmem_sf_copy):
            # Routed SF uses TmemSfRouteResource (LDS+STTM).
            # Non-routed SF uses standard TmemSfA/BResource (tcgen05_cp S2T).
            # Both get AsyncUmma pipeline for separate CopySf tasks.
            if cutlass.const_expr(cfg.is_swap_ab):
                # swapAB: SFB is routed (activations), SFA is not (weights)
                tmem_sfa = TmemSfAResource(
                    cfg=cfg, pipeline_config=pcfgs["tmem_sfa"], name="TmemSfA"
                )
                if cutlass.const_expr(
                    cfg.has_routed_sfs
                    or cfg.sfb_smem_to_tmem_copy == int(SfSmemToTmemCopy.LDS_STTM)
                ):
                    tmem_sfb = TmemSfRouteBResource(
                        cfg=cfg,
                        smem_sf_resource=smem_sfb,
                        pipeline_config=pcfgs["tmem_sfb"],
                        name="TmemSfB",
                    )
                else:
                    tmem_sfb = TmemSfBResource(
                        cfg=cfg, pipeline_config=pcfgs["tmem_sfb"], name="TmemSfB"
                    )
            else:
                # non-swapAB: SFA is routed (activations), SFB is not (weights)
                if cutlass.const_expr(cfg.has_routed_sfs):
                    tmem_sfa = TmemSfRouteAResource(
                        cfg=cfg,
                        smem_sf_resource=smem_sfa,
                        pipeline_config=pcfgs["tmem_sfa"],
                        name="TmemSfA",
                    )
                else:
                    tmem_sfa = TmemSfAResource(
                        cfg=cfg, pipeline_config=pcfgs["tmem_sfa"], name="TmemSfA"
                    )
                if cutlass.const_expr(
                    cfg.sfb_smem_to_tmem_copy == int(SfSmemToTmemCopy.LDS_STTM)
                ):
                    tmem_sfb = TmemSfRouteBResource(
                        cfg=cfg,
                        smem_sf_resource=smem_sfb,
                        pipeline_config=pcfgs["tmem_sfb"],
                        name="TmemSfB",
                    )
                else:
                    tmem_sfb = TmemSfBResource(
                        cfg=cfg, pipeline_config=pcfgs["tmem_sfb"], name="TmemSfB"
                    )
        else:
            # Not routed: fused S2T in MMA, no pipeline
            tmem_sfa = TmemSfAResource(cfg=cfg, pipeline_config=None, name="TmemSfA")
            tmem_sfb = TmemSfBResource(cfg=cfg, pipeline_config=None, name="TmemSfB")

    smem_dsfp8_sfab = None
    if cutlass.const_expr(cfg.has_deepseek_fp8):
        smem_dsfp8_sfab = SmemDeepSeekSfAbResource(
            cfg=cfg,
            sfa_gmem_base=cutlass.make_array_view(sfa_gmem_tensor),
            sfb_gmem_base=cutlass.make_array_view(sfb_gmem_tensor),
            tile_idx_view=tile_idx_view,
            mn_limit_view=mn_limit_view,
            route_map_view=cutlass.make_array_view(route_map_tensor),
            problem_m=problem_m,
            problem_n=problem_n,
            problem_k=problem_k,
            num_tokens=num_tokens,
            total_num_padded_tokens_tensor=total_num_padded_tokens_tensor,
            pipeline_config=pcfgs["smem_dsfp8_sfab"],
            name="SmemDeepSeekSfAb",
        )

    gmem_c = GmemCResource(
        cfg=cfg,
        c_tensor=c_tensor,
        sf_c_tensor=sf_c_tensor,
        tma_c_desc=tma_c_ptr,
        bias_tensor=bias_tensor,
        scale_c_tensor=scale_c_tensor,
        scale_gate_tensor=scale_gate_tensor,
        gemm1_alpha_tensor=gemm1_alpha_tensor,
        gemm1_beta_tensor=gemm1_beta_tensor,
        gemm1_clamp_limit_tensor=gemm1_clamp_limit_tensor,
        per_token_sf_a_tensor=per_token_sf_a_tensor,
        per_token_sf_b_tensor=per_token_sf_b_tensor,
        route_map_view=cutlass.make_array_view(route_map_tensor),
        tile_idx_view=tile_idx_view,
        mn_limit_view=mn_limit_view,
        total_num_padded_tokens_tensor=total_num_padded_tokens_tensor,
        problem_m=problem_m,
        problem_n=problem_n,
        name="GmemC",
    )

    # WorkQueue — CLC persistent or static (non-persistent)
    if cutlass.const_expr(cfg.is_persistent):
        if cutlass.const_expr(
            cfg.use_early_exit
            and (not cfg.use_pdl or _pdl_wait_completed_before_tasks(cfg))
        ):
            num_non_exiting_ctas_view = cutlass.make_array_view(
                num_non_exiting_ctas_tensor
            )
            num_non_exiting_ctas_value = num_non_exiting_ctas_view.load(
                idx=Int32(0), vector_size=1
            )[0]
        clc_response_ptr = cute.arch.alloc_smem(cutlass.Int128, cfg.num_stages_workid)
        tile_sched_cfg = (
            TileSchedulerConfig.create_clc_dynamic_persistent_tile_scheduler_params(
                tile_scheduler_params=tile_sched_params,
                response_ptr=clc_response_ptr,
            )
        )
        work_queue = BatchedGemmWorkQueue(
            tile_scheduler_config=tile_sched_cfg,
            cfg=cfg,
            num_non_exiting_ctas_tensor=num_non_exiting_ctas_tensor,
            num_non_exiting_ctas_value=num_non_exiting_ctas_value,
            pipeline_config=pcfgs["workid"],
            name="WorkQueue",
        )
    else:
        tile_sched_cfg = (
            TileSchedulerConfig.create_static_persistent_tile_scheduler_params(
                tile_scheduler_params=tile_sched_params,
            )
        )
        work_queue = WorkQueue(tile_scheduler_config=tile_sched_cfg, name="WorkQueue")

    work_throttle = None
    if cutlass.const_expr(cfg.use_work_throttle_barrier):
        work_throttle = WorkThrottleBarrierResource(
            pipeline_config=pcfgs["work_throttle"],
            name="WorkThrottle",
        )

    # SMEM allocator
    smem_allocator = SmemAllocator()
    smem_resources: tuple[Any, ...] = (smem_a, smem_b)
    if cutlass.const_expr(cfg.has_cast_a):
        smem_resources = smem_resources + (smem_sfa,)
    if cutlass.const_expr(cfg.has_scale_factors):
        smem_resources = smem_resources + (smem_sfa, smem_sfb)
    if cutlass.const_expr(cfg.has_deepseek_fp8):
        smem_resources = smem_resources + (smem_dsfp8_sfab,)
    if cutlass.const_expr(cfg.is_persistent):
        smem_resources = smem_resources + (work_queue,)
    smem_resources = smem_resources + (gmem_c,)
    for r in smem_resources:
        smem_allocator.add_resource(r)
    if cutlass.const_expr(cfg.aliases_c_scratch_with_ab):
        alloc_a = smem_a._alloc if hasattr(smem_a, "_alloc") else smem_a._alloc_a
        alloc_b = smem_b._alloc if hasattr(smem_b, "_alloc") else smem_b._alloc_b
        smem_allocator.add_alias_group(
            [
                [alloc_a, alloc_b],
                [gmem_c._alloc_sc],
            ]
        )
    tmem_ptr_alloc = smem_allocator.add_tmem_ptr(
        SmemAllocation("tmem_ptr_i32", dtype=cutlass.Int32, alignment=4)
    )
    tmem_dealloc_mbar_alloc = None
    if cutlass.const_expr(cfg.has_cluster):
        tmem_dealloc_mbar_alloc = smem_allocator.add(
            SmemAllocation("tmem_dealloc_mbar", dtype=cutlass.Int64, alignment=8)
        )
    smem_allocator.compute_layout()

    # TMEM allocator
    tmem_allocator = TmemAllocator()
    # C first (accumulator at offset 0, matching nvfp4_gemm reference)
    tmem_allocator.add_resource(tmem_c)
    if cutlass.const_expr(cfg.has_cast_a):
        tmem_allocator.add_resource(tmem_cast_a)
    if cutlass.const_expr(cfg.has_scale_factors and cfg.uses_unfused_tmem_sf_copy):
        if cutlass.const_expr(cfg.use_combined_sfab_copy):
            tmem_allocator.add_resource(tmem_sfab)
        else:
            tmem_allocator.add_resource(tmem_sfa)
            tmem_allocator.add_resource(tmem_sfb)
    tmem_allocator.compute_layout()

    pdl_wait_completed_before_tasks = cutlass.const_expr(
        _pdl_wait_completed_before_tasks(cfg)
    )
    pdl_wait_resource = (
        PdlWaitBarrier(name="PdlWait")
        if cutlass.const_expr(cfg.use_pdl and not pdl_wait_completed_before_tasks)
        else None
    )
    pdl_launch_resource = (
        PdlLaunchBarrier(name="PdlLaunch") if cutlass.const_expr(cfg.use_pdl) else None
    )

    # Tasks
    proxy_cluster = None
    if cutlass.const_expr(cfg.has_gather):
        if cutlass.const_expr(cfg.is_swap_ab):
            if cutlass.const_expr(cfg.fuse_operand_sf_loads):
                load_a = create_fused_load_a_sfa_task(
                    cfg,
                    gmem_a,
                    smem_a,
                    gmem_sfa,
                    smem_sfa,
                    work_queue,
                    k_tile_cnt,
                    work_throttle=work_throttle,
                )
                gather = create_fused_gather_sfb_task(
                    cfg,
                    gmem_b,
                    smem_b,
                    gmem_sfb,
                    smem_sfb,
                    work_queue,
                    k_tile_cnt,
                    pdl_wait_resource=pdl_wait_resource,
                    pdl_launch_resource=pdl_launch_resource,
                )
            else:
                load_a = create_load_a_task(
                    cfg,
                    gmem_a,
                    smem_a,
                    work_queue,
                    k_tile_cnt,
                    work_throttle=work_throttle,
                )
                gather = create_gather_task(
                    cfg,
                    gmem_b,
                    smem_b,
                    work_queue,
                    k_tile_cnt,
                    pdl_wait_resource=pdl_wait_resource,
                    pdl_launch_resource=pdl_launch_resource,
                )
            tma_smem = smem_a  # TMA resource for SyncTask
            gather_smem = smem_b  # Gather resource for SyncTask
            task_list = [load_a, gather]
        else:
            gather = create_gather_task(
                cfg,
                gmem_a,
                smem_a,
                work_queue,
                k_tile_cnt,
                pdl_wait_resource=pdl_wait_resource,
                pdl_launch_resource=pdl_launch_resource,
            )
            load_b = create_load_b_task(
                cfg,
                gmem_b,
                smem_b,
                work_queue,
                k_tile_cnt,
                work_throttle=work_throttle,
            )
            gather_smem = smem_a
            tma_smem = smem_b
            task_list = [gather, load_b]

        # 2-CTA + gather: add proxy barrier + sync task
        if cutlass.const_expr(cfg.has_cluster and cfg.num_sync_warps > 0):
            proxy_cluster = ProxyClusterBarrierResource(
                cfg=cfg,
                pipeline_config=pcfgs.get("proxy"),
                name="ProxyCluster",
            )
            sync = create_sync_task(
                cfg,
                proxy_cluster,
                gather_smem,
                tma_smem,
                work_queue,
                k_tile_cnt,
                sync_warp_idx=cfg.sync_warp_idx,
            )
            task_list.append(sync)
    else:
        load_a = create_load_a_task(
            cfg,
            gmem_a,
            smem_a,
            work_queue,
            k_tile_cnt,
            work_throttle=work_throttle,
            pdl_wait_resource=(
                None if cutlass.const_expr(cfg.is_swap_ab) else pdl_wait_resource
            ),
            pdl_launch_resource=(
                None if cutlass.const_expr(cfg.is_swap_ab) else pdl_launch_resource
            ),
        )
        load_b = create_load_b_task(
            cfg,
            gmem_b,
            smem_b,
            work_queue,
            k_tile_cnt,
            pdl_wait_resource=(
                pdl_wait_resource if cutlass.const_expr(cfg.is_swap_ab) else None
            ),
            pdl_launch_resource=(
                pdl_launch_resource if cutlass.const_expr(cfg.is_swap_ab) else None
            ),
        )
        task_list = [load_a, load_b]

    if cutlass.const_expr(cfg.has_cast_a):
        load_sfa = create_load_sfa_task(
            cfg,
            gmem_sfa,
            smem_sfa,
            work_queue,
            k_tile_cnt,
            pdl_wait_resource=(
                None if cutlass.const_expr(cfg.is_swap_ab) else pdl_wait_resource
            ),
            pdl_launch_resource=(
                None if cutlass.const_expr(cfg.is_swap_ab) else pdl_launch_resource
            ),
        )
        cast_a = create_cast_a_task(
            cfg,
            smem_a,
            smem_sfa,
            tmem_cast_a,
            work_queue,
            k_tile_cnt,
        )
        task_list += [load_sfa, cast_a]

    if cutlass.const_expr(cfg.has_scale_factors and not cfg.fuse_operand_sf_loads):
        load_sfa = create_load_sfa_task(
            cfg,
            gmem_sfa,
            smem_sfa,
            work_queue,
            k_tile_cnt,
            pdl_wait_resource=(
                None if cutlass.const_expr(cfg.is_swap_ab) else pdl_wait_resource
            ),
            pdl_launch_resource=(
                None if cutlass.const_expr(cfg.is_swap_ab) else pdl_launch_resource
            ),
        )
        load_sfb = create_load_sfb_task(
            cfg,
            gmem_sfb,
            smem_sfb,
            work_queue,
            k_tile_cnt,
            pdl_wait_resource=(
                pdl_wait_resource if cutlass.const_expr(cfg.is_swap_ab) else None
            ),
            pdl_launch_resource=(
                pdl_launch_resource if cutlass.const_expr(cfg.is_swap_ab) else None
            ),
        )
        task_list += [load_sfa, load_sfb]
        if cutlass.const_expr(cfg.uses_unfused_tmem_sf_copy):
            # Separate CopySf tasks for SMEM→TMEM when SF is routed.
            if cutlass.const_expr(cfg.use_combined_sfab_copy):
                copy_sfab = create_copy_sfab_task(
                    cfg, smem_sfa, smem_sfb, tmem_sfab, work_queue, k_tile_cnt
                )
                task_list.append(copy_sfab)
            else:
                copy_sfa = create_copy_sfa_task(
                    cfg, smem_sfa, tmem_sfa, work_queue, k_tile_cnt
                )
                copy_sfb = create_copy_sfb_task(
                    cfg, smem_sfb, tmem_sfb, work_queue, k_tile_cnt
                )
                task_list += [copy_sfa, copy_sfb]

    # With separate CopySf: MMA consumes TmemSf, not SmemSf.
    # Otherwise MMA consumes SmemSf and does fused S2T in producer_work.
    if cutlass.const_expr(cfg.has_scale_factors and cfg.uses_unfused_tmem_sf_copy):
        # MMA consumes TmemSf — SF already in TMEM from CopySf tasks
        smem_sfa_for_mma = None
        smem_sfb_for_mma = None
    elif cutlass.const_expr(cfg.has_scale_factors):
        # MMA consumes SmemSf — does fused S2T in producer_work
        smem_sfa_for_mma = smem_sfa
        smem_sfb_for_mma = smem_sfb
    else:
        smem_sfa_for_mma = None
        smem_sfb_for_mma = None
    # Pass TmemSf to MMA when separate CopySf tasks are enabled.
    tmem_sfa_for_mma = (
        tmem_sfa
        if cutlass.const_expr(
            cfg.has_scale_factors
            and cfg.uses_unfused_tmem_sf_copy
            and not cfg.use_combined_sfab_copy
        )
        else None
    )
    tmem_sfb_for_mma = (
        tmem_sfb
        if cutlass.const_expr(
            cfg.has_scale_factors
            and cfg.uses_unfused_tmem_sf_copy
            and not cfg.use_combined_sfab_copy
        )
        else None
    )
    tmem_sfab_for_mma = (
        tmem_sfab
        if cutlass.const_expr(cfg.has_scale_factors and cfg.use_combined_sfab_copy)
        else None
    )
    tmem_cast_a_for_mma = tmem_cast_a if cutlass.const_expr(cfg.has_cast_a) else None

    mma = create_mma_task(
        cfg,
        smem_a,
        smem_b,
        smem_sfa_for_mma,
        smem_sfb_for_mma,
        tmem_c,
        work_queue,
        k_tile_cnt,
        proxy_cluster=proxy_cluster,
        tmem_sfa=tmem_sfa_for_mma,
        tmem_sfb=tmem_sfb_for_mma,
        tmem_sfab=tmem_sfab_for_mma,
        tmem_cast_a=tmem_cast_a_for_mma,
    )
    if cutlass.const_expr(cfg.has_deepseek_fp8):
        load_sfab = create_load_sfab_task(
            cfg,
            smem_dsfp8_sfab,
            work_queue,
            k_tile_cnt,
            pdl_wait_resource=pdl_wait_resource,
            pdl_launch_resource=pdl_launch_resource,
        )
        epi = create_epilogue_task_dsfp8(
            cfg, tmem_c, smem_dsfp8_sfab, gmem_c, work_queue, k_tile_cnt
        )
        task_list += [mma, load_sfab, epi]
    else:
        epi = create_epilogue_task(cfg, tmem_c, gmem_c, work_queue, k_tile_cnt)
        task_list += [mma, epi]

    if cutlass.const_expr(cfg.num_padding_warps > 0):
        pad = create_padding_task(cfg, work_queue, k_tile_cnt)
        task_list.append(pad)
    # WorkScheduleTask (CLC persistent only)
    if cutlass.const_expr(cfg.is_persistent):
        workid = create_workid_task(
            cfg,
            work_queue,
            k_tile_cnt,
            work_throttle=work_throttle,
        )
        task_list.append(workid)

    resource_dependency_graph = {
        smem_a: [gmem_a],
        smem_b: [gmem_b],
        gmem_c: [tmem_c],
    }
    if cutlass.const_expr(cfg.has_scale_factors and cfg.uses_unfused_tmem_sf_copy):
        resource_dependency_graph[smem_sfa] = [gmem_sfa]
        resource_dependency_graph[smem_sfb] = [gmem_sfb]
        ab_dependencies = [smem_a, smem_b]
        if cutlass.const_expr(
            cfg.has_gather and cfg.has_cluster and cfg.num_sync_warps > 0
        ):
            resource_dependency_graph[proxy_cluster] = ab_dependencies
            ab_dependencies = [proxy_cluster]
        if cutlass.const_expr(cfg.use_combined_sfab_copy):
            resource_dependency_graph[tmem_sfab] = [smem_sfa, smem_sfb]
            resource_dependency_graph[tmem_c] = ab_dependencies + [tmem_sfab]
        else:
            resource_dependency_graph[tmem_sfa] = [smem_sfa]
            resource_dependency_graph[tmem_sfb] = [smem_sfb]
            # Separate CopySf: TmemC depends on the A/B readiness proxy (when
            # clustered gather is active) and both TMEM scale-factor rings.
            resource_dependency_graph[tmem_c] = ab_dependencies + [
                tmem_sfa,
                tmem_sfb,
            ]
    elif cutlass.const_expr(cfg.has_scale_factors):
        resource_dependency_graph[smem_sfa] = [gmem_sfa]
        resource_dependency_graph[smem_sfb] = [gmem_sfb]
        ab_dependencies = [smem_a, smem_b]
        if cutlass.const_expr(
            cfg.has_gather and cfg.has_cluster and cfg.num_sync_warps > 0
        ):
            resource_dependency_graph[proxy_cluster] = ab_dependencies
            ab_dependencies = [proxy_cluster]
        # Fused S2T+MMA waits for the clustered A/B readiness proxy (when
        # present) as well as both SMEM scale-factor rings.
        resource_dependency_graph[tmem_c] = ab_dependencies + [smem_sfa, smem_sfb]
    elif cutlass.const_expr(cfg.has_cast_a):
        resource_dependency_graph[smem_sfa] = [gmem_sfa]
        resource_dependency_graph[tmem_cast_a] = [smem_a, smem_sfa]
        if cutlass.const_expr(
            cfg.has_gather and cfg.has_cluster and cfg.num_sync_warps > 0
        ):
            resource_dependency_graph[proxy_cluster] = [smem_b]
            resource_dependency_graph[tmem_c] = [tmem_cast_a, proxy_cluster]
        else:
            resource_dependency_graph[tmem_c] = [tmem_cast_a, smem_b]
    elif cutlass.const_expr(
        cfg.has_gather and cfg.has_cluster and cfg.num_sync_warps > 0
    ):
        # 2-CTA+gather: MMA depends on proxy (which depends on gather+TMA)
        resource_dependency_graph[proxy_cluster] = [smem_a, smem_b]
        resource_dependency_graph[tmem_c] = [proxy_cluster]
    else:
        resource_dependency_graph[tmem_c] = [smem_a, smem_b]

    if cutlass.const_expr(cfg.has_deepseek_fp8):
        resource_dependency_graph[smem_dsfp8_sfab] = (
            [pdl_wait_resource]
            if cutlass.const_expr(pdl_wait_resource is not None)
            else []
        )
        resource_dependency_graph[gmem_c] = [tmem_c, smem_dsfp8_sfab]

    if cutlass.const_expr(cfg.is_persistent):
        for k in list(resource_dependency_graph.keys()):
            resource_dependency_graph[k].append(work_queue)
        if cutlass.const_expr(work_throttle is not None):
            throttle_producer = (
                smem_a
                if cutlass.const_expr(cfg.is_swap_ab or not cfg.has_gather)
                else smem_b
            )
            resource_dependency_graph[work_throttle] = [throttle_producer]
            resource_dependency_graph[work_queue] = [work_queue, work_throttle]
    if cutlass.const_expr(pdl_launch_resource is not None):
        resource_dependency_graph[pdl_launch_resource] = []
    if cutlass.const_expr(pdl_wait_resource is not None):
        resource_dependency_graph[pdl_wait_resource] = []
        if cutlass.const_expr(cfg.is_swap_ab):
            resource_dependency_graph[gmem_b] = [pdl_wait_resource]
            if cutlass.const_expr(cfg.has_scale_factors):
                resource_dependency_graph[gmem_sfb] = [pdl_wait_resource]
        else:
            resource_dependency_graph[gmem_a] = [pdl_wait_resource]
            if cutlass.const_expr(cfg.has_cast_a or cfg.has_scale_factors):
                resource_dependency_graph[gmem_sfa] = [pdl_wait_resource]
    task_manager_cls = (
        TaskManager if _task_manager_verify_enabled() else _ProductionTaskManager
    )
    task_manager = task_manager_cls(
        tasks=task_list,
        resource_dependency_graph=resource_dependency_graph,
        smem_allocator=smem_allocator,
        tmem_allocator=tmem_allocator,
        assume_pdl_wait_completed=pdl_wait_completed_before_tasks,
        exhaustive_deadlock_race_check=cutlass.const_expr(
            _exhaustive_deadlock_race_check_enabled()
        ),
    )

    # Early exit for dynamic-batch kernels: a max token-CTA grid is launched
    # for CUDA graph reuse and inactive CTAs skip execution.  The guard sits
    # here -- below resource/task construction -- so allocator registration
    # (trace-time layout bookkeeping) stays outside runtime control flow.
    if cutlass.const_expr(cfg.use_early_exit and not cfg.is_persistent):
        num_non_exiting_ctas_view = cutlass.make_array_view(num_non_exiting_ctas_tensor)
        num_non_exiting_ctas = num_non_exiting_ctas_view.load(
            idx=Int32(0), vector_size=1
        )[0]
        block_m, block_n, _ = cute.arch.block_idx()
        if cutlass.const_expr(cfg.is_swap_ab):
            token_cta_idx = block_n
        else:
            token_cta_idx = block_m
        is_active_cta = token_cta_idx < num_non_exiting_ctas
    else:
        is_active_cta = True

    if is_active_cta:
        # Setup
        task_manager.setup_resources_and_tasks()

        # Fence all mbarrier initializations before
        # any pipeline wait/arrive.  ``PipelineConfig`` creation uses
        # ``defer_sync=True``, so the common fence/sync has to happen here after
        # every resource has initialized its barriers.
        tmem_ptr_i32 = smem_allocator.get(tmem_ptr_alloc)
        tmem_dealloc_mbar = None
        tmem_dealloc_mbar_ptr = None
        if cutlass.const_expr(cfg.has_cluster):
            tmem_dealloc_mbar = smem_allocator.get(tmem_dealloc_mbar_alloc)
            tmem_dealloc_mbar_ptr = cute.make_ptr(
                cutlass.Int64,
                tmem_dealloc_mbar.data_ptr(),
                cutlass.AddressSpace.smem,
            )
            if warp_idx == 0:
                if prims.elect_sync():
                    cute.arch.mbarrier_init(tmem_dealloc_mbar_ptr, cute.arch.WARP_SIZE)
            cute.arch.mbarrier_init_fence()
        tmem_ptr_smem = cute.make_ptr(
            cutlass.Int32,
            tmem_ptr_i32.data_ptr(),
            cutlass.AddressSpace.smem,
        )
        cluster_vmnk = cute.make_layout((cfg.cluster_m, 1, 1, 1))
        pipeline.pipeline_init_arrive(cluster_vmnk, is_relaxed=True)
        pipeline.pipeline_init_wait(cluster_vmnk)

        # TMEM allocation
        # Allocate enough columns for both the explicit TS layout and generated
        # kernel fixed offsets. cfg.tmem_total_cols is derived from KernelTraits-style
        # formulas and includes the tile256 max-overlap lower bound.
        num_tmem_cols = _round_up_tmem_columns(
            max(tmem_allocator.total_tmem_columns, cfg.tmem_total_cols)
        )
        if warp_idx == 0:
            cute.arch.alloc_tmem(
                num_tmem_cols,
                tmem_ptr_smem,
                is_two_cta=cfg.has_cluster,
            )
            cute.arch.relinquish_tmem_alloc_permit(is_two_cta=cfg.has_cluster)

        # Sync warps that need TMEM
        prims.barrier_cta_sync()

        # Run
        task_manager.run()

        # TMEM deallocation.  Only the epilogue warpgroup is synchronized
        # before deallocating from the first epilogue warp.  For CTA_2 dealloc, the
        # first epilogue warp from each peer CTA performs the 32-thread rendezvous
        # before issuing the collective dealloc.
        if cutlass.const_expr(cfg.has_cluster):
            epilogue_end = cfg.epilogue_warp_idx + cfg.num_epilogue_warps
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            if (warp_idx >= Int32(cfg.epilogue_warp_idx)) & (
                warp_idx < Int32(epilogue_end)
            ):
                prims.barrier_cta_sync(
                    barrier_id=10,
                    thread_count=cfg.num_epilogue_warps * 32,
                )
                if warp_idx == Int32(cfg.epilogue_warp_idx):
                    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
                    peer_cta_rank = cta_rank_in_cluster ^ 1
                    cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr, peer_cta_rank)
                    cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
                    tmem_ptr_raw = prims.shfl_sync(
                        thread_mask=0xFFFFFFFF,
                        val=Int32(tmem_ptr_i32.load()),
                        offset=0,
                        mask_and_clamp=0x1F,
                        kind=prims.Shfl.IDX,
                        return_value_and_is_valid=False,
                    )
                    tmem_ptr = prims.make_tmem_ptr(tmem_ptr_raw, cutlass.Float32)
                    prims.tcgen05_dealloc(tmem_ptr, num_tmem_cols, group=cfg.cta_group)
        else:
            epilogue_end = cfg.epilogue_warp_idx + cfg.num_epilogue_warps
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            if cutlass.const_expr(cfg.use_full_tmem_dealloc_barrier):
                prims.barrier_cta_sync()
            if (warp_idx >= Int32(cfg.epilogue_warp_idx)) & (
                warp_idx < Int32(epilogue_end)
            ):
                prims.barrier_cta_sync(
                    barrier_id=10,
                    thread_count=cfg.num_epilogue_warps * 32,
                )
                if warp_idx == Int32(cfg.epilogue_warp_idx):
                    tmem_ptr_raw = prims.shfl_sync(
                        thread_mask=0xFFFFFFFF,
                        val=Int32(tmem_ptr_i32.load()),
                        offset=0,
                        mask_and_clamp=0x1F,
                        kind=prims.Shfl.IDX,
                        return_value_and_is_valid=False,
                    )
                    tmem_ptr = prims.make_tmem_ptr(tmem_ptr_raw, cutlass.Float32)
                    prims.tcgen05_dealloc(tmem_ptr, num_tmem_cols, group=cfg.cta_group)


@cute.kernel
def batched_gemm_kernel_bf16(
    tma_a_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_b_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_c_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_sfa_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_sfb_desc: cutlass.GridConstant[cuda.TensorMap],
    c_tensor: cute.Tensor,
    sf_c_tensor: cute.Tensor,
    bias_tensor: cute.Tensor,
    scale_c_tensor: cute.Tensor,
    scale_gate_tensor: cute.Tensor,
    gemm1_alpha_tensor: cute.Tensor,
    gemm1_beta_tensor: cute.Tensor,
    gemm1_clamp_limit_tensor: cute.Tensor,
    per_token_sf_a_tensor: cute.Tensor,
    per_token_sf_b_tensor: cute.Tensor,
    tile_idx_tensor: cute.Tensor,
    route_map_tensor: cute.Tensor,
    mn_limit_tensor: cute.Tensor,
    num_non_exiting_ctas_tensor: cute.Tensor,
    total_num_padded_tokens_tensor: cute.Tensor,
    act_tensor: cute.Tensor,
    sfa_gmem_tensor: cute.Tensor,
    sfb_gmem_tensor: cute.Tensor,
    problem_m: cutlass.Int32,
    problem_n: cutlass.Int32,
    problem_k: cutlass.Int32,
    num_tokens: cutlass.Int32,
    num_experts: cutlass.Int32,
    tile_sched_params: object,
    cfg: cutlass.Constexpr[BatchedGemmConfig],
    early_exit_max_token_ctas: cutlass.Int32,
) -> None:
    if cutlass.const_expr(cfg.do_pdl_wait_for_num_non_exiting_ctas):
        prims.griddepcontrol(kind=prims.GridDepAction.WAIT)

    if cutlass.const_expr(cfg.use_early_exit and not cfg.is_persistent):
        # Dynamic-batch kernels launch a max token-CTA grid for
        # CUDA graph reuse and skip inactive CTAs before touching routing tables.
        num_non_exiting_ctas_view = cutlass.make_array_view(num_non_exiting_ctas_tensor)
        num_non_exiting_ctas = num_non_exiting_ctas_view.load(
            idx=Int32(0), vector_size=1
        )[0]
        block_m, block_n, _ = cute.arch.block_idx()
        if cutlass.const_expr(cfg.is_swap_ab):
            token_cta_idx = block_n
        else:
            token_cta_idx = block_m
        if token_cta_idx < num_non_exiting_ctas:
            _batched_gemm_kernel_bf16_body(
                tma_a_desc,
                tma_b_desc,
                tma_c_desc,
                tma_sfa_desc,
                tma_sfb_desc,
                c_tensor,
                sf_c_tensor,
                bias_tensor,
                scale_c_tensor,
                scale_gate_tensor,
                gemm1_alpha_tensor,
                gemm1_beta_tensor,
                gemm1_clamp_limit_tensor,
                per_token_sf_a_tensor,
                per_token_sf_b_tensor,
                tile_idx_tensor,
                route_map_tensor,
                mn_limit_tensor,
                num_non_exiting_ctas_tensor,
                total_num_padded_tokens_tensor,
                act_tensor,
                sfa_gmem_tensor,
                sfb_gmem_tensor,
                problem_m,
                problem_n,
                problem_k,
                num_tokens,
                num_experts,
                tile_sched_params,
                cfg,
                early_exit_max_token_ctas,
            )
    else:
        _batched_gemm_kernel_bf16_body(
            tma_a_desc,
            tma_b_desc,
            tma_c_desc,
            tma_sfa_desc,
            tma_sfb_desc,
            c_tensor,
            sf_c_tensor,
            bias_tensor,
            scale_c_tensor,
            scale_gate_tensor,
            gemm1_alpha_tensor,
            gemm1_beta_tensor,
            gemm1_clamp_limit_tensor,
            per_token_sf_a_tensor,
            per_token_sf_b_tensor,
            tile_idx_tensor,
            route_map_tensor,
            mn_limit_tensor,
            num_non_exiting_ctas_tensor,
            total_num_padded_tokens_tensor,
            act_tensor,
            sfa_gmem_tensor,
            sfb_gmem_tensor,
            problem_m,
            problem_n,
            problem_k,
            num_tokens,
            num_experts,
            tile_sched_params,
            cfg,
            early_exit_max_token_ctas,
        )


# ---------------------------------------------------------------------------
# Host-side launcher
# ---------------------------------------------------------------------------


@cute.jit
def gemm(
    a_raw_ptr: cute.Pointer,
    b_raw_ptr: cute.Pointer,
    sfa_raw_ptr: cute.Pointer,
    sfb_raw_ptr: cute.Pointer,
    c_raw_ptr: cute.Pointer,
    sf_c_raw_ptr: cute.Pointer,
    tile_idx_raw_ptr: cute.Pointer,
    route_map_raw_ptr: cute.Pointer,  # Int32 route map (gather only)
    mn_limit_raw_ptr: cute.Pointer,  # Int32 per-tile mn limit (gather only)
    num_non_exiting_ctas_raw_ptr: cute.Pointer,  # Int32 scalar for early exit
    total_num_padded_tokens_raw_ptr: cute.Pointer,  # Int32 scalar for routed token stride
    act_raw_ptr: cute.Pointer,  # BF16 activation ptr (gather only)
    per_token_sf_a_raw_ptr: cute.Pointer,
    per_token_sf_b_raw_ptr: cute.Pointer,
    bias_raw_ptr,
    scale_c_raw_ptr,
    scale_gate_raw_ptr,
    gemm1_alpha_raw_ptr,
    gemm1_beta_raw_ptr,
    gemm1_clamp_limit_raw_ptr,
    problem_size: tuple,
    early_exit_max_token_ctas: cutlass.Int32,
    cfg: cutlass.Constexpr[BatchedGemmConfig],
    stream,
):
    """Host-side: create TMA descs and launch kernel."""
    m, n, k, num_experts_dim, num_tokens = problem_size
    compute_warp_layout(cfg)
    validate_config(cfg, problem_mnk=(m, n, k))
    from cutlass.experimental.cuda import TensorMapDataFormat

    # TensorMap lowers element strides through bit strides. Keep products
    # in 64 bits so large routed workspaces do not overflow int32 there.
    # Typed pointers passed from host-side make_ptr (cute.runtime.make_ptr)
    a_ptr = a_raw_ptr
    b_ptr = b_raw_ptr
    sfa_ptr = sfa_raw_ptr
    sfb_ptr = sfb_raw_ptr
    per_token_sf_a_ptr = per_token_sf_a_raw_ptr
    per_token_sf_b_ptr = per_token_sf_b_raw_ptr

    # TMA route (gather4): the routed operand is 2D (K, total_rows) flat.
    # TmaOobOpt: non-routed activations use a 4D out-of-bounds TMA descriptor.
    # Non-swapAB: A=activations (routed 2D), B=weights (3D).
    # SwapAB:     A=weights (3D), B=activations (routed 2D).
    # No route:   activation operand may be 4D OOB; weights stay 3D.
    if cutlass.const_expr(cfg.uses_block_major_k_weight_a):
        a_block_k = cfg.block_major_k_elems
        a_layout = cute.make_layout(
            (a_block_k, m, cute.assume(k // a_block_k, 1), num_experts_dim),
            stride=(
                1,
                a_block_k,
                cute.assume(m * a_block_k, 32),
                cute.assume(cutlass.Int64(m) * cutlass.Int64(k), 32),
            ),
        )
        a_tensor = cute.make_tensor(a_ptr, a_layout)
    elif cutlass.const_expr(cfg.has_tma_route and not cfg.is_swap_ab):
        a_layout = cute.make_layout(
            (cute.assume(k, 32), m), stride=(1, cute.assume(k, 32))
        )
        a_tensor = cute.make_tensor(a_ptr, a_layout)
    elif cutlass.const_expr(cfg.use_bf16_kbox_tma_a):
        a_layout = cute.make_layout(
            (64, m, cute.assume(k // 64, 1), num_experts_dim),
            stride=(1, cute.assume(k, 32), 64, cute.assume(cutlass.Int64(m) * cutlass.Int64(k), 32)),
        )
        a_tensor = cute.make_tensor(a_ptr, a_layout)
    elif cutlass.const_expr(cfg.use_tma_oob_opt_a):
        a_layout = cute.make_layout(
            (cute.assume(k, 32), cfg.tile_m, TMA_DIM_MAX, TMA_DIM_MAX),
            stride=(1, cute.assume(k, 32), TMA_XLARGE_N - k, cute.assume(k, 32)),
        )
        a_tensor = cute.make_tensor(a_ptr, a_layout)
    else:
        a_layout = cute.make_layout(
            (cute.assume(k, 32), m, num_experts_dim),
            stride=(1, cute.assume(k, 32), cute.assume(cutlass.Int64(m) * cutlass.Int64(k), 32)),
        )
        a_tensor = cute.make_tensor(a_ptr, a_layout)

    if cutlass.const_expr(cfg.uses_block_major_k_weight_b):
        b_block_k = cfg.block_major_k_elems
        b_layout = cute.make_layout(
            (b_block_k, n, cute.assume(k // b_block_k, 1), num_experts_dim),
            stride=(
                1,
                b_block_k,
                cute.assume(n * b_block_k, 32),
                cute.assume(cutlass.Int64(n) * cutlass.Int64(k), 32),
            ),
        )
    elif cutlass.const_expr(cfg.has_tma_route and cfg.is_swap_ab):
        b_layout = cute.make_layout(
            (cute.assume(k, 32), n), stride=(1, cute.assume(k, 32))
        )
    elif cutlass.const_expr(cfg.use_tma_oob_opt_b):
        b_layout = cute.make_layout(
            (cute.assume(k, 32), cfg.tile_n, TMA_DIM_MAX, TMA_DIM_MAX),
            stride=(1, cute.assume(k, 32), TMA_XLARGE_N - k, cute.assume(k, 32)),
        )
    else:
        b_layout = cute.make_layout(
            (cute.assume(k, 32), n, num_experts_dim),
            stride=(1, cute.assume(k, 32), cute.assume(cutlass.Int64(n) * cutlass.Int64(k), 32)),
        )
    b_tensor = cute.make_tensor(b_ptr, b_layout)

    def _tma_format_for_dtype(dtype_kind: int, native_fp4: bool = False):
        if cutlass.const_expr(dtype_kind == int(DType.BF16)):
            return TensorMapDataFormat.DEFAULT
        if cutlass.const_expr(dtype_kind in (int(DType.MXE4M3), int(DType.E4M3))):
            return TensorMapDataFormat.BYTE
        if cutlass.const_expr(dtype_kind == int(DType.MXE2M1)):
            if cutlass.const_expr(native_fp4):
                return TensorMapDataFormat.B4X16
            return TensorMapDataFormat.B4X16_P64
        return TensorMapDataFormat.B4X16

    def _tma_box_k_for_dtype(dtype_kind: int, native_fp4: bool = False):
        if cutlass.const_expr(dtype_kind == int(DType.BF16)):
            return 64
        if cutlass.const_expr(native_fp4):
            return cfg.tile_k
        if cutlass.const_expr(
            dtype_kind in (int(DType.MXE2M1), int(DType.MXE4M3), int(DType.E4M3))
        ):
            return 128
        return 256

    def _tma_swizzle_for_fastest_dim_bytes(num_bytes: int):
        if cutlass.const_expr(num_bytes % 128 == 0):
            return cuda.TensorMapSwizzle.s128b
        if cutlass.const_expr(num_bytes % 64 == 0):
            return cuda.TensorMapSwizzle.s64b
        if cutlass.const_expr(num_bytes % 32 == 0):
            return cuda.TensorMapSwizzle.s32b
        return cuda.TensorMapSwizzle.none

    # TMA descriptors for A and B: MX FP4 expands to 8-bit SMEM staging, so
    # A/B formats and box widths must be selected independently.
    tma_format_a = _tma_format_for_dtype(cfg.dtype_a_kind, cfg.has_cast_a)
    tma_format_b = _tma_format_for_dtype(cfg.dtype_b_kind)
    tma_swizzle = cuda.TensorMapSwizzle.s128b
    a_box_k = _tma_box_k_for_dtype(cfg.dtype_a_kind, cfg.has_cast_a)
    b_box_k = _tma_box_k_for_dtype(cfg.dtype_b_kind)

    if cutlass.const_expr(cfg.uses_block_major_k_weight_a):
        a_tile_block_k = cfg.block_major_k_tile_block_elems
        tma_a_desc = cuda.create_tensor_map_tiled_from_tensor(
            tensor=a_tensor,
            box_dims=(
                a_tile_block_k,
                cfg.tile_m,
                cfg.block_major_k_tile_slices,
                1,
            ),
            stride_order=(0, 1, 2, 3),
            swizzle=tma_swizzle,
            tma_format=tma_format_a,
        )
    elif cutlass.const_expr(cfg.has_tma_route and not cfg.is_swap_ab):
        tma_a_desc = cuda.create_tensor_map_tiled_from_tensor(
            tensor=a_tensor,
            box_dims=(a_box_k, 1),
            stride_order=(0, 1),
            swizzle=tma_swizzle,
            tma_format=tma_format_a,
        )
    elif cutlass.const_expr(cfg.use_bf16_kbox_tma_a):
        tma_a_desc = cuda.create_tensor_map_tiled_from_tensor(
            tensor=a_tensor,
            box_dims=(64, cfg.tile_m, cfg.tile_k // 64, 1),
            stride_order=(0, 1, 2, 3),
            swizzle=tma_swizzle,
            tma_format=tma_format_a,
        )
    elif cutlass.const_expr(cfg.use_tma_oob_opt_a):
        tma_a_desc = cuda.create_tensor_map_tiled_from_tensor(
            tensor=a_tensor,
            box_dims=(a_box_k, cfg.tile_m, 1, 1),
            stride_order=(0, 1, 2, 3),
            swizzle=tma_swizzle,
            tma_format=tma_format_a,
        )
    else:
        tma_a_desc = cuda.create_tensor_map_tiled_from_tensor(
            tensor=a_tensor,
            box_dims=(a_box_k, cfg.tile_m, 1),
            stride_order=(0, 1, 2),
            swizzle=tma_swizzle,
            tma_format=tma_format_a,
        )

    if cutlass.const_expr(cfg.has_tma_route and cfg.is_swap_ab):
        tma_b_desc = cuda.create_tensor_map_tiled_from_tensor(
            tensor=b_tensor,
            box_dims=(b_box_k, 1),
            stride_order=(0, 1),
            swizzle=tma_swizzle,
            tma_format=tma_format_b,
        )
    else:
        b_box_rows = cfg.tile_n
        if cutlass.const_expr(cfg.split_b_across_ctas):
            b_box_rows = cfg.tile_n // cfg.cluster_m
        if cutlass.const_expr(cfg.uses_block_major_k_weight_b):
            b_tile_block_k = cfg.block_major_k_tile_block_elems
            tma_b_desc = cuda.create_tensor_map_tiled_from_tensor(
                tensor=b_tensor,
                box_dims=(
                    b_tile_block_k,
                    b_box_rows,
                    cfg.block_major_k_tile_slices,
                    1,
                ),
                stride_order=(0, 1, 2, 3),
                swizzle=tma_swizzle,
                tma_format=tma_format_b,
            )
        elif cutlass.const_expr(cfg.use_tma_oob_opt_b):
            tma_b_desc = cuda.create_tensor_map_tiled_from_tensor(
                tensor=b_tensor,
                box_dims=(b_box_k, b_box_rows, 1, 1),
                stride_order=(0, 1, 2, 3),
                swizzle=tma_swizzle,
                tma_format=tma_format_b,
            )
        else:
            tma_b_desc = cuda.create_tensor_map_tiled_from_tensor(
                tensor=b_tensor,
                box_dims=(b_box_k, b_box_rows, 1),
                stride_order=(0, 1, 2),
                swizzle=tma_swizzle,
                tma_format=tma_format_b,
            )

    # SF TMA descriptors (only for FP4/FP8/CastA with scale factors).
    # E8M0 scale factors are packed 2 per uint16 for TMA transport.
    tma_sfa_desc = None
    tma_sfb_desc = None
    if cutlass.const_expr(cfg.has_scale_factor_a):
        sf_vec_size = cfg.sf_vec_size
        sf_tma_ptr_dtype = cutlass.Uint8
        # Each K-block = 4 SF atoms. dim0 = tile_mn * 4 / 2 (uint16 packing)
        k_atoms_per_block = 4
        rest_k_blocks = k // sf_vec_size // k_atoms_per_block
        rest_k_per_tile = cfg.tile_k // sf_vec_size // k_atoms_per_block

        sf_k_per_tile = cfg.tile_k // sf_vec_size  # SF elements per K-tile
        routed_sf_swizzle = _tma_swizzle_for_fastest_dim_bytes(sf_k_per_tile)

        # Folds expert L into the weight operand's outer row
        # extent. The no-route activation operand is already expanded/padded
        # by token row and does not use a separate expert dimension.
        sfa_outer_rows = m
        if cutlass.const_expr(cfg.is_swap_ab):
            sfa_outer_rows = m * num_experts_dim
        sfb_outer_rows = n
        if cutlass.const_expr(not cfg.is_swap_ab):
            sfb_outer_rows = n * num_experts_dim

        # SFA descriptor: routed TMA gather4 (2D linear) or regular tiled SF layout.
        # LDGSTS-routed SF bypasses TensorMap encoding and uses raw GMEM tensors.
        if cutlass.const_expr(uses_routed_sfa_tma_desc(cfg)):
            # 2D linear layout: (K/sf_vec_size, total_tokens) for gather4.
            # Pad sf_k to 16-byte alignment (TMA requirement).
            sf_k_total = ((k // sf_vec_size + 15) // 16) * 16
            sfa_2d_layout = cute.make_layout(
                (sf_k_total, m),
                stride=(1, sf_k_total),
            )
            sfa_tensor = cute.make_tensor(
                cute.recast_ptr(sfa_ptr, dtype=sf_tma_ptr_dtype), sfa_2d_layout
            )
            tma_sfa_desc = cuda.create_tensor_map_tiled_from_tensor(
                tensor=sfa_tensor,
                box_dims=(sf_k_per_tile, 1),
                stride_order=(0, 1),
                swizzle=routed_sf_swizzle,
                tma_format=cuda.TensorMapDataFormat.BYTE,
            )
        else:
            # Tiled SF layout. L is folded into the outer dimension for
            # weights and absent for expanded activations. Only the generated
            # split R128c4 descriptor remains rank-4; the legacy tiled
            # descriptor is rank-3.
            if cutlass.const_expr(
                cfg.use_tile256_tmem_overlap
                or cfg.is_mx_mma
                or cfg.has_cast_a
                or cfg.is_nvfp4_mma
            ):
                # Generated R128c4 SF TMA shape is [256, 2, K/(sfBlock*4), outer/128].
                # The split keeps the 512B scale-factor block TMA-legal and is
                # used for NVFP4 as well as MX/CastA paths.
                sfa_outer_tiles = (sfa_outer_rows + 127) // 128
                tma_sfa_desc = cuda.create_tensor_map_tiled(
                    global_address=sfa_ptr.toint(),
                    dtype=cutlass.Uint8,
                    tma_format=cuda.TensorMapDataFormat.BYTE,
                    global_dims=(
                        256,
                        2,
                        cute.assume(rest_k_blocks, 4),
                        sfa_outer_tiles,
                    ),
                    # create_tensor_map_tiled expects strides in 16-byte units.
                    global_strides=(
                        16,
                        32,
                        cute.assume(32 * rest_k_blocks, 32 * 4),
                    ),
                    box_dims=(256, 2, rest_k_per_tile, 1),
                    swizzle=cuda.TensorMapSwizzle.none,
                )
            else:
                sfa_dim0 = cfg.tile_m * k_atoms_per_block // 2
                sfa_rest_m = sfa_outer_rows // cfg.tile_m
                sfa_ptr_fp16 = cute.recast_ptr(sfa_ptr, dtype=cutlass.Float16)
                sfa_layout = cute.make_layout(
                    (
                        sfa_dim0,
                        cute.assume(rest_k_blocks, 4),
                        sfa_rest_m,
                    ),
                    stride=(
                        1,
                        sfa_dim0,
                        cute.assume(sfa_dim0 * rest_k_blocks, sfa_dim0 * 4),
                    ),
                )
                sfa_tensor_fp16 = cute.make_tensor(sfa_ptr_fp16, sfa_layout)
                tma_sfa_desc = cuda.create_tensor_map_tiled_from_tensor(
                    tensor=sfa_tensor_fp16,
                    box_dims=(sfa_dim0, rest_k_per_tile, 1),
                    stride_order=(0, 1, 2),
                    swizzle=cuda.TensorMapSwizzle.none,
                    tma_format=cuda.TensorMapDataFormat.DEFAULT,
                )

        # SFB descriptor: routed TMA gather4 (2D linear) or regular tiled SF layout.
        # LDGSTS-routed SF bypasses TensorMap encoding and uses raw GMEM tensors.
        if cutlass.const_expr(uses_routed_sfb_tma_desc(cfg)):
            # 2D linear layout: (K/sf_vec_size, total_tokens) for gather4
            sf_k_total = ((k // sf_vec_size + 15) // 16) * 16
            sfb_2d_layout = cute.make_layout(
                (sf_k_total, n),
                stride=(1, sf_k_total),
            )
            sfb_tensor = cute.make_tensor(
                cute.recast_ptr(sfb_ptr, dtype=sf_tma_ptr_dtype), sfb_2d_layout
            )
            tma_sfb_desc = cuda.create_tensor_map_tiled_from_tensor(
                tensor=sfb_tensor,
                box_dims=(sf_k_per_tile, 1),
                stride_order=(0, 1),
                swizzle=routed_sf_swizzle,
                tma_format=cuda.TensorMapDataFormat.BYTE,
            )
        else:
            # Tiled SF layout with the same folded-weight/no-L-activation
            # convention as SFA. Legacy tiled descriptors are rank-3.
            if cutlass.const_expr(cfg.use_tile256_tmem_overlap):
                sfb_outer_tiles = (sfb_outer_rows + 127) // 128
                sfb_tile_outer = (cfg.tile_n + 127) // 128
                sfb_layout = cute.make_layout(
                    (256, 2, cute.assume(rest_k_blocks, 4), sfb_outer_tiles),
                    stride=(
                        1,
                        256,
                        512,
                        cute.assume(512 * rest_k_blocks, 512 * 4),
                    ),
                )
                sfb_tensor = cute.make_tensor(
                    cute.recast_ptr(sfb_ptr, dtype=sf_tma_ptr_dtype),
                    sfb_layout,
                )
                tma_sfb_desc = cuda.create_tensor_map_tiled_from_tensor(
                    tensor=sfb_tensor,
                    box_dims=(256, 2, rest_k_per_tile, sfb_tile_outer),
                    stride_order=(0, 1, 2, 3),
                    swizzle=cuda.TensorMapSwizzle.none,
                    tma_format=cuda.TensorMapDataFormat.BYTE,
                )
            elif cutlass.const_expr(cfg.uses_sfb_8x4_load):
                sfb_outer_tiles = (sfb_outer_rows + 7) // 8
                sfb_outer_per_tile = (cfg.tile_n + 7) // 8
                sfb_layout = cute.make_layout(
                    (
                        32,
                        cute.assume(rest_k_blocks, 4),
                        sfb_outer_tiles,
                    ),
                    stride=(
                        1,
                        32,
                        cute.assume(32 * rest_k_blocks, 32 * 4),
                    ),
                )
                sfb_tensor = cute.make_tensor(
                    cute.recast_ptr(sfb_ptr, dtype=sf_tma_ptr_dtype),
                    sfb_layout,
                )
                tma_sfb_desc = cuda.create_tensor_map_tiled_from_tensor(
                    tensor=sfb_tensor,
                    box_dims=(32, rest_k_per_tile, sfb_outer_per_tile),
                    stride_order=(0, 1, 2),
                    swizzle=cuda.TensorMapSwizzle.none,
                    tma_format=cuda.TensorMapDataFormat.BYTE,
                )
            elif cutlass.const_expr(
                (cfg.is_mx_mma or cfg.has_cast_a)
                and cfg.smem_sfb_layout == int(SfLayout.R128c4)
            ):
                sfb_outer_tiles = (sfb_outer_rows + 127) // 128
                sfb_tile_outer = (cfg.tile_n + 127) // 128
                sfb_layout = cute.make_layout(
                    (256, 2, cute.assume(rest_k_blocks, 4), sfb_outer_tiles),
                    stride=(
                        1,
                        256,
                        512,
                        cute.assume(512 * rest_k_blocks, 512 * 4),
                    ),
                )
                sfb_tensor = cute.make_tensor(
                    cute.recast_ptr(sfb_ptr, dtype=sf_tma_ptr_dtype),
                    sfb_layout,
                )
                tma_sfb_desc = cuda.create_tensor_map_tiled_from_tensor(
                    tensor=sfb_tensor,
                    box_dims=(256, 2, rest_k_per_tile, sfb_tile_outer),
                    stride_order=(0, 1, 2, 3),
                    swizzle=cuda.TensorMapSwizzle.none,
                    tma_format=cuda.TensorMapDataFormat.BYTE,
                )
            else:
                sfb_dim0 = cfg.tile_n * k_atoms_per_block // 2
                sfb_rest_n = sfb_outer_rows // cfg.tile_n
                sfb_ptr_fp16 = cute.recast_ptr(sfb_ptr, dtype=cutlass.Float16)
                sfb_layout = cute.make_layout(
                    (
                        sfb_dim0,
                        cute.assume(rest_k_blocks, 4),
                        sfb_rest_n,
                    ),
                    stride=(
                        1,
                        sfb_dim0,
                        cute.assume(sfb_dim0 * rest_k_blocks, sfb_dim0 * 4),
                    ),
                )
                sfb_tensor_fp16 = cute.make_tensor(sfb_ptr_fp16, sfb_layout)
                tma_sfb_desc = cuda.create_tensor_map_tiled_from_tensor(
                    tensor=sfb_tensor_fp16,
                    box_dims=(sfb_dim0, rest_k_per_tile, 1),
                    stride_order=(0, 1, 2),
                    swizzle=cuda.TensorMapSwizzle.none,
                    tma_format=cuda.TensorMapDataFormat.DEFAULT,
                )

    # Grid and tile scheduler
    num_tiles_m = (m + cfg.tile_m - 1) // cfg.tile_m
    num_tiles_n = (n + cfg.tile_n - 1) // cfg.tile_n
    launch_num_tiles_m = num_tiles_m
    launch_num_tiles_n = num_tiles_n
    if cutlass.const_expr(cfg.use_early_exit):
        if cutlass.const_expr(cfg.is_swap_ab):
            launch_num_tiles_n = early_exit_max_token_ctas
        else:
            launch_num_tiles_m = early_exit_max_token_ctas
    block = cfg.threads_per_cta
    cluster_shape = (cfg.cluster_m, 1, 1)

    if cutlass.const_expr(cfg.is_persistent):
        clc_raster_along_m = True
        tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
            (launch_num_tiles_m, launch_num_tiles_n, 1),
            cluster_shape,
            1,
            clc_raster_along_m,
        )
        grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(tile_sched_params)
    else:
        # Non-persistent: launch one CTA per logical output tile.
        tile_sched_params = utils.PersistentTileSchedulerParams(
            (launch_num_tiles_m, launch_num_tiles_n, 1),
            cluster_shape,
        )
        grid = (launch_num_tiles_m, launch_num_tiles_n, 1)

    # C output tensor
    output_m_for_c = m
    if cutlass.const_expr(cfg.is_swap_ab and cfg.has_gated_epilogue):
        output_m_for_c = m // Int32(2)
    if cutlass.const_expr(cfg.is_swap_ab):
        # SwapAB: C is (M, N) in M-major (column-major) layout.
        # M = hidden dim (stride-1), N = tokens (stride-M).
        if cutlass.const_expr(cfg.use_tma_store and cfg.use_tma_oob_opt):
            if cutlass.const_expr(cfg.has_epilogue_quant):
                c_layout = cute.make_layout(
                    (
                        cute.assume(output_m_for_c, 32),
                        cfg.tile_n,
                        TMA_DIM_MAX,
                        TMA_DIM_MAX,
                    ),
                    stride=(
                        1,
                        cute.assume(output_m_for_c, 32),
                        TMA_XLARGE_N - output_m_for_c,
                        cute.assume(output_m_for_c, 32),
                    ),
                )
            else:
                c_layout = cute.make_layout(
                    (
                        cute.assume(output_m_for_c, 32),
                        cfg.tile_n,
                        TMA_DIM_MAX,
                        TMA_DIM_MAX,
                    ),
                    stride=(
                        1,
                        cute.assume(output_m_for_c, 32),
                        TMA_XLARGE_N - output_m_for_c,
                        cute.assume(output_m_for_c, 32),
                    ),
                )
        else:
            c_layout = cute.make_layout(
                (cute.assume(output_m_for_c, 32), cute.assume(n, 16)),
                stride=(1, cute.assume(output_m_for_c, 32)),
            )
    else:
        # Non-swapAB: C is (M, N) row-major.
        c_layout = cute.make_layout(
            (cute.assume(m, 32), cute.assume(n, 16)),
            stride=(cute.assume(n, 16), 1),
        )
    if cutlass.const_expr(cfg.has_epilogue_quant or cfg.uses_fp8_output):
        c_quant_dtype = (
            cutlass.Float8E4M3FN
            if cfg.uses_mxfp8_output_quant or cfg.uses_fp8_output
            else cutlass.Float4E2M1FN
        )
        c_tensor = cute.make_tensor(
            cute.recast_ptr(c_raw_ptr, dtype=c_quant_dtype),
            c_layout,
        )
    else:
        c_tensor = cute.make_tensor(c_raw_ptr, c_layout)
    if cutlass.const_expr(cfg.is_swap_ab and cfg.use_tma_store):
        tma_store_cols = cfg.epi_tile_n
    else:
        tma_store_cols = min(16, max(8, cfg.tile_n))
    if cutlass.const_expr(cfg.has_epilogue_quant):
        if cutlass.const_expr(cfg.use_tma_store):
            if cutlass.const_expr(cfg.uses_mxfp8_output_quant):
                tma_c_swizzle = cuda.TensorMapSwizzle.s64b
                tma_c_format = cuda.TensorMapDataFormat.BYTE
            else:
                tma_c_swizzle = cuda.TensorMapSwizzle.s32b
                tma_c_format = cuda.TensorMapDataFormat.B4X16
            if cutlass.const_expr(cfg.use_tma_oob_opt):
                tma_c_desc = cuda.create_tensor_map_tiled_from_tensor(
                    tensor=c_tensor,
                    box_dims=(cfg.tile_m // 2, cfg.epi_tile_n, 1, 1),
                    stride_order=(0, 1, 2, 3),
                    swizzle=tma_c_swizzle,
                    tma_format=tma_c_format,
                )
            else:
                tma_c_desc = cuda.create_tensor_map_tiled_from_tensor(
                    tensor=c_tensor,
                    box_dims=(cfg.tile_m // 2, cfg.epi_tile_n),
                    stride_order=(0, 1),
                    swizzle=tma_c_swizzle,
                    tma_format=tma_c_format,
                )
        else:
            # Quantized FC1 may use direct packed STG for C; keep a dummy
            # descriptor so the common kernel signature stays stable.
            tma_c_desc = tma_a_desc
    else:
        if cutlass.const_expr(cfg.is_swap_ab and cfg.use_tma_store):
            if cutlass.const_expr(cfg.uses_fp8_output):
                tma_c_format = cuda.TensorMapDataFormat.BYTE
                tma_c_swizzle = cuda.TensorMapSwizzle.s64b
            else:
                tma_c_format = cuda.TensorMapDataFormat.DEFAULT
                tma_c_swizzle = cuda.TensorMapSwizzle.s128b
            if cutlass.const_expr(cfg.use_tma_oob_opt):
                tma_c_desc = cuda.create_tensor_map_tiled_from_tensor(
                    tensor=c_tensor,
                    box_dims=(64, tma_store_cols, 1, 1),
                    stride_order=(0, 1, 2, 3),
                    swizzle=tma_c_swizzle,
                    tma_format=tma_c_format,
                )
            else:
                tma_c_desc = cuda.create_tensor_map_tiled_from_tensor(
                    tensor=c_tensor,
                    box_dims=(64, tma_store_cols),
                    stride_order=(0, 1),
                    swizzle=tma_c_swizzle,
                    tma_format=tma_c_format,
                )
        else:
            if cutlass.const_expr(cfg.uses_fp8_output):
                tma_c_format = cuda.TensorMapDataFormat.BYTE
            else:
                tma_c_format = cuda.TensorMapDataFormat.DEFAULT
            tma_c_desc = cuda.create_tensor_map_tiled_from_tensor(
                tensor=c_tensor,
                box_dims=(cfg.tile_m, tma_store_cols),
                stride_order=(1, 0),
                swizzle=cuda.TensorMapSwizzle.none,
                tma_format=tma_c_format,
            )

    output_m_for_sf = m
    if cutlass.const_expr(cfg.is_swap_ab and cfg.has_gated_epilogue):
        output_m_for_sf = m // Int32(2)
    if cutlass.const_expr(cfg.has_epilogue_quant):
        sf_block_m = Int32(cfg.output_sf_block_size_c)
        sf_group_m = sf_block_m * Int32(4)
        if cutlass.const_expr(cfg.sf_layout_c == int(SfLayout.R8c4)):
            sf_c_elems = ((n + Int32(7)) // Int32(8)) * (
                ((output_m_for_sf + sf_group_m - Int32(1)) // sf_group_m) * Int32(32)
            )
        else:
            sf_c_elems = ((n + Int32(127)) // Int32(128)) * (
                ((output_m_for_sf + sf_group_m - Int32(1)) // sf_group_m) * Int32(512)
            )
        sf_c_layout = cute.make_layout((sf_c_elems,), stride=(1,))
    elif cutlass.const_expr(cfg.has_deepseek_fp8_c_scale):
        if cutlass.const_expr(cfg.is_swap_ab):
            sf_c_elems = ((output_m_for_sf + Int32(127)) // Int32(128)) * n
        else:
            sf_c_elems = ((n + Int32(127)) // Int32(128)) * output_m_for_sf
        sf_c_layout = cute.make_layout((sf_c_elems,), stride=(1,))
    else:
        sf_c_layout = cute.make_layout((1,), stride=(1,))
    sf_c_tensor = cute.make_tensor(
        cute.recast_ptr(
            sf_c_raw_ptr,
            dtype=(
                cutlass.Float32
                if cfg.has_deepseek_fp8_c_scale
                else (
                    cutlass.Float8E8M0FNU
                    if cfg.uses_mx_output_quant
                    else cutlass.Float8E4M3FN
                )
            ),
        ),
        sf_c_layout,
    )

    bias_layout = cute.make_layout(
        (cute.assume(m, 32), num_experts_dim),
        stride=(1, cute.assume(m, 32)),
    )
    bias_tensor = cute.make_tensor(bias_raw_ptr, bias_layout)

    global_scale_layout = cute.make_layout((num_experts_dim,), stride=(1,))
    scale_c_tensor = cute.make_tensor(scale_c_raw_ptr, global_scale_layout)
    scale_gate_tensor = cute.make_tensor(scale_gate_raw_ptr, global_scale_layout)
    gemm1_alpha_tensor = cute.make_tensor(gemm1_alpha_raw_ptr, global_scale_layout)
    gemm1_beta_tensor = cute.make_tensor(gemm1_beta_raw_ptr, global_scale_layout)
    gemm1_clamp_limit_tensor = cute.make_tensor(
        gemm1_clamp_limit_raw_ptr,
        global_scale_layout,
    )

    if cutlass.const_expr(cfg.per_token_sf_dtype == int(DType.BF16)):
        per_token_sf_dtype = cutlass.BFloat16
    elif cutlass.const_expr(cfg.per_token_sf_dtype == int(DType.FP16)):
        per_token_sf_dtype = cutlass.Float16
    else:
        per_token_sf_dtype = cutlass.Float32
    per_token_sf_layout = cute.make_layout((max(m, n),), stride=(1,))
    per_token_sf_a_tensor = cute.make_tensor(
        cute.recast_ptr(per_token_sf_a_ptr, dtype=per_token_sf_dtype),
        per_token_sf_layout,
    )
    per_token_sf_b_tensor = cute.make_tensor(
        cute.recast_ptr(per_token_sf_b_ptr, dtype=per_token_sf_dtype),
        per_token_sf_layout,
    )

    # tile_idx and mn_limit: indexed by the TOKEN dimension tiles.
    # non-swapAB: tokens = M → num_tiles_m.  swapAB: tokens = N → num_tiles_n.
    if cutlass.const_expr(cfg.is_swap_ab):
        num_token_tiles = num_tiles_n
    else:
        num_token_tiles = num_tiles_m
    tile_idx_layout = cute.make_layout((num_token_tiles,), stride=(1,))
    tile_idx_tensor = cute.make_tensor(tile_idx_raw_ptr, tile_idx_layout)

    # Route map (gather only; dummy 1-element when off)
    if cutlass.const_expr(cfg.has_routed_act):
        num_total_tokens = n if cutlass.const_expr(cfg.is_swap_ab) else m
        route_map_layout = cute.make_layout((num_total_tokens,), stride=(1,))
    else:
        route_map_layout = cute.make_layout((1,), stride=(1,))
    route_map_tensor = cute.make_tensor(route_map_raw_ptr, route_map_layout)

    # mn_limit: TRT-LLM Gen absolute end-row limit per token tile.
    mn_limit_layout = cute.make_layout((num_token_tiles,), stride=(1,))
    mn_limit_tensor = cute.make_tensor(mn_limit_raw_ptr, mn_limit_layout)
    num_non_exiting_ctas_tensor = cute.make_tensor(
        num_non_exiting_ctas_raw_ptr, cute.make_layout((1,), stride=(1,))
    )
    total_num_padded_tokens_tensor = cute.make_tensor(
        total_num_padded_tokens_raw_ptr, cute.make_layout((1,), stride=(1,))
    )
    # Activation tensor for gather: flat (total_elems,) view
    act_layout = cute.make_layout((1,), stride=(1,))
    act_tensor = cute.make_tensor(
        cute.recast_ptr(act_raw_ptr, dtype=cutlass.Uint8), act_layout
    )

    # SF TMA descriptors: real only for operands that have scale factors.
    if cutlass.const_expr(not cfg.has_scale_factor_a):
        tma_sfa_desc = tma_a_desc  # dummy, never accessed
    if cutlass.const_expr(not cfg.has_scale_factor_b):
        tma_sfb_desc = tma_b_desc  # dummy, never accessed

    # SF GMEM tensors for LDGSTS SF loading.
    if cutlass.const_expr(cfg.has_deepseek_fp8):
        # DeepSeek FP8 uses FP32 per-K dequant scales;
        # routed MX/NVFP4 uses packed E4M3/E8M0 bytes.
        ds_k_blocks = (k + Int32(cfg.tile_k - 1)) // Int32(cfg.tile_k)
        ds_weight_tiles = (
            (m + Int32(cfg.tile_m - 1)) // Int32(cfg.tile_m)
            if cfg.is_swap_ab
            else (n + Int32(cfg.tile_n - 1)) // Int32(cfg.tile_n)
        )
        if cutlass.const_expr(cfg.has_routed_act):
            ds_act_rows = num_tokens
        else:
            ds_act_rows = n if cfg.is_swap_ab else m
        sfa_gmem_layout = cute.make_layout(
            (num_experts_dim * ds_weight_tiles * ds_k_blocks,),
            stride=(1,),
        )
        sfb_gmem_layout = cute.make_layout((ds_act_rows * ds_k_blocks,), stride=(1,))
        sfa_gmem_tensor = cute.make_tensor(
            cute.recast_ptr(sfa_ptr, dtype=cutlass.Float32), sfa_gmem_layout
        )
        sfb_gmem_tensor = cute.make_tensor(
            cute.recast_ptr(sfb_ptr, dtype=cutlass.Float32), sfb_gmem_layout
        )
    elif cutlass.const_expr(cfg.has_routed_sfs and cfg.uses_ldgsts_routed_sfs):
        # (flat 1D E4M3 view)
        sf_dtype = (
            cutlass.Float8E8M0FNU if cfg.uses_mx_scale_factors else cutlass.Float8E4M3FN
        )
        sfa_gmem_layout = cute.make_layout((m * (k // cfg.sf_vec_size),), stride=(1,))
        sfb_gmem_layout = cute.make_layout((n * (k // cfg.sf_vec_size),), stride=(1,))
        sfa_gmem_tensor = cute.make_tensor(
            cute.recast_ptr(sfa_ptr, dtype=sf_dtype), sfa_gmem_layout
        )
        sfb_gmem_tensor = cute.make_tensor(
            cute.recast_ptr(sfb_ptr, dtype=sf_dtype), sfb_gmem_layout
        )
    else:
        # Dummy 1-element tensors (never accessed)
        sfa_gmem_tensor = cute.make_tensor(
            cute.recast_ptr(a_ptr, dtype=cutlass.Float8E4M3FN),
            cute.make_layout((1,), stride=(1,)),
        )
        sfb_gmem_tensor = cute.make_tensor(
            cute.recast_ptr(b_ptr, dtype=cutlass.Float8E4M3FN),
            cute.make_layout((1,), stride=(1,)),
        )

    batched_gemm_kernel_bf16(
        tma_a_desc,
        tma_b_desc,
        tma_c_desc,
        tma_sfa_desc,
        tma_sfb_desc,
        c_tensor,
        sf_c_tensor,
        bias_tensor,
        scale_c_tensor,
        scale_gate_tensor,
        gemm1_alpha_tensor,
        gemm1_beta_tensor,
        gemm1_clamp_limit_tensor,
        per_token_sf_a_tensor,
        per_token_sf_b_tensor,
        tile_idx_tensor,
        route_map_tensor,
        mn_limit_tensor,
        num_non_exiting_ctas_tensor,
        total_num_padded_tokens_tensor,
        act_tensor,
        sfa_gmem_tensor,
        sfb_gmem_tensor,
        m,
        n,
        k,
        num_tokens,
        num_experts_dim,
        tile_sched_params,
        cfg,
        early_exit_max_token_ctas,
    ).launch(
        grid=grid,
        block=[block, 1, 1],
        cluster=list(cluster_shape),
        stream=stream,
        use_pdl=bool(cfg.use_pdl),
    )
