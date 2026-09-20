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

"""Throughput 2CTA MLA decode TS kernel implementation.

Warp-specialized MLA decode kernel using the CUTLASS task-scheduling framework.

The graph depends on dtype and scheduler policy. BF16 uses 12 warps and a
combined TMA/MMA schedule; CLC adds a scheduler task to that graph. FP8 uses
16 warps, separate K/V and QK/PV tasks, and two softmax groups. In both paths,
softmax and correction own the first two four-warp groups.

Entry points:
  - build_mla_decode_task_manager() -- pure Python, used for validation only
  - MlaDecodeTs                    -- class with @cute.kernel for GPU execution
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.experimental.cuda as cuda
from cutlass import Int32, Int64
from cutlass.cute.testing import assert_ as runtime_assert
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.experimental import primitives as prims

from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    PipelineConfig,
    TileSchedulerConfig,
)
from cutlass.experimental.task_scheduling.enums import SignalingThreads
from cutlass.experimental.task_scheduling.task_manager import TaskManager
from ...tensor_map import (
    create_tensor_map_ragged_from_tensor,
    create_tensor_map_tiled_from_view,
)

from .config import (
    LOG2_E,
    REDUCTION_ROWS_PER_CTA,
    REDUCTION_THREADS_PER_ROW,
    V_TMA_LATENT_ELEMENTS,
    MlaDecodeConfig,
    make_mla_decode_config,
)
from ..helpers.constants import MAX_MLA_SPLITS_KV, TMEM_DEALLOC_MBAR_THREADS
from ..helpers.mask import MaskType, mask_visible_k_length, normalize_mask_type
from ..helpers.query import (
    FlatQueryTileLayout,
    flat_query_row_state,
    query_batch_bounds,
    runtime_flat_query_tile_has_rows,
)
from .work_partition import (
    runtime_split_kv_cap,
    runtime_split_tile_range,
)
from .resources import (
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
    MlaWorkQueue,
    WorkThrottleBarrierResource,
)
from ..helpers.math import qkv_dtype
from ..parallel_reduction_topology import (
    ParallelReductionTopology,
    make_q128_wave_limited_parallel_reduction_topology,
    should_use_q128_g1_parallel_reducer,
    validate_parallel_reduction_workspace,
)
from .parallel_reduction import (
    PARALLEL_REDUCTION_HEAD_DIM,
    PARALLEL_REDUCTION_THREADS,
    run_parallel_reduction_kernel,
)
from .reduction import run_reduction_kernel
from ..helpers.tile_scheduler import (
    MLAStaticTileSchedulerParams,
    MLAStaticTileScheduler,
    create_mla_static_tile_scheduler_params,
    divmod_constexpr_power_of_two_or_fdd,
)
from .tasks import (
    MlaClcTask,
    MlaInterleavedTask,
    MlaTask,
    create_load_k_task,
    create_load_v_task,
    create_load_tma_task,
    create_mma_task,
    create_mma_qk_direct_task,
    create_mma_pv_direct_task,
    create_softmax_task,
    create_correction_task,
    create_padding_task,
    create_scheduler_task,
)


def ceil_div(a, b):
    """Return the ceiling of a divided by b."""
    return (a + b - 1) // b


def build_mla_decode_task_manager(
    cfg: MlaDecodeConfig,
    # SMEM arrays (None for validate-only)
    smem_q_latent_arr=None,
    smem_q_rope_arr=None,
    smem_kc_arr=None,
    smem_vc_arr=None,
    smem_p_arr=None,
    # TMA descriptors (None for validate-only)
    tma_desc_q_latent=None,
    tma_desc_q_rope=None,
    tma_desc_c_latent=None,
    tma_desc_c_rope=None,
    tma_desc_c_transpose=None,
    # Page-offset tensor
    page_offsets=None,
    # Runtime coordinates
    blk_coord=None,
    tidx=None,
    # GMEM output tensors
    output=None,
    acc_output=None,
    lse=None,
    acc_lse=None,
    # Domain (k_tile_count)
    domain=4,
    # Persistent loop support
    work_queue=None,
    # For k_index_base computation (split_kv support)
    cache_seqs=None,
    cu_seqlens_q=None,
    split_kv=None,
    logical_num_heads_q=128,
    logical_seq_len_q=1,
    tiled_mma_qk=None,
    verbose=False,
    exhaustive_deadlock_race_check=True,
) -> "tuple[TaskManager, list[MemoryResource], dict[str, MemoryResource]]":
    """Build the MLA decode TaskManager with all resources, tasks, and dependency graph.

    The throughput 2CTA graph uses Q/K/V TMA loads, a register-held BF16 page-ID
    window, QK/PV MMA, softmax, and correction/store tasks. Validation-only
    calls pass a concrete integer ``domain`` and no runtime tensors; JIT calls
    pass symbolic runtime state and skip schedule validation. The dependency
    graph relies on TaskManager DMA-order validation to keep SMEM producers
    alive until async TMEM consumers have launched.

    Parameters
    ----------
    cfg : MlaDecodeConfig
        Kernel-wide configuration.
    domain : int or symbolic
        Number of k-tile iterations (loop domain for all tasks).
        Use an integer for validation-only mode.

    Returns
    -------
    TaskManager, list[MemoryResource], dict
        Configured TaskManager, TMEM resources list, named resource dict.
    """
    # ──────────────────────────────────────────────────────────────
    # Cluster / CTA layout
    # ──────────────────────────────────────────────────────────────
    cluster_shape_vmnk = (cfg.num_mma_ctas, 1, 1, 1)
    WARP_SIZE = 32
    Agent = pipeline.Agent
    use_clc_dynamic = bool(work_queue is not None and work_queue.use_clc_dynamic)
    use_work_throttle = use_clc_dynamic and not cfg.use_fp8_split_mma_schedule
    non_interleaved_task_class = MlaClcTask if use_clc_dynamic else MlaTask
    task_domain = (
        MlaClcTask.get_domain
        if use_clc_dynamic and not isinstance(domain, int)
        else domain
    )

    # Cooperative groups
    tma_producer_group = pipeline.CooperativeGroup(Agent.Thread)  # elect_one TMA
    umma_hw_group = pipeline.CooperativeGroup(Agent.Thread)  # UMMA hardware

    # Softmax warps: 4 warps * 32 threads * 2 CTAs for cluster-scoped consumer
    compute_group_cluster = pipeline.CooperativeGroup(
        Agent.Thread, cfg.num_compute_warps * WARP_SIZE * cfg.num_mma_ctas
    )
    # Softmax warps: 4 warps * 32 threads (local CTA only)
    compute_group_local = pipeline.CooperativeGroup(
        Agent.Thread, cfg.num_compute_warps * WARP_SIZE
    )
    # Correction warps: same layout
    correction_group_cluster = pipeline.CooperativeGroup(
        Agent.Thread, cfg.num_compute_warps * WARP_SIZE * cfg.num_mma_ctas
    )
    correction_group_local = pipeline.CooperativeGroup(
        Agent.Thread, cfg.num_compute_warps * WARP_SIZE
    )
    # ──────────────────────────────────────────────────────────────
    # Pipeline configs
    # ──────────────────────────────────────────────────────────────

    # SmemQ: TmaUmma, 1 stage, LoadTma -> Mma
    smem_q_pipeline_cfg = PipelineConfig.create_tma_umma_pipeline_cfg(
        num_stages=cfg.load_q_stage,
        num_bytes=cfg.tma_copy_q_bytes,
        producer_group=tma_producer_group,
        consumer_group=umma_hw_group,
        cta_layout_vmnk=cluster_shape_vmnk,
        consumer_signaling_threads=SignalingThreads.CtaLeader,
        num_bytes_per_warp_per_cta=(cfg.tma_copy_q_bytes // cfg.num_mma_ctas),
    )

    if cfg.use_fp8_split_mma_schedule:
        # FP8 uses independent whole-tile K and V pipelines so QK and PV have
        # separate consumer states.  This mirrors the native FP8 schedule shape
        # without changing the BF16 combined-KV path.
        smem_k_pipeline_cfg = PipelineConfig.create_tma_umma_pipeline_cfg(
            num_stages=cfg.load_k_stage,
            num_bytes=cfg.tma_copy_k_tile_bytes,
            producer_group=tma_producer_group,
            consumer_group=umma_hw_group,
            cta_layout_vmnk=cluster_shape_vmnk,
            consumer_signaling_threads=SignalingThreads.CtaLeader,
            num_bytes_per_warp_per_cta=(cfg.tma_copy_k_tile_bytes // cfg.num_mma_ctas),
        )
        smem_v_pipeline_cfg = PipelineConfig.create_tma_umma_pipeline_cfg(
            num_stages=cfg.load_v_stage,
            num_bytes=cfg.tma_copy_v_tile_bytes,
            producer_group=tma_producer_group,
            consumer_group=umma_hw_group,
            cta_layout_vmnk=cluster_shape_vmnk,
            consumer_signaling_threads=SignalingThreads.CtaLeader,
            num_bytes_per_warp_per_cta=(cfg.tma_copy_v_tile_bytes // cfg.num_mma_ctas),
        )
        smem_kv_pipeline_cfg = None
    else:
        # SmemKV: TmaUmma, LoadTma -> QK/PV MMA.
        smem_kv_pipeline_cfg = PipelineConfig.create_tma_umma_pipeline_cfg(
            num_stages=cfg.load_kv_stage,
            num_bytes=cfg.tma_kc_subtile_bytes,
            producer_group=tma_producer_group,
            consumer_group=umma_hw_group,
            cta_layout_vmnk=cluster_shape_vmnk,
            consumer_signaling_threads=SignalingThreads.CtaLeader,
            num_bytes_per_warp_per_cta=(cfg.tma_kc_subtile_bytes // cfg.num_mma_ctas),
        )
        smem_k_pipeline_cfg = None
        smem_v_pipeline_cfg = None

    # TmemS: UmmaAsync, 2 stages, MmaTask -> SoftmaxTask
    tmem_s_pipeline_cfg = PipelineConfig.create_umma_async_pipeline_cfg(
        num_stages=cfg.mma_s_stage,
        producer_group=umma_hw_group,
        consumer_group=compute_group_cluster,
        cta_layout_vmnk=cluster_shape_vmnk,
        producer_signaling_threads=SignalingThreads.CtaLeader,
        interleave_stride=(1, 1, 2, 2) if cfg.use_fp8_dual_softmax_schedule else 1,
    )

    # SmemP: AsyncUmma, 2 stages, SoftmaxTask -> MmaTask
    smem_p_pipeline_cfg = PipelineConfig.create_async_umma_pipeline_cfg(
        num_stages=cfg.p_mma_stage,
        producer_group=compute_group_cluster,
        consumer_group=umma_hw_group,
        cta_layout_vmnk=cluster_shape_vmnk,
        consumer_signaling_threads=SignalingThreads.CtaLeader,
        interleave_stride=(2, 2, 1, 1) if cfg.use_fp8_dual_softmax_schedule else 1,
    )

    # TmemCorr: AsyncAsync, 2 stages, Softmax -> Correction
    tmem_corr_pipeline_cfg = PipelineConfig.create_async_async_pipeline_cfg(
        num_stages=cfg.p_cor_stage,
        producer_group=compute_group_local,
        consumer_group=correction_group_local,
        cta_layout_vmnk=cluster_shape_vmnk,
        interleave_stride=(2, 2, 1, 1) if cfg.use_fp8_dual_softmax_schedule else 1,
    )

    # TmemO: UmmaAsync, 1 stage, Mma -> Correction
    tmem_o_pipeline_cfg = PipelineConfig.create_umma_async_pipeline_cfg(
        num_stages=cfg.mma_o_stage,
        producer_group=umma_hw_group,
        consumer_group=correction_group_cluster,
        cta_layout_vmnk=cluster_shape_vmnk,
        producer_signaling_threads=SignalingThreads.CtaLeader,
    )

    # OInit removed: TMEM visibility is ensured by kernel-level named
    # barrier sync before task_manager.run(), so no pipeline needed.

    work_throttle = None
    if use_work_throttle:
        work_throttle = WorkThrottleBarrierResource(
            pipeline_config=PipelineConfig.create_async_async_pipeline_cfg(
                num_stages=2,
                producer_group=pipeline.CooperativeGroup(Agent.Thread, WARP_SIZE),
                consumer_group=pipeline.CooperativeGroup(Agent.Thread, WARP_SIZE),
                cta_layout_vmnk=cluster_shape_vmnk,
                producer_signaling_threads=SignalingThreads.CtaLeader,
                consumer_signaling_threads=SignalingThreads.CtaLeader,
            ),
            name="work_throttle",
        )

    # ──────────────────────────────────────────────────────────────
    # Create resource instances
    # ──────────────────────────────────────────────────────────────

    page_offset_window = PageOffsetWindowResource(
        page_offsets=page_offsets,
        cfg=cfg,
        pipeline_config=None,
        name="page_offset_window",
    )

    smem_q = SmemQResource(
        smem_q_latent=smem_q_latent_arr,
        smem_q_rope=smem_q_rope_arr,
        tma_desc_q_latent=tma_desc_q_latent,
        tma_desc_q_rope=tma_desc_q_rope,
        cu_seqlens_q=cu_seqlens_q,
        logical_num_heads_q=logical_num_heads_q,
        logical_seq_len_q=logical_seq_len_q,
        cfg=cfg,
        pipeline_config=smem_q_pipeline_cfg,
        name="smem_q",
    )

    if cfg.use_fp8_split_mma_schedule:
        smem_k = SmemKResource(
            smem_k=smem_kc_arr,
            page_offsets=page_offsets,
            tma_desc_c_latent=tma_desc_c_latent,
            tma_desc_c_rope=tma_desc_c_rope,
            logical_seq_len_q=logical_seq_len_q,
            cfg=cfg,
            pipeline_config=smem_k_pipeline_cfg,
            name="smem_k",
        )
        smem_v = SmemVResource(
            smem_v=smem_vc_arr,
            page_offsets=page_offsets,
            tma_desc_c_transpose=tma_desc_c_transpose,
            logical_seq_len_q=logical_seq_len_q,
            cfg=cfg,
            pipeline_config=smem_v_pipeline_cfg,
            name="smem_v",
        )
        smem_kv = None
    else:
        smem_kv = SmemKVResource(
            smem_kv=smem_kc_arr,
            tma_desc_c_latent=tma_desc_c_latent,
            tma_desc_c_rope=tma_desc_c_rope,
            tma_desc_c_transpose=tma_desc_c_transpose,
            cfg=cfg,
            pipeline_config=smem_kv_pipeline_cfg,
            name="smem_kv",
        )
        smem_k = None
        smem_v = None

    tmem_s = TmemSResource(
        smem_q_latent=smem_q_latent_arr,
        smem_q_rope=smem_q_rope_arr,
        smem_p=smem_p_arr,
        smem_exchange=None,  # Set at runtime
        softmax_scale_log2=None,  # Set at runtime
        cache_seqs=cache_seqs,
        cu_seqlens_q=cu_seqlens_q,
        split_kv=split_kv,
        logical_num_heads_q=logical_num_heads_q,
        logical_seq_len_q=logical_seq_len_q,
        tiled_mma_qk=tiled_mma_qk,
        cfg=cfg,
        pipeline_config=tmem_s_pipeline_cfg,
        name="tmem_s",
    )

    smem_p = SmemPResource(
        smem_p=smem_p_arr,
        cfg=cfg,
        pipeline_config=smem_p_pipeline_cfg,
        name="smem_p",
    )

    tmem_corr = TmemCorrResource(
        cfg=cfg,
        pipeline_config=tmem_corr_pipeline_cfg,
        name="tmem_corr",
    )

    tmem_o = TmemOResource(
        cfg=cfg,
        tmem_corr_ref=tmem_corr,
        pipeline_config=tmem_o_pipeline_cfg,
        name="tmem_o",
    )

    gmem_o = GmemOResource(
        cfg=cfg,
        output=output,
        partial_output=acc_output,
        lse=lse,
        partial_lse=acc_lse,
        tmem_o_ref=tmem_o,
        tmem_corr_ref=tmem_corr,
        output_scale=None,  # set at runtime
        softmax_scale_log2=None,  # set at runtime
        smem_exchange=None,  # set at runtime
        split_kv=None,  # set at runtime
        cu_seqlens_q=cu_seqlens_q,
        logical_num_heads_q=logical_num_heads_q,
        logical_seq_len_q=logical_seq_len_q,
        name="gmem_o",
    )

    # ──────────────────────────────────────────────────────────────
    # Create tasks
    # ──────────────────────────────────────────────────────────────

    # Warpgroup 2 tasks (warps 8-11) use the low-register producer budget.
    # Keep this explicit for both validation and codegen so the scheduler's
    # register-budget check accounts for the real MLA warpgroup layout.
    wg2_reg_count = cfg.other_reg_num

    if cfg.use_fp8_split_mma_schedule:
        load_k_task = create_load_k_task(
            smem_q,
            smem_k,
            work_queue=work_queue,
            domain=domain,
            num_registers=wg2_reg_count,
        )
        load_v_task = create_load_v_task(
            smem_v,
            work_queue=work_queue,
            domain=domain,
            num_registers=wg2_reg_count,
        )
        mma_task = None
        mma_qk_task = create_mma_qk_direct_task(
            smem_q,
            smem_k,
            tmem_s,
            iterations_qk=cfg.iterations_qk,
            work_queue=work_queue,
            domain=domain,
            num_registers=wg2_reg_count,
        )
        mma_pv_task = create_mma_pv_direct_task(
            smem_v,
            smem_p,
            tmem_o,
            iterations_pv=cfg.iterations_pv_k * cfg.iterations_pv_n,
            iterations_pv_k=cfg.iterations_pv_k,
            iterations_pv_n=cfg.iterations_pv_n,
            per_n_o_pipeline=cfg.use_fp8_split_mma_schedule,
            work_queue=work_queue,
            domain=domain,
            num_registers=wg2_reg_count,
        )
        load_tma_task = None
    else:
        load_k_task = None
        load_v_task = None
        load_tma_task = create_load_tma_task(
            page_offset_window,
            smem_q,
            smem_kv,
            iterations_qk=cfg.iterations_qk_stages,
            iterations_pv=cfg.iterations_pv_stages,
            work_queue=work_queue,
            task_class=non_interleaved_task_class,
            domain=task_domain,
            num_registers=wg2_reg_count,
        )
        mma_task = create_mma_task(
            smem_q,
            smem_kv,
            smem_p,
            tmem_s,
            tmem_o,
            iterations_qk=cfg.iterations_qk_stages,
            iterations_pv=cfg.iterations_pv_stages,
            work_queue=work_queue,
            work_throttle=work_throttle,
            task_class=non_interleaved_task_class,
            domain=task_domain,
            num_registers=wg2_reg_count,
        )
        mma_qk_task = None
        mma_pv_task = None

    # Warpgroup 0 (warps 0-3): all 4 warps are in SoftmaxTask,
    # so setmaxnreg.inc 192 is safe (all warps participate).
    softmax_task = create_softmax_task(
        tmem_s,
        tmem_corr,
        smem_p,
        work_queue=work_queue,
        task_class=(
            MlaInterleavedTask
            if cfg.use_fp8_dual_softmax_schedule
            else non_interleaved_task_class
        ),
        domain=(domain if cfg.use_fp8_dual_softmax_schedule else task_domain),
        domain_start=0,
        step=2 if cfg.use_fp8_dual_softmax_schedule else 1,
        num_registers=cfg.softmax_reg_num,
        softmax_group_id=0,
    )
    if cfg.use_fp8_dual_softmax_schedule:
        second_softmax_task = create_softmax_task(
            tmem_s,
            tmem_corr,
            smem_p,
            work_queue=work_queue,
            task_class=MlaInterleavedTask,
            domain=domain,
            domain_start=1,
            step=2,
            num_registers=cfg.softmax_reg_num,
            warp_idx=cfg.second_compute_warp_ids[0],
            name="SoftmaxOddTask",
            softmax_group_id=1,
        )
    else:
        second_softmax_task = None

    # Warpgroup 1 (warps 4-7): all 4 warps are in CorrectionTask,
    # so setmaxnreg.inc 208 is safe (all warps participate).
    correction_task = create_correction_task(
        tmem_corr,
        tmem_o,
        gmem_o,
        iterations_pv_n=cfg.iterations_pv_n,
        per_n_o_pipeline=cfg.use_fp8_split_mma_schedule,
        work_queue=work_queue,
        task_class=non_interleaved_task_class,
        domain=task_domain,
        num_registers=cfg.correction_reg_num,
    )

    if cfg.use_fp8_split_mma_schedule:
        task_list = [
            load_k_task,
            load_v_task,
        ]
        task_list.extend([mma_qk_task, softmax_task])
        if second_softmax_task is not None:
            task_list.append(second_softmax_task)
        task_list.extend([mma_pv_task, correction_task])
    else:
        task_list = [
            load_tma_task,
        ]
        task_list.extend([mma_task, softmax_task, correction_task])

    if not cfg.use_fp8_split_mma_schedule and use_clc_dynamic:
        padding_task = create_padding_task(
            work_queue=work_queue,
            task_class=non_interleaved_task_class,
            domain=task_domain,
            num_registers=wg2_reg_count,
            warp_idx=10,
        )
        task_list.append(padding_task)
        scheduler_task = create_scheduler_task(
            work_queue,
            work_throttle,
            task_class=non_interleaved_task_class,
            num_registers=wg2_reg_count,
        )
        task_list.append(scheduler_task)
    elif not cfg.use_fp8_split_mma_schedule:
        # BF16 uses only three one-warp producer tasks in warpgroup 2.  Keep an
        # explicit warp-11 placeholder so register validation sees the complete
        # four-warp group with the low-register producer budget.
        padding_task = create_padding_task(
            work_queue=work_queue,
            domain=domain,
            num_registers=wg2_reg_count,
            warp_idx=10,
            num_warps=2,
        )
        task_list.append(padding_task)

    # ──────────────────────────────────────────────────────────────
    # Dependency graph
    # ──────────────────────────────────────────────────────────────
    if cfg.use_fp8_split_mma_schedule:
        resource_dependency_graph = {
            smem_q: [],  # Q loads (independent)
            smem_k: [],  # K loads read page offsets directly
            smem_v: [],  # V loads read page offsets directly
            tmem_s: [smem_k, smem_q],  # QK MMA needs K and Q
            smem_p: [tmem_s],  # softmax reads S -> writes P
            tmem_corr: [tmem_s],  # softmax produces correction
            tmem_o: [smem_p, smem_v],  # PV MMA needs P and V
            gmem_o: [tmem_o, tmem_corr],  # epilogue needs O + correction
        }
        dma_consumer_release_labels = {
            (smem_k, tmem_s): {"k_desc"},
            (smem_v, tmem_o): {"v_desc", "v_desc_n_major"},
        }
    else:
        resource_dependency_graph = {
            page_offset_window: [],  # register-held page-table window
            smem_q: [],  # Q loads (independent)
            # Page offsets are cached into registers inside LoadTmaTask before K/V TMA.
            smem_kv: [],
            tmem_s: [smem_kv, smem_q],  # QK MMA needs K and Q
            smem_p: [tmem_s],  # softmax reads S -> writes P
            tmem_corr: [tmem_s],  # softmax produces correction
            tmem_o: [smem_p, smem_kv],  # PV MMA needs P and V
            gmem_o: [tmem_o, tmem_corr],  # epilogue needs O + correction
        }
        dma_consumer_release_labels = {
            (smem_kv, tmem_s): {"k_desc"},
            (smem_kv, tmem_o): {"v_desc"},
        }
    if work_queue is not None:
        if use_clc_dynamic:
            for resource, dependencies in tuple(resource_dependency_graph.items()):
                if (
                    resource is not work_queue
                    and resource is not page_offset_window
                    and work_queue not in dependencies
                ):
                    dependencies.append(work_queue)
            resource_dependency_graph[work_queue] = (
                [work_queue, work_throttle]
                if work_throttle is not None
                else [work_queue]
            )
            if work_throttle is not None:
                # The leader MMA produces both S and the throttle token after
                # it observes Q for the current work tile.
                resource_dependency_graph[work_throttle] = [tmem_s]
        else:
            resource_dependency_graph[work_queue] = []

    # ──────────────────────────────────────────────────────────────
    # Create TaskManager
    # ──────────────────────────────────────────────────────────────
    skip = not isinstance(domain, int)
    task_manager = TaskManager(
        tasks=task_list,
        resource_dependency_graph=resource_dependency_graph,
        dma_consumer_release_labels=dma_consumer_release_labels,
        skip_validation=skip,
        verbose=verbose,
        exhaustive_deadlock_race_check=exhaustive_deadlock_race_check,
    )

    tmem_resources = [tmem_s, tmem_o, tmem_corr]
    named_resources = {
        "tmem_s": tmem_s,
        "tmem_o": tmem_o,
        "tmem_corr": tmem_corr,
        "gmem_o": gmem_o,
    }
    return task_manager, tmem_resources, named_resources


# GPU Kernel Class
# =====================================================================


class MlaDecodeTs:
    """Warp-specialised MLA decode kernel using the TS framework.

    Usage::

        mla = MlaDecodeTs()
        mla(q_latent, q_rope, c_latent, c_rope, page_offsets,
            o, lse, workspace, split_kv, cache_seqs, cu_seqlens_q,
            block_split_kvs, softmax_scale, output_scale, stream)
    """

    def __init__(
        self,
        acc_dtype=None,
        lse_dtype=None,
        mma_qk_tiler_mn=(128, 128),
        mma_pv_tiler_mn=(128, 256),
        max_active_clusters=56,
        page_size=32,
        is_persistent=True,
        is_var_seq=False,
        is_var_split_kv=False,
        static_split_kv=None,
        static_seq_len_k=None,
        qkv_dtype="bf16",
        out_dtype="bf16",
        rope_dim=64,
        num_heads=128,
        seq_len_q=1,
        batch_size=1,
        mask_type: MaskType | str = MaskType.CAUSAL,
    ):
        """
        Parameters
        ----------
        acc_dtype : cutlass dtype, optional
            Accumulator dtype (default: Float32).
        lse_dtype : cutlass dtype, optional
            LSE dtype (default: Float32).
        mma_qk_tiler_mn : tuple, optional
            MMA tiler shape (M, N) for QK gemm (default: (128, 128)).
        mma_pv_tiler_mn : tuple, optional
            MMA tiler shape (M, N) for PV gemm (default: (128, 256)).
        max_active_clusters : int, optional
            Maximum number of active clusters (default: 56).
        page_size : int, optional
            KV cache page size in tokens (default: 32).
        is_persistent : bool, optional
            Use persistent kernel scheduling (default: True).
        is_var_seq : bool, optional
            Enable variable KV-cache sequence lengths (default: False).
        is_var_split_kv : bool, optional
            Use ``block_split_kvs[batch]`` as a per-batch split cap. Device
            scheduling further contracts that cap from each tile's valid K.
        static_split_kv : int or None, optional
            Compile-time maximum split-KV capacity. The grid and workspace use
            this value, while device scheduling skips the inactive split suffix
            for shorter runtime K lengths.
        static_seq_len_k : int or None, optional
            Compile-time K length for fixed-length launches.  Used only for
            non-empty split-KV-1 specialisation; variable cache_seqs launches
            keep the generic runtime domain path.
        qkv_dtype : str, optional
            Q/K/V dtype name.
        out_dtype : str, optional
            Output dtype name.
        rope_dim : int, optional
            RoPE head dimension.
        num_heads : int, optional
            Logical query-head count used to derive the flat-row tile count.
        seq_len_q : int, optional
            Logical query length used to derive the flat-row tile count.
        batch_size : int, optional
            Host-known batch size used to qualify the standalone reducer
            topology.
        mask_type : MaskType or str, optional
            ``causal`` (default) for bottom-right speculative decoding or
            ``dense`` for full per-batch KV visibility.
        """
        import cutlass as _cutlass

        if acc_dtype is None:
            acc_dtype = _cutlass.Float32
        if lse_dtype is None:
            lse_dtype = _cutlass.Float32

        self.acc_dtype = acc_dtype
        self.lse_dtype = lse_dtype
        self.mma_qk_tiler_mn = mma_qk_tiler_mn
        self.mma_pv_tiler_mn = mma_pv_tiler_mn
        self.max_active_clusters = max_active_clusters
        self.page_size = page_size
        self.is_persistent = is_persistent
        self.is_var_seq = is_var_seq
        self.is_var_split_kv = is_var_split_kv
        self.static_split_kv = static_split_kv
        self.reduction_split_capacity = (
            static_split_kv if static_split_kv is not None else MAX_MLA_SPLITS_KV
        )
        if not 1 <= self.reduction_split_capacity <= MAX_MLA_SPLITS_KV:
            raise ValueError(f"static_split_kv must be in [1, {MAX_MLA_SPLITS_KV}]")
        self.static_seq_len_k = static_seq_len_k
        self.qkv_dtype = qkv_dtype
        self.out_dtype = out_dtype
        self.rope_dim = rope_dim
        self.num_heads = num_heads
        self.seq_len_q = seq_len_q
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.mask_type = normalize_mask_type(mask_type)
        self.query_tile_layout = FlatQueryTileLayout.for_tile(
            num_heads, seq_len_q, mma_qk_tiler_mn[0]
        )
        self.num_q_tiles = self.query_tile_layout.num_tiles
        self.tail_q_rows = self.query_tile_layout.tail_rows
        self.parallel_reduction_topology: ParallelReductionTopology | None = None
        self.use_parallel_reduction = False
        self._parallel_reduction_shape_is_eligible = (
            not is_var_split_kv
            and static_split_kv is not None
            and 2 <= static_split_kv <= 128
            and mma_qk_tiler_mn[0] == 128
            and acc_dtype == _cutlass.Float32
            and lse_dtype == _cutlass.Float32
            and not is_persistent
        )
        self._configure_parallel_reduction_topology()

    def _effective_reduction_shape(self) -> tuple[int, int]:
        """Return physical row/tile extents used by the split workspace."""

        return self.mma_qk_tiler_mn[0], self.num_q_tiles

    def compile_signature(self) -> tuple[object, ...]:
        """Return the complete batch-independent JIT identity."""

        return (
            self.acc_dtype,
            self.lse_dtype,
            self.mma_qk_tiler_mn,
            self.mma_pv_tiler_mn,
            self.max_active_clusters,
            self.page_size,
            self.is_persistent,
            self.is_var_seq,
            self.is_var_split_kv,
            self.static_split_kv,
            self.static_seq_len_k,
            self.qkv_dtype,
            self.out_dtype,
            self.rope_dim,
            self.num_heads,
            self.seq_len_q,
            self.mask_type,
            self.reduction_split_capacity,
            self.query_tile_layout,
            self.num_q_tiles,
            self.tail_q_rows,
            self._parallel_reduction_shape_is_eligible,
            self.use_parallel_reduction,
            self.parallel_reduction_topology,
        )

    def _configure_parallel_reduction_topology(self) -> None:
        """Refresh reducer topology after any host-side launch-shape update."""

        self.parallel_reduction_topology = None
        self.use_parallel_reduction = False
        physical_tile_rows, num_query_tiles = self._effective_reduction_shape()

        # Every reducer shares this normalized partial workspace. Validate its
        # full configured capacity, not only high-split clustered launches.
        if self.reduction_split_capacity > 1:
            validate_parallel_reduction_workspace(
                batch_size=self.batch_size,
                num_heads_q=physical_tile_rows,
                seq_len_q=num_query_tiles,
                splits_kv=self.reduction_split_capacity,
                head_dim=PARALLEL_REDUCTION_HEAD_DIM,
            )

        if not self._parallel_reduction_shape_is_eligible:
            return
        assert self.static_split_kv is not None
        topology = make_q128_wave_limited_parallel_reduction_topology(
            self.static_split_kv,
            logical_rows=(self.num_heads * self.seq_len_q * self.batch_size),
            physical_sm_count=self.max_active_clusters * 2,
            max_cluster_size=8,
        )
        parallel_g1_grid_has_no_padded_rows = (
            self.query_tile_layout.total_rows == physical_tile_rows * num_query_tiles
        )
        # Small split counts use G1 only when row coarsening leaves a sub-wave
        # grid and the producer generated enough work to amortize one CTA per
        # physical row. Padded M128 tails and intermediate split counts retain
        # the compact reference grid; high split counts may use clusters.
        use_small_split_g1 = (
            self.static_split_kv <= 16
            and topology is not None
            and topology.cluster_size == 1
            and parallel_g1_grid_has_no_padded_rows
            and should_use_q128_g1_parallel_reducer(
                batch_size=self.batch_size,
                physical_rows_per_batch=(physical_tile_rows * num_query_tiles),
                producer_ctas=(
                    self.batch_size * num_query_tiles * self.static_split_kv * 2
                ),
                reference_rows_per_cta=REDUCTION_ROWS_PER_CTA,
                physical_sm_count=self.max_active_clusters * 2,
            )
        )
        use_high_split_cluster = (
            self.static_split_kv > 32
            and topology is not None
            and topology.cluster_size > 1
        )
        if use_small_split_g1 or use_high_split_cluster:
            self.parallel_reduction_topology = topology
            self.use_parallel_reduction = True

    @cute.jit
    def __call__(
        self,
        q_latent: cute.Tensor,
        q_rope: cute.Tensor,
        c_latent: cute.Tensor,
        c_rope: cute.Tensor,
        page_offsets: cute.Tensor,
        o: cute.Tensor,
        lse: cute.Tensor,
        workspace: cute.Tensor,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        cu_seqlens_q: cute.Tensor | None,
        block_split_kvs: cute.Tensor,
        softmax_scale: cutlass.Float32,
        output_scale: cutlass.Float32,
        stream: object,
    ):
        """Execute the MLA decode TS kernel."""
        cfg = make_mla_decode_config(
            mma_qk_tiler_mn=self.mma_qk_tiler_mn,
            mma_pv_tiler_mn=self.mma_pv_tiler_mn,
            rope_dim=self.rope_dim,
            page_size=self.page_size,
            qkv_dtype=self.qkv_dtype,
            o_dtype=self.out_dtype,
            max_active_clusters=self.max_active_clusters,
            is_persistent=self.is_persistent,
            is_var_seq=self.is_var_seq,
            is_var_split_kv=self.is_var_split_kv,
            mask_type=self.mask_type,
        )
        physical_tile_rows = self.mma_qk_tiler_mn[0]
        num_query_tiles = self.num_q_tiles
        # Fixed public tensors retain [H,D,SQ,B]/[H,SQ,B]; variable-Q tensors
        # compact batches into [H,D,totalQ]/[H,totalQ]. The scheduler/workspace
        # use physical flat-query tile coordinates, while resources map
        # each valid row back to its logical fixed or ragged storage location.
        if cutlass.const_expr(cu_seqlens_q is not None):
            runtime_assert(
                cute.size(cu_seqlens_q) == cute.size(cache_seqs) + Int32(1),
                "cu_seqlens_q must contain one more offset than cache_seqs",
            )
            batch_size = cute.size(cu_seqlens_q) - Int32(1)
        else:
            batch_size = cute.size(o.shape[3])
            runtime_assert(
                batch_size == cute.size(cache_seqs),
                "fixed output batch size must match cache_seqs",
            )

        runtime_assert(
            q_latent.stride[2] == q_latent.shape[0] * q_latent.stride[0],
            "q_latent must be compact across the head and query dimensions",
        )
        runtime_assert(
            q_rope.stride[2] == q_rope.shape[0] * q_rope.stride[0],
            "q_rope must be compact across the head and query dimensions",
        )
        runtime_assert(
            o.stride[1] == 1,
            "o must have a contiguous dimension axis",
        )
        runtime_assert(
            o.stride[0] == o.shape[1] * o.stride[1],
            "o must be compact from the dimension axis into the head axis",
        )
        runtime_assert(
            o.stride[2] == o.shape[0] * o.stride[0],
            "o must be compact from the head axis into the query axis",
        )
        if cutlass.const_expr(cu_seqlens_q is None):
            runtime_assert(
                o.stride[3] == o.shape[2] * o.stride[2],
                "o must be compact from the query axis into the batch axis",
            )
        runtime_assert(
            lse.stride[0] == 1,
            "lse must have a contiguous head axis",
        )
        runtime_assert(
            lse.stride[1] == lse.shape[0] * lse.stride[0],
            "lse must be compact from the head axis into the query axis",
        )
        if cutlass.const_expr(cu_seqlens_q is None):
            runtime_assert(
                lse.stride[2] == lse.shape[1] * lse.stride[1],
                "lse must be compact from the query axis into the batch axis",
            )

        def _flatten_query_rows_for_tma(t):
            """Expose logical ``(SQ, H)`` as one bounded TMA row dimension."""
            if cutlass.const_expr(cu_seqlens_q is not None):
                return cute.make_tensor(
                    t.iterator,
                    cute.make_layout(
                        (t.shape[1], t.shape[0] * t.shape[2]),
                        stride=(t.stride[1], t.stride[0]),
                    ),
                )
            return cute.make_tensor(
                t.iterator,
                cute.make_layout(
                    (t.shape[1], t.shape[0] * t.shape[2], t.shape[3]),
                    stride=(t.stride[1], t.stride[0], t.stride[3]),
                ),
            )

        # Create TMA descriptors (same as bare metal)

        # Keep the descriptor extent at the logical H*SQ row count.  The final
        # physical query tile still requests a full M tile; tensor-map OOB fill
        # supplies zeros for its tail without changing the public Q shape.
        q_latent_tma = _flatten_query_rows_for_tma(q_latent)
        if cutlass.const_expr(cu_seqlens_q is not None):
            tma_desc_q_latent = create_tensor_map_ragged_from_tensor(
                q_latent_tma,
                box_dims=(cfg.mma_qk_tiler[2], cfg.mma_qk_tiler[0] // 2),
                ragged_dim=1,
                stride_order=(0, 1),
                swizzle=cuda.TensorMapSwizzle.s128b,
                l2_promotion=cuda.TensorMapL2Promotion.l2_128b,
            )
        else:
            tma_desc_q_latent = create_tensor_map_tiled_from_view(
                q_latent_tma,
                box_dims=(cfg.mma_qk_tiler[2], cfg.mma_qk_tiler[0] // 2, 1),
                stride_order=(0, 1, 2),
                swizzle=cuda.TensorMapSwizzle.s128b,
                l2_promotion=cuda.TensorMapL2Promotion.l2_128b,
            )

        if cutlass.const_expr(cfg.rope_dim > 0):
            q_rope_tma = _flatten_query_rows_for_tma(q_rope)
            q_rope_swizzle = cuda.TensorMapSwizzle.s128b
            if cutlass.const_expr(cfg.is_fp8_qkv() and cfg.rope_dim == 64):
                q_rope_swizzle = cuda.TensorMapSwizzle.s64b
            if cutlass.const_expr(cu_seqlens_q is not None):
                tma_desc_q_rope = create_tensor_map_ragged_from_tensor(
                    q_rope_tma,
                    box_dims=(
                        cfg.mma_qk_rope_tiler[2],
                        cfg.mma_qk_rope_tiler[0] // 2,
                    ),
                    ragged_dim=1,
                    stride_order=(0, 1),
                    swizzle=q_rope_swizzle,
                    l2_promotion=cuda.TensorMapL2Promotion.l2_128b,
                )
            else:
                tma_desc_q_rope = create_tensor_map_tiled_from_view(
                    q_rope_tma,
                    box_dims=(
                        cfg.mma_qk_rope_tiler[2],
                        cfg.mma_qk_rope_tiler[0] // 2,
                        1,
                    ),
                    stride_order=(0, 1, 2),
                    swizzle=q_rope_swizzle,
                    l2_promotion=cuda.TensorMapL2Promotion.l2_128b,
                )
        else:
            tma_desc_q_rope = tma_desc_q_latent

        c_latent_tma = cute.make_tensor(
            c_latent.iterator,
            cute.select(c_latent.layout, mode=[1, 0, 2]),
        )
        tma_desc_c_latent = create_tensor_map_tiled_from_view(
            c_latent_tma,
            box_dims=(cfg.mma_qk_tiler[2], cfg.kc_page_tile_size, 1),
            stride_order=(0, 1, 2),
            swizzle=cuda.TensorMapSwizzle.s128b,
            l2_promotion=cuda.TensorMapL2Promotion.l2_128b,
        )

        if cutlass.const_expr(cfg.rope_dim > 0):
            c_rope_tma = cute.make_tensor(
                c_rope.iterator,
                cute.select(c_rope.layout, mode=[1, 0, 2]),
            )
            c_rope_swizzle = cuda.TensorMapSwizzle.s128b
            if cutlass.const_expr(cfg.is_fp8_qkv() and cfg.rope_dim == 64):
                c_rope_swizzle = cuda.TensorMapSwizzle.s64b
            tma_desc_c_rope = create_tensor_map_tiled_from_view(
                c_rope_tma,
                box_dims=(cfg.mma_qk_rope_tiler[2], cfg.kc_page_tile_size, 1),
                stride_order=(0, 1, 2),
                swizzle=c_rope_swizzle,
                l2_promotion=cuda.TensorMapL2Promotion.l2_128b,
            )
        else:
            tma_desc_c_rope = tma_desc_c_latent

        c_transpose_swizzle = cuda.TensorMapSwizzle.s128b
        if cutlass.const_expr(cfg.is_fp8_qkv() and cfg.mma_pv_tiler[2] == 64):
            c_transpose_swizzle = cuda.TensorMapSwizzle.s64b
        c_latent_transpose_layout = cute.select(c_latent.layout, mode=[1, 0, 2])
        c_latent_transpose = cute.make_tensor(
            c_latent.iterator, c_latent_transpose_layout
        )
        # The physical page controls only the GMEM coordinate.  TMA assembles
        # fixed K32 SMEM blocks for PV from one or more page-bounded copies.
        tma_desc_c_transpose = create_tensor_map_tiled_from_view(
            c_latent_transpose,
            box_dims=(V_TMA_LATENT_ELEMENTS, cfg.v_tma_token_count, 1),
            stride_order=(0, 1, 2),
            swizzle=c_transpose_swizzle,
            l2_promotion=cuda.TensorMapL2Promotion.l2_128b,
        )

        softmax_scale_log2 = softmax_scale * LOG2_E

        kernel_split_kv = (
            Int32(self.static_split_kv)
            if cutlass.const_expr(self.static_split_kv is not None)
            else split_kv
        )

        # Compute grid
        tile_sched_params = create_mla_static_tile_scheduler_params(
            self.is_persistent,
            batch_size,
            Int32(num_query_tiles),
            cfg.cluster_shape_mnk,
            kernel_split_kv,
        )
        use_clc_dynamic = self.is_persistent and not cfg.is_fp8_qkv()
        clc_tile_sched_params = None
        if cutlass.const_expr(use_clc_dynamic):
            # Keep the physical query-tile dimension in grid X and flatten
            # only split/batch into grid Z.  Besides avoiding a hot-path
            # S/B decode for every stolen tile, this preserves the natural
            # 2CTA query-cluster raster used by the nonpersistent launch.
            clc_tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
                (
                    cfg.cluster_shape_mnk[0] * Int32(num_query_tiles),
                    1,
                    batch_size * kernel_split_kv,
                ),
                cfg.cluster_shape_mnk,
            )
            grid = clc_tile_sched_params.get_grid_shape()
        else:
            grid = MLAStaticTileScheduler.get_grid_shape(
                tile_sched_params, self.max_active_clusters
            )

        # Initialize workspace for split_kv > 1
        acc_o, acc_lse = self.initialize_workspace(
            Int32(physical_tile_rows),
            cfg.latent_dim,  # D
            Int32(num_query_tiles),
            batch_size,
            kernel_split_kv,
            workspace,
        )
        # A one-wave producer can publish its dependent reducer launch while
        # retiring, hiding launch latency without admitting reducer CTAs into
        # an actively grid-striding persistent producer.
        use_one_wave_reducer_pdl = acc_o is not None and not self.is_persistent

        self.split_kv_kernel(
            tma_desc_q_latent,
            tma_desc_q_rope,
            tma_desc_c_latent,
            tma_desc_c_rope,
            tma_desc_c_transpose,
            c_latent,
            c_rope,
            page_offsets,
            o,
            lse,
            acc_o,
            acc_lse,
            kernel_split_kv,
            cache_seqs,
            cu_seqlens_q,
            block_split_kvs,
            softmax_scale_log2,
            output_scale,
            tile_sched_params,
            clc_tile_sched_params,
        ).launch(
            grid=grid,
            block=[cfg.threads_per_cta, 1, 1],
            cluster=cfg.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=use_one_wave_reducer_pdl,
        )

        # Reduction kernel: combine per-split results when split_kv > 1
        if cutlass.const_expr(acc_o is not None):
            if cutlass.const_expr(self.use_parallel_reduction):
                topology = self.parallel_reduction_topology
                self.parallel_reduction_kernel(
                    o,
                    lse,
                    acc_o,
                    acc_lse,
                    kernel_split_kv,
                    cache_seqs,
                    cu_seqlens_q,
                    block_split_kvs,
                ).launch(
                    grid=(
                        physical_tile_rows * topology.cluster_size,
                        num_query_tiles,
                        batch_size,
                    ),
                    block=[PARALLEL_REDUCTION_THREADS, 1, 1],
                    cluster=[topology.cluster_size, 1, 1],
                    stream=stream,
                    min_blocks_per_mp=1,
                    use_pdl=use_one_wave_reducer_pdl,
                )
            else:
                logical_query_rows = self.num_heads * self.seq_len_q
                self.reduction_kernel(
                    o,
                    lse,
                    acc_o,
                    acc_lse,
                    kernel_split_kv,
                    cache_seqs,
                    cu_seqlens_q,
                    block_split_kvs,
                ).launch(
                    grid=(
                        ceil_div(logical_query_rows, REDUCTION_ROWS_PER_CTA),
                        1,
                        batch_size,
                    ),
                    block=[REDUCTION_ROWS_PER_CTA * REDUCTION_THREADS_PER_ROW, 1, 1],
                    smem=(
                        REDUCTION_ROWS_PER_CTA
                        * self.reduction_split_capacity
                        * self.lse_dtype.width
                        // 8
                    ),
                    stream=stream,
                    min_blocks_per_mp=2,
                    use_pdl=use_one_wave_reducer_pdl,
                )

    @cute.kernel
    def split_kv_kernel(
        self,
        tma_desc_q_latent: cutlass.GridConstant[cuda.TensorMap],
        tma_desc_q_rope: cutlass.GridConstant[cuda.TensorMap],
        tma_desc_c_latent: cutlass.GridConstant[cuda.TensorMap],
        tma_desc_c_rope: cutlass.GridConstant[cuda.TensorMap],
        tma_desc_c_transpose: cutlass.GridConstant[cuda.TensorMap],
        c_latent: cute.Tensor,
        c_rope: cute.Tensor,
        page_offsets: cute.Tensor,
        o: cute.Tensor,
        lse: cute.Tensor,
        acc_o: cute.Tensor,
        acc_lse: cute.Tensor,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        cu_seqlens_q: cute.Tensor | None,
        block_split_kvs: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        output_scale: cutlass.Float32,
        tile_sched_params: MLAStaticTileSchedulerParams,
        clc_tile_sched_params: object,
    ) -> None:
        """MLA decode TS kernel: persistent tile-scheduled execution."""
        cfg = make_mla_decode_config(
            mma_qk_tiler_mn=self.mma_qk_tiler_mn,
            mma_pv_tiler_mn=self.mma_pv_tiler_mn,
            rope_dim=self.rope_dim,
            page_size=self.page_size,
            qkv_dtype=self.qkv_dtype,
            o_dtype=self.out_dtype,
            max_active_clusters=self.max_active_clusters,
            is_persistent=self.is_persistent,
            is_var_seq=self.is_var_seq,
            is_var_split_kv=self.is_var_split_kv,
            mask_type=self.mask_type,
        )
        num_query_tiles = self.num_q_tiles
        use_clc_dynamic = self.is_persistent and not cfg.is_fp8_qkv()
        tiled_mma_qk = None
        if cutlass.const_expr(cfg.is_fp8_qkv()):
            tiled_mma_qk = sm100_utils.make_trivial_tiled_mma(
                qkv_dtype(cfg),
                qkv_dtype(cfg),
                OperandMajorMode.K,
                OperandMajorMode.K,
                self.acc_dtype,
                tcgen05.CtaGroup.TWO,
                cfg.mma_qk_tiler[:2],
            )

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        cluster_idx, _, _ = cute.arch.block_idx()

        mma_tile_coord_v = cluster_idx % 2

        # Prefetch TMA descriptors on MMA warp
        if warp_idx == cfg.mma_warp_id:
            prims.prefetch_tensormap(tma_desc_q_latent.get_ptr())
            prims.prefetch_tensormap(tma_desc_q_rope.get_ptr())
            prims.prefetch_tensormap(tma_desc_c_latent.get_ptr())
            prims.prefetch_tensormap(tma_desc_c_rope.get_ptr())
            prims.prefetch_tensormap(tma_desc_c_transpose.get_ptr())

        # Allocate SMEM
        qkv_element_dtype = qkv_dtype(cfg)
        smem_q_latent_arr = cutlass.Array(
            qkv_element_dtype,
            cfg.smem_q_latent_elems,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        smem_q_rope_arr = cutlass.Array(
            qkv_element_dtype,
            max(1, cfg.smem_q_rope_elems),
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        smem_kc_arr = cutlass.Array(
            qkv_element_dtype,
            cfg.smem_kc_elems,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        smem_vc_arr = None
        if cutlass.const_expr(cfg.use_fp8_split_mma_schedule):
            smem_vc_arr = cutlass.Array(
                qkv_element_dtype,
                cfg.smem_vc_elems,
                space=cutlass.AddressSpace.smem,
                alignment=1024,
            )
        smem_p_arr = cutlass.Array(
            qkv_element_dtype,
            cfg.smem_p_elems,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        softmax_exchange_arr = cutlass.Array(
            self.acc_dtype,
            cfg.softmax_exchange_elems,
            space=cutlass.AddressSpace.smem,
            alignment=4,
        )
        epilogue_exchange_arr = cutlass.Array(
            self.acc_dtype,
            cfg.num_compute_warps * cfg.threads_per_warp,
            space=cutlass.AddressSpace.smem,
            alignment=4,
        )
        tmem_holding_buf_arr = cutlass.Array(
            Int32, 1, space=cutlass.AddressSpace.smem, alignment=4
        )
        tmem_dealloc_mbar_arr = cutlass.Array(
            Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
        )
        clc_response_ptr = None
        if cutlass.const_expr(use_clc_dynamic):
            # Keep both response stages in the ordinary dynamic-SMEM arena.
            # ``alloc_smem`` creates a separately rounded static section, which
            # needlessly exceeds this kernel's near-capacity SMEM budget.
            clc_response_arr = cutlass.Array(
                cutlass.Int128,
                2,
                space=cutlass.AddressSpace.smem,
                alignment=16,
            )
            clc_response_ptr = cute.make_ptr(
                cutlass.Int128,
                clc_response_arr.data_ptr(),
                mem_space=cutlass.AddressSpace.smem,
            )

        # Init dealloc mbarrier
        if warp_idx == cfg.mma_warp_id:
            if prims.elect_sync():
                prims.mbarrier_init(tmem_dealloc_mbar_arr, TMEM_DEALLOC_MBAR_THREADS)

        # setmaxnreg.dec: ALL warps in warpgroup 2 (warps 8-11)
        if warp_idx >= 8:
            prims.setmaxregister(cfg.other_reg_num, prims.SetMaxRegisterAction.DECREASE)

        # Workaround: avoid CSE-induced register spills. The tile
        # scheduler calls grid_dim(), producing nctaid SSA values in the shared
        # prologue; CSE merges them across tasks, preventing per-task tile
        # schedulers from getting independent copies and causing register spills.
        # Remove once the compiler scopes nctaid reads per task.
        # Tile decomposition — compute blk_coord from ctaid.x WITHOUT creating
        # a full tile scheduler.
        if cutlass.const_expr(use_clc_dynamic):
            query_cluster_idx, _, split_batch_idx = cute.arch.block_idx()
            seq_q_idx = query_cluster_idx // Int32(cfg.cluster_shape_mnk[0])
            cluster_idx = query_cluster_idx % Int32(cfg.cluster_shape_mnk[0])
            split_kv_idx, batch_idx = divmod_constexpr_power_of_two_or_fdd(
                split_batch_idx,
                None,
                tile_sched_params.problem_shape_b_fdd,
            )
            blk_coord = (cluster_idx, seq_q_idx, batch_idx, split_kv_idx)
        elif cutlass.const_expr(self.is_persistent):
            current_work_linear_idx = cute.arch.block_idx()[0]
            current_work_cluster_batch = current_work_linear_idx // Int32(
                cfg.cluster_shape_mnk[0]
            )
            cluster_idx = current_work_linear_idx % Int32(cfg.cluster_shape_mnk[0])
            current_work_after_seq_q, seq_q_idx = divmod_constexpr_power_of_two_or_fdd(
                current_work_cluster_batch,
                num_query_tiles,
                tile_sched_params.problem_shape_s_fdd,
            )
            current_work_after_batch, batch_idx = divmod_constexpr_power_of_two_or_fdd(
                current_work_after_seq_q,
                None,
                tile_sched_params.problem_shape_b_fdd,
            )
            _, split_kv_idx = divmod(
                current_work_after_batch, tile_sched_params.split_kv_fdd
            )
            blk_coord = (cluster_idx, seq_q_idx, batch_idx, split_kv_idx)
        else:
            cluster_idx, seq_batch_idx, split_kv_idx = cute.arch.block_idx()
            seq_q_idx, batch_idx = divmod_constexpr_power_of_two_or_fdd(
                seq_batch_idx,
                None,
                tile_sched_params.problem_shape_b_fdd,
            )
            blk_coord = (cluster_idx, seq_q_idx, batch_idx, split_kv_idx)
        tile_cluster_idx, tile_seq_q_idx, tile_batch_idx, tile_split_kv_idx = blk_coord
        del tile_cluster_idx

        fixed_nonempty_single_split = (
            not self.is_var_seq
            and not self.is_var_split_kv
            and self.static_split_kv == 1
            and cu_seqlens_q is None
        )
        if cutlass.const_expr(fixed_nonempty_single_split):
            max_split_kv = Int32(1)
            exit_early = False
        elif cutlass.const_expr(not self.is_persistent):
            # Every task already derives its graph-live K/Q domain through the
            # work queue. Let a zero-domain task skip its data schedule instead
            # of recomputing the same metadata in an all-warp CTA prologue.
            max_split_kv = (
                Int32(self.static_split_kv)
                if cutlass.const_expr(self.static_split_kv is not None)
                else split_kv
            )
            exit_early = False
        else:
            # Compute the initial tile's active Q/split state for static early
            # exit. Persistent CTAs stay live because a later grid-stride tile
            # can be active even when their initial logical tile is padded.
            # The full k_tile_count is recomputed per-task inside
            # MlaTask._run_task_body_persistent to keep SSA live ranges short.
            if cutlass.const_expr(self.static_seq_len_k is not None):
                K = Int32(self.static_seq_len_k)
            else:
                K = cache_seqs[tile_batch_idx]

            _, q_len = query_batch_bounds(
                cu_seqlens_q,
                tile_batch_idx,
                self.seq_len_q,
            )
            query_tile_has_rows = runtime_flat_query_tile_has_rows(
                tile_seq_q_idx,
                self.mma_qk_tiler_mn[0],
                self.num_heads,
                self.seq_len_q,
                cu_seqlens_q,
                tile_batch_idx,
            )
            if cutlass.const_expr(
                cfg.mask_type == MaskType.CAUSAL.value and self.seq_len_q > 1
            ):
                _, _, logical_q_idx, _, _ = flat_query_row_state(
                    Int32(self.mma_qk_tiler_mn[0] - 1),
                    tile_seq_q_idx,
                    self.mma_qk_tiler_mn[0],
                    self.num_heads,
                    self.seq_len_q,
                    cu_seqlens_q,
                    tile_batch_idx,
                )
                K = mask_visible_k_length(cfg.mask_type, K, logical_q_idx, q_len)
            K = K if query_tile_has_rows else Int32(0)
            # split_kv is the static launch/workspace capacity. Runtime K
            # contracts the optional per-batch cap to an active prefix.
            max_split_kv = (
                Int32(self.static_split_kv)
                if cutlass.const_expr(self.static_split_kv is not None)
                else split_kv
            )
            if cutlass.const_expr(
                self.static_split_kv is not None and not self.is_var_split_kv
            ):
                split_kv_cap = max_split_kv
            else:
                split_kv_cap = runtime_split_kv_cap(
                    max_split_kv,
                    self.is_var_split_kv,
                    block_split_kvs,
                    tile_batch_idx,
                )
            k_tile_total = (K + cfg.mma_qk_tiler[1] - 1) // cfg.mma_qk_tiler[1]
            _, k_tile_count = runtime_split_tile_range(
                k_tile_total,
                split_kv_cap,
                tile_split_kv_idx,
            )
            exit_early = k_tile_count <= Int32(0)
            if cutlass.const_expr(self.is_persistent):
                # A physical persistent CTA may own an empty initial split but
                # later grid-stride to a nonempty batch/Q tile. Let MlaTask
                # skip empty logical tiles instead of terminating the CTA.
                exit_early = False

        if exit_early:
            # The ptxas-generated CTA_2 TMEM lifecycle barrier is initialized
            # in the shared prologue.  Order that initialization across both
            # CTAs before tcgen05_alloc, matching the active path below.
            prims.fence_mbarrier_init()
            prims.barrier_cluster_arrive_relaxed()
            prims.barrier_cluster_wait()

            # TMEM alloc + sync even for empty CTAs to keep cluster in sync.
            if warp_idx == cfg.mma_warp_id:
                prims.tcgen05_alloc(
                    tmem_holding_buf_arr, cfg.num_tmem_cols, group="cta_2"
                )
                prims.tcgen05_relinquish_alloc_permit(group="cta_2")
            participates_in_tmem_sync = warp_idx <= cfg.mma_warp_id
            if cutlass.const_expr(cfg.use_fp8_split_mma_schedule):
                participates_in_tmem_sync = participates_in_tmem_sync or (
                    warp_idx == cfg.pv_mma_warp_id
                )
            if cutlass.const_expr(cfg.use_fp8_dual_softmax_schedule):
                participates_in_tmem_sync = participates_in_tmem_sync or (
                    warp_idx >= cfg.second_compute_warp_ids[0]
                    and warp_idx <= cfg.second_compute_warp_ids[-1]
                )
            if participates_in_tmem_sync:
                prims.barrier_cta_sync(
                    cfg.tmem_sync_bar_id, thread_count=cfg.tmem_sync_bar_threads
                )
        else:
            # Build TaskManager and initialize pipelines BEFORE tcgen05_alloc.
            # Pipeline mbarrier init must happen first because ptxas inserts
            # TMEM lifecycle barrier code around tcgen05_alloc that uses
            # static SMEM at offset 0x40, which overlaps with the region
            # that pipeline mbarrier init writes to. Without this ordering,
            # the lifecycle barrier is uninitialized and crashes.

            # Create one MLA-coordinate work queue for persistent scheduling.
            # Static dispatch wraps MLAStaticTileScheduler and needs no
            # response pipeline. CLC dispatch installs the fetch pipeline and
            # response parameters below; all participating tasks still consume
            # the same cached per-tile MLA coordinate and K-domain state.
            work_queue_pipeline_config = None
            work_queue_tile_scheduler_config = None
            if cutlass.const_expr(use_clc_dynamic):
                work_queue_pipeline_config = (
                    PipelineConfig.create_clc_fetch_async_pipeline_cfg(
                        num_stages=2,
                        num_bytes=16,
                        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                        consumer_group=pipeline.CooperativeGroup(
                            pipeline.Agent.Thread,
                            # CTA1 omits both the leader-only MMA warp and the
                            # leader-only scheduler warp. Exclude both from the
                            # cluster-wide empty-barrier arrival count.
                            cfg.threads_per_cta * cfg.num_mma_ctas
                            - 2 * cfg.threads_per_warp,
                        ),
                        cta_layout_vmnk=(cfg.num_mma_ctas, 1, 1, 1),
                        producer_signaling_threads=SignalingThreads.CtaLeader,
                        consumer_signaling_threads=SignalingThreads.All,
                    )
                )
                work_queue_tile_scheduler_config = TileSchedulerConfig.create_clc_dynamic_persistent_tile_scheduler_params(
                    tile_scheduler_params=clc_tile_sched_params,
                    response_ptr=clc_response_ptr,
                )

            mla_work_queue = MlaWorkQueue(
                tile_sched_params=tile_sched_params,
                cache_seqs=cache_seqs,
                split_kv=max_split_kv,
                block_split_kvs=block_split_kvs,
                is_var_split_kv=self.is_var_split_kv,
                cfg=cfg,
                static_split_kv=self.static_split_kv,
                static_seq_len_k=self.static_seq_len_k,
                cu_seqlens_q=cu_seqlens_q,
                logical_num_heads_q=self.num_heads,
                logical_seq_len_q=self.seq_len_q,
                static_problem_shape_b=None,
                static_problem_shape_s=num_query_tiles,
                use_clc_dynamic=use_clc_dynamic,
                tile_scheduler_config=work_queue_tile_scheduler_config,
                pipeline_config=work_queue_pipeline_config,
                name="mla_work_queue",
            )

            task_manager, _tmem_resources, named_res = build_mla_decode_task_manager(
                cfg=cfg,
                smem_q_latent_arr=smem_q_latent_arr,
                smem_q_rope_arr=smem_q_rope_arr,
                smem_kc_arr=smem_kc_arr,
                smem_vc_arr=smem_vc_arr,
                smem_p_arr=smem_p_arr,
                tma_desc_q_latent=tma_desc_q_latent.get_ptr(),
                tma_desc_q_rope=tma_desc_q_rope.get_ptr(),
                tma_desc_c_latent=tma_desc_c_latent.get_ptr(),
                tma_desc_c_rope=tma_desc_c_rope.get_ptr(),
                tma_desc_c_transpose=tma_desc_c_transpose.get_ptr(),
                page_offsets=page_offsets,
                blk_coord=blk_coord,
                tidx=tidx,
                output=o,
                acc_output=acc_o,
                lse=lse,
                acc_lse=acc_lse,
                domain=Int32(1),  # dummy; MlaTask recomputes per-task to avoid spills
                work_queue=mla_work_queue,
                cache_seqs=cache_seqs,
                cu_seqlens_q=cu_seqlens_q,
                split_kv=max_split_kv,
                logical_num_heads_q=self.num_heads,
                logical_seq_len_q=self.seq_len_q,
                tiled_mma_qk=tiled_mma_qk,
            )

            # Initialize pipelines (creates mbarriers in SMEM)
            task_manager.setup_resources_and_tasks()

            # Fence mbarrier init then cluster sync so both CTAs see
            # initialized barriers before tcgen05_alloc.  The alloc with
            # CTA_2 group uses ptxas-generated lifecycle barriers in
            # static SMEM [0,1024) that require cross-CTA coordination.
            prims.fence_mbarrier_init()
            prims.barrier_cluster_arrive_relaxed()
            prims.barrier_cluster_wait()

            # TMEM allocation AFTER lifecycle init + cluster sync (warp 8 only)
            if warp_idx == cfg.mma_warp_id:
                prims.tcgen05_alloc(
                    tmem_holding_buf_arr, cfg.num_tmem_cols, group="cta_2"
                )
                prims.tcgen05_relinquish_alloc_permit(group="cta_2")

            # tcgen05.alloc.cta_group::2 is itself cluster-synchronous; the
            # pre-allocation cluster barrier above protects lifecycle-mbarrier
            # initialization, while another full-cluster barrier here only
            # serializes the first consumers of the published TMEM base.

            # TMEM sync barrier: only warps 0-8 participate
            participates_in_tmem_sync = warp_idx <= cfg.mma_warp_id
            if cutlass.const_expr(cfg.use_fp8_split_mma_schedule):
                participates_in_tmem_sync = participates_in_tmem_sync or (
                    warp_idx == cfg.pv_mma_warp_id
                )
            if cutlass.const_expr(cfg.use_fp8_dual_softmax_schedule):
                participates_in_tmem_sync = participates_in_tmem_sync or (
                    warp_idx >= cfg.second_compute_warp_ids[0]
                    and warp_idx <= cfg.second_compute_warp_ids[-1]
                )
            if participates_in_tmem_sync:
                prims.barrier_cta_sync(
                    cfg.tmem_sync_bar_id, thread_count=cfg.tmem_sync_bar_threads
                )

            # Only the warps synchronized above consume TMEM resources.  Do
            # not let the loader/page/padding warps race the allocation's
            # publication slot merely to materialize an unused pointer.
            tmem_base_addr = Int32(0)
            if participates_in_tmem_sync:
                tmem_base_addr = tmem_holding_buf_arr.load()

            # Set TMEM base address on resources
            named_res["tmem_s"].tmem_base_addr = tmem_base_addr
            named_res["tmem_o"].tmem_base_addr = tmem_base_addr
            named_res["tmem_corr"].tmem_base_addr = tmem_base_addr

            # Set runtime params on TmemS (softmax)
            named_res["tmem_s"].softmax_scale_log2 = softmax_scale_log2
            named_res["tmem_s"].smem_exchange = softmax_exchange_arr

            named_res[
                "tmem_corr"
            ].smem_exchange = epilogue_exchange_arr.data_ptr().toint(Int32)

            # Set runtime params on GmemO (epilogue)
            named_res["gmem_o"].output_scale = output_scale
            named_res["gmem_o"].softmax_scale_log2 = softmax_scale_log2
            named_res["gmem_o"].smem_exchange = epilogue_exchange_arr.data_ptr().toint(
                Int32
            )
            named_res["gmem_o"].split_kv = max_split_kv

            task_manager.run()

        # TMEM deallocation (MMA warp)
        if warp_idx == cfg.mma_warp_id:
            cta_rank = cute.arch.make_warp_uniform(mma_tile_coord_v)
            peer_cta_rank = cta_rank ^ 1
            peer_mbar = prims.mapa(tmem_dealloc_mbar_arr, peer_cta_rank)
            prims.mbarrier_arrive(peer_mbar)
            while not prims.mbarrier_try_wait_parity(tmem_dealloc_mbar_arr, 0):
                pass
            tmem_arr_for_dealloc = prims.make_tmem_ptr(
                tmem_holding_buf_arr.load(), self.acc_dtype
            )
            prims.tcgen05_dealloc(
                tmem_arr_for_dealloc,
                cfg.num_tmem_cols,
                group="cta_2",
            )
            # Each producer CTA publishes completion only after its paired
            # TMEM lifetime and all scheduled output work have retired.
            if cutlass.const_expr(acc_o is not None and not self.is_persistent):
                if prims.elect_sync():
                    prims.griddepcontrol(kind=prims.GridDepAction.LAUNCH_DEPENDENTS)

    @cute.jit
    def initialize_workspace(
        self,
        H: cutlass.Int32,
        D: cutlass.Int32,
        S: cutlass.Int32,
        B: cutlass.Int32,
        split_kv: cutlass.Int32,
        workspace: cute.Tensor,
    ):
        """Construct acc_o and acc_lse tensors from the workspace buffer."""
        acc_o, acc_lse = None, None
        if cutlass.const_expr(workspace is not None):
            # Workspace strides are aligned to 256 bits, expressed in Float16
            # elements because split-KV partial O is BF16 even for FP8 output.
            align = 256 // cutlass.Float16.width
            acc_o_layout = cute.make_layout(
                (H, split_kv, D, S, B),
                stride=(
                    cute.assume(split_kv * D, align),
                    cute.assume(D, align),
                    1,
                    cute.assume(split_kv * H * D, align),
                    cute.assume(H * split_kv * S * D, align),
                ),
            )
            acc_o_iter = cute.recast_ptr(workspace.iterator, dtype=cutlass.BFloat16)
            acc_o = cute.make_tensor(acc_o_iter, acc_o_layout)
            acc_lse_layout = cute.make_layout(
                (H, split_kv, S, B),
                stride=(split_kv, 1, H * split_kv, H * split_kv * S),
            )
            acc_lse_iter = cute.recast_ptr(
                workspace.iterator
                + Int64(cute.cosize(acc_o_layout)) * Int64(cutlass.BFloat16.width // 8),
                dtype=self.lse_dtype,
            )
            acc_lse = cute.make_tensor(acc_lse_iter, acc_lse_layout)
        return acc_o, acc_lse

    @cute.kernel
    def reduction_kernel(
        self,
        output: cute.Tensor,
        lse: cute.Tensor,
        acc_output: cute.Tensor,
        acc_lse: cute.Tensor,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        cu_seqlens_q: cute.Tensor | None,
        block_split_kvs: cute.Tensor,
    ):
        """Dispatch the throughput 2CTA split-KV reduction body."""
        cfg = make_mla_decode_config(
            mma_qk_tiler_mn=self.mma_qk_tiler_mn,
            mma_pv_tiler_mn=self.mma_pv_tiler_mn,
            rope_dim=self.rope_dim,
            page_size=self.page_size,
            qkv_dtype=self.qkv_dtype,
            o_dtype=self.out_dtype,
            mask_type=self.mask_type,
        )
        if cutlass.const_expr(not self.is_persistent):
            prims.griddepcontrol(kind=prims.GridDepAction.WAIT)
        run_reduction_kernel(
            self,
            output,
            lse,
            acc_output,
            acc_lse,
            split_kv,
            cache_seqs,
            cu_seqlens_q,
            block_split_kvs,
            cfg,
            self.reduction_split_capacity,
            REDUCTION_ROWS_PER_CTA,
        )

    @cute.kernel
    def parallel_reduction_kernel(
        self,
        output: cute.Tensor,
        lse: cute.Tensor,
        acc_output: cute.Tensor,
        acc_lse: cute.Tensor,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        cu_seqlens_q: cute.Tensor | None,
        block_split_kvs: cute.Tensor,
    ):
        """Dispatch the high-split fixed-D512 cluster reducer."""

        cfg = make_mla_decode_config(
            mma_qk_tiler_mn=self.mma_qk_tiler_mn,
            mma_pv_tiler_mn=self.mma_pv_tiler_mn,
            rope_dim=self.rope_dim,
            page_size=self.page_size,
            qkv_dtype=self.qkv_dtype,
            o_dtype=self.out_dtype,
            mask_type=self.mask_type,
        )
        if cutlass.const_expr(not self.is_persistent):
            prims.griddepcontrol(kind=prims.GridDepAction.WAIT)
        topology = self.parallel_reduction_topology
        run_parallel_reduction_kernel(
            self.num_heads,
            self.seq_len_q,
            self.is_var_split_kv,
            output,
            lse,
            acc_output,
            acc_lse,
            split_kv,
            cache_seqs,
            cu_seqlens_q,
            block_split_kvs,
            cfg,
            topology.actual_splits,
            topology.cluster_size,
            topology.slots_per_rank,
        )
