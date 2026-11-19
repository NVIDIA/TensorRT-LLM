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
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Int32, Int64, cute, pipeline
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cute.testing import assert_ as runtime_assert
from cutlass.experimental import cuda
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.enums import SignalingThreads
from cutlass.experimental.task_scheduling.resources import MemoryResource, PipelineConfig

from .mla_decode_2cta_config import V_TMA_LATENT_ELEMENTS, MlaDecodeConfig, make_mla_decode_config
from .mla_decode_2cta_resources import (
    GmemOResource,
    MlaWorkQueue,
    PageOffsetWindowResource,
    SmemKResource,
    SmemPResource,
    SmemQResource,
    SmemVResource,
    TmemCorrResource,
    TmemOResource,
    TmemSResource,
)
from .mla_decode_2cta_scheduler import (
    MLAStaticTileScheduler,
    MLAStaticTileSchedulerParams,
    create_mla_static_tile_scheduler_params,
    divmod_constexpr_power_of_two_or_fdd,
)
from .mla_decode_2cta_tasks import (
    MlaTask,
    create_correction_task,
    create_load_k_task,
    create_load_v_task,
    create_mma_pv_direct_task,
    create_mma_qk_direct_task,
    create_padding_task,
    create_softmax_task,
    create_sparse_offsets_task,
)
from .mla_decode_helpers_gather import create_tensor_map_tiled_from_view, make_sparse_rope_map
from .mla_decode_helpers_math import (
    LOG2_E,
    MaskType,
    canonicalize_softmax_rescale_threshold_log2,
    normalize_mask_type,
    qkv_dtype,
)
from .mla_decode_helpers_schedule import FlatQueryTileLayout
from .mla_decode_helpers_schedule import MlaTaskManager as TaskManager


def build_mla_decode_task_manager(
    cfg: MlaDecodeConfig,
    # SMEM arrays (None for validate-only)
    smem_q_latent_arr=None,
    smem_q_rope_arr=None,
    smem_kc_arr=None,
    smem_vc_arr=None,
    smem_p_arr=None,
    sparse_offsets_smem=None,
    # TMA descriptors (None for validate-only)
    tma_desc_q_latent=None,
    tma_desc_q_rope=None,
    tma_desc_c_latent=None,
    tma_desc_c_rope=None,
    tma_desc_c_transpose=None,
    tma_desc_extra_v=None,
    tma_desc_sparse_rope=None,
    tma_desc_sparse_extra_rope=None,
    # Page-offset tensor
    primary_cache=None,
    extra_cache=None,
    page_offsets=None,
    # Runtime coordinates
    blk_coord=None,
    tidx=None,
    # GMEM output tensors
    output=None,
    attention_sinks=None,
    lse_in_natural_log=False,
    output_quant=None,
    output_quant_args=None,
    smem_o=None,
    tma_o_desc=None,
    lse=None,
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
    exhaustive_deadlock_race_check: bool | None = None,
) -> "tuple[TaskManager, list[MemoryResource], dict[str, MemoryResource]]":
    """Build the MLA decode TaskManager with all resources, tasks, and dependency graph.

    The throughput 2CTA graph uses Q/K/V TMA loads fed by a shared ring of
    sparse KV row IDs, QK/PV MMA, softmax, and correction/store tasks.
    Validation-only
    calls pass a concrete integer ``domain`` and no runtime tensors; JIT calls
    pass symbolic runtime state and retain structural checks as warnings.
    Exhaustive interleaving checks default to validation-only calls, whose
    concrete domains permit meaningful simulation; an explicit Boolean
    overrides that default. The dependency
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

    # Cooperative groups
    tma_producer_group = pipeline.CooperativeGroup(Agent.Thread)
    umma_hw_group = pipeline.CooperativeGroup(Agent.Thread)  # UMMA hardware

    # Softmax warps: 4 warps * 32 threads * 2 CTAs for cluster-scoped consumer
    compute_group_cluster = pipeline.CooperativeGroup(
        Agent.Thread, cfg.num_compute_warps * WARP_SIZE * cfg.num_mma_ctas
    )
    # Softmax warps: 4 warps * 32 threads (local CTA only)
    compute_group_local = pipeline.CooperativeGroup(Agent.Thread, cfg.num_compute_warps * WARP_SIZE)
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
        producer_signaling_threads=SignalingThreads.TaskWarpLeader
        if cfg.load_num_warps > 1
        else SignalingThreads.All,
        consumer_group=umma_hw_group,
        cta_layout_vmnk=cluster_shape_vmnk,
        consumer_signaling_threads=SignalingThreads.CtaLeader,
        num_bytes_per_warp_per_cta=(cfg.tma_copy_q_bytes // cfg.num_mma_ctas // cfg.load_num_warps),
    )

    if cfg.use_fp8_split_mma_schedule:
        # FP8 uses independent whole-tile K and V pipelines so QK and PV have
        # separate consumer states.  This mirrors the native FP8 schedule shape
        # without changing the BF16 combined-KV path.
        smem_k_pipeline_cfg = PipelineConfig.create_tma_umma_pipeline_cfg(
            num_stages=cfg.load_k_stage,
            num_bytes=cfg.tma_copy_k_tile_bytes,
            producer_group=tma_producer_group,
            producer_signaling_threads=SignalingThreads.TaskWarpLeader
            if cfg.load_num_warps > 1
            else SignalingThreads.All,
            consumer_group=umma_hw_group,
            cta_layout_vmnk=cluster_shape_vmnk,
            consumer_signaling_threads=SignalingThreads.CtaLeader,
            num_bytes_per_warp_per_cta=(
                cfg.tma_copy_k_tile_bytes // cfg.num_mma_ctas // cfg.load_num_warps
            ),
        )
        smem_v_pipeline_cfg = PipelineConfig.create_tma_umma_pipeline_cfg(
            num_stages=cfg.load_v_stage,
            num_bytes=cfg.tma_copy_v_tile_bytes,
            producer_group=tma_producer_group,
            producer_signaling_threads=SignalingThreads.TaskWarpLeader
            if cfg.load_num_warps > 1
            else SignalingThreads.All,
            consumer_group=umma_hw_group,
            cta_layout_vmnk=cluster_shape_vmnk,
            consumer_signaling_threads=SignalingThreads.CtaLeader,
            num_bytes_per_warp_per_cta=(
                cfg.tma_copy_v_tile_bytes // cfg.num_mma_ctas // cfg.load_num_warps
            ),
        )

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

    # ──────────────────────────────────────────────────────────────
    # Create resource instances
    # ──────────────────────────────────────────────────────────────

    sparse_offsets_enabled = cfg.page_size == 1 and cfg.use_fp8_split_mma_schedule
    shared_mask_enabled = (
        sparse_offsets_enabled
        and not cfg.use_fp8_dual_softmax_schedule
        and not getattr(page_offsets, "assume_valid_prefix", False)
    )
    # Prefix masks use live lengths only, so softmax does not consume the
    # offset ring and has no arrivals or task dependency on it; the K/V
    # loaders consume it with the same addressing and synchronization.
    offsets_cfg = None
    if sparse_offsets_enabled:
        offsets_cfg = PipelineConfig.create_async_async_pipeline_cfg(
            num_stages=cfg.sparse_offset_stages,
            producer_group=pipeline.CooperativeGroup(Agent.Thread, 32),
            consumer_group=pipeline.CooperativeGroup(
                Agent.Thread,
                (2 * cfg.load_num_warps + (4 if shared_mask_enabled else 0)) * 32,
            ),
            cta_layout_vmnk=cluster_shape_vmnk,
        )
    page_offset_window = PageOffsetWindowResource(
        primary_cache=primary_cache,
        extra_cache=extra_cache,
        page_offsets=page_offsets,
        smem_sparse_offsets=sparse_offsets_smem,
        cfg=cfg,
        pipeline_config=offsets_cfg,
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
            smem_sparse_offsets=sparse_offsets_smem,
            smem_k=smem_kc_arr,
            tma_desc_c_latent=tma_desc_c_latent,
            tma_desc_c_rope=tma_desc_c_rope,
            tma_desc_sparse_rope=tma_desc_sparse_rope,
            tma_desc_sparse_extra_rope=tma_desc_sparse_extra_rope,
            logical_seq_len_q=logical_seq_len_q,
            cfg=cfg,
            pipeline_config=smem_k_pipeline_cfg,
            name="smem_k",
        )
        smem_v = SmemVResource(
            smem_sparse_offsets=sparse_offsets_smem,
            smem_v=smem_vc_arr,
            tma_desc_c_transpose=tma_desc_c_transpose,
            tma_desc_extra_v=tma_desc_extra_v,
            tma_desc_sparse_rope=tma_desc_sparse_rope,
            tma_desc_sparse_extra_rope=tma_desc_sparse_extra_rope,
            logical_seq_len_q=logical_seq_len_q,
            cfg=cfg,
            pipeline_config=smem_v_pipeline_cfg,
            name="smem_v",
        )

    tmem_s = TmemSResource(
        page_offsets=page_offsets,
        smem_q_latent=smem_q_latent_arr,
        smem_q_rope=smem_q_rope_arr,
        smem_p=smem_p_arr,
        smem_exchange=None,  # Set at runtime
        scale_softmax_log2=None,  # Set at runtime
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

    output_pipeline = None
    gmem_o = GmemOResource(
        cfg=cfg,
        output=output,
        lse=lse,
        tmem_o_ref=tmem_o,
        tmem_corr_ref=tmem_corr,
        attention_sinks=attention_sinks,
        lse_in_natural_log=lse_in_natural_log,
        output_quant=output_quant,
        output_quant_args=output_quant_args,
        smem_o=smem_o,
        tma_o_desc=tma_o_desc,
        output_scale=None,  # set at runtime
        scale_softmax_log2=None,  # set at runtime
        smem_exchange=None,  # set at runtime
        cu_seqlens_q=cu_seqlens_q,
        logical_num_heads_q=logical_num_heads_q,
        logical_seq_len_q=logical_seq_len_q,
        name="gmem_o",
        pipeline_config=output_pipeline,
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
            page_offsets=page_offset_window if sparse_offsets_enabled else None,
            work_queue=work_queue,
            domain=domain,
            num_registers=wg2_reg_count,
        )
        load_v_task = create_load_v_task(
            smem_v,
            page_offsets=page_offset_window if sparse_offsets_enabled else None,
            work_queue=work_queue,
            domain=domain,
            num_registers=wg2_reg_count,
        )
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

    # Warpgroup 0 (warps 0-3): all 4 warps are in SoftmaxTask,
    # so setmaxnreg.inc 192 is safe (all warps participate).
    softmax_task = create_softmax_task(
        tmem_s,
        tmem_corr,
        smem_p,
        page_offsets=page_offset_window if shared_mask_enabled else None,
        work_queue=work_queue,
        task_class=MlaTask,
        domain=domain,
        domain_start=0,
        step=2 if cfg.use_fp8_dual_softmax_schedule else 1,
        num_registers=cfg.softmax_reg_num,
        softmax_group_id=0,
    )

    # Warpgroup 1 (warps 4-7): all 4 warps are in CorrectionTask,
    # so setmaxnreg.inc 208 is safe (all warps participate).
    correction_task = create_correction_task(
        tmem_corr,
        tmem_o,
        gmem_o,
        iterations_pv_n=cfg.iterations_pv_n,
        per_n_o_pipeline=cfg.use_fp8_split_mma_schedule,
        work_queue=work_queue,
        task_class=MlaTask,
        domain=domain,
        num_registers=cfg.correction_reg_num,
    )

    if cfg.use_fp8_split_mma_schedule:
        task_list = [load_k_task]
        if load_v_task is not None:
            task_list.append(load_v_task)
        task_list.extend([mma_qk_task, softmax_task])
        task_list.extend([mma_pv_task, correction_task])

    if sparse_offsets_enabled:
        task_list.append(
            create_sparse_offsets_task(
                page_offset_window,
                work_queue=work_queue,
                domain=domain,
                num_registers=wg2_reg_count,
            )
        )
    if cfg.use_fp8_split_mma_schedule and cfg.load_num_warps > 1:
        task_list.append(
            create_padding_task(
                work_queue=work_queue,
                domain=domain,
                num_registers=wg2_reg_count,
                warp_idx=10 if sparse_offsets_enabled else 9,
                num_warps=1 if sparse_offsets_enabled else 2,
            )
        )

    # ──────────────────────────────────────────────────────────────
    # Dependency graph
    # ──────────────────────────────────────────────────────────────
    if cfg.use_fp8_split_mma_schedule:
        resource_dependency_graph = {
            smem_q: [],  # Q loads (independent)
            smem_k: [],  # K loads take row IDs from the offset ring
            smem_v: [],  # V loads take row IDs from the offset ring
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
    if work_queue is not None:
        resource_dependency_graph[work_queue] = []

    if sparse_offsets_enabled:
        # K/V TMA reads register snapshots; metadata has no asynchronous SMEM read edge.
        resource_dependency_graph[page_offset_window] = (
            [work_queue] if work_queue is not None else []
        )

    # ──────────────────────────────────────────────────────────────
    # Create TaskManager
    # ──────────────────────────────────────────────────────────────
    skip = not isinstance(domain, int)
    # skip_validation only downgrades failures to warnings in the DSL. Avoid
    # repeating the expensive interleaving search for every JIT specialization
    # with a symbolic domain, which the checker approximates as one iteration.
    if exhaustive_deadlock_race_check is None:
        exhaustive_deadlock_race_check = not skip
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
        "smem_q": smem_q,
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
        mla(
            q_latent,
            q_rope,
            c_latent,
            c_rope,
            page_offsets,
            o,
            lse,
            split_kv,
            cache_seqs,
            cu_seqlens_q,
            block_split_kvs,
            softmax_scale,
            output_scale,
            stream,
        )
    """

    lse_base_on_e = False

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
        split_kv_capacity: int | None = None,
        static_seq_len_k=None,
        qkv_dtype="bf16",
        out_dtype="bf16",
        rope_dim=64,
        num_heads=128,
        seq_len_q=1,
        mask_type: MaskType | str = MaskType.CAUSAL,
        softmax_rescale_threshold_log2: float = 0.0,
        fixed_seqlen: bool = False,
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
            Compile-time split-KV count. Plans run one split (``1``): there is no
            split-KV reducer to merge partial outputs.
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
        mask_type : MaskType or str, optional
            ``causal`` (default) for bottom-right speculative decoding or
            ``dense`` for full per-batch KV visibility.
        softmax_rescale_threshold_log2 : float, optional
            Compile-time skip-correction threshold in log2 units. ``0.0``
            (default) keeps the exact online-softmax rescale.
        fixed_seqlen : bool, optional
            ``False`` (default) makes the query lengths runtime launch
            operands; ``True`` compiles for the exact static ``seq_len_q``.
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
            split_kv_capacity
            if split_kv_capacity is not None
            else static_split_kv
            if static_split_kv is not None
            else max_active_clusters * 2
        )
        self.static_seq_len_k = static_seq_len_k
        self.qkv_dtype = qkv_dtype
        self.out_dtype = out_dtype
        self.rope_dim = rope_dim
        self.num_heads = num_heads
        self.seq_len_q = seq_len_q
        self.fixed_seqlen = fixed_seqlen
        self.mask_type = normalize_mask_type(mask_type)
        # Validate eagerly: the config (and its own check) is only built when
        # the kernel is traced.
        self.softmax_rescale_threshold_log2 = canonicalize_softmax_rescale_threshold_log2(
            softmax_rescale_threshold_log2, qkv_dtype=qkv_dtype
        )
        self.fuse_sparse_epilogue = False
        self.output_quant = None
        # Sparse attention binds separate source lists; dense attention uses
        # the ordinary page table. Both reuse the staged KV loaders.
        self.direct_sparse = False
        self.sparse_gather_warps = 1
        self.sparse_kv_stages = 0
        self.sparse_uniform_pages = False
        self.sparse_balanced_registers = False
        self.assume_valid_prefix = False
        self.direct_sparse_capacities = (0, 0)
        self.query_tile_layout = FlatQueryTileLayout.for_tile(
            num_heads, seq_len_q, mma_qk_tiler_mn[0]
        )
        self.num_q_tiles = self.query_tile_layout.num_tiles
        self.tail_q_rows = self.query_tile_layout.tail_rows

    def compile_signature(self) -> tuple[object, ...]:
        """Return the complete batch-independent JIT identity."""

        return (
            self.acc_dtype,
            self.lse_dtype,
            self.lse_base_on_e,
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
            self.seq_len_q if self.fixed_seqlen else None,
            self.mask_type,
            self.softmax_rescale_threshold_log2,
            self.reduction_split_capacity,
            self.query_tile_layout if self.fixed_seqlen else None,
            self.num_q_tiles if self.fixed_seqlen else None,
            self.tail_q_rows if self.fixed_seqlen else None,
            self.fuse_sparse_epilogue,
            self.output_quant,
            self.fixed_seqlen,
            self.direct_sparse,
            self.assume_valid_prefix,
            tuple(int(x > 0) for x in self.direct_sparse_capacities),
            self.sparse_gather_warps,
            self.sparse_kv_stages,
            self.sparse_uniform_pages,
            self.sparse_balanced_registers,
        )

    def _make_config(self):
        """Share geometry and resource settings between launch and device code."""
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
            lse_base_on_e=self.lse_base_on_e,
            sparse_gather_warps=self.sparse_gather_warps,
            sparse_kv_stages=self.sparse_kv_stages,
            cache_uniform_sparse_pages=self.sparse_uniform_pages,
            balance_sparse_registers=self.sparse_balanced_registers,
            fuse_output_quant=self.output_quant is not None and self.fuse_sparse_epilogue,
            softmax_rescale_threshold_log2=self.softmax_rescale_threshold_log2,
        )

        # One flattened sparse query supplies a complete M128 tile. Its two
        # CTAs each stage 64 heads without row predicates. Warp 10 is otherwise
        # padding in the sparse FP8 schedule; using it for stores preserves
        # gather/MMA/softmax resources and overlaps output traffic with work.
        # Other geometries retain direct vector stores.
        cfg.use_tma_output = bool(
            self.output_quant is not None
            and self.fuse_sparse_epilogue
            and cfg.is_fp8_qkv()
            and cfg.page_size == 1
            and self.num_heads == 128
            and self.seq_len_q == 1
            and cfg.mma_pv_tiler_mn == (128, 256)
        )
        return cfg

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
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        cu_seqlens_q: cute.Tensor | None,
        block_split_kvs: object,
        softmax_scale: cutlass.Float32,
        output_scale: cutlass.Float32,
        stream: object,
        output_quant_args=None,
        max_q_len: cutlass.Int32 = 0,
    ):
        """Execute the MLA decode TS kernel."""
        seq_len_q = self.seq_len_q if self.fixed_seqlen else max_q_len
        if cutlass.const_expr(self.fuse_sparse_epilogue):
            # This CTA can finalize only an unsplit output. Split partials
            # must first merge their statistics; this is not a gather4 limit.
            pass
        if cutlass.const_expr(self.direct_sparse):
            pass
        cfg = self._make_config()
        num_query_tiles = (
            seq_len_q * self.num_heads + self.mma_qk_tiler_mn[0] - 1
        ) // self.mma_qk_tiler_mn[0]
        # Fixed public tensors retain [H,D,SQ,B]/[H,SQ,B]; variable-Q tensors
        # compact batches into [H,D,totalQ]/[H,totalQ]. The scheduler uses
        # physical flat-query tile coordinates, while resources map
        # each valid row back to its logical fixed or ragged storage location.
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
        if cutlass.const_expr(lse is not None):
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

        cute.make_tensor(
            c_latent.iterator,
            cute.select(c_latent.layout, mode=[1, 0, 2]),
        )
        if cutlass.const_expr(cfg.page_size == 1):
            # D512 has no separate Q/K RoPE maps. Reuse their ABI slots for
            # extra-K and extra-V descriptors, keeping dense launch ABI stable.
            primary_rows = cute.make_tensor(
                c_latent.iterator,
                cute.make_layout(
                    (c_latent.shape[1], c_latent.shape[2]),
                    stride=(c_latent.stride[1], c_latent.stride[2]),
                ),
            )
            extra_rows = cute.make_tensor(
                c_rope.iterator,
                cute.make_layout(
                    (c_rope.shape[1], c_rope.shape[2]),
                    stride=(c_rope.stride[1], c_rope.stride[2]),
                ),
            )
            tma_desc_c_latent = create_tensor_map_tiled_from_view(
                primary_rows,
                box_dims=(cfg.mma_qk_tiler[2], 1),
                stride_order=(0, 1),
                swizzle=cuda.TensorMapSwizzle.s128b,
                l2_promotion=cuda.TensorMapL2Promotion.l2_128b,
            )
            tma_desc_c_rope = create_tensor_map_tiled_from_view(
                extra_rows,
                box_dims=(cfg.mma_qk_tiler[2], 1),
                stride_order=(0, 1),
                swizzle=cuda.TensorMapSwizzle.s128b,
                l2_promotion=cuda.TensorMapL2Promotion.l2_128b,
            )
            v_swizzle = cuda.TensorMapSwizzle.s128b
            sparse_v_width = 128 if cfg.is_fp8_qkv() else V_TMA_LATENT_ELEMENTS
            tma_desc_c_transpose = create_tensor_map_tiled_from_view(
                primary_rows,
                box_dims=(sparse_v_width, 1),
                stride_order=(0, 1),
                swizzle=v_swizzle,
                l2_promotion=cuda.TensorMapL2Promotion.l2_128b,
            )
            tma_desc_extra_v = create_tensor_map_tiled_from_view(
                extra_rows,
                box_dims=(sparse_v_width, 1),
                stride_order=(0, 1),
                swizzle=v_swizzle,
                l2_promotion=cuda.TensorMapL2Promotion.l2_128b,
            )

        tma_desc_sparse_rope = tma_desc_c_latent
        tma_desc_sparse_extra_rope = tma_desc_c_rope
        if cutlass.const_expr(cfg.page_size == 1 and cfg.rope_dim):
            tma_desc_sparse_rope = make_sparse_rope_map(c_latent, cfg.qkv_dtype_bytes)
            tma_desc_sparse_extra_rope = make_sparse_rope_map(c_rope, cfg.qkv_dtype_bytes)

        tma_desc_o = tma_desc_q_latent

        scale_softmax_log2 = softmax_scale * LOG2_E

        kernel_split_kv = (
            Int32(self.static_split_kv)
            if cutlass.const_expr(self.static_split_kv is not None)
            else split_kv
        )

        runtime_assert(kernel_split_kv >= Int32(1), "split count must be positive")
        runtime_assert(
            kernel_split_kv <= Int32(self.reduction_split_capacity),
            "split count exceeds prepared capacity",
        )

        # Compute grid
        tile_sched_params = create_mla_static_tile_scheduler_params(
            self.is_persistent,
            batch_size,
            Int32(num_query_tiles),
            cfg.cluster_shape_mnk,
            kernel_split_kv,
        )
        clc_tile_sched_params = None
        launch_clusters = Int32(self.max_active_clusters)
        grid = MLAStaticTileScheduler.get_grid_shape(tile_sched_params, launch_clusters)

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
            kernel_split_kv,
            cache_seqs,
            cu_seqlens_q,
            block_split_kvs,
            scale_softmax_log2,
            output_scale,
            tile_sched_params,
            clc_tile_sched_params,
            output_quant_args,
            tma_desc_o,
            tma_desc_extra_v,
            tma_desc_sparse_rope,
            tma_desc_sparse_extra_rope,
            max_q_len,
        ).launch(
            grid=grid,
            block=[cfg.threads_per_cta, 1, 1],
            cluster=cfg.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=False,
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
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        cu_seqlens_q: cute.Tensor | None,
        block_split_kvs: object,
        scale_softmax_log2: cutlass.Float32,
        output_scale: cutlass.Float32,
        tile_sched_params: MLAStaticTileSchedulerParams,
        clc_tile_sched_params: object,
        output_quant_args=None,
        tma_desc_o: cutlass.GridConstant[cuda.TensorMap] = None,
        tma_desc_extra_v: cutlass.GridConstant[cuda.TensorMap] = None,
        tma_desc_sparse_rope: cutlass.GridConstant[cuda.TensorMap] = None,
        tma_desc_sparse_extra_rope: cutlass.GridConstant[cuda.TensorMap] = None,
        max_q_len: cutlass.Int32 = 0,
    ) -> None:
        """MLA decode TS kernel: persistent tile-scheduled execution."""
        seq_len_q = self.seq_len_q if self.fixed_seqlen else max_q_len
        if cutlass.const_expr(self.direct_sparse):
            from .mla_decode_helpers_gather import bind_sparse_kv_indices

            si, ci, sl, cl, sinks, ks = block_split_kvs
            capacity = ((Int32(si.shape[-1]) + 127) // 128) * 128
            capacity = cute.math.max(Int32(256), capacity)
            request = None
            if cutlass.const_expr(not self.is_persistent):
                _, row = divmod_constexpr_power_of_two_or_fdd(
                    cute.arch.block_idx()[1],
                    None,
                    tile_sched_params.problem_shape_b_fdd,
                )
                request = Int64(row)
            page_offsets, cache_seqs = bind_sparse_kv_indices(
                si,
                ci,
                sl,
                cl,
                capacity,
                query_row=request,
                assume_valid_prefix=self.assume_valid_prefix,
                has_extra=self.direct_sparse_capacities[1] > 0,
            )
        cfg = self._make_config()
        num_query_tiles = (
            seq_len_q * self.num_heads + self.mma_qk_tiler_mn[0] - 1
        ) // self.mma_qk_tiler_mn[0]
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
        # One shared metadata ring feeds both K/V issuers. Each holds its
        # stage through the last coordinate read; TMA consumes register
        # operands and its payload buffers have independent lifetimes.
        sparse_offsets_smem = None
        if cutlass.const_expr(cfg.page_size == 1 and cfg.use_fp8_split_mma_schedule):
            sparse_offsets_smem = cutlass.Array(
                Int32,
                cfg.sparse_offset_stages * cfg.tokens_per_k_tile,
                space=cutlass.AddressSpace.smem,
                alignment=128,
            )
        smem_o = None

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
        tmem_holding_buf_arr = cutlass.Array(Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)

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
        if cutlass.const_expr(self.is_persistent):
            current_work_linear_idx = cute.arch.block_idx()[0]
            current_work_cluster_batch = current_work_linear_idx // Int32(cfg.cluster_shape_mnk[0])
            cluster_idx = current_work_linear_idx % Int32(cfg.cluster_shape_mnk[0])
            current_work_after_seq_q, seq_q_idx = divmod_constexpr_power_of_two_or_fdd(
                current_work_cluster_batch,
                num_query_tiles,
                tile_sched_params.problem_shape_s_fdd,
            )
            _, batch_idx = divmod_constexpr_power_of_two_or_fdd(
                current_work_after_seq_q,
                None,
                tile_sched_params.problem_shape_b_fdd,
            )
            blk_coord = (cluster_idx, seq_q_idx, batch_idx, Int32(0))
        else:
            cluster_idx, seq_batch_idx, split_kv_idx = cute.arch.block_idx()
            seq_q_idx, batch_idx = divmod_constexpr_power_of_two_or_fdd(
                seq_batch_idx,
                None,
                tile_sched_params.problem_shape_b_fdd,
            )
            blk_coord = (cluster_idx, seq_q_idx, batch_idx, split_kv_idx)

        fixed_nonempty_single_split = (
            not self.is_var_seq
            and not self.is_var_split_kv
            and self.static_split_kv == 1
            and cu_seqlens_q is None
        )
        if cutlass.const_expr(fixed_nonempty_single_split):
            max_split_kv = Int32(1)
        elif cutlass.const_expr(not self.is_persistent):
            # Every task already derives its graph-live K/Q domain through the
            # work queue. Let a zero-domain task skip its data schedule instead
            # of recomputing the same metadata in an all-warp CTA prologue.
            max_split_kv = (
                Int32(self.static_split_kv)
                if cutlass.const_expr(self.static_split_kv is not None)
                else split_kv
            )

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
            logical_seq_len_q=seq_len_q,
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
            sparse_offsets_smem=sparse_offsets_smem,
            tma_desc_q_latent=tma_desc_q_latent.get_ptr(),
            tma_desc_q_rope=tma_desc_q_rope.get_ptr(),
            tma_desc_c_latent=tma_desc_c_latent.get_ptr(),
            tma_desc_c_rope=tma_desc_c_rope.get_ptr(),
            tma_desc_c_transpose=tma_desc_c_transpose.get_ptr(),
            tma_desc_extra_v=tma_desc_extra_v.get_ptr(),
            tma_desc_sparse_rope=tma_desc_sparse_rope.get_ptr(),
            tma_desc_sparse_extra_rope=tma_desc_sparse_extra_rope.get_ptr(),
            primary_cache=c_latent,
            extra_cache=c_rope,
            page_offsets=page_offsets,
            blk_coord=blk_coord,
            tidx=tidx,
            output=o,
            attention_sinks=sinks if self.fuse_sparse_epilogue else None,
            lse_in_natural_log=self.fuse_sparse_epilogue,
            output_quant=self.output_quant if self.fuse_sparse_epilogue else None,
            output_quant_args=output_quant_args,
            smem_o=smem_o,
            tma_o_desc=tma_desc_o.get_ptr(),
            lse=lse,
            domain=Int32(1),  # dummy; MlaTask recomputes per-task to avoid spills
            work_queue=mla_work_queue,
            cache_seqs=cache_seqs,
            cu_seqlens_q=cu_seqlens_q,
            split_kv=max_split_kv,
            logical_num_heads_q=self.num_heads,
            logical_seq_len_q=seq_len_q,
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
            prims.tcgen05_alloc(tmem_holding_buf_arr, cfg.num_tmem_cols, group="cta_2")
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
        if participates_in_tmem_sync:
            prims.barrier_cta_sync(cfg.tmem_sync_bar_id, thread_count=cfg.tmem_sync_bar_threads)

        # Only the warps synchronized above consume TMEM resources.  Do
        # not let the loader/page/padding warps race the allocation's
        # publication slot merely to materialize an unused pointer.
        tmem_base_addr = Int32(0)
        if participates_in_tmem_sync:
            tmem_base_addr = tmem_holding_buf_arr.load()

        # Set TMEM base address on resources
        named_res["tmem_s"].tmem_base_addr = tmem_base_addr
        named_res["smem_q"].tmem_base_addr = tmem_base_addr
        named_res["tmem_o"].tmem_base_addr = tmem_base_addr
        named_res["tmem_corr"].tmem_base_addr = tmem_base_addr

        # Set runtime params on TmemS (softmax)
        named_res["tmem_s"].scale_softmax_log2 = scale_softmax_log2
        named_res["tmem_corr"].page_offsets = page_offsets
        named_res["tmem_corr"].score_resource = named_res["tmem_s"]
        named_res["gmem_o"].page_offsets = page_offsets
        named_res["tmem_s"].smem_exchange = softmax_exchange_arr

        named_res["tmem_corr"].smem_exchange = epilogue_exchange_arr.data_ptr().toint(Int32)

        # Set runtime params on GmemO (epilogue)
        named_res["gmem_o"].output_scale = output_scale
        named_res["gmem_o"].scale_softmax_log2 = scale_softmax_log2
        named_res["gmem_o"].smem_exchange = epilogue_exchange_arr.data_ptr().toint(Int32)

        task_manager.run()

        # Pair teardown. The QK MMA warp (8) finishes its tasks before the PV MMA warp and the correction / epilogue
        # warps are done with the pair's TMEM, and its peer may exit at any time after its own dealloc: every thread
        # of both CTAs settles its TMEM accesses and meets on a cluster barrier, warp 8 deallocates the pair's TMEM
        # (cta_group::2), and every thread meets again, so neither CTA frees TMEM still in use nor exits while its
        # peer deallocates.
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        prims.barrier_cluster_arrive()
        prims.barrier_cluster_wait()
        if warp_idx == cfg.mma_warp_id:
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            tmem_arr_for_dealloc = prims.make_tmem_ptr(tmem_holding_buf_arr.load(), self.acc_dtype)
            prims.tcgen05_dealloc(
                tmem_arr_for_dealloc,
                cfg.num_tmem_cols,
                group="cta_2",
            )
        prims.barrier_cluster_arrive()
        prims.barrier_cluster_wait()
