# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch wrapper of the fused FC12 NVFP4 MoE kernel (SM107) for Kimi K3.

:func:`run_fc12` gathers, runs FC1 + SiTU + NVFP4 requant + FC2 and scatter-adds the weighted rows
into ``output`` in one persistent kernel (``Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel``,
the family's copy in :mod:`.fc12_kernel`), with the AutoTuner's MMA tactic for the shape. Beyond
the kernel's plain launch it takes:

* ``reserved_sms``: SMs left without an FC12 CTA, for the kernels another stream runs beside it;
* ``zeroed_counters``: the kernel's counters, zeroed together with ``output`` by the caller on a
  stream joined before the call, so neither the counter fill nor the output memset sits between
  the routing and the kernel;
* ``die_sched_min_tokens``: die-local tile claiming on die-alternating weights
  (:func:`..fc12.fc12_die_sm_words`).

Importing this module needs the CuTe DSL with Rubin support.
"""

from __future__ import annotations

import ast
from typing import List, Optional, Tuple

import cutlass
import cutlass.cute as cute
import torch

from tensorrt_llm._torch.autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from tensorrt_llm._torch.cute_dsl_utils import (
    IS_CUTLASS_DSL_RUBIN_AVAILABLE,
    reserved_max_active_clusters,
)
from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

if not IS_CUTLASS_DSL_RUBIN_AVAILABLE:
    raise ImportError("the Kimi K3 FC12 wrapper needs the CuTe DSL with Rubin support")

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
    Fc12FusedInputsHelper,
    GatherGroupedGemmInputsHelper,
    get_max_activate_clusters,
)
from tensorrt_llm._torch.kernels.blackwell.utils import make_ptr

from .fc12 import fc12_counter_layout, fc12_die_sm_words
from .fc12_kernel import Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel

#: AutoTuner key of the MMA tactic (the one the fused FC12 op tunes under).
TUNER_KEY = "trtllm::cute_dsl_nvfp4_fc12_fused_rubin"


class K3Fc12Runner(TunableRunner):
    """Rubin runner of the fused FC1+FC2 (FC12) NVFP4 MoE kernel for Kimi K3.

    The fused kernel replaces the two-op FC1 (gather+GEMM+gated-act+quant)
    and FC2 (GEMM+finalize) sequence with a single persistent kernel.
    The gated epilogue is SwiGLU or SiTU (trace-time specialization).
    The kernel takes three int32 atomic counters (fc1_ready /
    fc1_scheduler_counter / fc2_scheduler_counter), which are allocated and
    zeroed here on every launch unless the caller hands over
    ``zeroed_counters`` (see ``forward``).  Geometry follows the routing tile: tile_size=128
    runs 1-CTA (mma_tiler 128x{128,256}, cluster (1,1)) and
    tile_size=256 runs 2-CTA (mma_tiler 256x{128,256}, cluster (2,1));
    the autotuner picks between them per shape.  scheduler="l2_atomic".
    ``reserved_sms`` SMs are left without an FC12 CTA (whole clusters fewer,
    at least one launched): one CTA fills an SM's register file, so kernels
    another stream issues while FC12 runs otherwise get SMs only as its CTAs
    exit. FC12 hands tiles out through atomic counters, so fewer CTAs change
    only which CTA computes a tile.
    ``die_sched_min_tokens``: die-local tile claiming (see
    :func:`fc12_die_sm_words`) for calls of at least that many tokens on
    die-alternating weights; ``None``: never.
    """

    kernel_class = Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel
    kernel_cache = dict()
    tuning_config_cache = dict()

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int,
        tile_size: int,
        scaling_vector_size: int = 16,
        swiglu_limit: float = float("inf"),
        ep_size: int = 1,
        enable_alltoall: bool = False,
        activation_type: ActivationType = ActivationType.Swiglu,
        situ_beta: Optional[float] = None,
        situ_linear_beta: Optional[float] = None,
        reserved_sms: int = 0,
        die_sched_min_tokens: Optional[int] = None,
        zeroed_counters: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.local_expert_offset = local_expert_offset
        self.tile_size = tile_size
        self.scaling_vector_size = scaling_vector_size
        self.swiglu_limit = swiglu_limit
        self.activation_type = ActivationType(int(activation_type))
        self.situ_beta = situ_beta
        self.situ_linear_beta = situ_linear_beta
        if reserved_sms < 0:
            raise ValueError(f"reserved_sms must be non-negative, got {reserved_sms}")
        self.reserved_sms = reserved_sms
        self.die_sched_min_tokens = die_sched_min_tokens
        # int32 words the caller zeroed, together with the output, for
        # one launch (``None``: zeroed here). Not part of the tactic.
        self.zeroed_counters = zeroed_counters
        # Used only by the in-op output memset (moved here so the memset
        # is the fused kernel's immediate stream predecessor).
        self.ep_size = ep_size
        self.enable_alltoall = enable_alltoall
        if (sm_version := get_sm_version()) != 107:
            raise ValueError(
                f"{self.__class__.kernel_class.__name__} supports SM 107 "
                f"(Rubin) only, but got SM {sm_version}"
            )
        # Routing tiles: 128 (1-CTA) or 256 (2-CTA, cluster (2,1)).
        if self.tile_size not in (128, 256):
            raise ValueError(
                f"{self.__class__.kernel_class.__name__} supports "
                f"tile_size 128 (1-CTA) or 256 (2-CTA) only, but got "
                f"{self.tile_size}"
            )

    def unique_id(self):
        return (
            self.num_experts,
            self.top_k,
            self.num_local_experts,
            self.local_expert_offset,
            self.tile_size,
            self.scaling_vector_size,
            self.swiglu_limit,
            int(self.activation_type),
            self.situ_beta,
            self.situ_linear_beta,
            self.reserved_sms,
        )

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
        **kwargs,
    ) -> List[Tuple]:
        (
            fc1_a,
            fc1_b,
            fc1_sfa,
            fc1_sfb,
            fc1_alpha,
            tile_idx_to_group_idx,
            tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx,
            num_non_exiting_tiles,
            fc1_norm_const,
            fc2_b,
            *_,
        ) = inputs
        m = permuted_idx_to_expanded_idx.size(0)
        k = fc1_a.size(1) * 2
        l, fc1_n = fc1_b.size(0), fc1_b.size(1)  # noqa: E741
        fc2_n, fc2_k = fc2_b.size(1), fc2_b.size(2) * 2

        # Fixed K for FP4: mma_tiler_k=256, mma_inst_k=128.
        mma_tiler_k = 256
        mma_inst_k = 128
        # Mirror the CuteDSL grouped-GEMM runners: the MMA M-tile equals
        # the routing tile_size and the cluster M = tile_size // 128, so
        # tile_size=128 -> 1-CTA cluster (1,1), tile_size=256 -> 2-CTA
        # cluster (2,1). moe_sort tiles the tokens by tile_size, so the
        # kernel's M-tile always matches the routing tile. mma_n is free
        # {128, 256}.
        mma_tiler_m = self.tile_size
        mma_inst_m = self.tile_size
        cluster_shape_mn = (self.tile_size // 128, 1)
        mma_n_candidates = [128, 256]

        valid_tactics = []
        for mma_n in mma_n_candidates:
            # No "cluster CTAs > tiles" guard here (unlike the Sm100
            # runners): this kernel's per-CTA M-tile is 128, so one
            # 256-row logical tile is exactly one (2,1) cluster and a
            # single-tile problem is valid for 2-CTA.
            # The fused N-tile must divide both FC1 and FC2 output N.
            if fc1_n % mma_n != 0 or fc2_n % mma_n != 0:
                continue

            mma_tiler = (mma_tiler_m, mma_n, mma_tiler_k)
            mma_inst_shape = (mma_inst_m, mma_n, mma_inst_k)

            if self.__class__.kernel_class.can_implement(
                a_dtype=cutlass.Float4E2M1FN,
                b_dtype=cutlass.Float4E2M1FN,
                sf_dtype=cutlass.Float8E4M3FN,
                sf_vec_size=self.scaling_vector_size,
                fc1_c_dtype=cutlass.Float4E2M1FN,
                fc2_c_dtype=cutlass.BFloat16,
                mma_inst_shape=mma_inst_shape,
                mma_tiler=mma_tiler,
                cluster_shape_mn=cluster_shape_mn,
                fc1_gemm_shape=(m, fc1_n, k, l),
                fc2_gemm_shape=(m, fc2_n, fc2_k, l),
                a_major="k",
                b_major="k",
                fc1_c_major="n",
                fc2_c_major="n",
            ):
                valid_tactics.append((mma_tiler, mma_inst_shape, cluster_shape_mn))

        logger.debug(
            f"CuteDSL Rubin FusedFC12: Found {len(valid_tactics)} valid "
            f"tactics for M={m}, FC1_N={fc1_n}, FC2_N={fc2_n}, K={k}, "
            f"L={l}"
        )
        return valid_tactics

    def get_tuning_config(self) -> TuningConfig:
        key = self.unique_id()
        if key not in self.__class__.tuning_config_cache:
            helper = Fc12FusedInputsHelper(
                self.num_experts,
                self.top_k,
                self.num_local_experts,
                self.local_expert_offset,
                self.tile_size,
            )
            self.__class__.tuning_config_cache[key] = TuningConfig(
                dynamic_tensor_specs=(
                    DynamicTensorSpec(
                        GatherGroupedGemmInputsHelper.IDX_SHAPE_INFER,
                        0,
                        helper.gen_tuning_buckets,
                        helper.map_to_tuning_buckets,
                    ),
                ),
                constraint_specs=(
                    ConstraintSpec(0, 0, helper.infer_shape_num_tokens),
                    ConstraintSpec(2, 0, helper.infer_shape_num_tokens),
                    ConstraintSpec(5, 0, helper.infer_shape_max_num_tiles),
                    ConstraintSpec(6, 0, helper.infer_shape_max_num_tiles),
                    # fc2_c(13)/fc2_routing_scales(14) dim0=num_tokens
                    # follow the dynamic M bucket (avoid per-num_tokens
                    # redundant tune + context-shape fallback).
                    ConstraintSpec(13, 0, helper.infer_shape_num_tokens),
                    ConstraintSpec(14, 0, helper.infer_shape_num_tokens),
                    # expanded_idx_to_permuted_idx(15) dim0=num_tokens too
                    ConstraintSpec(15, 0, helper.infer_shape_num_tokens),
                ),
                inputs_pre_hook=helper.inputs_pre_hook,
            )
        return self.__class__.tuning_config_cache[key]

    def forward(
        self, inputs: List[torch.Tensor], tactic: Optional[tuple], **kwargs
    ) -> torch.Tensor:
        (
            fc1_a,
            fc1_b,
            fc1_sfa,
            fc1_sfb,
            fc1_alpha,
            tile_idx_to_group_idx,
            tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx,
            num_non_exiting_tiles,
            fc1_norm_const,
            fc2_b,
            fc2_sfb,
            fc2_alpha,
            fc2_c,
            fc2_routing_scales,
            expanded_idx_to_permuted_idx,
        ) = inputs

        assert fc1_a.dtype == torch.float4_e2m1fn_x2
        assert fc1_b.dtype == torch.float4_e2m1fn_x2
        assert fc2_b.dtype == torch.float4_e2m1fn_x2
        assert fc1_sfa.dtype == torch.uint8
        assert fc1_sfb.dtype == torch.uint8
        assert fc2_sfb.dtype == torch.uint8
        assert fc1_alpha.dtype == torch.float32
        assert fc2_alpha.dtype == torch.float32
        assert fc1_norm_const.dtype == torch.float32
        assert fc2_c.dtype == torch.bfloat16

        sf_vec = self.scaling_vector_size
        orig_m, k = fc1_a.size(0), fc1_a.size(1) * 2
        m = permuted_idx_to_expanded_idx.size(0)
        l, fc1_n = fc1_b.size(0), fc1_b.size(1)  # noqa: E741
        interm_size = fc1_n // 2  # SwiGLU (gated) halves N
        fc2_n, fc2_k = fc2_b.size(1), fc2_b.size(2) * 2
        num_tokens = fc2_c.size(0)
        num_tiles = m // self.tile_size
        assert m % self.tile_size == 0
        assert fc2_k == interm_size, f"FC2 K ({fc2_k}) must equal FC1 intermediate ({interm_size})"

        if isinstance(tactic, tuple):
            mma_tiler, mma_inst_shape, cluster_shape_mn = tactic
        else:
            # Fallback geometry must satisfy the kernel validator
            # (is_valid_mma_tiler_and_cluster_shape): inst_m ==
            # tile_m and cluster_m == inst_m // 128. Mirrors
            # _get_sm107_nvfp4_default_mma_config used by the sibling
            # runners; the old fixed cluster (1,1) was illegal for
            # tile_size=256 (a 2-CTA MMA in a 1-CTA cluster).
            mma_inst_m = min(self.tile_size, 256)
            mma_tiler = (self.tile_size, 128, 256)
            mma_inst_shape = (mma_inst_m, 128, 128)
            cluster_shape_mn = (mma_inst_m // 128, 1)
        # The MMA M-tile and cluster M must match the routing tile this
        # runner was built for; a mismatch (e.g. a tactic captured under
        # another tile_size) would index routing metadata wrongly and
        # can read uninitialised permuted-index padding.
        assert mma_tiler[0] == self.tile_size and cluster_shape_mn[0] == self.tile_size // 128, (
            f"FC12 tactic/tile mismatch: mma_tiler={mma_tiler} "
            f"mma_inst_shape={mma_inst_shape} "
            f"cluster_shape_mn={cluster_shape_mn} "
            f"tile_size={self.tile_size}"
        )

        # FC1 intermediate output (kept on-chip by the kernel; passed as
        # a scratch tensor) + its dynamic block scale.
        fc1_c = torch.empty(m, interm_size // 2, dtype=fc1_a.dtype, device=fc1_a.device)
        fc1_sfc = torch.empty(m * interm_size // sf_vec, dtype=fc1_sfa.dtype, device=fc1_sfa.device)
        # Three atomic counters, zero at launch (fc1_ready, then each
        # scheduler counter on its own 128 B line). The caller's
        # ``zeroed_counters`` come zeroed together with ``fc2_c`` (on a
        # stream the caller joined before this call) and serve one launch:
        # never while tuning, whose profiling launches reuse their
        # inputs. Otherwise one fill zeroes the counters here.
        # Die-local claiming: one counter per die and phase.
        die_sm_words = fc12_die_sm_words(
            fc1_b,
            fc2_b,
            cluster_shape_mn[0] * cluster_shape_mn[1],
            num_tokens,
            self.die_sched_min_tokens,
        )
        die_sched = die_sm_words is not None
        fc1_at, fc2_at, counter_words = fc12_counter_layout(num_tiles, die_sched=die_sched)
        # Words of a phase's scheduler counters: die 1's is word 32.
        sched_words = 33 if die_sched else 1
        tuning = AutoTuner.get().is_tuning_mode
        prezeroed = (
            self.zeroed_counters is not None
            and not tuning
            and self.zeroed_counters.numel() >= counter_words
        )
        if prezeroed:
            counters = self.zeroed_counters
        else:
            counters = torch.zeros(counter_words, dtype=torch.int32, device=fc1_a.device)
        fc1_ready = counters[:num_tiles]
        fc1_scheduler_counter = counters[fc1_at : fc1_at + sched_words]
        fc2_scheduler_counter = counters[fc2_at : fc2_at + sched_words]

        # Zero the scatter-add output on this stream right before the
        # fused kernel, unless it came zeroed with ``zeroed_counters``.
        # The zeroing must run on every other launch: autotune() keeps
        # is_tuning_mode set for its whole context, including the final
        # real launch, so gating the memset on it would leave real
        # outputs unzeroed. While tuning, the profiling inputs carry an
        # uninitialised expanded_idx (see Fc12FusedInputsHelper
        # .inputs_pre_hook), so use an index-free full zero there instead
        # of the sparse, index-driven memset; it is semantically equivalent
        # (a superset of the rows the finalize scatter-adds into) and still
        # charges a memset to the profiled time.
        if tuning:
            fc2_c.zero_()
        elif not prezeroed:
            torch.ops.trtllm.moe_output_memset_inplace(
                input=fc2_c,
                tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
                num_non_exiting_tiles=num_non_exiting_tiles,
                tile_tokens_dim=self.tile_size,
                top_k=self.top_k,
                ep_size=self.ep_size,
                enable_alltoall=self.enable_alltoall,
            )

        fc1_a_ptr = make_ptr(
            cutlass.Float4E2M1FN, fc1_a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
        )
        fc1_b_ptr = make_ptr(
            cutlass.Float4E2M1FN, fc1_b.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
        )
        fc1_c_ptr = make_ptr(
            cutlass.Float4E2M1FN, fc1_c.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
        )
        fc1_sfa_ptr = make_ptr(
            cutlass.Float8E4M3FN, fc1_sfa.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        )
        fc1_sfb_ptr = make_ptr(
            cutlass.Float8E4M3FN, fc1_sfb.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        )
        fc1_sfc_ptr = make_ptr(
            cutlass.Float8E4M3FN, fc1_sfc.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        )
        fc1_norm_const_ptr = make_ptr(
            cutlass.Float32, fc1_norm_const.data_ptr(), cute.AddressSpace.gmem
        )
        fc1_alpha_ptr = make_ptr(cutlass.Float32, fc1_alpha.data_ptr(), cute.AddressSpace.gmem)
        fc2_alpha_ptr = make_ptr(cutlass.Float32, fc2_alpha.data_ptr(), cute.AddressSpace.gmem)
        tile_idx_to_group_idx_ptr = make_ptr(
            cutlass.Int32, tile_idx_to_group_idx.data_ptr(), cute.AddressSpace.gmem
        )
        tile_idx_to_mn_limit_ptr = make_ptr(
            cutlass.Int32, tile_idx_to_mn_limit.data_ptr(), cute.AddressSpace.gmem
        )
        permuted_idx_to_expanded_idx_ptr = make_ptr(
            cutlass.Int32, permuted_idx_to_expanded_idx.data_ptr(), cute.AddressSpace.gmem
        )
        num_non_exiting_tiles_ptr = make_ptr(
            cutlass.Int32, num_non_exiting_tiles.data_ptr(), cute.AddressSpace.gmem
        )
        fc1_ready_ptr = make_ptr(cutlass.Int32, fc1_ready.data_ptr(), cute.AddressSpace.gmem)
        fc1_scheduler_counter_ptr = make_ptr(
            cutlass.Int32, fc1_scheduler_counter.data_ptr(), cute.AddressSpace.gmem
        )
        fc2_scheduler_counter_ptr = make_ptr(
            cutlass.Int32, fc2_scheduler_counter.data_ptr(), cute.AddressSpace.gmem
        )
        fc2_b_ptr = make_ptr(
            cutlass.Float4E2M1FN, fc2_b.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
        )
        fc2_sfb_ptr = make_ptr(
            cutlass.Float8E4M3FN, fc2_sfb.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        )
        fc2_c_ptr = make_ptr(
            cutlass.BFloat16, fc2_c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        )
        fc2_routing_scales_ptr = make_ptr(
            cutlass.Float32, fc2_routing_scales.data_ptr(), cute.AddressSpace.gmem
        )

        torch_stream = torch.cuda.current_stream()
        stream = cuda.CUstream(torch_stream.cuda_stream)
        # Cached occupancy lookup (same helper as the other Rubin
        # runners): avoids re-querying HardwareInfo on every forward
        # and during CUDA-graph capture. FC12 never runs inside a
        # locality-domain context, so this is the full-device value.
        cluster_size = cluster_shape_mn[0] * cluster_shape_mn[1]
        max_active_clusters = reserved_max_active_clusters(
            get_max_activate_clusters(cluster_size), cluster_size, self.reserved_sms
        )

        # The fused kernel exposes only ``__call__`` (cute.Tensor args),
        # no ptr-based wrapper. ``make_ordered_layout`` needs an MLIR
        # context, so build the cute tensors INSIDE a @cute.jit wrapper
        # (like the standalone FC1/FC2 kernels' ``wrapper`` methods do
        # internally) that captures ``gemm`` and takes ptrs + Int64
        # shapes. Shapes are dynamic so one compile serves all shapes.
        sf_vec_cx = self.scaling_vector_size
        tile_size_cx = self.tile_size
        top_k_cx = self.top_k
        sched_words_cx = sched_words
        cache_key = (
            self.scaling_vector_size,
            self.tile_size,
            self.top_k,
            mma_tiler,
            mma_inst_shape,
            cluster_shape_mn,
            max_active_clusters,
            self.swiglu_limit,
            int(self.activation_type),
            self.situ_beta,
            self.situ_linear_beta,
        )
        if die_sched:
            cache_key += (die_sm_words,)
        if cache_key not in self.__class__.kernel_cache:
            gemm = self.__class__.kernel_class(
                self.scaling_vector_size,
                mma_inst_shape,
                mma_tiler,
                cluster_shape_mn,
                True,  # vectorized_f32
                topk=self.top_k,
                use_pdl=True,
                swiglu_limit=self.swiglu_limit,
                scheduler="l2_atomic",
                activation_type=self.activation_type,
                situ_beta=self.situ_beta,
                situ_linear_beta=self.situ_linear_beta,
                die_sm_words=die_sm_words,
            )

            @cute.jit
            def _fc12_wrapper(
                a_ptr,
                b_ptr,
                c_ptr,
                sfa_ptr,
                sfb_ptr,
                sfc_ptr,
                norm_const_ptr,
                tile_grp_ptr,
                tile_mn_ptr,
                num_non_exiting_ptr,
                fc1_alpha_ptr,
                fc2_alpha_ptr,
                ready_ptr,
                fc1_sched_ptr,
                fc2_sched_ptr,
                w2_ptr,
                out_ptr,
                w2_sf_ptr,
                permuted_ptr,
                routing_ptr,
                orig_m: cutlass.Int64,
                m: cutlass.Int64,
                fc1_n: cutlass.Int64,
                k: cutlass.Int64,
                l: cutlass.Int64,  # noqa: E741
                fc2_n: cutlass.Int64,
                fc2_k: cutlass.Int64,
                num_tokens: cutlass.Int64,
                max_active_clusters: cutlass.Constexpr,
                stream: cuda.CUstream,
            ):
                interm_size = fc1_n // 2  # SwiGLU (gated) halves N
                scale_k = k // sf_vec_cx
                fc2_scale_k = fc2_k // sf_vec_cx
                num_tiles = m // tile_size_cx
                a = cute.make_tensor(
                    a_ptr, layout=cute.make_ordered_layout((orig_m, k, 1), order=(1, 0, 2))
                )
                b = cute.make_tensor(
                    b_ptr, layout=cute.make_ordered_layout((fc1_n, k, l), order=(1, 0, 2))
                )
                c = cute.make_tensor(
                    c_ptr,
                    layout=cute.make_layout(
                        (m, interm_size, 1), stride=(interm_size, 1, m * interm_size)
                    ),
                )
                sfa = cute.make_tensor(
                    sfa_ptr, layout=cute.make_ordered_layout((orig_m, scale_k, 1), order=(1, 0, 2))
                )
                sfb = cute.make_tensor(
                    sfb_ptr,
                    layout=cute.make_ordered_layout(
                        (32, 4, fc1_n // 128, 4, scale_k // 4, l), order=(2, 1, 4, 0, 3, 5)
                    ),
                )
                sfc = cute.make_tensor(
                    sfc_ptr,
                    layout=cute.make_ordered_layout(
                        (32, 4, m // 128, 4, interm_size // (sf_vec_cx * 4), l),
                        order=(2, 1, 4, 0, 3, 5),
                    ),
                )
                norm_const = cute.make_tensor(norm_const_ptr, layout=cute.make_layout((1,)))
                fc1_alpha = cute.make_tensor(fc1_alpha_ptr, layout=cute.make_layout((l,)))
                fc2_alpha = cute.make_tensor(fc2_alpha_ptr, layout=cute.make_layout((l,)))
                tile_grp = cute.make_tensor(tile_grp_ptr, layout=cute.make_layout((num_tiles,)))
                tile_mn = cute.make_tensor(tile_mn_ptr, layout=cute.make_layout((num_tiles,)))
                num_non_exiting = cute.make_tensor(
                    num_non_exiting_ptr, layout=cute.make_layout((1,))
                )
                ready = cute.make_tensor(ready_ptr, layout=cute.make_layout((num_tiles,)))
                fc1_sched = cute.make_tensor(
                    fc1_sched_ptr, layout=cute.make_layout((sched_words_cx,))
                )
                fc2_sched = cute.make_tensor(
                    fc2_sched_ptr, layout=cute.make_layout((sched_words_cx,))
                )
                w2 = cute.make_tensor(
                    w2_ptr, layout=cute.make_ordered_layout((fc2_n, fc2_k, l), order=(1, 0, 2))
                )
                w2_sf = cute.make_tensor(
                    w2_sf_ptr,
                    layout=cute.make_ordered_layout(
                        (32, 4, fc2_n // 128, 4, fc2_scale_k // 4, l), order=(2, 1, 4, 0, 3, 5)
                    ),
                )
                out = cute.make_tensor(
                    out_ptr,
                    layout=cute.make_layout(
                        (num_tokens, fc2_n, 1), stride=(fc2_n, 1, num_tokens * fc2_n)
                    ),
                )
                permuted = cute.make_tensor(permuted_ptr, layout=cute.make_layout((m,)))
                routing = cute.make_tensor(
                    routing_ptr,
                    layout=cute.make_ordered_layout((num_tokens, top_k_cx), order=(1, 0)),
                )
                gemm(
                    a,
                    b,
                    c,
                    sfa,
                    sfb,
                    sfc,
                    norm_const,
                    tile_grp,
                    tile_mn,
                    num_non_exiting,
                    fc1_alpha,
                    fc2_alpha,
                    ready,
                    fc1_sched,
                    fc2_sched,
                    w2,
                    out,
                    w2_sf,
                    permuted,
                    routing,
                    max_active_clusters=max_active_clusters,
                    stream=stream,
                )

            compiled_gemm = cute.compile(
                _fc12_wrapper,
                fc1_a_ptr,
                fc1_b_ptr,
                fc1_c_ptr,
                fc1_sfa_ptr,
                fc1_sfb_ptr,
                fc1_sfc_ptr,
                fc1_norm_const_ptr,
                tile_idx_to_group_idx_ptr,
                tile_idx_to_mn_limit_ptr,
                num_non_exiting_tiles_ptr,
                fc1_alpha_ptr,
                fc2_alpha_ptr,
                fc1_ready_ptr,
                fc1_scheduler_counter_ptr,
                fc2_scheduler_counter_ptr,
                fc2_b_ptr,
                fc2_c_ptr,
                fc2_sfb_ptr,
                permuted_idx_to_expanded_idx_ptr,
                fc2_routing_scales_ptr,
                orig_m,
                m,
                fc1_n,
                k,
                l,
                fc2_n,
                fc2_k,
                num_tokens,
                max_active_clusters=max_active_clusters,
                stream=stream,
            )
            self.__class__.kernel_cache[cache_key] = compiled_gemm
        else:
            compiled_gemm = self.__class__.kernel_cache[cache_key]

        compiled_gemm(
            fc1_a_ptr,
            fc1_b_ptr,
            fc1_c_ptr,
            fc1_sfa_ptr,
            fc1_sfb_ptr,
            fc1_sfc_ptr,
            fc1_norm_const_ptr,
            tile_idx_to_group_idx_ptr,
            tile_idx_to_mn_limit_ptr,
            num_non_exiting_tiles_ptr,
            fc1_alpha_ptr,
            fc2_alpha_ptr,
            fc1_ready_ptr,
            fc1_scheduler_counter_ptr,
            fc2_scheduler_counter_ptr,
            fc2_b_ptr,
            fc2_c_ptr,
            fc2_sfb_ptr,
            permuted_idx_to_expanded_idx_ptr,
            fc2_routing_scales_ptr,
            orig_m,
            m,
            fc1_n,
            k,
            l,
            fc2_n,
            fc2_k,
            num_tokens,
            stream=stream,
        )
        return fc2_c


def _soft_cap(beta: Optional[float]) -> Optional[float]:
    """A SiTU soft-cap as the kernel takes it: ``None`` when absent or non-positive."""
    return None if beta is None or beta <= 0 else float(beta)


def run_fc12(
    input: torch.Tensor,
    fc1_weight: torch.Tensor,
    input_scale: torch.Tensor,
    fc1_weight_scale: torch.Tensor,
    fc1_alpha: torch.Tensor,
    tile_idx_to_group_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    global_sf: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_weight_scale: torch.Tensor,
    fc2_alpha: torch.Tensor,
    output: torch.Tensor,
    token_final_scales: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
    num_local_experts: int,
    local_expert_offset: int,
    tile_size: int,
    ep_size: int,
    situ_beta: Optional[float],
    situ_linear_beta: Optional[float],
    reserved_sms: int = 0,
    die_sched_min_tokens: Optional[int] = None,
    zeroed_counters: Optional[torch.Tensor] = None,
    precomputed_tactic: Optional[str] = None,
) -> None:
    """Scatter-add the routed NVFP4 SiTU experts' weighted rows into ``output`` (BF16
    ``[num_tokens, hidden]``; zeroed here unless it came zeroed with ``zeroed_counters``).

    ``input`` / ``input_scale``: E2M1 rows and linear E4M3 block scales; the weights and block
    scales in the layout of the grouped-GEMM NVFP4 kernels (FC1 rows in the SM107 gate16/up16
    order); ``fc1_alpha`` / ``fc2_alpha`` per expert, ``global_sf`` the FC2 input global scale;
    the routing tables ``moe_sort``'s at ``tile_size`` (128 or 256).
    """
    runner = K3Fc12Runner(
        num_experts,
        top_k,
        num_local_experts,
        local_expert_offset,
        tile_size,
        16,
        swiglu_limit=float("inf"),
        ep_size=ep_size,
        enable_alltoall=False,
        activation_type=ActivationType.SiTu,
        situ_beta=_soft_cap(situ_beta),
        situ_linear_beta=_soft_cap(situ_linear_beta),
        reserved_sms=reserved_sms,
        die_sched_min_tokens=die_sched_min_tokens,
        zeroed_counters=zeroed_counters,
    )
    # Input order of Fc12FusedInputsHelper: the FC1 gather prefix 0..9, the FC2 tensors 10..14,
    # expanded_idx 15 for the output memset.
    inputs = [
        input,
        fc1_weight,
        input_scale,
        fc1_weight_scale,
        fc1_alpha,
        tile_idx_to_group_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
        global_sf,
        fc2_weight,
        fc2_weight_scale,
        fc2_alpha,
        output,
        token_final_scales,
        expanded_idx_to_permuted_idx,
    ]
    if precomputed_tactic is None:
        _, tactic = AutoTuner.get().choose_one(
            TUNER_KEY, [runner], runner.get_tuning_config(), inputs
        )
    else:
        tactic = ast.literal_eval(precomputed_tactic)
    runner(inputs, tactic=tactic)
