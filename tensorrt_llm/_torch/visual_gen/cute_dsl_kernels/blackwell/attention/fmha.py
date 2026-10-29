# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# ruff: noqa: I001, E501, F841

import math
from dataclasses import replace
from typing import Callable, Type, Tuple, Union, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda_exp
from cutlass import testing
from cutlass.cute.nvgpu import tcgen05
import cutlass.cute.nvgpu.cpasync as cpasync
from cutlass.cute.nvgpu.common import OperandMajorMode
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.typing import Int8, Int32, Int64, Float16, Float32
from cutlass.base_dsl.arch import Arch
from cutlass.cutlass_dsl import BaseDSL

from ...helpers import fmha_helpers as fmha_utils

"""
A fused multi-head attention (FMHA) kernel for the NVIDIA Blackwell SM100 architecture using CUTE DSL.

A TMA + Blackwell SM100 TensorCore warp-specialized persistent kernel that fuses Q*K^T, softmax
normalization and softmax(Q*K^T)*V, avoiding intermediate data movement through global memory.

Features include warp specialization (load, MMA, softmax, correction, epilogue), pipelined stages
between warps, causal/window/residual masking, and an optional paged KV cache: K/V are gathered
from fixed-size physical pages through a page table.

Constraints:
* Supported head dimensions: 32, 64, 128 (and 192/128 for MLA)
* Number of heads in Q must be divisible by number of heads in K
* mma_tiler_mn must be 128,128
* Paged KV requires ``page_size >= 8`` dividing the 128-token KV tile
"""


def make_thread_cooperative_group(size: int):
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, size)


_PAGED_KV_LAYOUTS = frozenset({"separate_nhd", "combined_hnd"})


class BlackwellFusedMultiHeadAttentionForward:
    arch_str: str = "sm_100"
    arch_name: str = "Blackwell SM100"
    _supported_head_dims: frozenset[int] = frozenset({32, 64, 128, 192})
    # Whether packed variable-length output may use TMA descriptors; the base
    # kernel's dense TMA output store would overwrite packed ragged tails, so
    # only frontends with ragged/bounded descriptors advertise support.
    _supports_packed_tma_o_store: bool = False

    @classmethod
    def _storage_dim_from_logical(cls, head_dim: int) -> int:
        """Return the physical storage width for a logical head dimension."""
        return head_dim

    def __init__(
        self,
        qk_acc_dtype: Type[cutlass.Numeric],
        pv_acc_dtype: Type[cutlass.Numeric],
        mma_tiler: Tuple[int, int],
        head_dim: Union[int, Tuple[int, int]],
        is_persistent: bool,
        mask_type: fmha_utils.MaskEnum,
        enable_ex2_emulation: bool,
        enable_skip_correction: bool,
        rescale_threshold: float = 8.0,
        use_tma_store: bool = True,
        merge_softmax_warp_groups: Optional[bool] = None,
        page_size: Optional[int] = None,
        paged_kv_layout: str = "separate_nhd",
    ):
        """Initializes the configuration for a Blackwell Fused Multi-Head Attention (FMHA) kernel.

        This configuration includes several key aspects:

        1.  Data Type Settings:
            - qk_acc_dtype: Data type for Q*K^T matrix multiplication accumulator
            - pv_acc_dtype: Data type for P*V matrix multiplication accumulator

        2.  MMA Instruction Settings:
            - mma_tiler: The shape of the MMA instruction unit: (M, N) for BMM1 and (M, K) for BMM2
            - head_dim: The head dimension, it can be a single integer or a tuple of two integers (D, Dv).
                        If it is a tuple, Dv is the head dimension of the value & output tensors.
                        It also determines the K dimension of the BMM1's MMA instruction unit
                        & N dimension of the BMM2's MMA instruction unit.
            - qk_mma_tiler: MMA shape for Q*K^T computation
            - pv_mma_tiler: MMA shape for P*V computation

        3.  Kernel Execution Mode:
            - is_persistent: Boolean indicating whether to use persistent kernel mode
            - mask_type: Specifies the type of mask to use (no mask, residual mask, or causal mask)
            - window_size_left/right: Sliding window size for attention masking
            - enable_ex2_emulation: Whether to enable exp2 emulation
            - enable_skip_correction: Whether to skip the correction when rowmax is not updated larger than a threshold

        :param qk_acc_dtype: Data type for Q*K^T matrix multiplication accumulator
        :type qk_acc_dtype: Type[cutlass.Numeric]
        :param pv_acc_dtype: Data type for P*V matrix multiplication accumulator
        :type pv_acc_dtype: Type[cutlass.Numeric]
        :param mma_tiler: The (M, N) shape of the MMA instruction
        :type mma_tiler: Tuple[int, int]
        :param head_dim: The head dimension, it can be a single integer or a tuple of two integers (D, Dv).
        :type head_dim: Union[int, Tuple[int, int]]
        :param is_persistent: Whether to use persistent kernel mode
        :type is_persistent: bool
        :param mask_type: Type of mask to use
        :type mask_type: fmha_utils.MaskEnum
        :param window_size_left: Left-side sliding window size for attention masking
        :type window_size_left: int
        :param window_size_right: Right-side sliding window size for attention masking
        :type window_size_right: int
        :param page_size: Number of tokens in each physical K/V page. ``None``
                          selects contiguous K/V storage. Must be at least 8 and
                          divide the KV tile (``mma_tiler[1]``).
        :type page_size: Optional[int]
        :param paged_kv_layout: Physical paged-KV storage contract.
                                ``separate_nhd`` uses independent token-major
                                K/V pools ``(num_pages, page_size, h_k, 1, d)``.
                                ``combined_hnd`` uses K/V plane views into one
                                ``(num_pages, 2, h_k, page_size, d)`` allocation.
        :type paged_kv_layout: str
        """

        # Architecture feature flags checked inline by the kernel body;
        # subclasses (e.g. Rubin) turn them on and supply the machinery.
        self.enable_fp16_mixed_softmax = False
        self.enable_mma_rowsum = False
        self.enable_q_mixed_residency = False
        self.qk_acc_dtype = qk_acc_dtype
        self.pv_acc_dtype = pv_acc_dtype
        if isinstance(head_dim, tuple):
            self.head_dim = head_dim[0]
            self.head_dim_v = head_dim[1]
            assert self.head_dim == 192 and self.head_dim_v == 128, (
                f"When Headdim is a tuple, it's for MLA. Must be (192, 128), but got {head_dim}"
            )
        else:
            self.head_dim = head_dim
            self.head_dim_v = head_dim
        self.storage_head_dim = self._storage_dim_from_logical(self.head_dim)
        self.storage_head_dim_v = self._storage_dim_from_logical(self.head_dim_v)
        self.cta_tiler = (
            2 * mma_tiler[0],  # 2 O tile per CTA
            mma_tiler[1],
            self.head_dim_v,
        )
        self.qk_mma_tiler = (
            *mma_tiler,
            self.head_dim,
        )
        self.pv_mma_tiler = (
            mma_tiler[0],
            self.head_dim_v,
            mma_tiler[1],
        )
        # Architecture extension points consumed by the shared kernel body
        # and its extension methods. Subclasses that implement these features
        # validate and override the attribute defaults in their constructors.
        # Paged KV: each KV tile gathers ``pages_per_kv_tile`` physical pages
        # through the page table; ``None`` keeps contiguous K/V.
        self.pages_per_kv_tile = 1
        if page_size is not None:
            if self.storage_head_dim != self.head_dim or self.storage_head_dim_v != self.head_dim_v:
                raise ValueError(
                    "paged KV currently requires logical and storage head dimensions to match"
                )
            if page_size < 8:
                raise ValueError(f"page_size must be at least 8, got {page_size}")
            if self.qk_mma_tiler[1] % page_size != 0:
                raise ValueError(
                    f"page_size ({page_size}) must divide the KV tile size ({self.qk_mma_tiler[1]})"
                )
            self.pages_per_kv_tile = self.qk_mma_tiler[1] // page_size
        if paged_kv_layout not in _PAGED_KV_LAYOUTS:
            raise ValueError(
                f"paged_kv_layout must be 'separate_nhd' or 'combined_hnd', got {paged_kv_layout!r}"
            )
        if page_size is None and paged_kv_layout != "separate_nhd":
            raise ValueError("combined_hnd requires paged KV")
        self.page_size = page_size
        # Set per call: keys are a shared prefix plus a window of the paged KV address space.
        self.prefix_window = False
        self.paged_kv_layout = paged_kv_layout
        self.num_q_heads = None
        self.use_cluster_launch = False
        self.use_clc_scheduler = False
        self.use_dynamic_work_scheduler = False
        self.use_ragged_tma_store = False
        self.uniform_varlen_o_extent = None
        self.persistent_raster_order = "mhb"
        self.cluster_shape_mn = (1, 1)
        self.is_persistent = is_persistent
        self.mask_type = mask_type
        self.enable_skip_correction = enable_skip_correction
        self.rescale_threshold = rescale_threshold if enable_skip_correction else 0.0
        # Stage counts for the (subclass-owned) hardware-CLC / dynamic
        # work-queue response pipelines; they also size the shared-storage
        # reservation below.
        self.clc_stage = 2
        self.dynamic_work_stage = 1
        self.enable_ex2_emulation = enable_ex2_emulation
        # For most cases, seq barrier is needed to help keep the pipeline stable
        # But sometimes, compiler will schedule the barrier at an unexpected place
        # if it hurts perf a lot, try to quickly fix it by disabling seq barrier
        self.enable_sequence_barrier = False
        self.use_tma_store = use_tma_store
        # Repetition of the TMEM copy atom used to load S rows in the softmax
        # warps. Architecture subclasses may widen it.
        self.softmax_tmem_load_repetition = 32
        # Architecture subclasses may replace the generic mask with the scalar
        # setp/selp inline-PTX specialization (causal, sliding-window, and
        # residual masks).
        self.use_causal_setp_selp_inline = False
        # TMEM columns loaded per correction-rescale copy; wider tiles expose
        # independent FMUL2 work across fewer, larger load/store groups.
        self.correction_rescale_tile_size = 16
        # ``None`` fully unrolls the correction epilogue; subclasses may cap
        # the unroll factor to bound the live swizzled sO addresses.
        self.correction_epilog_unroll = None

        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.softmax1_warp_ids = (4, 5, 6, 7)
        self.correction_warp_ids = (8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 13
        self.epilogue_warp_id = 14
        self.empty_warp_id = 15
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(self.arch_str)

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                self.epilogue_warp_id,
                self.empty_warp_id,
            )
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_warp
            * sum(
                (
                    len((self.mma_warp_id,)),
                    len(self.softmax0_warp_ids),
                    len(self.softmax1_warp_ids),
                    len(self.correction_warp_ids),
                )
            ),
        )
        # Ping-pong sequence barriers indexed by the waiting softmax stage
        # (IDs base+0 and base+1); see _sequence_barrier().
        self.sequence_barrier_base_id = 3
        self.sequence_barrier_num_threads = self.threads_per_warp * len(
            (*self.softmax0_warp_ids, *self.softmax1_warp_ids)
        )
        # Per-stage softmax warpgroup vote barriers (IDs base+0 and base+1);
        # see _softmax_warpgroup_barrier().
        self.softmax_warpgroup_barrier_base_id = 5
        self.softmax_warpgroup_barrier_num_threads = self.threads_per_warp * len(
            self.softmax0_warp_ids
        )
        self.tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=7,
            num_threads=self.threads_per_warp * len(self.correction_warp_ids),
        )

        self.tmem_s0_offset = 0
        self.tmem_s1_offset = 128
        self.tmem_o0_offset = 256
        self.tmem_o1_offset = 384
        # inplaced with s1
        self.tmem_p0_offset = 160
        # inplaced with s0
        self.tmem_p1_offset = 32
        # vec buffer for row_max & row_sum
        # inplaced with s0
        self.tmem_vec0_offset = 0
        # inplaced with s1
        self.tmem_vec1_offset = 128
        # skip mma pv flag offset regarding to the vec buffer
        # inplaced with s1
        self.tmem_skip_softmax0_offset = 136
        # inplaced with s0
        self.tmem_skip_softmax1_offset = 8

        self.num_regs_softmax = 192
        self.num_regs_correction = 88
        self.num_regs_other = 40
        self.buffer_align_bytes = 1024
        self.arch = BaseDSL._get_dsl().get_arch_enum()
        self.merge_softmax_warp_groups = (
            bool(self.arch >= Arch.sm_103)
            if merge_softmax_warp_groups is None
            else merge_softmax_warp_groups
        )

        if self.arch >= Arch.sm_103:
            assert not self.enable_ex2_emulation, (
                f"Don't enable exp2 emulation for {self.arch}, it doesn't help performance"
            )

        num_warps_per_warpgroup = 4
        self.softmax_warpgroup_count = (
            len((*self.softmax0_warp_ids, *self.softmax1_warp_ids)) // num_warps_per_warpgroup
        )

    def _sequence_barrier(self, waiting_stage):
        """Sequence named barrier the given (possibly runtime) stage waits on.

        The stage-k softmax group waits on ID ``base + k`` and signals the
        other group's barrier via ``_sequence_barrier(1 - k)``.
        """
        return pipeline.NamedBarrier(
            barrier_id=self.sequence_barrier_base_id + waiting_stage,
            num_threads=self.sequence_barrier_num_threads,
        )

    def _softmax_warpgroup_barrier(self, stage):
        """Skip-softmax vote barrier of the given (possibly runtime) stage."""
        return pipeline.NamedBarrier(
            barrier_id=self.softmax_warpgroup_barrier_base_id + stage,
            num_threads=self.softmax_warpgroup_barrier_num_threads,
        )

    def _make_qk_tiled_mma(self, cta_group):
        """Build the QK tiled MMA. Override in subclasses to target a different arch."""
        return sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.qk_mma_tiler[:2],
        )

    def _make_pv_tiled_mma(self, cta_group, p_major_mode, p_source):
        """Build the PV tiled MMA. Override in subclasses to target a different arch."""
        return sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            self.v_dtype,
            p_major_mode,
            self.v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.pv_mma_tiler[:2],
            p_source,
        )

    # ------------------------------------------------------------------
    # Architecture extension points
    #
    # The generic Blackwell kernel is written entirely against the hooks
    # below. Architecture subclasses (e.g. Rubin) override them to inject
    # arch-specific operands, TMEM/SMEM state, and math variants without the
    # base kernel knowing about any particular feature. Every hook's base
    # implementation is the plain Blackwell behavior.
    # ------------------------------------------------------------------

    def _make_q_smem_layout_staged(self, qk_tiled_mma):
        """Staged shared-memory layout for the Q operand."""
        return sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.q_dtype,
            self.q_stage,
        )

    def _make_arch_kernel_spec(self, cta_group, qk_tiled_mma):
        """Return ``(kernel_args, smem_scratch_cosize, smem_scratch_align)``.

        ``kernel_args`` is forwarded opaquely to :meth:`kernel` as
        ``arch_args``; the scratch cosize/align size the ``sArchScratch``
        shared-memory field (in ``v_dtype`` elements).
        """
        return (), 0, 1

    @cute.jit
    def _make_arch_smem_views(self, storage, arch_args):
        """Build arch-specific shared-memory views inside the kernel."""
        return None

    @cute.jit
    def _init_arch_smem_operands(self, arch_smem, warp_idx, tidx):
        """Initialize arch-specific shared-memory operands.

        Runs in the kernel's common prologue, so overrides may synchronize
        all threads to publish their stores.
        """
        return

    @cute.jit
    def _make_mma_warp_arch_state(self, arch_smem, arch_args, tmem_ptr):
        """Build per-MMA-warp architecture state from the opaque arch_args."""
        return None

    @cute.jit
    def _stage_arch_q_operand(self, arch_state, q_index, q_slot):
        """Stage the Q operand for one pipeline stage (no-op on Blackwell)."""
        return

    @cute.jit
    def _mma_qk_step(
        self,
        tiled_mma,
        tensor_args,
        pipeline_args,
        arch_state,
        q_slot,
        q_index,
    ):
        """One QK GEMM step; overrides may substitute arch-specific operands."""
        return self.mma_qk(tiled_mma, tensor_args, pipeline_args)

    @cute.jit
    def _mma_pv_step(
        self,
        tiled_mma,
        whether_acc,
        tensor_args,
        pipeline_args,
        skip_pv_args,
        arch_state,
        o_slot,
    ):
        """One PV GEMM step; overrides may issue extra arch-specific MMAs."""
        return self.mma_pv(tiled_mma, whether_acc, tensor_args, pipeline_args, skip_pv_args)

    @cute.jit
    def _make_correction_arch_state(self, arch_args, tmem_ptr, thread_idx):
        """Build per-correction-warp architecture state."""
        return None

    def _setup_attributes(self):
        """Set up configurations and parameters for the FMHA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the fused multi-head attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """

        self.q_stage = 2
        k_stage = 4 if self.q_dtype.width == 8 or not self.use_tma_store else 3
        v_stage = 4 if self.v_dtype.width == 8 or not self.use_tma_store else 3
        self.kv_stage = min(k_stage, v_stage)
        # For D192, the smem usage of Q & K is larger. So, we need to reduce the stage count.
        if self.head_dim == 192 and self.q_dtype.width == 16:
            self.kv_stage = 2
        self.p_mma_stage = 1
        self.acc_stage = 1
        self.softmax_corr_stage = 1
        self.mma_corr_stage = 2
        self.mma_softmax_stage = 1
        self.epi_stage = 2
        # A dedicated warp prefetches one KV tile's page IDs per stage. The
        # load warp holds each stage until both the matching K and V TMAs have
        # consumed it, eliminating the duplicate K/V global page-table reads.
        self.load_pt_stage = 4

        # Tunable parameters
        # FP8 P pre-scale: offset added to exp2 exponent so that P*2^offset fills
        # more of E4M3's [0, 448] range, improving quantization precision.
        # Derived from rescale_threshold to guarantee P*2^offset <= 448.
        self.p_fp8_prescale_log2 = max(0.0, math.floor(math.log2(448) - self.rescale_threshold))
        # ln(2) * offset correction for LSE when pre-scale is active
        self.p_fp8_prescale_lse_correction = self.p_fp8_prescale_log2 * math.log(2)
        # Optional double buffering for correction rescale.
        self.enable_correction_double_buffer = False
        # Correction epilogue tile width along the logical output dimension;
        # subclasses with a padded BMM2 N tile override it after super().
        self.correction_epilog_tile_size = 32 * 8 // self.o_dtype.width
        # ``None`` keeps the architecture-default depth of the softmax
        # exp->convert software pipeline; subclasses may deepen it.
        self.softmax_cvt_count = None

    @cute.jit
    def __call__(
        self,
        q_tensor: cute.Tensor,
        k_tensor: cute.Tensor,
        v_tensor: cute.Tensor,
        o_tensor: cute.Tensor,
        problem_size: Tuple[Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32],
        cum_seqlen_q: Optional[cute.Tensor],
        cum_seqlen_k: Optional[cute.Tensor],
        lse_tensor: Optional[cute.Tensor],
        sink_tensor: Optional[cute.Tensor],
        scale_softmax_log2: Float32,
        scale_softmax: Float32,
        scale_output: Float32,
        skip_softmax_threshold_log2: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        skip_softmax_count: Optional[cute.Tensor],
        total_softmax_count: Optional[cute.Tensor],
        stream: cuda.CUstream,
        use_pdl: bool,
        page_table: Optional[cute.Tensor] = None,
        o_sf_tensor: Optional[cute.Tensor] = None,
        o_sf_scale: Optional[Float32] = None,
        o_sf_start_index: Optional[Int32] = None,
        dynamic_work_counter: Optional[cute.Tensor] = None,
        scale_softmax_log2_tensor: Optional[cute.Tensor] = None,
        scale_softmax_tensor: Optional[cute.Tensor] = None,
        scale_output_tensor: Optional[cute.Tensor] = None,
        kv_len_tensor: Optional[cute.Tensor] = None,
        prefix_tokens_k: Optional[Int32] = None,
    ):
        """Execute the Fused Multi-Head Attention operation on the provided tensors.

        This method prepares the input tensors for processing, validates their shapes and types,
        configures the computation parameters, and launches the CUDA kernel.

        The method handles:
        1. Tensor layout transformations for specific memory access patterns
        2. Validation of tensor shapes and data types
        3. Initialization of hardware-specific parameters and memory layouts
        4. Configuration of TMA (Tensor Memory Access) operations
        5. Grid and work scheduling computation
        6. Kernel launch with appropriate parameters

        :param q_tensor: The query tensor with shape ``(b, s_q, h_k, h_r,
                         storage_d)``, or a packed rank-3 ``[tokens, h_q,
                         storage_d]`` buffer for variable sequence lengths
        :type q_tensor: cute.Tensor
        :param k_tensor: The key tensor with shape ``(b, s_k, h_k, 1, storage_d)``
                         for contiguous K/V. Paged ``separate_nhd`` uses
                         ``(num_pages, page_size, h_k, 1, storage_d)``. Paged
                         ``combined_hnd`` uses a zero-copy K-plane view of a
                         ``(num_pages, 2, h_k, page_size, storage_d)`` backing
                         allocation. Logical sequence masking removes unused K
                         slots in a partial final page from the score tensor.
        :type k_tensor: cute.Tensor
        :param v_tensor: The value tensor with shape
                         ``(b, s_v, h_k, 1, storage_dv)`` for contiguous K/V.
                         Paged ``separate_nhd`` uses a separate pool matching K.
                         Paged ``combined_hnd`` uses the V-plane view of the same
                         combined backing allocation. Unused V slots in a
                         partial final page must be finite because full physical
                         pages participate in the PV load pipeline.
        :type v_tensor: cute.Tensor
        :param o_tensor: The output tensor with shape ``(b, s_q, h_k, h_r,
                         storage_dv)``, or a packed rank-3 ``[tokens, h_q,
                         storage_dv]`` buffer for variable sequence lengths
        :type o_tensor: cute.Tensor
        :param problem_size: The problem size with shape [b, s_q_max, s_lse_max, s_k_max, h_q, h_k, d, dv]. If cum_seqlen_q or cum_seqlen_k is not None, s_q_max and s_k_max are the max of the per-batch sequence lengths respectively.
        :type problem_size: Tuple[Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32]
        :param cum_seqlen_q: The cumulative sequence length tensor for query
        :type cum_seqlen_q: Optional[cute.Tensor]
        :param cum_seqlen_k: The cumulative sequence length tensor for key.
                             Paged mode requires it even for fixed-length batches,
                             because it describes logical rather than physical-pool
                             sequence lengths.
        :type cum_seqlen_k: Optional[cute.Tensor]
        :param page_table: Logical-to-physical KV page mapping with shape
                           ``(batch, max_pages_per_batch)`` and Int32 elements.
                           Every used entry must index a valid physical K/V page.
        :type page_table: Optional[cute.Tensor]
        :param o_sf_tensor: Contiguous, unit-stride E4M3 scale-factor storage
                            using the NVFP4 Layout128x4 layout. Required only
                            for E2M1 output. A direct caller must provide at
                            least ``rows * columns`` elements, where ``rows``
                            covers the token capacity plus
                            ``o_sf_start_index`` rounded up to 128 and
                            ``columns`` covers the per-head 16-value blocks
                            rounded up to four.
        :type o_sf_tensor: Optional[cute.Tensor]
        :param o_sf_scale: Positive global normalization constant used by the
                           fused NVFP4 quantization epilogue.
        :type o_sf_scale: Optional[Float32]
        :param o_sf_start_index: Starting token-row offset in ``o_sf_tensor``.
        :type o_sf_start_index: Optional[Int32]
        :param dynamic_work_counter: Per-in-flight ``Int32[2]`` state:
                                     ``[next_work_idx, exact_compact_work]``.
                                     It must not be shared by overlapping
                                     launches or streams.
        :type dynamic_work_counter: Optional[cute.Tensor]
        :param scale_softmax_log2: The log2 scale factor for softmax
        :type scale_softmax_log2: Float32
        :param scale_softmax: The scale factor for softmax
        :type scale_softmax: Float32
        :param scale_output: The scale factor for the output
        :type scale_output: Float32
        :param scale_softmax_log2_tensor: Optional device scalar that replaces scale_softmax_log2
        :type scale_softmax_log2_tensor: Optional[cute.Tensor]
        :param scale_softmax_tensor: Optional device scalar that replaces scale_softmax
        :type scale_softmax_tensor: Optional[cute.Tensor]
        :param scale_output_tensor: Optional device scalar that replaces scale_output
        :type scale_output_tensor: Optional[cute.Tensor]
        :param kv_len_tensor: Fixed window attention with a prefix sequence (paged KV only), together
                              with ``prefix_tokens_k``. ``(batch,)`` Int32 key lengths. Sequence ``i``
                              attends to the logical key positions ``[cum_seqlen_k[i], cum_seqlen_k[i] +
                              kv_len_tensor[i])`` and to ``[0, prefix_tokens_k)``. ``cum_seqlen_k`` is
                              then ``(batch,)`` start positions with no prefix-sum requirement (equal
                              starts make sequences share keys), ``page_table`` is one shared row
                              ``(1, max_pages)`` indexed by absolute position // page_size, and
                              ``problem_size``'s ``s_k_max`` is not used. There is no query-dependent mask:
                              every query row of a sequence sees the same keys.
        :type kv_len_tensor: Optional[cute.Tensor]
        :param prefix_tokens_k: Number of leading logical key positions every sequence also attends to.
        :type prefix_tokens_k: Optional[Int32]
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[Int32]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[Int32]
        :param stream: The CUDA stream to execute the kernel on
        :type stream: cuda.CUstream
        :raises TypeError: If tensor data types don't match or aren't supported
        :raises RuntimeError: If tensor layouts aren't in supported formats
        """
        b, s_q_max, s_lse_max, s_k_max, h_q, h_k, d, dv = problem_size
        h_r = h_q // h_k
        # setup static attributes before smem/grid/tma computation
        self.q_dtype = q_tensor.element_type
        self.k_dtype = k_tensor.element_type
        self.v_dtype = v_tensor.element_type
        self.o_dtype = o_tensor.element_type
        self.enable_nvfp4_output = self.o_dtype is cutlass.Float4E2M1FN
        self.prefix_window = prefix_tokens_k is not None
        if cutlass.const_expr(self.prefix_window):
            if cutlass.const_expr(
                self.page_size is None or kv_len_tensor is None or cum_seqlen_k is None
            ):
                raise ValueError(
                    "prefix_tokens_k requires paged KV, cum_seqlen_k and kv_len_tensor"
                )
        elif cutlass.const_expr(kv_len_tensor is not None):
            raise ValueError("kv_len_tensor requires prefix_tokens_k")
        self._validate_arch_call_operands(
            k_tensor,
            v_tensor,
            cum_seqlen_q,
            cum_seqlen_k,
            page_table,
            o_sf_tensor,
            o_sf_scale,
            o_sf_start_index,
            dynamic_work_counter,
        )

        # s_q, s_k, s_v are the storage extents. Variable-length callers may
        # provide either the example's legacy rank-5 tensors or production
        # packed rank-3 [tokens, heads, storage_D] buffers.
        if cutlass.const_expr(len(q_tensor.shape) == 3):
            if cutlass.const_expr(cum_seqlen_q is None):
                raise RuntimeError("rank-3 Q/O requires variable sequence lengths")
            s_q = q_tensor.shape[0]
        else:
            s_q = q_tensor.shape[1]
        if cutlass.const_expr(self.page_size is not None):
            s_k = self.page_size
            s_v = self.page_size
        elif cutlass.const_expr(len(k_tensor.shape) == 3):
            if cutlass.const_expr(cum_seqlen_k is None):
                raise RuntimeError("rank-3 K/V requires variable sequence lengths")
            s_k = k_tensor.shape[0]
            s_v = v_tensor.shape[0]
        else:
            s_k = k_tensor.shape[1]
            s_v = v_tensor.shape[1]
        s_lse = s_lse_max
        # Important for performance
        align = 256 // self.o_dtype.width
        # Do not promise a stronger dynamic alignment than the static logical
        # dimension provides. This matters for non-power-of-two dimensions.
        d = cute.assume(Int32(d), math.gcd(align, self.head_dim))
        dv = cute.assume(Int32(dv), math.gcd(align, self.head_dim_v))

        storage_d = cute.assume(
            Int32(self.storage_head_dim), math.gcd(align, self.storage_head_dim)
        )
        storage_dv = cute.assume(
            Int32(self.storage_head_dim_v), math.gcd(align, self.storage_head_dim_v)
        )
        stride_b_q = h_r * h_k * s_q * storage_d if cum_seqlen_q is None else 0
        o_s_q = self.uniform_varlen_o_extent if self.uniform_varlen_o_extent is not None else s_q
        stride_b_o = (
            h_r * h_k * o_s_q * storage_dv
            if cum_seqlen_q is None or self.uniform_varlen_o_extent is not None
            else 0
        )
        stride_b_k = h_k * s_k * storage_d if cum_seqlen_k is None else 0
        stride_b_v = h_k * s_v * storage_dv if cum_seqlen_k is None else 0
        stride_b_lse = h_r * h_k * s_lse if cum_seqlen_q is None else 0

        # Tensor extents carry logical bounds while strides map coordinates to
        # physical storage. Architectures may therefore pad a logical head
        # dimension without exposing that padding in the public shape.
        # (b, s_q, h_k, h_r, storage_d) -> (s_q, logical_d, ((h_r, h_k), b))
        q_layout = cute.make_layout(
            (s_q, d, ((h_r, h_k), b)),
            stride=(
                storage_d * h_r * h_k,
                1,
                ((storage_d, storage_d * h_r), stride_b_q),
            ),
        )
        q = cute.make_tensor(q_tensor.iterator, q_layout)
        if cutlass.const_expr(self.page_size is not None):
            k_layout, v_layout = self._make_paged_kv_gmem_layouts(
                k_tensor, d, dv, h_r, h_k, storage_d, storage_dv
            )
        else:
            # (b, s_k, h_k, 1, storage_d) -> (s_k, logical_d, ((h_r, h_k), b))
            k_layout = cute.make_layout(
                (s_k, d, ((h_r, h_k), b)),
                stride=(storage_d * h_k, 1, ((0, storage_d), stride_b_k)),
            )
            # (b, s_v, h_k, 1, storage_dv) -> (logical_dv, s_v, ((h_r, h_k), b))
            v_layout = cute.make_layout(
                (dv, s_v, ((h_r, h_k), b)),
                stride=(1, storage_dv * h_k, ((0, storage_dv), stride_b_v)),
            )
        k = cute.make_tensor(k_tensor.iterator, k_layout)
        v = cute.make_tensor(v_tensor.iterator, v_layout)
        if cutlass.const_expr(page_table is not None):
            page_table = self._transpose_page_table(page_table)
        # Output stores cover the physical extent. The epilogue explicitly
        # defines any columns beyond the logical output width as zero.
        # (b, s_q, h_k, h_r, storage_dv) -> (s_q, storage_dv, ((h_r, h_k), b))
        o_layout = cute.make_layout(
            (o_s_q, storage_dv, ((h_r, h_k), b)),
            stride=(
                storage_dv * h_r * h_k,
                1,
                ((storage_dv, storage_dv * h_r), stride_b_o),
            ),
        )
        o = cute.make_tensor(o_tensor.iterator, o_layout)
        if cutlass.const_expr(self.enable_nvfp4_output):
            if cutlass.const_expr(isinstance(h_q, int) and self.num_q_heads != h_q):
                raise ValueError(
                    f"num_q_heads={self.num_q_heads} must match the problem's "
                    f"query-head count h_q={h_q}; the Layout128x4 scale pitch "
                    "would otherwise diverge from the kernel's stores"
                )
            self.o_sf_columns = ((self.num_q_heads * (self.storage_head_dim_v // 16) + 3) // 4) * 4
        if cutlass.const_expr(lse_tensor is not None):
            # (s, ((h_r, h_k), b)) - head stride=1 to match FlashInfer (total_q, h_q) convention
            lse_layout = cute.make_layout(
                (s_lse, ((h_r, h_k), b)),
                stride=(h_r * h_k, ((1, h_r), stride_b_lse)),
            )
            lse = cute.make_tensor(lse_tensor.iterator, lse_layout)
        else:
            lse = None

        if cutlass.const_expr(sink_tensor is not None):
            # sink_tensor is 1D with shape (h_q,) = (h_k * h_r,)
            # Create layout ((h_r, h_k), b) with stride 0 for batch so blk_coord[2] works
            sink_layout = cute.make_layout(
                ((h_r, h_k), b),
                stride=((1, h_r), 0),
            )
            sink = cute.make_tensor(sink_tensor.iterator, sink_layout)
        else:
            sink = None

        self.tile_sched_params, grid = fmha_utils.compute_grid(
            cute.shape((s_q_max, d, ((h_r, h_k), b))),
            self.cta_tiler,
            self.is_persistent,
            persistent_raster_order=self.persistent_raster_order,
        )
        self.q_major_mode = cutlass.tensor_utils.LayoutEnum.from_tensor(q).mma_major_mode()
        self.k_major_mode = cutlass.tensor_utils.LayoutEnum.from_tensor(k).mma_major_mode()
        self.v_major_mode = cutlass.tensor_utils.LayoutEnum.from_tensor(v).mma_major_mode()
        self.o_layout = cutlass.tensor_utils.LayoutEnum.from_tensor(o)

        if cutlass.const_expr(self.q_major_mode != OperandMajorMode.K):
            raise RuntimeError("The layout of q is not supported")
        if cutlass.const_expr(self.k_major_mode != OperandMajorMode.K):
            raise RuntimeError("The layout of k is not supported")
        if cutlass.const_expr(self.v_major_mode != OperandMajorMode.MN):
            raise RuntimeError("The layout of v is not supported")

        # check type consistency: Q and K must share the same dtype (qk_dtype);
        # V may use a different dtype (pv_dtype) to allow qk_dtype != pv_dtype.
        if cutlass.const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.ONE
        # the intermediate tensor p is from tmem & k-major
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = cute.nvgpu.OperandMajorMode.K
        qk_tiled_mma = self._make_qk_tiled_mma(cta_group)
        pv_tiled_mma = self._make_pv_tiled_mma(cta_group, p_major_mode, p_source)

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )
        self.epi_tile = self.pv_mma_tiler[:2]

        q_smem_layout_staged = self._make_q_smem_layout_staged(qk_tiled_mma)
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.k_dtype,
            self.kv_stage,
        )
        p_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            pv_tiled_mma,
            self.pv_mma_tiler,
            self.v_dtype,
            self.acc_stage,
        )
        (
            arch_kernel_args,
            arch_scratch_cosize,
            arch_scratch_align,
        ) = self._make_arch_kernel_spec(cta_group, qk_tiled_mma)
        v_smem_layout_staged_origin = sm100_utils.make_smem_layout_b(
            pv_tiled_mma,
            self.pv_mma_tiler,
            self.v_dtype,
            self.kv_stage,
        )
        # k & v share the same smem buffer. Pad the smaller-tile operand's stage stride to
        # match the larger tile. Stride is scaled to the target operand's element width.
        # sK_cosize covers all kv_stage slots at the larger tile's byte footprint.
        if cutlass.const_expr(self.k_dtype.width >= self.v_dtype.width):
            # k tile >= v tile: give v k's stage stride in v_dtype units
            k_tile_cosize = cute.cosize(cute.select(k_smem_layout_staged, mode=[0, 1, 2]))
            v_stage_stride = k_tile_cosize * self.k_dtype.width // self.v_dtype.width
            v_smem_layout_staged = cute.append(
                cute.select(v_smem_layout_staged_origin, mode=[0, 1, 2]),
                cute.make_layout(self.kv_stage, stride=v_stage_stride),
            )
            sK_cosize = cute.cosize(k_smem_layout_staged)
        else:
            # v tile > k tile: give k v's stage stride in k_dtype units
            v_tile_cosize = cute.cosize(cute.select(v_smem_layout_staged_origin, mode=[0, 1, 2]))
            k_stage_stride = v_tile_cosize * self.v_dtype.width // self.k_dtype.width
            k_smem_layout_staged = cute.append(
                cute.select(k_smem_layout_staged, mode=[0, 1, 2]),
                cute.make_layout(self.kv_stage, stride=k_stage_stride),
            )
            v_smem_layout_staged = v_smem_layout_staged_origin
            # cute.cosize gives (kv_stage-1)*k_stage_stride + k_tile_cosize, but the
            # last stage must also accommodate a full V tile, so size for kv_stage slots.
            sK_cosize = self.kv_stage * k_stage_stride

        if cutlass.const_expr(self.use_tma_store or not self.enable_nvfp4_output):
            o_smem_layout_staged = sm100_utils.make_smem_layout_epi(
                self.o_dtype,
                self.o_layout,
                self.epi_tile,
                self.epi_stage,
            )
        else:
            o_smem_layout_staged = self._nvfp4_o_smem_placeholder()

        # TMA load for Q
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q,
            q_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # TMA load for K
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        k_smem_layout_for_tma = k_smem_layout_staged
        if cutlass.const_expr(self.page_size is not None):
            tma_atom_k, tma_tensor_k, k_smem_layout_for_tma = self._make_paged_k_tma_atom(
                tma_load_op, k, k_smem_layout_staged
            )
        else:
            tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                k,
                k_smem_layout,
                self.qk_mma_tiler,
                qk_tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
        # TMA load for V
        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
        v_smem_layout_for_tma = v_smem_layout_staged
        if cutlass.const_expr(self.page_size is not None):
            tma_atom_v, tma_tensor_v, v_smem_layout_for_tma = self._make_paged_v_tma_atom(
                tma_load_op, v, v_smem_layout_staged
            )
        else:
            tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                v,
                v_smem_layout,
                self.pv_mma_tiler,
                pv_tiled_mma,
                self.cluster_layout_vmnk.shape,
            )

        o_smem_layout = cute.select(o_smem_layout_staged, mode=[0, 1])
        if cutlass.const_expr(self.use_tma_store or not self.enable_nvfp4_output):
            tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_store_op,
                o,
                o_smem_layout,
                self.epi_tile,
            )
        else:
            # Compile-time-dead placeholders for the epilogue warp.
            tma_atom_o, tma_tensor_o = tma_atom_q, tma_tensor_q
        ragged_tma_o_desc = None
        if cutlass.const_expr(self.use_ragged_tma_store):
            # The TMA chunk/swizzle selection and descriptor box need the
            # compile-time storage width, not the alignment-annotated runtime
            # value used by the gmem layout strides.
            ragged_tma_o_desc = self._make_ragged_o_tma_descriptor(
                o_tensor, cum_seqlen_q, s_q, h_q, self.storage_head_dim_v
            )

        q_copy_size = cute.size_in_bytes(self.q_dtype, q_smem_layout)
        k_copy_size = cute.size_in_bytes(self.k_dtype, k_smem_layout)
        v_copy_size = cute.size_in_bytes(self.v_dtype, v_smem_layout)
        self.tma_copy_q_bytes = q_copy_size
        self.tma_copy_k_bytes = k_copy_size
        self.tma_copy_v_bytes = v_copy_size

        # The cute.jit AST preprocessor drops class-body `if` statements, so
        # the sO field must exist unconditionally; direct-store kernels give
        # it zero elements instead.
        sO_cosize = cute.cosize(o_smem_layout_staged) if self.use_tma_store else 0

        @cute.struct
        class SharedStorage:
            # Pipeline barriers
            load_q_mbar_ptr: cute.struct.MemRange[Int64, self.q_stage * 2]
            load_kv_mbar_ptr: cute.struct.MemRange[Int64, self.kv_stage * 2]
            load_pt_mbar_ptr: cute.struct.MemRange[
                Int64, self.load_pt_stage * 2 if self.page_size is not None else 1
            ]
            # Each softmax pipeline pair is laid out as
            # [group0(full, empty), group1(full, empty)] so the softmax warps
            # can select their group with one runtime pointer offset.
            mma_s_mbar_ptr: cute.struct.MemRange[Int64, self.mma_softmax_stage * 2 * 2]
            p_mma_mbar_ptr: cute.struct.MemRange[Int64, self.p_mma_stage * 2 * 2]
            s_corr_mbar_ptr: cute.struct.MemRange[Int64, self.softmax_corr_stage * 2 * 2]
            corr_epi_mbar_ptr: cute.struct.MemRange[Int64, self.epi_stage * 2]
            mma_corr_mbar_ptr: cute.struct.MemRange[Int64, self.mma_corr_stage * 2]
            inplace_mbar_ptr: cute.struct.MemRange[Int64, self.p_mma_stage * 2 * 2]
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # One Int32 slot per producer lane keeps the stage stride fixed
            # while supporting every legal page size (at most 16 IDs/tile).
            # The ring fits in the padding before the 1024-byte-aligned sO.
            sPageIds: cute.struct.MemRange[
                Int32,
                (self.load_pt_stage * self.threads_per_warp if self.page_size is not None else 1),
            ]
            # Hardware CLC and software dynamic work-index scheduling are mutually
            # exclusive and reuse this response-pipeline storage.
            clc_pipeline_mbar: cute.struct.MemRange[Int64, self.clc_stage * 2]
            clc_response: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int128, self.clc_stage], 16
            ]
            # Smem tensors
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, sO_cosize],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, sK_cosize],
                self.buffer_align_bytes,
            ]
            # Architecture-specific shared-memory scratch (empty on Blackwell)
            sArchScratch: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, arch_scratch_cosize],
                arch_scratch_align,
            ]
            # Skip softmax and PV warpgroup votes, one 4-slot column per
            # softmax stage
            warp_wants_skip_softmax_exchange: cute.struct.MemRange[Int8, 4 * 2]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            qk_tiled_mma,
            pv_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_o,
            tma_tensor_o,
            ragged_tma_o_desc,
            o,
            o_sf_tensor,
            cum_seqlen_q,
            cum_seqlen_k,
            lse,
            sink,
            page_table,
            scale_softmax_log2,
            scale_softmax,
            scale_output,
            scale_softmax_log2_tensor,
            scale_softmax_tensor,
            scale_output_tensor,
            kv_len_tensor,
            prefix_tokens_k,
            o_sf_scale,
            o_sf_start_index,
            skip_softmax_threshold_log2,
            window_size_left,
            window_size_right,
            q_smem_layout_staged,
            k_smem_layout_staged,
            k_smem_layout_for_tma,
            p_tmem_layout_staged,
            v_smem_layout_staged,
            v_smem_layout_for_tma,
            o_smem_layout_staged,
            arch_kernel_args,
            skip_softmax_count,
            total_softmax_count,
            self.tile_sched_params,
            dynamic_work_counter,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(self.cluster_shape_mnk if self.use_cluster_launch else None),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=use_pdl,
        )

    def _nvfp4_o_smem_placeholder(self):
        """Legal sO placeholder for direct-store kernels that skip TMA O."""
        # Direct stores do not consume sO. Keep a legal placeholder layout
        # so packed sub-byte outputs do not attempt to build an unsupported
        # TMA tensor map.
        o_smem_outer = cute.make_layout(
            (self.epi_tile[0], self.epi_tile[1], self.epi_stage),
            stride=(
                self.epi_tile[1],
                1,
                self.epi_tile[0] * self.epi_tile[1],
            ),
        )
        o_smem_layout_staged = cute.make_composed_layout(
            cute.make_swizzle(0, 3, 4),
            0,
            o_smem_outer,
        )
        return o_smem_layout_staged

    def _validate_arch_call_operands(
        self,
        k_tensor,
        v_tensor,
        cum_seqlen_q,
        cum_seqlen_k,
        page_table,
        o_sf_tensor,
        o_sf_scale,
        o_sf_start_index,
        dynamic_work_counter,
    ):
        """Validate arch-specific kernel operands; overrides add their own contracts."""
        nvfp4_output_implemented = False
        if cutlass.const_expr(self.enable_nvfp4_output and not nvfp4_output_implemented):
            raise ValueError("NVFP4 output is not supported in this build")
        if cutlass.const_expr(
            self.use_tma_store
            and cum_seqlen_q is not None
            and not self._supports_packed_tma_o_store
        ):
            raise ValueError(
                "variable-length Q requires use_tma_store=False on this kernel; "
                "its dense TMA output store would overwrite packed ragged tails"
            )
        if cutlass.const_expr(self.page_size is not None):
            if cutlass.const_expr(page_table is None):
                raise ValueError("page_table is required when paged KV is enabled")
            if cutlass.const_expr(cum_seqlen_k is None):
                raise ValueError("cum_seqlen_k is required to describe logical paged-KV lengths")
            if cutlass.const_expr(self.paged_kv_layout == "separate_nhd"):
                if cutlass.const_expr(cute.rank(k_tensor) != 5 or cute.rank(v_tensor) != 5):
                    raise ValueError("separate_nhd paged K/V pools must be rank-5 tensors")
            else:
                if cutlass.const_expr(cute.rank(k_tensor) != 4 or cute.rank(v_tensor) != 4):
                    raise ValueError("combined_hnd K/V plane views must be rank-4 tensors")
                if cutlass.const_expr(self.k_dtype != self.v_dtype):
                    raise TypeError("combined_hnd requires matching K and V dtypes")
                if cutlass.const_expr(self.storage_head_dim != self.storage_head_dim_v):
                    raise ValueError("combined_hnd requires matching K and V storage dimensions")
            if cutlass.const_expr(cute.rank(page_table) != 2):
                raise ValueError("page_table must be a rank-2 tensor")
            if cutlass.const_expr(page_table.element_type != Int32):
                raise TypeError("page_table must have Int32 elements")
            if cutlass.const_expr(cute.rank(cum_seqlen_k) != 1):
                raise ValueError("cum_seqlen_k must have shape (batch + 1,)")
            if cutlass.const_expr(cum_seqlen_k.element_type != Int32):
                raise TypeError("cum_seqlen_k must have Int32 elements")
        elif cutlass.const_expr(page_table is not None):
            raise ValueError("page_table requires a configured page_size")
        if cutlass.const_expr(
            not self.enable_nvfp4_output
            and (o_sf_tensor is not None or o_sf_scale is not None or o_sf_start_index is not None)
        ):
            raise ValueError("output scale-factor arguments require NVFP4 output")
        if cutlass.const_expr(
            dynamic_work_counter is not None and not self.use_dynamic_work_scheduler
        ):
            raise ValueError("dynamic_work_counter requires the dynamic work-index scheduler")

    def _make_paged_kv_gmem_layouts(self, k_tensor, d, dv, h_r, h_k, storage_d, storage_dv):
        """Page-granular K/V gmem descriptors for both paged storage layouts."""
        num_pages = k_tensor.shape[0]
        if cutlass.const_expr(self.paged_kv_layout == "combined_hnd"):
            # K/V are plane views of [num_pages, 2, h_k, page_size, D].
            # Their iterators already carry the plane base offset. Rebuild
            # descriptors with the backing allocation's page/head strides.
            k_layout = cute.make_layout(
                (self.page_size, d, ((h_r, h_k), num_pages)),
                stride=(
                    storage_d,
                    1,
                    (
                        (0, self.page_size * storage_d),
                        2 * self.page_size * h_k * storage_d,
                    ),
                ),
            )
            v_layout = cute.make_layout(
                (dv, self.page_size, ((h_r, h_k), num_pages)),
                stride=(
                    1,
                    storage_dv,
                    (
                        (0, self.page_size * storage_dv),
                        2 * self.page_size * h_k * storage_dv,
                    ),
                ),
            )
        else:
            # Separate physical pools are
            # [num_pages, page_size, h_k, 1, storage_d/dv].
            k_layout = cute.make_layout(
                (self.page_size, d, ((h_r, h_k), num_pages)),
                stride=(
                    h_k * storage_d,
                    1,
                    ((0, storage_d), self.page_size * h_k * storage_d),
                ),
            )
            v_layout = cute.make_layout(
                (dv, self.page_size, ((h_r, h_k), num_pages)),
                stride=(
                    1,
                    h_k * storage_dv,
                    ((0, storage_dv), self.page_size * h_k * storage_dv),
                ),
            )
        return k_layout, v_layout

    def _transpose_page_table(self, page_table):
        """View the page table as internal ``[max_pages, batch]``."""
        # Externally [batch, max_pages], internally [max_pages, batch].
        return cute.make_tensor(
            page_table.iterator,
            cute.make_layout(
                (page_table.shape[1], page_table.shape[0]),
                stride=(page_table.stride[1], page_table.stride[0]),
            ),
        )

    def _partition_paged_kv(
        self,
        tma_atom_k,
        tma_atom_v,
        mK_kdl_,
        mV_dkl_,
        sK_for_tma,
        sV_for_tma,
        curr_block_coord,
    ):
        """TMA-partition paged K/V around the page-granular atoms."""
        head_coord = curr_block_coord[2][0]
        gK_kdl = cute.local_tile(
            mK_kdl_,
            (self.page_size, self.qk_mma_tiler[2]),
            coord=(0, None, (head_coord, None)),
        )
        tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
            tma_atom_k,
            0,  # no multicast
            cute.make_layout(1),
            sK_for_tma,
            cute.group_modes(gK_kdl, 0, 2),
        )
        gV_dkl = cute.local_tile(
            mV_dkl_,
            (self.pv_mma_tiler[1], self.page_size),
            coord=(None, 0, (head_coord, None)),
        )
        tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(
            tma_atom_v,
            0,  # no multicast
            cute.make_layout(1),
            sV_for_tma,
            cute.group_modes(gV_dkl, 0, 2),
        )
        return tKsK, tKgK, tVsV, tVgV

    @cute.jit
    def _prefix_window_plan(
        self, cum_seqlen_k, kv_len_tensor, prefix_tokens_k, batch_coord
    ) -> Tuple:
        """Tile plan of one sequence of fixed window attention with a prefix sequence."""
        return fmha_utils.PrefixWindowKv.plan(
            prefix_tokens_k,
            cum_seqlen_k[batch_coord],
            kv_len_tensor[batch_coord],
            self.cta_tiler[1],
        )

    @cute.jit
    def _kv_loop_steps(
        self,
        curr_block_coord,
        seqlen_q,
        seqlen_k,
        window_size_left,
        window_size_right,
        batch_coord,
        cum_seqlen_k,
        kv_len_tensor,
        prefix_tokens_k,
    ) -> Int32:
        """Number of KV tiles the work tile visits."""
        steps = Int32(0)
        if cutlass.const_expr(self.prefix_window):
            steps = self._prefix_window_plan(
                cum_seqlen_k, kv_len_tensor, prefix_tokens_k, batch_coord
            )[1]
        else:
            steps = fmha_utils.FusedMask.get_trip_count(
                self.mask_type,
                curr_block_coord,
                self.cta_tiler,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
            )
        return steps

    @cute.jit
    def _run_page_table_prefetch(
        self,
        tile_sched_params,
        cum_seqlen_q,
        cum_seqlen_k,
        kv_len_tensor,
        prefix_tokens_k,
        mQ_qdl: cute.Tensor,
        mK_kdl: cute.Tensor,
        mPageTable: cute.Tensor,
        window_size_left,
        window_size_right,
        sPageIds: cute.Tensor,
        load_pt_pipeline,
        clc_pipeline,
        clc_response_ptr,
        clc_consumer_state,
        tidx: Int32,
    ) -> None:
        """Prefetch each KV tile's page IDs and reserve CLC work indices."""
        cute.arch.griddepcontrol_wait()
        load_pt_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.load_pt_stage
        )
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS),
            Int32,
            num_bits_per_copy=Int32.width,
        )
        sPageIds_for_copy = cute.flat_divide(sPageIds, (1,))
        lane_idx = tidx % self.threads_per_warp
        load_pt_tile_sched = fmha_utils.create_fmha_static_tile_scheduler(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            cum_seqlen_q,
            self.cta_tiler[0],
        )
        load_pt_work_tile = load_pt_tile_sched.initial_work_tile_info()
        clc_producer_state = None
        if cutlass.const_expr(self.use_clc_scheduler):
            clc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.ProducerConsumer, self.clc_stage
            )
        while load_pt_work_tile.is_valid_tile:
            curr_block_coord = load_pt_work_tile.tile_idx
            batch_coord = curr_block_coord[2][1]
            continue_cond = False
            seqlen_q = mQ_qdl.shape[0]
            seqlen_k = mK_kdl.shape[0]
            if cutlass.const_expr(cum_seqlen_q is not None):
                cuseqlen_q = cum_seqlen_q[batch_coord]
                seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                continue_cond = (
                    not fmha_utils.FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.cta_tiler[0],
                        curr_block_coord[0],
                        seqlen_q,
                    )
                )
            if not continue_cond:
                if cutlass.const_expr(self.prefix_window):
                    seqlen_k = prefix_tokens_k + kv_len_tensor[batch_coord]
                if cutlass.const_expr(cum_seqlen_k is not None and not self.prefix_window):
                    seqlen_k = cum_seqlen_k[batch_coord + 1] - cum_seqlen_k[batch_coord]
                continue_cond = seqlen_k <= 0
            if not continue_cond:
                mPageTable_batch = mPageTable[None, batch_coord]
                if cutlass.const_expr(self.prefix_window):
                    # One logical address space for every sequence.
                    mPageTable_batch = mPageTable[None, 0]
                mPageTable_for_copy = cute.flat_divide(mPageTable_batch, (1,))
                seqlen_kv_loop_start = fmha_utils.FusedMask.get_trip_start(
                    self.mask_type,
                    curr_block_coord,
                    self.cta_tiler,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                )
                seqlen_kv_loop_steps = self._kv_loop_steps(
                    curr_block_coord,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                    batch_coord,
                    cum_seqlen_k,
                    kv_len_tensor,
                    prefix_tokens_k,
                )
                last_logical_page = cute.ceil_div(seqlen_k, self.page_size) - 1
                plan = None
                if cutlass.const_expr(self.prefix_window):
                    # Page ids of every tile in the plan's virtual order. Only the pages of
                    # the last visible position are valid to read; the rest repeat it.
                    plan = self._prefix_window_plan(
                        cum_seqlen_k, kv_len_tensor, prefix_tokens_k, batch_coord
                    )
                    last_logical_page = (
                        cute.ceil_div(max(plan[11], max(plan[9], 1)), self.page_size) - 1
                    )
                kv_coord = seqlen_kv_loop_start
                for _ in cutlass.range(0, seqlen_kv_loop_steps, 1, unroll=1):
                    load_pt_pipeline.producer_acquire(load_pt_producer_state)
                    if lane_idx < self.pages_per_kv_tile:
                        tile = kv_coord
                        if cutlass.const_expr(self.prefix_window):
                            tile = fmha_utils.PrefixWindowKv.tile_of(plan, kv_coord)
                        logical_page = tile * self.pages_per_kv_tile + lane_idx
                        # Full K/V tiles keep a fixed TMA transaction
                        # count. Repeating the final finite page is safe
                        # because logical masking removes tail tokens.
                        if logical_page > last_logical_page:
                            logical_page = last_logical_page
                        cute.copy(
                            atom_async_copy,
                            mPageTable_for_copy[None, logical_page],
                            sPageIds_for_copy[
                                None,
                                lane_idx,
                                load_pt_producer_state.index,
                            ],
                        )
                    load_pt_pipeline.producer_commit(load_pt_producer_state)
                    load_pt_producer_state.advance()
                    kv_coord += 1
            # Reserve the next work index only after the current work's
            # page IDs are in flight. Keeping the query late avoids
            # extending the response lifetime across page-table setup.
            if cutlass.const_expr(self.use_clc_scheduler):
                clc_pipeline.producer_acquire(clc_producer_state)
                stage_response_ptr = clc_response_ptr + clc_producer_state.index
                clc_mbar_ptr = clc_pipeline.producer_get_barrier(clc_producer_state)
                with cute.arch.elect_one():
                    cute.arch.issue_clc_query(
                        clc_mbar_ptr,
                        stage_response_ptr,
                        multicast=False,
                    )
                clc_producer_state.advance()
                (
                    load_pt_work_tile,
                    clc_consumer_state,
                ) = self._consume_next_clc_work(
                    clc_pipeline,
                    clc_response_ptr,
                    clc_consumer_state,
                )
            else:
                load_pt_tile_sched.advance_to_next_work()
                load_pt_work_tile = load_pt_tile_sched.get_current_work()
        if cutlass.const_expr(self.use_clc_scheduler):
            clc_pipeline.producer_tail(clc_producer_state)
        load_pt_pipeline.producer_tail(load_pt_producer_state)

    def _make_paged_k_tma_atom(self, tma_load_op, k, k_smem_layout_staged):
        """Build the page-granular K TMA atom (page_size x K-phase boxes)."""
        k_smem_layout_mk = cute.composition(
            k_smem_layout_staged,
            cute.make_layout(
                (
                    self.qk_mma_tiler[1],
                    self.qk_mma_tiler[2],
                    self.kv_stage,
                )
            ),
        )
        k_smem_layout_for_tma = cute.tiled_divide(
            k_smem_layout_mk, (self.page_size, self.qk_mma_tiler[2])
        )
        k_smem_layout_for_tma = cute.select(k_smem_layout_for_tma, mode=[0, 1, 3])
        tma_atom_k, tma_tensor_k = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_load_op,
            k,
            k_smem_layout_for_tma[0],
            (self.page_size, self.qk_mma_tiler[2]),
        )
        return tma_atom_k, tma_tensor_k, k_smem_layout_for_tma

    def _make_paged_v_tma_atom(self, tma_load_op, v, v_smem_layout_staged):
        """Build the page-granular V TMA atom (D x page_size boxes)."""
        v_smem_layout_mk = cute.composition(
            v_smem_layout_staged,
            cute.make_layout(
                (
                    self.pv_mma_tiler[1],
                    self.pv_mma_tiler[2],
                    self.kv_stage,
                )
            ),
        )
        v_smem_layout_for_tma = cute.tiled_divide(
            v_smem_layout_mk, (self.pv_mma_tiler[1], self.page_size)
        )
        v_smem_layout_for_tma = cute.select(v_smem_layout_for_tma, mode=[0, 2, 3])
        tma_atom_v, tma_tensor_v = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_load_op,
            v,
            v_smem_layout_for_tma[0],
            (self.pv_mma_tiler[1], self.page_size),
        )
        return tma_atom_v, tma_tensor_v, v_smem_layout_for_tma

    @cute.jit
    def _copy_k_tile(
        self,
        tma_atom_k: cute.CopyAtom,
        tKgK: cute.Tensor,
        tKsK: cute.Tensor,
        page_ids: Optional[cute.Tensor],
        page_id_stage: Int32,
        kv_coord: Int32,
        stage: Int32,
        tma_bar_ptr: cute.Pointer,
    ) -> None:
        """Copy one K tile, gathering its physical pages when KV is paged."""
        if cutlass.const_expr(self.page_size is not None):
            for page_idx in cutlass.range_constexpr(self.pages_per_kv_tile):
                physical_page = page_ids[page_idx, page_id_stage]
                cute.copy(
                    tma_atom_k,
                    tKgK[None, 0, physical_page],
                    tKsK[None, page_idx, stage],
                    tma_bar_ptr=tma_bar_ptr,
                )
        else:
            cute.copy(
                tma_atom_k,
                tKgK[None, kv_coord],
                tKsK[None, stage],
                tma_bar_ptr=tma_bar_ptr,
            )

    @cute.jit
    def _copy_v_tile(
        self,
        tma_atom_v: cute.CopyAtom,
        tVgV: cute.Tensor,
        tVsV: cute.Tensor,
        page_ids: Optional[cute.Tensor],
        page_id_stage: Int32,
        kv_coord: Int32,
        stage: Int32,
        tma_bar_ptr: cute.Pointer,
    ) -> None:
        """Copy one V tile, gathering its physical pages when KV is paged."""
        if cutlass.const_expr(self.page_size is not None):
            for page_idx in cutlass.range_constexpr(self.pages_per_kv_tile):
                physical_page = page_ids[page_idx, page_id_stage]
                cute.copy(
                    tma_atom_v,
                    tVgV[None, 0, physical_page],
                    tVsV[None, page_idx, stage],
                    tma_bar_ptr=tma_bar_ptr,
                )
        else:
            cute.copy(
                tma_atom_v,
                tVgV[None, kv_coord],
                tVsV[None, stage],
                tma_bar_ptr=tma_bar_ptr,
            )

    @cute.jit
    def _store_o_tile_direct(
        self,
        tTMrO: cute.Tensor,
        tTMEM_LOADdO_i: cute.Tensor,
        row_idx: Int32,
        seqlen_q: Int32,
    ) -> None:
        """Bounds-checked register-to-global store of one correction tile."""
        tDMrO = cute.make_rmem_tensor(tTMrO.shape, self.o_dtype)
        tDMrO.store(tTMrO.load().to(self.o_dtype))
        if row_idx < seqlen_q:
            cute.autovec_copy(tDMrO, tTMEM_LOADdO_i)

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_kdl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_dkl: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        mO_qdl: cute.Tensor,
        ragged_tma_o_desc: Optional[cutlass.GridConstant[cuda_exp.TensorMap]],
        mO: cute.Tensor,
        mO_sf: Optional[cute.Tensor],
        cum_seqlen_q: Optional[cute.Tensor],
        cum_seqlen_k: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor],
        mSink: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        scale_softmax_log2: Float32,
        scale_softmax: Float32,
        scale_output: Float32,
        scale_softmax_log2_tensor: Optional[cute.Tensor],
        scale_softmax_tensor: Optional[cute.Tensor],
        scale_output_tensor: Optional[cute.Tensor],
        kv_len_tensor: Optional[cute.Tensor],
        prefix_tokens_k: Optional[Int32],
        o_sf_scale: Optional[Float32],
        o_sf_start_index: Optional[Int32],
        skip_softmax_threshold_log2: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_for_tma: cute.ComposedLayout,
        p_tmem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_for_tma: cute.ComposedLayout,
        o_smem_layout_staged: cute.ComposedLayout,
        arch_args: Tuple,
        skip_softmax_count: Optional[cute.Tensor],
        total_softmax_count: Optional[cute.Tensor],
        tile_sched_params: fmha_utils.FmhaStaticTileSchedulerParams,
        dynamic_work_counter: Optional[cute.Tensor],
    ):
        """The device kernel implementation of the Fused Multi-Head Attention.

        This kernel coordinates multiple specialized warps to perform different phases of the FMHA computation:
        1. Load warp: Loads Q, K, V data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Softmax warps: Compute softmax normalization on attention scores
        4. Correction warps: Apply adjustments to intermediate results
        5. Epilogue warp: Handles final output transformation and storage

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases, and optional attention masking.

        :param qk_tiled_mma: Tiled MMA for Q*K^T
        :type qk_tiled_mma: cute.TiledMma
        :param pv_tiled_mma: Tiled MMA for P*V
        :type pv_tiled_mma: cute.TiledMma
        :param tma_atom_q: TMA copy atom for query tensor
        :type tma_atom_q: cute.CopyAtom
        :param mQ_qdl: Partitioned query tensor
        :type mQ_qdl: cute.Tensor
        :param tma_atom_k: TMA copy atom for key tensor
        :type tma_atom_k: cute.CopyAtom
        :param mK_kdl: Partitioned key tensor
        :type mK_kdl: cute.Tensor
        :param tma_atom_v: TMA copy atom for value tensor
        :type tma_atom_v: cute.CopyAtom
        :param mV_dkl: Partitioned value tensor
        :type mV_dkl: cute.Tensor
        :param tma_atom_o: TMA copy atom for output tensor
        :type tma_atom_o: cute.CopyAtom
        :param mO_qdl: Partitioned output tensor
        :type mO_qdl: cute.Tensor
        :param ragged_tma_o_desc: Synthetic-dimension TMA descriptor for
                                  arbitrary packed Q tails, or ``None``.
        :type ragged_tma_o_desc: Optional[GridConstant[TensorMap]]
        :param mO_sf: NVFP4 Layout128x4 E4M3 scale-factor storage, or ``None``
        :param mPageTable: Internal ``[max_pages, batch]`` page-table view for
                           paged K/V, or ``None``
        :param o_sf_scale: NVFP4 global normalization constant, or ``None``
        :param o_sf_start_index: Starting token row in ``mO_sf``, or ``None``
        :param k_smem_layout_for_tma: Page-granular K TMA smem view (equals
                                      ``k_smem_layout_staged`` for dense K/V)
        :param v_smem_layout_for_tma: Page-granular V TMA smem view (equals
                                      ``v_smem_layout_staged`` for dense K/V)
        :param dynamic_work_counter: ``Int32[2]`` software work-index queue
                                     state, or ``None``
        :param mO: Non-partitioned output tensor
        :type mO: cute.Tensor
        :param scale_softmax_log2: The log2 scale factor for softmax
        :type scale_softmax_log2: Float32
        :param scale_output: The scale factor for the output
        :type scale_output: Float32
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[Int32]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[Int32]
        :param q_smem_layout_staged: Shared memory layout for query tensor
        :type q_smem_layout_staged: cute.ComposedLayout
        :param k_smem_layout_staged: Shared memory layout for key tensor
        :type k_smem_layout_staged: cute.ComposedLayout
        :param p_tmem_layout_staged: Tensor memory layout for probability matrix
        :type p_tmem_layout_staged: cute.ComposedLayout
        :param v_smem_layout_staged: Shared memory layout for value tensor
        :type v_smem_layout_staged: cute.ComposedLayout
        :param o_smem_layout_staged: Shared memory layout for output tensor
        :type o_smem_layout_staged: cute.ComposedLayout
        :param tile_sched_params: Scheduling parameters for work distribution
        :type tile_sched_params: fmha_utils.FmhaStaticTileSchedulerParams
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        # A device scalar replaces the host value, so a caller whose scale is computed on
        # the device (quantization global scales) needs no host sync.
        if cutlass.const_expr(scale_softmax_tensor is not None):
            scale_softmax = scale_softmax_tensor[0]
        if cutlass.const_expr(scale_softmax_log2_tensor is not None):
            scale_softmax_log2 = scale_softmax_log2_tensor[0]
        if cutlass.const_expr(scale_output_tensor is not None):
            scale_output = scale_output_tensor[0]

        #
        # Prefetch tma desc
        #
        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            if cutlass.const_expr(self.use_tma_store):
                if cutlass.const_expr(self.use_ragged_tma_store):
                    self._prefetch_ragged_o_descriptor(ragged_tma_o_desc)
                else:
                    cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_o)

        # Alloc
        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.q_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_q_bytes,
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        load_kv_producer, load_kv_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.kv_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_k_bytes,
            barrier_storage=storage.load_kv_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        load_pt_pipeline = None
        if cutlass.const_expr(self.page_size is not None):
            load_pt_pipeline = pipeline.PipelineCpAsync.create(
                num_stages=self.load_pt_stage,
                producer_group=make_thread_cooperative_group(
                    self.threads_per_warp * len([self.empty_warp_id])
                ),
                consumer_group=make_thread_cooperative_group(
                    self.threads_per_warp * len([self.load_warp_id])
                ),
                barrier_storage=storage.load_pt_mbar_ptr.data_ptr(),
                defer_sync=True,
            )
        mma_s0_pipeline = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_softmax_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax0_warp_ids)
            ),
            barrier_storage=storage.mma_s_mbar_ptr.data_ptr(),
            defer_sync=True,
        )
        mma_s0_producer, mma_s0_consumer = mma_s0_pipeline.make_participants()
        mma_s1_pipeline = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_softmax_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax1_warp_ids)
            ),
            barrier_storage=storage.mma_s_mbar_ptr.data_ptr() + self.mma_softmax_stage * 2,
            defer_sync=True,
        )
        mma_s1_producer, mma_s1_consumer = mma_s1_pipeline.make_participants()
        p0_mma_pipeline = pipeline.PipelineAsyncUmma.create(
            num_stages=self.p_mma_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax0_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.p_mma_mbar_ptr.data_ptr(),
            defer_sync=True,
        )
        p0_mma_producer, p0_mma_consumer = p0_mma_pipeline.make_participants()
        p1_mma_pipeline = pipeline.PipelineAsyncUmma.create(
            num_stages=self.p_mma_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax1_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.p_mma_mbar_ptr.data_ptr() + self.p_mma_stage * 2,
            defer_sync=True,
        )
        p1_mma_producer, p1_mma_consumer = p1_mma_pipeline.make_participants()
        s0_corr_pipeline = pipeline.PipelineAsync.create(
            num_stages=self.softmax_corr_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len((*self.softmax0_warp_ids, self.mma_warp_id))
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            barrier_storage=storage.s_corr_mbar_ptr.data_ptr(),
            defer_sync=True,
        )
        s0_corr_producer, s0_corr_consumer = s0_corr_pipeline.make_participants()
        s1_corr_pipeline = pipeline.PipelineAsync.create(
            num_stages=self.softmax_corr_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len((*self.softmax1_warp_ids, self.mma_warp_id))
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            barrier_storage=storage.s_corr_mbar_ptr.data_ptr() + self.softmax_corr_stage * 2,
            defer_sync=True,
        )
        s1_corr_producer, s1_corr_consumer = s1_corr_pipeline.make_participants()
        corr_epi_producer, corr_epi_consumer = pipeline.PipelineAsync.create(
            num_stages=self.epi_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len([self.epilogue_warp_id])
            ),
            barrier_storage=storage.corr_epi_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        mma_corr_producer, mma_corr_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_corr_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            barrier_storage=storage.mma_corr_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        s0_p1_inplace_pipeline = pipeline.PipelineAsync.create(
            num_stages=self.p_mma_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax0_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax1_warp_ids)
            ),
            barrier_storage=storage.inplace_mbar_ptr.data_ptr(),
            defer_sync=True,
        )
        s0_p1_inplace_producer, s0_p1_inplace_consumer = s0_p1_inplace_pipeline.make_participants()
        s1_p0_inplace_pipeline = pipeline.PipelineAsync.create(
            num_stages=self.p_mma_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax0_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax1_warp_ids)
            ),
            barrier_storage=storage.inplace_mbar_ptr.data_ptr() + self.p_mma_stage * 2,
            defer_sync=True,
        )
        s1_p0_inplace_producer, s1_p0_inplace_consumer = s1_p0_inplace_pipeline.make_participants()
        clc_pipeline = None
        clc_response_ptr = None
        clc_consumer_state = None
        dynamic_work_pipeline = None
        dynamic_work_idx_ptr = None
        dynamic_work_consumer_state = None
        if cutlass.const_expr(self.use_clc_scheduler):
            clc_pipeline = pipeline.PipelineClcFetchAsync.create(
                num_stages=self.clc_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, self.threads_per_cta
                ),
                tx_count=16,
                barrier_storage=storage.clc_pipeline_mbar.data_ptr(),
                cta_layout_vmnk=(1, 1, 1, 1),
                defer_sync=True,
            )
            clc_response_ptr = storage.clc_response.data_ptr()
            clc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.clc_stage
            )
        elif cutlass.const_expr(self.use_dynamic_work_scheduler):
            dynamic_work_pipeline = pipeline.PipelineAsync.create(
                num_stages=self.dynamic_work_stage,
                producer_group=make_thread_cooperative_group(self.threads_per_warp),
                consumer_group=make_thread_cooperative_group(
                    self.threads_per_cta - self.threads_per_warp
                ),
                barrier_storage=storage.clc_pipeline_mbar.data_ptr(),
                defer_sync=True,
            )
            dynamic_work_idx_ptr = cute.recast_ptr(storage.clc_response.data_ptr(), dtype=Int32)
            dynamic_work_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.dynamic_work_stage
            )
        tmem = cutlass.memory.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            # Correction warp is the last one that accesses tmem
            allocator_warp_id=self.correction_warp_ids[0],
            arch=self.arch_str,
        )
        pipeline_init_arrive(is_relaxed=True)

        #  Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        sK = storage.sK.get_tensor(k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner)
        sK_for_tma = sK
        if cutlass.const_expr(self.page_size is not None):
            sK_for_tma = cute.make_tensor(sK.iterator, k_smem_layout_for_tma.outer)
        # (MMA, MMA_K, MMA_D, PIPE)
        # Reuse k's smem buffer for v. Recast element type so MMA descriptor matches v_dtype.
        sV_ptr = cute.recast_ptr(
            cute.recast_ptr(sK.iterator, dtype=self.v_dtype),
            v_smem_layout_staged.inner,
        )
        sV = cute.make_tensor(sV_ptr, v_smem_layout_staged.outer)
        sV_for_tma = sV
        if cutlass.const_expr(self.page_size is not None):
            sV_for_tma = cute.make_tensor(sV.iterator, v_smem_layout_for_tma.outer)
        sPageIds = None
        if cutlass.const_expr(self.page_size is not None):
            sPageIds = storage.sPageIds.get_tensor(
                cute.make_layout((self.threads_per_warp, self.load_pt_stage))
            )
        arch_smem = self._make_arch_smem_views(storage, arch_args)
        if cutlass.const_expr(self.use_tma_store):
            sO = storage.sO.get_tensor(
                o_smem_layout_staged.outer, swizzle=o_smem_layout_staged.inner
            )
        warp_wants_skip_softmax_exchange = storage.warp_wants_skip_softmax_exchange.get_tensor(
            cute.make_layout((4, 2))
        )

        # Views are built from a copy so later ACCUMULATE-field mutation of
        # the MMA handles cannot affect them.
        if cutlass.const_expr(self.enable_q_mixed_residency):
            # Keep fragment/view construction independent from the mutable MMA
            # handles whose ACCUMULATE fields change in mma_qk_mixed().
            qk_tiled_mma_view = self._make_qk_tiled_mma(tcgen05.CtaGroup.ONE)
        else:
            qk_tiled_mma_view = qk_tiled_mma
        qk_thr_mma = qk_tiled_mma_view.get_slice(0)  # default 1sm
        pv_thr_mma = pv_tiled_mma.get_slice(0)  # default 1sm
        tSrQ = qk_thr_mma.make_fragment_A(sQ)
        tSrK = qk_thr_mma.make_fragment_B(sK)
        tOrV = pv_thr_mma.make_fragment_B(sV)
        # Runs in the common prologue so it may synchronize all threads.
        self._init_arch_smem_operands(arch_smem, warp_idx, tidx)

        def make_tmem_tensors(
            self,
            qk_thr_mma: cute.TiledMma,
            pv_thr_mma: cute.TiledMma,
            p_tmem_layout_staged: cute.Layout,
            tmem_ptr: cute.Pointer,
        ):
            qk_acc_shape = qk_thr_mma.partition_shape_C(
                (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
            )
            tStS_fake = qk_thr_mma.make_fragment_C(qk_acc_shape)
            tStS = cute.make_tensor(tmem_ptr + self.tmem_s0_offset, tStS_fake.layout)
            pv_acc_shape = pv_thr_mma.partition_shape_C(
                (self.pv_mma_tiler[0], self.pv_mma_tiler[1])
            )
            tOtO = pv_thr_mma.make_fragment_C(pv_acc_shape)
            tStS0 = cute.make_tensor(tmem_ptr + self.tmem_s0_offset, tStS.layout)
            tStS1 = cute.make_tensor(tmem_ptr + self.tmem_s1_offset, tStS.layout)
            tOtO0 = cute.make_tensor(tmem_ptr + self.tmem_o0_offset, tOtO.layout)
            tOtO1 = cute.make_tensor(tmem_ptr + self.tmem_o1_offset, tOtO.layout)
            tP = cute.make_tensor(tStS.iterator, p_tmem_layout_staged.outer)
            tOrP = pv_thr_mma.make_fragment_A(tP)[None, None, None, 0]
            tOrP0 = cute.make_tensor(
                cute.recast_ptr(
                    tmem_ptr + self.tmem_p0_offset,
                    dtype=tOrP.dtype,
                ),
                tOrP.layout,
            )
            tOrP1 = cute.make_tensor(
                cute.recast_ptr(
                    tmem_ptr + self.tmem_p1_offset,
                    dtype=tOrP.dtype,
                ),
                tOrP.layout,
            )
            return tStS, tStS0, tStS1, tOtO0, tOtO1, tOrP0, tOrP1

        def make_pipeline_view(prototype, barrier_storage: cute.Pointer):
            """Rebase an initialized pipeline without reinitializing mbarriers."""
            full = prototype.sync_object_full
            empty = prototype.sync_object_empty
            assert isinstance(full, pipeline.MbarrierArray)
            assert isinstance(empty, pipeline.MbarrierArray)
            full_view = full.__new_from_mlir_values__([barrier_storage])
            empty_view = empty.__new_from_mlir_values__([barrier_storage + prototype.num_stages])
            return replace(
                prototype,
                sync_object_full=full_view,
                sync_object_empty=empty_view,
            )

        tile_sched = fmha_utils.create_fmha_static_tile_scheduler(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            cum_seqlen_q,
            self.cta_tiler[0],
        )
        if cutlass.const_expr(self.use_dynamic_work_scheduler):
            # Consumer roles obtain their first tile from the software queue.
            # Avoid an unused compact prefix search before that first response.
            work_tile = fmha_utils.WorkTileInfo(
                (Int32(0), 0, (Int32(0), Int32(0))),
                cutlass.Boolean(False),
            )
        else:
            work_tile = tile_sched.initial_work_tile_info()
        pipeline_init_wait()
        softmax_fn = partial(
            self.softmax,
            qk_thr_mma=qk_thr_mma,
            value_args=(
                mK_kdl.shape[0],
                mQ_qdl.shape[0],
                scale_softmax_log2,
                skip_softmax_threshold_log2,
            ),
            mask_args=(window_size_left, window_size_right),
            sched_args=(
                tile_sched,
                work_tile,
                clc_pipeline,
                clc_response_ptr,
                clc_consumer_state,
                dynamic_work_pipeline,
                dynamic_work_idx_ptr,
                dynamic_work_consumer_state,
            ),
        )
        # ///////////////////////////////////////////////////////////////////////////////
        #  WORK SCHEDULER / PAGE TABLE PREFETCH
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.empty_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            if cutlass.const_expr(self.use_dynamic_work_scheduler):
                self._produce_dynamic_work_idxs(
                    dynamic_work_pipeline,
                    dynamic_work_idx_ptr,
                    dynamic_work_counter,
                    tile_sched,
                )
            if cutlass.const_expr(self.page_size is not None):
                self._run_page_table_prefetch(
                    tile_sched_params,
                    cum_seqlen_q,
                    cum_seqlen_k,
                    kv_len_tensor,
                    prefix_tokens_k,
                    mQ_qdl,
                    mK_kdl,
                    mPageTable,
                    window_size_left,
                    window_size_right,
                    sPageIds,
                    load_pt_pipeline,
                    clc_pipeline,
                    clc_response_ptr,
                    clc_consumer_state,
                    tidx,
                )

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            cute.arch.griddepcontrol_wait()
            load_pt_consumer_state = None
            load_pt_release_state = None
            if cutlass.const_expr(self.page_size is not None):
                load_pt_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.load_pt_stage
                )
                # Acquisition advances at K_i, while release intentionally
                # trails until V_i has issued from the same page-ID stage.
                load_pt_release_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.load_pt_stage
                )
            if cutlass.const_expr(self.use_dynamic_work_scheduler):
                work_tile, dynamic_work_consumer_state = self._consume_next_dynamic_work(
                    dynamic_work_pipeline,
                    dynamic_work_idx_ptr,
                    dynamic_work_consumer_state,
                )
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                batch_coord = curr_block_coord[2][1]
                continue_cond = False
                cuseqlen_q = Int32(0)
                seqlen_q = mQ_qdl.shape[0]
                seqlen_k = mK_kdl.shape[0]
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = (
                        not fmha_utils.FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                            self.cta_tiler[0],
                            curr_block_coord[0],
                            seqlen_q,
                        )
                    )
                if not continue_cond:
                    if cutlass.const_expr(self.prefix_window):
                        seqlen_k = prefix_tokens_k + kv_len_tensor[batch_coord]
                    if cutlass.const_expr(cum_seqlen_k is not None and not self.prefix_window):
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cum_seqlen_k[batch_coord]
                    continue_cond = seqlen_k <= 0
                if not continue_cond:
                    mQ_qdl_ = mQ_qdl
                    mK_kdl_ = mK_kdl
                    mV_dkl_ = mV_dkl
                    if cutlass.const_expr(cum_seqlen_q is not None):
                        mQ_qdl_ = cute.domain_offset(
                            (cum_seqlen_q[batch_coord], 0, ((0, 0), 0)), mQ_qdl
                        )
                    if cutlass.const_expr(cum_seqlen_k is not None and self.page_size is None):
                        mK_kdl_ = cute.domain_offset(
                            (cum_seqlen_k[batch_coord], 0, ((0, 0), 0)), mK_kdl
                        )
                        mV_dkl_ = cute.domain_offset(
                            (0, cum_seqlen_k[batch_coord], ((0, 0), 0)), mV_dkl
                        )
                    # Local tile partition global tensors
                    gQ_qdl = cute.flat_divide(mQ_qdl_, cute.select(self.qk_mma_tiler, mode=[0, 2]))
                    tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
                    tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_q,
                        0,  # no multicast
                        cute.make_layout(1),
                        cute.group_modes(sQ, 0, 3),
                        cute.group_modes(tSgQ_qdl, 0, 3),
                    )
                    tQgQ = tQgQ_qdl[None, None, 0, curr_block_coord[2]]
                    if cutlass.const_expr(self.page_size is not None):
                        tKsK, tKgK, tVsV, tVgV = self._partition_paged_kv(
                            tma_atom_k,
                            tma_atom_v,
                            mK_kdl_,
                            mV_dkl_,
                            sK_for_tma,
                            sV_for_tma,
                            curr_block_coord,
                        )
                    else:
                        gK_kdl = cute.flat_divide(
                            mK_kdl_, cute.select(self.qk_mma_tiler, mode=[1, 2])
                        )
                        tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
                        tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_k,
                            0,  # no multicast
                            cute.make_layout(1),
                            cute.group_modes(sK, 0, 3),
                            cute.group_modes(tSgK_kdl, 0, 3),
                        )
                        tKgK = tKgK_kdl[None, None, 0, curr_block_coord[2]]
                        gV_dkl = cute.flat_divide(
                            mV_dkl_, cute.select(self.pv_mma_tiler, mode=[1, 2])
                        )
                        tSgV_dkl = pv_thr_mma.partition_B(gV_dkl)
                        tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_v,
                            0,  # no multicast
                            cute.make_layout(1),
                            cute.group_modes(sV, 0, 3),
                            cute.group_modes(tSgV_dkl, 0, 3),
                        )
                        tVgV = tVgV_dkl[None, 0, None, curr_block_coord[2]]
                    seqlen_kv_loop_start = fmha_utils.FusedMask.get_trip_start(
                        self.mask_type,
                        curr_block_coord,
                        self.cta_tiler,
                        seqlen_q,
                        seqlen_k,
                        window_size_left,
                    )
                    # Q0
                    q0_coord = 2 * curr_block_coord[0]
                    q0_handle = load_q_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_q,
                        tQgQ[None, q0_coord],
                        tQsQ[None, q0_handle.index],
                        tma_bar_ptr=q0_handle.barrier,
                    )
                    seqlen_kv_loop_steps = self._kv_loop_steps(
                        curr_block_coord,
                        seqlen_q,
                        seqlen_k,
                        window_size_left,
                        window_size_right,
                        batch_coord,
                        cum_seqlen_k,
                        kv_len_tensor,
                        prefix_tokens_k,
                    )
                    # K0
                    kv_coord = seqlen_kv_loop_start
                    page_id_stage = Int32(0)
                    if cutlass.const_expr(self.page_size is not None):
                        load_pt_pipeline.consumer_wait(load_pt_consumer_state)
                        page_id_stage = load_pt_consumer_state.index
                        load_pt_consumer_state.advance()
                    k_handle = load_kv_producer.acquire_and_advance()
                    self._copy_k_tile(
                        tma_atom_k,
                        tKgK,
                        tKsK,
                        sPageIds,
                        page_id_stage,
                        kv_coord,
                        k_handle.index,
                        k_handle.barrier,
                    )
                    # Q1
                    q1_coord = q0_coord + 1
                    q1_handle = load_q_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_q,
                        tQgQ[None, q1_coord],
                        tQsQ[None, q1_handle.index],
                        tma_bar_ptr=q1_handle.barrier,
                    )
                    kv_coord += 1

                    for i in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
                        # Ki
                        if cutlass.const_expr(self.page_size is not None):
                            load_pt_pipeline.consumer_wait(load_pt_consumer_state)
                            page_id_stage = load_pt_consumer_state.index
                            load_pt_consumer_state.advance()
                        k_handle = load_kv_producer.acquire_and_advance()
                        self._copy_k_tile(
                            tma_atom_k,
                            tKgK,
                            tKsK,
                            sPageIds,
                            page_id_stage,
                            kv_coord,
                            k_handle.index,
                            k_handle.barrier,
                        )
                        # Vi-1
                        v_handle = load_kv_producer.acquire_and_advance(
                            expected_tx=self.tma_copy_v_bytes
                        )
                        self._copy_v_tile(
                            tma_atom_v,
                            tVgV,
                            tVsV,
                            sPageIds,
                            (
                                load_pt_release_state.index
                                if self.page_size is not None
                                else Int32(0)
                            ),
                            kv_coord - 1,
                            v_handle.index,
                            v_handle.barrier,
                        )
                        if cutlass.const_expr(self.page_size is not None):
                            load_pt_pipeline.consumer_release(load_pt_release_state)
                            load_pt_release_state.advance()
                        kv_coord += 1
                    # End of seqlen_kv loop
                    # Vi_end
                    v_handle = load_kv_producer.acquire_and_advance(
                        expected_tx=self.tma_copy_v_bytes
                    )
                    self._copy_v_tile(
                        tma_atom_v,
                        tVgV,
                        tVsV,
                        sPageIds,
                        (load_pt_release_state.index if self.page_size is not None else Int32(0)),
                        kv_coord - 1,
                        v_handle.index,
                        v_handle.barrier,
                    )
                    if cutlass.const_expr(self.page_size is not None):
                        load_pt_pipeline.consumer_release(load_pt_release_state)
                        load_pt_release_state.advance()
                # End of if not continue_cond
                if cutlass.const_expr(self.use_dynamic_work_scheduler):
                    work_tile, dynamic_work_consumer_state = self._consume_next_dynamic_work(
                        dynamic_work_pipeline,
                        dynamic_work_idx_ptr,
                        dynamic_work_consumer_state,
                    )
                elif cutlass.const_expr(self.use_clc_scheduler):
                    work_tile, clc_consumer_state = self._consume_next_clc_work(
                        clc_pipeline,
                        clc_response_ptr,
                        clc_consumer_state,
                    )
                else:
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
                # End of persistent scheduler loop
        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            tStS, tStS0, tStS1, tOtO0, tOtO1, tOrP0, tOrP1 = make_tmem_tensors(
                self, qk_thr_mma, pv_thr_mma, p_tmem_layout_staged, tmem_ptr
            )
            mma_arch_state = self._make_mma_warp_arch_state(arch_smem, arch_args, tmem_ptr)
            enable_skip_softmax = skip_softmax_threshold_log2 is not None
            tiled_tmem_load_v = None
            tTMEM_LOADtS_v0, tTMEM_LOADtS_v1 = None, None
            tTMEM_LOADrS_v0, tTMEM_LOADrS_v1 = None, None
            if cutlass.const_expr(enable_skip_softmax):
                cS = cute.make_identity_tensor(cute.select(self.qk_mma_tiler, mode=[0, 1]))
                tScS = qk_thr_mma.partition_C(cS)
                tStS_v = cute.composition(tStS, cute.make_layout((self.threads_per_warp, 1)))
                tScS_v = cute.composition(tScS, cute.make_layout((self.threads_per_warp, 1)))
                tmem_load_v_atom = cute.make_copy_atom(
                    tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(1)),
                    self.qk_acc_dtype,
                )
                thread_idx = tidx % self.threads_per_warp

                tiled_tmem_load_v = tcgen05.make_tmem_copy(tmem_load_v_atom, tStS_v)
                thr_tmem_load_v = tiled_tmem_load_v.get_slice(thread_idx)
                tTMEM_LOADtS_v = thr_tmem_load_v.partition_S(tStS_v)
                tTMEM_LOADcS_v = thr_tmem_load_v.partition_D(tScS_v)
                tTMEM_LOADrS_v0 = cute.make_rmem_tensor(tTMEM_LOADcS_v.shape, self.qk_acc_dtype)
                tTMEM_LOADrS_v1 = cute.make_rmem_tensor(tTMEM_LOADcS_v.shape, self.qk_acc_dtype)
                tTMEM_LOADtS_v0 = cute.make_tensor(
                    tTMEM_LOADtS_v.iterator + self.tmem_skip_softmax0_offset,
                    tTMEM_LOADtS_v.layout,
                )
                tTMEM_LOADtS_v1 = cute.make_tensor(
                    tTMEM_LOADtS_v.iterator + self.tmem_skip_softmax1_offset,
                    tTMEM_LOADtS_v.layout,
                )

            if cutlass.const_expr(self.use_dynamic_work_scheduler):
                work_tile, dynamic_work_consumer_state = self._consume_next_dynamic_work(
                    dynamic_work_pipeline,
                    dynamic_work_idx_ptr,
                    dynamic_work_consumer_state,
                )
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                batch_coord = curr_block_coord[2][1]
                continue_cond = False
                seqlen_q = mQ_qdl.shape[0]
                seqlen_k = mK_kdl.shape[0]
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = (
                        not fmha_utils.FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                            self.cta_tiler[0],
                            curr_block_coord[0],
                            seqlen_q,
                        )
                    )
                if not continue_cond:
                    if cutlass.const_expr(self.prefix_window):
                        seqlen_k = prefix_tokens_k + kv_len_tensor[batch_coord]
                    if cutlass.const_expr(cum_seqlen_k is not None and not self.prefix_window):
                        cuseqlen_k = cum_seqlen_k[batch_coord]
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                    continue_cond = seqlen_k <= 0
                if not continue_cond:
                    # Wait for Q0
                    q0_handle = load_q_consumer.wait_and_advance()
                    tSrQ0 = tSrQ[None, None, None, q0_handle.index]
                    self._stage_arch_q_operand(mma_arch_state, q0_handle.index, 0)
                    # Wait for K0
                    k_handle = load_kv_consumer.wait_and_advance()
                    tSrK0 = tSrK[None, None, None, k_handle.index]
                    # GEMM_QK00 (Q0 * K0 -> S0)
                    mma_s0_producer, s0_corr_producer = self._mma_qk_step(
                        qk_tiled_mma,
                        (tSrQ0, tSrK0, tStS0),
                        (mma_s0_producer, s0_corr_producer),
                        mma_arch_state,
                        0,
                        q0_handle.index,
                    )
                    # Wait for Q1
                    q1_handle = load_q_consumer.wait_and_advance()
                    tSrQ1 = tSrQ[None, None, None, q1_handle.index]
                    self._stage_arch_q_operand(mma_arch_state, q1_handle.index, 1)
                    # GEMM_QK10 (Q1 * K0 -> S1), K0 is ready in GEMM_QK00
                    mma_s1_producer, s1_corr_producer = self._mma_qk_step(
                        qk_tiled_mma,
                        (tSrQ1, tSrK0, tStS1),
                        (mma_s1_producer, s1_corr_producer),
                        mma_arch_state,
                        1,
                        q1_handle.index,
                    )
                    # Release K0
                    k_handle.release()
                    # Note: Q0 & Q1 are still needed in the seqlen_kv loop
                    # so we need to release them after the seqlen_kv loop
                    seqlen_kv_loop_steps = self._kv_loop_steps(
                        curr_block_coord,
                        seqlen_q,
                        seqlen_k,
                        window_size_left,
                        window_size_right,
                        batch_coord,
                        cum_seqlen_k,
                        kv_len_tensor,
                        prefix_tokens_k,
                    )
                    # O1 hasn't been accumulated yet, its first MMA calculation doesn't need to accumulate
                    pv_whether_acc = False
                    for i in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
                        # Wait for Ki
                        k_handle = load_kv_consumer.wait_and_advance()
                        tSrKi = tSrK[None, None, None, k_handle.index]
                        # GEMM_QK0i (Q0 * Ki -> S0)
                        mma_s0_producer, s0_corr_producer = self._mma_qk_step(
                            qk_tiled_mma,
                            (tSrQ0, tSrKi, tStS0),
                            (mma_s0_producer, s0_corr_producer),
                            mma_arch_state,
                            0,
                            q0_handle.index,
                        )
                        # Wait for Vi-1
                        v_handle = load_kv_consumer.wait_and_advance()
                        tOrVi = tOrV[None, None, None, v_handle.index]
                        # GEMM_PV0(i-1) (P0 * Vi-1 -> O0_partial)
                        mma_corr_producer, p0_mma_consumer = self._mma_pv_step(
                            pv_tiled_mma,
                            pv_whether_acc,
                            (tOrP0, tOrVi, tOtO0),
                            (mma_corr_producer, p0_mma_consumer),
                            (
                                enable_skip_softmax,
                                tiled_tmem_load_v,
                                tTMEM_LOADtS_v0,
                                tTMEM_LOADrS_v0,
                            ),
                            mma_arch_state,
                            0,
                        )
                        # GEMM_QK1i (Q1 * Ki -> S1)
                        mma_s1_producer, s1_corr_producer = self._mma_qk_step(
                            qk_tiled_mma,
                            (tSrQ1, tSrKi, tStS1),
                            (mma_s1_producer, s1_corr_producer),
                            mma_arch_state,
                            1,
                            q1_handle.index,
                        )
                        # Release Ki
                        k_handle.release()
                        # GEMM_PV1(i-1) (P1 * Vi-1 -> O1_partial)
                        mma_corr_producer, p1_mma_consumer = self._mma_pv_step(
                            pv_tiled_mma,
                            pv_whether_acc,
                            (tOrP1, tOrVi, tOtO1),
                            (mma_corr_producer, p1_mma_consumer),
                            (
                                enable_skip_softmax,
                                tiled_tmem_load_v,
                                tTMEM_LOADtS_v1,
                                tTMEM_LOADrS_v1,
                            ),
                            mma_arch_state,
                            1,
                        )
                        pv_whether_acc = True
                        # Release Vi-1
                        v_handle.release()
                    # End of seqlen_kv loop
                    # release Q0 & Q1
                    q0_handle.release()
                    q1_handle.release()
                    # Wait for Vi_end
                    v_handle = load_kv_consumer.wait_and_advance()
                    tOrVi = tOrV[None, None, None, v_handle.index]
                    # GEMM_PV0(i_end) (P0 * Vi_end -> O0)
                    mma_corr_producer, p0_mma_consumer = self._mma_pv_step(
                        pv_tiled_mma,
                        pv_whether_acc,
                        (tOrP0, tOrVi, tOtO0),
                        (mma_corr_producer, p0_mma_consumer),
                        (
                            enable_skip_softmax,
                            tiled_tmem_load_v,
                            tTMEM_LOADtS_v0,
                            tTMEM_LOADrS_v0,
                        ),
                        mma_arch_state,
                        0,
                    )
                    # GEMM_PV1(i_end) (P1 * Vi_end -> O1)
                    mma_corr_producer, p1_mma_consumer = self._mma_pv_step(
                        pv_tiled_mma,
                        pv_whether_acc,
                        (tOrP1, tOrVi, tOtO1),
                        (mma_corr_producer, p1_mma_consumer),
                        (
                            enable_skip_softmax,
                            tiled_tmem_load_v,
                            tTMEM_LOADtS_v1,
                            tTMEM_LOADrS_v1,
                        ),
                        mma_arch_state,
                        1,
                    )
                    # Release Vi_end
                    v_handle.release()
                    # Empty step for correction epilog
                    vec0_handle = s0_corr_producer.acquire_and_advance()
                    vec0_handle.commit()
                    vec1_handle = s1_corr_producer.acquire_and_advance()
                    vec1_handle.commit()
                # End of if not continue_cond
                # Advance to next tile
                if cutlass.const_expr(self.use_dynamic_work_scheduler):
                    work_tile, dynamic_work_consumer_state = self._consume_next_dynamic_work(
                        dynamic_work_pipeline,
                        dynamic_work_idx_ptr,
                        dynamic_work_consumer_state,
                    )
                elif cutlass.const_expr(self.use_clc_scheduler):
                    work_tile, clc_consumer_state = self._consume_next_clc_work(
                        clc_pipeline,
                        clc_response_ptr,
                        clc_consumer_state,
                    )
                else:
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
            # End of persistent scheduler loop
        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue (TMA store path only)
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.epilogue_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            if cutlass.const_expr(self.use_tma_store):
                if cutlass.const_expr(self.use_dynamic_work_scheduler):
                    work_tile, dynamic_work_consumer_state = self._consume_next_dynamic_work(
                        dynamic_work_pipeline,
                        dynamic_work_idx_ptr,
                        dynamic_work_consumer_state,
                    )
                while work_tile.is_valid_tile:
                    curr_block_coord = work_tile.tile_idx
                    batch_coord = curr_block_coord[2][1]
                    continue_cond = False
                    cuseqlen_q = Int32(0)
                    seqlen_q = mQ_qdl.shape[0]

                    if cutlass.const_expr(cum_seqlen_q is not None):
                        cuseqlen_q = cum_seqlen_q[batch_coord]
                        seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                        continue_cond = (
                            not fmha_utils.FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                                self.cta_tiler[0],
                                curr_block_coord[0],
                                seqlen_q,
                            )
                        )
                    if not continue_cond:
                        o0_coord = 2 * curr_block_coord[0]
                        o1_coord = o0_coord + 1
                        if cutlass.const_expr(self.use_ragged_tma_store):
                            flat_head_coord = curr_block_coord[2][0]
                            seq_end = cuseqlen_q + seqlen_q
                            o0_row = cuseqlen_q + o0_coord * self.pv_mma_tiler[0]
                            o1_row = cuseqlen_q + o1_coord * self.pv_mma_tiler[0]
                        else:
                            mO_qdl_ = mO_qdl
                            if cutlass.const_expr(
                                cum_seqlen_q is not None and self.uniform_varlen_o_extent is None
                            ):
                                mO_qdl_ = cute.domain_offset(
                                    (cum_seqlen_q[batch_coord], 0, ((0, 0), 0)),
                                    mO_qdl,
                                )
                            gO_qdl = cute.flat_divide(
                                mO_qdl_,
                                cute.select(self.pv_mma_tiler, mode=[0, 1]),
                            )
                            gO = gO_qdl[None, None, None, 0, curr_block_coord[2]]
                            tOsO, tOgO = cute.nvgpu.cpasync.tma_partition(
                                tma_atom_o,
                                0,
                                cute.make_layout(1),
                                cute.group_modes(sO, 0, 2),
                                cute.group_modes(gO, 0, 2),
                            )

                        # O0 O1 using the same pipeline
                        # wait from corr, issue tma store on smem
                        # O0
                        # 1. Wait for O0 final
                        o0_handle = corr_epi_consumer.wait_and_advance()
                        # 2. Copy O0 to gmem
                        if cutlass.const_expr(self.use_ragged_tma_store):
                            self._store_o_tile_ragged(
                                ragged_tma_o_desc,
                                sO[None, None, 0],
                                flat_head_coord,
                                o0_row,
                                seq_end,
                            )
                        else:
                            cute.copy(
                                tma_atom_o,
                                tOsO[None, 0],
                                tOgO[None, o0_coord],
                            )
                        cute.arch.cp_async_bulk_commit_group()
                        # O1
                        # 1. Wait for O1 final
                        o1_handle = corr_epi_consumer.wait_and_advance()
                        # 2. Copy O1 to gmem
                        if cutlass.const_expr(self.use_ragged_tma_store):
                            self._store_o_tile_ragged(
                                ragged_tma_o_desc,
                                sO[None, None, 1],
                                flat_head_coord,
                                o1_row,
                                seq_end,
                            )
                        else:
                            cute.copy(
                                tma_atom_o,
                                tOsO[None, 1],
                                tOgO[None, o1_coord],
                            )
                        cute.arch.cp_async_bulk_commit_group()

                        # Ensure O0 buffer is ready to be released
                        cute.arch.cp_async_bulk_wait_group(1, read=True)
                        o0_handle.release()
                        # Ensure O1 buffer is ready to be released
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                        o1_handle.release()

                    # Advance to next tile
                    if cutlass.const_expr(self.use_dynamic_work_scheduler):
                        work_tile, dynamic_work_consumer_state = self._consume_next_dynamic_work(
                            dynamic_work_pipeline,
                            dynamic_work_idx_ptr,
                            dynamic_work_consumer_state,
                        )
                    elif cutlass.const_expr(self.use_clc_scheduler):
                        work_tile, clc_consumer_state = self._consume_next_clc_work(
                            clc_pipeline,
                            clc_response_ptr,
                            clc_consumer_state,
                        )
                    else:
                        tile_sched.advance_to_next_work()
                        work_tile = tile_sched.get_current_work()
                cute.arch.griddepcontrol_launch_dependents()
            # End of persistent scheduler loop
        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax0 + Softmax1
        # ///////////////////////////////////////////////////////////////////////////////
        # With the merged dispatch (default on B300/sm_103 and later, where
        # the duplicated per-stage bodies cause instruction-fetch stalls),
        # both warp groups trace one shared body whose per-stage resources
        # are selected by the runtime stage index, halving the softmax
        # instruction footprint. Otherwise each warp group traces its own
        # specialized copy with a compile-time stage index.
        softmax_stage_specializations = (None,) if self.merge_softmax_warp_groups else (0, 1)
        for stage_spec in softmax_stage_specializations:
            softmax_warp_lo = self.softmax1_warp_ids[0] if stage_spec == 1 else 0
            softmax_warp_hi = (
                self.softmax1_warp_ids[0] if stage_spec == 0 else self.correction_warp_ids[0]
            )
            if cutlass.const_expr(softmax_warp_lo == 0):
                softmax_warp_sel = warp_idx < softmax_warp_hi
            else:
                softmax_warp_sel = (warp_idx >= softmax_warp_lo) and (warp_idx < softmax_warp_hi)
            if softmax_warp_sel:
                cute.arch.setmaxregister_increase(self.num_regs_softmax)
                tmem.wait_for_alloc()
                tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
                tStS, _, _, _, _, _, _ = make_tmem_tensors(
                    self, qk_thr_mma, pv_thr_mma, p_tmem_layout_staged, tmem_ptr
                )
                stage = (
                    warp_idx // len(self.softmax0_warp_ids) if stage_spec is None else stage_spec
                )
                mma_si_pipeline = make_pipeline_view(
                    mma_s0_pipeline,
                    storage.mma_s_mbar_ptr.data_ptr() + stage * self.mma_softmax_stage * 2,
                )
                si_corr_pipeline = make_pipeline_view(
                    s0_corr_pipeline,
                    storage.s_corr_mbar_ptr.data_ptr() + stage * self.softmax_corr_stage * 2,
                )
                pi_mma_pipeline = make_pipeline_view(
                    p0_mma_pipeline,
                    storage.p_mma_mbar_ptr.data_ptr() + stage * self.p_mma_stage * 2,
                )
                inplace_producer_pipeline = make_pipeline_view(
                    s0_p1_inplace_pipeline,
                    storage.inplace_mbar_ptr.data_ptr() + stage * self.p_mma_stage * 2,
                )
                inplace_consumer_pipeline = make_pipeline_view(
                    s0_p1_inplace_pipeline,
                    storage.inplace_mbar_ptr.data_ptr() + (stage ^ 1) * self.p_mma_stage * 2,
                )

                mma_si_consumer = mma_si_pipeline.make_consumer()
                si_corr_producer = si_corr_pipeline.make_producer()
                pi_mma_producer = pi_mma_pipeline.make_producer()
                inplace_producer = inplace_producer_pipeline.make_producer()
                inplace_consumer = inplace_consumer_pipeline.make_consumer()

                tStSi = cute.make_tensor(
                    tStS.iterator + stage * (self.tmem_s1_offset - self.tmem_s0_offset),
                    tStS.layout,
                )
                softmax_fn(
                    stage=stage,
                    tensor_args=(
                        tStS,
                        tStSi,
                        cum_seqlen_k,
                        cum_seqlen_q,
                        kv_len_tensor,
                        prefix_tokens_k,
                        warp_wants_skip_softmax_exchange[None, stage],
                        skip_softmax_count,
                        total_softmax_count,
                    ),
                    pipeline_args=(
                        mma_si_consumer,
                        si_corr_producer,
                        pi_mma_producer,
                    ),
                    inplace_args=(inplace_producer, inplace_consumer),
                )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_correction)
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            tStS, tStS0, tStS1, tOtO0, tOtO1, tOrP0, tOrP1 = make_tmem_tensors(
                self, qk_thr_mma, pv_thr_mma, p_tmem_layout_staged, tmem_ptr
            )
            cS = cute.make_identity_tensor((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
            tScS = qk_thr_mma.partition_C(cS)

            tStS_vec_layout = cute.composition(tStS.layout, cute.make_layout((128, 2)))

            tStS_vec0 = cute.make_tensor(tStS.iterator + self.tmem_vec0_offset, tStS_vec_layout)
            tStS_vec1 = cute.make_tensor(tStS.iterator + self.tmem_vec1_offset, tStS_vec_layout)

            tScS_vec_layout = cute.composition(tScS.layout, cute.make_layout((128, 2)))
            tScS_vec = cute.make_tensor(tScS.iterator, tScS_vec_layout)
            tmem_load_v_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(2)),
                self.qk_acc_dtype,
            )
            tiled_tmem_load_vec = tcgen05.make_tmem_copy(tmem_load_v_atom, tStS_vec0)
            thread_idx = tidx % (self.threads_per_warp * len(self.correction_warp_ids))
            thr_tmem_load_vec = tiled_tmem_load_vec.get_slice(thread_idx)
            tTMEM_LOAD_VECtS0 = thr_tmem_load_vec.partition_S(tStS_vec0)
            tTMEM_LOAD_VECtS1 = thr_tmem_load_vec.partition_S(tStS_vec1)
            tTMEM_LOAD_VECcS = thr_tmem_load_vec.partition_D(tScS_vec)
            corr_arch_state = self._make_correction_arch_state(arch_args, tmem_ptr, thread_idx)
            if cutlass.const_expr(self.use_dynamic_work_scheduler):
                work_tile, dynamic_work_consumer_state = self._consume_next_dynamic_work(
                    dynamic_work_pipeline,
                    dynamic_work_idx_ptr,
                    dynamic_work_consumer_state,
                )
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                batch_coord = curr_block_coord[2][1]
                seqlen_k = mK_kdl.shape[0]
                row_idx = Int32(0)
                continue_cond = False
                cuseqlen_q = Int32(0)
                o_sf_row_offset = Int32(0)
                seqlen_q = mQ_qdl.shape[0]

                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    o_sf_row_offset = cuseqlen_q
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    continue_cond = (
                        not fmha_utils.FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                            self.cta_tiler[0],
                            curr_block_coord[0],
                            seqlen_q,
                        )
                    )
                else:
                    # Scale rows flatten fixed [B, S] output in token order.
                    o_sf_row_offset = batch_coord * seqlen_q
                if not continue_cond:
                    row_idx = curr_block_coord[0] * self.cta_tiler[0] + tTMEM_LOAD_VECcS[0][0]
                    if cutlass.const_expr(self.prefix_window):
                        seqlen_k = prefix_tokens_k + kv_len_tensor[batch_coord]
                    if cutlass.const_expr(cum_seqlen_k is not None and not self.prefix_window):
                        cuseqlen_k = cum_seqlen_k[batch_coord]
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                    continue_cond = seqlen_k <= 0
                if not continue_cond:
                    # Ignore first signal from softmax as no correction is required
                    vec0_handle = s0_corr_consumer.wait_and_advance()
                    vec0_handle.release()
                    vec1_handle = s1_corr_consumer.wait_and_advance()
                    vec1_handle.release()
                    seqlen_kv_loop_steps = self._kv_loop_steps(
                        curr_block_coord,
                        seqlen_q,
                        seqlen_k,
                        window_size_left,
                        window_size_right,
                        batch_coord,
                        cum_seqlen_k,
                        kv_len_tensor,
                        prefix_tokens_k,
                    )
                    # O0/O1 share the same mma_corr consumer state, so the
                    # Oi peek token rolls from O0 -> O1 -> next O0. Seed
                    # with a blocking token; correction_rescale refreshes
                    # it near the end of each iteration.
                    oi_peek_status = cutlass.Boolean(False)
                    for _ in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
                        (
                            (s0_corr_consumer, mma_corr_consumer),
                            oi_peek_status,
                        ) = self.correction_rescale(
                            pv_thr_mma,
                            tiled_tmem_load_vec,
                            scale_softmax_log2,
                            (tOtO0, tTMEM_LOAD_VECtS0, tTMEM_LOAD_VECcS),
                            (s0_corr_consumer, mma_corr_consumer),
                            oi_peek_status,
                            corr_arch_state,
                            0,
                        )
                        (
                            (s1_corr_consumer, mma_corr_consumer),
                            oi_peek_status,
                        ) = self.correction_rescale(
                            pv_thr_mma,
                            tiled_tmem_load_vec,
                            scale_softmax_log2,
                            (tOtO1, tTMEM_LOAD_VECtS1, tTMEM_LOAD_VECcS),
                            (s1_corr_consumer, mma_corr_consumer),
                            oi_peek_status,
                            corr_arch_state,
                            1,
                        )
                    # End of seqlen_corr_loop_steps
                    value_args = (
                        cuseqlen_q,
                        o_sf_row_offset,
                        seqlen_q,
                        curr_block_coord,
                        scale_softmax,
                        scale_output,
                        o_sf_scale,
                        o_sf_start_index,
                    )
                    if cutlass.const_expr(self.use_tma_store):
                        # TMA store path: write to sO, signal epilogue warp
                        # Normalize O0
                        s0_corr_consumer, mma_corr_consumer, corr_epi_producer = (
                            self.correction_epilog(
                                pv_thr_mma,
                                tiled_tmem_load_vec,
                                (
                                    tOtO0,
                                    tTMEM_LOAD_VECtS0,
                                    tTMEM_LOAD_VECcS,
                                    sO[None, None, 0],
                                    mLSE,
                                    mSink,
                                    mO_sf,
                                ),
                                (
                                    s0_corr_consumer,
                                    mma_corr_consumer,
                                    corr_epi_producer,
                                ),
                                (row_idx, *value_args),
                                corr_arch_state,
                                0,
                            )
                        )
                        row_idx += self.qk_mma_tiler[0]
                        # Normalize O1
                        s1_corr_consumer, mma_corr_consumer, corr_epi_producer = (
                            self.correction_epilog(
                                pv_thr_mma,
                                tiled_tmem_load_vec,
                                (
                                    tOtO1,
                                    tTMEM_LOAD_VECtS1,
                                    tTMEM_LOAD_VECcS,
                                    sO[None, None, 1],
                                    mLSE,
                                    mSink,
                                    mO_sf,
                                ),
                                (
                                    s1_corr_consumer,
                                    mma_corr_consumer,
                                    corr_epi_producer,
                                ),
                                (row_idx, *value_args),
                                corr_arch_state,
                                1,
                            )
                        )
                    else:
                        # st.global path: store directly to global memory
                        block_offset_o = Int32(0)
                        if cutlass.const_expr(cum_seqlen_q is not None):
                            block_offset_o = cum_seqlen_q[batch_coord]
                        mO_ = cute.make_tensor(
                            mO.iterator + block_offset_o * mO.stride[0],
                            cute.make_layout(
                                (seqlen_q, mO.shape[1], mO.shape[2]),
                                stride=mO.stride,
                            ),
                        )
                        o0_coord = 2 * curr_block_coord[0]
                        o1_coord = o0_coord + 1
                        gO_stg = cute.local_tile(
                            mO_,
                            (self.pv_mma_tiler[0], self.pv_mma_tiler[1]),
                            (None, None, None),
                        )
                        gO0 = gO_stg[None, None, o0_coord, 0, curr_block_coord[2]]
                        gO1 = gO_stg[None, None, o1_coord, 0, curr_block_coord[2]]
                        # Normalize O0 and store to global memory
                        s0_corr_consumer, mma_corr_consumer = self.correction_epilog(
                            pv_thr_mma,
                            tiled_tmem_load_vec,
                            (
                                tOtO0,
                                tTMEM_LOAD_VECtS0,
                                tTMEM_LOAD_VECcS,
                                gO0,
                                mLSE,
                                mSink,
                                mO_sf,
                            ),
                            (s0_corr_consumer, mma_corr_consumer),
                            (row_idx, *value_args),
                            corr_arch_state,
                            0,
                        )
                        row_idx += self.qk_mma_tiler[0]
                        # Normalize O1 and st.global to global memory
                        s1_corr_consumer, mma_corr_consumer = self.correction_epilog(
                            pv_thr_mma,
                            tiled_tmem_load_vec,
                            (
                                tOtO1,
                                tTMEM_LOAD_VECtS1,
                                tTMEM_LOAD_VECcS,
                                gO1,
                                mLSE,
                                mSink,
                                mO_sf,
                            ),
                            (s1_corr_consumer, mma_corr_consumer),
                            (row_idx, *value_args),
                            corr_arch_state,
                            1,
                        )
                # End of if not continue_cond
                # Advance to next tile
                if cutlass.const_expr(self.use_dynamic_work_scheduler):
                    work_tile, dynamic_work_consumer_state = self._consume_next_dynamic_work(
                        dynamic_work_pipeline,
                        dynamic_work_idx_ptr,
                        dynamic_work_consumer_state,
                    )
                elif cutlass.const_expr(self.use_clc_scheduler):
                    work_tile, clc_consumer_state = self._consume_next_clc_work(
                        clc_pipeline,
                        clc_response_ptr,
                        clc_consumer_state,
                    )
                else:
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
            if cutlass.const_expr(not self.use_tma_store):
                cute.arch.griddepcontrol_launch_dependents()
            # End of persistent scheduler loop
            tmem.relinquish_alloc_permit()
            # Synchronize before TMEM dealloc (done by the caller)
            self.tmem_dealloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr, self.num_tmem_alloc_cols)
        return

    @cute.jit
    def get_skip_softmax_flag(self, tiled_tmem_load_v, tTMEM_LOADtS_v, tTMEM_LOADrS_v):
        cute.copy(tiled_tmem_load_v, tTMEM_LOADtS_v, tTMEM_LOADrS_v)
        tTMEM_LOADrS_v_i32 = cute.recast_tensor(tTMEM_LOADrS_v, dtype=cutlass.Int32)
        skip_softmax_flag = cute.arch.make_warp_uniform(tTMEM_LOADrS_v_i32[0])
        return skip_softmax_flag

    @cute.jit
    def mma_qk(
        self,
        tiled_mma: cute.TiledMma,
        tensor_args: Tuple,
        pipeline_args: Tuple,
        pipeline_tokens: Tuple = (None, None),
    ) -> Tuple[pipeline.PipelineProducer, pipeline.PipelineProducer]:
        """Perform a single step of the QK GEMM computation on a block of attention scores.

        :param tiled_mma: Tiled MMA for QK GEMM
        :type tiled_mma: cute.TiledMma
        :param tensor_args: Tuple containing Qi, K, and Si
        :type tensor_args: Tuple
        :param pipeline_args: Tuple containing mma_si_producer and si_corr_producer
        :type pipeline_args: Tuple
        :param pipeline_tokens: Optional non-blocking peek tokens for the Si and
            vec_i producers, in the form ``(si_peek_status, veci_peek_status)``.
            ``None`` for either token falls back to a blocking acquire.
        :type pipeline_tokens: Tuple
        :return: Tuple containing mma_si_producer and si_corr_producer
        :rtype: Tuple[pipeline.PipelineProducer, pipeline.PipelineProducer]
        """
        tSrQi, tSrK, tStSi = tensor_args
        mma_si_producer, si_corr_producer = pipeline_args
        si_peek_status, veci_peek_status = pipeline_tokens
        # 0. Make sure Qi & K are ready when calling mma_qk
        # 1. acquire S0
        si_handle = mma_si_producer.acquire_and_advance(si_peek_status)
        # 2. make sure vec is already released in corr
        veci_handle = si_corr_producer.acquire_and_advance(veci_peek_status)
        veci_handle.commit()
        # 3. gemm
        num_kphases = cute.size(tSrQi, mode=[2])
        for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
            kphase_coord = (None, None, kphase_idx)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
            cute.gemm(
                tiled_mma,
                tStSi,
                tSrQi[kphase_coord],
                tSrK[kphase_coord],
                tStSi,
            )
        # 4. release S0
        si_handle.commit()
        return mma_si_producer, si_corr_producer

    @cute.jit
    def mma_pv(
        self,
        tiled_mma: cute.TiledMma,
        whether_acc: bool,
        tensor_args: Tuple,
        pipeline_args: Tuple,
        skip_pv_args: Tuple,
    ) -> Tuple[
        pipeline.PipelineProducer,
        pipeline.PipelineConsumer,
    ]:
        """Perform a single step of the PV GEMM computation on accumulating O.

        :param tiled_mma: Tiled MMA for PV GEMM
        :type tiled_mma: cute.TiledMma
        :param whether_acc: Whether to accumulate O
        :type whether_acc: bool
        :param tensor_args: Tuple containing Pi, Vi, and Oi
        :type tensor_args: Tuple
        :param pipeline_args: Tuple containing mma_corr_producer and pi_mma_consumer
        :type pipeline_args: Tuple
        :param skip_pv_args: Tuple containing enable_skip_softmax, tiled_tmem_load_v, tTMEM_LOADtS_v, tTMEM_LOADrS_v
        :type skip_pv_args: Tuple
        :return: Tuple containing mma_corr_producer and pi_mma_consumer
        :rtype: Tuple[pipeline.PipelineProducer, pipeline.PipelineConsumer]
        """
        tOrPi, tOrVi, tOtOi = tensor_args
        mma_corr_producer, pi_mma_consumer = pipeline_args
        enable_skip_softmax, tiled_tmem_load_v, tTMEM_LOADtS_v, tTMEM_LOADrS_v = skip_pv_args
        # 0. Make sure Vi is ready when calling mma_pv
        # 1. acquire Oi
        oi_handle = mma_corr_producer.acquire_and_advance()
        # 2. wait for Pi
        pi_handle = pi_mma_consumer.wait_and_advance()
        # 3. gemm
        num_kphases = cute.size(tOrPi, mode=[2])
        if cutlass.const_expr(enable_skip_softmax):
            skip_pv = self.get_skip_softmax_flag(tiled_tmem_load_v, tTMEM_LOADtS_v, tTMEM_LOADrS_v)
            if not skip_pv:
                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                    kphase_coord = (None, None, kphase_idx)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, whether_acc or kphase_idx != 0)
                    cute.gemm(
                        tiled_mma,
                        tOtOi,
                        tOrPi[kphase_coord],
                        tOrVi[kphase_coord],
                        tOtOi,
                    )
        else:
            for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                kphase_coord = (None, None, kphase_idx)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, whether_acc or kphase_idx != 0)
                cute.gemm(
                    tiled_mma,
                    tOtOi,
                    tOrPi[kphase_coord],
                    tOrVi[kphase_coord],
                    tOtOi,
                )
        # 4. commit Pi
        pi_handle.release()
        # 5. commit Oi
        oi_handle.commit()
        return mma_corr_producer, pi_mma_consumer

    @cute.jit
    def calculate_skip_softmax_flag(
        self,
        row_max,
        tile_row_max,
        scale_softmax_log2,
        skip_softmax_threshold_log2,
        seqlen_q,
        thread_idx,
        logical_offset,
        warp_wants_skip_softmax_exchange,
        stage,
        skip_softmax_count,
        total_softmax_count,
    ) -> Tuple[bool, float]:
        """Calculate the skip softmax flag and the row maximum.

        :param row_max: The row maximum.
        :type row_max: float
        :param tile_row_max: The tile row maximum.
        :type tile_row_max: float
        :param scale_softmax_log2: The scale softmax log2.
        :type scale_softmax_log2: float
        :param skip_softmax_threshold_log2: The skip softmax threshold log2.
        :type skip_softmax_threshold_log2: float
        :param seqlen_q: The sequence length q.
        :type seqlen_q: int
        :param thread_idx: The thread index.
        :type thread_idx: int
        :param logical_offset: The logical offset.
        :type logical_offset: Tuple[int, int]
        :param warp_wants_skip_softmax_exchange: The warp wants skip softmax exchange.
        :type warp_wants_skip_softmax_exchange: cute.Tensor
        :param stage: The stage.
        :type stage: int
        :param skip_softmax_count: The skip softmax count.
        :type skip_softmax_count: cute.Tensor
        :param total_softmax_count: The total softmax count.
        :type total_softmax_count: cute.Tensor
        :return: Tuple containing the skip softmax flag and the row maximum.
        :rtype: Tuple[bool, float]
        """
        thread_wants_skip = (
            tile_row_max * scale_softmax_log2 - row_max * scale_softmax_log2
        ) < skip_softmax_threshold_log2
        thread_wants_skip = thread_wants_skip or ((logical_offset[0] + thread_idx) >= seqlen_q)
        warp_wants_skip = cute.arch.vote_all_sync(thread_wants_skip)

        with cute.arch.elect_one():
            warp_wants_skip_softmax_exchange[cute.arch.warp_idx() % 4] = warp_wants_skip
        self._softmax_warpgroup_barrier(stage).arrive_and_wait()
        warp_wants_skip_softmax_exchange_i32 = cute.make_tensor(
            cute.recast_ptr(warp_wants_skip_softmax_exchange.iterator, dtype=cutlass.Int32),
            cute.make_layout((1,)),
        )
        skip_softmax = cute.arch.popc(warp_wants_skip_softmax_exchange_i32[0]) == 4

        if not skip_softmax:
            row_max = max(row_max, tile_row_max)

        if cutlass.const_expr(skip_softmax_count is not None):
            if thread_idx == 0:
                if skip_softmax:
                    cute.arch.atomic_add(skip_softmax_count.iterator.llvm_ptr, Int32(1))
                cute.arch.atomic_add(total_softmax_count.iterator.llvm_ptr, Int32(1))
        return skip_softmax, row_max

    @cute.jit
    def apply_exp_and_cvt(
        self,
        tTMEM_LOADrS,
        tTMEM_LOADrS_cvt,
        tTMEM_STORErS_x4_e_cvt,
        stage,
        scale,
        minus_row_max_scale,
        local_row_sum,
        pi_mma_producer,
        inplace_consumer,
        EXP2_EMULATION_OFFSET,
        EXP2_EMULATION_COUNT,
        CVT_COUNT,
        CVT_PER_STEP,
        FMA_COUNT,
        ARV_COUNT,
    ):
        """Apply the exp and conversion to the P data type on fragment.

        :param tTMEM_LOADrS: The tTMEM_LOADrS tensor.
        :type tTMEM_LOADrS: cute.Tensor
        :param tTMEM_LOADrS_cvt: The tTMEM_LOADrS_cvt tensor.
        :type tTMEM_LOADrS_cvt: cute.Tensor
        :param tTMEM_STORErS_x4_e_cvt: The tTMEM_STORErS_x4_e_cvt tensor.
        :type tTMEM_STORErS_x4_e_cvt: cute.Tensor
        :param stage: The stage.
        :type stage: int
        :param scale: The scale.
        :type scale: float
        :param minus_row_max_scale: The minus row maximum scale.
        :type minus_row_max_scale: float
        :param local_row_sum: The local row sum.
        :type local_row_sum: float
        :param pi_mma_producer: The P-buffer producer pipeline.
        :type pi_mma_producer: PipelineTmaAsync
        :param inplace_consumer: The inplace consumer.
        :type inplace_consumer: cute.Tensor
        :param EXP2_EMULATION_OFFSET: The exp2 emulation offset.
        :type EXP2_EMULATION_OFFSET: int
        :param EXP2_EMULATION_COUNT: The exp2 emulation count.
        :type EXP2_EMULATION_COUNT: int
        :param CVT_COUNT: The cvt count.
        :type CVT_COUNT: int
        :param CVT_PER_STEP: The cvt per step.
        :type CVT_PER_STEP: int
        :param FMA_COUNT: The fma count.
        :type FMA_COUNT: int
        :param ARV_COUNT: The arv count.
        :type ARV_COUNT: int
        :return: The local row sum, inplace consumer, and the two pipeline peek tokens.
        :rtype: Tuple[float, cute.Tensor, object, object]
        """
        pi_peek_status = None
        inplace_peek_status = None
        for i in cutlass.range_constexpr(0, EXP2_EMULATION_OFFSET, 2):
            if cutlass.const_expr(i >= CVT_COUNT):
                if cutlass.const_expr(i % CVT_PER_STEP == 0):
                    if cutlass.const_expr(self.v_dtype.width == 8):
                        fmha_utils.cvt_f32x4_to_f8x4(
                            tTMEM_LOADrS_cvt[None, (i - CVT_COUNT) // CVT_PER_STEP],
                            tTMEM_STORErS_x4_e_cvt[None, (i - CVT_COUNT) // CVT_PER_STEP],
                        )
                    else:
                        s_vec = tTMEM_LOADrS_cvt[None, (i - CVT_COUNT) // CVT_PER_STEP].load()
                        tTMEM_STORErS_x4_e_cvt[None, (i - CVT_COUNT) // CVT_PER_STEP].store(
                            s_vec.to(self.v_dtype)
                        )
                local_row_sum = cute.arch.add_packed_f32x2(
                    local_row_sum,
                    (
                        tTMEM_LOADrS[i - CVT_COUNT],
                        tTMEM_LOADrS[i - CVT_COUNT + 1],
                    ),
                )
            tTMEM_LOADrS[i] = cute.math.exp2(tTMEM_LOADrS[i], fastmath=True)
            if cutlass.const_expr(i == EXP2_EMULATION_OFFSET - 4):
                # Stagger the two non-blocking probes across the first two of
                # the final four exp2 instructions. Probe inplace first because it is
                # consumed before the P-buffer producer token in the step tail.
                inplace_peek_status = inplace_consumer.try_wait()
            if cutlass.const_expr(i + FMA_COUNT < EXP2_EMULATION_OFFSET):
                (
                    tTMEM_LOADrS[i + FMA_COUNT],
                    tTMEM_LOADrS[i + FMA_COUNT + 1],
                ) = cute.arch.fma_packed_f32x2(
                    (
                        tTMEM_LOADrS[i + FMA_COUNT],
                        tTMEM_LOADrS[i + FMA_COUNT + 1],
                    ),
                    (scale, scale),
                    (minus_row_max_scale, minus_row_max_scale),
                )
            tTMEM_LOADrS[i + 1] = cute.math.exp2(tTMEM_LOADrS[i + 1], fastmath=True)
            if cutlass.const_expr(i == EXP2_EMULATION_OFFSET - 4):
                pi_peek_status = pi_mma_producer.try_acquire()
            if cutlass.const_expr(i == EXP2_EMULATION_OFFSET - ARV_COUNT):
                if cutlass.const_expr(self.enable_sequence_barrier):
                    self._sequence_barrier(1 - stage).arrive()
        # The remaining conversion steps
        for i in cutlass.range_constexpr(
            EXP2_EMULATION_OFFSET - CVT_COUNT,
            EXP2_EMULATION_OFFSET,
            2,
        ):
            if cutlass.const_expr(i % CVT_PER_STEP == 0):
                if cutlass.const_expr(self.v_dtype.width == 8):
                    fmha_utils.cvt_f32x4_to_f8x4(
                        tTMEM_LOADrS_cvt[None, i // CVT_PER_STEP],
                        tTMEM_STORErS_x4_e_cvt[None, i // CVT_PER_STEP],
                    )
                else:
                    s_vec = tTMEM_LOADrS_cvt[None, i // CVT_PER_STEP].load()
                    tTMEM_STORErS_x4_e_cvt[None, i // CVT_PER_STEP].store(s_vec.to(self.v_dtype))
            local_row_sum = cute.arch.add_packed_f32x2(
                local_row_sum, (tTMEM_LOADrS[i], tTMEM_LOADrS[i + 1])
            )
        for i in cutlass.range_constexpr(
            EXP2_EMULATION_OFFSET, EXP2_EMULATION_OFFSET + EXP2_EMULATION_COUNT // 2, 2
        ):
            tTMEM_LOADrS[i], tTMEM_LOADrS[i + 1] = cute.arch.fma_packed_f32x2(
                (tTMEM_LOADrS[i], tTMEM_LOADrS[i + 1]),
                (scale, scale),
                (minus_row_max_scale, minus_row_max_scale),
            )
            tTMEM_LOADrS[i], tTMEM_LOADrS[i + 1] = fmha_utils.ex2_emulation_packed_f32x2(
                tTMEM_LOADrS[i], tTMEM_LOADrS[i + 1]
            )
            if cutlass.const_expr((i + 2) % CVT_PER_STEP == 0):
                if cutlass.const_expr(self.v_dtype.width == 8):
                    fmha_utils.cvt_f32x4_to_f8x4(
                        tTMEM_LOADrS_cvt[None, i // CVT_PER_STEP],
                        tTMEM_STORErS_x4_e_cvt[None, i // CVT_PER_STEP],
                    )
                else:
                    s_vec = tTMEM_LOADrS_cvt[None, i // CVT_PER_STEP].load()
                    tTMEM_STORErS_x4_e_cvt[None, i // CVT_PER_STEP].store(s_vec.to(self.v_dtype))

        for i in cutlass.range_constexpr(
            EXP2_EMULATION_OFFSET + EXP2_EMULATION_COUNT // 2,
            EXP2_EMULATION_OFFSET + EXP2_EMULATION_COUNT,
            2,
        ):
            tTMEM_LOADrS[i], tTMEM_LOADrS[i + 1] = cute.arch.fma_packed_f32x2(
                (tTMEM_LOADrS[i], tTMEM_LOADrS[i + 1]),
                (scale, scale),
                (minus_row_max_scale, minus_row_max_scale),
            )
            tTMEM_LOADrS[i], tTMEM_LOADrS[i + 1] = fmha_utils.ex2_emulation_packed_f32x2(
                tTMEM_LOADrS[i], tTMEM_LOADrS[i + 1]
            )
            if cutlass.const_expr((i + 2) % CVT_PER_STEP == 0):
                if cutlass.const_expr(self.v_dtype.width == 8):
                    fmha_utils.cvt_f32x4_to_f8x4(
                        tTMEM_LOADrS_cvt[None, i // CVT_PER_STEP],
                        tTMEM_STORErS_x4_e_cvt[None, i // CVT_PER_STEP],
                    )
                else:
                    s_vec = tTMEM_LOADrS_cvt[None, i // CVT_PER_STEP].load()
                    tTMEM_STORErS_x4_e_cvt[None, i // CVT_PER_STEP].store(s_vec.to(self.v_dtype))
        inplace_consumer.wait_and_advance(inplace_peek_status)
        return (
            local_row_sum,
            inplace_consumer,
            pi_peek_status,
            inplace_peek_status,
        )

    @cute.jit
    def _apply_prefix_window_mask(
        self,
        acc_qk: cute.Tensor,
        index_qk: cute.Tensor,
        seqlen_q: Int32,
        visible_range: Tuple,
    ) -> None:
        """Mask the scores of keys outside ``[0, prefix)`` and ``[lo, hi)``, and of rows past the query."""
        prefix, lo, hi = visible_range
        for i in cutlass.range(cute.size(acc_qk), unroll_full=True):
            index_q, index_k = index_qk[i]
            if index_k >= prefix and (index_k < lo or index_k >= hi):
                acc_qk[i] = -Float32.inf
            if index_q >= seqlen_q:
                acc_qk[i] = -Float32.inf

    @cute.jit
    def softmax_step(
        self,
        stage,
        whether_apply_mask: bool,
        iter_args: Tuple,
        stats_args: Tuple,
        pipeline_args: Tuple,
        value_args: Tuple,
        atom_args: Tuple,
        tensor_args: Tuple,
    ) -> Tuple[Tuple, Tuple]:
        """Perform a single step of the softmax computation on a block of attention scores.

        This method processes one block of the attention matrix, computing numerically stable
        softmax by first finding the row maximum, subtracting it from all elements, applying
        exponential function, and then normalizing by the sum of exponentials. It also handles
        optional masking of attention scores.

        The method involves several key operations:
        1. Loading attention scores from tensor memory
        2. Applying optional masking based on position
        3. Computing row-wise maximum values for numerical stability
        4. Transforming scores using exp2(x*scale - max*scale)
        5. Computing row sums for normalization
        6. Coordinating pipeline synchronization between different processing stages

        :param stage: Processing stage (0 for first half, 1 for second half)
        :type stage: int
        :param whether_apply_mask: Whether to apply attention masking
        :type whether_apply_mask: bool
        :param iter_args: Tuple containing the counting tensor, row_max, row_sum, and vector buffer's handle for current iteration
        :type iter_args: Tuple
        :param stats_args: Tuple containing row_sum and row_max
        :type stats_args: Tuple
        :param pipeline_args: Tuple containing pipeline related arguments for MMA, correction, and sequence synchronization
        :type pipeline_args: Tuple
        :param value_args: Tuple containing seqlen_k, seqlen_q, and scale_softmax_log2
        :type value_args: Tuple
        :param atom_args: Tuple containing mma & copy atoms
        :type atom_args: Tuple
        :param tensor_args: Tuple containing softmax related tensors
        :type tensor_args: Tuple
        :param fused_mask: Compute trip counts and apply masking for attention blocks
        :type fused_mask: fmha_utils.FusedMask
        :return: Updated stats_args and pipeline_args
        :rtype: Tuple[Tuple, Tuple]
        """
        row_sum, row_max = stats_args
        cS, is_last_iter = iter_args
        (
            seqlen_k,
            seqlen_q,
            scale_softmax_log2,
            window_size_left,
            window_size_right,
            skip_softmax_threshold_log2,
            thread_idx,
            logical_offset,
            visible_range,
        ) = value_args
        (
            si_peek_status,
            mma_si_consumer,
            si_corr_producer,
            pi_mma_producer,
            inplace_producer,
            inplace_consumer,
        ) = pipeline_args
        (
            qk_thr_mma,
            tiled_tmem_load,
            tiled_tmem_store,
            tiled_tmem_store_vec,
            thr_tmem_load,
            thr_tmem_store,
            thr_tmem_store_vec,
        ) = atom_args
        (
            tTMEM_LOADtS,
            tTMEM_STORE_VECtS,
            tTMEM_STORE_SKIP_SOFTMAX,
            tTMEM_STOREtS_x4,
            warp_wants_skip_softmax_exchange,
            skip_softmax_count,
            total_softmax_count,
        ) = tensor_args
        tilePlikeFP32 = self.qk_mma_tiler[1] // Float32.width * self.v_dtype.width
        tScS = qk_thr_mma.partition_C(cS)
        enable_skip_softmax = skip_softmax_threshold_log2 is not None
        tScS_vec_layout = cute.composition(tScS.layout, cute.make_layout((128, 2)))
        tScS_vec = cute.make_tensor(tScS.iterator, tScS_vec_layout)
        tScS_P_layout = cute.composition(tScS.layout, cute.make_layout((128, tilePlikeFP32)))
        tScS_P = cute.make_tensor(tScS.iterator, tScS_P_layout)
        tTMEM_LOADcS = thr_tmem_load.partition_D(tScS)
        tTMEM_STORE_VECcS = thr_tmem_store_vec.partition_S(tScS_vec)
        tTMEM_STOREcS = thr_tmem_store.partition_S(tScS_P)
        # Wait for Si
        si_handle = mma_si_consumer.wait_and_advance(si_peek_status)
        tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcS.shape, self.qk_acc_dtype)
        old_row_max = row_max
        skip_softmax = cutlass.Boolean(False)
        if whether_apply_mask:
            if cutlass.const_expr(self.arch >= Arch.sm_100 and self.arch <= Arch.sm_100f):
                cute.copy(tiled_tmem_load, tTMEM_LOADtS, tTMEM_LOADrS)
            else:
                tTMEM_LOADrMax = cute.make_rmem_tensor(
                    cute.make_layout((1, cute.size(tTMEM_LOADrS, mode=[1]))),
                    self.qk_acc_dtype,
                )
                for i in cutlass.range_constexpr(0, cute.size(tTMEM_LOADrS, mode=[1])):
                    cute.copy_atom_call(
                        tiled_tmem_load,
                        tTMEM_LOADtS[None, i, 0, 0],
                        (tTMEM_LOADrS[None, i, 0, 0], tTMEM_LOADrMax[None, i]),
                    )
                    if cutlass.const_expr(self.arch > Arch.sm_100f and i == 1):
                        cute.arch.fence_view_async_tmem_load()
            if cutlass.const_expr(self.prefix_window):
                self._apply_prefix_window_mask(tTMEM_LOADrS, tTMEM_LOADcS, seqlen_q, visible_range)
            else:
                fmha_utils.FusedMask.apply_mask(
                    self.mask_type,
                    tTMEM_LOADrS,
                    tTMEM_LOADcS,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                    use_causal_setp_selp_inline=cutlass.const_expr(
                        self.use_causal_setp_selp_inline
                        and (
                            window_size_left is not None
                            or window_size_right is not None
                            or self.mask_type is fmha_utils.MaskEnum.RESIDUAL_MASK
                        )
                    ),
                )
            tile_row_max = tTMEM_LOADrS.load().reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
            if cutlass.const_expr(not enable_skip_softmax):
                row_max = cute.arch.fmax(row_max, tile_row_max)
            else:
                skip_softmax, row_max = self.calculate_skip_softmax_flag(
                    row_max,
                    tile_row_max,
                    scale_softmax_log2,
                    skip_softmax_threshold_log2,
                    seqlen_q,
                    thread_idx,
                    logical_offset,
                    warp_wants_skip_softmax_exchange,
                    stage,
                    skip_softmax_count,
                    total_softmax_count,
                )
            si_handle.release()
            # S0 -> P1 / S1 -> P0
            inplace_producer.commit()
            inplace_producer.advance()
        else:
            if cutlass.const_expr(self.arch >= Arch.sm_100 and self.arch <= Arch.sm_100f):
                cute.copy(
                    tiled_tmem_load,
                    tTMEM_LOADtS[None, 0, None, None],
                    tTMEM_LOADrS[None, 0, None, None],
                )
                cute.copy(
                    tiled_tmem_load,
                    tTMEM_LOADtS[None, 1, None, None],
                    tTMEM_LOADrS[None, 1, None, None],
                )
                tile_row_max = -cutlass.Float32.inf
                tile_row_max_ = tile_row_max
                for i in cutlass.range_constexpr(0, cute.size(tTMEM_LOADrS, mode=[0]), 4):
                    tile_row_max = cute.arch.fmax(tile_row_max, tTMEM_LOADrS[i, 0, 0, 0])
                    tile_row_max = cute.arch.fmax(tile_row_max, tTMEM_LOADrS[i + 1, 0, 0, 0])
                    tile_row_max_ = cute.arch.fmax(tile_row_max_, tTMEM_LOADrS[i + 2, 0, 0, 0])
                    tile_row_max_ = cute.arch.fmax(tile_row_max_, tTMEM_LOADrS[i + 3, 0, 0, 0])
                cute.copy(
                    tiled_tmem_load,
                    tTMEM_LOADtS[None, 2, None, None],
                    tTMEM_LOADrS[None, 2, None, None],
                )
                for i in cutlass.range_constexpr(0, cute.size(tTMEM_LOADrS, mode=[0]), 4):
                    tile_row_max = cute.arch.fmax(tile_row_max, tTMEM_LOADrS[i, 1, 0, 0])
                    tile_row_max = cute.arch.fmax(tile_row_max, tTMEM_LOADrS[i + 1, 1, 0, 0])
                    tile_row_max_ = cute.arch.fmax(tile_row_max_, tTMEM_LOADrS[i + 2, 1, 0, 0])
                    tile_row_max_ = cute.arch.fmax(tile_row_max_, tTMEM_LOADrS[i + 3, 1, 0, 0])
                cute.copy(
                    tiled_tmem_load,
                    tTMEM_LOADtS[None, 3, None, None],
                    tTMEM_LOADrS[None, 3, None, None],
                )
                for i in cutlass.range_constexpr(0, cute.size(tTMEM_LOADrS, mode=[0]), 4):
                    tile_row_max = cute.arch.fmax(tile_row_max, tTMEM_LOADrS[i, 2, 0, 0])
                    tile_row_max = cute.arch.fmax(tile_row_max, tTMEM_LOADrS[i + 1, 2, 0, 0])
                    tile_row_max_ = cute.arch.fmax(tile_row_max_, tTMEM_LOADrS[i + 2, 2, 0, 0])
                    tile_row_max_ = cute.arch.fmax(tile_row_max_, tTMEM_LOADrS[i + 3, 2, 0, 0])
                cute.arch.fence_view_async_tmem_store()
                si_handle.release()
                # S0 -> P1 / S1 -> P0
                inplace_producer.commit()
                inplace_producer.advance()
                for i in cutlass.range_constexpr(0, cute.size(tTMEM_LOADrS, mode=[0]), 4):
                    tile_row_max = cute.arch.fmax(tile_row_max, tTMEM_LOADrS[i, 3, 0, 0])
                    tile_row_max = cute.arch.fmax(tile_row_max, tTMEM_LOADrS[i + 1, 3, 0, 0])
                    tile_row_max_ = cute.arch.fmax(tile_row_max_, tTMEM_LOADrS[i + 2, 3, 0, 0])
                    tile_row_max_ = cute.arch.fmax(tile_row_max_, tTMEM_LOADrS[i + 3, 3, 0, 0])
                tile_row_max = cute.arch.fmax(tile_row_max, tile_row_max_)
                if cutlass.const_expr(not enable_skip_softmax):
                    row_max = cute.arch.fmax(tile_row_max, row_max)
                else:
                    skip_softmax, row_max = self.calculate_skip_softmax_flag(
                        row_max,
                        tile_row_max,
                        scale_softmax_log2,
                        skip_softmax_threshold_log2,
                        seqlen_q,
                        thread_idx,
                        logical_offset,
                        warp_wants_skip_softmax_exchange,
                        stage,
                        skip_softmax_count,
                        total_softmax_count,
                    )
            else:
                tTMEM_LOADrMax = cute.make_rmem_tensor(
                    cute.make_layout((1, cute.size(tTMEM_LOADrS, mode=[1]))),
                    self.qk_acc_dtype,
                )
                for i in cutlass.range_constexpr(0, cute.size(tTMEM_LOADrS, mode=[1])):
                    cute.copy_atom_call(
                        tiled_tmem_load,
                        tTMEM_LOADtS[None, i, 0, 0],
                        (tTMEM_LOADrS[None, i, 0, 0], tTMEM_LOADrMax[None, i]),
                    )
                    if cutlass.const_expr(self.arch > Arch.sm_100f and i == 1):
                        cute.arch.fence_view_async_tmem_load()
                cute.arch.fence_view_async_tmem_store()
                tile_row_max = tTMEM_LOADrMax.load().reduce(
                    cute.ReductionOp.MAX, -cutlass.Float32.inf, 0
                )
                if cutlass.const_expr(not enable_skip_softmax):
                    row_max = cute.arch.fmax(tile_row_max, row_max)
                else:
                    skip_softmax, row_max = self.calculate_skip_softmax_flag(
                        row_max,
                        tile_row_max,
                        scale_softmax_log2,
                        skip_softmax_threshold_log2,
                        seqlen_q,
                        thread_idx,
                        logical_offset,
                        warp_wants_skip_softmax_exchange,
                        stage,
                        skip_softmax_count,
                        total_softmax_count,
                    )
                si_handle.release()
                # S0 -> P1 / S1 -> P0
                inplace_producer.commit()
                inplace_producer.advance()

        row_max_safe = row_max
        if row_max == -cutlass.Float32.inf:
            row_max_safe = 0.0
        if cutlass.const_expr(self.rescale_threshold > 0.0):
            if (row_max_safe - old_row_max) * scale_softmax_log2 <= self.rescale_threshold:
                row_max_safe = old_row_max
        tTMEM_STORE_VECrS = cute.make_rmem_tensor(tTMEM_STORE_VECcS.shape, self.qk_acc_dtype)
        tTMEM_STORE_VECrS[0] = old_row_max
        tTMEM_STORE_VECrS[1] = row_max_safe
        vec_i_peek_status = si_corr_producer.try_acquire()
        tTMEM_STORErS_x4 = cute.make_rmem_tensor(tTMEM_STOREcS.shape, self.qk_acc_dtype)
        tTMEM_STORErS_x4_e = cute.make_tensor(
            cute.recast_ptr(tTMEM_STORErS_x4.iterator, dtype=self.v_dtype),
            tTMEM_LOADrS.layout,
        )
        scale = scale_softmax_log2
        minus_row_max_scale = (0.0 - row_max_safe) * scale
        if cutlass.const_expr(self.v_dtype.width == 8 and self.p_fp8_prescale_log2 > 0):
            minus_row_max_scale = minus_row_max_scale + self.p_fp8_prescale_log2

        # B300's two softmax warp groups benefit from publishing the peer
        # sequence barrier after 100 of 128 exp2 instructions instead of after
        # 126.
        ARV_COUNT = 28 if self.arch > Arch.sm_100f else 4
        FMA_COUNT = 8
        default_cvt_count = (
            12
            if self.arch > Arch.sm_100f and self.v_dtype.width == 8
            else (8 if self.v_dtype.width == 8 else 4)
        )
        CVT_COUNT = default_cvt_count if self.softmax_cvt_count is None else self.softmax_cvt_count
        CVT_PER_STEP = 4 if self.v_dtype.width == 8 else 2
        assert CVT_COUNT % CVT_PER_STEP == 0, (
            f"CVT_COUNT {CVT_COUNT} must be divisible by CVT_PER_STEP {CVT_PER_STEP}"
        )
        tTMEM_STORErS_x4_e_cvt = cute.logical_divide(
            tTMEM_STORErS_x4_e, cute.make_layout(CVT_PER_STEP)
        )
        if cutlass.const_expr(self.enable_fp16_mixed_softmax):
            # The packed-FP16 exp results alias the already-consumed FP32
            # score registers; no FMA prologue is needed.
            exp_buf = cute.make_tensor(
                cute.recast_ptr(tTMEM_LOADrS.iterator, dtype=Float16),
                tTMEM_LOADrS.layout,
            )
            exp_buf_cvt = cute.logical_divide(exp_buf, cute.make_layout(CVT_PER_STEP))
        else:
            tTMEM_LOADrS_cvt = cute.logical_divide(tTMEM_LOADrS, cute.make_layout(CVT_PER_STEP))
            for i in cutlass.range_constexpr(0, FMA_COUNT, 2):
                tTMEM_LOADrS[i], tTMEM_LOADrS[i + 1] = cute.arch.fma_packed_f32x2(
                    (tTMEM_LOADrS[i], tTMEM_LOADrS[i + 1]),
                    (scale, scale),
                    (minus_row_max_scale, minus_row_max_scale),
                )
        vec_i_handle = si_corr_producer.acquire_and_advance(vec_i_peek_status)
        cute.copy(tiled_tmem_store_vec, tTMEM_STORE_VECrS, tTMEM_STORE_VECtS)
        cute.arch.fence_view_async_tmem_store()
        # Notify correction wg that row_max is ready
        vec_i_handle.commit()

        EXP2_EMULATION_COUNT = 20 if self.enable_ex2_emulation and not whether_apply_mask else 0
        EXP2_EMULATION_OFFSET = cute.size(tTMEM_LOADrS) - EXP2_EMULATION_COUNT
        acc_scale = None
        if cutlass.const_expr(not self.enable_mma_rowsum):
            # With the MMA row sum the correction warp rescales the TMEM
            # accumulator instead; the softmax warp carries no online sum.
            acc_scale_ = scale * (old_row_max - row_max_safe)
            acc_scale = cute.math.exp2(acc_scale_, fastmath=True) * 0.5
        if cutlass.const_expr(self.enable_sequence_barrier):
            self._sequence_barrier(stage).arrive_and_wait()
        if cutlass.const_expr(self.enable_fp16_mixed_softmax):
            # Probe the P-buffer empty barrier before packed softmax so its
            # readiness check overlaps the exp2 and conversion work.
            mixed_pi_peek_status = pi_mma_producer.try_acquire()

        if cutlass.const_expr(enable_skip_softmax):
            if not skip_softmax:
                if cutlass.const_expr(self.enable_mma_rowsum):
                    local_row_sum = (Float32(0.0), Float32(0.0))
                else:
                    row_sum *= acc_scale
                    local_row_sum = (row_sum, row_sum)
                if cutlass.const_expr(self.enable_fp16_mixed_softmax):
                    local_row_sum, inplace_consumer = self.apply_exp_and_cvt_fp16_mixed(
                        tTMEM_LOADrS,
                        exp_buf,
                        exp_buf_cvt,
                        tTMEM_STORErS_x4_e_cvt,
                        stage,
                        scale,
                        minus_row_max_scale,
                        local_row_sum,
                        inplace_consumer,
                        CVT_COUNT,
                        CVT_PER_STEP,
                        ARV_COUNT,
                    )
                else:
                    (
                        local_row_sum,
                        inplace_consumer,
                        regular_pi_peek_status,
                        regular_inplace_peek_status,
                    ) = self.apply_exp_and_cvt(
                        tTMEM_LOADrS,
                        tTMEM_LOADrS_cvt,
                        tTMEM_STORErS_x4_e_cvt,
                        stage,
                        scale,
                        minus_row_max_scale,
                        local_row_sum,
                        pi_mma_producer,
                        inplace_consumer,
                        EXP2_EMULATION_OFFSET,
                        EXP2_EMULATION_COUNT,
                        CVT_COUNT,
                        CVT_PER_STEP,
                        FMA_COUNT,
                        ARV_COUNT,
                    )
                tTMEM_STORE_VECrS_i32 = cute.recast_tensor(tTMEM_STORE_VECrS, dtype=cutlass.Int32)
                tTMEM_STORE_VECrS_i32[0] = 0
                if cutlass.const_expr(self.enable_fp16_mixed_softmax):
                    pi_handle = pi_mma_producer.acquire_and_advance(mixed_pi_peek_status)
                else:
                    pi_handle = pi_mma_producer.acquire_and_advance(regular_pi_peek_status)
                # store skip softmax flag
                cute.copy(
                    tiled_tmem_store_vec,
                    tTMEM_STORE_VECrS_i32,
                    tTMEM_STORE_SKIP_SOFTMAX,
                )
                # store P
                cute.copy(tiled_tmem_store, tTMEM_STORErS_x4, tTMEM_STOREtS_x4)
                cute.arch.fence_view_async_tmem_store()
                pi_handle.commit()
                for j in cutlass.range_constexpr(
                    EXP2_EMULATION_OFFSET,
                    EXP2_EMULATION_OFFSET + EXP2_EMULATION_COUNT,
                    2,
                ):
                    local_row_sum = cute.arch.add_packed_f32x2(
                        (tTMEM_LOADrS[j], tTMEM_LOADrS[j + 1]),
                        local_row_sum,
                    )
                if cutlass.const_expr(self.enable_mma_rowsum):
                    row_sum = Float32(0.0)
                else:
                    row_sum = local_row_sum[0] + local_row_sum[1]
                cute.arch.fence_view_async_tmem_store()
            else:
                if cutlass.const_expr(self.enable_sequence_barrier):
                    self._sequence_barrier(1 - stage).arrive()
                # This path has no exp2 work to hide the probes behind;
                # keep them adjacent to their consuming waits.
                skip_pi_peek_status = pi_mma_producer.try_acquire()
                skip_inplace_peek_status = inplace_consumer.try_wait()
                inplace_consumer.wait_and_advance(skip_inplace_peek_status)
                tTMEM_STORE_VECrS_i32 = cute.recast_tensor(tTMEM_STORE_VECrS, dtype=cutlass.Int32)
                tTMEM_STORE_VECrS_i32[0] = 1
                skip_pi_handle = pi_mma_producer.acquire_and_advance(skip_pi_peek_status)
                # store skip softmax flag
                cute.copy(
                    tiled_tmem_store_vec,
                    tTMEM_STORE_VECrS_i32,
                    tTMEM_STORE_SKIP_SOFTMAX,
                )
                cute.arch.fence_view_async_tmem_store()
                skip_pi_handle.commit()
        else:
            if cutlass.const_expr(self.enable_mma_rowsum):
                local_row_sum = (Float32(0.0), Float32(0.0))
            else:
                row_sum *= acc_scale
                local_row_sum = (row_sum, row_sum)
            if cutlass.const_expr(self.enable_fp16_mixed_softmax):
                local_row_sum, inplace_consumer = self.apply_exp_and_cvt_fp16_mixed(
                    tTMEM_LOADrS,
                    exp_buf,
                    exp_buf_cvt,
                    tTMEM_STORErS_x4_e_cvt,
                    stage,
                    scale,
                    minus_row_max_scale,
                    local_row_sum,
                    inplace_consumer,
                    CVT_COUNT,
                    CVT_PER_STEP,
                    ARV_COUNT,
                )
            else:
                (
                    local_row_sum,
                    inplace_consumer,
                    regular_pi_peek_status,
                    regular_inplace_peek_status,
                ) = self.apply_exp_and_cvt(
                    tTMEM_LOADrS,
                    tTMEM_LOADrS_cvt,
                    tTMEM_STORErS_x4_e_cvt,
                    stage,
                    scale,
                    minus_row_max_scale,
                    local_row_sum,
                    pi_mma_producer,
                    inplace_consumer,
                    EXP2_EMULATION_OFFSET,
                    EXP2_EMULATION_COUNT,
                    CVT_COUNT,
                    CVT_PER_STEP,
                    FMA_COUNT,
                    ARV_COUNT,
                )
            if cutlass.const_expr(self.enable_fp16_mixed_softmax):
                pi_handle = pi_mma_producer.acquire_and_advance(mixed_pi_peek_status)
            else:
                pi_handle = pi_mma_producer.acquire_and_advance(regular_pi_peek_status)
            # store P
            cute.copy(tiled_tmem_store, tTMEM_STORErS_x4, tTMEM_STOREtS_x4)
            cute.arch.fence_view_async_tmem_store()
            for j in cutlass.range_constexpr(
                EXP2_EMULATION_OFFSET,
                EXP2_EMULATION_OFFSET + EXP2_EMULATION_COUNT,
                2,
            ):
                local_row_sum = cute.arch.add_packed_f32x2(
                    (tTMEM_LOADrS[j], tTMEM_LOADrS[j + 1]),
                    local_row_sum,
                )
            if cutlass.const_expr(self.enable_mma_rowsum):
                row_sum = Float32(0.0)
            else:
                row_sum = local_row_sum[0] + local_row_sum[1]
            cute.arch.fence_view_async_tmem_store()
            # Notify tensor core warp that softmax(S->P) is ready
            pi_handle.commit()
        if not is_last_iter:
            si_peek_status = mma_si_consumer.try_wait()

        stats_args = (row_sum, row_max_safe)
        pipeline_args = (
            si_peek_status,
            mma_si_consumer,
            si_corr_producer,
            pi_mma_producer,
            inplace_producer,
            inplace_consumer,
        )
        return stats_args, pipeline_args

    # For both softmax0 and softmax1 warp group
    @cute.jit
    def softmax(
        self,
        stage,
        tensor_args: Tuple,
        pipeline_args: Tuple,
        inplace_args: Tuple,
        qk_thr_mma: cute.ThrMma,
        value_args: Tuple,
        mask_args: Tuple,
        sched_args: Tuple,
    ):
        """Compute softmax on attention scores from QK matrix multiplication.

        This method handles the softmax computation for either the first or second half of the
        attention matrix, depending on the 'stage' parameter. It calculates row-wise maximum
        and sum values needed for stable softmax computation, applies optional masking, and
        transforms raw attention scores into probability distributions.

        The implementation uses specialized memory access patterns and efficient math operations
        for computing exp(x) using exp2 functions. It also coordinates pipeline
        synchronization between MMA, correction, and sequence processing stages.

        :param stage: Processing stage (0 for first half, 1 for second half of attention matrix)
        :type stage: int
        :param seqlen_k: Length of the key sequence
        :type seqlen_k: Int32
        :param seqlen_q: Length of the query sequence
        :type seqlen_q: Int32
        :param cum_seqlen_q: Cumulative sequence lengths for queries
        :type cum_seqlen_q: cute.Tensor | None
        :param cum_seqlen_k: Cumulative sequence lengths for keys
        :type cum_seqlen_k: cute.Tensor | None
        :param scale_softmax_log2: Log2 scale factor for softmax operation
        :type scale_softmax_log2: Float32
        :param qk_thr_mma: Thread MMA operation for QK matrix multiplication
        :type qk_thr_mma: cute.ThrMma
        :param tStS: Shared tensor for softmax input/output
        :type tStS: cute.Tensor
        :param tStSi: Input tensor containing attention scores
        :type tStSi: cute.Tensor
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[Int32]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[Int32]
        :param mma_si_consumer: Pipeline for synchronizing with Si tensors
        :type mma_si_consumer: pipeline.PipelineConsumer
        :param si_corr_producer: Pipeline for synchronizing with correction operations
        :type si_corr_producer: pipeline.PipelineProducer
        :param pi_mma_producer: Pipeline for synchronizing with Pi tensors
        :type pi_mma_producer: pipeline.PipelineProducer
        :param tile_sched_params: Parameters for tile scheduling
        :type tile_sched_params: fmha_utils.FmhaStaticTileSchedulerParams
        :param fused_mask: Compute trip counts and apply masking for attention blocks
        :type fused_mask: fmha_utils.FusedMask
        """
        (
            tStS,
            tStSi,
            cum_seqlen_k,
            cum_seqlen_q,
            kv_len_tensor,
            prefix_tokens_k,
            warp_wants_skip_softmax_exchange,
            skip_softmax_count,
            total_softmax_count,
        ) = tensor_args
        mma_si_consumer, si_corr_producer, pi_mma_producer = pipeline_args
        inplace_producer, inplace_consumer = inplace_args
        (
            seqlen_k,
            seqlen_q,
            scale_softmax_log2,
            skip_softmax_threshold_log2,
        ) = value_args
        window_size_left, window_size_right = mask_args
        (
            tile_sched,
            work_tile,
            clc_pipeline,
            clc_response_ptr,
            clc_consumer_state,
            dynamic_work_pipeline,
            dynamic_work_idx_ptr,
            dynamic_work_consumer_state,
        ) = sched_args

        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax0_warp_ids))

        cS_base = cute.make_identity_tensor((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
        tilePlikeFP32 = self.qk_mma_tiler[1] // 32 * self.v_dtype.width
        tScS = qk_thr_mma.partition_C(cS_base)
        tStS_vec_layout = cute.composition(tStS.layout, cute.make_layout((128, 2)))
        tmem_vec_offset = self.tmem_vec0_offset + stage * (
            self.tmem_vec1_offset - self.tmem_vec0_offset
        )
        tStS_vec = cute.make_tensor(tStS.iterator + tmem_vec_offset, tStS_vec_layout)
        tmem_skip_softmax_offset = self.tmem_skip_softmax0_offset + stage * (
            self.tmem_skip_softmax1_offset - self.tmem_skip_softmax0_offset
        )
        tStS_skip_softmax = cute.make_tensor(
            tStS.iterator + tmem_skip_softmax_offset, tStS_vec_layout
        )
        tScS_vec_layout = cute.composition(tScS.layout, cute.make_layout((128, 2)))
        tScS_vec = cute.make_tensor(tScS.iterator, tScS_vec_layout)
        tStS_P_layout = cute.composition(tStS.layout, cute.make_layout((128, tilePlikeFP32)))
        tmem_p_offset = self.tmem_p0_offset + stage * (self.tmem_p1_offset - self.tmem_p0_offset)
        tStS_P = cute.make_tensor(tStS.iterator + tmem_p_offset, tStS_P_layout)
        if cutlass.const_expr(self.arch >= Arch.sm_100 and self.arch <= Arch.sm_100f):
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
                self.qk_acc_dtype,
            )
        else:
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.LdRed32x32bOp(
                    tcgen05.copy.Repetition(self.softmax_tmem_load_repetition),
                    redOp=tcgen05.TmemLoadRedOp.MAX,
                ),
                self.qk_acc_dtype,
            )

        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStSi)
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        tTMEM_LOADtS = thr_tmem_load.partition_S(tStSi)
        tmem_store_vec_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(2)),
            self.qk_acc_dtype,
        )
        tiled_tmem_store_vec = tcgen05.make_tmem_copy(tmem_store_vec_atom, tStS_vec)
        thr_tmem_store_vec = tiled_tmem_store_vec.get_slice(thread_idx)
        tTMEM_STORE_VECtS = thr_tmem_store_vec.partition_D(tStS_vec)
        tTMEM_STORE_VECcS = thr_tmem_store_vec.partition_S(tScS_vec)
        tTMEM_STORE_SKIP_SOFTMAX = thr_tmem_store_vec.partition_D(tStS_skip_softmax)
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)),
            self.qk_acc_dtype,
        )
        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStS_P)
        thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)
        tTMEM_STOREtS_x4 = thr_tmem_store.partition_D(tStS_P)
        if cutlass.const_expr(self.enable_sequence_barrier):
            if stage == 1:
                self._sequence_barrier(0).arrive()
        if cutlass.const_expr(self.use_dynamic_work_scheduler):
            work_tile, dynamic_work_consumer_state = self._consume_next_dynamic_work(
                dynamic_work_pipeline,
                dynamic_work_idx_ptr,
                dynamic_work_consumer_state,
            )
        while work_tile.is_valid_tile:
            curr_block_coord = work_tile.tile_idx
            batch_coord = curr_block_coord[2][1]
            seqlen_k_ = seqlen_k
            seqlen_q_ = seqlen_q
            continue_cond = False
            cuseqlen_q = Int32(0)
            seqlen_q_ = seqlen_q
            if cutlass.const_expr(cum_seqlen_q is not None):
                cuseqlen_q = cum_seqlen_q[batch_coord]
                seqlen_q_ = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                continue_cond = (
                    not fmha_utils.FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.cta_tiler[0],
                        curr_block_coord[0],
                        seqlen_q_,
                    )
                )
            if not continue_cond:
                if cutlass.const_expr(self.prefix_window):
                    seqlen_k_ = prefix_tokens_k + kv_len_tensor[batch_coord]
                if cutlass.const_expr(cum_seqlen_k is not None and not self.prefix_window):
                    cuseqlen_k = cum_seqlen_k[batch_coord]
                    seqlen_k_ = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                continue_cond = seqlen_k_ <= 0
            if not continue_cond:
                logical_offset = (
                    curr_block_coord[0] * self.cta_tiler[0] + stage * self.qk_mma_tiler[0],
                    0,
                )
                cS = cute.domain_offset(logical_offset, cS_base)
                plan = None
                visible_range = None
                if cutlass.const_expr(self.prefix_window):
                    plan = self._prefix_window_plan(
                        cum_seqlen_k, kv_len_tensor, prefix_tokens_k, batch_coord
                    )
                    visible_range = fmha_utils.PrefixWindowKv.visible_range(plan)
                value_args_ = (
                    seqlen_k_,
                    seqlen_q_,
                    scale_softmax_log2,
                    window_size_left,
                    window_size_right,
                    skip_softmax_threshold_log2,
                    thread_idx,
                    logical_offset,
                    visible_range,
                )
                atom_args = (
                    qk_thr_mma,
                    tiled_tmem_load,
                    tiled_tmem_store,
                    tiled_tmem_store_vec,
                    thr_tmem_load,
                    thr_tmem_store,
                    thr_tmem_store_vec,
                )
                tensor_args_ = (
                    tTMEM_LOADtS,
                    tTMEM_STORE_VECtS,
                    tTMEM_STORE_SKIP_SOFTMAX,
                    tTMEM_STOREtS_x4,
                    warp_wants_skip_softmax_exchange,
                    skip_softmax_count,
                    total_softmax_count,
                )
                if cutlass.const_expr(self.prefix_window):
                    # The masked tiles come first in the plan's virtual order.
                    st_cnt = Int32(0)
                    end_cnt = plan[1]
                    ld_mask_cnt = plan[0]
                    unmask_cnt = plan[1] - plan[0]
                    tl_mask_cnt = Int32(0)
                else:
                    st_cnt, end_cnt, ld_mask_cnt, unmask_cnt, tl_mask_cnt = (
                        fmha_utils.FusedMask.get_masked_info(
                            self.mask_type,
                            curr_block_coord,
                            self.cta_tiler,
                            seqlen_q_,
                            seqlen_k_,
                            window_size_left,
                            window_size_right,
                        )
                    )
                row_max = -Float32.inf
                row_sum = 0.0
                stats_args = (row_sum, row_max)

                def softmax_loop(
                    whether_apply_mask: bool,
                    loop_args: Tuple,
                    stats_args: Tuple,
                    pipeline_args: Tuple,
                    inner_fn: Callable,
                    value_args: Tuple,
                    atom_args: Tuple,
                    tensor_args: Tuple,
                    cS: cute.Tensor,
                ) -> Tuple[Tuple, Tuple]:
                    start_index, iter_num, upper_bound = loop_args
                    for i in cutlass.range(start_index, start_index + iter_num, 1, unroll=1):
                        kv_tile = i
                        if cutlass.const_expr(self.prefix_window):
                            kv_tile = fmha_utils.PrefixWindowKv.tile_of(plan, i)
                        cS_iter = cute.domain_offset((0, kv_tile * self.qk_mma_tiler[1]), cS)
                        iter_args = (cS_iter, i == upper_bound - 1)
                        stats_args, pipeline_args = inner_fn(
                            stage,
                            whether_apply_mask,
                            iter_args,
                            stats_args,
                            pipeline_args,
                            value_args,
                            atom_args,
                            tensor_args,
                        )
                    return stats_args, pipeline_args

                softmax_step_fn = self.softmax_step
                softmax_loop_fn = partial(
                    softmax_loop,
                    inner_fn=softmax_step_fn,
                    value_args=value_args_,
                    atom_args=atom_args,
                    tensor_args=tensor_args_,
                    cS=cS,
                )
                si_peek_status = mma_si_consumer.try_wait()
                if stage == 1:
                    inplace_consumer.wait_and_advance()
                pipeline_args_ = (
                    si_peek_status,
                    mma_si_consumer,
                    si_corr_producer,
                    pi_mma_producer,
                    inplace_producer,
                    inplace_consumer,
                )
                # 1. Leading mask loop
                loop_args = (st_cnt, ld_mask_cnt, end_cnt)
                stats_args, pipeline_args_ = softmax_loop_fn(
                    True, loop_args, stats_args, pipeline_args_
                )
                # 2. Unmasked loop
                loop_args = (st_cnt + ld_mask_cnt, unmask_cnt, end_cnt)
                stats_args, pipeline_args_ = softmax_loop_fn(
                    False, loop_args, stats_args, pipeline_args_
                )
                # 3. Trailing mask loop
                loop_args = (st_cnt + ld_mask_cnt + unmask_cnt, tl_mask_cnt, end_cnt)
                stats_args, pipeline_args_ = softmax_loop_fn(
                    True, loop_args, stats_args, pipeline_args_
                )

                # Unpack pipeline_args
                (
                    _,
                    mma_si_consumer,
                    si_corr_producer,
                    pi_mma_producer,
                    inplace_producer,
                    inplace_consumer,
                ) = pipeline_args_
                if stage == 0:
                    inplace_producer.commit()
                    inplace_producer.advance()
                # 4. Copy the final stats for correction epilog
                tTMEM_STORE_VECrS = cute.make_rmem_tensor(
                    tTMEM_STORE_VECcS.shape, self.qk_acc_dtype
                )
                tTMEM_STORE_VECrS[0] = stats_args[0]
                tTMEM_STORE_VECrS[1] = stats_args[1]
                vec_i_handle = si_corr_producer.acquire_and_advance()
                cute.copy(tiled_tmem_store_vec, tTMEM_STORE_VECrS, tTMEM_STORE_VECtS)
                cute.arch.fence_view_async_tmem_store()
                vec_i_handle.commit()
            # End of if not continue_cond
            # Advance to next tile
            if cutlass.const_expr(self.use_dynamic_work_scheduler):
                work_tile, dynamic_work_consumer_state = self._consume_next_dynamic_work(
                    dynamic_work_pipeline,
                    dynamic_work_idx_ptr,
                    dynamic_work_consumer_state,
                )
            elif cutlass.const_expr(self.use_clc_scheduler):
                work_tile, clc_consumer_state = self._consume_next_clc_work(
                    clc_pipeline,
                    clc_response_ptr,
                    clc_consumer_state,
                )
            else:
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
        # End of persistent scheduler loop

    @cute.jit
    def correction_rescale(
        self,
        thr_mma: cute.ThrMma,
        tiled_tmem_load_vec: cute.TiledCopy,
        scale_softmax_log2: Float32,
        tensor_args: Tuple,
        pipeline_args: Tuple,
        oi_peek_status=None,
        arch_state: Tuple = None,
        o_slot: int = 0,
    ):
        """Rescale intermediate attention results based on softmax normalization factor.

        This method performs a crucial correction step in the attention computation pipeline.
        When processing attention in blocks, the softmax normalization factors may change
        as new blocks are processed. This method rescales previously computed partial
        output values to account for updated normalization factors.

        The implementation uses efficient tensor memory operations to:
        1. Load existing partial attention output from tensor memory
        2. Apply the scaling factor to all elements
        3. Store the rescaled results back to tensor memory

        When ``self.enable_correction_double_buffer`` is True, the rescale loop
        runs as a 2-buffer tensor-memory load / multiply / store pipeline.

        :param thr_mma: Thread MMA operation for the computation
        :type thr_mma: cute.ThrMma
        :param tiled_tmem_load_vec: Tiled memory load operation for the vectorized row-wise max
        :type tiled_tmem_load_vec: cute.TiledCopy
        :param scale_softmax_log2: Log2 of the softmax factor
        :type scale_softmax_log2: Float32
        :param tensor_args: Tuple containing the tensors for the correction
        :type tensor_args: Tuple[cute.Tensor, cute.Tensor, cute.Tensor]
        :param pipeline_args: Tuple containing the pipeline arguments for the correction
        :type pipeline_args: Tuple[pipeline.PipelineConsumer, pipeline.PipelineConsumer]
        :param oi_peek_status: Optional non-blocking token for the Oi consumer
            wait. ``None`` or ``False`` falls back to a blocking wait.
        :param arch_state: Opaque per-correction-warp architecture state (see
            ``_make_correction_arch_state``).
        :param o_slot: Which O accumulator (0 or 1) this rescale operates on.
        :return: ``((si_corr_consumer, mma_corr_consumer), next_oi_peek_status)``
            where ``next_oi_peek_status`` is the peek for the next Oi (only
            refreshed when a rescale actually ran; otherwise stays
            ``cutlass.Boolean(False)``).
        """
        tOtO, tTMEM_LOAD_VECtSi, tTMEM_LOAD_VECcS = tensor_args
        si_corr_consumer, mma_corr_consumer = pipeline_args

        pv_tiled_mma_shape = (
            self.pv_mma_tiler[0],
            self.pv_mma_tiler[1],
        )
        cO = cute.make_identity_tensor(pv_tiled_mma_shape)
        tOcO = thr_mma.partition_C(cO)
        corr_tile_size = self.correction_rescale_tile_size
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tOtO_i_layout = cute.composition(tOtO.layout, cute.make_layout((128, corr_tile_size)))
        tOcO_i_layout = cute.composition(tOcO.layout, cute.make_layout((128, corr_tile_size)))
        tOtO_i = cute.make_tensor(tOtO.iterator, tOtO_i_layout)
        tOcO_i = cute.make_tensor(tOcO.iterator, tOcO_i_layout)
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_i)
        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_i)
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.correction_warp_ids))
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)
        tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i)
        tTMEM_LOADcO = thr_tmem_load.partition_D(tOcO_i)
        tTMEM_STOREtO = thr_tmem_store.partition_D(tOtO_i)
        num_tiles = self.cta_tiler[2] // corr_tile_size
        if cutlass.const_expr(self.enable_correction_double_buffer):
            # Double buffer: 2 register buffers to pipeline tensor-memory load / multiply / store.
            tTMrO = cute.make_rmem_tensor((tTMEM_LOADcO.shape, 2), self.pv_acc_dtype)
            # Rank-matching views for cute.copy (TMEM partitions are rank-3,
            # raw tTMrO[None, idx] is rank-1; composition restores the rank).
            copy_layout = cute.make_layout(tTMrO.shape[0])
            view_0 = tTMrO[None, 0]
            view_1 = tTMrO[None, 1]
            tTMrO_copy = (
                cute.make_tensor(
                    view_0.iterator,
                    cute.composition(view_0.layout, copy_layout),
                ),
                cute.make_tensor(
                    view_1.iterator,
                    cute.composition(view_1.layout, copy_layout),
                ),
            )
        else:
            tTMrO = cute.make_rmem_tensor((tTMEM_LOADcO.shape, num_tiles), self.pv_acc_dtype)
        tTMEM_LOAD_VECrS = cute.make_rmem_tensor(tTMEM_LOAD_VECcS.shape, self.qk_acc_dtype)
        # Wait for vec_i (row_wise current max & previous max)
        vec_i_handle = si_corr_consumer.wait_and_advance()
        cute.copy(tiled_tmem_load_vec, tTMEM_LOAD_VECtSi, tTMEM_LOAD_VECrS)
        cute.arch.fence_view_async_tmem_load()
        vec_i_handle.release()
        # Wait for Oi (peek-token aware: None/False falls back to blocking).
        oi_handle = mma_corr_consumer.wait_and_advance(oi_peek_status)
        next_oi_peek_status = cutlass.Boolean(False)
        vote_ballot_cnt = cute.arch.vote_ballot_sync(tTMEM_LOAD_VECrS[0] != tTMEM_LOAD_VECrS[1])
        should_rescale = vote_ballot_cnt != 0
        if should_rescale:
            scale_ = scale_softmax_log2 * (tTMEM_LOAD_VECrS[0] - tTMEM_LOAD_VECrS[1])
            scale = cute.math.exp2(scale_, fastmath=True)
            if cutlass.const_expr(self.enable_mma_rowsum):
                tiled_tmem_load_rowsum = arch_state[0]
                tiled_tmem_store_rowsum = arch_state[1]
                tTMEM_LOAD_ROWSUMt = arch_state[2] if o_slot == 0 else arch_state[3]
                tTMEM_LOAD_ROWSUMc = arch_state[4]
                tTMEM_STORE_ROWSUMt = arch_state[5] if o_slot == 0 else arch_state[6]
                # C shares O's two-stage mma_corr lifetime. Rescale C while
                # holding the matching token so the next row-sum MMA observes
                # C = old_sum * exp(old_max - new_max) before adding tile_sum.
                tTMEM_LOAD_ROWSUMr = cute.make_rmem_tensor(
                    tTMEM_LOAD_ROWSUMc.shape, self.pv_acc_dtype
                )
                cute.copy(
                    tiled_tmem_load_rowsum,
                    tTMEM_LOAD_ROWSUMt,
                    tTMEM_LOAD_ROWSUMr,
                )
                cute.arch.fence_view_async_tmem_load()
                tTMEM_LOAD_ROWSUMr.store(scale * tTMEM_LOAD_ROWSUMr.load())
                cute.copy(
                    tiled_tmem_store_rowsum,
                    tTMEM_LOAD_ROWSUMr,
                    tTMEM_STORE_ROWSUMt,
                )
            if cutlass.const_expr(self.enable_correction_double_buffer):
                num_elems = cute.size(tTMrO, mode=[0])
                # Prologue: load first tile into buffer 0
                cute.copy(tiled_tmem_load, tTMEM_LOADtO, tTMrO_copy[0])
                # Steady state. Refresh the next Oi probe near the end of the
                # current rescale pipeline.
                for i in cutlass.range_constexpr(1, num_tiles):
                    cute.copy(
                        tiled_tmem_load,
                        cute.make_tensor(
                            tTMEM_LOADtO.iterator + i * corr_tile_size,
                            tTMEM_LOADtO.layout,
                        ),
                        tTMrO_copy[i % 2],
                    )
                    for j in range(0, num_elems, 2):
                        tTMrO[j, (i - 1) % 2], tTMrO[j + 1, (i - 1) % 2] = (
                            cute.arch.mul_packed_f32x2(
                                (
                                    tTMrO[j, (i - 1) % 2],
                                    tTMrO[j + 1, (i - 1) % 2],
                                ),
                                (scale, scale),
                            )
                        )
                    cute.copy(
                        tiled_tmem_store,
                        tTMrO_copy[(i - 1) % 2],
                        cute.make_tensor(
                            tTMEM_STOREtO.iterator + (i - 1) * corr_tile_size,
                            tTMEM_STOREtO.layout,
                        ),
                    )
                next_oi_peek_status = mma_corr_consumer.try_wait()
                # Epilogue: compute and store last tile
                last = (num_tiles - 1) % 2
                for j in range(0, num_elems, 2):
                    tTMrO[j, last], tTMrO[j + 1, last] = cute.arch.mul_packed_f32x2(
                        (tTMrO[j, last], tTMrO[j + 1, last]),
                        (scale, scale),
                    )
                cute.copy(
                    tiled_tmem_store,
                    tTMrO_copy[last],
                    cute.make_tensor(
                        tTMEM_STOREtO.iterator + (num_tiles - 1) * corr_tile_size,
                        tTMEM_STOREtO.layout,
                    ),
                )
            else:
                for i in cutlass.range_constexpr(0, num_tiles):
                    tTMrO_i_ = tTMrO[None, i]
                    tTMrO_i = cute.make_tensor(
                        tTMrO_i_.iterator,
                        cute.composition(tTMrO_i_.layout, cute.make_layout(tTMrO.shape[0])),
                    )
                    cute.copy(
                        tiled_tmem_load,
                        cute.make_tensor(
                            tTMEM_LOADtO.iterator + i * corr_tile_size,
                            tTMEM_LOADtO.layout,
                        ),
                        tTMrO_i,
                    )
                    for j in range(0, cute.size(tTMrO_i), 2):
                        tTMrO_i[j], tTMrO_i[j + 1] = cute.arch.mul_packed_f32x2(
                            (tTMrO_i[j], tTMrO_i[j + 1]),
                            (scale, scale),
                        )
                    cute.copy(
                        tiled_tmem_store,
                        tTMrO_i,
                        cute.make_tensor(
                            tTMEM_STOREtO.iterator + i * corr_tile_size,
                            tTMEM_STOREtO.layout,
                        ),
                    )
                next_oi_peek_status = mma_corr_consumer.try_wait()
        # Release Oi
        cute.arch.fence_view_async_tmem_store()
        oi_handle.release()
        return (si_corr_consumer, mma_corr_consumer), next_oi_peek_status

    @cute.jit
    def correction_epilog(
        self,
        thr_mma: cute.ThrMma,
        tiled_tmem_load_vec: cute.TiledCopy,
        tensor_args: Tuple,
        pipeline_args: Tuple,
        value_args: Tuple,
        arch_state: Tuple = None,
        o_slot: int = 0,
    ):
        """Apply final scaling and transformation to attention output.

        When use_tma_store=True: writes to shared memory and signals epilogue warp for TMA store.
        When use_tma_store=False: writes directly to global memory via st.global.

        :param thr_mma: Thread MMA operation for the computation
        :type thr_mma: cute.ThrMma
        :param tiled_tmem_load_vec: Tiled memory load operation for the vectorized row-wise max
        :type tiled_tmem_load_vec: cute.TiledCopy
        :param tensor_args: Tuple containing (tOtO, tTMEM_LOAD_VECtSi,
                            tTMEM_LOAD_VECcS, sO_or_gO, mLSE, mSink, mO_sf)
        :type tensor_args: Tuple
        :param pipeline_args: When use_tma_store: (si_corr_consumer, mma_corr_consumer, corr_epi_producer).
                              When not use_tma_store: (si_corr_consumer, mma_corr_consumer).
        :type pipeline_args: Tuple
        :param value_args: Tuple containing (row_idx, cuseqlen_q,
                            o_sf_row_offset, seqlen_q, blk_coord,
                            scale_softmax, scale_output, o_sf_scale,
                            o_sf_start_index)
        :type value_args: Tuple
        :param arch_state: Opaque per-correction-warp architecture state (see
            ``_make_correction_arch_state``).
        :param o_slot: Which O accumulator (0 or 1) this epilog operates on.
        """
        (
            tOtO,
            tTMEM_LOAD_VECtSi,
            tTMEM_LOAD_VECcS,
            dest_O,
            mLSE,
            mSink,
            mO_sf,
        ) = tensor_args
        (
            row_idx,
            cuseqlen_q,
            o_sf_row_offset,
            seqlen_q,
            blk_coord,
            scale_softmax,
            scale_output,
            o_sf_scale,
            o_sf_start_index,
        ) = value_args

        pv_tiled_mma_shape = (
            self.pv_mma_tiler[0],
            self.pv_mma_tiler[1],
        )
        cO = cute.make_identity_tensor(pv_tiled_mma_shape)

        corr_tile_size = self.correction_epilog_tile_size
        tOdO = thr_mma.partition_C(dest_O)
        tOcO = thr_mma.partition_C(cO)
        tOtO_i = cute.logical_divide(tOtO, cute.make_layout((128, corr_tile_size)))
        tOcO_i = cute.logical_divide(tOcO, cute.make_layout((128, corr_tile_size)))
        tOdO_i = cute.logical_divide(tOdO, cute.make_layout((128, corr_tile_size)))

        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.correction_warp_ids))
        epi_subtile = (self.epi_tile[0], corr_tile_size)
        tmem_copy_atom = sm100_utils.get_tmem_load_op(
            self.pv_mma_tiler,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=False,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_i[(None, None), 0])
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tTMEM_LOADdO = thr_tmem_load.partition_D(tOdO_i[(None, None), None])
        tTMEM_LOADoO = thr_tmem_load.partition_D(tOcO_i[(None, None), None])

        if cutlass.const_expr(self.use_tma_store):
            si_corr_consumer, mma_corr_consumer, corr_epi_producer = pipeline_args
            smem_copy_atom = sm100_utils.get_smem_store_op(
                self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
            )
            tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)
        else:
            si_corr_consumer, mma_corr_consumer = pipeline_args

        # Wait for vec_i (row_wise global sum)
        vec_i_handle = si_corr_consumer.wait_and_advance()
        tTMEM_LOAD_VECrS = cute.make_rmem_tensor(tTMEM_LOAD_VECcS.shape, self.qk_acc_dtype)
        cute.copy(tiled_tmem_load_vec, tTMEM_LOAD_VECtSi, tTMEM_LOAD_VECrS)
        cute.arch.fence_view_async_tmem_load()
        vec_i_handle.release()

        # Wait for Oi
        oi_handle = mma_corr_consumer.wait_and_advance()
        if cutlass.const_expr(self.use_tma_store):
            oi_final_handle = corr_epi_producer.acquire_and_advance()
        if cutlass.const_expr(self.enable_mma_rowsum):
            tiled_tmem_load_rowsum = arch_state[0]
            tTMEM_LOAD_ROWSUMt = arch_state[2] if o_slot == 0 else arch_state[3]
            tTMEM_LOAD_ROWSUMc = arch_state[4]
            tTMEM_LOAD_ROWSUMr = cute.make_rmem_tensor(tTMEM_LOAD_ROWSUMc.shape, self.pv_acc_dtype)
            cute.copy(
                tiled_tmem_load_rowsum,
                tTMEM_LOAD_ROWSUMt,
                tTMEM_LOAD_ROWSUMr,
            )
            cute.arch.fence_view_async_tmem_load()
            row_sum = tTMEM_LOAD_ROWSUMr[0]
        else:
            row_sum = tTMEM_LOAD_VECrS[0]
        if cutlass.const_expr(mSink is not None):
            sink_val = mSink[blk_coord[2]]
            row_max_raw = tTMEM_LOAD_VECrS[1]
            # sink is already in scaled logit space, row_max_raw is unscaled
            # exp2((sink - max_scaled) * log2(e)) = exp(sink - max_scaled)
            log2_e = Float32(1.4426950408889634)
            sink_exp = cute.math.exp2(
                (sink_val - row_max_raw * scale_softmax) * log2_e, fastmath=True
            )
            if cutlass.const_expr(
                self.enable_mma_rowsum and self.v_dtype.width == 8 and self.p_fp8_prescale_log2 > 0
            ):
                # Row-MMA sums the prescaled FP8 P values, so the sink must be
                # brought into the same domain before final normalization.
                sink_exp = sink_exp * Float32(2**self.p_fp8_prescale_log2)
            row_sum = row_sum + sink_exp
        scale = scale_output / row_sum

        epilog_steps = (
            self.storage_head_dim_v // corr_tile_size
            if self.enable_nvfp4_output
            else self.cta_tiler[2] // corr_tile_size
        )
        epilog_unroll = (
            epilog_steps if self.correction_epilog_unroll is None else self.correction_epilog_unroll
        )
        for i in cutlass.range(0, epilog_steps, 1, unroll=epilog_unroll):
            tTMEM_LOADtO_i = tTMEM_LOADtO[None, 0, 0, i]
            tTMEM_LOADdO_i = tTMEM_LOADdO[None, 0, 0, i]
            tTMrO = cute.make_rmem_tensor(tTMEM_LOADoO[None, 0, 0, i].shape, self.pv_acc_dtype)
            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO)
            for j in range(0, cute.size(tTMrO), 2):
                tTMrO[j], tTMrO[j + 1] = cute.arch.mul_packed_f32x2(
                    (tTMrO[j], tTMrO[j + 1]),
                    (scale, scale),
                )
            if cutlass.const_expr(
                self.enable_nvfp4_output and self.storage_head_dim_v > self.head_dim_v
            ):
                # Logical D72 uses physical D80. The final 16-value block has
                # eight logical values followed by eight defined zero values.
                if i == epilog_steps - 1:
                    tail_values = self.head_dim_v % 16
                    for j in cutlass.range_constexpr(tail_values, cute.size(tTMrO)):
                        tTMrO[j] = Float32(0.0)

            if cutlass.const_expr(self.enable_nvfp4_output):
                # Production NVFP4 contract:
                #   sf = E4M3(o_sf_scale * amax / 6)
                #   q  = E2M1(output * o_sf_scale / float(sf))
                # The inverse multiplier is formed after E4M3 rounding.
                abs_o = cute.math.abs(tTMrO.load())
                amax = abs_o.reduce(cute.ReductionOp.MAX, Float32(0.0), 0)
                sf_f32_reg = cute.make_rmem_tensor((1,), Float32)
                sf_f32_reg[0] = o_sf_scale * amax * Float32(1.0 / 6.0)
                sf_reg = cute.make_rmem_tensor((1,), cutlass.Float8E4M3FN)
                sf_reg.store(sf_f32_reg.load().to(cutlass.Float8E4M3FN))
                sf_value = sf_reg[0]
                rounded_sf = sf_value.to(Float32)
                quant_scale = Float32(0.0)
                if rounded_sf != Float32(0.0):
                    quant_scale = o_sf_scale * cute.arch.rcp_approx(rounded_sf)
                for j in range(0, cute.size(tTMrO), 2):
                    tTMrO[j], tTMrO[j + 1] = cute.arch.mul_packed_f32x2(
                        (tTMrO[j], tTMrO[j + 1]),
                        (quant_scale, quant_scale),
                    )
            if cutlass.const_expr(self.use_tma_store):
                tDMrO = cute.make_rmem_tensor(tTMrO.shape, self.o_dtype)
                o_vec = tTMrO.load()
                tDMrO.store(o_vec.to(self.o_dtype))
                # TMA store path: write to shared memory
                cute.copy(tiled_smem_store, tDMrO, tTMEM_LOADdO_i)
            else:
                # st.global path: write directly to global memory with bounds check
                self._store_o_tile_direct(tTMrO, tTMEM_LOADdO_i, row_idx, seqlen_q)

            if cutlass.const_expr(self.enable_nvfp4_output):
                if row_idx < seqlen_q:
                    # Layout128x4 address transform for
                    # [flattened_token, flattened_head_group].
                    sf_row = o_sf_start_index + o_sf_row_offset + row_idx
                    sf_col = blk_coord[2][0] * (self.storage_head_dim_v // 16) + i
                    sf_row_in_atom = sf_row % 128
                    sf_offset = (
                        (sf_row // 128) * self.o_sf_columns * 128
                        + (sf_col // 4) * 512
                        + (sf_row_in_atom % 32) * 16
                        + (sf_row_in_atom // 32) * 4
                        + sf_col % 4
                    )
                    mO_sf[sf_offset] = sf_value

        if cutlass.const_expr(
            not self.enable_nvfp4_output and self.storage_head_dim_v > self.head_dim_v
        ):
            self._store_o_d80_padding(
                tTMEM_LOADdO,
                tTMEM_LOADoO if self.use_tma_store else None,
                tiled_smem_store if self.use_tma_store else None,
                row_idx,
                seqlen_q,
            )

        if cutlass.const_expr(mLSE is not None):
            scaled_tmp = scale_softmax * tTMEM_LOAD_VECrS[1]
            # Convert LSE from natural log to log2 space, consistent with flashinfer trtllm backend
            lse = (cute.math.log(row_sum, fastmath=True) + scaled_tmp) * Float32(1.4426950408889634)
            # Pre-scale correction: row_sum was inflated by 2^offset, so the
            # log2-space LSE is too large by exactly `p_fp8_prescale_log2`.
            if cutlass.const_expr(self.v_dtype.width == 8 and self.p_fp8_prescale_log2 > 0):
                lse = lse - self.p_fp8_prescale_log2
            if row_idx < seqlen_q:
                mLSE[row_idx + cuseqlen_q, blk_coord[2]] = lse
        if cutlass.const_expr(self.use_tma_store):
            # fence view async shared
            cute.arch.fence_view_async_shared()
            oi_handle.release()
            oi_final_handle.commit()
            return (si_corr_consumer, mma_corr_consumer, corr_epi_producer)
        else:
            oi_handle.release()
            return (si_corr_consumer, mma_corr_consumer)

    def check_supported_dtypes(
        self,
        qk_dtype: Type[cutlass.Numeric],
        pv_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        qk_acc_dtype: Type[cutlass.Numeric],
        pv_acc_dtype: Type[cutlass.Numeric],
    ):
        supported = {cutlass.Float8E4M3FN, cutlass.Float16, cutlass.BFloat16}
        if qk_dtype not in supported:
            raise testing.CantImplementError("Unsupported qk_dtype")
        if pv_dtype not in supported:
            raise testing.CantImplementError("Unsupported pv_dtype")
        if out_dtype not in supported:
            raise testing.CantImplementError("Unsupported out_dtype")
        if qk_acc_dtype not in {cutlass.Float32}:
            raise testing.CantImplementError("Unsupported qk_acc_dtype")
        if pv_acc_dtype not in {cutlass.Float32}:
            raise testing.CantImplementError("Unsupported pv_acc_dtype")

    def check_invalid_shape(
        self,
        qk_dtype: Type[cutlass.Numeric],
        q_shape: Tuple[int, int, int, int],
        k_shape: Tuple[int, int, int, int],
    ):
        # Shapes are passed around this example as (batch, seq_len, num_heads, head_dim).
        b, s_q, h_q, d = q_shape
        b_, s_k, h_k, d_ = k_shape

        if b != b_:
            raise testing.CantImplementError("q & k must have the same batch size")
        if d != d_:
            raise testing.CantImplementError("q & k must have the same head dimension")
        if d != self.head_dim:
            raise testing.CantImplementError(
                "tensor shape does not match the configured logical head dimension"
            )
        if self.head_dim not in self._supported_head_dims:
            raise testing.CantImplementError("Unsupported logical head dimension")
        if h_q % h_k != 0:
            raise testing.CantImplementError("h_q must be divisible by h_k")
        if isinstance(s_q, tuple) and len(s_q) != b:
            raise testing.CantImplementError(
                "variable_seqlen s_q must have the length of batch size"
            )
        if isinstance(s_k, tuple) and len(s_k) != b:
            raise testing.CantImplementError(
                "variable_seqlen s_k must have the length of batch size"
            )
        if d == 192 and qk_dtype not in {cutlass.Float8E4M3FN}:
            raise testing.CantImplementError("unimplemented dtypes for headdim 192")

    def can_implement(
        self,
        q_shape: Tuple[int, int, int, int],
        k_shape: Tuple[int, int, int, int],
        qk_dtype: Type[cutlass.Numeric],
        pv_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        qk_acc_dtype: Type[cutlass.Numeric],
        pv_acc_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """
        :param q_shape: Shape of the query tensor.
        :type q_shape: Tuple[int, int, int, int]
        :param k_shape: Shape of the key tensor.
        :type k_shape: Tuple[int, int, int, int]
        :param qk_dtype: Data type for Q and K (Bmm1 inputs).
        :type qk_dtype: Type[cutlass.Numeric]
        :param pv_dtype: Data type for P and V (Bmm2 inputs).
        :type pv_dtype: Type[cutlass.Numeric]
        :param out_dtype: Data type of the output tensor.
        :type out_dtype: Type[cutlass.Numeric]
        :param qk_acc_dtype: Data type of the qk accumulator tensor.
        :type qk_acc_dtype: Type[cutlass.Numeric]
        :param pv_acc_dtype: Data type of the pv accumulator tensor.
        :type pv_acc_dtype: Type[cutlass.Numeric]
        :return: True if the kernel can be implemented, False otherwise.
        :rtype: bool
        """
        try:
            # Skip unsupported types
            self.check_supported_dtypes(
                qk_dtype,
                pv_dtype,
                out_dtype,
                qk_acc_dtype,
                pv_acc_dtype,
            )
            # Skip invalid shape
            self.check_invalid_shape(
                qk_dtype,
                q_shape,
                k_shape,
            )
        except testing.CantImplementError:
            return False
        return True
