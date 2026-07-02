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

# ruff: noqa: I001, E501, F841, E712, E741

import math
from typing import Callable, Type, Tuple, Union, Optional
from functools import partial

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.nvgpu.common import OperandMajorMode
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.torch as cutlass_torch
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.typing import Int8, Int32, Int64, Float32
from cutlass.base_dsl.arch import Arch
from cutlass.cutlass_dsl import BaseDSL

from ...helpers import fmha_helpers as fmha_utils

"""
A fused multi-head attention (FMHA) example where Bmm1(Q@K) is block-scaled with MXFP8 or NVFP4 for
NVIDIA Blackwell SM100 architecture using CUTE DSL

This example demonstrates an implementation of fused multi-head attention using a TMA + Blackwell SM100
TensorCore warp-specialized persistent kernel. The implementation integrates the Q*K^T matrix multiplication,
softmax normalization, and softmax(Q*K^T)*V into a single kernel, avoiding intermediate data movement between
global memory and shared memory, thus improving computational efficiency.

The kernel implements key optimizations including:
- Warp specialization for different computation phases (load, MMA, softmax, correction, epilogue)
- Pipeline stages between different warps for overlapping computation and memory access
- Support for different precision data types
- Optional causal masking for autoregressive models

To run this example:

.. code-block:: bash

    python nvidia-internal/fmha_blockscaled.py                            \
      --qk_mode MXFP8                                                     \
      --qk_acc_dtype Float32 --pv_acc_dtype Float32                       \
      --mma_tiler_mn 128,128                                              \
      --q_shape 4,1024,8,64 --k_shape 4,1024,8,64                         \
      --is_persistent

The above example runs FMHA with batch size 4, sequence length 1024, 8 attention heads, and head
dimension 64. The Blackwell tcgen05 MMA tile shape is (128, 128), and the default runner uses BF16
for V/output with FP32 for accumulation.

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python nvidia-internal/fmha_blockscaled.py                        \
      --qk_mode NVFP4                                                     \
      --qk_acc_dtype Float32 --pv_acc_dtype Float32                       \
      --mma_tiler_mn 128,128                                              \
      --q_shape 4,1024,8,64 --k_shape 4,1024,8,64                         \
      --is_persistent --warmup_iterations 10                              \
      --iterations 10 --skip_ref_check

Constraints for this example:
* Q and K use supports two microscaling schema:
  MXFP8 (qk_dtype in {Float8E4M3FN, Float8E5M2}, qk_sf_dtype=Float8E8M0FNU, qk_sf_vec_size=32)
  and NVFP4 (qk_dtype=Float4E2M1FN, qk_sf_dtype=Float8E4M3FN, qk_sf_vec_size=16)
* Bmm2 (P@V) remains dense FP8 or BF16 through pv_dtype
* Supported QK head dimensions: 128 (due to current limitations in blockscaled_utils)
* Number of heads in Q must be divisible by number of heads in K
* mma_tiler_mn must be 128,128
* Batch size must be the same for Q, K, and V tensors
* For causal masking, use --is_causal (note: specify without =True/False)
* For persistent scheduling, use --is_persistent (note: specify without =True/False)

For details on the skip softmax algorithm, please refer to the paper: https://arxiv.org/abs/2512.12087.
"""


def make_thread_cooperative_group(size: int):
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, size)


def create_scale_factor_tensor(
    mn: int,
    k: int,
    l: int,
    sf_vec_size: int,
    sf_dtype: Type[cutlass.Numeric],
):
    """
    Create dense reference SFs and blocked device SF storage for QK block-scaled MMA.

    Generates per-position SFs sampled from {0.5, 1.0, 2.0} (which are exactly representable in both
    E8M0 and E4M3) and arranges them into the blocked storage that tile_atom_to_shape_SF,
    BlockScaledBasicChunk consume on the kernel side. The SF atom layout is:
        ((32,4),(sf_vec,4)) : ((16,4),(0,1))
    so for a logical mn in [0,128) inside an atom, the SF byte lives at offset:
        (mn % 32) * 16 + (mn // 32) * 4 + k_inner
    """

    def ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b

    atom_m = 32 * 4
    atom_k = 4
    m_atoms = ceil_div(mn, atom_m)
    sf_k = ceil_div(k, sf_vec_size)
    sf_k_atoms = ceil_div(sf_k, atom_k)

    # Generate SFs in the LOGICAL (mn, sf_k, l) layout first, then re-arrange into the blocked
    # storage layout explained in docstring above.
    # CuTe decomposes mn in [0,128) column-major as (mn % 32, mn // 32), so the size-32 axis
    # gets the LOW 5 bits of mn (stride 16) and the size-4 axis gets the HIGH 2 bits (stride 4).
    mn_padded = m_atoms * atom_m
    sf_k_padded = sf_k_atoms * atom_k
    sf_choices = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
    sf_logical_padded = sf_choices[torch.randint(0, len(sf_choices), (mn_padded, sf_k_padded, l))]
    sf_storage_f32 = (
        # K-mode: k_padded -> (sf_k_atoms, atom_k=4)
        # MN-mode: mn_padded -> (sf_m_atoms, proton_m=4, neutron_m=32)
        sf_logical_padded.reshape(m_atoms, 4, 32, sf_k_atoms, atom_k, l)
        # Arrangement (last-idx-major): (l, sf_m_atoms, sf_k_atoms, neutron_m, proton_m, atom_k)
        # (l, sf_m_atoms, sf_k_atoms) follows a traditional K-major interblock layout
        # (atom_k) is the last index -> reproduces (sf_vec,atom_k):(0,1)
        # (neutron_m, proton_m) -> reproduces
        #   prod((neutron_m,proton_m):(proton_m:1), 1:atom_k) = (32,4):(16,4)
        .permute(5, 0, 3, 2, 1, 4)
        .contiguous()
    )

    sf_tensor, _ = cutlass_torch.cute_tensor_like(
        sf_storage_f32,
        sf_dtype,
        is_dynamic_layout=True,
        assumed_align=16,
    )

    sf_logical = sf_logical_padded[:mn, :sf_k, :]
    # Broadcast each SF across its sf_vec_size K elements to build a elementwise (mn, k, l)
    # reference whose (m, j, b) entry is the SF the kernel multiplies into the same logical position
    ref_elementwise = (
        sf_logical.unsqueeze(2)
        .expand(-1, -1, sf_vec_size, -1)
        .reshape(mn, sf_k * sf_vec_size, l)[:, :k, :]
        .contiguous()
    )

    return ref_elementwise, sf_tensor


def compact_fp4_data(torch_underlying: torch.Tensor, dtype: Type[cutlass.Numeric]) -> None:
    """Compact packed FP4 rows to match CuTe's sub-byte stride interpretation."""
    if dtype is not cutlass.Float4E2M1FN:
        return

    # torch lacks fill_/copy_ kernels for Float4_e2m1fn_x2 on CUDA, but the
    # storage is byte-packed (two FP4 per byte), so a uint8 view is a free
    # reinterpret and supports the in-place primitives we need.
    d = torch_underlying.shape[-1]
    packed_size = d // (8 // dtype.width)
    underlying_u8 = torch_underlying.view(torch.uint8)
    rows = underlying_u8.reshape(-1, d)
    packed = rows[:, :packed_size].contiguous()
    underlying_u8.fill_(0)
    underlying_u8.flatten()[: packed.numel()].copy_(packed.flatten())


class BlackwellFusedMultiHeadBlockScaledAttentionForward:
    arch_str: str = "sm_100"
    arch_name: str = "Blackwell SM100"

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
        qk_sf_vec_size: int,
        use_tma_store: bool = True,
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
        """

        self.qk_acc_dtype = qk_acc_dtype
        self.pv_acc_dtype = pv_acc_dtype
        if mma_tiler != (128, 128):
            raise ValueError(
                "This standalone kernel uses a static TMEM map and currently supports only mma_tiler=(128, 128)"
            )
        if isinstance(head_dim, tuple):
            self.head_dim = head_dim[0]
            self.head_dim_v = head_dim[1]
            assert self.head_dim == 192 and self.head_dim_v == 128, (
                f"When Headdim is a tuple, it's for MLA. Must be (192, 128), but got {head_dim}"
            )
        else:
            self.head_dim = head_dim
            self.head_dim_v = head_dim
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
        self.cluster_shape_mn = (1, 1)
        self.is_persistent = is_persistent
        self.mask_type = mask_type
        self.enable_skip_correction = enable_skip_correction
        self.enable_ex2_emulation = enable_ex2_emulation
        self.qk_sf_vec_size = qk_sf_vec_size
        self.qk_mma_inst_bits_k = 256
        if qk_sf_vec_size == 16:
            # NVFP4: a 256-bit MMA operand tile covers 64 logical K values.
            self.qk_mma_inst_tile_k = self.head_dim // (self.qk_mma_inst_bits_k // 4)
        elif qk_sf_vec_size == 32:
            # MXFP8: a 256-bit MMA operand tile covers 32 logical K values.
            self.qk_mma_inst_tile_k = self.head_dim // (self.qk_mma_inst_bits_k // 8)
        else:
            self.qk_mma_inst_tile_k = 0
        self.use_tma_store = use_tma_store

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
        self.sequence_s0_s1_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.threads_per_warp
            * len((*self.softmax0_warp_ids, *self.softmax1_warp_ids)),
        )
        self.sequence_s1_s0_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_warp
            * len((*self.softmax0_warp_ids, *self.softmax1_warp_ids)),
        )
        self.s0_warpgroup_barrier = pipeline.NamedBarrier(
            barrier_id=5,
            num_threads=self.threads_per_warp * len(self.softmax0_warp_ids),
        )
        self.s1_warpgroup_barrier = pipeline.NamedBarrier(
            barrier_id=6,
            num_threads=self.threads_per_warp * len(self.softmax1_warp_ids),
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
        # Block-scaled QK scale factors are live only while issuing QK MMA.
        # Keep each stage's scale buffers in the opposite S/P tile's tail columns.
        self.tmem_qk0_sf_offset = 224
        self.tmem_qk1_sf_offset = 96
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
        self.num_regs_correction = 96
        self.num_regs_other = 32
        self.buffer_align_bytes = 1024
        self.arch = BaseDSL._get_dsl().get_arch_enum()

        if self.arch >= Arch.sm_103:
            assert self.enable_ex2_emulation == False, (
                f"Don't enable exp2 emulation for {self.arch}, it doesn't help performance"
            )

        num_warps_per_warpgroup = 4
        self.softmax_warpgroup_count = (
            len((*self.softmax0_warp_ids, *self.softmax1_warp_ids)) // num_warps_per_warpgroup
        )

    def _make_qk_tiled_mma(self, cta_group):
        """Build the QK tiled MMA. Override in subclasses to target a different arch."""
        return sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.q_dtype,
            self.k_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_sf_dtype,
            self.qk_sf_vec_size,
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

    def _setup_attributes(self):
        """Set up configurations and parameters for the FMHA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the fused multi-head attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """

        self.q_stage = 2
        k_stage = 4 if self.q_dtype.width == 8 else 3
        v_stage = 4 if self.v_dtype.width == 8 else 3
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

        # Tunable parameters
        self.rescale_threshold = 8.0 if self.enable_skip_correction else 0.0
        # FP8 P pre-scale: offset added to exp2 exponent so that P*2^offset fills
        # more of E4M3's [0, 448] range, improving quantization precision.
        # Derived from rescale_threshold to guarantee P*2^offset <= 448.
        self.p_fp8_prescale_log2 = max(0.0, math.floor(math.log2(448) - self.rescale_threshold))
        # ln(2) * offset correction for LSE when pre-scale is active
        self.p_fp8_prescale_lse_correction = self.p_fp8_prescale_log2 * math.log(2)
        # For most cases, seq barrier is needed to help keep the pipeline stable
        # But sometimes, compiler will schedule the barrier at an unexpected place
        # if it hurts perf a lot, try to quickly fix it by disabling seq barrier
        self.enable_sequence_barrier = False
        # Optional double buffering for correction rescale.
        self.enable_correction_double_buffer = False

    @cute.jit
    def __call__(
        self,
        q_tensor: cute.Tensor,
        k_tensor: cute.Tensor,
        q_sf_tensor: cute.Tensor,
        k_sf_tensor: cute.Tensor,
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
        scale_v_channels: Optional[cute.Tensor],
        skip_softmax_threshold_log2: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        skip_softmax_count: Optional[cute.Tensor],
        total_softmax_count: Optional[cute.Tensor],
        stream: cuda.CUstream,
        use_pdl: bool,
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

        :param q_tensor: The query tensor with shape (b, s_q, h_k, h_r, d)
        :type q_tensor: cute.Tensor
        :param k_tensor: The key tensor with shape (b, s_k, h_k, 1, d)
        :type k_tensor: cute.Tensor
        :param v_tensor: The value tensor with shape (b, s_v, h_k, 1, dv)
        :type v_tensor: cute.Tensor
        :param o_tensor: The output tensor with shape (b, s_q, h_k, h_r, dv)
        :type o_tensor: cute.Tensor
        :param problem_size: The problem size with shape [b, s_q_max, s_lse_max, s_k_max, h_q, h_k, d, dv]. If cum_seqlen_q or cum_seqlen_k is not None, s_q_max and s_k_max are the max of the per-batch sequence lengths respectively.
        :type problem_size: Tuple[Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32]
        :param cum_seqlen_q: The cumulative sequence length tensor for query
        :type cum_seqlen_q: Optional[cute.Tensor]
        :param cum_seqlen_k: The cumulative sequence length tensor for key
        :type cum_seqlen_k: Optional[cute.Tensor]
        :param scale_softmax_log2: The log2 scale factor for softmax
        :type scale_softmax_log2: Float32
        :param scale_softmax: The scale factor for softmax
        :type scale_softmax: Float32
        :param scale_output: The scale factor for the output
        :type scale_output: Float32
        :param window_size_left: Left-side sliding window size for attention masking.
        :type window_size_left: Optional[Int32]
        :param window_size_right: Right-side sliding window size for attention masking.
        :type window_size_right: Optional[Int32]
        :param stream: The CUDA stream to execute the kernel on
        :type stream: cuda.CUstream
        :raises TypeError: If tensor data types don't match or aren't supported
        :raises RuntimeError: If tensor layouts aren't in supported formats
        """
        b, s_q_max, s_lse_max, s_k_max, h_q, h_k, d_, dv_ = problem_size
        h_r = h_q // h_k
        # setup static attributes before smem/grid/tma computation
        self.q_dtype = q_tensor.element_type
        self.k_dtype = k_tensor.element_type
        self.qk_sf_dtype = q_sf_tensor.element_type
        self.v_dtype = v_tensor.element_type
        self.o_dtype = o_tensor.element_type

        # s_q, s_k, s_v are the actual tensor dimensions (total seqlen for varlen)
        s_q = q_tensor.shape[1]
        s_k = k_tensor.shape[1]
        s_v = v_tensor.shape[1]
        s_lse = s_lse_max
        d = self.head_dim
        dv = self.head_dim_v
        # Important for performance
        align = 256 // self.o_dtype.width
        assert d % align == 0, f"head_dim must be multiple of {align} for the given datatypes."
        assert dv % align == 0, f"head_dim_v must be multiple of {align} for the given datatypes."

        stride_b_q = h_r * h_k * s_q * d if cum_seqlen_q is None else 0
        stride_b_o = h_r * h_k * s_q * dv if cum_seqlen_q is None else 0
        stride_b_k = h_k * s_k * d if cum_seqlen_k is None else 0
        stride_b_v = h_k * s_v * dv if cum_seqlen_k is None else 0
        stride_b_lse = h_r * h_k * s_lse if cum_seqlen_q is None else 0

        # (b, s_q, h_k, h_r, d) -> (s_q, d, ((h_r, h_k), b))
        q_layout = cute.make_layout(
            (s_q, d, ((h_r, h_k), b)),
            stride=(d * h_r * h_k, 1, ((d, d * h_r), stride_b_q)),
        )
        q = cute.make_tensor(q_tensor.iterator, q_layout)
        # (b, s_k, h_k, 1, d) -> (s_k, d, ((h_r, h_k), b)), 0-stride for h_r to broadcast
        k_layout = cute.make_layout(
            (s_k, d, ((h_r, h_k), b)),
            stride=(d * h_k, 1, ((0, d), stride_b_k)),
        )
        k = cute.make_tensor(k_tensor.iterator, k_layout)
        q_sf_layout = blockscaled_utils.tile_atom_to_shape_SF(q.shape, self.qk_sf_vec_size)
        q_sf = cute.make_tensor(q_sf_tensor.iterator, q_sf_layout)
        k_sf_layout = blockscaled_utils.tile_atom_to_shape_SF(
            (s_k, d, (h_k, b)), self.qk_sf_vec_size
        )
        k_sf = cute.make_tensor(k_sf_tensor.iterator, k_sf_layout)
        # (b, s_v, h_k, 1, dv) -> (dv, s_v, ((h_r, h_k), b)), 0-stride for h_r to broadcast
        v_layout = cute.make_layout(
            (dv, s_v, ((h_r, h_k), b)),
            stride=(1, dv * h_k, ((0, dv), stride_b_v)),
        )
        v = cute.make_tensor(v_tensor.iterator, v_layout)
        # (b, s_q, h_k, h_r, dv) -> (s_q, dv, ((h_r, h_k), b))
        o_layout = cute.make_layout(
            (s_q, dv, ((h_r, h_k), b)),
            stride=(dv * h_r * h_k, 1, ((dv, dv * h_r), stride_b_o)),
        )
        o = cute.make_tensor(o_tensor.iterator, o_layout)
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

        if cutlass.const_expr(scale_v_channels is not None):
            # scale_v_channels is per (h_k, dv): shape (h_k * dv,) row-major.
            # Expose as (dv, ((h_r, h_k), b)) where h_r and b are 0-stride broadcasts.
            scale_v_channels_layout = cute.make_layout(
                (dv, ((h_r, h_k), b)),
                stride=(1, ((0, dv), 0)),
            )
            m_scale_v_channels = cute.make_tensor(
                scale_v_channels.iterator, scale_v_channels_layout
            )
        else:
            m_scale_v_channels = None

        self.tile_sched_params, grid = fmha_utils.compute_grid(
            cute.shape((s_q_max, d, ((h_r, h_k), b))),
            self.cta_tiler,
            self.is_persistent,
        )
        self.q_major_mode = utils.LayoutEnum.from_tensor(q).mma_major_mode()
        self.k_major_mode = utils.LayoutEnum.from_tensor(k).mma_major_mode()
        self.v_major_mode = utils.LayoutEnum.from_tensor(v).mma_major_mode()
        self.o_layout = utils.LayoutEnum.from_tensor(o)

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
        if cutlass.const_expr(self.qk_sf_dtype != k_sf_tensor.element_type):
            raise TypeError(
                f"Q/K scale type mismatch: {self.qk_sf_dtype} != {k_sf_tensor.element_type}"
            )
        if cutlass.const_expr(self.qk_mma_inst_tile_k <= 0):
            raise RuntimeError(
                f"Invalid QK scale-factor tiling for head_dim={self.head_dim}, qk_sf_vec_size={self.qk_sf_vec_size}"
            )
        # SFQ + SFK share one 32-column TMEM slot per stage (see kernel TMEM layout
        # below). Validate the static footprint on the host so we don't smuggle a
        # `raise` into @cute.kernel.
        _sf_atom_mn = 32
        _qk_sf_tmem_cols = (self.qk_mma_tiler[0] // _sf_atom_mn) * self.qk_mma_inst_tile_k
        _kqk_sf_tmem_cols = (
            ((self.qk_mma_tiler[1] + 127) // 128 * 128) // _sf_atom_mn
        ) * self.qk_mma_inst_tile_k
        if cutlass.const_expr(_qk_sf_tmem_cols + _kqk_sf_tmem_cols > 32):
            raise RuntimeError(
                "QK scale-factor TMEM footprint exceeds the static 32-column slot "
                f"(qk_sf_tmem_cols={_qk_sf_tmem_cols}, k_sf_tmem_cols={_kqk_sf_tmem_cols})"
            )
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

        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.q_dtype,
            self.q_stage,
        )
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.k_dtype,
            self.kv_stage,
        )
        q_sf_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.qk_sf_vec_size,
            self.q_stage,
        )
        k_sf_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.qk_sf_vec_size,
            self.kv_stage,
        )
        p_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            pv_tiled_mma,
            self.pv_mma_tiler,
            self.v_dtype,
            self.acc_stage,
        )
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

        o_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.o_dtype,
            self.o_layout,
            self.epi_tile,
            self.epi_stage,
        )

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
        q_sf_smem_layout = cute.slice_(q_sf_smem_layout_staged, (None, None, None, 0))
        tma_atom_q_sf, tma_tensor_q_sf = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q_sf,
            q_sf_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # TMA load for K
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            k,
            k_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        k_sf_smem_layout = cute.slice_(k_sf_smem_layout_staged, (None, None, None, 0))
        tma_atom_k_sf, tma_tensor_k_sf = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            k_sf,
            k_sf_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        # TMA load for V
        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            v,
            v_smem_layout,
            self.pv_mma_tiler,
            pv_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        o_smem_layout = cute.select(o_smem_layout_staged, mode=[0, 1])
        tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_store_op,
            o,
            o_smem_layout,
            self.epi_tile,
        )

        q_copy_size = cute.size_in_bytes(self.q_dtype, q_smem_layout)
        k_copy_size = cute.size_in_bytes(self.k_dtype, k_smem_layout)
        q_sf_copy_size = cute.size_in_bytes(self.qk_sf_dtype, q_sf_smem_layout)
        k_sf_copy_size = cute.size_in_bytes(self.qk_sf_dtype, k_sf_smem_layout)
        v_copy_size = cute.size_in_bytes(self.v_dtype, v_smem_layout)
        self.tma_copy_q_bytes = q_copy_size + q_sf_copy_size
        self.tma_copy_k_bytes = k_copy_size + k_sf_copy_size
        self.tma_copy_v_bytes = v_copy_size

        @cute.struct
        class SharedStorage:
            # Pipeline barriers
            load_q_mbar_ptr: cute.struct.MemRange[Int64, self.q_stage * 2]
            load_kv_mbar_ptr: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mma_s0_mbar_ptr: cute.struct.MemRange[Int64, self.mma_softmax_stage * 2]
            mma_s1_mbar_ptr: cute.struct.MemRange[Int64, self.mma_softmax_stage * 2]
            p0_mma_mbar_ptr: cute.struct.MemRange[Int64, self.p_mma_stage * 2]
            p1_mma_mbar_ptr: cute.struct.MemRange[Int64, self.p_mma_stage * 2]
            s0_corr_mbar_ptr: cute.struct.MemRange[Int64, self.softmax_corr_stage * 2]
            s1_corr_mbar_ptr: cute.struct.MemRange[Int64, self.softmax_corr_stage * 2]
            corr_epi_mbar_ptr: cute.struct.MemRange[Int64, self.epi_stage * 2]
            mma_corr_mbar_ptr: cute.struct.MemRange[Int64, self.mma_corr_stage * 2]
            s0_p1_inplace_barrier_ptr: cute.struct.MemRange[Int64, self.p_mma_stage * 2]
            s1_p0_inplace_barrier_ptr: cute.struct.MemRange[Int64, self.p_mma_stage * 2]
            # Softmax_{1-j} signals MMA that S_{1-j}'s TMEM region (whose tail
            # columns hold SFQ_j / SFK_j) is no longer being read, so mma_qk's
            # SFQ/SFK S2T copy is safe to issue. One pipeline per QK stage.
            qk_sf_inplace_0_barrier_ptr: cute.struct.MemRange[Int64, 1 * 2]
            qk_sf_inplace_1_barrier_ptr: cute.struct.MemRange[Int64, 1 * 2]
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # Smem tensors
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, cute.cosize(o_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sQSF: cute.struct.Align[
                cute.struct.MemRange[self.qk_sf_dtype, cute.cosize(q_sf_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, sK_cosize],
                self.buffer_align_bytes,
            ]
            sKSF: cute.struct.Align[
                cute.struct.MemRange[self.qk_sf_dtype, cute.cosize(k_sf_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # Skip softmax and PV warpgroup votes
            s0_warp_wants_skip_softmax_exchange: cute.struct.MemRange[Int8, 4]
            s1_warp_wants_skip_softmax_exchange: cute.struct.MemRange[Int8, 4]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            qk_tiled_mma,
            pv_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_q_sf,
            tma_tensor_q_sf,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_k_sf,
            tma_tensor_k_sf,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_o,
            tma_tensor_o,
            o,
            cum_seqlen_q,
            cum_seqlen_k,
            lse,
            sink,
            scale_softmax_log2,
            scale_softmax,
            scale_output,
            m_scale_v_channels,
            skip_softmax_threshold_log2,
            window_size_left,
            window_size_right,
            q_smem_layout_staged,
            k_smem_layout_staged,
            q_sf_smem_layout_staged,
            k_sf_smem_layout_staged,
            p_tmem_layout_staged,
            v_smem_layout_staged,
            o_smem_layout_staged,
            skip_softmax_count,
            total_softmax_count,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=use_pdl,
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        tma_atom_q_sf: cute.CopyAtom,
        mQSF_qdl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_kdl: cute.Tensor,
        tma_atom_k_sf: cute.CopyAtom,
        mKSF_kdl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_dkl: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        mO_qdl: cute.Tensor,
        mO: cute.Tensor,
        cum_seqlen_q: Optional[cute.Tensor],
        cum_seqlen_k: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor],
        mSink: Optional[cute.Tensor],
        scale_softmax_log2: Float32,
        scale_softmax: Float32,
        scale_output: Float32,
        m_scale_v_channels: Optional[cute.Tensor],
        skip_softmax_threshold_log2: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        q_sf_smem_layout_staged: cute.Layout,
        k_sf_smem_layout_staged: cute.Layout,
        p_tmem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        o_smem_layout_staged: cute.ComposedLayout,
        skip_softmax_count: Optional[cute.Tensor],
        total_softmax_count: Optional[cute.Tensor],
        tile_sched_params: fmha_utils.FmhaStaticTileSchedulerParams,
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

        #
        # Prefetch tma desc
        #
        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q_sf)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k_sf)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            if cutlass.const_expr(self.use_tma_store):
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_o)

        # Alloc
        smem = utils.SmemAllocator()
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
        load_kv_full_mbar_ptr = storage.load_kv_mbar_ptr.data_ptr()
        load_kv_empty_mbar_ptr = load_kv_full_mbar_ptr + self.kv_stage
        mma_s0_producer, mma_s0_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_softmax_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax0_warp_ids)
            ),
            barrier_storage=storage.mma_s0_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        mma_s1_producer, mma_s1_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_softmax_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax1_warp_ids)
            ),
            barrier_storage=storage.mma_s1_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        p0_mma_producer, p0_mma_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.p_mma_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax0_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.p0_mma_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        p1_mma_producer, p1_mma_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.p_mma_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax1_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.p1_mma_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        s0_corr_producer, s0_corr_consumer = pipeline.PipelineAsync.create(
            num_stages=self.softmax_corr_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len((*self.softmax0_warp_ids, self.mma_warp_id))
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            barrier_storage=storage.s0_corr_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        s1_corr_producer, s1_corr_consumer = pipeline.PipelineAsync.create(
            num_stages=self.softmax_corr_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len((*self.softmax1_warp_ids, self.mma_warp_id))
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.correction_warp_ids)
            ),
            barrier_storage=storage.s1_corr_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
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
        s0_p1_inplace_producer, s0_p1_inplace_consumer = pipeline.PipelineAsync.create(
            num_stages=self.p_mma_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax0_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax1_warp_ids)
            ),
            barrier_storage=storage.s0_p1_inplace_barrier_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        s1_p0_inplace_producer, s1_p0_inplace_consumer = pipeline.PipelineAsync.create(
            num_stages=self.p_mma_stage,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax0_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax1_warp_ids)
            ),
            barrier_storage=storage.s1_p0_inplace_barrier_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        # QK SF TMEM-slot release pipelines: softmax_{1-j} (producer) signals
        # MMA (consumer) once its T2R-load of S_{1-j} is done, so mma_qk(j)'s
        # SFQ/SFK S2T copy is safe to overwrite the tail of TMEM[S_{1-j}].
        qk_sf_inplace_0_producer, qk_sf_inplace_0_consumer = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax1_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.qk_sf_inplace_0_barrier_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        qk_sf_inplace_1_producer, qk_sf_inplace_1_consumer = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.softmax0_warp_ids)
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.qk_sf_inplace_1_barrier_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        tmem = utils.TmemAllocator(
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
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQSF = storage.sQSF.get_tensor(q_sf_smem_layout_staged)
        # (MMA, MMA_K, MMA_D, PIPE)
        sK = storage.sK.get_tensor(k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        sKSF = storage.sKSF.get_tensor(k_sf_smem_layout_staged)
        # (MMA, MMA_K, MMA_D, PIPE)
        # Reuse k's smem buffer for v. Recast element type so MMA descriptor matches v_dtype.
        sV_ptr = cute.recast_ptr(
            cute.recast_ptr(sK.iterator, dtype=self.v_dtype),
            v_smem_layout_staged.inner,
        )
        sV = cute.make_tensor(sV_ptr, v_smem_layout_staged.outer)
        sO = storage.sO.get_tensor(o_smem_layout_staged.outer, swizzle=o_smem_layout_staged.inner)
        s0_warp_wants_skip_softmax_exchange = (
            storage.s0_warp_wants_skip_softmax_exchange.get_tensor(cute.make_layout((4,)))
        )
        s1_warp_wants_skip_softmax_exchange = (
            storage.s1_warp_wants_skip_softmax_exchange.get_tensor(cute.make_layout((4,)))
        )

        qk_thr_mma = qk_tiled_mma.get_slice(0)  # default 1sm
        pv_thr_mma = pv_tiled_mma.get_slice(0)  # default 1sm
        tSrQ = qk_thr_mma.make_fragment_A(sQ)
        tSrK = qk_thr_mma.make_fragment_B(sK)
        tOrV = pv_thr_mma.make_fragment_B(sV)

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

        tile_sched = fmha_utils.create_fmha_static_tile_scheduler(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
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
            sched_args=(tile_sched, work_tile),
            # Each softmax warpgroup picks its producer by stage: softmax0
            # (stage=0) produces on qk_sf_inplace_1, softmax1 (stage=1) on
            # qk_sf_inplace_0.
            qk_sf_inplace_producers=(
                qk_sf_inplace_1_producer,
                qk_sf_inplace_0_producer,
            ),
        )
        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.empty_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            cute.arch.griddepcontrol_wait()
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
                    if cutlass.const_expr(cum_seqlen_k is not None):
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cum_seqlen_k[batch_coord]
                    continue_cond = seqlen_k <= 0
                if not continue_cond:
                    mQ_qdl_ = mQ_qdl
                    mK_kdl_ = mK_kdl
                    mQSF_qdl_ = mQSF_qdl
                    mKSF_kdl_ = mKSF_kdl
                    mV_dkl_ = mV_dkl
                    if cutlass.const_expr(cum_seqlen_q is not None):
                        mQ_qdl_ = cute.domain_offset(
                            (cum_seqlen_q[batch_coord], 0, ((0, 0), 0)), mQ_qdl
                        )
                        mQSF_qdl_ = cute.domain_offset(
                            (cum_seqlen_q[batch_coord], 0, (0, 0)), mQSF_qdl
                        )
                    if cutlass.const_expr(cum_seqlen_k is not None):
                        mK_kdl_ = cute.domain_offset(
                            (cum_seqlen_k[batch_coord], 0, ((0, 0), 0)), mK_kdl
                        )
                        mKSF_kdl_ = cute.domain_offset(
                            (cum_seqlen_k[batch_coord], 0, (0, 0)), mKSF_kdl
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
                    gQSF_qdl = cute.local_tile(
                        mQSF_qdl_,
                        cute.select(self.qk_mma_tiler, mode=[0, 2]),
                        (None, None, None),
                    )
                    tSgQSF_qdl = qk_thr_mma.partition_A(gQSF_qdl)
                    tQsQSF, tQgQSF_qdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_q_sf,
                        0,  # no multicast
                        cute.make_layout(1),
                        cute.group_modes(sQSF, 0, 3),
                        cute.group_modes(tSgQSF_qdl, 0, 3),
                    )
                    tQsQSF = cute.filter_zeros(tQsQSF)
                    tQgQSF_qdl = cute.filter_zeros(tQgQSF_qdl)
                    # tile_atom_to_shape_SF coalesces Q's ((h_r, h_k), b) mode into (1, h_r*h_k*b),
                    # One needs to reconstruct the correct layout here to correctly map block_coord
                    # onto this linearlized L-like mode.
                    sf_l_q = cute.make_layout(
                        mQ_qdl.shape[2],
                        stride=(
                            (1, mQ_qdl.shape[2][0][0]),
                            cute.size(mQ_qdl.shape[2][0]),
                        ),
                    )(curr_block_coord[2])
                    tQgQSF = tQgQSF_qdl[None, None, 0, sf_l_q]
                    gK_kdl = cute.flat_divide(mK_kdl_, cute.select(self.qk_mma_tiler, mode=[1, 2]))
                    tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
                    tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_k,
                        0,  # no multicast
                        cute.make_layout(1),
                        cute.group_modes(sK, 0, 3),
                        cute.group_modes(tSgK_kdl, 0, 3),
                    )
                    tKgK = tKgK_kdl[None, None, 0, curr_block_coord[2]]
                    gKSF_kdl = cute.local_tile(
                        mKSF_kdl_,
                        cute.select(self.qk_mma_tiler, mode=[1, 2]),
                        (None, None, None),
                    )
                    tSgKSF_kdl = qk_thr_mma.partition_B(gKSF_kdl)
                    tKsKSF, tKgKSF_kdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_k_sf,
                        0,  # no multicast
                        cute.make_layout(1),
                        cute.group_modes(sKSF, 0, 3),
                        cute.group_modes(tSgKSF_kdl, 0, 3),
                    )
                    tKsKSF = cute.filter_zeros(tKsKSF)
                    tKgKSF_kdl = cute.filter_zeros(tKgKSF_kdl)
                    # Same L-mode trick as Q's SF (see comment above). K's SF is shared across h_r heads
                    # (it indexes h_k, not h_q), which we express by a 0-stride for the h_r sub-mode.
                    sf_l_k = cute.make_layout(
                        mK_kdl.shape[2],
                        stride=((0, 1), mK_kdl.shape[2][0][1]),
                    )(curr_block_coord[2])
                    tKgKSF = tKgKSF_kdl[None, None, 0, sf_l_k]
                    gV_dkl = cute.flat_divide(mV_dkl_, cute.select(self.pv_mma_tiler, mode=[1, 2]))
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
                    cute.copy(
                        tma_atom_q_sf,
                        tQgQSF[None, q0_coord],
                        tQsQSF[None, q0_handle.index],
                        tma_bar_ptr=q0_handle.barrier,
                    )
                    seqlen_kv_loop_steps = fmha_utils.FusedMask.get_trip_count(
                        self.mask_type,
                        curr_block_coord,
                        self.cta_tiler,
                        seqlen_q,
                        seqlen_k,
                        window_size_left,
                        window_size_right,
                    )
                    # K0
                    kv_coord = seqlen_kv_loop_start
                    k_handle = load_kv_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_k,
                        tKgK[None, kv_coord],
                        tKsK[None, k_handle.index],
                        tma_bar_ptr=k_handle.barrier,
                    )
                    cute.copy(
                        tma_atom_k_sf,
                        tKgKSF[None, kv_coord],
                        tKsKSF[None, k_handle.index],
                        tma_bar_ptr=k_handle.barrier,
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
                    cute.copy(
                        tma_atom_q_sf,
                        tQgQSF[None, q1_coord],
                        tQsQSF[None, q1_handle.index],
                        tma_bar_ptr=q1_handle.barrier,
                    )
                    kv_coord += 1

                    for i in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
                        # Ki
                        k_handle = load_kv_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_k,
                            tKgK[None, kv_coord],
                            tKsK[None, k_handle.index],
                            tma_bar_ptr=k_handle.barrier,
                        )
                        cute.copy(
                            tma_atom_k_sf,
                            tKgKSF[None, kv_coord],
                            tKsKSF[None, k_handle.index],
                            tma_bar_ptr=k_handle.barrier,
                        )
                        # Vi-1
                        v_handle, load_kv_producer = self.kv_producer_update_tx_acquire_and_advance(
                            load_kv_producer,
                            load_kv_empty_mbar_ptr,
                            load_kv_full_mbar_ptr,
                            self.tma_copy_v_bytes,
                        )
                        cute.copy(
                            tma_atom_v,
                            tVgV[None, kv_coord - 1],
                            tVsV[None, v_handle.index],
                            tma_bar_ptr=load_kv_full_mbar_ptr + v_handle.index,
                        )
                        kv_coord += 1
                    # End of seqlen_kv loop
                    # Vi_end
                    v_handle, load_kv_producer = self.kv_producer_update_tx_acquire_and_advance(
                        load_kv_producer,
                        load_kv_empty_mbar_ptr,
                        load_kv_full_mbar_ptr,
                        self.tma_copy_v_bytes,
                    )
                    cute.copy(
                        tma_atom_v,
                        tVgV[None, kv_coord - 1],
                        tVsV[None, v_handle.index],
                        tma_bar_ptr=load_kv_full_mbar_ptr + v_handle.index,
                    )
                # End of if not continue_cond
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
            q_sf_tmem_layout = blockscaled_utils.make_tmem_layout_sfa(
                qk_tiled_mma,
                self.qk_mma_tiler,
                self.qk_sf_vec_size,
                cute.slice_(q_sf_smem_layout_staged, (None, None, None, 0)),
            )
            k_sf_tmem_layout = blockscaled_utils.make_tmem_layout_sfb(
                qk_tiled_mma,
                self.qk_mma_tiler,
                self.qk_sf_vec_size,
                cute.slice_(k_sf_smem_layout_staged, (None, None, None, 0)),
            )
            # TMEM offsets are in u32 columns. Follow FA4's explicit SFQ footprint
            # calculation instead of deriving the offset from the recast FP8 tensor
            # view, which can under-count for block-scaled layouts.
            sf_atom_mn = 32
            q_sf_tmem_cols = (self.qk_mma_tiler[0] // sf_atom_mn) * self.qk_mma_inst_tile_k
            k_sf_tmem_cols = (
                ((self.qk_mma_tiler[1] + 127) // 128 * 128) // sf_atom_mn
            ) * self.qk_mma_inst_tile_k
            tCtQSF0 = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + self.tmem_qk0_sf_offset, dtype=self.qk_sf_dtype),
                q_sf_tmem_layout,
            )
            tCtKSF0 = cute.make_tensor(
                cute.recast_ptr(
                    tmem_ptr + self.tmem_qk0_sf_offset + q_sf_tmem_cols,
                    dtype=self.qk_sf_dtype,
                ),
                k_sf_tmem_layout,
            )
            tCtQSF1 = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + self.tmem_qk1_sf_offset, dtype=self.qk_sf_dtype),
                q_sf_tmem_layout,
            )
            tCtKSF1 = cute.make_tensor(
                cute.recast_ptr(
                    tmem_ptr + self.tmem_qk1_sf_offset + q_sf_tmem_cols,
                    dtype=self.qk_sf_dtype,
                ),
                k_sf_tmem_layout,
            )
            tiled_copy_s2t_qsf0, tCsQSF_s2t, tCtQSF0_s2t = self.mainloop_s2t_copy_and_partition(
                sQSF, tCtQSF0
            )
            tiled_copy_s2t_ksf0, tCsKSF_s2t, tCtKSF0_s2t = self.mainloop_s2t_copy_and_partition(
                sKSF, tCtKSF0
            )
            tiled_copy_s2t_qsf1, tCsQSF1_s2t, tCtQSF1_s2t = self.mainloop_s2t_copy_and_partition(
                sQSF, tCtQSF1
            )
            tiled_copy_s2t_ksf1, tCsKSF1_s2t, tCtKSF1_s2t = self.mainloop_s2t_copy_and_partition(
                sKSF, tCtKSF1
            )
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
                    if cutlass.const_expr(cum_seqlen_k is not None):
                        cuseqlen_k = cum_seqlen_k[batch_coord]
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                    continue_cond = seqlen_k <= 0
                if not continue_cond:
                    # Wait for Q0
                    q0_handle = load_q_consumer.wait_and_advance()
                    tSrQ0 = tSrQ[None, None, None, q0_handle.index]
                    # Wait for K0
                    k_handle = load_kv_consumer.wait_and_advance()
                    tSrK0 = tSrK[None, None, None, k_handle.index]
                    q0_sf_stage = (None, None, None, None, q0_handle.index)
                    k_sf_stage = (None, None, None, None, k_handle.index)
                    # GEMM_QK00 (Q0 * K0 -> S0)
                    mma_s0_producer, s0_corr_producer, qk_sf_inplace_0_consumer = self.mma_qk(
                        qk_tiled_mma,
                        (tSrQ0, tSrK0, tStS0),
                        (
                            tiled_copy_s2t_qsf0,
                            tCsQSF_s2t[q0_sf_stage],
                            tCtQSF0_s2t,
                            tCtQSF0,
                            tiled_copy_s2t_ksf0,
                            tCsKSF_s2t[k_sf_stage],
                            tCtKSF0_s2t,
                            tCtKSF0,
                        ),
                        (mma_s0_producer, s0_corr_producer),
                        qk_sf_inplace_0_consumer,
                    )
                    # Wait for Q1
                    q1_handle = load_q_consumer.wait_and_advance()
                    tSrQ1 = tSrQ[None, None, None, q1_handle.index]
                    q1_sf_stage = (None, None, None, None, q1_handle.index)
                    # GEMM_QK10 (Q1 * K0 -> S1), K0 is ready in GEMM_QK00
                    mma_s1_producer, s1_corr_producer, qk_sf_inplace_1_consumer = self.mma_qk(
                        qk_tiled_mma,
                        (tSrQ1, tSrK0, tStS1),
                        (
                            tiled_copy_s2t_qsf1,
                            tCsQSF1_s2t[q1_sf_stage],
                            tCtQSF1_s2t,
                            tCtQSF1,
                            tiled_copy_s2t_ksf1,
                            tCsKSF1_s2t[k_sf_stage],
                            tCtKSF1_s2t,
                            tCtKSF1,
                        ),
                        (mma_s1_producer, s1_corr_producer),
                        qk_sf_inplace_1_consumer,
                    )
                    # Release K0
                    k_handle.release()
                    # Note: Q0 & Q1 are still needed in the seqlen_kv loop
                    # so we need to release them after the seqlen_kv loop
                    seqlen_kv_loop_steps = fmha_utils.FusedMask.get_trip_count(
                        self.mask_type,
                        curr_block_coord,
                        self.cta_tiler,
                        seqlen_q,
                        seqlen_k,
                        window_size_left,
                        window_size_right,
                    )
                    # O1 hasn't been accumulated yet, its first MMA calculation doesn't need to accumulate
                    pv_whether_acc = False
                    for i in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
                        # Wait for Ki
                        k_handle = load_kv_consumer.wait_and_advance()
                        tSrKi = tSrK[None, None, None, k_handle.index]
                        k_sf_stage = (None, None, None, None, k_handle.index)
                        # GEMM_QK0i (Q0 * Ki -> S0)
                        mma_s0_producer, s0_corr_producer, qk_sf_inplace_0_consumer = self.mma_qk(
                            qk_tiled_mma,
                            (tSrQ0, tSrKi, tStS0),
                            (
                                tiled_copy_s2t_qsf0,
                                tCsQSF_s2t[q0_sf_stage],
                                tCtQSF0_s2t,
                                tCtQSF0,
                                tiled_copy_s2t_ksf0,
                                tCsKSF_s2t[k_sf_stage],
                                tCtKSF0_s2t,
                                tCtKSF0,
                            ),
                            (mma_s0_producer, s0_corr_producer),
                            qk_sf_inplace_0_consumer,
                        )
                        # Wait for Vi-1
                        v_handle = load_kv_consumer.wait_and_advance()
                        tOrVi = tOrV[None, None, None, v_handle.index]
                        # GEMM_PV0(i-1) (P0 * Vi-1 -> O0_partial)
                        mma_corr_producer, p0_mma_consumer = self.mma_pv(
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
                        )
                        # GEMM_QK1i (Q1 * Ki -> S1)
                        mma_s1_producer, s1_corr_producer, qk_sf_inplace_1_consumer = self.mma_qk(
                            qk_tiled_mma,
                            (tSrQ1, tSrKi, tStS1),
                            (
                                tiled_copy_s2t_qsf1,
                                tCsQSF1_s2t[q1_sf_stage],
                                tCtQSF1_s2t,
                                tCtQSF1,
                                tiled_copy_s2t_ksf1,
                                tCsKSF1_s2t[k_sf_stage],
                                tCtKSF1_s2t,
                                tCtKSF1,
                            ),
                            (mma_s1_producer, s1_corr_producer),
                            qk_sf_inplace_1_consumer,
                        )
                        # Release Ki
                        k_handle.release()
                        # GEMM_PV1(i-1) (P1 * Vi-1 -> O1_partial)
                        mma_corr_producer, p1_mma_consumer = self.mma_pv(
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
                    mma_corr_producer, p0_mma_consumer = self.mma_pv(
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
                    )
                    # GEMM_PV1(i_end) (P1 * Vi_end -> O1)
                    mma_corr_producer, p1_mma_consumer = self.mma_pv(
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
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            # End of persistent scheduler loop
        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue (TMA store path only)
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.epilogue_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            if cutlass.const_expr(self.use_tma_store):
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
                        mO_qdl_ = mO_qdl
                        if cutlass.const_expr(cum_seqlen_q is not None):
                            mO_qdl_ = cute.domain_offset(
                                (cum_seqlen_q[batch_coord], 0, ((0, 0), 0)), mO_qdl
                            )

                        o0_coord = 2 * curr_block_coord[0]
                        o1_coord = o0_coord + 1
                        gO_qdl = cute.flat_divide(
                            mO_qdl_, cute.select(self.pv_mma_tiler, mode=[0, 1])
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
                        cute.copy(tma_atom_o, tOsO[None, 0], tOgO[None, o0_coord])
                        cute.arch.cp_async_bulk_commit_group()
                        # O1
                        # 1. Wait for O1 final
                        o1_handle = corr_epi_consumer.wait_and_advance()
                        # 2. Copy O1 to gmem
                        cute.copy(tma_atom_o, tOsO[None, 1], tOgO[None, o1_coord])
                        cute.arch.cp_async_bulk_commit_group()

                        # Ensure O0 buffer is ready to be released
                        cute.arch.cp_async_bulk_wait_group(1, read=True)
                        o0_handle.release()
                        # Ensure O1 buffer is ready to be released
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                        o1_handle.release()

                    # Advance to next tile
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
                cute.arch.griddepcontrol_launch_dependents()
            # End of persistent scheduler loop
        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax0
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx < self.softmax1_warp_ids[0]:
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            tStS, tStS0, tStS1, tOtO0, tOtO1, tOrP0, tOrP1 = make_tmem_tensors(
                self, qk_thr_mma, pv_thr_mma, p_tmem_layout_staged, tmem_ptr
            )
            softmax_fn(
                stage=0,
                tensor_args=(
                    tStS,
                    tStS0,
                    cum_seqlen_k,
                    cum_seqlen_q,
                    s0_warp_wants_skip_softmax_exchange,
                    skip_softmax_count,
                    total_softmax_count,
                ),
                pipeline_args=(mma_s0_consumer, s0_corr_producer, p0_mma_producer),
                inplace_args=(s0_p1_inplace_producer, s1_p0_inplace_consumer),
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax1
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx < self.correction_warp_ids[0] and warp_idx >= self.softmax1_warp_ids[0]:
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            tStS, tStS0, tStS1, tOtO0, tOtO1, tOrP0, tOrP1 = make_tmem_tensors(
                self, qk_thr_mma, pv_thr_mma, p_tmem_layout_staged, tmem_ptr
            )
            softmax_fn(
                stage=1,
                tensor_args=(
                    tStS,
                    tStS1,
                    cum_seqlen_k,
                    cum_seqlen_q,
                    s1_warp_wants_skip_softmax_exchange,
                    skip_softmax_count,
                    total_softmax_count,
                ),
                pipeline_args=(mma_s1_consumer, s1_corr_producer, p1_mma_producer),
                inplace_args=(s1_p0_inplace_producer, s0_p1_inplace_consumer),
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
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                batch_coord = curr_block_coord[2][1]
                seqlen_k = mK_kdl.shape[0]
                row_idx = Int32(0)
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
                    row_idx = curr_block_coord[0] * self.cta_tiler[0] + tTMEM_LOAD_VECcS[0][0]
                    if cutlass.const_expr(cum_seqlen_k is not None):
                        cuseqlen_k = cum_seqlen_k[batch_coord]
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                    continue_cond = seqlen_k <= 0
                if not continue_cond:
                    # Ignore first signal from softmax as no correction is required
                    vec0_handle = s0_corr_consumer.wait_and_advance()
                    vec0_handle.release()
                    vec1_handle = s1_corr_consumer.wait_and_advance()
                    vec1_handle.release()
                    # O0/O1 share the same mma_corr consumer state, so the Oi
                    # peek token rolls from O0 -> O1 -> next O0. Seed with a
                    # blocking token; the rescale helper refreshes it near the
                    # end of each iteration.
                    oi_peek_status = cutlass.Boolean(False)
                    seqlen_kv_loop_steps = fmha_utils.FusedMask.get_trip_count(
                        self.mask_type,
                        curr_block_coord,
                        self.cta_tiler,
                        seqlen_q,
                        seqlen_k,
                        window_size_left,
                        window_size_right,
                    )
                    for i in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
                        # Rescale O0
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
                        )
                        # Rescale O1
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
                        )
                    # End of seqlen_corr_loop_steps
                    value_args = (
                        cuseqlen_q,
                        seqlen_q,
                        curr_block_coord,
                        scale_softmax,
                        scale_output,
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
                                    m_scale_v_channels,
                                ),
                                (
                                    s0_corr_consumer,
                                    mma_corr_consumer,
                                    corr_epi_producer,
                                ),
                                (row_idx, *value_args),
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
                                    m_scale_v_channels,
                                ),
                                (
                                    s1_corr_consumer,
                                    mma_corr_consumer,
                                    corr_epi_producer,
                                ),
                                (row_idx, *value_args),
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
                                m_scale_v_channels,
                            ),
                            (s0_corr_consumer, mma_corr_consumer),
                            (row_idx, *value_args),
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
                                m_scale_v_channels,
                            ),
                            (s1_corr_consumer, mma_corr_consumer),
                            (row_idx, *value_args),
                        )
                # End of if not continue_cond
                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            if cutlass.const_expr(not self.use_tma_store):
                cute.arch.griddepcontrol_launch_dependents()
            # End of persistent scheduler loop
            tmem.relinquish_alloc_permit()
            # Synchronize before TMEM dealloc (done by the caller)
            self.tmem_dealloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
        return

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """Partition one block-scaled QK scale-factor tensor for SMEM to TMEM copy."""
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)

        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE),
            self.qk_sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    @cute.jit
    def kv_producer_update_tx_acquire_and_advance(
        self, tma_producer, empty_mbar_ptr, full_mbar_ptr, tx_bytes
    ):
        # This utility function is a special version of tma_producer.acquire_and_advance().
        # This is used to customize the tx bytes which is different from
        # the initialized tx bytes of tma_producer.
        state = tma_producer._PipelineProducer__state.clone()
        cute.arch.mbarrier_wait(empty_mbar_ptr + state.index, state.phase)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(
                full_mbar_ptr + state.index,
                tx_bytes,
            )
        tma_producer.advance()
        return state, tma_producer

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
        scale_args: Tuple,
        pipeline_args: Tuple,
        qk_sf_inplace_consumer: pipeline.PipelineConsumer,
        pipeline_tokens: Tuple = (None, None),
    ) -> Tuple[
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineConsumer,
    ]:
        """Perform a single step of the QK GEMM computation on a block of attention scores.

        :param tiled_mma: Tiled MMA for QK GEMM
        :type tiled_mma: cute.TiledMma
        :param tensor_args: Tuple containing Qi, K, and Si
        :type tensor_args: Tuple
        :param pipeline_args: Tuple containing mma_si_producer and si_corr_producer
        :type pipeline_args: Tuple
        :param qk_sf_inplace_consumer: PipelineAsync consumer guarding the SFQ/SFK TMEM slot
            (the tail of the OPPOSITE-stage S region).
        :type qk_sf_inplace_consumer: pipeline.PipelineConsumer
        :param pipeline_tokens: Optional non-blocking peek tokens for the Si and
            vec_i producers, in the form ``(si_peek_status, veci_peek_status)``.
            ``None`` for either token falls back to a blocking acquire.
        :type pipeline_tokens: Tuple
        :return: Tuple containing mma_si_producer, si_corr_producer, and the
            (advanced) qk_sf_inplace_consumer.
        :rtype: Tuple[pipeline.PipelineProducer, pipeline.PipelineProducer,
            pipeline.PipelineConsumer]
        """
        tSrQi, tSrK, tStSi = tensor_args
        (
            tiled_copy_s2t_qsf,
            tCsQSF_stage,
            tCtQSF_s2t,
            tCtQSF,
            tiled_copy_s2t_ksf,
            tCsKSF_stage,
            tCtKSF_s2t,
            tCtKSF,
        ) = scale_args
        mma_si_producer, si_corr_producer = pipeline_args
        si_peek_status, veci_peek_status = pipeline_tokens
        qk_tiled_mma = cutlass.new_from_mlir_values(
            tiled_mma, cutlass.extract_mlir_values(tiled_mma)
        )
        # 0. Make sure Qi & K are ready when calling mma_qk
        # 1. acquire S0
        si_handle = mma_si_producer.acquire_and_advance(si_peek_status)
        # 2. make sure vec is already released in corr
        veci_handle = si_corr_producer.acquire_and_advance(veci_peek_status)
        veci_handle.commit()
        # 3. Wait until softmax_{1-j} has finished T2R-loading S_{1-j} to avoid SFQK_j race cond
        qk_sf_inplace_consumer.wait_and_advance()
        # 4. Copy MXFP8 scale factors and issue block-scaled gemm
        cute.copy(tiled_copy_s2t_qsf, tCsQSF_stage, tCtQSF_s2t)
        cute.copy(tiled_copy_s2t_ksf, tCsKSF_stage, tCtKSF_s2t)
        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        cute.gemm(
            qk_tiled_mma,
            tStSi,
            [tSrQi, tCtQSF],
            [tSrK, tCtKSF],
            tStSi,
        )
        # 4. release S0
        si_handle.commit()
        return mma_si_producer, si_corr_producer, qk_sf_inplace_consumer

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
        softmax_barrier = self.s0_warpgroup_barrier if stage == 0 else self.s1_warpgroup_barrier
        softmax_barrier.arrive_and_wait()
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
        :return: The local row sum and the inplace consumer.
        :rtype: Tuple[float, cute.Tensor]
        """
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
            if cutlass.const_expr(i == EXP2_EMULATION_OFFSET - ARV_COUNT):
                if cutlass.const_expr(self.enable_sequence_barrier):
                    if cutlass.const_expr(stage == 0):
                        self.sequence_s1_s0_barrier.arrive()
                    else:
                        self.sequence_s0_s1_barrier.arrive()

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

        inplace_peek_status = inplace_consumer.try_wait()
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
        return local_row_sum, inplace_consumer

    @cute.jit
    def softmax_step(
        self,
        stage: int,
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
            qk_sf_inplace_producer,
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
        tilePlikeFP32 = self.qk_mma_tiler[1] // Float32.width * self.o_dtype.width
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
            fmha_utils.FusedMask.apply_mask(
                self.mask_type,
                tTMEM_LOADrS,
                tTMEM_LOADcS,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
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
            # Fence the T2R load above, then signal MMA that the opposite-stage S is loaded.
            cute.arch.fence_view_async_tmem_load()
            qk_sf_inplace_producer.commit()
            qk_sf_inplace_producer.advance()
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
                # Fence T2R loads then release the QK SF TMEM slot before we release S to MMA
                cute.arch.fence_view_async_tmem_load()
                qk_sf_inplace_producer.commit()
                qk_sf_inplace_producer.advance()
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
                # Fence the T2R loads then release the QK SF TMEM slot.
                cute.arch.fence_view_async_tmem_load()
                qk_sf_inplace_producer.commit()
                qk_sf_inplace_producer.advance()
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

        ARV_COUNT = 4
        FMA_COUNT = 8
        CVT_COUNT = 8 if self.v_dtype.width == 8 else 4
        CVT_PER_STEP = 4 if self.v_dtype.width == 8 else 2
        assert CVT_COUNT % CVT_PER_STEP == 0, (
            f"CVT_COUNT {CVT_COUNT} must be divisible by CVT_PER_STEP {CVT_PER_STEP}"
        )
        tTMEM_LOADrS_cvt = cute.logical_divide(tTMEM_LOADrS, cute.make_layout(CVT_PER_STEP))
        tTMEM_STORErS_x4_e_cvt = cute.logical_divide(
            tTMEM_STORErS_x4_e, cute.make_layout(CVT_PER_STEP)
        )
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
        acc_scale_ = scale * (old_row_max - row_max_safe)
        acc_scale = cute.math.exp2(acc_scale_, fastmath=True) * 0.5
        if cutlass.const_expr(self.enable_sequence_barrier):
            if cutlass.const_expr(stage == 0):
                self.sequence_s0_s1_barrier.arrive_and_wait()
            else:
                self.sequence_s1_s0_barrier.arrive_and_wait()

        if cutlass.const_expr(enable_skip_softmax):
            if not skip_softmax:
                row_sum *= acc_scale
                local_row_sum = (row_sum, row_sum)
                local_row_sum, inplace_consumer = self.apply_exp_and_cvt(
                    tTMEM_LOADrS,
                    tTMEM_LOADrS_cvt,
                    tTMEM_STORErS_x4_e_cvt,
                    stage,
                    scale,
                    minus_row_max_scale,
                    local_row_sum,
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
                pi_handle = pi_mma_producer.acquire_and_advance()
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
                row_sum = local_row_sum[0] + local_row_sum[1]
                cute.arch.fence_view_async_tmem_store()
            else:
                if cutlass.const_expr(self.enable_sequence_barrier):
                    if cutlass.const_expr(stage == 0):
                        self.sequence_s1_s0_barrier.arrive()
                    else:
                        self.sequence_s0_s1_barrier.arrive()
                inplace_peek_status = inplace_consumer.try_wait()
                inplace_consumer.wait_and_advance(inplace_peek_status)
                tTMEM_STORE_VECrS_i32 = cute.recast_tensor(tTMEM_STORE_VECrS, dtype=cutlass.Int32)
                tTMEM_STORE_VECrS_i32[0] = 1
                pi_handle = pi_mma_producer.acquire_and_advance()
                # store skip softmax flag
                cute.copy(
                    tiled_tmem_store_vec,
                    tTMEM_STORE_VECrS_i32,
                    tTMEM_STORE_SKIP_SOFTMAX,
                )
                cute.arch.fence_view_async_tmem_store()
                pi_handle.commit()
        else:
            row_sum *= acc_scale
            local_row_sum = (row_sum, row_sum)
            local_row_sum, inplace_consumer = self.apply_exp_and_cvt(
                tTMEM_LOADrS,
                tTMEM_LOADrS_cvt,
                tTMEM_STORErS_x4_e_cvt,
                stage,
                scale,
                minus_row_max_scale,
                local_row_sum,
                inplace_consumer,
                EXP2_EMULATION_OFFSET,
                EXP2_EMULATION_COUNT,
                CVT_COUNT,
                CVT_PER_STEP,
                FMA_COUNT,
                ARV_COUNT,
            )
            pi_handle = pi_mma_producer.acquire_and_advance()
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
        stage: int,
        tensor_args: Tuple,
        pipeline_args: Tuple,
        inplace_args: Tuple,
        qk_thr_mma: cute.ThrMma,
        value_args: Tuple,
        mask_args: Tuple,
        sched_args: Tuple,
        qk_sf_inplace_producers: Tuple[pipeline.PipelineProducer, pipeline.PipelineProducer],
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
        tile_sched, work_tile = sched_args

        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax0_warp_ids))

        cS_base = cute.make_identity_tensor((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
        tilePlikeFP32 = self.qk_mma_tiler[1] // 32 * self.o_dtype.width
        tScS = qk_thr_mma.partition_C(cS_base)
        tStS_vec_layout = cute.composition(tStS.layout, cute.make_layout((128, 2)))
        tmem_vec_offset = self.tmem_vec0_offset if stage == 0 else self.tmem_vec1_offset
        tStS_vec = cute.make_tensor(tStS.iterator + tmem_vec_offset, tStS_vec_layout)
        tmem_skip_softmax_offset = (
            self.tmem_skip_softmax0_offset if stage == 0 else self.tmem_skip_softmax1_offset
        )
        tStS_skip_softmax = cute.make_tensor(
            tStS.iterator + tmem_skip_softmax_offset, tStS_vec_layout
        )
        tScS_vec_layout = cute.composition(tScS.layout, cute.make_layout((128, 2)))
        tScS_vec = cute.make_tensor(tScS.iterator, tScS_vec_layout)
        tStS_P_layout = cute.composition(tStS.layout, cute.make_layout((128, tilePlikeFP32)))
        tmem_p_offset = self.tmem_p0_offset if stage == 0 else self.tmem_p1_offset
        tStS_P = cute.make_tensor(tStS.iterator + tmem_p_offset, tStS_P_layout)
        if cutlass.const_expr(self.arch >= Arch.sm_100 and self.arch <= Arch.sm_100f):
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
                self.qk_acc_dtype,
            )
        else:
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.LdRed32x32bOp(
                    tcgen05.copy.Repetition(32), redOp=tcgen05.TmemLoadRedOp.MAX
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
        # Each softmax warpgroup releases the OPPOSITE-stage S region (where
        # SFQ_{1-stage}/SFK_{1-stage} live in the tail) by committing to its qk_sf_inplace producer
        # once per kv block:
        # softmax0 (stage=0) drives qk_sf_inplace_1
        # softmax1 (stage=1) drives qk_sf_inplace_0
        qk_sf_inplace_producer = qk_sf_inplace_producers[stage]
        # Bootstrap: MMA's first mma_qk(stage=0) waits on qk_sf_inplace_0 before
        # softmax1 has produced any in-loop commit. Each thread of softmax1
        # pre-commits once so the first wait succeeds. qk_sf_inplace_1 needs no
        # pre-commit because mma_qk(stage=1) runs after mma_qk(stage=0),
        # giving softmax0 enough time to consume S0 and commit naturally.
        if cutlass.const_expr(stage == 1):
            qk_sf_inplace_producer.commit()
            qk_sf_inplace_producer.advance()
        if cutlass.const_expr(self.enable_sequence_barrier):
            if cutlass.const_expr(stage == 1):
                self.sequence_s0_s1_barrier.arrive()
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
                if cutlass.const_expr(cum_seqlen_k is not None):
                    cuseqlen_k = cum_seqlen_k[batch_coord]
                    seqlen_k_ = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                continue_cond = seqlen_k_ <= 0
            if not continue_cond:
                logical_offset = (
                    curr_block_coord[0] * self.cta_tiler[0] + stage * self.qk_mma_tiler[0],
                    0,
                )
                cS = cute.domain_offset(logical_offset, cS_base)
                value_args_ = (
                    seqlen_k_,
                    seqlen_q_,
                    scale_softmax_log2,
                    window_size_left,
                    window_size_right,
                    skip_softmax_threshold_log2,
                    thread_idx,
                    logical_offset,
                    qk_sf_inplace_producer,
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
                        cS_iter = cute.domain_offset((0, i * self.qk_mma_tiler[1]), cS)
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
                if cutlass.const_expr(stage == 1):
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
                if cutlass.const_expr(stage == 0):
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
        corr_tile_size = 16  # tuneable parameter
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
    ):
        """Apply final scaling and transformation to attention output.

        When use_tma_store=True: writes to shared memory and signals epilogue warp for TMA store.
        When use_tma_store=False: writes directly to global memory via st.global.

        :param thr_mma: Thread MMA operation for the computation
        :type thr_mma: cute.ThrMma
        :param tiled_tmem_load_vec: Tiled memory load operation for the vectorized row-wise max
        :type tiled_tmem_load_vec: cute.TiledCopy
        :param tensor_args: Tuple containing (tOtO, tTMEM_LOAD_VECtSi, tTMEM_LOAD_VECcS, sO_or_gO, mLSE, mSink)
        :type tensor_args: Tuple
        :param pipeline_args: When use_tma_store: (si_corr_consumer, mma_corr_consumer, corr_epi_producer).
                              When not use_tma_store: (si_corr_consumer, mma_corr_consumer).
        :type pipeline_args: Tuple
        :param value_args: Tuple containing (row_idx, cuseqlen_q, seqlen_q, blk_coord, scale_softmax, scale_output)
        :type value_args: Tuple
        """
        (
            tOtO,
            tTMEM_LOAD_VECtSi,
            tTMEM_LOAD_VECcS,
            dest_O,
            mLSE,
            mSink,
            m_scale_v_channels,
        ) = tensor_args
        row_idx, cuseqlen_q, seqlen_q, blk_coord, scale_softmax, scale_output = value_args

        pv_tiled_mma_shape = (
            self.pv_mma_tiler[0],
            self.pv_mma_tiler[1],
        )
        cO = cute.make_identity_tensor(pv_tiled_mma_shape)

        corr_tile_size = 32 * 8 // self.o_dtype.width
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
            row_sum = row_sum + sink_exp
        scale = scale_output / row_sum

        if cutlass.const_expr(m_scale_v_channels is not None):
            scale_v_ch_h = m_scale_v_channels[None, blk_coord[2]]
        for i in range(self.cta_tiler[2] // corr_tile_size):
            tTMEM_LOADtO_i = tTMEM_LOADtO[None, 0, 0, i]
            tTMEM_LOADdO_i = tTMEM_LOADdO[None, 0, 0, i]
            tTMEM_LOADoO_i = tTMEM_LOADoO[None, 0, 0, i]
            tTMrO = cute.make_rmem_tensor(tTMEM_LOADoO_i.shape, self.pv_acc_dtype)
            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO)
            for j in range(0, cute.size(tTMrO), 2):
                tTMrO[j], tTMrO[j + 1] = cute.arch.mul_packed_f32x2(
                    (tTMrO[j], tTMrO[j + 1]),
                    (scale, scale),
                )
            if cutlass.const_expr(m_scale_v_channels is not None):
                for j in range(0, cute.size(tTMrO), 2):
                    _, n0 = tTMEM_LOADoO_i[j]
                    _, n1 = tTMEM_LOADoO_i[j + 1]
                    tTMrO[j], tTMrO[j + 1] = cute.arch.mul_packed_f32x2(
                        (tTMrO[j], tTMrO[j + 1]),
                        (scale_v_ch_h[n0], scale_v_ch_h[n1]),
                    )
            tDMrO = cute.make_rmem_tensor(tTMrO.shape, self.o_dtype)
            o_vec = tTMrO.load()
            tDMrO.store(o_vec.to(self.o_dtype))
            if cutlass.const_expr(self.use_tma_store):
                # TMA store path: write to shared memory
                cute.copy(tiled_smem_store, tDMrO, tTMEM_LOADdO_i)
            else:
                # st.global path: write directly to global memory with bounds check
                if row_idx < seqlen_q:
                    cute.autovec_copy(tDMrO, tTMEM_LOADdO_i)

        if cutlass.const_expr(mLSE is not None):
            scaled_tmp = scale_softmax * tTMEM_LOAD_VECrS[1]
            # Convert LSE from natural log to log2 space, consistent with flashinfer trtllm-gen backend
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
        qk_sf_dtype: Type[cutlass.Numeric],
        qk_sf_vec_size: int,
        qk_acc_dtype: Type[cutlass.Numeric],
        pv_acc_dtype: Type[cutlass.Numeric],
    ):
        if qk_dtype in {cutlass.Float8E4M3FN, cutlass.Float8E5M2}:
            if qk_sf_dtype is not cutlass.Float8E8M0FNU:
                raise NotImplementedError("MXFP8 QK requires qk_sf_dtype Float8E8M0FNU")
            if qk_sf_vec_size != 32:
                raise NotImplementedError("MXFP8 QK requires qk_sf_vec_size 32")
        elif qk_dtype is cutlass.Float4E2M1FN:
            if qk_sf_dtype is not cutlass.Float8E4M3FN:
                raise NotImplementedError("NVFP4 QK requires qk_sf_dtype Float8E4M3FN")
            if qk_sf_vec_size != 16:
                raise NotImplementedError("NVFP4 QK requires qk_sf_vec_size 16")
        else:
            raise NotImplementedError(
                "qk_dtype must be Float8E4M3FN/Float8E5M2 for MXFP8 or Float4E2M1FN for NVFP4"
            )

        if pv_dtype not in {cutlass.Float8E4M3FN, cutlass.BFloat16}:
            raise NotImplementedError("pv_dtype must be Float8E4M3FN or BFloat16")
        if out_dtype not in {cutlass.Float8E4M3FN, cutlass.Float16, cutlass.BFloat16}:
            raise NotImplementedError("Unsupported out_dtype")
        if qk_acc_dtype not in {cutlass.Float32}:
            raise NotImplementedError("Unsupported qk_acc_dtype")
        if pv_acc_dtype not in {cutlass.Float32}:
            raise NotImplementedError("Unsupported pv_acc_dtype")

    def check_invalid_shape(
        self,
        qk_dtype: Type[cutlass.Numeric],
        qk_sf_vec_size: int,
        q_shape: Tuple[int, int, int, int],
        k_shape: Tuple[int, int, int, int],
    ):
        # Shapes are passed around this example as (batch, seq_len, num_heads, head_dim).
        b, s_q, h_q, d = q_shape
        b_, s_k, h_k, d_ = k_shape

        if b != b_:
            raise NotImplementedError("q & k must have the same batch size")
        if d != d_:
            raise NotImplementedError("q & k must have the same head dimension")
        if qk_dtype is cutlass.Float4E2M1FN:
            if d not in {64, 128}:
                raise NotImplementedError("NVFP4 QK supports headdim 64 or 128")
        elif d not in {32, 64, 128}:
            raise NotImplementedError("MXFP8 QK supports headdim 32, 64, or 128")
        if d % qk_sf_vec_size != 0:
            raise NotImplementedError("head dimension must be divisible by qk_sf_vec_size")
        if h_q % h_k != 0:
            raise NotImplementedError("h_q must be divisible by h_k")
        if isinstance(s_q, tuple) and len(s_q) != b:
            raise NotImplementedError("variable_seqlen s_q must have the length of batch size")
        if isinstance(s_k, tuple) and len(s_k) != b:
            raise NotImplementedError("variable_seqlen s_k must have the length of batch size")

    def can_implement(
        self,
        q_shape: Tuple[int, int, int, int],
        k_shape: Tuple[int, int, int, int],
        qk_dtype: Type[cutlass.Numeric],
        pv_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        qk_sf_dtype: Type[cutlass.Numeric],
        qk_sf_vec_size: int,
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
                qk_sf_dtype,
                qk_sf_vec_size,
                qk_acc_dtype,
                pv_acc_dtype,
            )
            # Skip invalid shape
            self.check_invalid_shape(
                qk_dtype,
                qk_sf_vec_size,
                q_shape,
                k_shape,
            )
        except NotImplementedError:
            return False
        return True
