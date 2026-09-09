# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from the CUTLASS CuTe DSL Rubin blockwise GEMM example.

#
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

from typing import Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.rubin_helpers as sm107_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05 import CollectorOp
from cutlass.cutlass_dsl import Int32, T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

from ...blackwell.blockwise_gemm.blockwise_gemm import Sm100BlockwiseGemmKernel


@dsl_user_op
def _store_global_b32(gmem_ptr, value: Int32, *, loc=None, ip=None) -> None:
    gmem_addr = gmem_ptr.toint(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [
            gmem_addr.ir_value(loc=loc, ip=ip),
            Int32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _store_global_b64(gmem_ptr, value_lo: Int32, value_hi: Int32, *, loc=None, ip=None) -> None:
    gmem_addr = gmem_ptr.toint(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [
            gmem_addr.ir_value(loc=loc, ip=ip),
            Int32(value_lo).ir_value(loc=loc, ip=ip),
            Int32(value_hi).ir_value(loc=loc, ip=ip),
        ],
        "{\n  .reg .b64 packed;\n  mov.b64 packed, {$1, $2};\n  st.global.u64 [$0], packed;\n}",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _load_shared_b32(smem_ptr, *, loc=None, ip=None) -> Int32:
    smem_addr = smem_ptr.toint(loc=loc, ip=ip)
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_addr.ir_value(loc=loc, ip=ip)],
            "ld.shared.b32 $0, [$1];",
            "=r,r",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _store_shared_b128(
    smem_ptr,
    value0: Int32,
    value1: Int32,
    value2: Int32,
    value3: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    smem_addr = smem_ptr.toint(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [
            smem_addr.ir_value(loc=loc, ip=ip),
            Int32(value0).ir_value(loc=loc, ip=ip),
            Int32(value1).ir_value(loc=loc, ip=ip),
            Int32(value2).ir_value(loc=loc, ip=ip),
            Int32(value3).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _scale_bf16x4_to_e4m3x4(
    values01: Int32, values23: Int32, scale: Int32, *, loc=None, ip=None
) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int32(values01).ir_value(loc=loc, ip=ip),
                Int32(values23).ir_value(loc=loc, ip=ip),
                Int32(scale).ir_value(loc=loc, ip=ip),
            ],
            "{\n"
            "  .reg .b32 scaled01;\n"
            "  .reg .b32 scaled23;\n"
            "  .reg .b16 lo;\n"
            "  .reg .b16 hi;\n"
            "  mul.rn.bf16x2 scaled01, $1, $3;\n"
            "  mul.rn.bf16x2 scaled23, $2, $3;\n"
            "  cvt.rn.satfinite.e4m3x2.bf16x2 lo, scaled01;\n"
            "  cvt.rn.satfinite.e4m3x2.bf16x2 hi, scaled23;\n"
            "  mov.b32 $0, {lo, hi};\n"
            "}",
            "=r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


"""
A high-performance persistent blockwise dense GEMM
(C = (SFA * A) * (SFB * B)) example for the NVIDIA Rubin SM107 architecture
using CUTE DSL, extending the Blackwell implementation.

Comparison: SM107 (Rubin) vs. SM100 (Blackwell)
- Shared memory (SMEM): 328 KiB on SM107; 228 KiB on SM100
- Tensor memory (TMEM): 576 columns for SM107; 512 columns for SM100
- MMA K dimension: SM107 supports both K=32 and K=64 (SM100 only supports K=32)

- Matrix A is MxKxL, L is batch dimension, A can be row-major("K")
- Matrix B is NxKxL, L is batch dimension, B can be column-major("K")
- Matrix C is MxNxL, L is batch dimension, C can be row-major("N")
- Each block will apply the scale factor SFA
- Each row will apply the scale factor SFB
- For each iteration, the kernel will compute C = A * B and then apply the scale factor C *= SFA * SFB

This GEMM kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Rubin's tcgen05.mma for matrix multiply-accumulate (MMA) operations
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Support persistent tile scheduling to better overlap memory load/store with mma between tiles
    - Support warp specialization to avoid explicit pipelining between mainloop load and mma
    - Support Bkeep-Breuse pattern to reuse B matrix when mma_tiler_m = 2 * mma_inst_shape_m

.. code-block:: bash

    python examples/rubin/blockwise_gemm/blockwise_gemm.py                       \\
      --a_dtype Float8E4M3FN --b_dtype Float8E5M2 --c_dtype BFloat16             \\
      --acc_dtype Float32 --scale_dtype Float32                                  \\
      --mma_tiler 256,128,128 --mma_inst_shape 256,128,64                        \\
      --cluster_shape_mn 2,2                                                     \\
      --mnkl 4096,4096,4096,4

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/rubin/blockwise_gemm/blockwise_gemm.py                   \\
      --a_dtype Float8E4M3FN --b_dtype Float8E5M2 --c_dtype BFloat16             \\
      --acc_dtype Float32 --scale_dtype Float32                                  \\
      --mma_tiler 256,128,128 --mma_inst_shape 256,128,64                        \\
      --cluster_shape_mn 2,2                                                     \\
      --mnkl 4096,4096,4096,4                                                    \\
      --warmup_iterations 1 --iterations 10 --skip_ref_check


Additional constraints:
- Only FP8 inputs are supported (Float8E4M3FN or Float8E5M2) for now
- For K=64: M in the MMA instruction shape must be 128 (1 CTA) or 256 (2 CTAs)
- For K=32: same constraints as Blackwell
"""


class SM107BlockwiseGemmKernel(Sm100BlockwiseGemmKernel):
    """Persistent blockwise dense GEMM kernel for Rubin.

    Extends `Sm100BlockwiseGemmKernel` with SM107-specific behavior and limits.

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param use_2cta_instrs: Whether to use CTA group 2 for advanced thread cooperation
    :type use_2cta_instrs: bool
    :param mma_tiler: MMA tile shape (M, N, K). K may be 32 or 64 on SM107
    :type mma_tiler: Tuple[int, int, int]
    :param mma_inst_shape: MMA instruction shape (M, N, K)
    :type mma_inst_shape: Tuple[int, int, int]
    :param cluster_shape_mn: Cluster dimensions (M, N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]

    :note: Supported A/B data types:
        - Float8E4M3FN
        - Float8E5M2

    :note: Supported accumulator data types:
        - Float32

    :note: Supported C data types:
        - Float16/BFloat16
        - Float32

    :note: Constraints:
        - MMA tiler M must be 64/128/256/512
        - MMA tiler N must be 128, align with the scaleB requirement
        - K=64 constraint: M in the MMA instruction shape must be 128 (1 CTA) or 256 (2 CTAs)
        - Cluster shape M must be multiple of 2 if use_2cta_instrs=True
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16
        - Resources: larger SMEM (328 KiB) and TMEM (576 columns)

    **Example:**

    .. code-block:: python

        gemm = SM107BlockwiseGemmKernel(
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=True,
            mma_tiler=(256, 128, 128),
            mma_inst_shape=(256, 128, 64),
            cluster_shape_mn=(2, 1),
        )
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler: Tuple[int, int, int],
        mma_inst_shape: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        fp8_quantize_1x128: bool = False,
    ):
        """Initialize the Rubin persistent blockwise dense GEMM kernel.

        :param acc_dtype: Data type of the accumulator.
        :type acc_dtype: type[cutlass.Numeric]
        :param use_2cta_instrs: Boolean, True to use cta_group=2 MMA variant.
        :type use_2cta_instrs: bool
        :param mma_tiler: MMA tiler (M, N, K).
        :type mma_tiler: Tuple[int, int, int]
        :param mma_inst_shape: MMA instruction shape (M, N, K).
        :type mma_inst_shape: Tuple[int, int, int]
        :param cluster_shape_mn: Tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: Tuple[int, int]
        """
        # Call parent constructor with mma_tiler_mn (M, N only)
        super().__init__(
            acc_dtype,
            use_2cta_instrs,
            (mma_tiler[0], mma_tiler[1]),
            cluster_shape_mn,
        )
        # Match the register split used by the Rubin source kernel's Blackwell
        # base. TRT-LLM's vendored Blackwell kernel uses a balanced 216/216
        # split, while Rubin dedicates more registers to accumulator updates.
        self.num_regs_epilogue_warps = 168
        self.num_regs_acc_update_warps = 256
        self.arch = "sm_107"
        self.num_smem_capacity = utils.get_smem_capacity_in_bytes(self.arch)
        self.mma_tiler = mma_tiler
        self.mma_inst_shape = mma_inst_shape
        self.fp8_quantize_1x128 = fp8_quantize_1x128
        # Bkeep-Breuse pattern is controlled by mma_inst_shape and mma_tiler
        self.enable_breuse = True if mma_tiler[0] // mma_inst_shape[0] == 2 else False
        # TMEM allocation columns for SM107
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(self.arch)

    def _get_mma_permutation_mnk(self):
        """Get MMA permutation for Bkeep-Breuse pattern.

        Fold the 2x B-reuse factor (and the CTA split, for 2CTA) into the
        atom M layout so that sA (built from full mma_tiler) and gA (tiled
        per full mma_tiler) agree on MMA_M in tma_partition. This is needed
        for both 2CTA+breuse and non-2CTA+breuse paths.
        """
        if cutlass.const_expr(self.enable_breuse):
            cta_group = 2 if self.use_2cta_instrs else 1
            m_layout = cute.make_layout(
                shape=(self.mma_inst_shape[0] // cta_group, cta_group, 2),
                stride=(1, self.mma_inst_shape[0], self.mma_inst_shape[0] // cta_group),
            )
            return (m_layout, self.mma_inst_shape[1], self.mma_inst_shape[2])
        else:
            return (1, 1, 1)

    def _create_tiled_mma(self):
        """Create TiledMma for SM107."""
        return sm107_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_inst_shape,
            permutation_mnk=self._get_mma_permutation_mnk(),
        )

    def _create_tiled_mma_bkeep(self):
        """Create TiledMma for keep operation (with fill collector for B).

        This is used in the Bkeep-Breuse pattern for the first GEMM operation.
        The 'fill' collector operation indicates that B data should be kept
        for reuse in subsequent operations.
        """
        return sm107_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_inst_shape,
            permutation_mnk=self._get_mma_permutation_mnk(),
            b_collector_op=CollectorOp.FILL,
        )

    def _create_tiled_mma_breuse(self):
        """Create TiledMma for reuse operation (with lastuse collector for B).

        This is used in the Bkeep-Breuse pattern for the second GEMM operation.
        The 'lastuse' collector operation indicates that this is the last use
        of the B data that was kept from the previous operation.
        """
        return sm107_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_inst_shape,
            permutation_mnk=self._get_mma_permutation_mnk(),
            b_collector_op=CollectorOp.LASTUSE,
        )

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs.

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C stage counts in shared memory
        - Computing A/B/C shared memory layout
        - Computing tensor memory allocation columns
        """
        # Configure tiled mma
        tiled_mma = self._create_tiled_mma()

        # Compute mma/cluster/tile shapes
        self.mma_inst_tile_k = self.mma_tiler[2] // self.mma_inst_shape[2]

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        if self.fp8_quantize_1x128 and self.cta_tile_shape_mnk[:2] != (128, 128):
            raise ValueError("Fused FP8 quantization requires a 128x128 CTA output tile")

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        # Scale granularity settings
        self.scale_granularity_m = 1
        self.scale_granularity_n = 128
        self.scale_granularity_k = 128
        self.scale_m_per_tile = self.cta_tile_shape_mnk[0] // self.scale_granularity_m
        self.scale_n_per_tile = self.cta_tile_shape_mnk[1] // self.scale_granularity_n
        self.scale_k_per_tile = self.cta_tile_shape_mnk[2] // self.scale_granularity_k

        if self.scale_k_per_tile != 1:
            raise ValueError("scale_k_per_tile must be 1")
        if self.scale_m_per_tile != self.cta_tile_shape_mnk[0]:
            raise ValueError("scale_m_per_tile must be cta_tile_m")
        if self.scale_n_per_tile != 1:
            raise ValueError("scale_n_per_tile must be 1")

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Compute epilogue subtile
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        # Setup A/B/C/Scale stage count in shared memory and ACC stage count in tensor memory
        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_c_stage,
            self.num_scale_stage,
            self.num_tile_stage,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sfa_dtype,
            self.sfb_dtype,
            self.scale_m_per_tile * self.scale_k_per_tile,
            self.scale_n_per_tile * self.scale_k_per_tile,
            self.num_smem_capacity,
            self.occupancy,
        )
        if self.fp8_quantize_1x128:
            # The direct fused epilogue reuses this allocation as one
            # row-major 128x128 FP8 scratch tile (16 KiB). Two 128x32 BF16
            # epilogue stages provide exactly that capacity.
            self.num_c_stage = 2

        # SM107 TMEM (576 cols) can not fit multiple staging stages when
        # MMA_M=2 (cta_tile_M==256), so only keep one staging + one final.
        if self.cta_tile_shape_mnk[0] == 256:
            self.num_acc_stage = 1
        elif self.cta_tile_shape_mnk[0] == 128:
            self.num_acc_stage = 3
        else:
            self.num_acc_stage = min(self.num_acc_stage, 6)

        # Compute A/B/C/Scale shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )
        self.sfa_smem_layout_staged = cute.make_layout(
            (
                (self.scale_granularity_m, self.scale_m_per_tile),
                (self.scale_granularity_k, self.scale_k_per_tile),
                self.num_scale_stage,
            ),
            stride=(
                (0, self.scale_k_per_tile),
                (0, 1),
                self.scale_k_per_tile * self.scale_m_per_tile,
            ),
        )
        self.sfb_smem_layout_staged = cute.make_layout(
            (
                (self.scale_granularity_n, self.scale_n_per_tile),
                (self.scale_granularity_k, self.scale_k_per_tile),
                self.num_scale_stage,
            ),
            stride=(
                (0, self.scale_k_per_tile),
                (0, 1),
                self.scale_k_per_tile * self.scale_n_per_tile,
            ),
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
        fp8_scale: cute.Tensor = None,
    ):
        """Execute the GEMM operation in steps for SM107.

        This overrides the parent method to use SM107-specific tiled_mma creation.

        :param a: Input tensor A
        :type a: cute.Tensor
        :param b: Input tensor B
        :type b: cute.Tensor
        :param c: Output tensor C
        :type c: cute.Tensor
        :param sfa: Scale factor tensor A
        :type sfa: cute.Tensor
        :param sfb: Scale factor tensor B
        :type sfb: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        :param epilogue_op: Optional elementwise lambda function to apply to the output tensor
        :type epilogue_op: cutlass.Constexpr
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        """
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = (
            cutlass.BFloat16 if self.fp8_quantize_1x128 else c.element_type
        )
        self.sfa_dtype: Type[cutlass.Numeric] = sfa.element_type
        self.sfb_dtype: Type[cutlass.Numeric] = sfb.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(self.fp8_quantize_1x128):
            if cutlass.const_expr(c.element_type is not cutlass.Float8E4M3FN):
                raise TypeError("Fused FP8 quantization requires Float8E4M3FN C")
            if cutlass.const_expr(fp8_scale is None or fp8_scale.element_type is not cutlass.Uint8):
                raise TypeError("Fused FP8 quantization requires Uint8 output scales")
            # The fused epilogue stores whole 128-column tiles with row-only
            # predication, so C must be N-contiguous and N a multiple of 128
            # (checked in can_implement).
            if cutlass.const_expr(self.c_layout != utils.LayoutEnum.ROW_MAJOR):
                raise TypeError("Fused FP8 quantization requires a row-major (N-contiguous) C")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        # Use SM107-specific tiled_mma creation
        tiled_mma = self._create_tiled_mma()
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Create bkeep/breuse tiled_mma variants for SM107
        tiled_mma_bkeep = self._create_tiled_mma_bkeep()
        tiled_mma_breuse = self._create_tiled_mma_breuse()

        # Setup TMA load for A
        a_op = self._get_tma_atom_kind(atom_thr_size, self.is_a_mcast)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(cutlass.TFloat32 if a.element_type is cutlass.Float32 else None),
        )

        # Setup TMA load for B
        b_op = self._get_tma_atom_kind(atom_thr_size, self.is_b_mcast)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(cutlass.TFloat32 if b.element_type is cutlass.Float32 else None),
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        # Setup TMA store for C
        tma_atom_c = None
        tma_tensor_c = None
        if cutlass.const_expr(not self.fp8_quantize_1x128):
            c_cta_v_layout = cute.composition(cute.make_identity_layout(c.shape), self.epi_tile)
            epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
            tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                c,
                epi_smem_layout,
                c_cta_v_layout,
            )
        else:
            tma_tensor_c = c

        tensor_sfa = cute.make_tensor(
            sfa.iterator,
            cute.make_layout(
                (
                    (self.scale_granularity_m, sfa.shape[0]),
                    (self.scale_granularity_k, sfa.shape[1]),
                    sfa.shape[2],
                ),
                stride=(
                    (0, sfa.layout.stride[0]),
                    (0, sfa.layout.stride[1]),
                    sfa.layout.stride[2],
                ),
            ),
        )
        tensor_sfb = cute.make_tensor(
            sfb.iterator,
            cute.make_layout(
                (
                    (self.scale_granularity_n, sfb.shape[0]),
                    (self.scale_granularity_k, sfb.shape[1]),
                    sfb.shape[2],
                ),
                stride=(
                    (0, sfb.layout.stride[0]),
                    (0, sfb.layout.stride[1]),
                    sfb.layout.stride[2],
                ),
            ),
        )

        # Compute grid size
        self.tile_sched_params, grid = self._compute_grid(
            c, self.cta_tile_shape_mnk, self.cluster_shape_mn, max_active_clusters
        )

        self.buffer_align_bytes = 1024

        c_smem_size = cute.cosize(self.c_smem_layout_staged.outer)

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            # (bidx, bidy, bidz, valid)
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 4 * self.num_tile_stage],
                # 1 byte alignment
                1,
            ]
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            scale_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_scale_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_tile_stage * 2]
            epi_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1 * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    c_smem_size,
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (granularity_m, repeat_m), (granularity_k, repeat_k), num_scale_stage)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sfa_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sfb_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tiled_mma_bkeep,
            tiled_mma_breuse,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            tensor_sfa,
            tensor_sfb,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
            fp8_scale,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )
        return

    # GPU device kernel with SM107 Bkeep-Breuse support
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_bkeep: cute.TiledMma,
        tiled_mma_breuse: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        mSFA_mkl: cute.Tensor,
        mSFB_nkl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        fp8_scale: cute.Tensor,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation for SM107.

        This kernel implements SM107-specific features including the Bkeep-Breuse pattern
        for efficient B matrix reuse when enable_breuse is True.

        :param tiled_mma: Standard TiledMma for non-breuse path
        :param tiled_mma_bkeep: TiledMma with FILL collector op for B matrix keep
        :param tiled_mma_breuse: TiledMma with LASTUSE collector op for B matrix reuse
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_idx = cute.arch.lane_idx()

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            if cutlass.const_expr(not self.fp8_quantize_1x128):
                cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        # CUTLASS DSL 4.5 expresses multicast signaling through the consumer
        # thread count; Agent.Warp with enable_multicast_signaling=True needs a
        # newer release.
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Initialize mainloop scale_pipeline (barrier) and states
        scale_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 1,
        )
        scale_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * len(self.epilog_warp_id),
        )
        scale_pipeline = pipeline.PipelineCpAsync.create(
            barrier_storage=storage.scale_mbar_ptr.data_ptr(),
            num_stages=self.num_scale_stage,
            producer_group=scale_pipeline_producer_group,
            consumer_group=scale_pipeline_consumer_group,
            defer_sync=True,
        )

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Initialize epilogue pipeline (barrier) and states
        epi_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * len(self.acc_update_warp_id),
        )
        epi_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * len(self.epilog_warp_id),
        )
        epi_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.epi_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=epi_pipeline_producer_group,
            consumer_group=epi_pipeline_consumer_group,
            defer_sync=True,
        )

        # Initialize tile info pipeline (barrier) and states
        tile_info_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 1,
        )
        tile_info_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_wo_sched,
        )
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.tile_info_mbar_ptr.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=tile_info_pipeline_producer_group,
            consumer_group=tile_info_pipeline_consumer_group,
            defer_sync=True,
        )

        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
            arch=self.arch,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        #
        # Setup smem tensor A/B/C/Scale
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        # (granularity_m, repeat_m), (granularity_k, repeat_k), num_scale_stage)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        # (bidx, bidy, bidz, valid)
        info_layout = cute.make_layout((4, self.num_tile_stage), stride=(1, 4))
        sInfo = storage.sInfo.get_tensor(info_layout)

        #
        # Compute multicast mask for A/B buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, loopM, loopK, loopL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, loopN, loopK, loopL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bN, loopM, loopN, loopL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        # (bM, bK, loopM, loopK, loopL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl,
            cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        # (bN, bK, loopN, loopK, loopL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.cta_tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        # coordinate
        cC_mnl = cute.make_identity_tensor(cute.shape(mC_mnl))
        cSFA_mkl = cute.make_identity_tensor(cute.shape(mSFA_mkl))
        cSFB_nkl = cute.make_identity_tensor(cute.shape(mSFB_nkl))
        cC = cute.local_tile(
            cC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        # (bM, bK, loopM, loopK, loopL)
        cSFA = cute.local_tile(
            cSFA_mkl,
            cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        # (bN, bK, loopN, loopK, loopL)
        cSFB = cute.local_tile(
            cSFB_nkl,
            cute.slice_(self.cta_tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, loopM, loopK, loopL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, loopN, loopK, loopL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_N, loopM, loopN, loopL)
        tCgC = thr_mma.partition_C(gC_mnl)
        tCcC = thr_mma.partition_C(cC)

        # scale viewed as C tensor
        sSFA_view_as_C_layout = cute.make_layout(
            (
                (self.scale_granularity_m, self.scale_m_per_tile),
                self.cta_tile_shape_mnk[1],
                self.num_scale_stage,
            ),
            stride=((0, 1), 0, self.scale_m_per_tile),
        )
        sSFB_view_as_C_layout = cute.make_layout(
            (
                self.cta_tile_shape_mnk[0],
                (self.scale_granularity_n, self.scale_n_per_tile),
                self.num_scale_stage,
            ),
            stride=(0, (0, 1), self.scale_n_per_tile),
        )
        sSFA_view_as_C = cute.make_tensor(sSFA.iterator, sSFA_view_as_C_layout)
        sSFB_view_as_C = cute.make_tensor(sSFB.iterator, sSFB_view_as_C_layout)

        #
        # Partition global/shared tensor for TMA load A/B
        #
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        #
        # Partition global/shared tensor for scaleA/scaleB load
        #
        atom_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mSFA_mkl.element_type,
            num_bits_per_copy=mSFA_mkl.element_type.width,
        )
        tiled_copy_sfa = cute.make_tiled_copy_tv(
            atom_copy, cute.make_layout((32,)), cute.make_layout((1,))
        )
        tiled_copy_sfb = cute.make_tiled_copy_tv(
            atom_copy, cute.make_layout((32,)), cute.make_layout((1,))
        )
        thr_copy_sfa = tiled_copy_sfa.get_slice(lane_idx)
        thr_copy_sfb = tiled_copy_sfb.get_slice(lane_idx)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
        tAgSFA_mkl = thr_copy_sfa.partition_S(gSFA_mkl)
        tAsSFA = thr_copy_sfa.partition_D(sSFA)
        tAcSFA = thr_copy_sfa.partition_S(cSFA)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopN, loopK, loopL)
        tBgSFB_nkl = thr_copy_sfb.partition_S(gSFB_nkl)
        tBsSFB = thr_copy_sfb.partition_D(sSFB)
        tBcSFB = thr_copy_sfb.partition_S(cSFB)

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        #
        # Specialized Schedule warp
        #
        if warp_idx == self.sched_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_sched_warps)
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            # First tile
            work_tile = tile_sched.initial_work_tile_info()

            tile_info_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_tile_stage
            )

            while work_tile.is_valid_tile:
                # query next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

                # acquire tile info pipeline
                tile_info_pipeline.producer_acquire(tile_info_producer_state)

                # store the tile info
                cur_tile_coord = work_tile.tile_idx
                with cute.arch.elect_one():
                    sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[0]
                    sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[1]
                    sInfo[(2, tile_info_producer_state.index)] = cur_tile_coord[2]
                    sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                        work_tile.is_valid_tile
                    )

                # fence view async shared
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                self.sched_sync_barrier.arrive_and_wait()
                # commit tile info pipeline
                tile_info_pipeline.producer_commit(tile_info_producer_state)
                tile_info_producer_state.advance()

            tile_info_pipeline.producer_tail(tile_info_producer_state)

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_uniform_warps)
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            # First tile
            work_tile = tile_sched.initial_work_tile_info()

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            # Get tile coord from tile scheduler
            cur_tile_coord = work_tile.tile_idx
            # initialize the tile info
            tile_info[0] = cur_tile_coord[0]
            tile_info[1] = cur_tile_coord[1]
            tile_info[2] = cur_tile_coord[2]
            tile_info[3] = work_tile.is_valid_tile

            is_valid_tile = cutlass.Boolean(1)
            is_valid_tile = tile_info[3] == 1

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), loopK)
                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                # ((atom_v, rest_v), loopK)
                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

                # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    tAgA_k = tAgA_slice[(None, ab_producer_state.count)]
                    tBgB_k = tBgB_slice[(None, ab_producer_state.count)]
                    tAsA_pipe = tAsA[(None, ab_producer_state.index)]
                    tBsB_pipe = tBsB[(None, ab_producer_state.index)]

                    tma_bar = ab_pipeline.producer_get_barrier(ab_producer_state)

                    # Conditionally wait for AB buffer empty
                    ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                    # TMA load A/B
                    cute.copy(
                        tma_atom_a,
                        tAgA_k,
                        tAsA_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=b_full_mcast_mask,
                    )

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            #
            # Wait A/B buffer empty
            #
            ab_pipeline.producer_tail(ab_producer_state)

        #
        # Specialized Scale load warp
        #
        if warp_idx == self.scale_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_uniform_warps)
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            # First tile
            work_tile = tile_sched.initial_work_tile_info()

            scale_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_scale_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            # Get tile coord from tile scheduler
            cur_tile_coord = work_tile.tile_idx
            # initialize the tile info
            tile_info[0] = cur_tile_coord[0]
            tile_info[1] = cur_tile_coord[1]
            tile_info[2] = cur_tile_coord[2]
            tile_info[3] = work_tile.is_valid_tile

            is_valid_tile = cutlass.Boolean(1)
            is_valid_tile = tile_info[3] == 1

            while is_valid_tile:
                #
                # Prepare the mask for scaleA/scaleB
                #
                tApSFA = cute.make_rmem_tensor(
                    cute.make_layout(
                        cute.filter_zeros(cute.slice_(tAsSFA, (None, None, None, 0))).shape
                    ),
                    cutlass.Boolean,
                )
                tBpSFB = cute.make_rmem_tensor(
                    cute.make_layout(
                        cute.filter_zeros(cute.slice_(tBsSFB, (None, None, None, 0))).shape
                    ),
                    cutlass.Boolean,
                )

                # Peek (try_wait) SCALE buffer empty
                scale_producer_state.reset_count()
                peek_scale_empty_status = cutlass.Boolean(1)
                if scale_producer_state.count < k_tile_cnt:
                    peek_scale_empty_status = scale_pipeline.producer_try_acquire(
                        scale_producer_state
                    )

                #
                # load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    #
                    # Slice to per mma tile index
                    #
                    tAsSFA_pipe = cute.filter_zeros(
                        tAsSFA[(None, None, None, scale_producer_state.index)]
                    )
                    tBsSFB_pipe = cute.filter_zeros(
                        tBsSFB[(None, None, None, scale_producer_state.index)]
                    )
                    tAgSFA_k = cute.filter_zeros(
                        tAgSFA_mkl[
                            (
                                None,
                                None,
                                None,
                                tile_info[0],
                                scale_producer_state.count,
                                tile_info[2],
                            )
                        ]
                    )
                    tBgSFB_k = cute.filter_zeros(
                        tBgSFB_nkl[
                            (
                                None,
                                None,
                                None,
                                tile_info[1],
                                scale_producer_state.count,
                                tile_info[2],
                            )
                        ]
                    )

                    tAcSFA_compact = cute.filter_zeros(
                        cute.slice_(
                            tAcSFA,
                            (
                                None,
                                None,
                                None,
                                tile_info[0],
                                scale_producer_state.count,
                                tile_info[2],
                            ),
                        )
                    )
                    tBcSFB_compact = cute.filter_zeros(
                        cute.slice_(
                            tBcSFB,
                            (
                                None,
                                None,
                                None,
                                tile_info[1],
                                scale_producer_state.count,
                                tile_info[2],
                            ),
                        )
                    )
                    for i in cutlass.range_constexpr(cute.size(tApSFA, mode=[1])):
                        tApSFA[((0, 0), i, (0, 0))] = cute.elem_less(
                            tAcSFA_compact[(i)][0], mSFA_mkl.shape[0]
                        )
                    for i in cutlass.range_constexpr(cute.size(tBpSFB, mode=[1])):
                        tBpSFB[((0, 0), i, (0, 0))] = cute.elem_less(
                            tBcSFB_compact[(i)][0], mSFB_nkl.shape[0]
                        )

                    # Conditionally wait for Scale buffer empty
                    scale_pipeline.producer_acquire(scale_producer_state, peek_scale_empty_status)

                    # load scaleA/scaleB
                    cute.copy(tiled_copy_sfa, tAgSFA_k, tAsSFA_pipe, pred=tApSFA)
                    cute.copy(tiled_copy_sfb, tBgSFB_k, tBsSFB_pipe, pred=tBpSFB)

                    scale_pipeline.producer_commit(scale_producer_state)

                    # Peek (try_wait) Scale buffer empty
                    scale_producer_state.advance()
                    peek_scale_empty_status = cutlass.Boolean(1)
                    if scale_producer_state.count < k_tile_cnt:
                        peek_scale_empty_status = scale_pipeline.producer_try_acquire(
                            scale_producer_state
                        )

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            #
            # Wait Scale buffer empty
            #
            scale_pipeline.producer_tail(scale_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_uniform_warps)
            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            # Get tile coord from tile scheduler
            cur_tile_coord = work_tile.tile_idx
            # initialize the tile info
            tile_info[0] = cur_tile_coord[0]
            tile_info[1] = cur_tile_coord[1]
            tile_info[2] = cur_tile_coord[2]
            tile_info[3] = work_tile.is_valid_tile

            is_valid_tile = cutlass.Boolean(1)
            is_valid_tile = tile_info[3] == 1

            while is_valid_tile:
                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                # Peek (try_wait) Acc buffer empty for k_tile = 0
                acc_producer_state.reset_count()
                peek_acc_empty_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_acc_empty_status = acc_pipeline.producer_try_acquire(acc_producer_state)

                #
                # Mma mainloop with SM107 Bkeep-Breuse pattern
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Set tensor memory buffer for current tile
                    # (MMA, MMA_M, MMA_N)
                    tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                    #
                    # Wait for accumulator buffer empty
                    #
                    if is_leader_cta:
                        acc_pipeline.producer_acquire(acc_producer_state, peek_acc_empty_status)

                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)

                        # tCtAcc += tCrA * tCrB
                        tile_crd = (None, None, None, ab_consumer_state.index)

                        # Get current stage tensors (3D)
                        # tCrA has shape (MMA, MMA_M, MMA_K, STAGE) -> (MMA, MMA_M, MMA_K)
                        tCrA_stage = tCrA[tile_crd]
                        # tCrB has shape (MMA, MMA_N, MMA_K, STAGE) -> (MMA, MMA_N, MMA_K)
                        tCrB_stage = tCrB[tile_crd]

                        # Check if we should use Bkeep-Breuse pattern
                        if cutlass.const_expr(self.enable_breuse):
                            # Slice accumulator once (shared across k_phase)
                            tCtAcc_keep = tCtAcc[(None, 0, 0)]
                            tCtAcc_reuse = tCtAcc[(None, 1, 0)]

                            for k_phase in cutlass.range(self.mma_inst_tile_k, unroll_full=True):
                                # Bkeep-Breuse pattern

                                # B slice - select N=0 from (MMA, MMA_N, MMA_K) -> (MMA, MMA_K)
                                tCrB_slice = tCrB_stage[(None, 0, k_phase)]

                                # Keep operation - first A slice
                                # Select M=0 from (MMA, MMA_M, MMA_K) -> (MMA, MMA_K)
                                tCrA_keep = tCrA_stage[(None, 0, k_phase)]

                                # Blockwise consumer reads TMEM per k_tile and scales it,
                                # so we must overwrite on the first k_phase of every
                                # k_tile, not running-sum across k_tile boundaries.
                                tiled_mma_bkeep.set(
                                    tcgen05.Field.ACCUMULATE,
                                    k_phase != 0,
                                )
                                cute.gemm(
                                    tiled_mma_bkeep,
                                    tCtAcc_keep,
                                    tCrA_keep,
                                    tCrB_slice,
                                    tCtAcc_keep,
                                )

                                # Reuse operation - second A slice
                                # Select M=1 from (MMA, MMA_M, MMA_K) -> (MMA, MMA_K)
                                tCrA_reuse = tCrA_stage[(None, 1, k_phase)]

                                tiled_mma_breuse.set(
                                    tcgen05.Field.ACCUMULATE,
                                    k_phase != 0,
                                )
                                cute.gemm(
                                    tiled_mma_breuse,
                                    tCtAcc_reuse,
                                    tCrA_reuse,
                                    tCrB_slice,
                                    tCtAcc_reuse,
                                )
                        else:
                            # Regular kernel pattern (non-breuse)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA_stage,
                                tCrB_stage,
                                tCtAcc,
                            )

                        # Async arrive AB buffer empty
                        ab_pipeline.consumer_release(ab_consumer_state)

                    # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                    #
                    # Async arrive accumulator buffer full(each kblock)
                    #
                    if is_leader_cta:
                        acc_pipeline.producer_commit(acc_producer_state)

                    # Peek (try_wait) Acc buffer empty for k_tile = k_tile + 1
                    acc_producer_state.advance()
                    if acc_producer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_acc_empty_status = acc_pipeline.producer_try_acquire(
                                acc_producer_state
                            )

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Specialized acc update warps
        #
        if warp_idx <= self.acc_update_warp_id[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_acc_update_warps)
            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            # Use dynamic cosize of tCtAcc_base to place the final accumulator
            # right after the staging region, so it does not overlap when MMA_M>1.
            tCtAcc_final = cute.make_tensor(
                tCtAcc_base.iterator + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base),
                tCtAcc_base.layout,
            )

            #
            # Partition for epilogue
            #
            epi_tidx = tidx % 128
            (
                tiled_copy_t2r,
                tiled_copy_r2t,
                tTR_tAcc_base,
                tTR_rAcc,
                tTR_rAcc_final,
                tTR_sSFA,
                tTR_sSFB,
                tRT_rAcc,
                tRT_tAcc_base,
            ) = self.acc_update_tmem_copy_and_partition(
                epi_tidx,
                tCtAcc_base,
                tCtAcc_final,
                tCgC,
                sSFA_view_as_C,
                sSFB_view_as_C,
                epi_tile,
            )

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            scale_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_scale_stage
            )

            epi_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            # Get tile coord from tile scheduler
            cur_tile_coord = work_tile.tile_idx
            # initialize the tile info
            tile_info[0] = cur_tile_coord[0]
            tile_info[1] = cur_tile_coord[1]
            tile_info[2] = cur_tile_coord[2]
            tile_info[3] = work_tile.is_valid_tile

            is_valid_tile = cutlass.Boolean(1)
            is_valid_tile = tile_info[3] == 1

            while is_valid_tile:
                # initialize the final accumulator
                tTR_rAcc_final.fill(0.0)

                # Keep both EPI_M and EPI_N so per-subtile_idx scale is correct
                # when cta_tile_M > EPI_TILE_M (i.e. EPI_M > 1).
                tTR_rSFA_ = cute.make_rmem_tensor(
                    cute.slice_(tTR_sSFA, (None, None, None, None, None, 0)).shape,
                    self.acc_dtype,
                )
                tTR_rSFB_ = cute.make_rmem_tensor(
                    cute.slice_(tTR_sSFB, (None, None, None, None, None, 0)).shape,
                    self.acc_dtype,
                )
                # Group (EPI_M, EPI_N) into one mode so tTR_rSFA[.., subtile_idx]
                # matches the grouped tTR_tAcc indexing below.
                tTR_rSFA = cute.group_modes(tTR_rSFA_, 3, cute.rank(tTR_rSFA_))
                tTR_rSFB = cute.group_modes(tTR_rSFB_, 3, cute.rank(tTR_rSFB_))

                scale_consumer_state.reset_count()
                peek_scale_full_status = cutlass.Boolean(1)
                if scale_consumer_state.count < k_tile_cnt:
                    peek_scale_full_status = scale_pipeline.consumer_try_wait(scale_consumer_state)

                acc_consumer_state.reset_count()
                peek_acc_full_status = cutlass.Boolean(1)
                if acc_consumer_state.count < k_tile_cnt:
                    peek_acc_full_status = acc_pipeline.consumer_try_wait(acc_consumer_state)

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Set tensor memory buffer for current tile
                    # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                    tTR_tAcc = tTR_tAcc_base[
                        (None, None, None, None, None, acc_consumer_state.index)
                    ]

                    #
                    # Wait for scale buffer full
                    #
                    scale_pipeline.consumer_wait(scale_consumer_state, peek_scale_full_status)

                    tTR_sSFA_slice = cute.slice_(
                        tTR_sSFA,
                        (None, None, None, None, None, scale_consumer_state.index),
                    )
                    tTR_sSFB_slice = cute.slice_(
                        tTR_sSFB,
                        (None, None, None, None, None, scale_consumer_state.index),
                    )

                    scale_atom_copy = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        self.acc_dtype,
                        num_bits_per_copy=self.acc_dtype.width,
                    )

                    # Copy into the un-grouped rmem tensors (shape matches slice);
                    # tTR_rSFA / tTR_rSFB share storage via group_modes aliasing.
                    cute.copy(scale_atom_copy, tTR_sSFA_slice, tTR_rSFA_)
                    cute.copy(scale_atom_copy, tTR_sSFB_slice, tTR_rSFB_)

                    #
                    # Wait for accumulator buffer full
                    #

                    acc_pipeline.consumer_wait(acc_consumer_state, peek_acc_full_status)

                    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                    #
                    # Update accumulator by scale factor in subtiles
                    #
                    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                    for subtile_idx in cutlass.range(subtile_cnt):
                        #
                        # Load accumulator from tensor memory buffer to register
                        #
                        tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                        #
                        # Update accumulator by scale factor
                        #
                        tTR_rAcc_subtile = tTR_rAcc_final[(None, None, None, subtile_idx)]
                        tTR_rSFA_subtile = tTR_rSFA[(None, None, None, subtile_idx)]
                        tTR_rSFB_subtile = tTR_rSFB[(None, None, None, subtile_idx)]

                        acc_vec = tTR_rAcc.load()
                        final_vec = tTR_rAcc_subtile.load()
                        scale_a = tTR_rSFA_subtile.load()
                        scale_b = tTR_rSFB_subtile.load()
                        scale = scale_a * scale_b
                        final_vec = acc_vec * scale + final_vec
                        tTR_rAcc_subtile.store(final_vec.to(self.acc_dtype))

                    #
                    # Async arrive scale buffer empty
                    #
                    scale_pipeline.consumer_release(scale_consumer_state)
                    scale_consumer_state.advance()

                    peek_scale_full_status = cutlass.Boolean(1)
                    if scale_consumer_state.count < k_tile_cnt:
                        peek_scale_full_status = scale_pipeline.consumer_try_wait(
                            scale_consumer_state
                        )
                    #
                    # Async arrive accumulator buffer empty
                    #
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                    peek_acc_full_status = cutlass.Boolean(1)
                    if acc_consumer_state.count < k_tile_cnt:
                        peek_acc_full_status = acc_pipeline.consumer_try_wait(acc_consumer_state)

                tRT_tAcc = tRT_tAcc_base[(None, None, None, None, None, 0)]
                tRT_tAcc = cute.group_modes(tRT_tAcc, 3, cute.rank(tRT_tAcc))

                #
                # Wait for epilogue buffer empty
                #
                epi_pipeline.producer_acquire(epi_producer_state)

                # copy the accumulator to tensor memory buffer
                cute.copy(tiled_copy_r2t, tTR_rAcc_final, tRT_tAcc)
                cute.arch.fence_view_async_tmem_store()

                #
                # Async arrive epilogue buffer full
                #
                epi_pipeline.producer_commit(epi_producer_state)
                epi_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

        #
        # Specialized epilogue warps
        #
        if warp_idx <= self.epilog_warp_id[-1] and warp_idx >= self.epilog_warp_id[0]:
            cute.arch.setmaxregister_increase(self.num_regs_epilogue_warps)
            #
            # Alloc tensor memory buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base_ = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tCtAcc_final = cute.make_tensor(
                tCtAcc_base_.iterator + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base_),
                tCtAcc_base_.layout,
            )

            #
            # Partition for epilogue
            #
            epi_tidx = tidx % 128
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_final, tCgC, epi_tile, use_2cta_instrs
            )
            tTR_cC = None
            if cutlass.const_expr(not self.fp8_quantize_1x128):
                cC_t = transform_partitioned_tensor_layout(tCcC)
                cC_epi = cute.flat_divide(cC_t, epi_tile)
                tTR_cC = tiled_copy_t2r.get_slice(epi_tidx).partition_D(cC_epi)

            tTR_rC = None
            tiled_copy_r2s = None
            tRS_rC = None
            tRS_sC = None
            bSG_sC = None
            bSG_gC_partitioned = None
            if cutlass.const_expr(not self.fp8_quantize_1x128):
                tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
                tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                    tiled_copy_t2r, tTR_rC, epi_tidx, sC
                )
                (
                    tma_atom_c,
                    bSG_sC,
                    bSG_gC_partitioned,
                ) = self.epilog_gmem_copy_and_partition(epi_tidx, tma_atom_c, tCgC, epi_tile, sC)

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            epi_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)

            c_pipeline = None
            if cutlass.const_expr(not self.fp8_quantize_1x128):
                # Threads/warps participating in tma store pipeline
                c_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    32 * len(self.epilog_warp_id),
                )
                c_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.num_c_stage,
                    producer_group=c_producer_group,
                )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            # Get tile coord from tile scheduler
            cur_tile_coord = work_tile.tile_idx
            # initialize the tile info
            tile_info[0] = cur_tile_coord[0]
            tile_info[1] = cur_tile_coord[1]
            tile_info[2] = cur_tile_coord[2]
            tile_info[3] = work_tile.is_valid_tile

            is_valid_tile = cutlass.Boolean(1)
            is_valid_tile = tile_info[3] == 1

            num_prev_subtiles = cutlass.Int32(0)

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                #
                # Slice to per mma tile index
                #
                bSG_gC = None
                if cutlass.const_expr(not self.fp8_quantize_1x128):
                    # ((ATOM_V, REST_V), EPI_M, EPI_N)
                    bSG_gC = bSG_gC_partitioned[
                        (
                            None,
                            None,
                            None,
                            mma_tile_coord_mnl[0],
                            mma_tile_coord_mnl[1],
                            mma_tile_coord_mnl[2],
                        )
                    ]

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, epi_consumer_state.index)]

                #
                # Wait for accumulator buffer full
                #
                epi_pipeline.consumer_wait(epi_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                tTR_cC_tile = None
                if cutlass.const_expr(not self.fp8_quantize_1x128):
                    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
                    tTR_cC_tile = tTR_cC[(None, None, None, None, None, *mma_tile_coord_mnl)]
                    tTR_cC_tile = cute.group_modes(tTR_cC_tile, 3, cute.rank(tTR_cC_tile))

                #
                # Store accumulator to global memory in subtiles
                #
                if cutlass.const_expr(self.fp8_quantize_1x128):
                    assert fp8_scale is not None
                    self._epilog_fp8_register_scratch(
                        tiled_copy_t2r,
                        tTR_tAcc,
                        tTR_rAcc,
                        sC,
                        epi_consumer_state,
                        epi_pipeline,
                        epi_tidx // 32,
                        lane_idx,
                        mma_tile_coord_mnl,
                        mma_tile_coord_v,
                        epilogue_op,
                        mC_mnl,
                        fp8_scale,
                    )
                else:
                    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                    for subtile_idx in cutlass.range(subtile_cnt):
                        # Load accumulator from tensor memory buffer to register.
                        tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                        # Convert to C type.
                        acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                        acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                        tRS_rC.store(acc_vec)

                        # Store C to shared memory.
                        num_prev_subtiles = num_prev_subtiles + 1
                        c_buffer = num_prev_subtiles % self.num_c_stage
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rC,
                            tRS_sC[(None, None, None, c_buffer)],
                        )
                        # Fence and barrier to make the shared-memory store
                        # visible to the TMA store.
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )
                        self.epilog_sync_barrier.arrive_and_wait()

                        # TMA store C to global memory.
                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(
                                tma_atom_c,
                                bSG_sC[(None, c_buffer)],
                                bSG_gC[(None, subtile_idx)],
                            )
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        self.epilog_sync_barrier.arrive_and_wait()

                #
                # Async arrive accumulator buffer empty
                #
                if cutlass.const_expr(not self.fp8_quantize_1x128):
                    epi_pipeline.consumer_release(epi_consumer_state)
                    epi_consumer_state.advance()

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
            #
            # Wait for C store complete
            #
            if cutlass.const_expr(not self.fp8_quantize_1x128):
                c_pipeline.producer_tail()

    @cute.jit
    def _epilog_fp8_register_scratch(
        self,
        tiled_copy_t2r,
        tTR_tAcc,
        tTR_rAcc,
        sC,
        epi_consumer_state,
        epi_pipeline,
        epi_warp_idx,
        lane_idx,
        mma_tile_coord_mnl,
        mma_tile_coord_v,
        epilogue_op,
        mC_mnl,
        fp8_scale,
    ):
        """Release TMEM before quantization and coalesce FP8 output stores."""
        tile_m_base = (
            mma_tile_coord_mnl[0] * self.mma_tiler[0]
            + mma_tile_coord_v * self.cta_tile_shape_mnk[0]
        )
        # The scheduler never emits a tile wholly beyond M. Keep the remaining
        # row count as a DSL integer because Python min/max cannot consume its
        # dynamic Boolean comparisons during lowering.
        valid_rows = mC_mnl.shape[0] - tile_m_base
        global_n_base = mma_tile_coord_mnl[1] * self.mma_tiler[1]
        num_sf_cols = (mC_mnl.shape[1] * mC_mnl.shape[2]) // 32
        global_sf_col = (mma_tile_coord_mnl[2] * mC_mnl.shape[1] + global_n_base) // 32
        num_sf_col_tiles = (num_sf_cols + 3) // 4
        output_sf_tile_base = (tile_m_base // 128) * num_sf_col_tiles * 512 + (
            global_sf_col // 4
        ) * 512

        # The 32x32b tcgen05.ld shape gives each epilogue thread one complete row
        # fragment.
        # Retain all four N32 fragments in registers so each thread computes
        # one complete 1x128 scale without a cross-lane reduction.
        row_values = cute.make_rmem_tensor(
            cute.make_layout((16, 4), stride=(1, 16)),
            cutlass.Int32,
        )
        flat_tTR_rAcc = cute.make_tensor(tTR_rAcc.iterator, cute.make_layout((32,)))
        local_absmax = cutlass.Float32(0.0)
        for segment_idx in range(4):
            cute.copy(
                tiled_copy_t2r,
                tTR_tAcc[(None, None, None, segment_idx)],
                tTR_rAcc,
            )
            rounded_values = cute.make_rmem_tensor((32,), cutlass.BFloat16)
            rounded_values.store(epilogue_op(flat_tTR_rAcc.load().to(cutlass.BFloat16)))
            packed_values = cute.recast_tensor(rounded_values, cutlass.Int32)
            for packed_idx in range(16):
                row_values[(packed_idx, segment_idx)] = packed_values[packed_idx]
            for elem_idx in range(32):
                value_f32 = cutlass.Float32(rounded_values[elem_idx])
                local_absmax = cute.arch.fmax(
                    local_absmax,
                    cute.arch.fmax(value_f32, -value_f32),
                )

        # The complete row is now independent of TMEM. Release it before the
        # quantization and global stores so accumulator-update warps can begin
        # publishing the next persistent tile.
        cute.arch.fence_view_async_tmem_load()
        epi_pipeline.consumer_release(epi_consumer_state)
        epi_consumer_state.advance()

        block_absmax = cute.arch.fmax(local_absmax, cutlass.Float32(1e-10))
        output_sf = (block_absmax * cutlass.Float32(1.0 / 448.0)).to(cutlass.Float8E8M0FNU)
        output_sf_vec = cute.make_rmem_tensor((4,), cutlass.Float8E8M0FNU)
        for sf_copy_idx in range(4):
            output_sf_vec[sf_copy_idx] = output_sf
        output_sf_packed = cutlass.Uint32(cute.recast_tensor(output_sf_vec, cutlass.Uint32)[0])
        output_sf_code = output_sf_packed & cutlass.Uint32(0xFF)
        output_scale_bf16_bits = (cutlass.Uint32(254) - output_sf_code) << cutlass.Uint32(7)
        output_scale_bf16x2 = cutlass.Int32(
            output_scale_bf16_bits | (output_scale_bf16_bits << cutlass.Uint32(16))
        )

        epi_row = epi_warp_idx * 32 + lane_idx
        if epi_row < valid_rows:
            output_sf_offset = output_sf_tile_base + (epi_row % 32) * 16 + (epi_row // 32) * 4
            _store_global_b32(
                fp8_scale.iterator + output_sf_offset,
                cutlass.Int32(output_sf_packed),
            )

        # Quantize into the row-major 128x128 FP8 scratch tile. The sC
        # allocation is 1024-byte aligned and has exactly 16 KiB in this path.
        scratch_i32_ptr = cute.recast_ptr(sC.iterator, dtype=cutlass.Int32)
        for segment_idx in range(4):
            packed_output = cute.make_rmem_tensor((8,), cutlass.Int32)
            for output_idx in range(8):
                packed_idx = output_idx * 2
                packed_output[output_idx] = _scale_bf16x4_to_e4m3x4(
                    row_values[(packed_idx, segment_idx)],
                    row_values[(packed_idx + 1, segment_idx)],
                    output_scale_bf16x2,
                )
            scratch_word_offset = epi_row * 32 + segment_idx * 8
            _store_shared_b128(
                scratch_i32_ptr + scratch_word_offset,
                packed_output[0],
                packed_output[1],
                packed_output[2],
                packed_output[3],
            )
            _store_shared_b128(
                scratch_i32_ptr + scratch_word_offset + 4,
                packed_output[4],
                packed_output[5],
                packed_output[6],
                packed_output[7],
            )

        # Four adjacent lanes cooperatively store one contiguous output row;
        # each lane transfers eight bytes, producing coalesced 32-byte sectors.
        # Epilogue warps own disjoint 32-row scratch slices, so warp-local
        # synchronization is sufficient.
        cute.arch.sync_warp()
        row_group_idx = lane_idx // 4
        row_lane_idx = lane_idx % 4
        for row_batch_idx in range(4):
            output_row = epi_warp_idx * 32 + row_batch_idx * 8 + row_group_idx
            if output_row < valid_rows:
                for segment_idx in range(4):
                    scratch_word_offset = output_row * 32 + segment_idx * 8 + row_lane_idx * 2
                    value_lo = _load_shared_b32(scratch_i32_ptr + scratch_word_offset)
                    value_hi = _load_shared_b32(scratch_i32_ptr + scratch_word_offset + 1)
                    _store_global_b64(
                        cute.domain_offset(
                            (
                                tile_m_base + output_row,
                                global_n_base + segment_idx * 32 + row_lane_idx * 8,
                                mma_tile_coord_mnl[2],
                            ),
                            mC_mnl,
                        ).iterator,
                        value_lo,
                        value_hi,
                    )

        # Prevent this warp from overwriting its scratch slice on the next
        # persistent tile until all lanes have drained the current one.
        cute.arch.sync_warp()

    def acc_update_tmem_copy_and_partition(
        self,
        tidx,
        tAcc,
        tAcc_final,
        gC_mnl,
        sSFA,
        sSFB,
        epi_tile,
    ):
        """SM107 override: fold MMA_M/MMA_N into leading M/N dims via
        transform_partitioned_tensor_layout, so cta_tile_M=256 (MMA_M=2)
        is handled correctly. Semantically equivalent to the parent for
        MMA_M=MMA_N=1.
        """
        if cutlass.const_expr(self.mma_tiler[0] == 64):
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)),
                self.acc_dtype,
            )
            tmem_store_atom = cute.make_copy_atom(
                tcgen05.copy.St16x256bOp(tcgen05.copy.Repetition(8)),
                self.acc_dtype,
            )
        else:
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
                self.acc_dtype,
            )
            tmem_store_atom = cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)),
                self.acc_dtype,
            )

        tAcc_t = transform_partitioned_tensor_layout(tAcc)
        tAcc_final_t = transform_partitioned_tensor_layout(tAcc_final)
        gC_mnl_t = transform_partitioned_tensor_layout(gC_mnl)

        tAcc_epi = cute.flat_divide(tAcc_t, epi_tile)
        tAcc_final_epi = cute.flat_divide(tAcc_final_t, epi_tile)

        tiled_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tAcc_epi[(None, None, 0, 0, 0)])
        tiled_copy_r2t = tcgen05.make_tmem_copy(
            tmem_store_atom, tAcc_final_epi[(None, None, 0, 0, 0)]
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        thr_copy_r2t = tiled_copy_r2t.get_slice(tidx)

        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        gC_mnl_epi = cute.flat_divide(gC_mnl_t, epi_tile)
        sSFA_epi = cute.flat_divide(sSFA, epi_tile)
        sSFB_epi = cute.flat_divide(sSFB, epi_tile)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_sSFA = thr_copy_t2r.partition_D(sSFA_epi)
        tTR_sSFB = thr_copy_t2r.partition_D(sSFB_epi)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        tTR_rAcc_final_ = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, None, None, 0, 0, 0)].shape, self.acc_dtype
        )
        tTR_rAcc_final = cute.group_modes(tTR_rAcc_final_, 3, cute.rank(tTR_rAcc_final_))

        tRT_gC = thr_copy_r2t.partition_S(gC_mnl_epi)
        tRT_tAcc_final = thr_copy_r2t.partition_D(tAcc_final_epi)
        tRT_rAcc_final_ = cute.make_rmem_tensor(
            tRT_gC[(None, None, None, None, None, 0, 0, 0)].shape, self.acc_dtype
        )
        tRT_rAcc_final = cute.group_modes(tRT_rAcc_final_, 3, cute.rank(tRT_rAcc_final_))

        return (
            tiled_copy_t2r,
            tiled_copy_r2t,
            tTR_tAcc,
            tTR_rAcc,
            tTR_rAcc_final,
            tTR_sSFA,
            tTR_sSFB,
            tRT_rAcc_final,
            tRT_tAcc_final,
        )

    def epilog_tmem_copy_and_partition(
        self,
        tidx,
        tAcc,
        gC_mnl,
        epi_tile,
        use_2cta_instrs,
    ):
        """SM107 override: same as parent but applies
        transform_partitioned_tensor_layout so MMA_M>1 is handled.
        """
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        tAcc_t = transform_partitioned_tensor_layout(tAcc)
        gC_mnl_t = transform_partitioned_tensor_layout(gC_mnl)

        tAcc_epi = cute.flat_divide(tAcc_t, epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        gC_mnl_epi = cute.flat_divide(gC_mnl_t, epi_tile)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_gmem_copy_and_partition(
        self,
        tidx,
        atom,
        gC_mnl,
        epi_tile,
        sC,
    ):
        """SM107 override: apply transform_partitioned_tensor_layout to
        gC before flat_divide so MMA_M>1 is handled.
        """
        gC_mnl_t = transform_partitioned_tensor_layout(gC_mnl)
        gC_epi = cute.flat_divide(gC_mnl_t, epi_tile)
        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, loopM, loopN, loopL)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    @cute.jit
    def wrapper(
        self,
        m: cutlass.Int32,
        n: cutlass.Int32,
        k: cutlass.Int32,
        sf_m: cutlass.Int32,
        sf_n: cutlass.Int32,
        sf_k: cutlass.Int32,
        batch_size: cutlass.Int32,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        c_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Run the kernel with dynamic TRT-LLM tensor shapes and pointers."""

        a_tensor = cute.make_tensor(
            a_ptr,
            layout=cute.make_ordered_layout((m, k, batch_size), order=(1, 0, 2)),
        )
        b_tensor = cute.make_tensor(
            b_ptr,
            layout=cute.make_ordered_layout((n, k, batch_size), order=(1, 0, 2)),
        )
        sfa_tensor = cute.make_tensor(
            a_sf_ptr,
            layout=cute.make_ordered_layout(
                (sf_m, sf_k, batch_size),
                order=(0, 1, 2),
            ),
        )
        sfb_tensor = cute.make_tensor(
            b_sf_ptr,
            layout=cute.make_ordered_layout(
                (sf_n, sf_k, batch_size),
                order=(1, 0, 2),
            ),
        )

        self(
            a_tensor,
            b_tensor,
            c_tensor,
            sfa_tensor,
            sfb_tensor,
            max_active_clusters,
            stream,
        )

    @cute.jit
    def wrapper_fp8_quantized(
        self,
        m: cutlass.Int32,
        n: cutlass.Int32,
        k: cutlass.Int32,
        sf_m: cutlass.Int32,
        sf_n: cutlass.Int32,
        sf_k: cutlass.Int32,
        batch_size: cutlass.Int32,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        c_tensor: cute.Tensor,
        scale_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Run blockwise GEMM with fused packed FP8 1x128 output."""
        a_tensor = cute.make_tensor(
            a_ptr,
            layout=cute.make_ordered_layout((m, k, batch_size), order=(1, 0, 2)),
        )
        b_tensor = cute.make_tensor(
            b_ptr,
            layout=cute.make_ordered_layout((n, k, batch_size), order=(1, 0, 2)),
        )
        sfa_tensor = cute.make_tensor(
            a_sf_ptr,
            layout=cute.make_ordered_layout((sf_m, sf_k, batch_size), order=(0, 1, 2)),
        )
        sfb_tensor = cute.make_tensor(
            b_sf_ptr,
            layout=cute.make_ordered_layout((sf_n, sf_k, batch_size), order=(1, 0, 2)),
        )
        self(
            a_tensor,
            b_tensor,
            c_tensor,
            sfa_tensor,
            sfb_tensor,
            max_active_clusters,
            stream,
            fp8_scale=scale_tensor,
        )

    @staticmethod
    def can_implement(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler: Tuple[int, int, int],
        mma_inst_shape: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        batch_size: int,
        a_major: str,
        b_major: str,
        c_major: str,
        fuse_fp8_quantize_1x128: bool = False,
    ) -> bool:
        """
        Check if the gemm can be implemented on SM107.
        """
        can_impl = True
        # The fused packed-FP8 epilogue writes full 128-column tiles and only
        # predicates rows.
        if fuse_fp8_quantize_1x128 and (n % 128 != 0 or c_major != "n"):
            can_impl = False

        # Check data types
        if a_dtype not in {cutlass.Float8E4M3FN, cutlass.Float8E5M2}:
            can_impl = False
        if b_dtype not in {cutlass.Float8E4M3FN, cutlass.Float8E5M2}:
            can_impl = False
        if acc_dtype not in {cutlass.Float32}:
            can_impl = False
        if c_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
        }:
            can_impl = False

        # Check MMA instruction shape for K=64 constraint
        if mma_inst_shape[2] == 64:
            if not use_2cta_instrs and mma_inst_shape[0] != 128:
                can_impl = False
            elif use_2cta_instrs and mma_inst_shape[0] != 256:
                can_impl = False

        # Check MMA tiler and instruction shape relationship
        if (
            mma_tiler[0] // mma_inst_shape[0] != 2 and mma_tiler[0] // mma_inst_shape[0] != 1
        ) or mma_tiler[1] != mma_inst_shape[1]:
            can_impl = False

        # MMA tiler N must be 128
        if mma_tiler[1] not in (128,):
            can_impl = False

        # Cluster shape validation
        def is_power_of_2(x):
            return x > 0 and (x & (x - 1)) == 0

        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            or not is_power_of_2(cluster_shape_mn[0])
            or not is_power_of_2(cluster_shape_mn[1])
        ):
            can_impl = False

        # Cluster shape M must be multiple of 2 if use_2cta_instrs
        if use_2cta_instrs and cluster_shape_mn[0] % 2 != 0:
            can_impl = False

        # Skip unsupported A/B layout
        if not (a_major == "k" and b_major == "k"):
            can_impl = False

        # Check tensor alignment
        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contigous_16B_alignment(a_dtype, a_major == "m", (m, k, batch_size))
            or not check_contigous_16B_alignment(b_dtype, b_major == "n", (n, k, batch_size))
            or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n, batch_size))
        ):
            can_impl = False

        return can_impl
