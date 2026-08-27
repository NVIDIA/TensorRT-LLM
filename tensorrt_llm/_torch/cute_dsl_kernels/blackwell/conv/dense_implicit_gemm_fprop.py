# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Adapted from dlarch-fastkernels/dynamic-kernel-generator!20906 for TensorRT-LLM.
# ruff: noqa: E501

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

import argparse
from functools import lru_cache
from typing import Literal, Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
import torch.nn.functional as F
from cutlass import testing
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.torch import dtype as torch_dtype
from cutlass.utils import is_fp8_dtype

from tensorrt_llm.logger import logger

from .dense_gemm_persistent_dynamic import _compute_stages
from .dense_gemm_persistent_dynamic_preferred_cluster import (
    PersistentDenseGemmKernelDynamicPreferredCluster,
)

"""
A high-performance 3D implicit-GEMM based fprop convolution example for the NVIDIA Blackwell SM100 architecture using CUTE DSL.
- Input tensor A is NxDxHxWxC, must be C major.
- Filter tensor B is KxTxRxSxC, must be C major.
- Output tensor C is NxZxPxQxK, must be K major.

This kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations and for the im2col transformation of the input tensor A.
    - Utilizes Blackwell's tcgen05 MMA for matrix multiply-accumulate (MMA) operations (including 2cta mma instructions)
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Utilizes a CLC dynamic persistent dense GEMM kernel for the implicit GEMM.

This implicit-GEMM based convolution works by converting the convolution into a GEMM problem with the following mapping:
- GEMM M dimension maps to NxZxPxQ
- GEMM N dimension maps to K
- GEMM K dimension maps to TxRxSxC
During the load of input tensor to SMEM, the TMA operation performs the im2col transformation on the input tensor A.
This transforms the A matrix into the required shape for the GEMM operation (NxZxPxQ by TxRxSxC), and may involve replication of the input elements.
Filter tensor can be loaded to SMEM without any transformation.
The output tensor C is then stored to GMEM via the im2col TMA store (no transformation necessary).

To run this example:

.. code-block:: bash

    python -m tensorrt_llm._torch.cute_dsl_kernels.blackwell.conv.dense_implicit_gemm_fprop \
      --ncdhw 1,128,32,32,32 --ktrs 256,3,3,3                         \
      --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32        \
      --use_2cta_instrs --mma_tiler_mn 256,128                        \
      --preferred_cluster_shape_mn 2,1 --fallback_cluster_shape_mn 1,1 \
      --upper_pad_dhw 1,1,1 --lower_pad_dhw 1,1,1                     \
      --stride_dhw 1,1,1 --dil_dhw 1,1,1

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python -m tensorrt_llm._torch.cute_dsl_kernels.blackwell.conv.dense_implicit_gemm_fprop \
      --ncdhw 1,128,32,32,32 --ktrs 256,3,3,3                         \
      --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32        \
      --use_2cta_instrs --mma_tiler_mn 256,128                        \
      --preferred_cluster_shape_mn 2,1 --fallback_cluster_shape_mn 1,1 \
      --upper_pad_dhw 1,1,1 --lower_pad_dhw 1,1,1                     \
      --stride_dhw 1,1,1 --dil_dhw 1,1,1                                \
      --warmup_iterations 1 --iterations 10 --skip_ref_check

Constraints:
* Supported input data types: fp16, bf16, tf32, int8, uint8, fp8 (e4m3fn, e5m2),
  see detailed valid dtype combinations in below PersistentConvKernel class documentation
* A/B tensor must have the same data type
* Mma tiler M must be 64/128 (use_2cta_instrs=False) or 128/256 (use_2cta_instrs=True)
* Mma tiler N must be 32-256, step 32
* Cluster shape M/N must be positive and power of 2, total cluster size <= 16
* Cluster shape M must be multiple of 2 if use_2cta_instrs=True
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 4, 8, and 16 for TFloat32,
  Float16/BFloat16, and Int8/Uint8/Float8, respectively.
"""


def _check_tensor_alignment(
    c: int,
    k: int,
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
):
    """Check if the tensor alignment is valid for convolution.

    :param c: The number of input channels
    :type c: int
    :param k: The number of output channels
    :type k: int
    :param ab_dtype: The data type of the A and B operands
    :type ab_dtype: Type[cutlass.Numeric]
    :param c_dtype: The data type of the output tensor
    :type c_dtype: Type[cutlass.Numeric]
    """

    def check_contiguous_16B_alignment(dtype, num_major_elements):
        num_contiguous_elements = 16 * 8 // dtype.width
        return num_major_elements % num_contiguous_elements == 0

    if not check_contiguous_16B_alignment(c_dtype, k) or not check_contiguous_16B_alignment(
        ab_dtype, c
    ):
        raise testing.CantImplementError(
            f"Invalid tensor alignment: C = {c}, K = {k}, ab_dtype = {ab_dtype}, c_dtype = {c_dtype}"
        )


def _check_swizzle_size(
    m: int,
    n: int,
    mma_tiler_mn: Tuple[int, int],
    use_2cta_instrs: bool,
    preferred_cluster_shape_mn: Tuple[int, int],
    fallback_cluster_shape_mn: Tuple[int, int],
    swizzle_size: int,
    raster_along: str,
):
    """Check that swizzle_size does not exceed the cluster count in the swizzled dimension.

    :param m: GEMM M dimension (N*Z*P*Q for convolution)
    :param n: GEMM N dimension (K for convolution)
    :param mma_tiler_mn: MMA tiler shape (M, N)
    :param use_2cta_instrs: Whether 2-CTA instructions are used
    :param preferred_cluster_shape_mn: Preferred cluster shape (M, N)
    :param fallback_cluster_shape_mn: Fallback cluster shape (M, N)
    :param swizzle_size: Swizzle size to validate
    :param raster_along: Rasterization order ("m" or "n")
    """
    if swizzle_size <= 1:
        return

    cta_v = 2 if use_2cta_instrs else 1
    cta_tile_m = mma_tiler_mn[0] // cta_v
    cta_tile_n = mma_tiler_mn[1]
    m_tiles = -(-m // cta_tile_m)
    n_tiles = -(-n // cta_tile_n)

    for cs in [preferred_cluster_shape_mn, fallback_cluster_shape_mn]:
        if raster_along == "m":
            nclusters = -(-n_tiles // cs[1])
        else:
            nclusters = -(-m_tiles // cs[0])
        if nclusters < swizzle_size:
            dim_name = "N" if raster_along == "m" else "M"
            raise testing.CantImplementError(
                f"swizzle_size ({swizzle_size}) exceeds the number of "
                f"{dim_name} clusters ({nclusters}) for cluster shape "
                f"{cs}. Use a smaller swizzle_size or increase the "
                f"{dim_name} dimension."
            )


class PersistentConvKernel(PersistentDenseGemmKernelDynamicPreferredCluster):
    """
    Persistent 3D convolution kernel.
    The input (A) is expected to be in 5D tensor (NDHWC) format and is loaded via the im2col TMA load atom.
    The filter (B) is expected to be in 5D tensor (KTRSC) format and is loaded via the TMA load atom.
    The output (C) is expected to be in 5D tensor (NZPQK) format and is stored via the im2col TMA store.
    This class reuses the kernel from the PersistentDenseGemmKernelDynamicPreferredCluster class for the implicit GEMM.

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param use_2cta_instrs: Whether to use CTA group 2 for advanced thread cooperation
    :type use_2cta_instrs: bool
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tiler (M,N)
    :type mma_tiler_mn: Tuple[int, int]
    :param preferred_cluster_shape_mn: Preferred cluster dimensions (M,N) for optimal performance
    :type preferred_cluster_shape_mn: Tuple[int, int]
    :param fallback_cluster_shape_mn: Fallback cluster dimensions (M,N) for parallel processing
    :type fallback_cluster_shape_mn: Tuple[int, int]
    :param filter_trs: Filter dimensions (T, R, S)
    :type filter_trs: Tuple[int, int, int]
    :param upper_padding_dhw: Upper padding (D, H, W)
    :type upper_padding_dhw: Tuple[int, int, int]
    :param lower_padding_dhw: Lower padding (D, H, W)
    :type lower_padding_dhw: Tuple[int, int, int]
    :param stride_dhw: Stride (Sd, Sh, Sw)
    :type stride_dhw: Tuple[int, int, int]
    :param dilation_dhw: Dilation (DilD, DilH, DilW)
    :type dilation_dhw: Tuple[int, int, int]
    :param swizzle_size: Swizzling size in the unit of cluster. 1 means no swizzle
    :type swizzle_size: int
    :param raster_along: Rasterization order of clusters. Only used when swizzle_size > 1.
    :type raster_along: Literal["m", "n"]

    :note: In current version, A and B tensor must be C major. C tensor must be K major.

    :note: In current version, A and B tensor must have the same data type
        - i.e., Float8E4M3FN for A and Float8E5M2 for B is not supported

    :note: Supported A/B data types:
        - TFloat32
        - Float16/BFloat16
        - Int8/Uint8
        - Float8E4M3FN/Float8E5M2

    :note: Supported accumulator data types:
        - Float32 (for all floating point A/B data types)
        - Float16 (only for fp16 and fp8 A/B data types)
        - Int32 (only for uint8/int8 A/B data types)

    :note: Supported C data types:
        - Float32 (for float32 and int32 accumulator data types)
        - Int32 (for float32 and int32 accumulator data types)
        - Float16/BFloat16 (for fp16 and fp8 accumulator data types)
        - Int8/Uint8 (for uint8/int8 accumulator data types)
        - Float8E4M3FN/Float8E5M2 (for float32 accumulator data types)

    :note: Constraints:
        - MMA tiler M must be 64/128 (use_2cta_instrs=False) or 128/256 (use_2cta_instrs=True)
        - MMA tiler N must be 32-256, step 32
        - Cluster shape M must be multiple of 2 if use_2cta_instrs=True
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16

    **Example:**

    .. code-block:: python
        conv = PersistentConvKernel(
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=True,
            mma_tiler_mn=(128, 128),
            preferred_cluster_shape_mn=(4, 2),
            fallback_cluster_shape_mn=(2, 1),
            filter_trs=(3, 3, 3),
            upper_padding_dhw=(1, 1, 1),
            lower_padding_dhw=(1, 1, 1),
            stride_dhw=(1, 1, 1),
            dilation_dhw=(1, 1, 1),
        )
        conv(
            a,
            b,
            c,
            upper_pad_d,
            upper_pad_h,
            upper_pad_w,
            lower_pad_d,
            lower_pad_h,
            lower_pad_w,
            stride_d,
            stride_h,
            stride_w,
            dilation_d,
            dilation_h,
            dilation_w,
            stream,
            epilogue_op,
        )
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        preferred_cluster_shape_mn: Tuple[int, int],
        fallback_cluster_shape_mn: Tuple[int, int],
        filter_trs: Tuple[int, int, int],
        upper_padding_dhw: Tuple[int, int, int],
        lower_padding_dhw: Tuple[int, int, int],
        stride_dhw: Tuple[int, int, int],
        dilation_dhw: Tuple[int, int, int],
        swizzle_size: int = 1,
        raster_along: Literal["m", "n"] = "m",
    ):
        super().__init__(
            acc_dtype=acc_dtype,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=mma_tiler_mn,
            preferred_cluster_shape_mn=preferred_cluster_shape_mn,
            fallback_cluster_shape_mn=fallback_cluster_shape_mn,
            use_tma_store=True,  # Conv always uses im2col TMA store
            swizzle_size=swizzle_size,
            raster_along=raster_along,
        )

        # Store convolution parameters for im2col TMA
        self.filter_trs = filter_trs
        self.upper_padding_dhw = upper_padding_dhw
        self.lower_padding_dhw = lower_padding_dhw
        self.stride_dhw = stride_dhw
        self.dilation_dhw = dilation_dhw

    def _setup_conv_input_attrs(self, a, b, c):
        """Validate and set input-dependent attributes.

        Sets a_dtype, b_dtype, c_dtype, a_major_mode, b_major_mode, c_layout.

        :param a: Input tensor A - (N, D, H, W, C) layout
        :param b: Filter tensor B - (K, T, R, S, C) layout
        :param c: Output tensor C - (N, Z, P, Q, K) layout
        """
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        # Only C major accepted
        if cutlass.const_expr(a.leading_dim != 4):
            raise RuntimeError("The layout of a is not supported")
        if cutlass.const_expr(b.leading_dim != 4):
            raise RuntimeError("The layout of b is not supported")
        if cutlass.const_expr(c.leading_dim != 4):
            raise RuntimeError("The layout of c is not supported")
        self.a_major_mode = cute.nvgpu.OperandMajorMode.K
        self.b_major_mode = cute.nvgpu.OperandMajorMode.K
        self.c_layout = utils.LayoutEnum.ROW_MAJOR  # K dimension contiguous

        # Check if input data types are compatible with MMA instruction
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

    def _setup_conv_tma(self, a, b, c, tiled_mma, upper_pad_op, lower_pad_op, stride_op, dil_op):
        """Set up TMA atoms and tensors for im2col convolution with dual cluster shapes.

        Creates preferred and fallback TMA atoms for A and B tensors, and a single
        TMA atom for C (im2col store is cluster-independent).

        The pad/stride/dilation operands feed the im2col A descriptor corners.
        Threading them as runtime cutlass.Int32 lets one compiled cubin serve
        any pad/stride/dilation config.

        :param a: Input tensor A - (N, D, H, W, C) layout
        :param b: Filter tensor B - (K, T, R, S, C) layout
        :param c: Output tensor C - (N, Z, P, Q, K) layout
        :param tiled_mma: Tiled MMA configuration
        :param upper_pad_op: Upper padding (D, H, W) as runtime cutlass.Int32 tuple
        :param lower_pad_op: Lower padding (D, H, W) as runtime cutlass.Int32 tuple
        :param stride_op: Convolution stride (D, H, W) as runtime cutlass.Int32 tuple
        :param dil_op: Dilation (D, H, W) as runtime cutlass.Int32 tuple
        :returns: (tma_atom_a_preferred, tma_tensor_a_preferred,
                tma_atom_a_fallback, tma_tensor_a_fallback,
                tma_atom_b_preferred, tma_tensor_b_preferred,
                tma_atom_b_fallback, tma_tensor_b_fallback,
                tma_atom_c, tma_tensor_c)
        """
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Create 2-mode hierarchical tensor layout: (N, D, H, W, C) -> ((W, H, D, N), C)
        mA = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[3, 2, 1, 0, 4]))
        mA = cute.group_modes(mA, begin=0, end=4)

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        a_internal_type = cutlass.TFloat32 if mA.element_type is cutlass.Float32 else None

        # Filter T/R/S sourced from the filter tensor b (K, T, R, S, C) rather than
        # the host-int self.filter_trs: under the dynamic tensor layout these
        # extents are runtime Int32, so one compiled cubin serves any T/R/S.
        rt_filter_trs = (b.shape[1], b.shape[2], b.shape[3])

        # --- A preferred: im2col TMA load ---
        a_copy_atom_preferred = (
            cpasync.CopyBulkTensorIm2ColG2SMulticastOp(cta_group=self.cta_group)
            if self.is_preferred_a_mcast
            else cpasync.CopyBulkTensorIm2ColG2SOp(cta_group=self.cta_group)
        )
        tma_atom_a_preferred, tma_tensor_a_preferred = cute.nvgpu.make_im2col_tma_atom_A(
            a_copy_atom_preferred,
            mA,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            rt_filter_trs,
            upper_pad_op,
            lower_pad_op,
            stride_op,
            dil_op,
            self.preferred_cluster_layout_vmnk.shape,
            internal_type=a_internal_type,
        )

        # --- A fallback: im2col TMA load ---
        a_copy_atom_fallback = (
            cpasync.CopyBulkTensorIm2ColG2SMulticastOp(cta_group=self.cta_group)
            if self.is_fallback_a_mcast
            else cpasync.CopyBulkTensorIm2ColG2SOp(cta_group=self.cta_group)
        )
        tma_atom_a_fallback, tma_tensor_a_fallback = cute.nvgpu.make_im2col_tma_atom_A(
            a_copy_atom_fallback,
            mA,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            rt_filter_trs,
            upper_pad_op,
            lower_pad_op,
            stride_op,
            dil_op,
            self.fallback_cluster_layout_vmnk.shape,
            internal_type=a_internal_type,
        )

        # --- B: tiled TMA load ---
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        # Change view of filter tensor from (K, T, R, S, C) to (K, (C, S, R, T))
        mB = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[0, 4, 3, 2, 1]))
        mB = cute.group_modes(mB, begin=1, end=5)
        b_internal_type = cutlass.TFloat32 if mB.element_type is cutlass.Float32 else None

        # --- B preferred ---
        b_op_preferred = sm100_utils.cluster_shape_to_tma_atom_B(
            self.preferred_cluster_shape_mn, tiled_mma.thr_id
        )
        tma_atom_b_preferred, tma_tensor_b_preferred = cute.nvgpu.make_tiled_tma_atom_B(
            b_op_preferred,
            mB,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.preferred_cluster_layout_vmnk.shape,
            internal_type=b_internal_type,
        )

        # --- B fallback ---
        b_op_fallback = sm100_utils.cluster_shape_to_tma_atom_B(
            self.fallback_cluster_shape_mn, tiled_mma.thr_id
        )
        tma_atom_b_fallback, tma_tensor_b_fallback = cute.nvgpu.make_tiled_tma_atom_B(
            b_op_fallback,
            mB,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.fallback_cluster_layout_vmnk.shape,
            internal_type=b_internal_type,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size
        # Response size is 4B * 4 elements
        self.num_clc_response_bytes = 16

        # --- C: im2col TMA store (cluster-independent) ---
        # Change view of output tensor from (N, Z, P, Q, K) to ((Q, P, Z, N), K)
        mC = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[3, 2, 1, 0, 4]))
        mC = cute.group_modes(mC, begin=0, end=4)

        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, (None, 0)))

        tma_atom_c, tma_tensor_c = cpasync.make_im2col_tma_atom(
            cpasync.CopyBulkTensorIm2ColS2GOp(),
            mC,
            epi_smem_layout,
            self.epi_tile,
        )
        tma_tensor_c = cute.coalesce(tma_tensor_c, target_profile=(1, 1))

        # Add dummy batch dimension to all tensors (GEMM expects batch dimension)
        def add_dummy_batch_dimension(tensor):
            new_layout = cute.append(tensor.layout, cute.make_layout(1))
            tensor = cute.make_tensor(tensor.iterator, new_layout)
            return tensor

        tma_tensor_a_preferred = add_dummy_batch_dimension(tma_tensor_a_preferred)
        tma_tensor_a_fallback = add_dummy_batch_dimension(tma_tensor_a_fallback)
        tma_tensor_b_preferred = add_dummy_batch_dimension(tma_tensor_b_preferred)
        tma_tensor_b_fallback = add_dummy_batch_dimension(tma_tensor_b_fallback)
        tma_tensor_c = add_dummy_batch_dimension(tma_tensor_c)

        return (
            tma_atom_a_preferred,
            tma_tensor_a_preferred,
            tma_atom_a_fallback,
            tma_tensor_a_fallback,
            tma_atom_b_preferred,
            tma_tensor_b_preferred,
            tma_atom_b_fallback,
            tma_tensor_b_fallback,
            tma_atom_c,
            tma_tensor_c,
        )

    def _setup_attributes(self):
        """Set up configurations that are dependent on convolution inputs."""
        # Configure tiled mma
        tiled_mma = self._create_tiled_mma()

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            (mma_inst_shape_k * mma_inst_tile_k,),
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Compute epilogue subtile
        self.epi_tile = utils.sm100.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        c_smem_layout = utils.sm100.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, 1
        )

        self.smem_capacity = utils.get_smem_capacity_in_bytes()

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = _compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
            self.use_tma_store,
            c_smem_layout,
        )

        # Setup clc stage by default
        self.num_clc_stage = 1
        assert self.num_clc_stage == 1, "Only single-stage CLC pipeline is supported"

        # Compute A/B/C shared memory layout
        self.a_smem_layout_staged = utils.sm100.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = utils.sm100.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )

        self.c_smem_layout_staged = utils.sm100.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage
        )

        # Compute the number of tensor memory allocation columns
        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
            tiled_mma, self.mma_tiler, self.num_acc_stage, self.arch
        )

        # Compute preferred cluster layout (in addition to fallback already set by parent)
        tiled_mma = self._create_tiled_mma()
        self.preferred_cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.preferred_cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        # Calculate multicast CTA counts for preferred cluster
        self.num_preferred_mcast_ctas_a = cute.size(self.preferred_cluster_layout_vmnk.shape[2])
        self.num_preferred_mcast_ctas_b = cute.size(self.preferred_cluster_layout_vmnk.shape[1])
        self.is_preferred_a_mcast = self.num_preferred_mcast_ctas_a > 1
        self.is_preferred_b_mcast = self.num_preferred_mcast_ctas_b > 1

        self.fallback_cluster_layout_vmnk = self.cluster_layout_vmnk
        self.num_fallback_mcast_ctas_a = self.num_mcast_ctas_a
        self.num_fallback_mcast_ctas_b = self.num_mcast_ctas_b
        self.is_fallback_a_mcast = self.is_a_mcast
        self.is_fallback_b_mcast = self.is_b_mcast

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        rt_upper_pad_d: cutlass.Int32,
        rt_upper_pad_h: cutlass.Int32,
        rt_upper_pad_w: cutlass.Int32,
        rt_lower_pad_d: cutlass.Int32,
        rt_lower_pad_h: cutlass.Int32,
        rt_lower_pad_w: cutlass.Int32,
        rt_stride_d: cutlass.Int32,
        rt_stride_h: cutlass.Int32,
        rt_stride_w: cutlass.Int32,
        rt_dil_d: cutlass.Int32,
        rt_dil_h: cutlass.Int32,
        rt_dil_w: cutlass.Int32,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the persistent convolution operation with dynamic preferred cluster scheduling.

        :param a: Input tensor A - (N, D, H, W, C) layout
        :param b: Filter tensor B - (K, T, R, S, C) layout
        :param c: Output tensor C - (N, Z, P, Q, K) layout
        :param rt_upper_pad_d/h/w: Runtime upper padding (D, H, W) as Int32, so one
            compiled cubin serves any padding config without recompilation
        :param rt_lower_pad_d/h/w: Runtime lower padding (D, H, W) as Int32
        :param rt_stride_d/h/w: Runtime convolution stride (D, H, W) as Int32
        :param rt_dil_d/h/w: Runtime dilation (D, H, W) as Int32
        :param stream: CUDA stream for asynchronous execution
        :param epilogue_op: Optional elementwise lambda function to apply to the output tensor
        """
        self._setup_conv_input_attrs(a, b, c)
        self._setup_attributes()

        # Pack the runtime pad/stride/dilation scalars into (D, H, W) tuples that
        # feed the im2col A descriptor corners. Keeping them as runtime Int32 lets
        # a single compiled cubin run any pad/stride/dilation configuration.
        upper_pad_op = (rt_upper_pad_d, rt_upper_pad_h, rt_upper_pad_w)
        lower_pad_op = (rt_lower_pad_d, rt_lower_pad_h, rt_lower_pad_w)
        stride_op = (rt_stride_d, rt_stride_h, rt_stride_w)
        dil_op = (rt_dil_d, rt_dil_h, rt_dil_w)

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

        (
            tma_atom_a_preferred,
            tma_tensor_a_preferred,
            tma_atom_a_fallback,
            tma_tensor_a_fallback,
            tma_atom_b_preferred,
            tma_tensor_b_preferred,
            tma_atom_b_fallback,
            tma_tensor_b_fallback,
            tma_atom_c,
            tma_tensor_c,
        ) = self._setup_conv_tma(a, b, c, tiled_mma, upper_pad_op, lower_pad_op, stride_op, dil_op)

        # Compute grid size and scheduler params for both cluster shapes
        self.fallback_tile_sched_params, _ = self._compute_grid(
            tma_tensor_c,
            self.cta_tile_shape_mnk,
            self.fallback_cluster_shape_mn,
            self.swizzle_size,
            self.raster_along,
        )
        self.preferred_tile_sched_params, preferred_grid = self._compute_grid(
            tma_tensor_c,
            self.cta_tile_shape_mnk,
            self.preferred_cluster_shape_mn,
            self.swizzle_size,
            self.raster_along,
        )

        # Launch the megakernel synchronously
        self.kernel(
            tiled_mma,
            tma_atom_a_preferred,
            tma_tensor_a_preferred,
            tma_atom_a_fallback,
            tma_tensor_a_fallback,
            tma_atom_b_preferred,
            tma_tensor_b_preferred,
            tma_atom_b_fallback,
            tma_tensor_b_fallback,
            tma_atom_c,
            tma_tensor_c,
            self.preferred_cluster_layout_vmnk,
            self.fallback_cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.preferred_tile_sched_params,
            self.fallback_tile_sched_params,
            epilogue_op,
        ).launch(
            grid=preferred_grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.preferred_cluster_shape_mn, 1),
            fallback_cluster=(*self.fallback_cluster_shape_mn, 1),
            stream=stream,
            smem_merge_branch_allocs=True,
        )
        return

    def can_implement(
        self,
        ncdhw: Tuple[int, int, int, int, int],
        k: int,
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """Determine if the given tensor configuration can be implemented by this kernel."""
        try:
            self.check_supported_dtypes(ab_dtype, ab_dtype, c_dtype)
            # Validate fallback cluster shape (base class stores it as cluster_shape_mn)
            self.check_mma_tiler_and_cluster_shape()
            _check_tensor_alignment(ncdhw[1], k, ab_dtype, c_dtype)
            # Compute implicit GEMM M
            z, p, q = compute_zpq(
                ncdhw[2:],
                self.filter_trs,
                self.stride_dhw,
                self.upper_padding_dhw,
                self.lower_padding_dhw,
                self.dilation_dhw,
            )
            self.check_epilog_store_option(ncdhw[0] * z * p * q, k)
            _check_swizzle_size(
                ncdhw[0] * z * p * q,
                k,
                self.mma_tiler_mn,
                self.use_2cta_instrs,
                self.preferred_cluster_shape_mn,
                self.fallback_cluster_shape_mn,
                self.swizzle_size,
                self.raster_along,
            )
        except testing.CantImplementError as error:
            logger.debug(f"PersistentConvKernel cannot implement this config: {error}")
            return False
        return True

    # GPU device kernel - megakernel dispatcher for preferred/fallback cluster
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a_preferred: cute.CopyAtom,
        mA_mkl_preferred: cute.Tensor,
        tma_atom_a_fallback: cute.CopyAtom,
        mA_mkl_fallback: cute.Tensor,
        tma_atom_b_preferred: cute.CopyAtom,
        mB_nkl_preferred: cute.Tensor,
        tma_atom_b_fallback: cute.CopyAtom,
        mB_nkl_fallback: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: cute.Tensor,
        preferred_cluster_layout_vmnk: cute.Layout,
        fallback_cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        preferred_tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
        fallback_tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
    ):
        """
        GPU device kernel entry point for kernel with preferred cluster shape.
        """
        # Get cluster coordinates to determine if this is a preferred cluster
        cbdim_x, cbdim_y, cbdim_z = cute.arch.block_in_cluster_dim()
        is_preferred_cluster = (
            cbdim_x == self.preferred_cluster_shape_mn[0]
            and cbdim_y == self.preferred_cluster_shape_mn[1]
            and cbdim_z == 1
        )

        # Megakernel approach: two mutually exclusive code branches, only one path runs per launch.
        # smem_merge_branch_allocs=True at launch enables shared memory reuse between two paths.
        if is_preferred_cluster:
            self.cluster_specific_kernel(
                tiled_mma,
                tma_atom_a_preferred,
                mA_mkl_preferred,
                tma_atom_b_preferred,
                mB_nkl_preferred,
                tma_atom_c,
                mC_mnl,
                preferred_cluster_layout_vmnk,
                a_smem_layout_staged,
                b_smem_layout_staged,
                c_smem_layout_staged,
                epi_tile,
                preferred_tile_sched_params,
                epilogue_op,
                self.is_preferred_a_mcast,
                self.is_preferred_b_mcast,
                self.preferred_cluster_shape_mn,
            )
        else:
            self.cluster_specific_kernel(
                tiled_mma,
                tma_atom_a_fallback,
                mA_mkl_fallback,
                tma_atom_b_fallback,
                mB_nkl_fallback,
                tma_atom_c,
                mC_mnl,
                fallback_cluster_layout_vmnk,
                a_smem_layout_staged,
                b_smem_layout_staged,
                c_smem_layout_staged,
                epi_tile,
                fallback_tile_sched_params,
                epilogue_op,
                self.is_fallback_a_mcast,
                self.is_fallback_b_mcast,
                self.fallback_cluster_shape_mn,
            )

    @cute.jit()
    def cluster_specific_kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        effective_is_a_mcast: bool,
        effective_is_b_mcast: bool,
        cluster_shape: Tuple[int, int],
    ):
        """
        GPU device kernel performing the CLC dynamic persistent convolution computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, _bidy, _bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        is_first_cta_in_cluster = cta_rank_in_cluster == 0
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, CLC, tensor memory dealloc barrier
        #
        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            clc_response: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 4],
                16,  # Align bytes
            ]

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Warp)
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            enable_multicast_signaling=True,
            defer_sync=True,
        ).make_participants()

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilogue_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Initialize clc_pipeline (barrier) and states
        clc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cluster_size = cute.size(cluster_shape)
        num_clc_consumer_threads = 32 * (1 + cluster_size * (1 + len(self.epilogue_warp_id) + 1))
        clc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_clc_consumer_threads
        )
        clc_pipeline = pipeline.PipelineClcFetchAsync.create(
            barrier_storage=storage.clc_mbar_ptr.data_ptr(),
            num_stages=self.num_clc_stage,
            producer_group=clc_pipeline_producer_group,
            consumer_group=clc_pipeline_consumer_group,
            tx_count=self.num_clc_response_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        _tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_dealloc_sync_bar_id,
            num_threads=32 * len(self.epilogue_warp_id),
        )
        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=cluster_shape, is_relaxed=True)

        # Initial clc response pointer
        clc_response_ptr = storage.clc_response.data_ptr()

        clc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_clc_stage
        )

        #
        # Setup smem tensor A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )

        #
        # Compute multicast mask for A/B buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(effective_is_a_mcast or effective_is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)

        #
        # Partition global/shared tensor for TMA load A/B
        #
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
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
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

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
        pipeline_init_wait(cluster_shape_mn=cluster_shape)

        #
        # Construct the scheduler
        #
        tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            clc_response_ptr,
        )
        work_tile = tile_sched.initial_work_tile_info()

        #
        # Specialized TMA load warp
        #

        if warp_idx == self.tma_warp_id:
            #
            # Persistent tile scheduling loop
            #
            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), RestK)
                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                # ((atom_v, rest_v), RestK)
                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

                # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                ab_producer.reset()
                peek_ab_empty_status = ab_producer.try_acquire()

                #
                # Tma load loop
                #

                # Set up coord iterator to avoid incurring idx2crd at runtime
                # Permuted iteration order: K-tuple layout is (C,S,R,T) but we
                # iterate in (S,R,T,C) colex order so S advances innermost.
                # Tensor/TMA layouts are unchanged; we un-permute the coord at
                # slice time so A and B see the original (C,S,R,T) coord.
                k_shape_orig = cute.shape(tAgA_slice, mode=1)
                k_shape_perm = (
                    k_shape_orig[1],
                    k_shape_orig[2],
                    k_shape_orig[3],
                    k_shape_orig[0],
                )
                coord_perm = cute.repeat_like(0, k_shape_perm)
                for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Conditionally wait for AB buffer empty
                    handle = ab_producer.acquire_and_advance(peek_ab_empty_status)

                    # Un-permute (s,r,t,c) -> (c,s,r,t) for slice
                    coord_iter = (
                        coord_perm[3],
                        coord_perm[0],
                        coord_perm[1],
                        coord_perm[2],
                    )

                    # TMA load A/B
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, coord_iter)],
                        tAsA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, coord_iter)],
                        tBsB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=b_full_mcast_mask,
                    )

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_ab_empty_status = ab_producer.try_acquire()

                    coord_perm = cute.increment_coord(coord_perm, k_shape_perm)

                #
                # Advance to next tile
                #
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            #
            # Wait A/B buffer empty
            #
            ab_producer.tail()

        #
        # Specialized scheduler warp
        #
        if warp_idx == self.sched_warp_id and is_first_cta_in_cluster:
            #
            # Persistent tile scheduling loop
            #
            clc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.ProducerConsumer, self.num_clc_stage
            )

            while work_tile.is_valid_tile:
                #
                # Advance to next tile
                #
                clc_pipeline.producer_acquire(clc_producer_state)
                mbarrier_addr = clc_pipeline.producer_get_barrier(clc_producer_state)
                tile_sched.advance_to_next_work(mbarrier_addr)
                clc_producer_state.advance()

                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            clc_pipeline.producer_tail(clc_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # Persistent tile scheduling loop
            #
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Set tensor memory buffer for current tile
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer.reset()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_ab_full_status = ab_consumer.try_wait()

                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                #
                # Mma mainloop
                #
                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)

                        # tCtAcc += tCrA * tCrB
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                        tile_crd = (None, None, None, handle.index)
                        cute.gemm(tiled_mma, tCtAcc, tCrA[tile_crd], tCrB[tile_crd], tCtAcc)

                        # Async arrive AB buffer empty
                        handle.release()

                        # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                        peek_ab_full_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()

                #
                # Async arrive accumulator buffer full
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                #
                # Advance to next tile
                #
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)

        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = smem.allocate_tensor(
            element_type=self.c_dtype,
            layout=c_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=c_smem_layout_staged.inner,
        )

        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            #
            # Alloc tensor memory buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # Persistent tile scheduling loop for epilogue
            #
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            assert tma_atom_c is not None and sC is not None
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilogue_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage, producer_group=c_producer_group
            )
            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                num_tiles_executed = tile_sched.num_tiles_executed
                acc_consumer_state = utils.gemm.sm100.epilogue_tma_store(
                    self,
                    tidx,
                    warp_idx,
                    tma_atom_c,
                    tCtAcc_base,
                    sC,
                    tCgC,
                    epi_tile,
                    num_tiles_executed,
                    epilogue_op,
                    mma_tile_coord_mnl,
                    acc_consumer_state,
                    acc_pipeline,
                    c_pipeline,
                )
                #
                # Advance to next tile
                #
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            # Wait for C store complete
            c_pipeline.producer_tail()

            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)


@lru_cache(maxsize=1)
def compile_conv(
    ncdhw: Tuple[int, int, int, int, int],
    ktrs: Tuple[int, int, int, int],
    input: cute.Tensor,
    filter: cute.Tensor,
    output: cute.Tensor,
    acc_dtype: Type[cutlass.Numeric],
    mma_tiler: Tuple[int, int] = (256, 256),
    preferred_cluster_shape_mn: Tuple[int, int] = (2, 1),
    fallback_cluster_shape_mn: Tuple[int, int] = (1, 1),
    swizzle_size: int = 1,
    raster_along: Literal["m", "n"] = "m",
    use_2cta_instrs: bool = True,
    upper_padding_dhw: Tuple[int, int, int] = (0, 0, 0),
    lower_padding_dhw: Tuple[int, int, int] = (0, 0, 0),
    stride_dhw: Tuple[int, int, int] = (1, 1, 1),
    dilation_dhw: Tuple[int, int, int] = (1, 1, 1),
    epilogue_op: cutlass.Constexpr = lambda x: x,
):
    """
    Compile a 3D convolution kernel with CLC dynamic preferred cluster scheduling.

    :param ncdhw: Problem shape (N, C, D, H, W)
    :param ktrs: Problem shape (K, T, R, S)
    :param input: Input tensor (N, D, H, W, C) with C contiguous
    :param filter: Filter tensor in KTRSC format (K, T, R, S, C) with C contiguous
    :param output: Output tensor (N, Z, P, Q, K) with K contiguous
    :param acc_dtype: Accumulator data type
    :param mma_tiler: MMA tile shape (M, N)
    :param preferred_cluster_shape_mn: Preferred cluster shape (M, N)
    :param fallback_cluster_shape_mn: Fallback cluster shape (M, N)
    :param swizzle_size: Swizzling size in the unit of cluster. 1 means no swizzle
    :param raster_along: Rasterization order of clusters. Only used when swizzle_size > 1
    :param use_2cta_instrs: Whether to use 2CTA instructions
    :param upper_padding_dhw: Upper padding (D, H, W)
    :param lower_padding_dhw: Lower padding (D, H, W)
    :param stride_dhw: Stride (Sd, Sh, Sw)
    :param dilation_dhw: Dilation (DilD, DilH, DilW)
    :param epilogue_op: Epilogue operation
    :return: Compiled kernel function
    """
    from cutlass.cute.runtime import make_fake_stream

    # Create convolution kernel object
    conv_op = PersistentConvKernel(
        acc_dtype,
        use_2cta_instrs,
        mma_tiler,
        preferred_cluster_shape_mn,
        fallback_cluster_shape_mn,
        ktrs[1:],
        upper_padding_dhw,
        lower_padding_dhw,
        stride_dhw,
        dilation_dhw,
        swizzle_size,
        raster_along,
    )

    # Check if configuration can be implemented
    can_implement = conv_op.can_implement(
        ncdhw,
        ktrs[0],
        input.element_type,
        output.element_type,
    )
    if not can_implement:
        raise testing.CantImplementError("The current config is invalid/unsupported.")

    stream = make_fake_stream()
    # Box pad/stride/dilation as runtime Int32 so cute.compile lowers them to
    # SSA values and keeps them out of the mangled kernel name; one cubin then
    # serves any pad/stride/dilation without recompilation.
    return cute.compile(
        conv_op,
        input,
        filter,
        output,
        cutlass.Int32(upper_padding_dhw[0]),
        cutlass.Int32(upper_padding_dhw[1]),
        cutlass.Int32(upper_padding_dhw[2]),
        cutlass.Int32(lower_padding_dhw[0]),
        cutlass.Int32(lower_padding_dhw[1]),
        cutlass.Int32(lower_padding_dhw[2]),
        cutlass.Int32(stride_dhw[0]),
        cutlass.Int32(stride_dhw[1]),
        cutlass.Int32(stride_dhw[2]),
        cutlass.Int32(dilation_dhw[0]),
        cutlass.Int32(dilation_dhw[1]),
        cutlass.Int32(dilation_dhw[2]),
        stream,
        epilogue_op,
    )


def compute_zpq(
    dhw: Tuple[int, int, int],
    trs: Tuple[int, int, int],
    stride_dhw: Tuple[int, int, int],
    upper_padding_dhw: Tuple[int, int, int],
    lower_padding_dhw: Tuple[int, int, int],
    dilation_dhw: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """Compute output spatial dimensions Z, P, and Q with asymmetric padding."""
    D, H, W = dhw
    T, R, S = trs
    Sd, Sh, Sw = stride_dhw
    UpperPadD, UpperPadH, UpperPadW = upper_padding_dhw
    LowerPadD, LowerPadH, LowerPadW = lower_padding_dhw
    DilD, DilH, DilW = dilation_dhw
    Z = ((D + UpperPadD + LowerPadD - DilD * (T - 1) - 1) // Sd) + 1
    P = ((H + UpperPadH + LowerPadH - DilH * (R - 1) - 1) // Sh) + 1
    Q = ((W + UpperPadW + LowerPadW - DilW * (S - 1) - 1) // Sw) + 1
    return Z, P, Q


def create_cute_tensor(
    source_f32_tensor: torch.Tensor,
    dtype: Type[cutlass.Numeric],
    leading_dim: int | None = None,
) -> Tuple[cute.Tensor, torch.Tensor]:
    """Create a cute tensor with dynamic layout from a source f32 tensor.

    :param source_f32_tensor: Source f32 tensor
    :type source_f32_tensor: torch.Tensor
    :param dtype: Data type
    :type dtype: Type[cutlass.Numeric]
    :param leading_dim: Leading dimension for dynamic layout
    :type leading_dim: int
    :return: Tuple of cute tensor and storage tensor
    :rtype: Tuple[cute.Tensor, torch.Tensor]
    """
    import cutlass.torch as cutlass_torch

    storage_type = torch.int8 if is_fp8_dtype(dtype) else torch_dtype(dtype)
    storage_tensor = source_f32_tensor.to(dtype=storage_type)

    cute_tensor = from_dlpack(
        storage_tensor, assumed_align=16, force_tf32=dtype == cutlass.TFloat32
    )
    cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
    if is_fp8_dtype(dtype):
        cute_tensor.element_type = dtype
        cute_tensor = cutlass_torch.convert_cute_tensor(
            source_f32_tensor, cute_tensor, dtype, is_dynamic_layout=True
        )
        # Cast the underlying storage tensor to the correct dtype
        storage_tensor = storage_tensor.view(dtype=torch_dtype(dtype))
    return cute_tensor, storage_tensor


def prepare_tensors(
    ncdhw: Tuple[int, int, int, int, int],
    ktrs: Tuple[int, int, int, int],
    zpq: Tuple[int, int, int],
    ab_dtype: Type[cutlass.Numeric],
):
    """Prepare f32 tensors for 3D convolution.

    :param ncdhw: Input tensor shape (N, C, D, H, W)
    :type ncdhw: Tuple[int, int, int, int, int]
    :param ktrs: Filter tensor shape components (K, T, R, S)
    :type ktrs: Tuple[int, int, int, int]
    :param zpq: Output spatial dimensions (Z, P, Q)
    :type zpq: Tuple[int, int, int]
    :param ab_dtype: Data type for A/B input tensors
    :type ab_dtype: Type[cutlass.Numeric]
    :return: Tuple of input, filter, and output tensors
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    N, C, D, H, W = ncdhw
    K, T, R, S = ktrs
    Z, P, Q = zpq

    # Initialize with small random values for numerical stability
    if ab_dtype == cutlass.Uint8:
        input_range = (0, 2)
    else:
        input_range = (-1, 2)
    input_tensor = torch.randint(
        input_range[0],
        input_range[1],
        (N, D, H, W, C),
        dtype=torch.float32,
        device="cuda",
    )
    filter_tensor = torch.randint(
        input_range[0],
        input_range[1],
        (K, T, R, S, C),
        dtype=torch.float32,
        device="cuda",
    )
    output_tensor = torch.empty((N, Z, P, Q, K), dtype=torch.float32, device="cuda")

    return input_tensor, filter_tensor, output_tensor


def run(
    ncdhw: Tuple[int, int, int, int, int],
    ktrs: Tuple[int, int, int, int],
    stride_dhw: Tuple[int, int, int] = (1, 1, 1),
    upper_pad_dhw: Tuple[int, int, int] = (0, 0, 0),
    lower_pad_dhw: Tuple[int, int, int] = (0, 0, 0),
    dil_dhw: Tuple[int, int, int] = (1, 1, 1),
    ab_dtype: Type[cutlass.Numeric] = cutlass.Float16,
    c_dtype: Type[cutlass.Numeric] = cutlass.Float16,
    acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    preferred_cluster_shape_mn: Tuple[int, int] = (1, 1),
    fallback_cluster_shape_mn: Tuple[int, int] = (1, 1),
    swizzle_size: int = 1,
    raster_along: Literal["m", "n"] = "m",
    use_2cta_instrs: bool = False,
    tolerance: float = 1e-02,
    warmup_iterations: int = 0,
    iterations: int = 1,
    use_cold_l2: bool = False,
    skip_ref_check: bool = False,
    **kwargs,
):
    """Run 3D convolution and compare against PyTorch reference.

    The filter tensor uses native KTRSC layout (K, T, R, S, C) with C contiguous.

    :param ncdhw: Input tensor shape (N, C, D, H, W)
    :param ktrs: Filter tensor shape components (K, T, R, S)
    :param stride_dhw: Stride (Sd, Sh, Sw)
    :param upper_pad_dhw: Upper padding (D, H, W)
    :param lower_pad_dhw: Lower padding (D, H, W)
    :param dil_dhw: Dilation (DilD, DilH, DilW)
    :param ab_dtype: Data type for A/B input tensors
    :param c_dtype: Data type for output tensor C
    :param acc_dtype: Accumulator data type
    :param mma_tiler_mn: MMA tiler shape
    :param preferred_cluster_shape_mn: Preferred cluster shape (M, N)
    :param fallback_cluster_shape_mn: Fallback cluster shape (M, N)
    :param swizzle_size: Swizzling size in the unit of cluster. 1 means no swizzle
    :param raster_along: Rasterization order of clusters. Only used when swizzle_size > 1
    :param use_2cta_instrs: Whether to use 2CTA instructions
    :param tolerance: Tolerance for result comparison
    :param warmup_iterations: Number of warmup iterations
    :param iterations: Number of benchmark iterations
    :param use_cold_l2: Whether to flush L2 cache between iterations
    :param skip_ref_check: Whether to skip reference checking
    """
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    N, C, D, H, W = ncdhw
    K, T, R, S = ktrs

    Z, P, Q = compute_zpq(
        (D, H, W),
        (T, R, S),
        stride_dhw,
        upper_pad_dhw,
        lower_pad_dhw,
        dil_dhw,
    )

    print("Running Blackwell 3D Convolution test with:")
    print(f"  Input shape (N, C, D, H, W): {ncdhw}")
    print(f"  Filter shape (K, C, T, R, S): ({K}, {C}, {T}, {R}, {S})")
    print(f"  Output shape (N, K, Z, P, Q): ({N}, {K}, {Z}, {P}, {Q})")
    print(f"  Stride (Sd, Sh, Sw): {stride_dhw}")
    print(f"  Upper padding (D, H, W): {upper_pad_dhw}")
    print(f"  Lower padding (D, H, W): {lower_pad_dhw}")
    print(f"  Dilation (DilD, DilH, DilW): {dil_dhw}")
    print(f"  A/B data type: {ab_dtype}")
    print(f"  C data type: {c_dtype}")
    print(f"  Accumulator type: {acc_dtype}")
    print(f"  MMA tiler (M, N): {mma_tiler_mn}")
    print(f"  Preferred cluster shape (M, N): {preferred_cluster_shape_mn}")
    print(f"  Fallback cluster shape (M, N): {fallback_cluster_shape_mn}")
    print(f"  Swizzle size: {swizzle_size}")
    print(f"  Raster along: {raster_along}")
    print(f"  Use 2CTA instructions: {use_2cta_instrs}\n")

    # Create input and filter tensors
    input_tensor, filter_tensor, output_tensor = prepare_tensors(ncdhw, ktrs, (Z, P, Q), ab_dtype)

    # Prepare cute tensors
    input_, input_storage = create_cute_tensor(input_tensor, ab_dtype, leading_dim=4)
    filter_, filter_storage = create_cute_tensor(filter_tensor, ab_dtype, leading_dim=4)
    output_, output_storage = create_cute_tensor(output_tensor, c_dtype, leading_dim=4)

    # Compile convolution kernel
    print("Compiling kernel with cute.compile ...")
    compiled_fn = compile_conv(
        ncdhw,
        ktrs,
        input_,
        filter_,
        output_,
        acc_dtype,
        mma_tiler=mma_tiler_mn,
        preferred_cluster_shape_mn=preferred_cluster_shape_mn,
        fallback_cluster_shape_mn=fallback_cluster_shape_mn,
        swizzle_size=swizzle_size,
        raster_along=raster_along,
        use_2cta_instrs=use_2cta_instrs,
        upper_padding_dhw=upper_pad_dhw,
        lower_padding_dhw=lower_pad_dhw,
        stride_dhw=stride_dhw,
        dilation_dhw=dil_dhw,
    )

    # Get current CUDA stream
    torch_stream = torch.cuda.Stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    # Run convolution. Pad/stride/dilation are passed as runtime Int32 so the
    # cubin cached by compile_conv runs any pad/stride/dilation config.
    print("Running Blackwell 3D convolution...")
    compiled_fn(
        input_,
        filter_,
        output_,
        cutlass.Int32(upper_pad_dhw[0]),
        cutlass.Int32(upper_pad_dhw[1]),
        cutlass.Int32(upper_pad_dhw[2]),
        cutlass.Int32(lower_pad_dhw[0]),
        cutlass.Int32(lower_pad_dhw[1]),
        cutlass.Int32(lower_pad_dhw[2]),
        cutlass.Int32(stride_dhw[0]),
        cutlass.Int32(stride_dhw[1]),
        cutlass.Int32(stride_dhw[2]),
        cutlass.Int32(dil_dhw[0]),
        cutlass.Int32(dil_dhw[1]),
        cutlass.Int32(dil_dhw[2]),
        current_stream,
    )

    with torch.backends.cudnn.flags(enabled=False):
        if not skip_ref_check:
            # Run PyTorch reference convolution
            print("Running PyTorch reference 3D convolution...")
            # Pytorch expects tensors to be in NCDHW format
            input_ncdhw = input_tensor.permute(0, 4, 1, 2, 3)
            if upper_pad_dhw != lower_pad_dhw:
                # F.conv3d only supports symmetric padding, so manually pad the input
                # F.pad takes padding in reverse dimension order: (W_before, W_after, H_before, H_after, D_before, D_after)
                pad_arg = (
                    lower_pad_dhw[2],
                    upper_pad_dhw[2],
                    lower_pad_dhw[1],
                    upper_pad_dhw[1],
                    lower_pad_dhw[0],
                    upper_pad_dhw[0],
                )
                input_ncdhw = F.pad(input_ncdhw, pad_arg)
                conv_padding = (0, 0, 0)
            else:
                conv_padding = upper_pad_dhw
            output_ref = (
                F.conv3d(
                    input_ncdhw,
                    filter_tensor.permute(0, 4, 1, 2, 3),
                    stride=stride_dhw,
                    padding=conv_padding,
                    dilation=dil_dhw,
                )
                .to(dtype=torch_dtype(c_dtype))
                .to(dtype=torch.float32)
            )
            # Compare results
            print("Comparing results...")

            # Convert to float32 for comparison
            # Transform output from (N, Z, P, Q, K) -> (N, K, Z, P, Q)
            output_f32 = output_storage.permute(0, 4, 1, 2, 3).to(torch.float32)
            output_ref_f32 = output_ref.to(torch.float32)

            # Verify results
            torch.testing.assert_close(
                output_f32,
                output_ref_f32,
                atol=tolerance,
                rtol=1e-03,
            )
            print("✓ Results match within tolerance!")

    # Benchmark if requested
    if iterations > 0:
        print(f"\nBenchmarking with {warmup_iterations} warmup and {iterations} iterations...")

        def generate_tensors():
            input_tensor, filter_tensor, output_tensor = prepare_tensors(
                ncdhw, ktrs, (Z, P, Q), ab_dtype
            )
            input_, input_storage = create_cute_tensor(
                input_tensor,
                ab_dtype,
                leading_dim=4,
            )
            filter_, filter_storage = create_cute_tensor(
                filter_tensor,
                ab_dtype,
                leading_dim=4,
            )
            output_, output_storage = create_cute_tensor(
                output_tensor,
                c_dtype,
                leading_dim=4,
            )
            return testing.JitArguments(
                input_,
                filter_,
                output_,
                cutlass.Int32(upper_pad_dhw[0]),
                cutlass.Int32(upper_pad_dhw[1]),
                cutlass.Int32(upper_pad_dhw[2]),
                cutlass.Int32(lower_pad_dhw[0]),
                cutlass.Int32(lower_pad_dhw[1]),
                cutlass.Int32(lower_pad_dhw[2]),
                cutlass.Int32(stride_dhw[0]),
                cutlass.Int32(stride_dhw[1]),
                cutlass.Int32(stride_dhw[2]),
                cutlass.Int32(dil_dhw[0]),
                cutlass.Int32(dil_dhw[1]),
                cutlass.Int32(dil_dhw[2]),
                current_stream,
            )

        workspace_count = 1
        if use_cold_l2:
            one_workspace_bytes = (
                input_storage.numel() * input_storage.element_size()
                + filter_storage.numel() * filter_storage.element_size()
                + output_storage.numel() * output_storage.element_size()
            )
            workspace_count = testing.get_workspace_count(
                one_workspace_bytes, warmup_iterations, iterations
            )

        # exec_time is in microseconds
        exec_time = testing.benchmark(
            compiled_fn,
            workspace_generator=generate_tensors,
            workspace_count=workspace_count,
            stream=current_stream,
            warmup_iterations=warmup_iterations,
            iterations=iterations,
            use_cuda_graphs=True,
        )
        runtime_s = exec_time / 1.0e6
        fmas = (N * Z * P * Q) * K * (C * T * R * S)
        flop = 2 * fmas
        gflop = flop / 1.0e9
        gflops = gflop / runtime_s

        print("Average Runtime : ", exec_time / 1000, "ms")
        print("GFLOPS          : ", gflops)

        return exec_time


def _parse_comma_separated_ints(s: str) -> Tuple[int, ...]:
    try:
        return tuple(int(x.strip()) for x in s.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Invalid format. Expected comma-separated integers."
        ) from error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blackwell 3D convolution")

    # Convolution parameters
    parser.add_argument(
        "--ncdhw",
        type=_parse_comma_separated_ints,
        default=(1, 64, 3, 3, 3),
        help="Input tensor shape (N,C,D,H,W)",
    )
    parser.add_argument(
        "--ktrs",
        type=_parse_comma_separated_ints,
        default=(128, 3, 3, 3),
        help="Filter tensor shape components (K,T,R,S)",
    )
    parser.add_argument(
        "--stride_dhw",
        type=_parse_comma_separated_ints,
        default=(1, 1, 1),
        help="Stride (Sd,Sh,Sw)",
    )
    parser.add_argument(
        "--upper_pad_dhw",
        type=_parse_comma_separated_ints,
        default=(1, 1, 1),
        help="Upper padding (D,H,W)",
    )
    parser.add_argument(
        "--lower_pad_dhw",
        type=_parse_comma_separated_ints,
        default=(1, 1, 1),
        help="Lower padding (D,H,W)",
    )
    parser.add_argument(
        "--dil_dhw",
        type=_parse_comma_separated_ints,
        default=(1, 1, 1),
        help="Dilation (DilD,DilH,DilW)",
    )

    # Data type parameters
    parser.add_argument(
        "--ab_dtype",
        type=cutlass.dtype,
        choices=[
            cutlass.TFloat32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Int8,
            cutlass.Uint8,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        ],
        default=cutlass.Float16,
        help="Data type for A/B input tensors",
    )
    parser.add_argument(
        "--c_dtype",
        type=cutlass.dtype,
        choices=[
            cutlass.Float32,
            cutlass.Int32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Int8,
            cutlass.Uint8,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        ],
        default=cutlass.Float16,
        help="Data type for output tensor C",
    )
    parser.add_argument(
        "--acc_dtype",
        type=cutlass.dtype,
        choices=[cutlass.Float32, cutlass.Float16, cutlass.Int32],
        default=cutlass.Float32,
        help="Accumulator data type",
    )

    # Kernel parameters
    parser.add_argument(
        "--mma_tiler_mn",
        type=_parse_comma_separated_ints,
        default=(128, 128),
        help="MMA tiler shape (M,N)",
    )
    parser.add_argument(
        "--preferred_cluster_shape_mn",
        type=_parse_comma_separated_ints,
        default=(2, 1),
        help="Preferred cluster shape (M,N)",
    )
    parser.add_argument(
        "--fallback_cluster_shape_mn",
        type=_parse_comma_separated_ints,
        default=(1, 1),
        help="Fallback cluster shape (M,N)",
    )
    parser.add_argument(
        "--use_2cta_instrs",
        action="store_true",
        help="Enable 2CTA MMA instructions",
    )
    parser.add_argument(
        "--swizzle_size",
        type=int,
        default=1,
        help="Swizzling size in the unit of cluster for improving L2 cache hit rate",
    )
    parser.add_argument(
        "--raster_order",
        type=str,
        choices=["m", "n"],
        default="m",
        help="Rasterization order of clusters. Only used when swizzle_size > 1",
    )
    # Testing parameters
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-02,
        help="Tolerance for result comparison",
    )
    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=0,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--skip_ref_check",
        action="store_true",
        help="Skip reference checking",
    )
    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="Use circular buffer tensor sets to ensure L2 cold cache",
    )

    args = parser.parse_args()

    run(
        ncdhw=args.ncdhw,
        ktrs=args.ktrs,
        stride_dhw=args.stride_dhw,
        upper_pad_dhw=args.upper_pad_dhw,
        lower_pad_dhw=args.lower_pad_dhw,
        dil_dhw=args.dil_dhw,
        ab_dtype=args.ab_dtype,
        c_dtype=args.c_dtype,
        acc_dtype=args.acc_dtype,
        mma_tiler_mn=args.mma_tiler_mn,
        preferred_cluster_shape_mn=args.preferred_cluster_shape_mn,
        fallback_cluster_shape_mn=args.fallback_cluster_shape_mn,
        swizzle_size=args.swizzle_size,
        raster_along=args.raster_order,
        use_2cta_instrs=args.use_2cta_instrs,
        tolerance=args.tolerance,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        use_cold_l2=args.use_cold_l2,
        skip_ref_check=args.skip_ref_check,
    )
    print("PASS")
