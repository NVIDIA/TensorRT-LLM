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

# ruff: noqa: E501, E741

import argparse
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
import torch.nn.functional as F
from cutlass._mlir.dialects import nvvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import if_generate as _if_generate
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    PipelineAsync,
    PipelineOp,
    PipelineState,
    agent_sync,
    pipeline_init_arrive,
    pipeline_init_wait,
)

from .dense_gemm_persistent_dynamic_preferred_cluster import (
    PersistentDenseGemmKernelDynamicPreferredCluster,
)
from .dense_implicit_gemm_fprop import PersistentConvKernel


@dataclass(frozen=True)
class PipelineTmaCpAsyncUmma(PipelineAsync):
    """
    Merged pipeline for TMA (A+B+SFB) producers AND cp.async (SFA) producers sharing a single mbarrier set.

    Full mbarrier uses BOTH independent mbarrier counters:
      - tx_count   : incremented via mbarrier.expect_tx for TMA bytes, decremented by TMA completion
      - arrive_cnt : initialized to cp.async producer group size; decremented by cp.async.mbarrier.arrive.noinc

    Barrier phase flips only when tx_count == 0 AND arrive_cnt == 0, so the UMMA consumer performs
    a single wait/release per stage covering both producer streams.

    General variant: supports arbitrary cluster shapes (including cluster_shape_n > 1).
    """

    is_leader_cta: bool
    cta_group: cute.nvgpu.tcgen05.CtaGroup
    tx_count: int

    @staticmethod
    def create(
        *,
        num_stages: int,
        cpasynd_producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        tx_count: int,
        barrier_storage: cute.Pointer,
        cta_layout_vmnk: Optional[cute.Layout] = None,
        mcast_mode_mn: Tuple[int, int] = (1, 1),
        defer_sync: bool = False,
    ) -> "PipelineTmaCpAsyncUmma":
        """Creates and initializes a merged TMA+cp.async / UMMA pipeline."""
        if not isinstance(barrier_storage, cute.Pointer):
            raise ValueError(
                f"Expected barrier_storage to be a cute.Pointer, but got {type(barrier_storage)}"
            )

        producer = (PipelineOp.AsyncLoad, cpasynd_producer_group)
        consumer = (PipelineOp.TCGen05Mma, consumer_group)

        sync_object_full = pipeline.PipelineTmaUmma._make_sync_object(
            barrier_storage.align(min_align=8), num_stages, producer, tx_count
        )
        sync_object_empty = pipeline.PipelineTmaUmma._make_sync_object(
            barrier_storage.align(min_align=8) + num_stages, num_stages, consumer
        )

        cta_group = (
            cute.nvgpu.tcgen05.CtaGroup.ONE
            if cta_layout_vmnk is None or cute.size(cta_layout_vmnk, mode=[0]) == 1
            else cute.nvgpu.tcgen05.CtaGroup.TWO
        )

        if cta_layout_vmnk is None or cute.size(cta_layout_vmnk) == 1:
            consumer_mask = None
            is_leader_cta = True
            # No cross-CTA; remote arrives degenerate to self-arrive.
            producer_mask = cutlass.Int32(0)
        else:
            # Compute the empty-drain multicast mask: OR over A-axis and
            # B-axis TMA multicast sets (plus their V-peers).
            # PipelineTmaUmma._compute_mcast_arrival_mask returns the same
            # value but invoking it here produces stale IR for the merged
            # sync_object_empty; inline the equivalent computation instead.
            cta_rank_here = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            coord_self = cta_layout_vmnk.get_flat_coord(cta_rank_here)
            coord_peer = (coord_self[0] ^ 1, *coord_self[1:])
            # SFA cp.async remote-arrive target. Only meaningful for 2CTA, where
            # the peer CTA's cp.async lands in the leader's SMEM and both CTAs
            # must remote-arrive on the leader's sfa_full to close the peer-sSFA
            # race. The 2CTA flat rank is v + m*V + ..., so clearing the V bit
            # yields the leader rank. In a non-2CTA cluster (V=1) every CTA owns
            # its SFA locally, so producer_arrive_remote stays on the local
            # barrier and this value is unused; keep it as self-rank.
            producer_mask = (
                cta_rank_here & ~cutlass.Int32(1)
                if cta_group == cute.nvgpu.tcgen05.CtaGroup.TWO
                else cta_rank_here
            )
            mask_a_self = cute.nvgpu.cpasync.create_tma_multicast_mask(
                cta_layout_vmnk, coord_self, mcast_mode=2
            )
            mask_b_self = cute.nvgpu.cpasync.create_tma_multicast_mask(
                cta_layout_vmnk, coord_self, mcast_mode=1
            )
            mask_a_peer = cute.nvgpu.cpasync.create_tma_multicast_mask(
                cta_layout_vmnk, coord_peer, mcast_mode=2
            )
            mask_b_peer = cute.nvgpu.cpasync.create_tma_multicast_mask(
                cta_layout_vmnk, coord_peer, mcast_mode=1
            )
            if mcast_mode_mn[0] == 1 and mcast_mode_mn[1] == 1:
                consumer_mask = (
                    cutlass.Int32(mask_a_self)
                    | cutlass.Int32(mask_b_self)
                    | cutlass.Int32(mask_a_peer)
                    | cutlass.Int32(mask_b_peer)
                )
            elif mcast_mode_mn[1] == 1:
                consumer_mask = cutlass.Int32(mask_b_self) | cutlass.Int32(mask_b_peer)
            else:
                assert mcast_mode_mn[0] == 1
                consumer_mask = cutlass.Int32(mask_a_self) | cutlass.Int32(mask_a_peer)
            is_leader_cta = pipeline.PipelineTmaUmma._compute_is_leader_cta(cta_layout_vmnk)

        if not defer_sync:
            cute.arch.mbarrier_init_fence()
            if cta_layout_vmnk is None or cute.size(cta_layout_vmnk) == 1:
                agent_sync(Agent.ThreadBlock)
            else:
                agent_sync(Agent.ThreadBlockCluster, is_relaxed=True)

        return PipelineTmaCpAsyncUmma(
            sync_object_full,
            sync_object_empty,
            num_stages,
            producer_mask,
            consumer_mask,
            is_leader_cta,
            cta_group,
            tx_count,
        )

    def producer_acquire_tma(
        self,
        state: PipelineState,
        try_acquire_token: Optional[cutlass.Boolean] = None,
        *,
        expected_tx: Optional[cutlass.Int32] = None,
    ) -> None:
        """TMA producer path.

        Only the leader CTA issues ``arrive_and_expect_tx``. The multicast
        variant broadcasts BOTH the arrive and the tx-expect to every peer
        barrier in the cluster, so peer CTAs must do nothing here. Any peer
        ``arrive(1)`` would double-count the arrive on the peer barrier and
        opens a race window against cp.async producers draining first.
        """
        _if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase),
        )
        tx = self.tx_count if expected_tx is None else expected_tx

        def _leader_arrive_and_expect_tx() -> None:
            self.sync_object_full.arrive_and_expect_tx(state.index, tx)

        _if_generate(
            self.is_leader_cta,
            _leader_arrive_and_expect_tx,
        )

    def consumer_release(self, state: PipelineState) -> None:
        """UMMA consumer releases the shared empty barrier once per stage (multicast to peer CTAs)."""
        self.sync_object_empty.arrive(state.index, self.consumer_mask, self.cta_group)

    # ------------------------------------------------------------------
    # Shift-by-K cross-CTA producer commit primitives.
    # In 2CTA clusters, cp.async.mbarrier.arrive.noinc only signals the
    # local CTA's sfa_full, so the peer's cp.async landing on leader SMEM
    # is not covered. Split the commit into commit_group + wait_group(K)
    # + cross-CTA mbarrier.arrive(dst=leader_rank) so that arrives are
    # deferred by K iters, giving cp.async time to land on both CTAs
    # before the leader MMA fires.
    def producer_cp_async_commit(self) -> None:
        """Register all prior cp.async copies as one cp.async group."""
        cute.arch.cp_async_commit_group()

    def producer_cp_async_wait(self, num_inflight: int) -> None:
        """Block until at most ``num_inflight`` cp.async groups remain pending."""
        cute.arch.cp_async_wait_group(num_inflight)

    def producer_arrive_remote(self, stage_index) -> None:
        """Signal SFA cp.async completion on sfa_full[stage_index].

        In a 2CTA cluster the peer CTA's cp.async lands in the leader's SMEM,
        so both CTAs must arrive on the leader's sfa_full via DSMEM (producer_mask
        holds the leader rank). In a non-2CTA cluster every CTA owns its SFA
        locally, so the arrive stays on the local barrier with no DSMEM
        retargeting.
        """
        bar = self.sync_object_full.get_barrier(stage_index)
        if cutlass.const_expr(self.cta_group == cute.nvgpu.tcgen05.CtaGroup.TWO):
            cute.arch.mbarrier_arrive(bar, self.producer_mask)
        else:
            cute.arch.mbarrier_arrive(bar)


"""
A high-performance 3D implicit-GEMM based fprop convolution example for the NVIDIA Blackwell SM100 architecture using CUTE DSL.
- Input tensor A is NxDxHxWxC, must be C major.
- Filter tensor B is KxTxRxSxC, must be C major.
- Output tensor D is NxZxPxQxK, must be K major.

This kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations and for the im2col transformation of the input tensor A.
    - Utilizes Blackwell's tcgen05 MMA for matrix multiply-accumulate (MMA) operations (including 2cta mma instructions)
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Utilizes either a dense GEMM kernel or a persistent dense GEMM kernel for the implicit GEMM.

This implicit-GEMM based convolution works by converting the convolution into a GEMM problem with the following mapping:
- GEMM M dimension maps to NxZxPxQ
- GEMM N dimension maps to K
- GEMM K dimension maps to TxRxSxC
During the load of input tensor to SMEM, the TMA operation performs the im2col transformation on the input tensor A.
This transforms the A matrix into the required shape for the GEMM operation (NxZxPxQ by TxRxSxC), and may involve replication of the input elements.
Filter tensor can be loaded to SMEM without any transformation.
The output tensor D is then stored to GMEM via TMA im2col store (no transformation necessary).

To run this example:

.. code-block:: bash

    python examples/blackwell/kernel/conv/dense_blockscaled_implicit_gemm_fprop.py \
      --ncdhw 1,128,32,32,32 --ktrs 256,3,3,3                         \
      --use_2cta_instrs --mma_tiler_mn 256,128                        \
      --preferred_cluster_shape_mn 2,1 --fallback_cluster_shape_mn 1,1 \
      --upper_pad_dhw 1,1,1 --lower_pad_dhw 1,1,1                     \
      --stride_dhw 1,1,1 --dil_dhw 1,1,1

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/blackwell/kernel/conv/dense_blockscaled_implicit_gemm_fprop.py \
      --ncdhw 1,128,32,32,32 --ktrs 256,3,3,3                         \
      --use_2cta_instrs --mma_tiler_mn 256,128                        \
      --preferred_cluster_shape_mn 2,1 --fallback_cluster_shape_mn 1,1 \
      --upper_pad_dhw 1,1,1 --lower_pad_dhw 1,1,1                     \
      --stride_dhw 1,1,1 --dil_dhw 1,1,1                              \
      --warmup_iterations 1 --iterations 10 --skip_ref_check

Constraints:
* A/B data type is Float4E2M1FN (NVFP4) or Float8E4M3FN (MXFP8). Each pins its
  scale-factor format: NVFP4 takes a Float8E4M3FN scale over 16 channels, MXFP8 a
  Float8E8M0FNU scale over 32. A block-scaled narrow output takes the same format
  for its own scale factor, which pairs NVFP4 with an FP4 output and MXFP8 with an
  FP8 E4M3 one; a 16-bit output carries no scale factor.
* A/B tensor must have the same data type
* Mma tiler M must be 128 with use_2cta_instrs=False or 256 with True, the two
  shapes giving the per-CTA M of 128 that the block-scaled MMA requires
* Mma tiler N must be a multiple of 32 up to 256. Only 64, 128, 192 and 256 have a
  per-tile offset into the staged scale-factor chunk, so the others serve problems
  needing a single N tile
* Cluster shape M/N must be positive and power of 2, total cluster size <= 16
* Cluster shape M must be multiple of 2 if use_2cta_instrs=True
* The contiguous dimension of A/B/D tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 4, 8, and 16 for TFloat32,
  Float16/BFloat16, and Int8/Uint8/Float8, respectively.
* The GEMM-K tile is an independent compile-time choice: 64, 128 or 256 channels
  for NVFP4 (one, two or four MMA K instructions), 128 for MXFP8. It need not
  divide the input channel count C. A K tile wider than the channels a filter
  position has left is a partial tile: the im2col TMA zero-fills the A and B
  channels past C, the SFA transfer is clamped back inside the pixel's
  scale-factor row, and SFB is addressed per filter position over a span padded
  out to whole K tiles, so the tail contributes nothing.
"""


def _check_tensor_alignment(
    c: int,
    k: int,
    ab_dtype: Type[cutlass.Numeric],
    d_dtype: Type[cutlass.Numeric],
):
    """Check if the tensor alignment is valid for convolution.

    :param c: The number of input channels
    :type c: int
    :param k: The number of output channels
    :type k: int
    :param ab_dtype: The data type of the A and B operands
    :type ab_dtype: Type[cutlass.Numeric]
    :param d_dtype: The data type of the output tensor
    :type d_dtype: Type[cutlass.Numeric]
    """

    def check_contiguous_16B_alignment(dtype, num_major_elements):
        num_contiguous_elements = 16 * 8 // dtype.width
        return num_major_elements % num_contiguous_elements == 0

    if not check_contiguous_16B_alignment(d_dtype, k) or not check_contiguous_16B_alignment(
        ab_dtype, c
    ):
        raise testing.CantImplementError(
            f"Invalid tensor alignment: C = {c}, K = {k}, ab_dtype = {ab_dtype}, d_dtype = {d_dtype}"
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
    """Check that swizzle_size does not exceed the cluster count in the swizzled dimension."""
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


class Sm100BlockScaledPersistentDenseImplicitGemmKernel(PersistentConvKernel):
    """
    Persistent 3D convolution kernel.
    The input (A) is expected to be in 5D tensor (NDHWC) format and is loaded via TMA im2col load atom.
    The filter (B) is expected to be in 5D tensor (KTRSC) format and is loaded via TMA load atom.
    The output (D) is expected to be in 5D tensor (NZPQK) format and is stored via TMA im2col store.
    This class reuses the kernel from the PersistentDenseGemmKernel class for the implicit GEMM.

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param use_2cta_instrs: Whether to use CTA group 2 for advanced thread cooperation
    :type use_2cta_instrs: bool
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tiler (M,N)
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]
    :param filter_trs: Filter dimensions (T, R, S)
    :type filter_trs: Tuple[int, int, int]
    :param upper_padding_dhw: Upper padding in depth, height, width order
    :type upper_padding_dhw: Tuple[int, int, int]
    :param lower_padding_dhw: Lower padding in depth, height, width order
    :type lower_padding_dhw: Tuple[int, int, int]
    :param stride_dhw: Stride (Sd, Sh, Sw)
    :type stride_dhw: Tuple[int, int, int]
    :param dilation_dhw: Dilation (DilD, DilH, DilW)
    :type dilation_dhw: Tuple[int, int, int]
    :param swizzle_size: Swizzling size in the unit of cluster. 1 means no swizzle
    :type swizzle_size: int
    :param raster_along: Rasterization order of clusters. Only used when swizzle_size > 1.
    :type raster_along: Literal["m", "n"]

    :note: In current version, A and B tensor must be C major. D tensor must be K major.

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
        conv = Sm100BlockScaledPersistentDenseImplicitGemmKernel(
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=True,
            mma_tiler_mn=(128, 128),
            preferred_cluster_shape_mn=(2, 2),
            fallback_cluster_shape_mn=(1, 1),
            filter_trs=(3, 3, 3),
            upper_padding_dhw=(1, 1, 1),
            lower_padding_dhw=(1, 1, 1),
            stride_dhw=(1, 1, 1),
            dilation_dhw=(1, 1, 1),
        )
        conv(a, b, d, sfa, sfb, alpha, epilogue_op, stream=stream)
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        preferred_cluster_shape_mn: Tuple[int, int],
        fallback_cluster_shape_mn: Tuple[int, int],
        cta_tile_k: int,
        swizzle_size: int = 1,
        raster_along: Literal["m", "n"] = "m",
    ):
        # Skip PersistentConvKernel.__init__ and initialize the GEMM grandparent
        # directly. PersistentConvKernel takes trace-const filter/padding/stride/
        # dilation as host ints and stores them as static fields, but this kernel
        # carries those as runtime Int32 ops instead, so it has no static values
        # to hand up. Every field PersistentConvKernel would set is therefore
        # unused on this path, and every conv method it defines is overridden
        # here, so bypassing its __init__ loses nothing.
        PersistentDenseGemmKernelDynamicPreferredCluster.__init__(
            self,
            acc_dtype=acc_dtype,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=mma_tiler_mn,
            preferred_cluster_shape_mn=preferred_cluster_shape_mn,
            fallback_cluster_shape_mn=fallback_cluster_shape_mn,
            use_tma_store=True,  # Conv always uses im2col TMA store
            swizzle_size=swizzle_size,
            raster_along=raster_along,
        )
        self.sf_vec_size = sf_vec_size

        # cta_tile_k is the compile-time K tile (shapes the SMEM allocation).
        # The input channel count C is read from the dynamic tensors at runtime and
        # bounded only by the alignment of the channel axis, so one cubin serves
        # every C this kernel accepts. All output geometry (N/Z/P/Q/K) is runtime
        # as well.
        self.cta_tile_k = cta_tile_k

        # Override warp specialization for fp4 conv: 4 cp.async SFA warps + sched warp.
        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.cpasync_sfa_warp_id = (6, 7, 8, 9)
        self.sched_warp_id = 10
        # Dedicated residual G2S warp, only spawned when beta != 0 (has_residual).
        # threads_per_cta is finalized in __call__ once has_residual is known.
        self.residual_warp_id = 11
        self.base_warp_ids = (
            self.mma_warp_id,
            self.tma_warp_id,
            *self.cpasync_sfa_warp_id,
            self.sched_warp_id,
            *self.epilogue_warp_id,
        )
        self.threads_per_cta = 32 * len(self.base_warp_ids)
        # Override barrier ids; parent's preferred_cluster init reserves bar 0 for cta_sync.
        self.epilog_sync_bar_id = 1
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=self.epilog_sync_bar_id,
            num_threads=32 * len(self.epilogue_warp_id),
        )
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3

    def _setup_conv_input_attrs(self, a_tensor, b_tensor, d_tensor):
        """Validate and set input-dependent attributes.

        Sets a_dtype, b_dtype, d_dtype, a_major_mode, b_major_mode, d_layout.
        """
        self.a_dtype: Type[cutlass.Numeric] = a_tensor.element_type
        self.b_dtype: Type[cutlass.Numeric] = b_tensor.element_type
        self.d_dtype: Type[cutlass.Numeric] = d_tensor.element_type
        if cutlass.const_expr(a_tensor.leading_dim != 4):
            raise RuntimeError("The layout of a_tensor is not supported")
        if cutlass.const_expr(b_tensor.leading_dim != 4):
            raise RuntimeError("The layout of b_tensor is not supported")
        if cutlass.const_expr(d_tensor.leading_dim != 4):
            raise RuntimeError("The layout of d_tensor is not supported")
        self.a_major_mode = cute.nvgpu.OperandMajorMode.K
        self.b_major_mode = cute.nvgpu.OperandMajorMode.K
        self.d_layout = utils.LayoutEnum.ROW_MAJOR

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # Block-scaled D (SFD) output. Caller opts in by passing sfd_tensor;
        # gen_sfd is decided in __call__ from it. A narrow output carries one scale
        # factor per sfd_vec_size elements; a 16-bit output carries none. The SFD
        # block is the input SF block -- same element type (bound in
        # __call__) and same vector size -- so a quantized output feeds the next
        # operation directly: NVFP4 is E4M3 over 16 channels, MXFP8 E8M0 over 32.
        self.sfd_vec_size: int = self.sf_vec_size
        # M_D is the largest absolute value representable in the output dtype,
        # the full-scale target when quantizing each block to it. FP4 (E2M1)
        # magnitudes are {0,.5,1,1.5,2,3,4,6} so max is 6.0; E4M3 max normal is
        # 448.0. Only narrow SFD outputs use it, hence None otherwise.
        self.M_D: Optional[float] = {
            cutlass.Float4E2M1FN: 6.0,
            cutlass.Float8E4M3FN: 448.0,
        }.get(self.d_dtype, None)

    def _setup_conv_tma(
        self,
        a_tensor,
        b_tensor,
        d_tensor,
        c_tensor,
        tiled_mma,
        upper_pad_op,
        lower_pad_op,
        stride_op,
        dil_op,
    ):
        """Set up dual-cluster TMA atoms and tensors for im2col convolution.

        Creates preferred and fallback TMA atoms for A and B tensors, and a single
        TMA atom for D (im2col store is cluster-independent).

        upper_pad_op/lower_pad_op/stride_op/dil_op are the pad/stride/dilation
        operand tuples that build the im2col A descriptor corner. They are
        runtime Int32 tuples so one compiled cubin serves any pad/stride/dil
        config without recompilation.

        :returns: (tma_atom_a_preferred, tma_tensor_a_preferred,
                   tma_atom_a_fallback, tma_tensor_a_fallback,
                   tma_atom_b_preferred, tma_tensor_b_preferred,
                   tma_atom_b_fallback, tma_tensor_b_fallback,
                   tma_atom_d, tma_tensor_d)
        """
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Create 2-mode hierarchical tensor layout: (N, D, H, W, C) -> ((W, H, D, N), C)
        mA = cute.make_tensor(a_tensor.iterator, cute.select(a_tensor.layout, mode=[3, 2, 1, 0, 4]))
        mA = cute.group_modes(mA, begin=0, end=4)

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        a_internal_type = cutlass.TFloat32 if mA.element_type is cutlass.Float32 else None

        # Filter T/R/S sourced from the filter tensor b (K, T, R, S, C) rather
        # than host ints. The filter tensor is marked dynamic-layout, so these
        # extents are runtime Int32 and one compiled cubin serves any T/R/S.
        rt_filter_trs = (b_tensor.shape[1], b_tensor.shape[2], b_tensor.shape[3])

        # --- A preferred ---
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

        # --- A fallback ---
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
        # Filter (K, T, R, S, C) -> reorder to (K, (C, S, R, T)) for coord_iter indexing.
        mB = cute.make_tensor(b_tensor.iterator, cute.select(b_tensor.layout, mode=[0, 4, 3, 2, 1]))
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
        # CLC response size is 4B * 4 elements
        self.num_clc_response_bytes = 16

        # --- D: TMA im2col store (cluster-independent) ---
        mD = cute.make_tensor(d_tensor.iterator, cute.select(d_tensor.layout, mode=[3, 2, 1, 0, 4]))
        mD = cute.group_modes(mD, begin=0, end=4)

        epi_smem_layout = cute.slice_(self.d_smem_layout_staged, (None, None, (None, 0)))

        tma_atom_d, tma_tensor_d = cpasync.make_im2col_tma_atom(
            cpasync.CopyBulkTensorIm2ColS2GOp(),
            mD,
            epi_smem_layout,
            self.epi_tile,
        )
        tma_tensor_d = cute.coalesce(tma_tensor_d, target_profile=(1, 1))

        # --- Residual: im2col G2S load, the inverse of the D im2col store ---
        # The residual has the same (N,Z,P,Q,K) shape/layout as D, so it reuses
        # mD's hierarchical view and D's epilogue smem layout. Load mode (G2S)
        # requires the full im2col descriptor corner set; the residual is an
        # identity element map (a 1x1x1 tap with no padding/stride/dilation), so
        # every corner/stride is trivial: lower/upper corners and padding are 0,
        # the DHW stride is 1, and the SRT lower/stride are 0/1.
        if cutlass.const_expr(self.has_residual):
            mC = cute.make_tensor(
                c_tensor.iterator, cute.select(c_tensor.layout, mode=[3, 2, 1, 0, 4])
            )
            mC = cute.group_modes(mC, begin=0, end=4)
            tma_atom_c, tma_tensor_c = cpasync.make_im2col_tma_atom(
                cpasync.CopyBulkTensorIm2ColG2SOp(),
                mC,
                epi_smem_layout,
                self.epi_tile,
                lower_corner_whd=(0, 0, 0),
                upper_corner_whd=(0, 0, 0),
                lower_padding_whd=(0, 0, 0),
                upper_padding_whd=(0, 0, 0),
                stride_whd=(1, 1, 1),
                lower_srt=(0, 0, 0),
                stride_srt=(1, 1, 1),
            )
            tma_tensor_c = cute.coalesce(tma_tensor_c, target_profile=(1, 1))
        else:
            tma_atom_c, tma_tensor_c = None, None

        def add_dummy_batch_dimension(tensor):
            new_layout = cute.append(tensor.layout, cute.make_layout(1))
            return cute.make_tensor(tensor.iterator, new_layout)

        tma_tensor_a_preferred = add_dummy_batch_dimension(tma_tensor_a_preferred)
        tma_tensor_a_fallback = add_dummy_batch_dimension(tma_tensor_a_fallback)
        tma_tensor_b_preferred = add_dummy_batch_dimension(tma_tensor_b_preferred)
        tma_tensor_b_fallback = add_dummy_batch_dimension(tma_tensor_b_fallback)
        tma_tensor_d = add_dummy_batch_dimension(tma_tensor_d)
        if cutlass.const_expr(self.has_residual):
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
            tma_atom_d,
            tma_tensor_d,
            tma_atom_c,
            tma_tensor_c,
        )

    def _setup_attributes(self):
        """Set up configurations that are dependent on convolution inputs."""
        # Compute mma instruction shapes
        # (MMA_Tile_Shape_M, MMA_Tile_Shape_N, MMA_Inst_Shape_K)
        self.mma_inst_shape_mn = (
            self.mma_tiler[0],
            self.mma_tiler[1],
        )
        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        # cta_tile_k is a compile-time choice (64, 128 or 256 for FP4). It sets
        # how many MMA-instruction K blocks a K tile spans.
        mma_inst_tile_k = self.cta_tile_k // mma_inst_shape_k
        # Expose for overlapping-accum SF column accounting (see below).
        self.mma_inst_tile_k = mma_inst_tile_k
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            (mma_inst_shape_k * mma_inst_tile_k,),
        )
        # Scale factors tile K on their own cadence: an SF atom cannot be staged
        # in part, so a narrower A/B K tile still stages a whole one and the A/B
        # tiles sharing it each read their own part. Only MXFP8 gets there, with a
        # 128-channel atom against a 64 tile; NVFP4's atom is 64, at or below
        # every legal tile, so its ratio is 1 and every SF path keeps the A/B one.
        self.sf_tile_k = sf_k_tile_channels(self.cta_tile_k, self.sf_vec_size)
        self.ab_k_tiles_per_sf_k_tile = self.sf_tile_k // self.cta_tile_k
        self.sf_mma_inst_tile_k = self.sf_tile_k // mma_inst_shape_k
        # Flat-K tiler the SF layouts (smem, tmem, TMA box) are built from.
        self.flat_mma_tiler_sf = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            self.sf_tile_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            self.sf_tile_k,
        )
        # SFA 4-byte cp.async per SF K tile: each carries 4 SF blocks, and an SF
        # K tile is a whole number of atoms at every legal tile_k, so the count is
        # at least one and the power of two the group rotation below needs.
        self.num_sfa_cpasync = self.sf_tile_k // (self.sf_vec_size * 4)
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        # CTA-level SFB tile shape (used for SFB TMEM column accounting in
        # overlapping-accum). N-mode rounds MMA_N up to 128 like SFB MMA.
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.sf_tile_k,
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # Compute epilogue subtile
        self.epi_tile = utils.sm100.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.d_layout,
            self.d_dtype,
        )
        # N-extent of one epilogue subtile (used by overlapping-accum early release).
        self.epi_tile_n = cute.size(self.epi_tile[1])

        self.smem_capacity = utils.get_smem_capacity_in_bytes()

        # Setup A/B/D stage count in shared memory and ACC stage count in tensor memory
        self.num_acc_stage, self.num_ab_stage, self.num_d_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.d_dtype,
            self.d_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.flat_mma_tiler_sf,
            self.smem_capacity,
            self.occupancy,
            self.has_residual,
        )

        # Overlapping-accum: when only one accumulator stage fits in TMEM
        # (MMA_N == 256), squeeze a second logical acc buffer into the columns
        # otherwise reserved for SFA/SFB so the math mainloop of the next tile
        # can overlap the epilogue of the current tile.
        self.overlapping_accum = self.num_acc_stage == 1
        # SF columns follow the SF K cadence: TMEM holds whole chunks.
        self.num_sfa_tmem_cols = (self.cta_tile_shape_mnk[0] // 32) * self.sf_mma_inst_tile_k
        self.num_sfb_tmem_cols = (self.cta_tile_shape_mnk_sfb[1] // 32) * self.sf_mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        # Reverse-subtile index (raw loop counter) at which acc can release
        # early: the number of whole epilogue subtiles the SFA/SFB-aliased
        # columns span. Once the reverse walk has drained that many subtiles the
        # shared columns are fully read out and the next tile's MMA may reuse
        # them. When the shared columns fit within one subtile (num_sf_tmem_cols
        # <= epi_tile_n) this is 0, i.e. release right after the first subtile.
        self.iter_acc_early_release_in_epilogue = self.num_sf_tmem_cols // self.epi_tile_n

        # Setup CLC stage (single-stage CLC pipeline)
        self.num_clc_stage = 1
        assert self.num_clc_stage == 1, "Only single-stage CLC pipeline is supported"

        # Compute A/B/D shared memory layout
        self.a_smem_layout_staged = utils.sm100.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = utils.sm100.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )
        # The SF layouts take the flat-K SF tiler: blockscaled utils expect a
        # scalar K where conv carries a (K,) tuple, and K runs on the SF cadence.
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.flat_mma_tiler_sf,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.flat_mma_tiler_sf,
            self.sf_vec_size,
            self.num_ab_stage,
        )

        self.d_smem_layout_staged = utils.sm100.make_smem_layout_epi(
            self.d_dtype, self.d_layout, self.epi_tile, self.num_d_stage
        )

        # Compute the number of tensor memory allocation columns.
        # For overlapping-accum the second acc buffer is strided into the
        # SFA/SFB columns, so the inherited helper (which builds a contiguous
        # 2-stage fake) would under-count. Derive the column count from the
        # same fake tensor used at the injection sites instead.
        if cutlass.const_expr(self.overlapping_accum):
            self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(
                self._make_acc_fake_tensor(tiled_mma, self.mma_tiler),
                arch=self.arch,
            )
        else:
            self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
                tiled_mma, self.mma_tiler, self.num_acc_stage, self.arch
            )

        # Compute preferred cluster layout for dual-cluster scheduling
        self.preferred_cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.preferred_cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.num_preferred_mcast_ctas_a = cute.size(self.preferred_cluster_layout_vmnk.shape[2])
        self.num_preferred_mcast_ctas_b = cute.size(self.preferred_cluster_layout_vmnk.shape[1])
        self.is_preferred_a_mcast = self.num_preferred_mcast_ctas_a > 1
        self.is_preferred_b_mcast = self.num_preferred_mcast_ctas_b > 1

        # Fallback cluster layout was already computed as cluster_layout_vmnk above
        self.fallback_cluster_layout_vmnk = self.cluster_layout_vmnk
        self.num_fallback_mcast_ctas_a = self.num_mcast_ctas_a
        self.num_fallback_mcast_ctas_b = self.num_mcast_ctas_b
        self.is_fallback_a_mcast = self.is_a_mcast
        self.is_fallback_b_mcast = self.is_b_mcast

        # SFB cluster layouts (SFB always uses 1CTA group, thr_id shape = 1)
        self.preferred_cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.preferred_cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )
        self.fallback_cluster_layout_sfb_vmnk = self.cluster_layout_sfb_vmnk

    @cute.jit
    def __call__(
        self,
        a_tensor: cute.Tensor,
        b_tensor: cute.Tensor,
        d_tensor: cute.Tensor,
        sfa_tensor: cute.Tensor,
        sfb_tensor: cute.Tensor,
        alpha: cute.Tensor,
        epilogue_op: cutlass.Constexpr = lambda x: x,
        sfd_tensor: Optional[cute.Tensor] = None,
        norm_const_tensor: Optional[cute.Tensor] = None,
        bias_tensor: Optional[cute.Tensor] = None,
        c_tensor: Optional[cute.Tensor] = None,
        beta: cutlass.Constexpr = 0.0,
        # Conv pad/stride/dilation as cute.compile entry scalars. The caller
        # passes boxed cutlass.Int32(...) so they lower to runtime SSA: one
        # cubin then serves any pad/stride/dil because the SAME scalars feed
        # BOTH the im2col A descriptor corner (host-side, in _setup_conv_tma)
        # AND the device per-tile coord math. Boxing matters: a raw Python int
        # here would embed its value in the mangled function name, defeating
        # reuse and forcing recompilation per config.
        rt_upper_pad_d: cutlass.Int32 = None,
        rt_upper_pad_h: cutlass.Int32 = None,
        rt_upper_pad_w: cutlass.Int32 = None,
        rt_lower_pad_d: cutlass.Int32 = None,
        rt_lower_pad_h: cutlass.Int32 = None,
        rt_lower_pad_w: cutlass.Int32 = None,
        rt_stride_d: cutlass.Int32 = None,
        rt_stride_h: cutlass.Int32 = None,
        rt_stride_w: cutlass.Int32 = None,
        rt_dil_d: cutlass.Int32 = None,
        rt_dil_h: cutlass.Int32 = None,
        rt_dil_w: cutlass.Int32 = None,
        stream: cuda.CUstream = None,
    ):
        """Execute the persistent convolution operation with dynamic preferred cluster scheduling.

        :param a_tensor: Input tensor A - (N, D, H, W, C) layout
        :param b_tensor: Filter tensor B - (K, T, R, S, C) layout
        :param d_tensor: Output tensor D - (N, Z, P, Q, K) layout
        :param sfa_tensor: Block scale factor tensor for A
        :param sfb_tensor: Block scale factor tensor for B
        :param alpha: 1-element FP32 device tensor; applied to the accumulator in FP32
            before output quantization
        :param epilogue_op: Optional elementwise lambda function to apply to the output tensor
        :param sfd_tensor: Output scale factor tensor (NVFP4 SFD); pass None to skip SFD generation
        :param norm_const_tensor: Optional 1-element FP32 device tensor; per-tensor amax-derived
            global FP4 scale, multiplied into the SFD encode step. Only used when
            sfd_tensor is provided.
        :param bias_tensor: Optional length-K (output channel) device tensor; a
            per-output-channel bias added to the accumulator in FP32 as
            D = epilogue_op(alpha * acc + bias). Pass None to skip bias.
        :param c_tensor: Optional (N, Z, P, Q, K) device tensor with the
            same shape/layout as the output D; a per-element residual added to
            the accumulator in FP32 as D = epilogue_op(alpha * acc + bias +
            beta * residual). Loaded through the epilogue via a TMA im2col G2S
            copy (the inverse of the output store) into shared memory, then read
            to registers. Required when beta != 0.
        :param beta: compile-time residual scaling constant. beta == 0 removes
            the entire residual path (smem, pipeline, TMA load, add) via const
            DCE; beta != 0 scales the residual as D = alpha * acc + bias +
            beta * residual and requires c_tensor. A distinct beta value
            yields a distinct cubin.
        :param stream: CUDA stream for asynchronous execution; defaults to None
        """
        self._setup_conv_input_attrs(a_tensor, b_tensor, d_tensor)
        self.sf_dtype: Type[cutlass.Numeric] = sfa_tensor.element_type
        # SFD shares the input sf_dtype (E4M3 for NVFP4, E8M0 for MXFP8).
        self.sfd_dtype: Type[cutlass.Numeric] = self.sf_dtype
        # SFD opt-in: caller supplies sfd_tensor. SFD is valid only with a narrow
        # output (FP4 or FP8 E4M3). The normalization constant is required only
        # for the block-scaled output path.
        self.gen_sfd: bool = sfd_tensor is not None and norm_const_tensor is not None
        # Bias opt-in: caller supplies a length-K per-output-channel tensor.
        self.has_bias: bool = bias_tensor is not None
        # Residual opt-in via compile-time beta: beta == 0 removes the whole
        # residual path through const DCE; beta != 0 scales the per-element
        # residual as alpha*acc + bias + beta*residual and needs c_tensor.
        self.beta: cutlass.Constexpr = beta
        self.has_residual: bool = beta != 0.0
        if cutlass.const_expr(self.has_residual and c_tensor is None):
            raise ValueError("beta != 0 requires a c_tensor")
        # Finalize the CTA thread count now that has_residual is known: the
        # dedicated residual G2S warp only exists when beta != 0, so beta == 0
        # launches with no extra idle warp.
        if cutlass.const_expr(self.has_residual):
            self.threads_per_cta = 32 * (len(self.base_warp_ids) + 1)
        else:
            self.threads_per_cta = 32 * len(self.base_warp_ids)
        if cutlass.const_expr(
            self.gen_sfd and self.d_dtype not in (cutlass.Float4E2M1FN, cutlass.Float8E4M3FN)
        ):
            raise ValueError(
                "SFD output is only supported for Float4E2M1FN (NVFP4) or "
                f"Float8E4M3FN (MXFP8) output; got d_dtype={self.d_dtype}"
            )
        # sf_dtype is derived from SFA alone but also drives the SFB smem layout
        # and barrier transaction bytes; SFB/SFD must match it or scales get
        # misinterpreted and barrier byte counts diverge.
        if cutlass.const_expr(sfb_tensor.element_type is not self.sf_dtype):
            raise ValueError(
                f"SFB dtype ({sfb_tensor.element_type}) must match SFA/sf_dtype ({self.sf_dtype})"
            )
        if cutlass.const_expr(self.gen_sfd and sfd_tensor.element_type is not self.sfd_dtype):
            raise ValueError(
                f"SFD dtype ({sfd_tensor.element_type}) must match sf_dtype ({self.sfd_dtype})"
            )
        # sBias is allocated with d_dtype but the staging cp.async width is
        # taken from the bias tensor's dtype, so a mismatch overruns the smem
        # buffer. Require bias to match the output dtype.
        if cutlass.const_expr(self.has_bias and bias_tensor.element_type is not self.d_dtype):
            raise ValueError(
                f"bias dtype ({bias_tensor.element_type}) must match output "
                f"d_dtype ({self.d_dtype})"
            )
        self._setup_attributes()

        # Pad/stride/dilation operands that BOTH the im2col A descriptor corner
        # and the device per-tile coord math consume. They are the runtime
        # Int32 scalars passed at the cute.compile entry, so one cubin serves
        # any config. Threading the SAME values into the descriptor and the
        # kernel is required: runtime-ing only one side would let A's load
        # coords (descriptor) desync from SFA's cp.async coords (device),
        # zeroing the accumulator on any non-default config.
        upper_pad_op = (rt_upper_pad_d, rt_upper_pad_h, rt_upper_pad_w)
        lower_pad_op = (rt_lower_pad_d, rt_lower_pad_h, rt_lower_pad_w)
        stride_op = (rt_stride_d, rt_stride_h, rt_stride_w)
        dil_op = (rt_dil_d, rt_dil_h, rt_dil_w)

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
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
            tma_atom_d,
            tma_tensor_d,
            tma_atom_c,
            tma_tensor_c,
        ) = self._setup_conv_tma(
            a_tensor,
            b_tensor,
            d_tensor,
            c_tensor,
            tiled_mma,
            upper_pad_op,
            lower_pad_op,
            stride_op,
            dil_op,
        )

        # SFB tiled_mma (always 1CTA group for SFB)
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # SFB tensor reshape: match B's GEMM view (K, C*S*R*T) for correct TMA tiling.
        # B filter is (K, T, R, S, C) -> reorder to (K, C, S, R, T) -> group to (K, C*S*R*T).
        # This makes SFB N dim = K (GEMM N) and SFB K dim = C*S*R*T/16 (GEMM K / sf_vec).
        # Use a flat integer K extent for tile_atom_to_shape_SF, NOT the hierarchical
        # shape from group_modes: group_modes produces (K, (C, S, R, T)) with nested
        # strides, which tile_atom_to_shape_SF interprets incorrectly vs the flat
        # swizzled SFB storage.
        #
        # The N extent MUST be the dynamic b.shape[0], not a trace-time-static int.
        # For cta_n==192 the SFB layout is reshaped into overlapping 256-wide windows
        # ((2,2),y) with y=ceil_div(N_sf,4) below. A static N collapses y to a literal
        # 1, and CuTe coalesces the size-1 sub-mode away, flattening the window RestN
        # from (2,?) to 2. The scheduler still emits ceil(N/192) cta-tiles, so for a
        # partial last tile slice_n runs OOB on the flattened RestN and the TMA load
        # clamps to zeros -> wrong (zero) SFB scaling on that tile. A dynamic N keeps y
        # symbolic, blocks the coalesce, and slice_n decomposes to an in-bounds coord.
        # C*T*R*S = GEMM-K; T/R/S come from the dynamic filter tensor (runtime). The
        # channel extent is the span SFB was allocated at -- C rounded up to whole SF
        # K tiles -- on two counts. The N-mode rest stride tile_atom_to_shape_SF
        # derives is the tensor's whole atom count, so a short extent walks the
        # output-channel axis at the wrong pitch. And the rounding puts every filter
        # position on a tile boundary, which keeps a K tile inside one position and
        # lets the consumer's flat K index address the tile directly. The span comes
        # from the runtime channel count, so one cubin serves every C.
        sfb_c_extent = cute.ceil_div(b_tensor.shape[4], self.sf_tile_k) * self.sf_tile_k
        b_shape_for_sfb = (
            b_tensor.shape[0],
            sfb_c_extent * b_tensor.shape[1] * b_tensor.shape[2] * b_tensor.shape[3],
            1,
        )
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b_shape_for_sfb, self.sf_vec_size)
        sfb_tensor = cute.make_tensor(sfb_tensor.iterator, sfb_layout)

        # SFB TMA setup (dual atoms for preferred + fallback cluster shapes)
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))

        def _make_sfb_tma(cluster_shape_mn, cluster_layout_sfb_vmnk):
            sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(cluster_shape_mn, tiled_mma.thr_id)
            atom, tensor = cute.nvgpu.make_tiled_tma_atom_B(
                sfb_op,
                sfb_tensor,
                sfb_smem_layout,
                self.mma_tiler_sfb,
                tiled_mma_sfb,
                cluster_layout_sfb_vmnk.shape,
                internal_type=cutlass.Int16,
            )
            # Special-case for N=192: align the SFB scale-factor layout for Tensor Memory packing
            if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                x = tensor.stride[0][1]
                y = cute.ceil_div(tensor.shape[0][1], 4)
                new_shape = (
                    (tensor.shape[0][0], ((2, 2), y)),
                    tensor.shape[1],
                    tensor.shape[2],
                )
                x_times_3 = 3 * x
                new_stride = (
                    (tensor.stride[0][0], ((x, x), x_times_3)),
                    tensor.stride[1],
                    tensor.stride[2],
                )
                tensor = cute.make_tensor(
                    tensor.iterator,
                    cute.make_layout(new_shape, stride=new_stride),
                )
            return atom, tensor

        tma_atom_sfb_preferred, tma_tensor_sfb_preferred = _make_sfb_tma(
            self.preferred_cluster_shape_mn, self.preferred_cluster_layout_sfb_vmnk
        )
        tma_atom_sfb_fallback, tma_tensor_sfb_fallback = _make_sfb_tma(
            self.fallback_cluster_shape_mn, self.fallback_cluster_layout_sfb_vmnk
        )

        # Compute A, B, SFB copy sizes for separate TMA pipelines.
        # SFB has its own pipeline because B and SFB use different mcast
        # masks (cluster_layout_vmnk vs cluster_layout_sfb_vmnk), so bundling
        # them into one pipeline can stall TX accumulation in cluster > 1.
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_a_bytes = a_copy_size * atom_thr_size
        self.num_tma_load_b_bytes = b_copy_size * atom_thr_size
        self.num_tma_load_sfb_bytes = sfb_copy_size * atom_thr_size

        # Residual reuses the D epilogue smem layout (same shape/dtype as the
        # output); one epilogue subtile's worth of bytes per TMA load.
        c_smem_layout = cute.slice_(self.d_smem_layout_staged, (None, None, 0))
        self.num_tma_load_c_bytes = cute.size_in_bytes(self.d_dtype, c_smem_layout)

        self.buffer_align_bytes = 1024

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            # Shared TMA pipeline barriers for A, B and SFB
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            # CPASYNC SFA pipeline barriers
            sfa_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # CLC pipeline barriers and response buffer
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_clc_stage * 2]
            clc_response: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Int32, self.num_clc_response_bytes // 4 * self.num_clc_stage
                ],
                16,
            ]
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sD: cute.struct.Align[
                cute.struct.MemRange[
                    self.d_dtype,
                    cute.cosize(self.d_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (cta_tile_n,) staged bias row for the shared-memory broadcast.
            # One value per output channel in the CTA's N-tile; sized to zero
            # when no bias is supplied so it costs no smem.
            sBias: cute.struct.Align[
                cute.struct.MemRange[
                    self.d_dtype,
                    self.cta_tile_shape_mnk[1] if self.has_bias else 0,
                ],
                self.buffer_align_bytes,
            ]
            # (EPI_TILE_M, EPI_TILE_N, STAGE) residual tile, same layout as sD.
            # Sized to zero when no residual is supplied so it costs no smem.
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.d_dtype,
                    cute.cosize(self.d_smem_layout_staged.outer) if self.has_residual else 0,
                ],
                self.buffer_align_bytes,
            ]
            # Residual TMA load pipeline barriers (full/empty per stage).
            c_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_d_stage * 2 if self.has_residual else 0
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
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Compute grid size and scheduler params for both cluster shapes
        self.fallback_tile_sched_params, _ = self._compute_grid(
            tma_tensor_d,
            self.cta_tile_shape_mnk,
            self.fallback_cluster_shape_mn,
            self.swizzle_size,
            self.raster_along,
        )
        self.preferred_tile_sched_params, preferred_grid = self._compute_grid(
            tma_tensor_d,
            self.cta_tile_shape_mnk,
            self.preferred_cluster_shape_mn,
            self.swizzle_size,
            self.raster_along,
        )
        # Extract conv spatial dims from input/output tensors
        # a_tensor: (N, D, H, W, C), d_tensor: (N, Z, P, Q, K)
        conv_N = cute.size(a_tensor, mode=[0])
        conv_D = cute.size(a_tensor, mode=[1])
        conv_H = cute.size(a_tensor, mode=[2])
        conv_W = cute.size(a_tensor, mode=[3])
        conv_C = cute.size(a_tensor, mode=[4])
        conv_Z = cute.size(d_tensor, mode=[1])
        conv_P = cute.size(d_tensor, mode=[2])
        conv_Q = cute.size(d_tensor, mode=[3])
        # Filter T/R/S from the dynamic-layout filter tensor b_tensor (K, T, R, S, C):
        # runtime Int32 so one cubin serves any T/R/S. Fed to the device
        # kernel's conv_T/R/S params.
        conv_T, conv_R, conv_S = b_tensor.shape[1], b_tensor.shape[2], b_tensor.shape[3]
        # stride/dil/pad come from the SAME operand tuples that built the A
        # descriptor corner above, so the device's SFA cp.async coords stay in
        # lockstep with A's im2col load coords under any runtime config.
        stride_d, stride_h, stride_w = stride_op
        dil_d, dil_h, dil_w = dil_op
        # SFA cp.async path uses the leading (lower) pad to mirror what the
        # im2col TMA A descriptor subtracts when mapping output->input coords:
        #   d_in = z*stride - lower_pad + t*dil
        # Using upper_padding here desyncs SFA from A on asymmetric pads,
        # making A's nonzero positions multiply SFA's zero positions -> acc=0.
        pad_d, pad_h, pad_w = lower_pad_op
        K_gemm_tile = self.mma_tiler[2][0]  # mma_inst_shape_k * mma_inst_tile_k

        # Build mSFD tensor with the (M, K, 1) profile of mD, where M is the
        # flat NZPQ output-pixel axis and the K dimension is expressed as
        # (sfd_vec_size, sf_k_padded) with strides (0, 1) so that sfd_vec_size
        # consecutive K elements share one physical scale factor. Storage layout
        # matches SFA's K-contig-in-M form: (mn=NZPQ, sf_k_padded, 1) with
        # mn-stride = sf_k_padded.
        if cutlass.const_expr(self.gen_sfd):
            # Source N/Z/P/Q/K from the dynamic output tensor (runtime Int) so
            # one cubin serves any output geometry. The SFD store is a scalar
            # per-element STG with a static (unrolled) element count, so only the
            # per-thread tile is static; the global extent/strides can be runtime.
            sfd_N = conv_N
            sfd_Z = conv_Z
            sfd_P = conv_P
            sfd_Q = conv_Q
            conv_K = cute.size(d_tensor, mode=[4])
            sf_k = (conv_K + self.sfd_vec_size - 1) // self.sfd_vec_size
            sf_k_padded = ((sf_k + 3) // 4) * 4  # pad sf_k to multiple of 4
            stride_mn = sf_k_padded
            # The M axis is a single flat extent N*Z*P*Q with a uniform row
            # stride, not a hierarchical (Q,P,Z,N) tuple. local_tile splits the
            # leading M mode by the CTA tile, and a runtime-valued hierarchical
            # mode does not divide correctly there: the store then lands on wrong
            # rows whenever N>1 (or any spatial extent grows past one tile),
            # leaving some scale-factor rows unwritten. A flat runtime extent
            # tiles cleanly because the SFD gmem buffer is dense row-major NZPQ
            # and a plain STG needs no hierarchical structure.
            sfd_M = sfd_N * sfd_Z * sfd_P * sfd_Q
            mSFD_layout = cute.make_layout(
                (
                    sfd_M,
                    (self.sfd_vec_size, sf_k_padded),
                    1,
                ),
                stride=(
                    stride_mn,
                    (0, 1),
                    sfd_M * stride_mn,
                ),
            )
            mSFD_mnl = cute.make_tensor(sfd_tensor.iterator, mSFD_layout)
        else:
            mSFD_mnl = None

        # Build mBias tensor: a per-output-channel bias broadcast to the same
        # ((Q,P,Z,N), K, 1) = (M, N, L) profile as mD. The bias varies along the
        # output channel K (GEMM-N, real stride 1) and broadcasts across every
        # spatial output position (GEMM-M = Q,P,Z,N modes carry stride 0), so the
        # epilogue reads it through the same partition_C chain as the accumulator
        # and each thread lands on its own N-column bias scalar.
        if cutlass.const_expr(bias_tensor is not None):
            # The M (spatial) modes are stride-0 broadcast: every output row
            # reads the same bias[k], so their extent never enters an address and
            # can be a compile-time value. Using a static M extent keeps the
            # partitioned smem-read layout static (needed for a vectorized smem load)
            # while the N (K/channel) axis carries a runtime stride so one cubin
            # serves any channel count. The M extent only needs to cover the CTA
            # tile; the real output-M size is supplied at the store site through
            # the runtime tile coordinate, not through this layout.
            mBias_layout = cute.make_layout(
                (
                    (self.cta_tile_shape_mnk[0], 1, 1, 1),
                    cute.size(d_tensor, mode=[4]),
                    1,
                ),
                stride=((0, 0, 0, 0), 1, 0),
            )
            mBias_mnl = cute.make_tensor(bias_tensor.iterator, mBias_layout)
        else:
            mBias_mnl = None

        # Launch the megakernel synchronously (dispatcher selects preferred vs fallback)
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a_preferred,
            tma_tensor_a_preferred,
            tma_atom_a_fallback,
            tma_tensor_a_fallback,
            tma_atom_b_preferred,
            tma_tensor_b_preferred,
            tma_atom_b_fallback,
            tma_tensor_b_fallback,
            sfa_tensor,
            tma_atom_sfb_preferred,
            tma_tensor_sfb_preferred,
            tma_atom_sfb_fallback,
            tma_tensor_sfb_fallback,
            tma_atom_d,
            tma_tensor_d,
            tma_atom_c,
            tma_tensor_c,
            mSFD_mnl,
            alpha,
            norm_const_tensor,
            mBias_mnl,
            self.preferred_cluster_layout_vmnk,
            self.fallback_cluster_layout_vmnk,
            self.preferred_cluster_layout_sfb_vmnk,
            self.fallback_cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.d_smem_layout_staged,
            self.epi_tile,
            self.preferred_tile_sched_params,
            self.fallback_tile_sched_params,
            epilogue_op,
            conv_T,
            conv_R,
            conv_S,
            stride_d,
            stride_h,
            stride_w,
            dil_d,
            dil_h,
            dil_w,
            pad_d,
            pad_h,
            pad_w,
            conv_D,
            conv_H,
            conv_W,
            conv_Z,
            conv_P,
            conv_Q,
            conv_N,
            conv_C,
            K_gemm_tile,
        ).launch(
            grid=preferred_grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.preferred_cluster_shape_mn, 1),
            fallback_cluster=(*self.fallback_cluster_shape_mn, 1),
            stream=stream,
            smem_merge_branch_allocs=True,
        )
        return

    # GPU device kernel - megakernel dispatcher for preferred/fallback cluster
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a_preferred: cute.CopyAtom,
        mA_mkl_preferred: cute.Tensor,
        tma_atom_a_fallback: cute.CopyAtom,
        mA_mkl_fallback: cute.Tensor,
        tma_atom_b_preferred: cute.CopyAtom,
        mB_nkl_preferred: cute.Tensor,
        tma_atom_b_fallback: cute.CopyAtom,
        mB_nkl_fallback: cute.Tensor,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb_preferred: cute.CopyAtom,
        mSFB_nkl_preferred: cute.Tensor,
        tma_atom_sfb_fallback: cute.CopyAtom,
        mSFB_nkl_fallback: cute.Tensor,
        tma_atom_d: cute.CopyAtom,
        mD_mnl: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: Optional[cute.Tensor],
        mSFD_mnl: Optional[cute.Tensor],
        alpha: cute.Tensor,
        norm_const_tensor: Optional[cute.Tensor],
        mBias_mnl: Optional[cute.Tensor],
        preferred_cluster_layout_vmnk: cute.Layout,
        fallback_cluster_layout_vmnk: cute.Layout,
        preferred_cluster_layout_sfb_vmnk: cute.Layout,
        fallback_cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        d_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        preferred_tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
        fallback_tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        # Conv parameters for SFA im2col coordinate mapping. conv_T/R/S are
        # runtime Int32 so one cubin serves any filter size; they only feed
        # register comparisons in the K-loop (S->R->T wrap), adding no IDIV.
        conv_T: cutlass.Int32,
        conv_R: cutlass.Int32,
        conv_S: cutlass.Int32,
        # pad/stride/dilation are runtime Int32 so one cubin serves any conv
        # config; they feed the per-tile base-coord arithmetic (z*stride - pad)
        # and the K-loop dilation step, swapping immediates for registers with
        # no new IDIV.
        stride_d: cutlass.Int32,
        stride_h: cutlass.Int32,
        stride_w: cutlass.Int32,
        dil_d: cutlass.Int32,
        dil_h: cutlass.Int32,
        dil_w: cutlass.Int32,
        pad_d: cutlass.Int32,
        pad_h: cutlass.Int32,
        pad_w: cutlass.Int32,
        # Coordinate geometry: runtime Int32 so the per-tile NDHW decomposition
        # (m_global -> n,z,p,q) lowers to real IDIV and one cubin serves any
        # N/D/H/W/Z/P/Q.
        input_D: cutlass.Int32,
        input_H: cutlass.Int32,
        input_W: cutlass.Int32,
        output_Z: cutlass.Int32,
        output_P: cutlass.Int32,
        output_Q: cutlass.Int32,
        input_N: cutlass.Int32,
        input_C: cutlass.Int32,
        K_gemm_tile: cutlass.Constexpr,
    ):
        """
        GPU device kernel entry point: dispatches to preferred or fallback cluster path.
        """
        # Determine if this CTA is launched in a preferred-shape cluster
        cbdim_x, cbdim_y, cbdim_z = cute.arch.block_in_cluster_dim()
        is_preferred_cluster = (
            cbdim_x == self.preferred_cluster_shape_mn[0]
            and cbdim_y == self.preferred_cluster_shape_mn[1]
            and cbdim_z == 1
        )

        # Megakernel: only one branch executes per launch.
        # smem_merge_branch_allocs=True at launch enables shared memory reuse between two paths.
        if is_preferred_cluster:
            self.cluster_specific_kernel(
                tiled_mma,
                tiled_mma_sfb,
                tma_atom_a_preferred,
                mA_mkl_preferred,
                tma_atom_b_preferred,
                mB_nkl_preferred,
                mSFA_mkl,
                tma_atom_sfb_preferred,
                mSFB_nkl_preferred,
                tma_atom_d,
                mD_mnl,
                tma_atom_c,
                mC_mnl,
                mSFD_mnl,
                alpha,
                norm_const_tensor,
                mBias_mnl,
                preferred_cluster_layout_vmnk,
                preferred_cluster_layout_sfb_vmnk,
                a_smem_layout_staged,
                b_smem_layout_staged,
                sfa_smem_layout_staged,
                sfb_smem_layout_staged,
                d_smem_layout_staged,
                epi_tile,
                preferred_tile_sched_params,
                epilogue_op,
                self.num_preferred_mcast_ctas_a + self.num_preferred_mcast_ctas_b - 1,
                self.is_preferred_a_mcast,
                self.is_preferred_b_mcast,
                self.preferred_cluster_shape_mn,
                conv_T,
                conv_R,
                conv_S,
                stride_d,
                stride_h,
                stride_w,
                dil_d,
                dil_h,
                dil_w,
                pad_d,
                pad_h,
                pad_w,
                input_D,
                input_H,
                input_W,
                output_Z,
                output_P,
                output_Q,
                input_N,
                input_C,
                K_gemm_tile,
            )
        else:
            self.cluster_specific_kernel(
                tiled_mma,
                tiled_mma_sfb,
                tma_atom_a_fallback,
                mA_mkl_fallback,
                tma_atom_b_fallback,
                mB_nkl_fallback,
                mSFA_mkl,
                tma_atom_sfb_fallback,
                mSFB_nkl_fallback,
                tma_atom_d,
                mD_mnl,
                tma_atom_c,
                mC_mnl,
                mSFD_mnl,
                alpha,
                norm_const_tensor,
                mBias_mnl,
                fallback_cluster_layout_vmnk,
                fallback_cluster_layout_sfb_vmnk,
                a_smem_layout_staged,
                b_smem_layout_staged,
                sfa_smem_layout_staged,
                sfb_smem_layout_staged,
                d_smem_layout_staged,
                epi_tile,
                fallback_tile_sched_params,
                epilogue_op,
                self.num_fallback_mcast_ctas_a + self.num_fallback_mcast_ctas_b - 1,
                self.is_fallback_a_mcast,
                self.is_fallback_b_mcast,
                self.fallback_cluster_shape_mn,
                conv_T,
                conv_R,
                conv_S,
                stride_d,
                stride_h,
                stride_w,
                dil_d,
                dil_h,
                dil_w,
                pad_d,
                pad_h,
                pad_w,
                input_D,
                input_H,
                input_W,
                output_Z,
                output_P,
                output_Q,
                input_N,
                input_C,
                K_gemm_tile,
            )

    @cute.jit()
    def cluster_specific_kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_d: cute.CopyAtom,
        mD_mnl: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: Optional[cute.Tensor],
        mSFD_mnl: Optional[cute.Tensor],
        alpha: cute.Tensor,
        norm_const_tensor: Optional[cute.Tensor],
        mBias_mnl: Optional[cute.Tensor],
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        d_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        num_tma_producer: int,
        effective_is_a_mcast: bool,
        effective_is_b_mcast: bool,
        cluster_shape: Tuple[int, int],
        # Conv parameters for SFA im2col coordinate mapping. conv_T/R/S are
        # runtime Int32 so one cubin serves any filter size; they only feed
        # register comparisons in the K-loop (S->R->T wrap), adding no IDIV.
        conv_T: cutlass.Int32,
        conv_R: cutlass.Int32,
        conv_S: cutlass.Int32,
        # pad/stride/dilation are runtime Int32 so one cubin serves any conv
        # config; they feed the per-tile base-coord arithmetic (z*stride - pad)
        # and the K-loop dilation step, swapping immediates for registers with
        # no new IDIV.
        stride_d: cutlass.Int32,
        stride_h: cutlass.Int32,
        stride_w: cutlass.Int32,
        dil_d: cutlass.Int32,
        dil_h: cutlass.Int32,
        dil_w: cutlass.Int32,
        pad_d: cutlass.Int32,
        pad_h: cutlass.Int32,
        pad_w: cutlass.Int32,
        # Coordinate geometry: runtime Int32 so the per-tile NDHW decomposition
        # (m_global -> n,z,p,q) lowers to real IDIV and one cubin serves any
        # N/D/H/W/Z/P/Q.
        input_D: cutlass.Int32,
        input_H: cutlass.Int32,
        input_W: cutlass.Int32,
        output_Z: cutlass.Int32,
        output_P: cutlass.Int32,
        output_Q: cutlass.Int32,
        input_N: cutlass.Int32,
        input_C: cutlass.Int32,
        K_gemm_tile: cutlass.Constexpr,
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
            # SFA has no TMA descriptor to prefetch: it is loaded via cp.async.
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_d)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        is_first_cta_in_cluster = cta_rank_in_cluster == 0
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Pipeline Init: merged pipeline shares one mbarrier between TMA (A+B+SFB)
        # and cp.async (SFA) producers, for any cluster shape.
        # tx_count covers A+B+SFB TMA bytes; arrive_count = 128 cp.async lanes +
        # 1 TMA arrive. The TMA arrive is produced by the leader's multicast
        # arrive_and_expect_tx and propagated to every peer barrier by hardware,
        # so peer CTAs must not issue an additional arrive.
        # For 2CTA (use_2cta_instrs), peer CTA's 128 cpasync threads also
        # remote-arrive on leader's sfa_full via DSMEM (shift-by-K pattern)
        # to eliminate the peer-sSFA race where leader MMA proceeded before
        # peer's cp.async landed. Leader's arrive_count must therefore be
        # bumped to 129 + 128 = 257.
        # num_tma_producer is provided as kernel parameter (per-cluster)
        sfa_cpasync_arrive_count = 257 if use_2cta_instrs else 129
        cpasynd_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, sfa_cpasync_arrive_count
        )
        merged_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producer)
        merged_tx_count = (
            self.num_tma_load_a_bytes + self.num_tma_load_b_bytes + self.num_tma_load_sfb_bytes
        )
        ab_sfa_pipeline = PipelineTmaCpAsyncUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            cpasynd_producer_group=cpasynd_producer_group,
            consumer_group=merged_consumer_group,
            tx_count=merged_tx_count,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Pipeline Init: Initialize acd_pipeline (barrier) and states
        acd_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilogue_warp_id) * (2 if use_2cta_instrs else 1)
        acd_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acd_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acd_pipeline_producer_group,
            consumer_group=acd_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Pipeline Init: residual TMA load. The dedicated residual warp issues
        # the TMA load (single-thread producer arrive) into sC; all epilogue
        # warps consume it (one release arrive per warp). No multicast: each CTA
        # loads its own output-tile residual.
        if cutlass.const_expr(mC_mnl is not None):
            c_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            c_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(self.epilogue_warp_id)
            )
            c_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.c_full_mbar_ptr.data_ptr(),
                num_stages=self.num_d_stage,
                producer_group=c_producer_group,
                consumer_group=c_consumer_group,
                tx_count=self.num_tma_load_c_bytes,
                defer_sync=True,
            )
        else:
            c_pipeline = None

        # Pipeline Init: Initialize clc_pipeline (CLC fetch async)
        # Consumers of CLC response: TMA(1) + cp.async SFA(4) + MMA(1) + Epilogue(4)
        # [+ residual(1) when beta != 0] per CTA, * cluster_size; plus sched(1) on
        # the first CTA only. The tile-coord query is broadcast to all CTAs in the
        # cluster. The residual term must gate on the SAME const as the residual
        # warp's CLC-consume block (mC_mnl is not None) or CLC producer/consumer
        # counts diverge and the fetch pipeline hangs.
        clc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cluster_size = cute.size(cluster_shape)
        num_residual_clc_warps = 1 if cutlass.const_expr(mC_mnl is not None) else 0
        num_clc_consumer_threads = 32 * (
            1
            + cluster_size
            * (
                1
                + len(self.cpasync_sfa_warp_id)
                + len(self.epilogue_warp_id)
                + 1
                + num_residual_clc_warps
            )
        )
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

        # Pipeline Init: Tensor memory alloc/dealloc barrier init
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=cluster_shape, is_relaxed=True)

        # CLC response buffer pointer + consumer state
        clc_response_ptr = storage.clc_response.data_ptr()
        clc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_clc_stage
        )

        #
        # Setup smem tensor A/B/SFA/SFB/D
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sD = storage.sD.get_tensor(d_smem_layout_staged.outer, swizzle=d_smem_layout_staged.inner)
        # (cta_tile_n,) bias row for the shared-memory-staged broadcast.
        if cutlass.const_expr(mBias_mnl is not None):
            sBias = storage.sBias.get_tensor(cute.make_layout(self.cta_tile_shape_mnk[1]))
        else:
            sBias = None
        # (EPI_TILE_M, EPI_TILE_N, STAGE) residual tile, same layout as sD.
        if cutlass.const_expr(mC_mnl is not None):
            sC = storage.sC.get_tensor(
                d_smem_layout_staged.outer, swizzle=d_smem_layout_staged.inner
            )
        else:
            sC = None
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        # (MMA, MMA_M, MMA_K, STAGE)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (MMA, MMA_N, MMA_K, STAGE)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        #
        # Compute multicast mask for A/B/SFA/SFB buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfb_full_mcast_mask = None

        if cutlass.const_expr(effective_is_a_mcast or effective_is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
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
        # (bN, bK, RestN, RestK, RestL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        # (bM, bN, RestM, RestN, RestL)
        gD_mnl = cute.local_tile(
            mD_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        # (bM, bN, RestM, RestN, RestL) - same MNL tiling as gD; the K-axis (N
        # of MMA) carries a stride-0 broadcast over self.sfd_vec_size groups.
        if cutlass.const_expr(self.gen_sfd):
            gSFD_mnl = cute.local_tile(
                mSFD_mnl,
                cute.slice_(self.mma_tiler, (None, None, 0)),
                (None, None, None),
            )
        else:
            gSFD_mnl = None
        # (bM, bN, RestM, RestN, RestL) - same MNL tiling as gD; the M axis
        # carries a stride-0 broadcast so every spatial output position shares
        # one per-output-channel (N) bias.
        if cutlass.const_expr(mBias_mnl is not None):
            gBias_mnl = cute.local_tile(
                mBias_mnl,
                cute.slice_(self.mma_tiler, (None, None, 0)),
                (None, None, None),
            )
        else:
            gBias_mnl = None
        # (bM, bN, RestM, RestN, RestL) - residual shares D's MNL tiling; it is a
        # full per-element tensor (no broadcast) loaded via TMA, so it is tiled
        # exactly like gD.
        if cutlass.const_expr(mC_mnl is not None):
            gC_mnl = cute.local_tile(
                mC_mnl,
                cute.slice_(self.mma_tiler, (None, None, 0)),
                (None, None, None),
            )
        else:
            gC_mnl = None

        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgD = thr_mma.partition_C(gD_mnl)
        # SFD: same partition shape as gD; broadcast strides preserved through partition.
        if cutlass.const_expr(self.gen_sfd):
            tCgSFD = thr_mma.partition_C(gSFD_mnl)
        else:
            tCgSFD = None
        # Bias: same partition_C chain as the accumulator; the M-axis stride-0
        # broadcast is preserved so each thread reads its own N-column bias.
        if cutlass.const_expr(mBias_mnl is not None):
            tCgBias = thr_mma.partition_C(gBias_mnl)
        else:
            tCgBias = None
        # Residual: same partition_C chain as the accumulator/D, so the residual
        # TMA load walks the output tile exactly like the D store.
        if cutlass.const_expr(mC_mnl is not None):
            tCgC = thr_mma.partition_C(gC_mnl)
        else:
            tCgC = None

        #
        # Partition global/shared tensor for TMA load B/SFB
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
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        # TMA load SFB partition_S/D
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N, STAGE)
        # For overlapping-accum this carries the strided 2-buffer layout so the
        # accumulator's second stage aliases the SFA/SFB columns.
        tCtAcc_fake = self._make_acc_fake_tensor(tiled_mma, self.mma_tiler)

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=cluster_shape)

        #
        # Specialized TMA load A/B/SFA/SFB warp
        #
        if warp_idx == self.tma_warp_id:
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
                tile_sched_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                clc_response_ptr,
            )
            work_tile = tile_sched.initial_work_tile_info()

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

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

                # Special-case for N=64: SFB scale-factor slicing to match Tensor Memory packing
                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2
                # ((atom_v, rest_v), RestK)
                tBgSFB_slice = tBgSFB[(None, slice_n, None, mma_tile_coord_mnl[2])]

                # Peek (try_wait) shared AB/SFB buffer empty
                ab_producer_state.reset_count()
                peek_ab_empty_status = ab_sfa_pipeline.producer_try_acquire(ab_producer_state)

                #
                # Tma load loop with increment_coord for im2col K traversal
                # Traversal order: S innermost -> R -> T -> C outermost
                #
                # A's RestK shape from im2col TMA is (C_tiles, S, R, T) where
                # C_tiles is leftmost. To traverse S->R->T->C, we build a
                # permuted traversal shape and remap coords when indexing.
                k_shape = cute.shape(tAgA_slice, mode=1)  # (C_tiles, S, R, T)
                k_C_tiles = cute.size(k_shape, mode=0)
                k_S = cute.size(k_shape, mode=1)
                k_R = cute.size(k_shape, mode=2)
                k_T = cute.size(k_shape, mode=3)
                # SFB boxes per filter position: one box is a whole SF K tile, so a
                # position holds that many fewer boxes than A/B tiles. Rounding up
                # matches the span the host padded SFB to.
                sfb_C_tiles = cute.ceil_div(k_C_tiles, self.ab_k_tiles_per_sf_k_tile)
                # Traversal shape in S->R->T->C order (colexicographic on this)
                trav_shape = (k_S, k_R, k_T, k_C_tiles)
                trav_coord = cute.repeat_like(0, trav_shape)

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # TMA producer path on the merged pipeline (leader does
                    # arrive_and_expect_tx, peer does nothing).
                    ab_sfa_pipeline.producer_acquire_tma(ab_producer_state, peek_ab_empty_status)

                    # Remap (s,r,t,c) -> (c,s,r,t) to match A/B RestK layout
                    ab_coord = (
                        trav_coord[3],  # C_tiles
                        trav_coord[0],  # S
                        trav_coord[1],  # R
                        trav_coord[2],  # T
                    )

                    # TMA load A (use multi-dim coord for im2col K indexing)
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, ab_coord)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_sfa_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=a_full_mcast_mask,
                    )
                    # TMA load B (multi-dim K layout, use same coord)
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, ab_coord)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_sfa_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=b_full_mcast_mask,
                    )
                    # TMA load SFB: compute flat K tile index from (c,s,r,t)
                    # SFB uses flat layout with C->S->R->T physical order. The
                    # channel coordinate is on the SF cadence, so the A/B tiles
                    # sharing one chunk name the same box and each stages it whole.
                    sfb_flat_k = (
                        ab_coord[0] // self.ab_k_tiles_per_sf_k_tile
                        + ab_coord[1] * sfb_C_tiles
                        + ab_coord[2] * k_S * sfb_C_tiles
                        + ab_coord[3] * k_R * k_S * sfb_C_tiles
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_slice[(None, sfb_flat_k)],
                        tBsSFB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_sfa_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfb_full_mcast_mask,
                    )

                    # Peek (try_wait) shared AB/SFB buffer empty for next iteration
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_sfa_pipeline.producer_try_acquire(
                            ab_producer_state
                        )

                    # Advance traversal coord in S->R->T->C order
                    trav_coord = cute.increment_coord(trav_coord, trav_shape)

                #
                # Advance to next tile (CLC consumer)
                #
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            #
            # Wait shared AB/SFB buffer empty
            #
            ab_sfa_pipeline.producer_tail(ab_producer_state)

        #
        # Specialized cp.async SFA warp
        #
        if warp_idx >= self.cpasync_sfa_warp_id[0] and warp_idx <= self.cpasync_sfa_warp_id[-1]:
            #
            # Setup SFA CPASYNC copy atom
            #
            sfa_atom_copy = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mSFA_mkl.element_type,
                num_bits_per_copy=32,
            )
            tidx_in_warpgroup = tidx % 128

            # SFA predicate: dynamically computed per cp.async for conv3x3
            sfa_predicate_tensor = cute.make_rmem_tensor(
                cute.make_layout((1,)),
                cutlass.Boolean,
            )

            # Identity M offset (no permutation) + CTA offset within cluster
            cta_m_offset = mma_tile_coord_v * self.cta_tile_shape_mnk[0]
            sfa_m_offset = (
                cta_m_offset
                + 8 * (tidx_in_warpgroup // 32)
                + 32 * ((tidx_in_warpgroup % 32) // 8)
                + (tidx_in_warpgroup % 8)
            )

            # Slice sSFA for this thread's M-row and sub-row
            tAsSFA = sSFA[
                (
                    (
                        (
                            (
                                8 * (tidx_in_warpgroup // 32) + (tidx_in_warpgroup % 8),
                                (tidx_in_warpgroup % 32) // 8,
                            ),
                            None,
                        ),
                        None,
                    ),
                    None,
                    None,
                    None,
                )
            ]

            # SFA global tensor: shape (MN, SF_K, L) with K contiguous
            # For conv3x3 with arbitrary C, each cp.async may load from a different
            # input element (different M address) depending on the filter position.
            # When C < K_gemm_tile, a single K tile spans multiple filter positions.

            # Precompute conv spatial constants
            K_gemm_tile = self.mma_tiler[2][0]
            ZPQ = output_Z * output_P * output_Q
            PQ = output_P * output_Q
            DHW = input_D * input_H * input_W
            HW = input_H * input_W
            CTA_M = self.cta_tile_shape_mnk[0]
            # SF blocks one SF K tile holds. The SF tile widens to a whole atom
            # when the A/B K tile is narrower, and every A/B tile sharing it stages
            # the whole thing.
            sf_k_per_ktile = self.sf_tile_k // self.sf_vec_size
            # C tiles per filter position, rounded up: a partial trailing tile still
            # consumes a k_tile, and its tail reads zero-filled A and B from the TMA.
            c_tiles_per_fpos = (input_C + K_gemm_tile - 1) // K_gemm_tile
            # SF blocks a pixel's row holds. The host rounds it up to a whole 4-block
            # atom and zero-fills the tail, so a group on the last one is in bounds.
            sfa_sf_k_extent = mSFA_mkl.shape[1]
            sfa_last_group = sfa_sf_k_extent - 4
            # Per-carry address deltas for the linear im2col walk: the unclamped
            # gmem M address decomposes as m_base + t*dil_d*HW + r*dil_h*W +
            # s*dil_w, so each carry level advances one running delta by a
            # precomputed step. Steps carry the SFA row stride, so the loop
            # adds element offsets directly.
            sfa_step_s = dil_w * sfa_sf_k_extent
            sfa_step_r = (dil_h * input_W - conv_S * dil_w) * sfa_sf_k_extent
            sfa_step_t = (dil_d * HW - conv_R * dil_h * input_W) * sfa_sf_k_extent

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
                tile_sched_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                clc_response_ptr,
            )
            work_tile = tile_sched.initial_work_tile_info()

            sfa_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Compute this thread's global M index (output pixel)
                # Use mma_tile_coord_mnl[0] (M tile index without V) * full MMA tile M
                # because sfa_m_offset already includes the V (CTA-within-pair) offset
                cta_m_tile_offset = (
                    mma_tile_coord_mnl[0] * CTA_M * cute.size(tiled_mma.thr_id.shape)
                )
                m_global = cta_m_tile_offset + sfa_m_offset

                # Decompose m_global -> (n, z, p, q) output coordinates
                n_idx = m_global // ZPQ
                zpq_rem = m_global % ZPQ
                z_idx = zpq_rem // PQ
                pq_rem = zpq_rem % PQ
                p_idx = pq_rem // output_Q
                q_idx = pq_rem % output_Q

                # Check if m_global is within valid output range
                m_valid = m_global < (input_N * ZPQ)

                # Initialize base spatial coordinates for filter pos (t=0, r=0, s=0)
                d_in_base = z_idx * stride_d - pad_d
                h_in_base = p_idx * stride_h - pad_h
                w_in_base = q_idx * stride_w - pad_w
                n_clamped = n_idx if m_valid else 0

                # Maintain (s,r,t,c) coords for S->R->T->C traversal order
                # matching TMA warp's coord_iter order
                sfa_s_idx = 0
                sfa_r_idx = 0
                sfa_t_idx = 0
                sfa_c_tile_idx = 0
                sfa_m_base = (
                    n_clamped * DHW + d_in_base * HW + h_in_base * input_W + w_in_base
                ) * sfa_sf_k_extent
                sfa_delta = cutlass.Int32(0)
                # Running unclamped coords for the predicate, advanced with the
                # same carry chain as the delta.
                sfa_d_in = d_in_base + 0
                sfa_h_in = h_in_base + 0
                sfa_w_in = w_in_base + 0

                # Peek (try_wait) SFA buffer empty
                sfa_producer_state.reset_count()
                peek_sfa_empty_status = cutlass.Boolean(1)
                if sfa_producer_state.count < k_tile_cnt:
                    peek_sfa_empty_status = ab_sfa_pipeline.producer_try_acquire(sfa_producer_state)

                #
                # Shift-by-K cross-CTA commit ringbuffer (K=2). Per iter,
                # commit the prior copies as a cp.async group and wait_group(K);
                # then remote-arrive on leader's sfa_full for the stage from
                # K iters ago. Guarantees that cp.async has landed globally
                # before leader MMA sees the barrier full.
                #
                sfa_shift_K = 2
                pending_a = cutlass.Int32(0)
                pending_b = cutlass.Int32(0)

                #
                # CPASYNC SFA load loop with coordinate-based im2col
                # Traversal order: S->R->T->C (matching TMA warp)
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Conditionally wait for SFA buffer empty
                    ab_sfa_pipeline.producer_acquire(sfa_producer_state, peek_sfa_empty_status)

                    cur_stage = sfa_producer_state.index

                    tAsSFA_ktile = tAsSFA[(None, None, None, None, cur_stage)]

                    # Predicate: m_valid AND spatial bounds check
                    sfa_pred_val = cutlass.Boolean(0)
                    if m_valid:
                        if sfa_d_in >= 0 and sfa_d_in < input_D:
                            if sfa_h_in >= 0 and sfa_h_in < input_H:
                                if sfa_w_in >= 0 and sfa_w_in < input_W:
                                    sfa_pred_val = cutlass.Boolean(1)

                    # A predicated-off row reads address 0, always in bounds;
                    # its predicate suppresses the actual copy.
                    sfa_row_off = (sfa_m_base + sfa_delta) if sfa_pred_val else 0

                    # SF K base offset for current C tile within this fpos. A/B
                    # tiles sharing one SF tile take the same base: they sit in one
                    # filter position, so one transfer covers them all.
                    sf_k_base = (sfa_c_tile_idx // self.ab_k_tiles_per_sf_k_tile) * sf_k_per_ktile

                    # 4-byte cp.async SFA -- all SI share same fpos.
                    # Each thread must cover every SI in [0, num_sfa_cpasync) exactly
                    # once so all SF-K groups of its row get written. XOR (q ^ i) is a
                    # complete cover only when num_sfa_cpasync is a power of 2; for a
                    # non-power-of-2 count (e.g. 3 when cta_tile_k == 3 * mma_inst_shape_k)
                    # it drops SI values on some quarter-warps, leaving SF-K groups
                    # unwritten. Additive rotation (q + i) % num is a complete cyclic
                    # cover for any count and keeps quarter-warps on distinct groups
                    # per timestep.
                    for i in range(self.num_sfa_cpasync):
                        SI = ((tidx_in_warpgroup % 32) // 8 + i) % self.num_sfa_cpasync

                        # One 4-byte cp.async moves 4 contiguous SF blocks (4 x 1-byte
                        # E4M3 = the 4-byte transfer). SI indexes which group of 4,
                        # so its SF-block base is SI * 4 and the copy shape is (4,).
                        sf_k_smem = SI * 4
                        local_sf_k = sf_k_base + (sf_k_smem % sf_k_per_ktile)
                        # A partial trailing tile reaches past the scale factors this
                        # pixel owns. Clamping the group start to the row's last whole
                        # atom keeps it in bounds and finite, which the MMA needs since
                        # it applies the scale before seeing the zero operand.
                        local_sf_k = local_sf_k if local_sf_k <= sfa_last_group else sfa_last_group

                        sfa_predicate_tensor[0] = sfa_pred_val

                        # Gmem: this fpos's row offset plus the local SF-K offset
                        tAgSFA_slice_ptr = mSFA_mkl.iterator + (sfa_row_off + local_sf_k)
                        tAgSFA_slice = cute.make_tensor(
                            tAgSFA_slice_ptr, layout=cute.make_layout((4,))
                        )

                        # Smem: write to swizzled position. Adjacent SF-block
                        # groups are 512 bytes apart in the SFA smem layout:
                        # CTA_M (128 rows) x 4 bytes per group = 512.
                        tAsSFA_slice_ptr = tAsSFA_ktile.iterator + 512 * SI
                        tAsSFA_slice = cute.make_tensor(tAsSFA_slice_ptr, cute.make_layout((4,)))

                        cute.copy_atom_call(
                            sfa_atom_copy,
                            tAgSFA_slice,
                            tAsSFA_slice,
                            pred=sfa_predicate_tensor,
                        )

                    # Increment in S->R->T->C order (S innermost, C outermost)
                    sfa_s_idx = sfa_s_idx + 1
                    sfa_delta = sfa_delta + sfa_step_s
                    sfa_w_in = sfa_w_in + dil_w
                    if sfa_s_idx >= conv_S:
                        sfa_s_idx = 0
                        sfa_w_in = w_in_base + 0
                        sfa_r_idx = sfa_r_idx + 1
                        sfa_delta = sfa_delta + sfa_step_r
                        sfa_h_in = sfa_h_in + dil_h
                        if sfa_r_idx >= conv_R:
                            sfa_r_idx = 0
                            sfa_h_in = h_in_base + 0
                            sfa_t_idx = sfa_t_idx + 1
                            sfa_delta = sfa_delta + sfa_step_t
                            sfa_d_in = sfa_d_in + dil_d
                            if sfa_t_idx >= conv_T:
                                sfa_t_idx = 0
                                sfa_d_in = d_in_base + 0
                                sfa_delta = 0
                                sfa_c_tile_idx = sfa_c_tile_idx + 1
                                if sfa_c_tile_idx >= c_tiles_per_fpos:
                                    sfa_c_tile_idx = 0

                    # Commit this iter's copies as one cp.async group and
                    # gate on <=K inflight. After wait_group(K), the cp.async
                    # for the stage from K iters ago is guaranteed complete
                    # on every CTA in the cluster.
                    ab_sfa_pipeline.producer_cp_async_commit()
                    ab_sfa_pipeline.producer_cp_async_wait(sfa_shift_K)

                    # Shift-by-K: arrive on stage from K iters ago.
                    if k_tile >= sfa_shift_K:
                        ab_sfa_pipeline.producer_arrive_remote(pending_a)

                    # Advance ringbuffer (oldest <- 2nd oldest <- current).
                    pending_a = pending_b
                    pending_b = cur_stage

                    # Peek (try_wait) SFA buffer empty for next iteration
                    sfa_producer_state.advance()
                    peek_sfa_empty_status = cutlass.Boolean(1)
                    if sfa_producer_state.count < k_tile_cnt:
                        peek_sfa_empty_status = ab_sfa_pipeline.producer_try_acquire(
                            sfa_producer_state
                        )

                #
                # Shift-by-K tail: drain all inflight cp.async, then arrive on
                # the K stages whose arrives were deferred.
                #
                ab_sfa_pipeline.producer_cp_async_wait(0)
                if k_tile_cnt >= sfa_shift_K:
                    ab_sfa_pipeline.producer_arrive_remote(pending_a)
                if k_tile_cnt >= 1:
                    ab_sfa_pipeline.producer_arrive_remote(pending_b)

                #
                # Advance to next tile (CLC consumer)
                #
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            #
            # Wait SFA buffer empty
            #
            ab_sfa_pipeline.producer_tail(sfa_producer_state)

        #
        # Specialized residual G2S warp (only spawned when beta != 0).
        #
        # The residual load owns a dedicated warp so it no longer serializes
        # behind A/B/SFB issue on the TMA warp. Producer and epilogue consumer
        # still sit on physically separate warps (required: a same-warp
        # producer/consumer pair on the residual barrier self-deadlocks the
        # epilogue). It runs its own persistent tile-scheduler loop and consumes
        # one CLC response per tile, so it is counted in num_clc_consumer_threads
        # under the SAME const gate (mC_mnl is not None).
        #
        if warp_idx == self.residual_warp_id:
            if cutlass.const_expr(mC_mnl is not None):
                tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
                    tile_sched_params,
                    cute.arch.block_idx(),
                    cute.arch.grid_dim(),
                    clc_response_ptr,
                )
                work_tile = tile_sched.initial_work_tile_info()

                # The gmem/smem TMA partition is thread-invariant, so recompute
                # the exact handles the epilogue consumer expects.
                (
                    tma_atom_c,
                    bGS_sC,
                    bGS_gC_partitioned,
                ) = self.epilog_gmem_copy_and_partition(tidx, tma_atom_c, tCgC, epi_tile, sC)
                c_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_d_stage
                )
                # Overlapping-accum walks output subtiles back-to-front on the
                # phase-0 tile; mirror the epilogue's acc-consumer phase with a
                # shadow state so the producer stages subtiles in the same order
                # the consumer reads them.
                if cutlass.const_expr(self.overlapping_accum):
                    c_acc_shadow_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, self.num_acc_stage
                    )

                while work_tile.is_valid_tile:
                    cur_tile_coord = work_tile.tile_idx
                    mma_tile_coord_mnl = (
                        cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                        cur_tile_coord[1],
                        cur_tile_coord[2],
                    )

                    #
                    # Issue this output tile's residual TMA loads (gmem -> smem),
                    # one per output N-subtile, walking subtiles in lockstep with
                    # the epilogue consumer. Whole-warp producer arrive (no
                    # elect_one): PipelineTmaAsync's single-thread producer group
                    # self-elects the signaling lane.
                    #
                    bGS_gC = bGS_gC_partitioned[
                        (
                            None,
                            None,
                            None,
                            *mma_tile_coord_mnl,
                        )
                    ]
                    bGS_gC = cute.group_modes(bGS_gC, 1, cute.rank(bGS_gC))
                    subtile_cnt = cute.size(bGS_gC.shape, mode=[1])
                    if cutlass.const_expr(self.overlapping_accum):
                        reverse_subtile = c_acc_shadow_state.phase == 0
                    for subtile_idx in cutlass.range(subtile_cnt):
                        real_subtile_idx = subtile_idx
                        if cutlass.const_expr(self.overlapping_accum):
                            if reverse_subtile:
                                real_subtile_idx = (
                                    self.cta_tile_shape_mnk[1] // self.epi_tile_n - 1 - subtile_idx
                                )
                        # The ring slot comes from the pipeline state, the same
                        # place the barrier below and the epilogue consumer take
                        # theirs, so the ring holds at any stage count.
                        c_pipeline.producer_acquire(c_producer_state)
                        cute.copy(
                            tma_atom_c,
                            bGS_gC[(None, real_subtile_idx)],
                            bGS_sC[(None, c_producer_state.index)],
                            tma_bar_ptr=c_pipeline.producer_get_barrier(c_producer_state),
                        )
                        c_producer_state.advance()
                    if cutlass.const_expr(self.overlapping_accum):
                        c_acc_shadow_state.advance()

                    #
                    # Advance to next tile (CLC consumer)
                    #
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()

                #
                # Drain the residual TMA load pipeline before exiting.
                #
                c_pipeline.producer_tail(c_producer_state)

        #
        # Specialized scheduler warp (drives CLC fetch, only first CTA in cluster)
        #
        if warp_idx == self.sched_warp_id and is_first_cta_in_cluster:
            clc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.ProducerConsumer, self.num_clc_stage
            )

            tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
                tile_sched_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                clc_response_ptr,
            )
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
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
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator/SFA/SFB tensor
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # Make accumulator tmem tensor
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Make SFA tmem tensor
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base),
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_M, MMA_K)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.flat_mma_tiler_sf,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # Make SFB tmem tensor
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr
                + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
                + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_N, MMA_K)
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.flat_mma_tiler_sf,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            #
            # Partition for S2T copy of SFA/SFB
            #
            (
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t,
                tCtSFA_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            (
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t,
                tCtSFB_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
                tile_sched_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                clc_response_ptr,
            )
            work_tile = tile_sched.initial_work_tile_info()

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            sfa_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            # Filter positions one channel tile sweeps. The K loop walks S->R->T->C
            # with C outermost, so the chunk part below has to come from the channel
            # tile: the running k_tile count keeps advancing across positions.
            if cutlass.const_expr(self.ab_k_tiles_per_sf_k_tile > 1):
                sf_fpos_per_c_tile = conv_T * conv_R * conv_S

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Channel tile and its filter-position counter, reset per work tile.
                if cutlass.const_expr(self.ab_k_tiles_per_sf_k_tile > 1):
                    sf_c_tile_idx = cutlass.Int32(0)
                    sf_fpos_idx = cutlass.Int32(0)

                # Set tensor memory buffer for current tile
                # (MMA, MMA_M, MMA_N)
                # Overlapping-accum squeezes two acc buffers into 512-col TMEM by
                # aliasing the second buffer onto the SFA/SFB columns. The producer
                # writes the buffer the consumer is NOT currently draining, which is
                # acc_producer_state.phase ^ 1 (producer phase inits to 1, consumer to
                # 0, so tile 0 picks stage 0 to match the consumer).
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_producer_state.phase ^ 1
                else:
                    acc_stage_index = acc_producer_state.index
                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                # Peek (try_wait) shared AB/SFB + SFA buffer full for k_tile = 0
                ab_consumer_state.reset_count()
                sfa_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_sfa_pipeline.consumer_try_wait(ab_consumer_state)
                # Merged pipeline: sfa shares ab mbarrier, no separate try_wait needed.

                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acd_pipeline.producer_acquire(acc_producer_state)

                # Special-case for N=192/N=64: shift the SFB Tensor Memory pointer to match its packing
                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                    # If this is an ODD tile, shift the Tensor Memory start address for cta_tile_shape_n=192 case by two words (ignores first 64 columns of SFB)
                    offset = (
                        cutlass.Int32(2) if mma_tile_coord_mnl[1] % 2 == 1 else cutlass.Int32(0)
                    )
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
                        + tcgen05.find_tmem_tensor_col_offset(tCtSFA)
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                elif cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    # Move in increments of 64 columns of SFB
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
                        + tcgen05.find_tmem_tensor_col_offset(tCtSFA)
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)

                #
                # Reset the ACCUMULATE field for each tile
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                #
                # Mma mainloop
                #
                for k_tile in cutlass.range(k_tile_cnt, unroll=1):
                    if cutlass.const_expr(self.ab_k_tiles_per_sf_k_tile > 1):
                        sf_kblock_base = (
                            sf_c_tile_idx % self.ab_k_tiles_per_sf_k_tile
                        ) * self.mma_inst_tile_k
                    else:
                        sf_kblock_base = 0

                    if is_leader_cta:
                        # Merged pipeline: single consumer_wait covers both TMA and cp.async producers.
                        ab_sfa_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)

                        #  Copy SFA/SFB from smem to tmem
                        sfa_s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            sfa_consumer_state.index,
                        )
                        sfb_s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            ab_consumer_state.index,
                        )
                        tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[sfa_s2t_stage_coord]
                        tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[sfb_s2t_stage_coord]
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t_staged,
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t_staged,
                            tCtSFB_compact_s2t,
                        )

                        # Block-scaled MMA: acc += (A * SFA) * (B * SFB)
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
                            )

                            # Set SFA/SFB tensor to tiled_mma. The K blocks this
                            # tile owns start at its own part of the shared chunk
                            # (offset 0 for NVFP4, whose ratio is 1).
                            sf_kblock_coord = (
                                None,
                                None,
                                sf_kblock_base + kblock_idx,
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFA,
                                tCtSFA[sf_kblock_coord].iterator,
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB,
                                tCtSFB_mma[sf_kblock_coord].iterator,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )

                            # Enable accumulate on tCtAcc after first kblock
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Merged pipeline: single consumer_release covers both TMA and cp.async producers.
                        ab_sfa_pipeline.consumer_release(ab_consumer_state)

                    # Advance the channel tile once the whole filter has been
                    # swept, mirroring the S->R->T->C walk of the load warps.
                    if cutlass.const_expr(self.ab_k_tiles_per_sf_k_tile > 1):
                        sf_fpos_idx = sf_fpos_idx + 1
                        if sf_fpos_idx >= sf_fpos_per_c_tile:
                            sf_fpos_idx = cutlass.Int32(0)
                            sf_c_tile_idx = sf_c_tile_idx + 1

                    # Peek (try_wait) shared AB/SFB + SFA buffer full for k_tile+1
                    ab_consumer_state.advance()
                    sfa_consumer_state.advance()

                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_sfa_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )
                    # Merged pipeline: sfa shares ab mbarrier, try_wait above handles both.

                #
                # Async arrive accumulator buffer full
                #
                if is_leader_cta:
                    acd_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                #
                # Advance to next tile (CLC consumer)
                #
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            #
            # Wait for accumulator buffer empty
            #
            acd_pipeline.producer_tail(acc_producer_state)

        #
        # Specialized epilogue warps
        #
        if warp_idx <= self.epilogue_warp_id[-1]:
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
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            #
            # Partition for epilogue
            #
            epi_tidx = tidx
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, tCgD, epi_tile, use_2cta_instrs
            )

            tTR_rD = cute.make_rmem_tensor(tTR_rAcc.shape, self.d_dtype)
            tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rD, epi_tidx, sD
            )
            (
                tma_atom_d,
                bSG_sD,
                bSG_gD_partitioned,
            ) = self.epilog_gmem_copy_and_partition(epi_tidx, tma_atom_d, tCgD, epi_tile, sD)

            # Residual smem->reg partition: mirrors the accumulator's T2R
            # fragment so each thread's residual register tile lines up with its
            # acc tile. The gmem->smem TMA producer runs on the TMA warp; the
            # epilogue only consumes the staged smem tile here.
            if cutlass.const_expr(mC_mnl is not None):
                tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.d_dtype)
                (
                    tiled_copy_s2r_c,
                    tSR_rC,
                    tSR_sC,
                ) = self.epilog_smem_load_copy_and_partition(tiled_copy_t2r, tTR_rC, epi_tidx, sC)
            else:
                tTR_rC = None
                tiled_copy_s2r_c = None
                tSR_rC = None
                tSR_sC = None

            # SFD partition: same epi-tile/T2R structure as gD, but written
            # directly to GMEM via STG (no SMEM/TMA path).
            if cutlass.const_expr(self.gen_sfd):
                thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
                # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
                gSFD_epi = cute.flat_divide(
                    tCgSFD[((None, None), 0, 0, None, None, None)], epi_tile
                )
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
                tTR_gSFD_partitioned = thr_copy_t2r.partition_D(gSFD_epi)
                # The SFD store is a plain STG whose partition rounds M up to a
                # tile multiple, so the last m-tile has overhang rows with no
                # backing storage. Build an identity-coordinate mirror through
                # the D path (which has no stride-0 broadcast mode), partitioned
                # identically, so the store site can recover each thread's M
                # coordinate and predicate the STG against the real M extent.
                sfd_ceil_m, sfd_ceil_n, _ = cute.ceil_div(
                    mD_mnl.shape, (self.mma_tiler[0], self.mma_tiler[1], 1)
                )
                mcSFD = cute.make_identity_tensor(
                    (
                        cute.size(sfd_ceil_m) * self.mma_tiler[0],
                        cute.size(sfd_ceil_n) * self.mma_tiler[1],
                        1,
                    )
                )
                gcSFD = cute.local_tile(
                    mcSFD,
                    cute.slice_(self.mma_tiler, (None, None, 0)),
                    (None, None, None),
                )
                tCgcSFD = thr_mma.partition_C(gcSFD)
                gcSFD_epi = cute.flat_divide(
                    tCgcSFD[((None, None), 0, 0, None, None, None)], epi_tile
                )
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
                tTR_cSFD_partitioned = thr_copy_t2r.partition_D(gcSFD_epi)
            else:
                tTR_gSFD_partitioned = None
                tTR_cSFD_partitioned = None

            # Bias partition: same epi-tile/T2R structure as the accumulator, so
            # each epilogue thread's bias register fragment lines up with its acc
            # fragment. The M-axis stride-0 broadcast (built into mBias_mnl) means
            # each thread only needs its own N-column bias values.
            if cutlass.const_expr(mBias_mnl is not None):
                thr_copy_t2r_bias = tiled_copy_t2r.get_slice(epi_tidx)
                # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
                gBias_epi = cute.flat_divide(
                    tCgBias[((None, None), 0, 0, None, None, None)], epi_tile
                )
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
                tTR_gBias_partitioned = thr_copy_t2r_bias.partition_D(gBias_epi)
                # smem-load atom: each thread reads its own N columns back from the
                # CTA-staged sBias row. The per-tile read view is built inside
                # the loop off the sliced gmem layout (whose N modes already
                # address [0, cta_tile_n) with the RestN tile-selector removed).
                simt_atom_bias = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), mBias_mnl.element_type
                )
                # cp.async transfers at least 32 bits, so each lane moves a
                # 32-bit-wide vector of bias elements (2 bf16/fp16, 1 fp32).
                bias_elems_per_copy = 32 // mBias_mnl.element_type.width
                bias_g2s_atom = cute.make_copy_atom(
                    cute.nvgpu.cpasync.CopyG2SOp(),
                    mBias_mnl.element_type,
                    num_bits_per_copy=32,
                )
            else:
                tTR_gBias_partitioned = None
                simt_atom_bias = None
                bias_g2s_atom = None
                bias_elems_per_copy = None

            # Read the device-resident scale once per epilogue warp. Dynamic
            # activation quantization produces this value on the GPU, so keeping
            # it as a tensor avoids a host synchronization before every conv.
            alpha_scalar = cutlass.Float32(alpha[0])
            if cutlass.const_expr(self.gen_sfd):
                norm_const_scalar = cutlass.Float32(norm_const_tensor[0])
            else:
                norm_const_scalar = cutlass.Float32(1.0)

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
                tile_sched_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                clc_response_ptr,
            )
            work_tile = tile_sched.initial_work_tile_info()

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            # Pipeline Init: Threads/warps participating in tma store pipeline
            d_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilogue_warp_id),
            )
            d_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_d_stage,
                producer_group=d_producer_group,
            )

            # The residual producer state lives on the dedicated residual warp;
            # the epilogue only tracks the consumer side of the residual load
            # pipeline.
            if cutlass.const_expr(mC_mnl is not None):
                c_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_d_stage
                )
            else:
                c_consumer_state = None

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
                # ((ATOM_V, REST_V), EPI_M, EPI_N)
                bSG_gD = bSG_gD_partitioned[
                    (
                        None,
                        None,
                        None,
                        *mma_tile_coord_mnl,
                    )
                ]
                if cutlass.const_expr(self.gen_sfd):
                    # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
                    tTR_gSFD = tTR_gSFD_partitioned[
                        (
                            None,
                            None,
                            None,
                            None,
                            None,
                            *mma_tile_coord_mnl,
                        )
                    ]
                    # Identity-coordinate mirror, sliced identically, for the
                    # STG M-bound predicate at the SFD store site.
                    tTR_cSFD = tTR_cSFD_partitioned[
                        (
                            None,
                            None,
                            None,
                            None,
                            None,
                            *mma_tile_coord_mnl,
                        )
                    ]
                if cutlass.const_expr(mBias_mnl is not None):
                    # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
                    tTR_gBias = tTR_gBias_partitioned[
                        (
                            None,
                            None,
                            None,
                            None,
                            None,
                            *mma_tile_coord_mnl,
                        )
                    ]
                    # The CTA cp.async's this tile's contiguous cta_tile_n bias
                    # values (K stride 1) into sBias, then syncs so every
                    # epilogue thread can load its columns back from smem. One value per
                    # output channel is fetched once and broadcast to all M rows.
                    cta_n = self.cta_tile_shape_mnk[1]
                    n_base = mma_tile_coord_mnl[1] * cta_n
                    # Each lane cp.async's a contiguous 32-bit vector
                    # (bias_elems_per_copy elements) of this tile's bias row.
                    # Layout (elems, lanes) with column-major stride so lane t
                    # owns the contiguous block [t*elems : (t+1)*elems); the K
                    # (output-channel) axis of mBias_mnl has stride 1, so the
                    # segment is contiguous. n_active lanes cover cta_n.
                    n_active = cta_n // bias_elems_per_copy
                    row_layout = cute.make_layout(
                        (bias_elems_per_copy, n_active),
                        stride=(1, bias_elems_per_copy),
                    )
                    # cp.async needs 32-bit source/dest alignment; the tile
                    # base (n_base is a multiple of cta_tile_n) keeps both on a
                    # 4-byte boundary, so re-annotate the pointers to satisfy
                    # the 32-bit atom.
                    gBias_row = cute.make_tensor(
                        (mBias_mnl.iterator + n_base).align(min_align=4),
                        row_layout,
                    )
                    sBias_tiled = cute.make_tensor(sBias.iterator.align(min_align=4), row_layout)
                    # Predicated cp.async: a CTA N-tile rounds up to cta_tile_n,
                    # but K (= output channels = GEMM-N) need not divide it, so
                    # the tail lanes address bias columns n >= K that have
                    # no backing storage. Guard each lane's contiguous vector on
                    # its base channel: in-bounds lanes cp.async from gmem,
                    # out-of-bounds lanes zero-fill sBias (cp.async writes 0 on a
                    # false predicate). K % 8 == 0 with a 32-bit (2 bf16) vector
                    # means every vector lies wholly in- or out-of-bounds, so one
                    # predicate per lane is exact. The zero tail is only read back
                    # for overhang output that the TMA store clamps away anyway.
                    if epi_tidx < n_active:
                        bias_pred = cute.make_rmem_tensor(cute.make_layout((1,)), cutlass.Boolean)
                        bias_pred[0] = cutlass.Boolean(
                            n_base + epi_tidx * bias_elems_per_copy < mD_mnl.shape[1]
                        )
                        cute.copy_atom_call(
                            bias_g2s_atom,
                            gBias_row[(None, epi_tidx)],
                            sBias_tiled[(None, epi_tidx)],
                            pred=bias_pred,
                        )
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    self.epilog_sync_barrier.arrive_and_wait()
                    # Read view: the sliced gmem fragment layout already maps
                    # this thread's N columns into [0, cta_tile_n) with all M
                    # modes stride-0 and the tile-selecting RestN removed, so
                    # the same layout over sBias makes each per-subtile read an
                    # smem load of exactly this thread's columns. make_tensor flattens
                    # the trailing (EPI_M, EPI_N) hierarchy, so regroup it to
                    # match tTR_gBias's rank-4 (..., subtile) profile.
                    tTR_sBias = cute.group_modes(
                        cute.make_tensor(sBias.iterator, tTR_gBias.layout), 3, 5
                    )

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                # Overlapping-accum: the consumer reads the buffer indexed by its
                # phase (0/1). Stage-1 acc is strided into TMEM so that its high
                # subtiles alias the SFA/SFB columns; to drain those shared columns
                # first (before the next tile's MMA overwrites them) the epilogue
                # walks subtiles in REVERSE when phase==0.
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_consumer_state.phase
                    reverse_subtile = acc_stage_index == 0
                else:
                    acc_stage_index = acc_consumer_state.index
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]

                #
                # Wait for accumulator buffer full
                #
                acd_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gD = cute.group_modes(bSG_gD, 1, cute.rank(bSG_gD))
                if cutlass.const_expr(self.gen_sfd):
                    # ((T2R, T2R_M, T2R_N), SUBTILE_CNT)
                    tTR_gSFD = cute.group_modes(tTR_gSFD, 3, cute.rank(tTR_gSFD))
                    tTR_cSFD = cute.group_modes(tTR_cSFD, 3, cute.rank(tTR_cSFD))

                #
                # Store accumulator to global memory in subtiles
                #
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
                for subtile_idx in cutlass.range(subtile_cnt):
                    # Map the loop counter to the true output N-subtile. Under
                    # overlapping-accum with reverse_subtile we walk the output
                    # subtiles back-to-front so the SFA/SFB-aliased columns are
                    # drained (and released) before the next tile's MMA reuses
                    # them. real_subtile_idx addresses every actual output
                    # position (TMEM acc, gmem D, gmem SFD); the raw subtile_idx
                    # stays a sequential counter (SMEM ring, release).
                    real_subtile_idx = subtile_idx
                    if cutlass.const_expr(self.overlapping_accum):
                        if reverse_subtile:
                            real_subtile_idx = (
                                self.cta_tile_shape_mnk[1] // self.epi_tile_n - 1 - subtile_idx
                            )
                    #
                    # Load accumulator from tensor memory buffer to register
                    #
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    #
                    # Async arrive accumulator buffer empty earlier when
                    # overlapping_accum is enabled. Trigger keyed on the raw loop
                    # counter so it fires exactly once after the shared columns
                    # (drained first under reverse) have been read out.
                    #
                    if cutlass.const_expr(self.overlapping_accum):
                        if subtile_idx == self.iter_acc_early_release_in_epilogue:
                            cute.arch.fence_view_async_tmem_load()
                            with cute.arch.elect_one():
                                acd_pipeline.consumer_release(acc_consumer_state)
                            acc_consumer_state.advance()

                    # Apply per-tensor alpha.
                    tTR_rAcc.store(tTR_rAcc.load() * alpha_scalar)

                    # Add per-output-channel bias in FP32 (D = alpha*acc + bias).
                    # Load this subtile's bias from the CTA-staged sBias row into a
                    # register fragment shaped like the acc fragment, then add.
                    # The M-axis stride-0 broadcast means each thread reads only
                    # its own N-column bias.
                    if cutlass.const_expr(mBias_mnl is not None):
                        tTR_rBias = cute.make_rmem_tensor(tTR_rAcc.shape, mBias_mnl.element_type)
                        cute.copy(
                            simt_atom_bias,
                            tTR_sBias[(None, None, None, real_subtile_idx)],
                            tTR_rBias,
                        )
                        tTR_rAcc.store(tTR_rAcc.load() + tTR_rBias.load().to(self.acc_dtype))

                    # Add the per-element residual in FP32 (D = alpha*acc + bias
                    # + beta*residual). Wait for this subtile's TMA load to land
                    # in smem, copy it to registers aligned with the acc
                    # fragment, then accumulate. The residual shares the output
                    # dtype and is upconverted to the accumulator type; beta is a
                    # compile-time constant folded into the scaled add.
                    if cutlass.const_expr(mC_mnl is not None):
                        c_pipeline.consumer_wait(c_consumer_state)
                        cute.copy(
                            tiled_copy_s2r_c,
                            tSR_sC[(None, None, None, c_consumer_state.index)],
                            tSR_rC,
                        )
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )
                        # consumer_release self-elects its signaling threads
                        # (is_signaling_thread by tidx); all epilogue threads
                        # call it directly, no explicit elect_one.
                        c_pipeline.consumer_release(c_consumer_state)
                        c_consumer_state.advance()
                        # tSR_rC is the S2R-partition view of tTR_rC
                        # (shared storage); read back through the acc-shaped view
                        # so the add lines up with tTR_rAcc.
                        tTR_rAcc.store(
                            tTR_rAcc.load() + self.beta * tTR_rC.load().to(self.acc_dtype)
                        )

                    #
                    # SFD generation (NVFP4 output only): per sfd_vec_size
                    # abs-max -> pvscale_f32 -> cast to sf_dtype (E4M3) -> STG.
                    # Then rescale acc by norm_const_tensor * rcp(qpvscale_f32), where
                    # qpvscale_f32 is the SFD value read back after the E4M3 cast
                    # so D_quant * SFD stays self-consistent, with NaN/inf clamp
                    # via fmin.
                    #
                    if cutlass.const_expr(self.gen_sfd):
                        # Slice gSFD for this subtile and collapse stride-0 broadcast.
                        gSFD_subtile = tTR_gSFD[(None, None, None, real_subtile_idx)]
                        t2r_gSFD = cute.filter_zeros(gSFD_subtile)
                        # The plain SFD STG has no TMA extent clamp, so it must be
                        # predicated on BOTH the real M and N (channel) extents:
                        # the cute partition rounds the cta tile up to a full
                        # mma_tiler multiple in both axes, leaving overhang rows
                        # (m >= M) and, when cta_n does not divide K, an overhang
                        # N-subtile (n >= K). An unguarded N-overhang store writes
                        # zeros at a gmem offset that -- since the mn-row stride
                        # equals the global sf_k count -- folds onto the next row's
                        # low sf_k columns, corrupting valid scale factors. The
                        # fragment lies on one (m, n_base), so the first element's
                        # coordinates gate the whole store. Index via
                        # real_subtile_idx: under overlapping-accum reverse_subtile
                        # the raw loop counter and real output subtile are mirror
                        # images, so a raw-indexed predicate guards the wrong one.
                        cSFD_subtile = tTR_cSFD[(None, None, None, real_subtile_idx)]
                        sfd_m_in_bounds = cute.elem_less(cSFD_subtile[0][0], mD_mnl.shape[0])
                        sfd_n_in_bounds = cute.elem_less(cSFD_subtile[0][1], mD_mnl.shape[1])
                        sfd_in_bounds = sfd_m_in_bounds and sfd_n_in_bounds
                        # Partition tTR_rAcc into vec_size groups along the contig (K) mode.
                        sfgen_rAcc = cute.logical_divide(tTR_rAcc, self.sfd_vec_size)
                        n_sf = cute.size[1](sfgen_rAcc)
                        rSFD = cute.make_rmem_tensor((1, n_sf), dtype=self.sfd_dtype)
                        # Reciprocal of the output dtype's full-scale max (M_D).
                        rcp_max = cutlass.Float32(1.0 / self.M_D)
                        fp32_max = cutlass.Float32(3.40282346638528859812e38)
                        # Single-pass: compute amax, quantize SFD, and rescale acc
                        # in the same loop. The acc rescale reads the SFD value
                        # back after the E4M3 cast (qpvscale_f32), so the dequant
                        # identity D_quant * SFD reproduces the pre-quant
                        # accumulator to within the FP4 data round-off.
                        for i_sf in cutlass.range(n_sf, unroll_full=True):
                            sfgen_slice = sfgen_rAcc[(None, i_sf)]
                            red_ssa = sfgen_slice.load()
                            red_abs_ssa = cute.math.absf(red_ssa)
                            amax = cutlass.Float32(
                                red_abs_ssa.reduce(
                                    cute.ReductionOp.MAX,
                                    cutlass.Float32(0.0),
                                    0,
                                )
                            )
                            pvscale_f32 = amax * rcp_max * norm_const_scalar
                            rSFD[(0, i_sf)] = pvscale_f32.to(self.sfd_dtype)
                            qpvscale_f32 = cutlass.Float32(rSFD[(0, i_sf)].to(cutlass.Float32))
                            acc_scale = norm_const_scalar * cute.arch.rcp_approx(qpvscale_f32)
                            acc_scale = cutlass.Float32(
                                nvvm.fmin(
                                    acc_scale.ir_value(),
                                    fp32_max.ir_value(),
                                    nan=True,
                                )
                            )
                            sfgen_slice.store(sfgen_slice.load() * acc_scale)
                        # Store SFD to gmem (predicated on both the M and N
                        # bounds; the last m-tile and partial-N overhangs have no
                        # backing SFD storage). Scalar STG per SF element with a
                        # static (unrolled) index: the gmem address is base +
                        # i * runtime_stride, so the tensor's global extent may be
                        # a runtime Int -- only the per-thread element count is
                        # static. (autovec_copy would require a fully static
                        # destination and cannot take a runtime-extent tensor.)
                        if sfd_in_bounds:
                            for i_st in cutlass.range(n_sf, unroll_full=True):
                                t2r_gSFD[i_st] = rSFD[(0, i_st)]

                    #
                    # Convert to D type
                    #
                    # epilogue_op is applied after the cast, but SFD was computed
                    # above from the pre-op accumulator -- consistent only for an
                    # identity op; a non-trivial op would desync D from SFD.
                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                    acc_vec = epilogue_op(acc_vec.to(self.d_dtype))
                    tRS_rD.store(acc_vec)

                    #
                    # Store D to shared memory
                    #
                    d_buffer = (num_prev_subtiles + subtile_idx) % self.num_d_stage
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rD,
                        tRS_sD[(None, None, None, d_buffer)],
                    )
                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    self.epilog_sync_barrier.arrive_and_wait()

                    #
                    # TMA store D to global memory
                    #
                    if warp_idx == self.epilogue_warp_id[0]:
                        cute.copy(
                            tma_atom_d,
                            bSG_sD[(None, d_buffer)],
                            bSG_gD[(None, real_subtile_idx)],
                        )
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        d_pipeline.producer_commit()
                        d_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()

                #
                # Async arrive accumulator buffer empty. Under overlapping-accum
                # the release already happened mid-loop (early release above), so
                # only the non-overlapping path releases here.
                #
                if cutlass.const_expr(not self.overlapping_accum):
                    with cute.arch.elect_one():
                        acd_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                #
                # Advance to next tile (CLC consumer)
                #
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)
            #
            # Wait for D store complete
            #
            d_pipeline.producer_tail()

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to partition smem memory (source) and tensor memory (destination).

        :param sSF: The scale factor tensor in smem
        :type sSF: cute.Tensor
        :param tSF: The scale factor tensor in tmem
        :type tSF: cute.Tensor

        :return: A tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
            - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
            - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
            - tSF_compact_s2t: The partitioned scale factor tensor in tmem
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)

        # Make S2T CopyAtom and tiledCopy
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gD_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array (destination).

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
        :param gD_mnl: The global tensor D
        :type gD_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled
        :type use_2cta_instrs: bool

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc: The accumulated tensor in register used to hold t2r results
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.d_layout,
            self.d_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gD_mnl_epi = cute.flat_divide(gD_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
        tTR_gD = thr_copy_t2r.partition_D(gD_mnl_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gD[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rD: cute.Tensor,
        tidx: cutlass.Int32,
        sD: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register array (source) and shared memory (destination).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rD: The partitioned accumulator tensor
        :type tTR_rD: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param sD: The shared memory tensor to be copied and partitioned
        :type sD: cute.Tensor
        :type sepi: cute.Tensor

        :return: A tuple containing (tiled_copy_r2s, tRS_rD, tRS_sD) where:
            - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
            - tRS_rD: The partitioned tensor D (register source)
            - tRS_sD: The partitioned tensor D (smem destination)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.d_layout, self.d_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sD)
        # (R2S, R2S_M, R2S_N)
        tRS_rD = tiled_copy_r2s.retile(tTR_rD)
        return tiled_copy_r2s, tRS_rD, tRS_sD

    def epilog_smem_load_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory load, then use it to partition register
        array (destination) and shared memory (source). Used to read a residual
        tile that was TMA-loaded into smem back into registers, aligned to the
        accumulator's T2R fragment.

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rC: The register tensor shaped like the accumulator fragment
        :type tTR_rC: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor

        :return: A tuple containing (tiled_copy_s2r, tSR_rC, tSR_sC) where:
            - tiled_copy_s2r: The tiled copy operation for smem to register copy(s2r)
            - tSR_rC: The partitioned register tensor (destination)
            - tSR_sC: The partitioned shared memory tensor (source)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_s2r = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.d_dtype)
        tiled_copy_s2r = cute.make_tiled_copy_D(copy_atom_s2r, tiled_copy_t2r)
        # (S2R, S2R_M, S2R_N, PIPE)
        thr_copy_s2r = tiled_copy_s2r.get_slice(tidx)
        tSR_sC = thr_copy_s2r.partition_D(sC)
        # (S2R, S2R_M, S2R_N)
        tSR_rC = tiled_copy_s2r.retile(tTR_rC)
        return tiled_copy_s2r, tSR_rC, tSR_sC

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gD_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sD: cute.Tensor,
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """Make tiledCopy for global memory store, then use it to:
        partition shared memory (source) and global memory (destination) for TMA store version.

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param atom: The copy_atom_d to be used for TMA store version, or tiled_copy_t2r for none TMA store version
        :type atom: cute.CopyAtom or cute.TiledCopy
        :param gD_mnl: The global tensor D
        :type gD_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param sD: The shared memory tensor to be copied and partitioned
        :type sD: cute.Tensor

        :return: A tuple containing (tma_atom_d, bSG_sD, bSG_gD) where:
            - tma_atom_d: The TMA copy atom
            - bSG_sD: The partitioned shared memory tensor D
            - bSG_gD: The partitioned global tensor D
        :rtype: Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gD_epi = cute.flat_divide(gD_mnl[((None, None), 0, 0, None, None, None)], epi_tile)

        tma_atom_d = atom
        sD_for_tma_partition = cute.group_modes(sD, 0, 2)
        gD_for_tma_partition = cute.group_modes(gD_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, RestM, RestN, RestL)
        bSG_sD, bSG_gD = cpasync.tma_partition(
            tma_atom_d,
            0,
            cute.make_layout(1),
            sD_for_tma_partition,
            gD_for_tma_partition,
        )
        return tma_atom_d, bSG_sD, bSG_gD

    def _make_acc_fake_tensor(self, tiled_mma, mma_tiler):
        """Build the accumulator fake tensor used for TMEM column accounting.

        For overlapping-accum the second logical acc buffer is strided into the
        columns otherwise reserved for SFA/SFB (so the next tile's math can
        overlap the current tile's epilogue). The stride for the stage dim is
        (cta_tile_n - num_sf_tmem_cols) * stride[0][1], the block-scaled GEMM
        overlapping-accum layout. Otherwise fall back to a plain num_acc_stage fake.
        """
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        if cutlass.const_expr(self.overlapping_accum):
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 2))
            s = tCtAcc_fake.stride
            return cute.make_tensor(
                tCtAcc_fake.iterator,
                cute.make_layout(
                    tCtAcc_fake.shape,
                    stride=(
                        s[0],
                        s[1],
                        s[2],
                        (self.cta_tile_shape_mnk[1] - self.num_sf_tmem_cols) * s[0][1],
                    ),
                ),
            )
        return tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

    def _compute_stages(
        self,
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        d_dtype: Type[cutlass.Numeric],
        d_layout: utils.LayoutEnum,
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        flat_mma_tiler_sf: Tuple[int, int, int],
        smem_capacity: int,
        occupancy: int,
        has_residual: bool = False,
    ) -> Tuple[int, int, int]:
        """Computes the number of stages for A/B/D operands.

        The A/B stage count is chosen so the full SharedStorage (every pipeline
        mbar, whose count scales with the stage count, plus every operand array
        with its 1024B alignment padding) fits in smem. The number is derived by
        constructing the actual SharedStorage struct for a candidate stage count
        and reading its exact size_in_bytes(), then taking the largest count that
        fits -- rather than dividing capacity by a per-stage byte estimate that
        omits the alignment padding and the stage-scaled mbar bytes.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
        :param a_dtype: Data type of operand A.
        :param b_dtype: Data type of operand B.
        :param epi_tile: The epilogue tile shape.
        :param d_dtype: Data type of operand D (output).
        :param d_layout: Layout enum of operand D.
        :param sf_dtype: Data type of Scale factor.
        :param sf_vec_size: Scale factor vector size.
        :param flat_mma_tiler_sf: The flat-K mma tiler the SF layouts are built
            from: the A/B tiler's M and N with the SF cadence on K.
        :param smem_capacity: Total available shared memory capacity in bytes.
        :param occupancy: CTAs per SM. Always 1 for this persistent kernel.

        :return: (ACC stages, A/B operand stages, D stages)
        :rtype: tuple[int, int, int]
        """
        # ACC stages
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2

        # D stages
        num_d_stage = 2

        buffer_align_bytes = 1024
        num_clc_stage = 1
        num_clc_response_bytes = 16
        cta_tile_n = mma_tiler_mnk[1]

        def smem_bytes(ab_stage: int, d_stage: int) -> int:
            """Exact SharedStorage byte size for a candidate stage count.

            Builds the operand/SF/epi layouts for this ``ab_stage``/``d_stage``
            and assembles the SharedStorage this kernel will allocate. Every field
            mirrors the runtime SharedStorage struct defined in __call__; keep the
            two in sync so the byte total is exact.
            """
            a_smem = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, ab_stage)
            b_smem = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b_dtype, ab_stage)
            sfa_smem = blockscaled_utils.make_smem_layout_sfa(
                tiled_mma, flat_mma_tiler_sf, sf_vec_size, ab_stage
            )
            sfb_smem = blockscaled_utils.make_smem_layout_sfb(
                tiled_mma, flat_mma_tiler_sf, sf_vec_size, ab_stage
            )
            d_smem = sm100_utils.make_smem_layout_epi(d_dtype, d_layout, epi_tile, d_stage)

            @cute.struct
            class ProbeStorage:
                """Field-for-field mirror of the runtime SharedStorage, sized for
                this stage count so size_in_bytes() gives the exact allocation."""

                ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stage * 2]
                sfa_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stage * 2]
                acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_acc_stage * 2]
                tmem_dealloc_mbar_ptr: cutlass.Int64
                tmem_holding_buf: cutlass.Int32
                clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_clc_stage * 2]
                clc_response: cute.struct.Align[
                    cute.struct.MemRange[
                        cutlass.Int32, num_clc_response_bytes // 4 * num_clc_stage
                    ],
                    16,
                ]
                sD: cute.struct.Align[
                    cute.struct.MemRange[d_dtype, cute.cosize(d_smem.outer)],
                    buffer_align_bytes,
                ]
                sBias: cute.struct.Align[
                    cute.struct.MemRange[d_dtype, cta_tile_n if self.has_bias else 0],
                    buffer_align_bytes,
                ]
                sC: cute.struct.Align[
                    cute.struct.MemRange[
                        d_dtype,
                        cute.cosize(d_smem.outer) if has_residual else 0,
                    ],
                    buffer_align_bytes,
                ]
                c_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, d_stage * 2 if has_residual else 0
                ]
                sA: cute.struct.Align[
                    cute.struct.MemRange[a_dtype, cute.cosize(a_smem.outer)],
                    buffer_align_bytes,
                ]
                sB: cute.struct.Align[
                    cute.struct.MemRange[b_dtype, cute.cosize(b_smem.outer)],
                    buffer_align_bytes,
                ]
                sSFA: cute.struct.Align[
                    cute.struct.MemRange[sf_dtype, cute.cosize(sfa_smem)],
                    buffer_align_bytes,
                ]
                sSFB: cute.struct.Align[
                    cute.struct.MemRange[sf_dtype, cute.cosize(sfb_smem)],
                    buffer_align_bytes,
                ]

            return ProbeStorage.size_in_bytes()

        # A single A/B stage must fit: a tile whose one-stage SharedStorage already
        # exceeds capacity is unimplementable and is rejected upstream by
        # can_implement, so the lower-bound search below can assume >= 1 fits.
        if smem_bytes(1, num_d_stage) > smem_capacity:
            raise RuntimeError(
                "tile too large for even one A/B stage; can_implement should have "
                "rejected it before reaching stage selection"
            )

        # Compute a lower bound that is guaranteed to fit, then scan upward once
        # (never downward) to the largest A/B stage whose exact SharedStorage
        # fits. Each added stage grows the four staged arrays (sA/sB/sSFA/sSFB) by
        # their raw per-stage bytes; the Align[1024] rounding of each array is a
        # one-time boundary crossing over the whole growth, so it contributes at
        # most 4*1024 B total. Bounding the alignment loss as that constant in the
        # numerator (rather than inflating every stage's slope) keeps the lower
        # bound within a few stages of the true maximum while still guaranteeing
        # cost(lb) <= cost(1) + (lb-1)*slope + 4*1024 <= capacity. A/B depth is
        # filled first (it dominates mainloop overlap); any smem left over then
        # deepens the epilogue. size_in_bytes is monotonic in each stage count, so
        # the upward passes land on the true maximum.
        a_one = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, 1)
        b_one = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b_dtype, 1)
        sfa_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, flat_mma_tiler_sf, sf_vec_size, 1
        )
        sfb_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, flat_mma_tiler_sf, sf_vec_size, 1
        )
        slope = (
            cute.cosize(a_one.outer) * a_dtype.width // 8
            + cute.cosize(b_one.outer) * b_dtype.width // 8
            + cute.cosize(sfa_one) * sf_dtype.width // 8
            + cute.cosize(sfb_one) * sf_dtype.width // 8
            + 2 * 8  # ab_full mbar (Int64)
            + 2 * 8  # sfa_full mbar (Int64)
        )
        align_loss = 4 * buffer_align_bytes
        num_ab_stage = max(
            1, 1 + (smem_capacity - smem_bytes(1, num_d_stage) - align_loss) // slope
        )
        while smem_bytes(num_ab_stage + 1, num_d_stage) <= smem_capacity:
            num_ab_stage += 1
        while smem_bytes(num_ab_stage, num_d_stage + 1) <= smem_capacity:
            num_d_stage += 1

        return num_acc_stage, num_ab_stage, num_d_stage

    def can_implement(
        self,
        c: int,
        k: int,
        ab_dtype: Type[cutlass.Numeric],
        d_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        output: cute.Tensor,
        filter_trs: Tuple[int, int, int],
        upper_padding_dhw: Tuple[int, int, int],
        lower_padding_dhw: Tuple[int, int, int],
        stride_dhw: Tuple[int, int, int],
        dil_dhw: Tuple[int, int, int],
        c_dtype: Optional[Type[cutlass.Numeric]] = None,
        has_bias: bool = False,
        epilogue_op: Optional[cutlass.Constexpr] = None,
    ) -> bool:
        """Determine if the given tensor configuration can be implemented by this kernel."""
        try:
            # Residual (C) reuses the output D im2col TMA descriptor, so it must
            # match D's dtype exactly. c_dtype is None when no residual is passed.
            if c_dtype is not None and c_dtype is not d_dtype:
                raise testing.CantImplementError(
                    f"Residual c_dtype ({c_dtype}) must equal d_dtype ({d_dtype})."
                )
            # Two input formats, each pinned to one scale-factor format:
            #   NVFP4  : Float4E2M1FN A/B, Float8E4M3FN scale over 16 channels.
            #   MXFP8  : Float8E4M3FN A/B, Float8E8M0FNU scale over 32 channels.
            # The element type is pinned alongside the vector size because it drives
            # the SFB SMEM layout, the barrier transaction bytes and the SFD rounding
            # -- a mismatched one is misinterpreted on the device, not refused. SFD
            # takes the input's format, so this pins both sides.
            sf_format = {
                cutlass.Float4E2M1FN: (cutlass.Float8E4M3FN, 16),
                cutlass.Float8E4M3FN: (cutlass.Float8E8M0FNU, 32),
            }.get(ab_dtype)
            if sf_format is None:
                raise testing.CantImplementError(
                    f"Only Float4E2M1FN (NVFP4) and Float8E4M3FN (MXFP8) A/B "
                    f"are supported, got {ab_dtype}."
                )
            want_sf_dtype, want_sf_vec_size = sf_format
            if sf_dtype is not want_sf_dtype or self.sf_vec_size != want_sf_vec_size:
                raise testing.CantImplementError(
                    f"{ab_dtype} A/B requires sf_dtype={want_sf_dtype} over "
                    f"{want_sf_vec_size} channels, got sf_dtype={sf_dtype}, "
                    f"sf_vec_size={self.sf_vec_size}."
                )
            # D is either the narrow block-scaled output that pairs with the input
            # format -- NVFP4 with FP4, MXFP8 with FP8 E4M3 -- or a 16-bit wide output.
            # The pairing is forced by the SFD: the epilogue rescales the accumulator by
            # an approximate reciprocal of the quantized scale factor, exact for E8M0's
            # powers of two but not for E4M3's arbitrary values. FP4's eight magnitudes
            # absorb that error; FP8 E4M3's few hundred do not, leaving a data-dependent
            # fraction of elements one step off. A wide output carries no SFD and stops
            # at 16 bits: FP32 D doubles the epilogue buffer and takes A/B stages with
            # it, and what reads this kernel's output is narrow or 16-bit.
            paired_narrow_output = {
                cutlass.Float4E2M1FN: cutlass.Float4E2M1FN,
                cutlass.Float8E4M3FN: cutlass.Float8E4M3FN,
            }[ab_dtype]
            if d_dtype not in (
                paired_narrow_output,
                cutlass.BFloat16,
                cutlass.Float16,
            ):
                raise testing.CantImplementError(
                    f"d_dtype must be {paired_narrow_output} (the block-scaled narrow "
                    f"output paired with {ab_dtype} A/B), BFloat16 or Float16, got "
                    f"{d_dtype}."
                )
            # A block-scaled output constrains the whole epilogue. bias and residual
            # take D's own dtype, so the addend would be FP4 or FP8 -- eight magnitudes
            # in one case, a few hundred in the other -- making the term's quantization
            # error the same order as the term itself. And the SFD is taken from the
            # accumulator before the epilogue op runs, so anything but identity leaves D
            # scaled by a factor that no longer describes it. The op is matched against
            # the named non-identity activations: a caller's own pass-through lambda is
            # identity in effect and must not be refused for being a different object.
            if d_dtype in (cutlass.Float4E2M1FN, cutlass.Float8E4M3FN):
                if has_bias:
                    raise testing.CantImplementError(
                        f"A block-scaled {d_dtype} output does not support a bias."
                    )
                if c_dtype is not None:
                    raise testing.CantImplementError(
                        f"A block-scaled {d_dtype} output does not support a residual."
                    )
                if epilogue_op in {
                    entry["device"]
                    for name, entry in EPILOGUE_ACTIVATIONS.items()
                    if name != "identity"
                }:
                    raise testing.CantImplementError(
                        f"A block-scaled {d_dtype} output only supports the identity epilogue op."
                    )
            # The kernel emits the NZPQK output with K (the GEMM-N axis) contiguous,
            # i.e. n-major. The SFD store and TMA epilogue assume this; any other
            # leading dim would write the wrong axis. Reject it up front.
            if output.leading_dim != 4:
                raise testing.CantImplementError(
                    "D must be n-major (NZPQK with K contiguous, leading_dim=4), "
                    f"got leading_dim={output.leading_dim}"
                )
            self.check_mma_tiler_and_cluster_shape()
            # Blockscaled MMA requires a per-CTA M of 128: make_blockscaled_trivial_tiled_mma
            # has no M=64 atom and raises "expects the M-mode to be 128, but got 64" at
            # cute.compile time. The per-CTA M is mma_tiler_m halved under 2CTA, so both
            # mma_tiler_m=64 (1CTA) and mma_tiler_m=128 (2CTA) hit M=64 and fail. The
            # parent check_mma_tiler_and_cluster_shape admits both (it allows 64/128 for
            # 1CTA and 128/256 for 2CTA from the dense-GEMM contract), so reject them here
            # with a clear message instead of the opaque MMA-builder error. The two viable
            # shapes are mma_tiler_m=128 (1CTA) and mma_tiler_m=256 (2CTA), both giving M=128.
            per_cta_m = self.mma_tiler_mn[0] // (2 if self.use_2cta_instrs else 1)
            if per_cta_m != 128:
                raise testing.CantImplementError(
                    f"Blockscaled conv requires per-CTA M=128, got {per_cta_m} from "
                    f"mma_tiler_m={self.mma_tiler_mn[0]}, use_2cta_instrs="
                    f"{self.use_2cta_instrs}. Use mma_tiler_m=128 with 1CTA or "
                    f"mma_tiler_m=256 with 2CTA."
                )
            # SFB stages 128 output channels at a time, so an mma_tiler_n below that
            # puts several N tiles in one chunk, each needing its own Tensor Memory
            # read offset. That offset exists for 64 (two tiles per chunk) and 192
            # (the chunk pair reshaped into overlapping 256-wide windows); 128 and 256
            # sit on chunk boundaries. The rest silently read the chunk's first tile
            # once there is more than one N tile, so they serve a single one.
            cta_tile_n = self.mma_tiler_mn[1]
            if cta_tile_n not in (64, 128, 192, 256):
                n_tiles = -(-k // cta_tile_n)
                if n_tiles > 1:
                    raise testing.CantImplementError(
                        f"mma_tiler_n={cta_tile_n} has no per-tile SFB chunk offset, "
                        f"so it only serves a single N tile, but K={k} needs "
                        f"{n_tiles}. Use mma_tiler_n of 64, 128, 192 or 256."
                    )
            # 16B output alignment forces Kout % 32 == 0 for FP4 D, which is
            # what makes the SFD N-bound predicate (gates a whole subtile on its
            # first-element N coord) exact -- so no separate Kout guard is needed.
            _check_tensor_alignment(c, k, ab_dtype, d_dtype)
            # C legality. Either format takes a partial trailing K tile -- its A and B
            # arrive zero-filled from the TMA and its scale factors are read in bounds --
            # so C only has to be aligned, not a whole number of K tiles. The modulus is
            # the 16-byte alignment of the channel axis: 32 channels for NVFP4's 4-bit
            # elements, 16 for MXFP8's 8-bit ones.
            # MXFP8 states its bound explicitly even though the alignment check already
            # enforces the same 16, because what bounds C on this path is the SFA
            # cp.async geometry, not the alignment: one 4-byte transfer carries 4 SF
            # blocks of 32 channels, exactly one 128-channel SF tile. Should either
            # side move, this has to fail on its own terms rather than silently ride
            # on a number that happens to agree today.
            if ab_dtype is cutlass.Float8E4M3FN:
                if c % 16 != 0:
                    raise testing.CantImplementError(
                        f"MXFP8 requires C to be a multiple of 16, got C = {c}."
                    )
            # cta_tile_k legality. Both formats need a whole number of MMA K
            # instructions: mma_inst_tile_k truncates the division, so a tile_k off
            # that multiple silently runs a narrower tile than asked for. 192 is a
            # legal multiple but excluded: its SF-K atom count is three, and an odd
            # count above one cannot be halved when the SFB multicast splits along K
            # under 2CTA, which deadlocks the tmem-alloc mbarrier. A K tile need not
            # divide C, so nothing needs it.
            # MXFP8 stops at 128, the width of its SF atom. Below it the atom is
            # shared: a 64-channel K tile stages the whole 128-channel atom and reads
            # half of it, which only works because the two halves divide it exactly.
            # Above it a K tile would span several atoms per tile, which no
            # configuration here exercises.
            # NVFP4's atom is 64 channels, at or below every legal tile, so its upper
            # end is set by validation instead: 256 is four MMA K instructions, wider
            # tiles are unvalidated and tracing time climbs past them -- 768 does not
            # finish in 400 s.
            allowed_tile_k = (64, 128) if ab_dtype is cutlass.Float8E4M3FN else (64, 128, 256)
            if self.cta_tile_k not in allowed_tile_k:
                raise testing.CantImplementError(
                    f"{ab_dtype} requires cta_tile_k in {allowed_tile_k}, got {self.cta_tile_k}."
                )
            # Three fields of the 5D im2col tensor map are narrower than the
            # convolution parameters feeding them, and a 3D convolution always builds
            # a 5D descriptor. The widths come from the encodings: a 5-bit signed
            # corner per spatial dimension ([-16, 15]), an unsigned 5-bit coordinate
            # offset per dimension ([0, 31]), and a 3-bit traversal stride holding the
            # stride minus one ([1, 8]). A corner is not the padding itself -- the
            # leading one is -lower_padding, the trailing one
            # upper_padding - (filter - 1) * dilation -- so padding and dilation only
            # bind in combination. Overflowing one truncates it and moves the pixel
            # box, surfacing as an illegal instruction when the shifted box leaves
            # mapped memory and as a wrong answer when it does not, so computing
            # correctly past a bound is luck, not contract.
            corner_lo = -16
            corner_hi = 15
            max_filter_offset = 31
            max_element_stride = 8
            for dim, flt, dil, pad_up, pad_lo, stride in zip(
                ("D", "H", "W"),
                filter_trs,
                dil_dhw,
                upper_padding_dhw,
                lower_padding_dhw,
                stride_dhw,
                strict=True,
            ):
                leading_corner = -pad_lo
                trailing_corner = pad_up - (flt - 1) * dil
                if not corner_lo <= leading_corner <= corner_hi:
                    raise testing.CantImplementError(
                        f"{dim} leading im2col corner is -lower_padding = "
                        f"{leading_corner}, outside the [{corner_lo}, {corner_hi}] the "
                        f"descriptor's signed 5-bit corner encodes; lower_padding_"
                        f"{dim.lower()} must be at most {-corner_lo}"
                    )
                if not corner_lo <= trailing_corner <= corner_hi:
                    raise testing.CantImplementError(
                        f"{dim} trailing im2col corner is upper_padding - (filter - 1) "
                        f"* dilation = {pad_up} - ({flt} - 1) * {dil} = "
                        f"{trailing_corner}, outside the [{corner_lo}, {corner_hi}] the "
                        f"descriptor's signed 5-bit corner encodes"
                    )
                if stride > max_element_stride:
                    raise testing.CantImplementError(
                        f"stride_{dim.lower()}={stride} exceeds the {max_element_stride}"
                        f" a traversal stride encodes, holding the stride minus one in "
                        f"3 bits"
                    )
                # A filter offset is dilation * tap index, so the largest one a
                # dimension reaches is (filter - 1) * dilation. Overflowing the
                # unsigned field shifts the filter taps and silently changes the
                # result, so it is bounded even when both corners are in range.
                filter_offset = (flt - 1) * dil
                if filter_offset > max_filter_offset:
                    raise testing.CantImplementError(
                        f"{dim} filter offset is (filter - 1) * dilation = ({flt} - 1) "
                        f"* {dil} = {filter_offset}, past the {max_filter_offset} an "
                        f"unsigned 5-bit im2col coordinate offset encodes"
                    )
        except testing.CantImplementError as e:
            print(e)
            return False
        return True


def sf_k_tile_channels(cta_tile_k: int, sf_vec_size: int) -> int:
    """Channels one SF K tile spans.

    The SF atom holds 4 scale-factor blocks along K, so it covers
    4 * sf_vec_size channels and cannot be staged in part: a narrower A/B K tile
    still stages a whole atom, and several A/B tiles then share one SF tile. Host
    and kernel both derive the SF cadence from here so they cannot drift apart.
    """
    return max(cta_tile_k, 4 * sf_vec_size)


def sfb_per_position_channels(c: int, cta_tile_k: int, sf_vec_size: int) -> int:
    """Channels of SFB one filter position spans: C rounded up to whole SF K tiles.

    SFB runs channels and filter positions together in one flat K mode, so a K tile
    that does not divide C would straddle two positions; rounding each position out
    to whole SF K tiles puts every position on a tile boundary. The bound is the SF
    tile, the width of the TMA box: a C that a 64-channel A/B tile divides still
    straddles two positions once a 128-channel SF chunk serves two of those tiles.
    """
    span = sf_k_tile_channels(cta_tile_k, sf_vec_size)
    return -(-c // span) * span


# Exempt from the frontend cache-key rule (see indexer/mqa_logits_util.py
# frontend_cache_token): the key below contains cute.Tensor arguments, which
# hash by identity and are rebuilt via from_dlpack on every run(), so this
# cache can never hit across calls, let alone across frontends. If the Tensor
# arguments are ever dropped from the key or Tensor gains structural hashing,
# add the frontend token here.
@lru_cache(maxsize=1)
def compile_conv(
    ncdhw: Tuple[int, int, int, int, int],
    ktrs: Tuple[int, int, int, int],
    input: cute.Tensor,
    filter: cute.Tensor,
    output: cute.Tensor,
    acc_dtype: Type[cutlass.Numeric],
    sfa: cute.Tensor,
    sfb: cute.Tensor,
    sfd: Optional[cute.Tensor],
    alpha: cute.Tensor,
    norm_const_tensor: Optional[cute.Tensor],
    bias: Optional[cute.Tensor],
    residual: Optional[cute.Tensor],
    beta: float,
    sf_vec_size: int,
    cta_tile_k: int,
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
    Compile a 3D convolution kernel.

    :param ncdhw: Problem shape (N, C, D, H, W)
    :param ktrs: Problem shape (K, T, R, S)
    :param input: Input tensor (N, D, H, W, C) with C contiguous
    :param filter: Filter tensor in KTRSC format (K, T, R, S, C) with C contiguous
    :param output: Output tensor (N, Z, P, Q, K) with K contiguous
    :param acc_dtype: Accumulator data type
    :param mma_tiler: MMA tile shape (M, N)
    :param preferred_cluster_shape_mn: Preferred cluster shape (M, N) for CLC dynamic scheduling
    :param fallback_cluster_shape_mn: Fallback cluster shape (M, N) for CLC dynamic scheduling
    :param swizzle_size: Swizzling size in the unit of cluster. 1 means no swizzle
    :param raster_along: Rasterization order of clusters. Only used when swizzle_size > 1
    :param use_2cta_instrs: Whether to use 2CTA instructions
    :param upper_padding_dhw: Upper padding in depth, height, width order
    :param lower_padding_dhw: Lower padding in depth, height, width order
    :param stride_dhw: Stride (Sd, Sh, Sw)
    :param dilation_dhw: Dilation (DilD, DilH, DilW)
    :param epilogue_op: Epilogue operation
    :return: Compiled kernel function
    """
    from cutlass.cute.runtime import make_fake_stream

    # Output spatial dims for the SFD global descriptor (host int, trace-const).
    zpq = compute_zpq(
        ncdhw[2:],
        ktrs[1:],
        stride_dhw,
        upper_padding_dhw,
        lower_padding_dhw,
        dilation_dhw,
    )
    # Host-side swizzle_size guard: reject a swizzle that exceeds the cluster
    # count in the swizzled dimension (GEMM-M = N*Z*P*Q spatial, GEMM-N = Kout).
    _check_swizzle_size(
        ncdhw[0] * zpq[0] * zpq[1] * zpq[2],
        ktrs[0],
        mma_tiler,
        use_2cta_instrs,
        preferred_cluster_shape_mn,
        fallback_cluster_shape_mn,
        swizzle_size,
        raster_along,
    )

    # Create convolution kernel object. Only cta_tile_k (compile-time K tile) is
    # a build-time shape parameter; the input channel count C and all output
    # geometry (N/Z/P/Q/K) are read from the dynamic tensors at runtime, so one
    # cubin serves every C. Filter T/R/S and pad/stride/dil are likewise runtime
    # (dynamic filter extents + boxed Int32).
    conv_op = Sm100BlockScaledPersistentDenseImplicitGemmKernel(
        acc_dtype,
        sf_vec_size,
        use_2cta_instrs,
        mma_tiler,
        preferred_cluster_shape_mn,
        fallback_cluster_shape_mn,
        cta_tile_k,
        swizzle_size,
        raster_along,
    )

    # Check if configuration can be implemented
    can_implement = conv_op.can_implement(
        ncdhw[1],
        ktrs[0],
        input.element_type,
        output.element_type,
        sfa.element_type,
        output,
        ktrs[1:],
        upper_padding_dhw,
        lower_padding_dhw,
        stride_dhw,
        dilation_dhw,
        residual.element_type if residual is not None else None,
        bias is not None,
        epilogue_op,
    )
    if not can_implement:
        raise testing.CantImplementError("The current config is invalid/unsupported.")

    stream = make_fake_stream()
    # Box pad/stride/dilation as cutlass.Int32 so the cute.compile entry scalars
    # lower to runtime SSA AND keep their values out of the mangled function
    # name; that combination is what lets one cubin be reused across configs. A
    # raw Python int here would fold the value into the name and force
    # recompilation per config. The filter T/R/S are already runtime (taken from
    # the dynamic-layout filter tensor extents).
    return cute.compile(
        conv_op,
        input,
        filter,
        output,
        sfa,
        sfb,
        alpha,
        epilogue_op,
        sfd,
        norm_const_tensor,
        bias,
        residual,
        beta,
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
    leading_dim: int = None,
) -> Tuple[cute.Tensor, torch.Tensor]:
    """Create a dynamic-layout cute tensor from a source f32 tensor.

    The tensor is always marked dynamic-layout: its non-leading extents lower to
    runtime SSA so one compiled cubin serves any N/C/D/H/W/K/T/R/S config.

    :param source_f32_tensor: Source f32 tensor
    :type source_f32_tensor: torch.Tensor
    :param dtype: Data type
    :type dtype: Type[cutlass.Numeric]
    :param leading_dim: Leading dimension kept contiguous in the dynamic layout
    :type leading_dim: int
    :return: Tuple of cute tensor and storage tensor
    :rtype: Tuple[cute.Tensor, torch.Tensor]
    """

    # FP4 needs packed storage: 2 elements per byte. The shared cute_tensor_like
    # over-allocates 2x for FP4 (uint8 buffer with same shape as source);
    # quantization writes only first half, kernel reads back second-half garbage
    # via the FP4 layout (0.5 byte/elem stride).
    if dtype == cutlass.Float4E2M1FN:
        shape = tuple(source_f32_tensor.shape)
        assert shape[-1] % 2 == 0, (
            f"FP4 packed storage requires trailing dim even, got shape={shape}"
        )
        packed_shape = shape[:-1] + (shape[-1] // 2,)
        storage_int8 = torch.empty(packed_shape, dtype=torch.int8, device="cuda")
        storage_view = storage_int8.view(dtype=torch.float4_e2m1fn_x2)
        cute_tensor = from_dlpack(storage_view, assumed_align=16)
        cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)

        if source_f32_tensor.numel() > 0:
            f32_gpu = (
                source_f32_tensor.cuda()
                if source_f32_tensor.device.type == "cpu"
                else source_f32_tensor
            )
            f32_cute = from_dlpack(f32_gpu)
            f32_cute = f32_cute.mark_layout_dynamic(leading_dim=leading_dim)
            cute.testing.convert(f32_cute, cute_tensor)
        return cute_tensor, storage_view

    cute_tensor, storage_tensor = cutlass_torch.cute_tensor_like(
        source_f32_tensor, dtype, is_dynamic_layout=True, assumed_align=16
    )

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


@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
    sf_ref_tensor: cute.Tensor,
    sf_mma_tensor: cute.Tensor,
):
    """Convert scale factor tensor from MKL layout to mma specification M(32x4xrest_m)xK(4xrest_k)xL layout"""
    # sf_mma_tensor has flatten shape (32, 4, rest_m, 4, rest_k, l)
    # group to ((32, 4, rest_m), (4, rest_k), l)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_ref_tensor)):
        mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
        sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]


# Compile-time epilogue activations, folded into the cubin as a Constexpr op
# (one activation per cubin). Each entry pairs the device-side op applied to the
# output fragment with the torch op used to build the reference.
EPILOGUE_ACTIVATIONS = {
    "identity": {
        "device": lambda x: x,
        "ref": lambda x: x,
    },
    "relu": {
        "device": lambda x: cute.where(x > 0, x, cute.full_like(x, 0)),
        "ref": torch.nn.functional.relu,
    },
}


def run(
    ncdhw: Tuple[int, int, int, int, int],
    ktrs: Tuple[int, int, int, int],
    stride_dhw: Tuple[int, int, int] = (1, 1, 1),
    upper_pad_dhw: Tuple[int, int, int] = (0, 0, 0),
    lower_pad_dhw: Tuple[int, int, int] = (0, 0, 0),
    dil_dhw: Tuple[int, int, int] = (1, 1, 1),
    ab_dtype: Type[cutlass.Numeric] = cutlass.Float4E2M1FN,
    d_dtype: Type[cutlass.Numeric] = cutlass.Float4E2M1FN,
    c_dtype: Optional[Type[cutlass.Numeric]] = None,
    acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
    sf_dtype: Type[cutlass.Numeric] = cutlass.Float8E4M3FN,
    sf_vec_size: int = 16,
    cta_tile_k: Optional[int] = None,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    preferred_cluster_shape_mn: Tuple[int, int] = (2, 1),
    fallback_cluster_shape_mn: Tuple[int, int] = (1, 1),
    swizzle_size: int = 1,
    raster_along: Literal["m", "n"] = "m",
    use_2cta_instrs: bool = False,
    tolerance: float = 1e-02,
    warmup_iterations: int = 0,
    iterations: int = 1,
    use_cold_l2: bool = False,
    skip_ref_check: bool = False,
    use_bias: bool = False,
    beta: float = 0.0,
    activation: str = "identity",
    **kwargs,
):
    """Run 3D convolution and compare against PyTorch reference.

    The filter tensor uses native KTRSC layout (K, T, R, S, C) with C contiguous.

    :param ncdhw: Input tensor shape (N, C, D, H, W)
    :param ktrs: Filter tensor shape components (K, T, R, S)
    :param stride_dhw: Stride (Sd, Sh, Sw)
    :param upper_pad_dhw: Upper padding in depth, height, width order
    :param lower_pad_dhw: Lower padding in depth, height, width order
    :param dil_dhw: Dilation (DilD, DilH, DilW)
    :param ab_dtype: Data type for A/B input tensors
    :param d_dtype: Data type for output tensor D
    :param acc_dtype: Accumulator data type
    :param mma_tiler_mn: MMA tiler shape
    :param preferred_cluster_shape_mn: Preferred cluster shape (M, N) for CLC dynamic scheduling
    :param fallback_cluster_shape_mn: Fallback cluster shape (M, N) for CLC dynamic scheduling
    :param swizzle_size: Swizzling size in the unit of cluster. 1 means no swizzle
    :param raster_along: Rasterization order of clusters
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

    # Residual (C) shares the output's shape and im2col TMA descriptor, so its
    # dtype defaults to the output dtype when the caller does not set one.
    if c_dtype is None:
        c_dtype = d_dtype

    # cta_tile_k is the compile-time K tile. When unset, take the largest legal tile
    # that fits in the channel count. A C the tile does not divide is fine -- the
    # trailing tile is partial. A caller may pass one explicitly to reuse a cubin
    # across several C, or to trade tile width against channel padding.
    #
    # MXFP8 defaults to tile_k=128, the width of its SF atom, so a K tile stages
    # exactly one atom. tile_k=64 is legal too -- two K tiles then share one staged
    # atom -- and halves the channel padding on a C that 128 overshoots.
    if cta_tile_k is None:
        if ab_dtype is cutlass.Float8E4M3FN:
            cta_tile_k = 128
        else:
            cta_tile_k = max(t for t in (64, 128, 256) if t <= max(64, min(256, C)))

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
    print(f"  Upper padding (depth, height, width): {upper_pad_dhw}")
    print(f"  Lower padding (depth, height, width): {lower_pad_dhw}")
    print(f"  Dilation (DilD, DilH, DilW): {dil_dhw}")
    print(f"  A/B data type: {ab_dtype}")
    print(f"  D data type: {d_dtype}")
    print(f"  Accumulator type: {acc_dtype}")
    print(f"  sf data type: {sf_dtype}")
    print(f"  sf vec size: {sf_vec_size}")
    print(f"  MMA tiler (M, N): {mma_tiler_mn}")
    print(f"  Preferred cluster shape (M, N): {preferred_cluster_shape_mn}")
    print(f"  Fallback cluster shape (M, N): {fallback_cluster_shape_mn}")
    print(f"  Swizzle size: {swizzle_size}")
    print(f"  Raster along: {raster_along}")
    print(f"  Use 2CTA instructions: {use_2cta_instrs}")
    print()

    # Create input and filter tensors
    input_tensor, filter_tensor, output_tensor = prepare_tensors(ncdhw, ktrs, (Z, P, Q), ab_dtype)

    # Prepare cute tensors
    input_, input_storage = create_cute_tensor(input_tensor, ab_dtype, leading_dim=4)
    filter_, filter_storage = create_cute_tensor(filter_tensor, ab_dtype, leading_dim=4)
    output_, output_storage = create_cute_tensor(output_tensor, d_dtype, leading_dim=4)

    # Create scale factor tensor SFA/SFB
    def create_scale_factor_tensor_swizzled(L, mn, k, sf_vec_size, dtype):
        def ceil_div(a, b):
            return (a + b - 1) // b

        sf_k = ceil_div(k, sf_vec_size)
        ref_shape = (L, mn, sf_k)

        atom_m = (32, 4)
        atom_k = 4
        mma_shape = (
            L,
            ceil_div(mn, atom_m[0] * atom_m[1]),
            ceil_div(sf_k, atom_k),
            atom_m[0],
            atom_m[1],
            atom_k,
        )

        ref_permute_order = (1, 2, 0)
        mma_permute_order = (3, 4, 1, 5, 2, 0)

        # Create f32 ref torch tensor (cpu)
        ref_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
            ref_shape,
            torch.float32,
            permute_order=ref_permute_order,
            init_type=cutlass_torch.TensorInitType.RANDOM,
            init_config=cutlass_torch.RandomInitConfig(
                min_val=1,
                max_val=3,
            ),
        )
        # Create f32 cute torch tensor (cpu)
        cute_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
            mma_shape,
            torch.float32,
            permute_order=mma_permute_order,
            init_type=cutlass_torch.TensorInitType.RANDOM,
            init_config=cutlass_torch.RandomInitConfig(
                min_val=0,
                max_val=1,
            ),
        )

        # convert ref f32 tensor to cute f32 tensor
        cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
            from_dlpack(ref_f32_torch_tensor_cpu),
            from_dlpack(cute_f32_torch_tensor_cpu),
        )
        cute_f32_torch_tensor = cute_f32_torch_tensor_cpu.cuda()

        # reshape makes memory contiguous
        ref_f32_torch_tensor_cpu = (
            ref_f32_torch_tensor_cpu.permute(2, 0, 1)
            .unsqueeze(-1)
            .expand(L, mn, sf_k, sf_vec_size)
            .reshape(L, mn, sf_k * sf_vec_size)
            .permute(*ref_permute_order)
        )
        # prune to mkl for reference check.
        ref_f32_torch_tensor_cpu = ref_f32_torch_tensor_cpu[:, :k, :]

        # Round-trip the reference scale factors through the storage dtype so the
        # reference dequant uses the exact values the kernel reads. E4M3 keeps
        # fractional values (near-lossless for the 1..3 init range), but E8M0
        # (MXFP8) snaps every scale to a power of two, so an un-rounded f32
        # reference would disagree with the kernel on every non-pow2 block.
        ref_f32_torch_tensor_cpu = ref_f32_torch_tensor_cpu.to(cutlass_torch.dtype(dtype)).to(
            torch.float32
        )

        # Create dtype cute torch tensor (cpu)
        cute_tensor, cute_torch_tensor = cutlass_torch.cute_tensor_like(
            cute_f32_torch_tensor_cpu,
            dtype,
            is_dynamic_layout=True,
            assumed_align=16,
        )

        # Convert f32 cute tensor to dtype cute tensor
        cute_tensor = cutlass_torch.convert_cute_tensor(
            cute_f32_torch_tensor,
            cute_tensor,
            dtype,
            is_dynamic_layout=True,
        )
        return ref_f32_torch_tensor_cpu, cute_tensor, cute_torch_tensor

    def create_scale_factor_tensor_unswizzled(
        L, mn, k, sf_vec_size, dtype, mark_dynamic_leading_dim=None
    ):
        def ceil_div(a, b):
            return (a + b - 1) // b

        sf_k = ceil_div(k, sf_vec_size)
        # Pad sf_k to multiple of 4 so that the M-direction stride (= sf_k_padded)
        # satisfies the 4-byte cp.async alignment requirement.
        atom_k = 4
        sf_k_padded = ceil_div(sf_k, atom_k) * atom_k

        # Use pure PyTorch to create the tensor, bypassing cute_tensor_like which
        # requires the leading mode to be divisible by 4 for 8-bit types.
        sf_raw = torch.randint(0, 3, (L, mn, sf_k_padded), dtype=torch.uint8).permute(1, 2, 0)
        if sf_k_padded > sf_k:
            sf_raw[:, sf_k:, :] = 0

        sf_torch = sf_raw.to(dtype=cutlass_torch.dtype(dtype)).cuda()
        sf_tensor = from_dlpack(sf_torch, assumed_align=16)
        # When the kernel indexes this tensor directly (SFA: the device reads its
        # gmem through mSFA_mkl's own strides, no host rebuild), the sf_k extent
        # is C/sf_vec_size and its row stride must stay runtime so one cubin
        # serves every C. Without this the stride is frozen to the compile C and
        # a different C reads every row at the wrong offset.
        if mark_dynamic_leading_dim is not None:
            sf_tensor = sf_tensor.mark_layout_dynamic(leading_dim=mark_dynamic_leading_dim)

        # Build f32 reference for verification (only the original sf_k, not
        # padded). Decode the scale factors through the storage dtype so the
        # reference matches the values the kernel reads: E8M0 (MXFP8) has no
        # exact zero (0 stores as the smallest exponent, ~5.9e-39) and only
        # represents powers of two, so the raw uint8 -> f32 cast would disagree
        # with the kernel on those blocks. E4M3 decodes the small integers
        # losslessly, leaving the NVFP4 path unchanged.
        sf_ref = (
            sf_raw[:, :sf_k, :]
            .to(dtype=cutlass_torch.dtype(dtype))
            .float()
            .permute(2, 0, 1)
            .unsqueeze(-1)
            .expand(L, mn, sf_k, sf_vec_size)
            .reshape(L, mn, sf_k * sf_vec_size)
            .permute(1, 2, 0)
        )
        sf_ref = sf_ref[:, :k, :]

        return sf_ref, sf_tensor, sf_torch

    sfa_ref, sfa_, sfa_storage = create_scale_factor_tensor_unswizzled(
        1,
        N * D * H * W,
        C,
        sf_vec_size,
        sf_dtype,
        mark_dynamic_leading_dim=1,
    )
    # Allocate SFB at the per-position span: the channels past C pair with the
    # zero-filled B the TMA produces there, so what they scale is zero.
    sfb_c_span = sfb_per_position_channels(C, cta_tile_k, sf_vec_size)
    sfb_ref, sfb_, sfb_storage = create_scale_factor_tensor_swizzled(
        1,
        K,
        sfb_c_span * T * R * S,
        sf_vec_size,
        sf_dtype,
    )
    # SFD: emitted for a narrow (block-scaled) output; FP16 and BF16 carry none. It
    # takes the input's scale format outright, so NVFP4 stays E4M3 over 16 channels
    # and MXFP8 E8M0 over 32 on both sides.
    gen_sfd = d_dtype in (cutlass.Float4E2M1FN, cutlass.Float8E4M3FN)
    sfd_vec_size = sf_vec_size
    sfd_dtype = sf_dtype
    if gen_sfd:
        _, sfd_, sfd_storage = create_scale_factor_tensor_unswizzled(
            1,
            N * Z * P * Q,
            K,
            sfd_vec_size,
            sfd_dtype,
        )
        # Helper randomizes by default -- we want to read what the kernel writes.
        sfd_storage.zero_()
    else:
        sfd_, sfd_storage = None, None

    # Per-output-channel bias (length K), dtype matches the output. Added to the
    # accumulator in FP32 as D = alpha*acc + bias, broadcast across every spatial
    # output position.
    if use_bias:
        bias_storage = torch.randn(K, dtype=torch.float32, device="cuda").to(
            cutlass_torch.dtype(d_dtype)
        )
        bias_ = from_dlpack(bias_storage, assumed_align=16)
    else:
        bias_storage, bias_ = None, None

    # Per-element residual (C), same (N, Z, P, Q, K) shape as the output D.
    # Added to the accumulator in FP32 as D = alpha*acc + bias + beta*C.
    # Built as a dynamic-layout cute tensor (leading_dim=4, K contiguous) so it
    # is described by the same im2col TMA descriptor as the output store.
    if beta != 0.0:
        c_f32_src = torch.randn((N, Z, P, Q, K), dtype=torch.float32, device="cuda")
        c_, c_storage = create_cute_tensor(c_f32_src, c_dtype, leading_dim=4)
    else:
        c_f32_src, c_storage, c_ = None, None, None

    # Keep runtime scales device-resident, matching the product path where a
    # dynamic activation scale is produced asynchronously on the GPU.
    alpha_storage = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    alpha_ = from_dlpack(alpha_storage, assumed_align=4)
    if gen_sfd:
        norm_const_storage = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        norm_const_ = from_dlpack(norm_const_storage, assumed_align=4)
    else:
        norm_const_storage, norm_const_ = None, None

    # Resolve the compile-time epilogue activation. The device op is folded into
    # the cubin as a Constexpr; the reference op mirrors it on the host.
    if activation not in EPILOGUE_ACTIVATIONS:
        raise ValueError(
            f"Unsupported activation {activation!r}; choose from {sorted(EPILOGUE_ACTIVATIONS)}"
        )
    epilogue_op = EPILOGUE_ACTIVATIONS[activation]["device"]

    # Compile convolution kernel
    print("Compiling kernel with cute.compile ...")
    compiled_fn = compile_conv(
        ncdhw,
        ktrs,
        input_,
        filter_,
        output_,
        acc_dtype,
        sfa_,
        sfb_,
        sfd_,
        alpha_,
        norm_const_,
        bias_,
        c_,
        beta,
        sf_vec_size,
        cta_tile_k,
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
        epilogue_op=epilogue_op,
    )

    # Get current CUDA stream
    torch_stream = torch.cuda.Stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    # The compiled entry expects the 12 pad/stride/dilation runtime scalars
    # (upper d/h/w, lower d/h/w, stride d/h/w, dil d/h/w) matching the boxed
    # cutlass.Int32 params baked into cute.compile; pass them as the same host
    # config values so one cubin serves any config.
    print("Running Blackwell 3D convolution...")
    compiled_fn(
        input_,
        filter_,
        output_,
        sfa_,
        sfb_,
        alpha_,
        sfd_,
        norm_const_,
        bias_,
        c_,
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
    torch_stream.synchronize()

    if not skip_ref_check:
        print("Verifying results with block-scaled reference...")
        # Block-scaled reference using F.conv3d with pre-scaled inputs
        # This handles padding/stride/dilation correctly

        # Pre-scale input by SFA: sfa_ref shape (N*D*H*W, C, 1)
        sfa_expanded = sfa_ref.squeeze(-1).reshape(N, D, H, W, C).cuda()
        scaled_input = input_tensor.cuda().float() * sfa_expanded

        # Pre-scale filter by SFB: sfb_ref shape (K, C*T*R*S, 1)
        # Drop the per-position padding channels: they scale zero-filled B and have
        # no counterpart in the reference convolution.
        sfb_expanded = sfb_ref.squeeze(-1).reshape(K, T, R, S, sfb_c_span)[..., :C].cuda()
        scaled_filter = filter_tensor.cuda().float() * sfb_expanded

        # F.conv3d expects (N, C, D, H, W) and (K, C, T, R, S)
        scaled_input_ncdhw = scaled_input.permute(0, 4, 1, 2, 3).contiguous()
        scaled_filter_kctrs = scaled_filter.permute(0, 4, 1, 2, 3).contiguous()

        if upper_pad_dhw == lower_pad_dhw:
            ref_nkzpq = F.conv3d(
                scaled_input_ncdhw,
                scaled_filter_kctrs,
                padding=upper_pad_dhw,
                stride=stride_dhw,
                dilation=dil_dhw,
            )
        else:
            # Asymmetric padding: F.pad then conv3d without padding.
            # Convention: lower_pad = leading (left), upper_pad = trailing (right).
            # This must match the im2col TMA descriptor which subtracts lower_pad
            # when mapping output coords to input coords (d_in = z*stride - lower_pad + t*dil).
            padded = F.pad(
                scaled_input_ncdhw,
                (
                    lower_pad_dhw[2],
                    upper_pad_dhw[2],
                    lower_pad_dhw[1],
                    upper_pad_dhw[1],
                    lower_pad_dhw[0],
                    upper_pad_dhw[0],
                ),
            )
            ref_nkzpq = F.conv3d(
                padded,
                scaled_filter_kctrs,
                stride=stride_dhw,
                dilation=dil_dhw,
            )

        # Convert to NZPQK layout (matching kernel output)
        ref = ref_nkzpq.permute(0, 2, 3, 4, 1).contiguous()  # (N, Z, P, Q, K)
        # Add per-output-channel bias (D = alpha*acc + bias) in the FP32 domain,
        # broadcast across N/Z/P/Q. alpha is 1.0 here so acc == conv result.
        if use_bias:
            ref = ref + bias_storage.float().view(1, 1, 1, 1, K)
        # Add the per-element residual (D = alpha*acc + bias + beta*residual) in
        # the FP32 domain. c_storage holds the exact bf16 values the
        # kernel loads, so float() reproduces them bit-for-bit.
        if beta != 0.0:
            ref = ref + beta * c_storage.float().reshape(N, Z, P, Q, K).cuda()
        # Apply the epilogue activation on the full linear combination
        # (D = activation(alpha*acc + bias + beta*residual)), matching the
        # device op folded into the kernel.
        ref = EPILOGUE_ACTIVATIONS[activation]["ref"](ref)
        # Snapshot the un-quantized FP32 ref BEFORE the in-place quantize round-trip
        # below mutates `ref` (shares GPU storage with ref_device).
        ref_unquant_cpu = ref.detach().cpu().clone()

        # Convert kernel FP4 output to f32 for comparison
        d_ref_device = torch.empty((N, Z, P, Q, K), dtype=torch.float32, device="cuda")
        cute.testing.convert(
            output_,
            from_dlpack(d_ref_device, assumed_align=16).mark_layout_dynamic(leading_dim=4),
        )
        d_ref_result = d_ref_device.cpu()

        # Quantize reference: f32 -> d_dtype -> f32 (mutates ref in-place via shared
        # storage). Scratch storage dtype follows the kernel-convert rule: sub-byte
        # FP4 and <=8-bit floats pack into a uint8 byte buffer, while wider types
        # (f16/bf16/f32) use their native torch dtype so the buffer is not
        # under-allocated (f16 needs 2 bytes/elem, not 1).
        ref_quant_byte_storage = (d_dtype.is_float and d_dtype.width <= 8) or (
            d_dtype.is_integer and d_dtype.width == 4
        )
        ref_quant_storage_dtype = (
            torch.uint8 if ref_quant_byte_storage else cutlass_torch.dtype(d_dtype)
        )
        ref_f4_ = torch.empty((N, Z, P, Q, K), dtype=ref_quant_storage_dtype, device="cuda")
        ref_f4 = from_dlpack(ref_f4_, assumed_align=16).mark_layout_dynamic(leading_dim=4)
        ref_f4.element_type = d_dtype
        ref_device = ref.contiguous().cuda()
        ref_tensor = from_dlpack(ref_device, assumed_align=16).mark_layout_dynamic(leading_dim=4)
        cute.testing.convert(ref_tensor, ref_f4)
        cute.testing.convert(ref_f4, ref_tensor)
        ref_quantized = ref_device.cpu()

        if gen_sfd:
            # SFD path: the kernel rescales acc by norm_const_tensor / SFD before the
            # FP4 cast, so the raw kernel D is only meaningful together with SFD.
            # Recompute the reference SFD and the SFD-rescaled reference output
            # fully on the host -- per-vector amax, scale factor, and rescale are
            # all derived from the un-quantized reference, independent of the
            # kernel's own SFD -- then compare SFD and the quantized output
            # elementwise. The reference block scale is read back after its E4M3
            # cast so the rescale is self-consistent, matching the kernel.
            sf_k = (K + sfd_vec_size - 1) // sfd_vec_size
            # norm_const_ is a device Float32; mirror it as a host float for the
            # torch-side replay (host path always uses the 1.0 default).
            norm_const_host = 1.0
            fp32_max = torch.finfo(torch.float32).max

            # 1. Reference SFD and SFD-rescaled reference output, host-side.
            #    per-vector amax over sfd_vec_size contiguous K elements
            #    -> pvscale = amax * norm_const_tensor / M_D
            #    -> cast to sfd_dtype and read back (sfd_ref)
            #    -> rescale ref by norm_const_tensor / sfd_ref.
            # M_D is the output dtype's full-scale max (FP4=6.0, E4M3=448.0);
            # sfd_dtype is the input sf_dtype (E4M3 for NVFP4, E8M0 for MXFP8).
            m_d = 6.0 if d_dtype is cutlass.Float4E2M1FN else 448.0
            sfd_torch_dtype = cutlass_torch.dtype(sfd_dtype)
            k_pad = sf_k * sfd_vec_size
            ref_pad = torch.zeros((N, Z, P, Q, k_pad), dtype=torch.float32)
            ref_pad[..., :K] = ref_unquant_cpu
            amax = (
                ref_pad.reshape(N, Z, P, Q, sf_k, sfd_vec_size).abs().amax(dim=5)
            )  # (N, Z, P, Q, sf_k)
            pvscale = amax * norm_const_host / m_d
            if sfd_dtype is cutlass.Float8E8M0FNU:
                # E8M0 SFD (MXFP8) rounds toward +inf to the next power of two,
                # matching the device cvt.rp.ue8m0; a plain round-to-nearest cast
                # would disagree by up to one exponent step. exp2/log2 run on CPU
                # (the CUDA path goes through nvrtc jiterator). E8M0 has no zero;
                # its smallest value is 2^-127, which cvt.rp of 0.0 also yields.
                sfd_ref = torch.exp2(torch.ceil(torch.log2(pvscale)).clamp(-127.0, 127.0))
            else:
                # E4M3 SFD (NVFP4) has a mantissa; round-to-nearest is exact enough.
                sfd_ref = pvscale.to(sfd_torch_dtype).to(torch.float32)
            acc_scale = (norm_const_host / sfd_ref.clamp(min=1e-30)).clamp(max=fp32_max)
            d_scaled = (
                ref_unquant_cpu
                * acc_scale.unsqueeze(-1)
                .expand(N, Z, P, Q, sf_k, sfd_vec_size)
                .reshape(N, Z, P, Q, k_pad)[..., :K]
            )

            # 2. Quantize the rescaled reference through d_dtype (f32 -> FP4 ->
            #    f32) via the same convert path the kernel output went through.
            d_ref_narrow_ = torch.empty(
                (N, Z, P, Q, K), dtype=ref_quant_storage_dtype, device="cuda"
            )
            d_ref_narrow = from_dlpack(d_ref_narrow_, assumed_align=16).mark_layout_dynamic(
                leading_dim=4
            )
            d_ref_narrow.element_type = d_dtype
            d_scaled_dev = d_scaled.contiguous().cuda()
            d_scaled_tensor = from_dlpack(d_scaled_dev, assumed_align=16).mark_layout_dynamic(
                leading_dim=4
            )
            cute.testing.convert(d_scaled_tensor, d_ref_narrow)
            cute.testing.convert(d_ref_narrow, d_scaled_tensor)
            d_ref_quant = d_scaled_dev.cpu()

            # 3. Decode the kernel-written SFD (plain (NZPQ, sf_k_padded, 1)
            #    layout) to f32 in the same (N, Z, P, Q, sf_k) shape as sfd_ref,
            #    reinterpreting the raw bytes through the actual SFD dtype.
            sfd_cpu = sfd_storage.cpu()
            sfd_bytes_view = sfd_cpu.view(torch.uint8)
            sfd_kernel = (
                sfd_bytes_view[:, :sf_k, :]
                .view(sfd_torch_dtype)
                .float()
                .squeeze(-1)
                .reshape(N, Z, P, Q, sf_k)
            )
            n_nonzero = (sfd_bytes_view[:, :sf_k, :] != 0).sum().item()
            assert n_nonzero > 0, "SFD is all zero -- kernel did not write SFD"

            # Compare SFD first: the output rescale depends on it, so an SFD
            # mismatch is the more fundamental signal.
            torch.testing.assert_close(sfd_kernel, sfd_ref, atol=1e-01, rtol=1e-01)
            torch.testing.assert_close(d_ref_result, d_ref_quant, atol=1e-01, rtol=1e-01)
            print("SFD dequant refcheck passed.")
        else:
            torch.testing.assert_close(
                d_ref_result,
                ref_quantized,
                atol=tolerance,
                rtol=1e-02,
            )
            print("Results match within tolerance!")

    # Benchmark if requested
    if iterations > 0:
        print(f"\nBenchmarking with {warmup_iterations} warmup and {iterations} iterations...")

        def generate_tensors():
            input_tensor, filter_tensor, output_tensor = prepare_tensors(
                ncdhw, ktrs, (Z, P, Q), ab_dtype
            )
            input_, input_storage = create_cute_tensor(input_tensor, ab_dtype, leading_dim=4)
            filter_, filter_storage = create_cute_tensor(filter_tensor, ab_dtype, leading_dim=4)
            output_, output_storage = create_cute_tensor(output_tensor, d_dtype, leading_dim=4)
            # Arg order must match the compiled entry exactly (epilogue_op and
            # beta are Constexpr, folded in at cute.compile time, so they are
            # NOT runtime args): a, b, d, sfa, sfb, alpha, sfd, norm_const_tensor,
            # bias, residual, then the 12 pad/stride/dil runtime Int32, then
            # stream last. Only the A/B/D tensors rotate per cold-L2 workspace;
            # the SF tensors, alpha/norm_const_tensor/bias/residual, and the geometry
            # scalars are reused.
            return testing.JitArguments(
                input_,
                filter_,
                output_,
                sfa_,
                sfb_,
                alpha_,
                sfd_,
                norm_const_,
                bias_,
                c_,
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
                + sfa_storage.numel() * sfa_storage.element_size()
                + sfb_storage.numel() * sfb_storage.element_size()
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
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid format. Expected comma-separated integers.")


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
        help="Upper padding in depth,height,width order",
    )
    parser.add_argument(
        "--lower_pad_dhw",
        type=_parse_comma_separated_ints,
        default=(1, 1, 1),
        help="Lower padding in depth,height,width order",
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
            cutlass.Float4E2M1FN,
            cutlass.Float8E4M3FN,
        ],
        default=cutlass.Float4E2M1FN,
        help="Data type for A/B input tensors",
    )
    parser.add_argument(
        "--d_dtype",
        type=cutlass.dtype,
        choices=[
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float32,
        ],
        default=cutlass.Float4E2M1FN,
        help="Output D data type. SFD is generated only when width <= 8.",
    )
    parser.add_argument(
        "--c_dtype",
        type=cutlass.dtype,
        default=None,
        help="Residual (C) input data type. Defaults to --d_dtype; must equal "
        "it since the residual shares the output's im2col TMA descriptor.",
    )
    parser.add_argument(
        "--acc_dtype",
        type=cutlass.dtype,
        choices=[cutlass.Float32, cutlass.Float16, cutlass.Int32],
        default=cutlass.Float32,
        help="Accumulator data type",
    )
    parser.add_argument(
        "--sf_dtype",
        type=cutlass.dtype,
        choices=[
            cutlass.Float8E4M3FN,
            cutlass.Float8E8M0FNU,
        ],
        default=cutlass.Float8E4M3FN,
        help="Data type for A/B/D scaling factor tensors (NVFP4 default: E4M3)",
    )
    parser.add_argument("--sf_vec_size", type=int, default=16)
    parser.add_argument(
        "--cta_tile_k",
        type=int,
        default=None,
        help="Compile-time K tile (64/128/192/256). Defaults to min(256, C).",
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
        help="Preferred cluster shape (M,N) for CLC dynamic scheduling",
    )
    parser.add_argument(
        "--fallback_cluster_shape_mn",
        type=_parse_comma_separated_ints,
        default=(1, 1),
        help="Fallback cluster shape (M,N) for CLC dynamic scheduling",
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
        help="Rasterization order of clusters",
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
    parser.add_argument(
        "--use_bias",
        action="store_true",
        default=False,
        help="Add a per-output-channel bias (D = alpha*acc + bias)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="Residual scaling (D = alpha*acc + bias + beta*residual); "
        "beta != 0 enables the residual path, beta == 0 disables it",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="identity",
        choices=sorted(EPILOGUE_ACTIVATIONS),
        help="Compile-time epilogue activation applied as "
        "D = activation(alpha*acc + bias + beta*residual). One activation per "
        "cubin (folded in as a Constexpr). Non-identity is unsupported for FP4 "
        "output.",
    )
    args = parser.parse_args()

    run(
        args.ncdhw,
        args.ktrs,
        args.stride_dhw,
        args.upper_pad_dhw,
        args.lower_pad_dhw,
        args.dil_dhw,
        args.ab_dtype,
        args.d_dtype,
        args.c_dtype,
        args.acc_dtype,
        args.sf_dtype,
        args.sf_vec_size,
        args.cta_tile_k,
        args.mma_tiler_mn,
        args.preferred_cluster_shape_mn,
        args.fallback_cluster_shape_mn,
        args.swizzle_size,
        args.raster_order,
        args.use_2cta_instrs,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.use_cold_l2,
        args.skip_ref_check,
        args.use_bias,
        args.beta,
        args.activation,
    )
    print("PASS")
