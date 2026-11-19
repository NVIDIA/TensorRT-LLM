# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""SM107 fused FC12 kernel whose die-local tile claim runs on the whole scheduler warp (Rubin).

``RubinFc12UniformClaimKernel`` is the engine kernel
``Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel``
(``rubin_fc12_kernel.py``, this family's copy of the engine kernel with die-local claiming)
with another die-local claim. The engine kernel claims a tile with a search loop that lane 0 of
the scheduler warp runs alone, dividing by runtime divisors. Built with the public CuTe DSL, that
loop costs the kernel the uniform datapath: the TMA-B and MMA warps then issue every TMA and
tcgen05 instruction from a waterfall loop. Here the whole warp runs the same search on values lane 0
broadcasts (lane 0 alone issues the atomics, as ``_claim_next_linear_idx`` does), and the claim
divides by FastDivmod divisors built on the host. The atomics are the engine claim's, in the same
order, so every CTA claims the same tiles and the arithmetic of each tile is unchanged.

The subclass overrides the three methods this touches and adds the claim:

* ``__init__``: the engine constructor; the kernel exists only for die-local claiming, so it
  refuses a build without ``die_sm_words``; it also takes ``l2_prefetch`` (below);
* ``__call__``: the engine's host launch, plus the two phases' claim divisors, built from the die
  geometry and passed as four more kernel parameters;
* ``kernel``: the engine's device kernel, with those four parameters and its four claim call sites
  on :meth:`_die_local_claim_uniform`;
* :meth:`_die_claim_divisors`, :meth:`_die_local_claim_uniform`, :meth:`_die_claim_search`,
  :meth:`_die_claim_add` and :meth:`_die_claim_max`.

``l2_prefetch`` (P, 0..8, default 0): the TMA-B warp also prefetches the B / SFB weight stream into
L2 P K-blocks ahead of its loads (``cute.prefetch`` on the loads' own TMA atoms and slices, as the
family dense GEMM does): the first P K-blocks at each FC1 tile start, then K-block ``k + P`` after
the load of ``k``; at the FC1 -> FC2 switch, before the FC1 stages drain, the first P K-blocks of
the first FC2 tile; the same for each later FC2 tile as soon as the warp reads it, and in the FC2
K loop. Every such statement sits in a ``cutlass.const_expr(self.l2_prefetch > 0)`` block: at P 0
the kernel traces the statements of today's, and at any P the loads, barriers and pipeline states
are today's (a prefetch only warms L2; it writes no shared memory and arrives on no barrier).

The FastDivmod divisions are exact while ``tiles_per_m * tile_bytes + 2 MiB + tile_bytes < 2^31``
for both phases; the launcher (``rubin_fc12_launch.fc12_die_claim``) takes this kernel only for
calls inside that bound and the engine kernel otherwise. ``__call__`` and ``kernel`` are the
engine's but for these edits and the prefetch blocks. The family tests check the bound at its edge
and that the rows this kernel writes equal the engine claim kernel's. Nothing here is registered as
a torch op.
"""

from typing import Optional, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.rubin_helpers as sm107_utils
from cutlass._mlir.dialects import math
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05.mma import CollectorOp
from cutlass.cute.testing import assert_ as runtime_assert
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

from tensorrt_llm._torch.moe.kernels.rubin import manual_mma_128dp

from .rubin_fc12_kernel import (
    DIE_CHUNK_SHIFT,
    DIE_COUNTER_STRIDE,
    END_PHASE,
    FC1_PHASE,
    FC2_PHASE,
    L2AtomicPersistentTileScheduler,
    PipelineCpAsyncUmma,
    Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel,
    atom_add_relaxed_gpu,
    atom_max_relaxed_gpu,
    blk_reduce_bf16,
    fmin,
    ld_acquire_gpu,
    nanosleep,
    red_add_release_gpu,
)

__all__ = ["L2_PREFETCH_MAX", "RubinFc12UniformClaimKernel"]

#: The deepest weight-stream L2 prefetch the kernel takes, in K-blocks ahead of the loads.
L2_PREFETCH_MAX = 8


class RubinFc12UniformClaimKernel(Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel):
    """The engine fused FC12 kernel with the die-local claim on the whole scheduler warp.

    Constructor arguments as the engine kernel's; ``die_sm_words`` is required. ``l2_prefetch``:
    K-blocks of the B / SFB weight stream prefetched into L2 ahead of the loads (0: none, today's
    kernel).
    """

    def __init__(self, *args, l2_prefetch: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.use_die_scheduler:
            raise ValueError(
                "RubinFc12UniformClaimKernel claims tiles die-locally: it needs die_sm_words"
            )
        if type(l2_prefetch) is not int or not 0 <= l2_prefetch <= L2_PREFETCH_MAX:
            raise ValueError(
                f"RubinFc12UniformClaimKernel: l2_prefetch must be an int in 0..{L2_PREFETCH_MAX}, "
                f"got {l2_prefetch!r}"
            )
        self.l2_prefetch = l2_prefetch

    @cute.jit
    def __call__(
        self,
        fc1_a: cute.Tensor,
        fc1_b: cute.Tensor,
        fc1_c: cute.Tensor,
        fc1_sfa: cute.Tensor,
        fc1_sfb: cute.Tensor,
        fc1_sfc: Optional[cute.Tensor],
        fc1_norm_const: Optional[cute.Tensor],
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        fc1_alpha: cute.Tensor,
        fc2_alpha: cute.Tensor,
        fc1_ready: cute.Tensor,
        fc1_scheduler_counter: cute.Tensor,
        fc2_scheduler_counter: cute.Tensor,
        fc2_b: cute.Tensor,
        fc2_c: cute.Tensor,
        fc2_sfb: cute.Tensor,
        permuted_idx_to_expanded_idx: cute.Tensor,
        fc2_routing_scales: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the contiguous grouped GEMM with gather operation and SiTU fusion.

        This method performs FC1 layer computation:
        1. GEMM: acc = fc1_alpha * (SFA * A[token_ids]) * (SFB * B)
        2. Gated act: SiTU, decoded from gate16/up16 accumulator groups (N/2)
        3. Optional Quant: When c_dtype is Float4E2M1FN, generates SFC and quantizes output

        Data loading:
        - A and SFA are loaded using CpAsync instructions with token-based gather
        - B and SFB are loaded using TMA instructions with multicast
        - FC1 B and SFB both use gate16/up16 N order

        Execution steps:
        1. Setup static attributes before smem/grid computation
        2. Setup TMA load/store atoms for B, SFB, and C (no TMA for A/SFA)
        3. Compute grid size with regard to hardware constraints
        4. Define shared storage for kernel
        5. Launch the kernel synchronously with warp specialization:
           - Scheduler warp: Dispatches tile information
           - CpAsync warps: Load A and SFA with gather
           - TMA warp: Load B and SFB with multicast
           - MMA warp: Perform matrix multiply-accumulate
           - Epilogue warps: Apply gated activation, optional quantization, and store results

        :param fc1_a: FC1 input A (MxKx1), gathered with the shared route mapping
        :type fc1_a: cute.Tensor
        :param fc1_b: FC1 expert weight B (NxKxL), in gate16/up16 N order
        :type fc1_b: cute.Tensor
        :param fc1_c: Quantized FC1 SiTU output and FC2 input (Mx(N/2)x1)
        :type fc1_c: cute.Tensor
        :param fc1_sfa: FC1 A scale factors, gathered with the route mapping
        :type fc1_sfa: cute.Tensor
        :param fc1_sfb: FC1 B scale factors
        :type fc1_sfb: cute.Tensor
        :param fc1_sfc: FC1 C / FC2 A scale factors (None if not quantizing)
        :type fc1_sfc: Optional[cute.Tensor]
        :param fc1_norm_const: FC1 output normalization constant
            (None if not quantizing)
        :type fc1_norm_const: Optional[cute.Tensor]
        :param tile_idx_to_expert_idx: Mapping from tile index to expert ID,
            shape (permuted_m/cta_tile_m,) where cta_tile_m is the CTA tile M size
        :type tile_idx_to_expert_idx: cute.Tensor
        :param tile_idx_to_mn_limit: Mapping from tile index to M-N dimension limit
            for boundary checking, shape (permuted_m/cta_tile_m,)
        :type tile_idx_to_mn_limit: cute.Tensor
        :param num_non_exiting_tiles: Number of valid tiles to process (valid_m/cta_tile_m), shape (1,)
        :type num_non_exiting_tiles: cute.Tensor
        :param fc1_alpha: FC1 dequantization alpha tensor for each group
        :type fc1_alpha: cute.Tensor
        :param fc2_alpha: FC2 dequantization alpha tensor for each group
        :type fc2_alpha: cute.Tensor
        :param fc1_ready: Per-M uint32 counters released by completed FC1 N tiles
        :type fc1_ready: cute.Tensor
        :param fc1_scheduler_counter: FC1 L2-atomic work-ID counter
        :param fc2_scheduler_counter: FC2 L2-atomic work-ID counter
        :param fc2_b: FC2 expert weights, shape (FC2_N, FC2_K, L)
        :param fc2_c: FC2 final output, shape (sequence_M, FC2_N, 1)
        :param fc2_sfb: FC2 expert weight scale factors
        :param permuted_idx_to_expanded_idx: Shared FC1 gather and FC2 scatter
            mapping. Each entry is ``receive_row * topk + topk_slot``.
        :param fc2_routing_scales: Per-token final routing scales
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        :param epilogue_op: Optional elementwise lambda function to apply to the output tensor
        :type epilogue_op: cutlass.Constexpr
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        """
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = fc1_a.element_type
        self.b_dtype: Type[cutlass.Numeric] = fc1_b.element_type
        self.fc1_c_dtype: Type[cutlass.Numeric] = fc1_c.element_type
        self.sf_dtype: Type[cutlass.Numeric] = fc1_sfa.element_type
        self.fc2_c_dtype: Type[cutlass.Numeric] = fc2_c.element_type
        if cutlass.const_expr(self.fc2_c_dtype != cutlass.BFloat16):
            raise TypeError("fused FC2 finalize currently requires BF16 output")
        self.a_major_mode = cutlass.tensor_utils.LayoutEnum.from_tensor(fc1_a).mma_major_mode()
        self.b_major_mode = cutlass.tensor_utils.LayoutEnum.from_tensor(fc1_b).mma_major_mode()
        self.fc1_c_layout = cutlass.tensor_utils.LayoutEnum.from_tensor(fc1_c)
        self.fc2_c_layout = cutlass.tensor_utils.LayoutEnum.from_tensor(fc2_c)
        fc2_output_n = fc2_c.shape[1]
        runtime_assert(fc2_output_n > 0, "FC2 output N must be positive")
        runtime_assert(
            fc2_output_n % self.mma_tiler[1] == 0,
            "FC2 output N must be divisible by CTA tile N",
        )

        # Note: Rubin supports mixed A/B dtypes (e.g., Float8E4M3FN x Float8E5M2)

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        # Setup sfb tensor by filling B tensor to scale factor atom layout
        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        fc1_sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(fc1_b.shape, self.sf_vec_size)
        fc1_sfb = cute.make_tensor(fc1_sfb.iterator, fc1_sfb_layout)

        # Setup sfc tensor by filling C tensor to scale factor atom layout
        self.fc1_generate_sfc = fc1_sfc is not None and fc1_norm_const is not None
        if cutlass.const_expr(self.fc1_generate_sfc):
            fc1_sfc_layout = blockscaled_utils.tile_atom_to_shape_SF(fc1_c.shape, self.sf_vec_size)
            fc1_sfc = cute.make_tensor(fc1_sfc.iterator, fc1_sfc_layout)
        else:
            raise ValueError("fused FC12 requires quantized FC1 output and SFC")

        # Both phases reuse one tiled MMA and one set of A/B/SFB buffers, so
        # their corresponding operand and scale-factor types must match.
        if cutlass.const_expr(fc1_c.element_type != self.a_dtype):
            raise TypeError("FC1 C / FC2 A must use the shared A dtype")
        if cutlass.const_expr(fc2_b.element_type != self.b_dtype):
            raise TypeError("FC2 B must use the shared B dtype")
        if cutlass.const_expr(fc1_sfb.element_type != self.sf_dtype):
            raise TypeError("FC1 SFB must use the shared scale-factor dtype")
        if cutlass.const_expr(fc1_sfc.element_type != self.sf_dtype):
            raise TypeError("FC1 SFC / FC2 SFA must use the shared scale-factor dtype")
        if cutlass.const_expr(fc2_sfb.element_type != self.sf_dtype):
            raise TypeError("FC2 SFB must use the shared scale-factor dtype")
        fc2_sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(fc2_b.shape, self.sf_vec_size)
        fc2_sfb = cute.make_tensor(fc2_sfb.iterator, fc2_sfb_layout)

        atom_layout_mnk = (1, 1, 1)

        tiled_mma = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
            atom_layout_mnk=atom_layout_mnk,
            permutation_mnk=(1, 1, 1),
        )
        tiled_mma.set(tcgen05.Field.NEGATE_A, False)
        tiled_mma.set(tcgen05.Field.NEGATE_B, False)

        # SFB's tiled MMA: the instruction N rounded up to 128.
        tiled_mma_sfb = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_sfb,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
        )
        tiled_mma_sfb.set(tcgen05.Field.NEGATE_A, False)
        tiled_mma_sfb.set(tcgen05.Field.NEGATE_B, False)

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # CpAsync128.CG gmem → sA SMEM. Four gather warps × 32
        # threads issue cp.async.cg.16B per (token row, k chunk). a_num_loads
        # (in _setup_attributes) controls CpAsync iterations per thread per k_tile.
        # The same warps gather SFA into the same pipeline stage.

        # Setup TMA load for B
        fc1_b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.fc1_b_smem_layout_staged, (None, None, None, 0))
        tma_atom_fc1_b, tma_tensor_fc1_b = cute.nvgpu.make_tiled_tma_atom_B(
            fc1_b_op,
            fc1_b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for SFB
        fc1_sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(self.fc1_sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_fc1_sfb, tma_tensor_fc1_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            fc1_sfb_op,
            fc1_sfb,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # FC2 reuses the FC1 A/B/SF stage buffers but owns separate TMA
        # descriptors and mbarriers. A/SFA and B/SFB intentionally remain two
        # independent producer pipelines so only the A side waits on FC1.
        fc2_a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        fc2_a_smem_layout = cute.select(
            self.a_smem_layout_staged,
            mode=list(range(cute.rank(self.a_smem_layout_staged) - 1)),
        )
        tma_atom_fc2_a, tma_tensor_fc2_a = cute.nvgpu.make_tiled_tma_atom_A(
            fc2_a_op,
            fc1_c,
            fc2_a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        fc2_sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        fc2_sfa_smem_layout = cute.slice_(self.fc2_sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_fc2_sfa, tma_tensor_fc2_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            fc2_sfa_op,
            fc1_sfc,
            fc2_sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        fc2_b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        fc2_b_smem_layout = cute.select(
            self.fc2_b_smem_layout_staged,
            mode=list(range(cute.rank(self.fc2_b_smem_layout_staged) - 1)),
        )
        tma_atom_fc2_b, tma_tensor_fc2_b = cute.nvgpu.make_tiled_tma_atom_B(
            fc2_b_op,
            fc2_b,
            fc2_b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        fc2_sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        fc2_sfb_smem_layout = cute.slice_(self.fc2_sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_fc2_sfb, tma_tensor_fc2_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            fc2_sfb_op,
            fc2_sfb,
            fc2_sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        fc1_b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        fc1_sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.fc1_b_tma_load_bytes = (fc1_b_copy_size + fc1_sfb_copy_size) * atom_thr_size
        self.fc2_a_tma_load_bytes = (
            cute.size_in_bytes(self.a_dtype, fc2_a_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, fc2_sfa_smem_layout)
        ) * atom_thr_size
        self.fc2_b_tma_load_bytes = (
            cute.size_in_bytes(self.b_dtype, fc2_b_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, fc2_sfb_smem_layout)
        ) * atom_thr_size

        # Setup TMA store for C
        tma_atom_fc1_c = None
        tma_tensor_fc1_c = None
        fc1_epi_smem_layout = cute.slice_(self.fc1_c_smem_layout_staged, (None, None, 0))
        tma_atom_fc1_c, tma_tensor_fc1_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            fc1_c,
            fc1_epi_smem_layout,
            self.fc1_epi_tile,
        )

        # Both phases share padded M and group dimensions but own independent
        # N task spaces. Build scheduler parameters from logical GEMM shapes;
        # the physical persistent grid is sized once from the larger FC2 task
        # space used by the initial integration.
        fc1_gemm_shape = fc1_c.shape
        fc2_gemm_shape = (
            fc1_gemm_shape[0],
            fc2_output_n,
            fc1_gemm_shape[2],
        )
        self.fc1_tile_sched_params = self._compute_tile_sched_params(
            fc1_gemm_shape,
            self.fc1_cta_tile_shape_mnk_c,
            self.cluster_shape_mn,
        )
        self.fc2_tile_sched_params = self._compute_tile_sched_params(
            fc2_gemm_shape,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
        )
        fc1_ready_expected = cutlass.Int32(cute.ceil_div(fc1_b.shape[0], self.mma_tiler[1]))
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            self.fc2_tile_sched_params, max_active_clusters
        )

        # Define shared storage for kernel.
        @cute.struct
        class SharedStorageCpasync1cta:
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 6 * self.num_tile_stage],
                1,
            ]
            # cpasync mode: A and B use separate pipelines (CpAsync A is
            # CpAsync type, B is TmaUmma type — they can't share one mbar).
            # Each mbar set holds two barriers (full + empty) per phase stage.
            fc1_a_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.fc1_num_ab_stage * 2]
            fc1_b_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.fc1_num_ab_stage * 2]
            fc2_a_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.fc2_num_ab_stage * 2]
            fc2_b_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.fc2_num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_tile_stage * 2]
            fc1_scheduler_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_scheduler_stage * 2
            ]
            fc1_scheduler_response: cute.struct.MemRange[cutlass.Int64, self.num_scheduler_stage]
            fc2_scheduler_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_scheduler_stage * 2
            ]
            fc2_scheduler_response: cute.struct.MemRange[cutlass.Int64, self.num_scheduler_stage]
            scheduler_throttle_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_scheduler_stage * 2
            ]
            fc2_meta_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.fc2_num_meta_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sEpilogue: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Uint8,
                    self.epilogue_smem_alloc_bytes,
                ],
                self.buffer_align_bytes,
            ]
            # sFC1SFA placed BEFORE sA so SFA gets a low SMEM offset: CpAsync128.CG
            # destination addr stays under the 248KB threshold (above which the
            # .CG cache-mode hint would degrade to a plain CpAsync).
            sSFAStorage: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, self.sfa_smem_alloc_bytes],
                self.buffer_align_bytes,
            ]
            sMainloop: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, self.mainloop_smem_alloc_bytes],
                self.buffer_align_bytes,
            ]
            fc2_meta_token_idx: cute.struct.MemRange[
                cutlass.Int32,
                self.cta_tile_shape_mnk[0] * self.fc2_num_meta_stage,
            ]
            fc2_meta_scale: cute.struct.MemRange[
                cutlass.Float32,
                self.cta_tile_shape_mnk[0] * self.fc2_num_meta_stage,
            ]

        self.shared_storage = SharedStorageCpasync1cta

        # Die-local claiming: per phase, the N tiles per M tile and, for the die of a tile, the
        # B base address, the B bytes per expert and per N tile (rows of one N tile x all of K).
        fc1_die_geometry = (
            cutlass.Int32(0),
            cutlass.Int64(0),
            cutlass.Int64(0),
            cutlass.Int64(0),
        )
        fc2_die_geometry = fc1_die_geometry
        # The claim's divisors are built here, on the host, so that the device divides by a
        # multiply-high and a shift instead of a division sequence (and its 64-bit call).
        fc1_die_divisors = (None, None)
        fc2_die_divisors = (None, None)
        if cutlass.const_expr(self.use_die_scheduler):
            b_bits = self.b_dtype.width
            fc1_k_bytes = cutlass.Int64(fc1_b.shape[1]) * b_bits // 8
            fc2_k_bytes = cutlass.Int64(fc2_b.shape[1]) * b_bits // 8
            fc1_die_geometry = (
                cutlass.Int32(fc1_b.shape[0]) // self.mma_tiler[1],
                cutlass.Int64(fc1_b.iterator.toint()),
                cutlass.Int64(fc1_b.shape[0]) * fc1_k_bytes,
                fc1_k_bytes * self.mma_tiler[1],
            )
            fc2_die_geometry = (
                cutlass.Int32(fc2_b.shape[0]) // self.mma_tiler[1],
                cutlass.Int64(fc2_b.iterator.toint()),
                cutlass.Int64(fc2_b.shape[0]) * fc2_k_bytes,
                fc2_k_bytes * self.mma_tiler[1],
            )
            fc1_die_divisors = self._die_claim_divisors(fc1_die_geometry)
            fc2_die_divisors = self._die_claim_divisors(fc2_die_geometry)

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            fc1_a,
            tma_atom_fc1_b,
            tma_tensor_fc1_b,
            fc1_sfa,
            tma_atom_fc1_sfb,
            tma_tensor_fc1_sfb,
            tma_atom_fc2_a,
            tma_tensor_fc2_a,
            tma_atom_fc2_sfa,
            tma_tensor_fc2_sfa,
            tma_atom_fc2_b,
            tma_tensor_fc2_b,
            tma_atom_fc2_sfb,
            tma_tensor_fc2_sfb,
            fc2_c,
            permuted_idx_to_expanded_idx,
            fc2_routing_scales,
            tma_atom_fc1_c,
            tma_tensor_fc1_c,
            fc1_sfc,
            fc1_norm_const,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            num_non_exiting_tiles,
            fc1_alpha,
            fc2_alpha,
            fc1_ready,
            fc1_scheduler_counter,
            fc2_scheduler_counter,
            fc1_ready_expected,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.fc1_b_smem_layout_staged,
            self.fc2_b_smem_layout_staged,
            self.fc1_sfa_smem_layout_staged,
            self.fc2_sfa_smem_layout_staged,
            self.fc1_sfb_smem_layout_staged,
            self.fc2_sfb_smem_layout_staged,
            self.tCtFC1SFA_layout,
            self.tCtFC2SFA_layout,
            self.tCtSFB_layout,
            self.fc1_c_smem_layout_staged,
            self.fc2_c_smem_layout_staged,
            self.fc1_epi_tile,
            self.fc2_epi_tile,
            self.fc1_tile_sched_params,
            self.fc2_tile_sched_params,
            *fc1_die_geometry,
            *fc2_die_geometry,
            *fc1_die_divisors,
            *fc2_die_divisors,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        mFC1A_mkl: cute.Tensor,
        tma_atom_fc1_b: cute.CopyAtom,
        mFC1B_nkl: cute.Tensor,
        mFC1SFA_mkl: cute.Tensor,
        tma_atom_fc1_sfb: cute.CopyAtom,
        mFC1SFB_nkl: cute.Tensor,
        tma_atom_fc2_a: cute.CopyAtom,
        mFC2A_mkl: cute.Tensor,
        tma_atom_fc2_sfa: cute.CopyAtom,
        mFC2SFA_mkl: cute.Tensor,
        tma_atom_fc2_b: cute.CopyAtom,
        mFC2B_nkl: cute.Tensor,
        tma_atom_fc2_sfb: cute.CopyAtom,
        mFC2SFB_nkl: cute.Tensor,
        mFC2C_mnl: cute.Tensor,
        permuted_idx_to_expanded_idx: cute.Tensor,
        fc2_routing_scales: cute.Tensor,
        tma_atom_fc1_c: cute.CopyAtom,
        mFC1C_mnl: cute.Tensor,
        mFC1SFC_mnl: Optional[cute.Tensor],
        fc1_norm_const: Optional[cute.Tensor],
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        fc1_alpha: cute.Tensor,
        fc2_alpha: cute.Tensor,
        fc1_ready: cute.Tensor,
        fc1_scheduler_counter: cute.Tensor,
        fc2_scheduler_counter: cute.Tensor,
        fc1_ready_expected: cutlass.Int32,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        fc1_b_smem_layout_staged: cute.ComposedLayout,
        fc2_b_smem_layout_staged: cute.ComposedLayout,
        fc1_sfa_smem_layout_staged: cute.Layout,
        fc2_sfa_smem_layout_staged: cute.Layout,
        fc1_sfb_smem_layout_staged: cute.Layout,
        fc2_sfb_smem_layout_staged: cute.Layout,
        tCtFC1SFA_layout: cute.Layout,
        tCtFC2SFA_layout: cute.Layout,
        tCtSFB_layout: cute.Layout,
        fc1_c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        fc2_c_smem_layout_staged: cute.Layout,
        fc1_epi_tile: cute.Tile,
        fc2_epi_tile: cute.Tile,
        fc1_tile_sched_params: utils.PersistentTileSchedulerParams,
        fc2_tile_sched_params: utils.PersistentTileSchedulerParams,
        fc1_die_tiles_per_m: cutlass.Int32,
        fc1_die_b_base: cutlass.Int64,
        fc1_die_expert_bytes: cutlass.Int64,
        fc1_die_tile_bytes: cutlass.Int64,
        fc2_die_tiles_per_m: cutlass.Int32,
        fc2_die_b_base: cutlass.Int64,
        fc2_die_expert_bytes: cutlass.Int64,
        fc2_die_tile_bytes: cutlass.Int64,
        fc1_die_tiles_per_m_fdd: Optional[cute.FastDivmodDivisorV2],
        fc1_die_tile_bytes_fdd: Optional[cute.FastDivmodDivisorV2],
        fc2_die_tiles_per_m_fdd: Optional[cute.FastDivmodDivisorV2],
        fc2_die_tile_bytes_fdd: Optional[cute.FastDivmodDivisorV2],
        epilogue_op: cutlass.Constexpr,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_b_warp_id:
            cpasync.prefetch_descriptor(tma_atom_fc1_b)
            cpasync.prefetch_descriptor(tma_atom_fc1_sfb)
            cpasync.prefetch_descriptor(tma_atom_fc2_b)
            cpasync.prefetch_descriptor(tma_atom_fc2_sfb)
            cpasync.prefetch_descriptor(tma_atom_fc1_c)
        if warp_idx == self.gather_a_warp_id[0]:
            cpasync.prefetch_descriptor(tma_atom_fc2_a)
            cpasync.prefetch_descriptor(tma_atom_fc2_sfa)

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        is_scheduler_leader_cta = cta_rank_in_cluster == 0
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # (fc1_a_pipeline created below alongside fc1_b_pipeline.)

        # Four gather warps issue A and SFA cp.async copies, then arrive once
        # on the shared stage. The MMA warp consumes both operands from this
        # single CpAsyncUmma pipeline.
        fc1_a_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * len(self.gather_a_warp_id),
        )
        fc1_a_pipeline = PipelineCpAsyncUmma.create(
            barrier_storage=storage.fc1_a_mbar_ptr.data_ptr(),
            num_stages=self.fc1_num_ab_stage,
            producer_group=fc1_a_pipeline_producer_group,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            defer_sync=True,
        )

        fc1_b_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_b
        fc1_b_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        fc1_b_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.fc1_b_mbar_ptr.data_ptr(),
            num_stages=self.fc1_num_ab_stage,
            producer_group=fc1_b_pipeline_producer_group,
            consumer_group=fc1_b_pipeline_consumer_group,
            tx_count=self.fc1_b_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            mcast_mode_mn=(0, 1),
            defer_sync=True,
        )

        fc2_a_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.fc2_a_mbar_ptr.data_ptr(),
            num_stages=self.fc2_num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mcast_ctas_a),
            tx_count=self.fc2_a_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            mcast_mode_mn=(1, 0),
            defer_sync=True,
        )
        fc2_b_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.fc2_b_mbar_ptr.data_ptr(),
            num_stages=self.fc2_num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mcast_ctas_b),
            tx_count=self.fc2_b_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            mcast_mode_mn=(0, 1),
            defer_sync=True,
        )

        # Pipeline Init: Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * self.threads_per_warp
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

        # Pipeline Init:Initialize tile info pipeline (barrier) and states
        tile_info_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 1,
        )
        # All four gather A/SFA warps consume every tile descriptor.
        tile_info_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_wo_sched,
        )
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.tile_info_mbar_ptr.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=tile_info_pipeline_producer_group,
            consumer_group=tile_info_pipeline_consumer_group,
        )

        # A one-stage local handshake prevents the scheduler from claiming
        # work faster than the TMA-B warp starts the current descriptor.
        scheduler_throttle_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.scheduler_throttle_mbar_ptr.data_ptr(),
            num_stages=self.num_scheduler_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_warp),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_warp),
            defer_sync=True,
        )

        # Warp 11 stages the per-row FC2 scatter metadata independently from
        # the A/B load pipelines. Only the epilogue warpgroup consumes it.
        meta_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.fc2_meta_mbar_ptr.data_ptr(),
            num_stages=self.fc2_num_meta_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_warp),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.epilog_warp_id),
            ),
        )

        # Tensor memory dealloc barrier init
        tmem = cutlass.memory.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
            arch=self.arch,
        )

        # Cluster arrive after barrier init (Rubin uses pipeline_init_arrive)
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        #
        # Setup smem tensor A/B/C/Scale
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sEpilogueBytes = storage.sEpilogue.get_tensor(
            cute.make_layout((self.epilogue_smem_alloc_bytes,))
        )
        sFC1C = cute.make_tensor(
            cute.recast_ptr(
                sEpilogueBytes.iterator,
                fc1_c_smem_layout_staged.inner,
                dtype=self.fc1_c_dtype,
            ),
            fc1_c_smem_layout_staged.outer,
        )
        sMainloopBytes = storage.sMainloop.get_tensor(
            cute.make_layout((self.mainloop_smem_alloc_bytes,))
        )
        fc2_c_smem_ptr = sEpilogueBytes.iterator
        if cutlass.const_expr(self.overlay_fc2_c_on_fc1_ab_tail):
            # No additional hand-off barrier is required. Both FC2 operand
            # producers execute the corresponding FC1 ``producer_tail`` before
            # publishing their first FC2 stage. FC2 MMA therefore cannot make
            # an accumulator available to this epilogue until every aliased
            # FC1 operand stage has been released by its consumer.
            fc2_c_smem_ptr = sMainloopBytes.iterator + self.fc2_c_smem_offset_bytes
        sFC2C = cute.make_tensor(
            cute.recast_ptr(fc2_c_smem_ptr, dtype=self.fc2_c_dtype),
            fc2_c_smem_layout_staged,
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = cute.make_tensor(
            cute.recast_ptr(
                sMainloopBytes.iterator + self.a_smem_offset_bytes,
                a_smem_layout_staged.inner,
                dtype=self.a_dtype,
            ),
            a_smem_layout_staged.outer,
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sFC1B = cute.make_tensor(
            cute.recast_ptr(
                sMainloopBytes.iterator + self.fc1_b_smem_offset_bytes,
                fc1_b_smem_layout_staged.inner,
                dtype=self.b_dtype,
            ),
            fc1_b_smem_layout_staged.outer,
        )
        sFC2B = cute.make_tensor(
            cute.recast_ptr(
                sMainloopBytes.iterator + self.fc2_b_smem_offset_bytes,
                fc2_b_smem_layout_staged.inner,
                dtype=self.b_dtype,
            ),
            fc2_b_smem_layout_staged.outer,
        )
        # SFA SMEM (linear+pad layout for the cp.async gather).
        sFC1SFA = storage.sSFAStorage.get_tensor(fc1_sfa_smem_layout_staged)
        # FC2 overlays the same raw SFA storage with the standard blockscaled
        # layout expected by TMA and the FC2 MMA descriptor.
        sFC2SFA = storage.sSFAStorage.get_tensor(fc2_sfa_smem_layout_staged)
        # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
        sFC1SFB = cute.make_tensor(
            cute.recast_ptr(
                sMainloopBytes.iterator + self.fc1_sfb_smem_offset_bytes,
                dtype=self.sf_dtype,
            ),
            fc1_sfb_smem_layout_staged,
        )
        sFC2SFB = cute.make_tensor(
            cute.recast_ptr(
                sMainloopBytes.iterator + self.fc2_sfb_smem_offset_bytes,
                dtype=self.sf_dtype,
            ),
            fc2_sfb_smem_layout_staged,
        )
        # (bidx, bidy, bidz, valid, mn_limit)
        # (m_tile, n_tile, expert, valid, mn_limit, phase)
        info_layout = cute.make_layout((6, self.num_tile_stage), stride=(1, 6))
        sInfo = storage.sInfo.get_tensor(info_layout)
        fc2_meta_layout = cute.make_layout(
            (self.cta_tile_shape_mnk[0], self.fc2_num_meta_stage),
            stride=(1, self.cta_tile_shape_mnk[0]),
        )
        sFC2MetaTokenIdx = storage.fc2_meta_token_idx.get_tensor(fc2_meta_layout)
        sFC2MetaScale = storage.fc2_meta_scale.get_tensor(fc2_meta_layout)

        #
        # Compute multicast mask for A/B buffer full
        #
        b_full_mcast_mask = None
        sfb_full_mcast_mask = None
        # FC1 uses cpasync A, while FC2 uses TMA A/SFA and may multicast both
        # along cluster N.
        a_full_mcast_mask = None
        sfa_full_mcast_mask = None

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, loopM, loopK, loopL)
        gFC1A_mkl = cute.local_tile(
            mFC1A_mkl,
            cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        # (bN, bK, loopN, loopK, loopL)
        gFC1B_nkl = cute.local_tile(
            mFC1B_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )

        # (bM, bK, RestM, RestK, RestL)
        gFC1SFA_mkl = cute.local_tile(
            mFC1SFA_mkl,
            cute.slice_(self.fc1_cta_tile_shape_mnk_sfa, (None, 0, None)),
            (None, None, None),
        )

        # (bN, bK, RestN, RestK, RestL)
        gFC1SFB_nkl = cute.local_tile(
            mFC1SFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )

        gToken_ml = cute.local_tile(
            permuted_idx_to_expanded_idx,
            cute.slice_(self.cta_tile_shape_mnk, (None, 0, 0)),
            (None,),
        )

        # (bM, bN, loopM, loopN, loopL)
        gFC1C_mnl = cute.local_tile(
            mFC1C_mnl,
            cute.slice_(self.fc1_mma_tiler_c, (None, None, 0)),
            (None, None, None),
        )
        k_tile_cnt = cutlass.Int32(cute.size(gFC1A_mkl, mode=[3]))

        gFC2A_mkl = cute.local_tile(
            mFC2A_mkl,
            cute.slice_(self.mma_tiler, (None, 0, None)),
            (None, None, None),
        )
        gFC2B_nkl = cute.local_tile(
            mFC2B_nkl,
            cute.slice_(self.mma_tiler, (0, None, None)),
            (None, None, None),
        )
        gFC2SFA_mkl = cute.local_tile(
            mFC2SFA_mkl,
            cute.slice_(self.mma_tiler, (None, 0, None)),
            (None, None, None),
        )
        gFC2SFB_nkl = cute.local_tile(
            mFC2SFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        gFC2C_mnl = cute.local_tile(
            mFC2C_mnl,
            cute.slice_(self.mma_tiler, (None, None, 0)),
            (None, None, None),
        )
        fc2_k_tile_cnt = cutlass.Int32(cute.size(gFC2A_mkl, mode=[3]))

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        # (MMA, MMA_N, MMA_K, loopN, loopK, loopL)
        tCgFC1B = thr_mma.partition_B(gFC1B_nkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgFC1SFB = thr_mma_sfb.partition_B(gFC1SFB_nkl)
        # (MMA, MMA_M, MMA_N, loopM, loopN, loopL)
        tCgFC1C = thr_mma.partition_C(gFC1C_mnl)
        tCgFC2A = thr_mma.partition_A(gFC2A_mkl)
        tCgFC2B = thr_mma.partition_B(gFC2B_nkl)
        tCgFC2SFA = thr_mma.partition_A(gFC2SFA_mkl)
        tCgFC2SFB = thr_mma_sfb.partition_B(gFC2SFB_nkl)
        tCgFC2C = thr_mma.partition_C(gFC2C_mnl)

        #
        # Partition global/shared tensor for TMA load B
        #
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
        tFC1BsB, tFC1BgB = cpasync.tma_partition(
            tma_atom_fc1_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sFC1B, 0, 3),
            cute.group_modes(tCgFC1B, 0, 3),
        )

        # TMA load SFB partition_S/D
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tFC1BsSFB, tFC1BgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_fc1_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sFC1SFB, 0, 3),
            cute.group_modes(tCgFC1SFB, 0, 3),
        )
        tFC1BsSFB = cute.filter_zeros(tFC1BsSFB)
        tFC1BgSFB = cute.filter_zeros(tFC1BgSFB)

        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tFC2AsA, tFC2AgA = cpasync.tma_partition(
            tma_atom_fc2_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgFC2A, 0, 3),
        )
        tFC2AsSFA, tFC2AgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_fc2_sfa,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sFC2SFA, 0, 3),
            cute.group_modes(tCgFC2SFA, 0, 3),
        )
        tFC2AsSFA = cute.filter_zeros(tFC2AsSFA)
        tFC2AgSFA = cute.filter_zeros(tFC2AgSFA)
        tFC2BsB, tFC2BgB = cpasync.tma_partition(
            tma_atom_fc2_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sFC2B, 0, 3),
            cute.group_modes(tCgFC2B, 0, 3),
        )
        tFC2BsSFB, tFC2BgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_fc2_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sFC2SFB, 0, 3),
            cute.group_modes(tCgFC2SFB, 0, 3),
        )
        tFC2BsSFB = cute.filter_zeros(tFC2BsSFB)
        tFC2BgSFB = cute.filter_zeros(tFC2BgSFB)

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrFC1B = tiled_mma.make_fragment_B(sFC1B)
        tCrFC2B = tiled_mma.make_fragment_B(sFC2B)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        cute.arch.griddepcontrol_wait()

        #
        # Specialized Schedule Warp
        #
        if warp_idx == self.sched_warp_id:
            scheduler_throttle_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.num_scheduler_stage,
            )
            fc1_tile_sched = L2AtomicPersistentTileScheduler.create(
                fc1_tile_sched_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                storage.fc1_scheduler_response.data_ptr(),
                fc1_scheduler_counter.iterator,
            )
            # Die-local claiming: this SM's die, and the die whose counter the CTA claims from
            # (its own until that die's tiles run out, then the other one).
            die_own = cutlass.Int32(0)
            die_queue = cutlass.Int32(0)
            if cutlass.const_expr(self.use_die_scheduler):
                die_own = self._sm_die(cute.arch.smid())
                fc1_die_num_linear = num_non_exiting_tiles[0] * fc1_die_tiles_per_m
                work_tile, die_queue = self._die_local_claim_uniform(
                    fc1_scheduler_counter,
                    die_own,
                    die_own,
                    fc1_die_num_linear,
                    fc1_die_tiles_per_m,
                    tile_idx_to_expert_idx,
                    fc1_die_b_base,
                    fc1_die_expert_bytes,
                    fc1_die_tile_bytes,
                    fc1_die_tiles_per_m_fdd,
                    fc1_die_tile_bytes_fdd,
                )
            else:
                work_tile = fc1_tile_sched.initial_work_tile_info()

            tile_info_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_tile_stage
            )

            num_non_exiting_tiles_value = num_non_exiting_tiles[0]

            is_continue = cutlass.Boolean(1)
            while work_tile.is_valid_tile and is_continue:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_m = cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape)
                if mma_tile_coord_m < num_non_exiting_tiles_value:
                    tile_info_pipeline.producer_acquire(tile_info_producer_state)
                    cur_tile_coord = work_tile.tile_idx
                    expert_idx = tile_idx_to_expert_idx[mma_tile_coord_m]
                    mn_limit = tile_idx_to_mn_limit[mma_tile_coord_m]
                    with cute.arch.elect_one():
                        sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[0]
                        sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[1]
                        sInfo[(2, tile_info_producer_state.index)] = expert_idx
                        sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                            work_tile.is_valid_tile
                        )
                        sInfo[(4, tile_info_producer_state.index)] = mn_limit
                        sInfo[(5, tile_info_producer_state.index)] = cutlass.Int32(0)
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )

                    self.sched_sync_barrier.arrive_and_wait()
                    tile_info_pipeline.producer_commit(tile_info_producer_state)
                    tile_info_producer_state.advance()

                    if is_scheduler_leader_cta:
                        scheduler_throttle_pipeline.consumer_wait(scheduler_throttle_consumer_state)
                        scheduler_throttle_pipeline.consumer_release(
                            scheduler_throttle_consumer_state
                        )
                        scheduler_throttle_consumer_state.advance()
                else:
                    is_continue = cutlass.Boolean(0)

                if is_continue:
                    if cutlass.const_expr(self.use_die_scheduler):
                        work_tile, die_queue = self._die_local_claim_uniform(
                            fc1_scheduler_counter,
                            die_queue,
                            die_own,
                            fc1_die_num_linear,
                            fc1_die_tiles_per_m,
                            tile_idx_to_expert_idx,
                            fc1_die_b_base,
                            fc1_die_expert_bytes,
                            fc1_die_tile_bytes,
                            fc1_die_tiles_per_m_fdd,
                            fc1_die_tile_bytes_fdd,
                        )
                    else:
                        work_tile = fc1_tile_sched.claim_next_work_local()

            # FC2 owns an independent work-ID stream. There is deliberately no
            # grid-wide phase fence: a fast cluster may start claiming FC2 as
            # soon as its FC1 stream ends, while FC2 A enforces per-M readiness.
            fc2_tile_sched = L2AtomicPersistentTileScheduler.create(
                fc2_tile_sched_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                storage.fc2_scheduler_response.data_ptr(),
                fc2_scheduler_counter.iterator,
            )
            if cutlass.const_expr(self.use_die_scheduler):
                fc2_die_num_linear = num_non_exiting_tiles_value * fc2_die_tiles_per_m
                fc2_work_tile, die_queue = self._die_local_claim_uniform(
                    fc2_scheduler_counter,
                    die_own,
                    die_own,
                    fc2_die_num_linear,
                    fc2_die_tiles_per_m,
                    tile_idx_to_expert_idx,
                    fc2_die_b_base,
                    fc2_die_expert_bytes,
                    fc2_die_tile_bytes,
                    fc2_die_tiles_per_m_fdd,
                    fc2_die_tile_bytes_fdd,
                )
            else:
                fc2_work_tile = fc2_tile_sched.initial_work_tile_info()
            fc2_is_continue = cutlass.Boolean(1)
            while fc2_work_tile.is_valid_tile and fc2_is_continue:
                fc2_tile_coord = fc2_work_tile.tile_idx
                fc2_m_tile = fc2_tile_coord[0] // cute.size(tiled_mma.thr_id.shape)
                if fc2_m_tile < num_non_exiting_tiles_value:
                    tile_info_pipeline.producer_acquire(tile_info_producer_state)
                    expert_idx = tile_idx_to_expert_idx[fc2_m_tile]
                    mn_limit = tile_idx_to_mn_limit[fc2_m_tile]
                    with cute.arch.elect_one():
                        sInfo[(0, tile_info_producer_state.index)] = fc2_tile_coord[0]
                        sInfo[(1, tile_info_producer_state.index)] = fc2_tile_coord[1]
                        sInfo[(2, tile_info_producer_state.index)] = expert_idx
                        sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(1)
                        sInfo[(4, tile_info_producer_state.index)] = mn_limit
                        sInfo[(5, tile_info_producer_state.index)] = cutlass.Int32(FC2_PHASE)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.sched_sync_barrier.arrive_and_wait()
                    tile_info_pipeline.producer_commit(tile_info_producer_state)
                    tile_info_producer_state.advance()

                    if is_scheduler_leader_cta:
                        scheduler_throttle_pipeline.consumer_wait(scheduler_throttle_consumer_state)
                        scheduler_throttle_pipeline.consumer_release(
                            scheduler_throttle_consumer_state
                        )
                        scheduler_throttle_consumer_state.advance()
                else:
                    fc2_is_continue = cutlass.Boolean(0)

                if fc2_is_continue:
                    if cutlass.const_expr(self.use_die_scheduler):
                        fc2_work_tile, die_queue = self._die_local_claim_uniform(
                            fc2_scheduler_counter,
                            die_queue,
                            die_own,
                            fc2_die_num_linear,
                            fc2_die_tiles_per_m,
                            tile_idx_to_expert_idx,
                            fc2_die_b_base,
                            fc2_die_expert_bytes,
                            fc2_die_tile_bytes,
                            fc2_die_tiles_per_m_fdd,
                            fc2_die_tile_bytes_fdd,
                        )
                    else:
                        fc2_work_tile = fc2_tile_sched.claim_next_work_local()

            tile_info_pipeline.producer_acquire(tile_info_producer_state)
            with cute.arch.elect_one():
                sInfo[(0, tile_info_producer_state.index)] = -1
                sInfo[(1, tile_info_producer_state.index)] = -1
                sInfo[(2, tile_info_producer_state.index)] = -1
                sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(4, tile_info_producer_state.index)] = -1
                sInfo[(5, tile_info_producer_state.index)] = cutlass.Int32(END_PHASE)
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            self.sched_sync_barrier.arrive_and_wait()
            tile_info_pipeline.producer_commit(tile_info_producer_state)
            tile_info_producer_state.advance()
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        # Gather A/SFA warps (warps 4-7). Four warps issue CpAsync128.CG for
        # every K tile; the (16, 8) thread layout covers 16 M rows by eight
        # K chunks, and padding rows are predicated off.
        if warp_idx <= self.gather_a_warp_id[-1] and warp_idx >= self.gather_a_warp_id[0]:
            cute.arch.setmaxregister_decrease(self.num_regs_gather_a_warps)
            a_atom_copy = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                mFC1A_mkl.element_type,
                num_bits_per_copy=128,
            )
            a_thread_layout = cute.make_layout((16, 8), stride=(8, 1))
            a_value_layout = cute.make_layout((1, 32), stride=(32, 1))
            a_tiled_copy = cute.make_tiled_copy_tv(
                a_atom_copy,
                a_thread_layout,
                a_value_layout,
            )
            tidx_in_warpgroup = tidx % 128

            sA_tiled = cute.make_tensor(
                sA.iterator,
                layout=cute.make_layout(
                    (
                        self.cta_tile_shape_mnk[0],
                        self.cta_tile_shape_mnk[2],
                        self.fc1_num_ab_stage,
                    ),
                    stride=(
                        self.cta_tile_shape_mnk[2],
                        1,
                        self.a_smem_bytes_per_stage * 8 // self.a_dtype.width,
                    ),
                ),
            )
            a_thr_copy = a_tiled_copy.get_slice(tidx_in_warpgroup)
            tFC1AsA_tiled = a_thr_copy.partition_D(sA_tiled)

            a_token_offset_tensor = cute.make_rmem_tensor(
                cute.make_layout((self.fc1_a_num_loads,)),
                cutlass.Int32,
            )
            a_predicate_tensor = cute.make_rmem_tensor(
                cute.make_layout((self.fc1_a_num_loads,)),
                cutlass.Boolean,
            )

            # SFA gather rides these same warps. One CpAsync128.CG per
            # thread per row per k_tile (16B = 16 FP8 SFs); sFC1SFA is plain
            # linear (M, tile_K_sf, stage). Same 128 threads as A, so
            # Each thread gathers one row at cta_tile_M=128, or two rows
            # in the inherited cta_tile_M=256 layout.
            sfa_tile_k_sf = self.cta_tile_shape_mnk[2] // self.sf_vec_size
            sfa_gather_threads = len(self.gather_a_warp_id) * self.threads_per_warp
            sfa_rows_per_thread = self.cta_tile_shape_mnk[0] // sfa_gather_threads
            sfa_atom_copy = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                mFC1SFA_mkl.element_type,
                num_bits_per_copy=128,
            )
            sFC1SFA_tiled = cute.make_tensor(
                sFC1SFA.iterator,
                layout=cute.make_layout(
                    (
                        self.cta_tile_shape_mnk[0],
                        sfa_tile_k_sf,
                        self.fc1_num_ab_stage,
                    ),
                    stride=(
                        sfa_tile_k_sf,
                        1,
                        self.cta_tile_shape_mnk[0] * sfa_tile_k_sf,
                    ),
                ),
            )

            fc1_a_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.fc1_num_ab_stage
            )
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            tile_info = cute.make_rmem_tensor((6,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
            tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
            tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
            tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
            is_valid_tile = (tile_info[3] == 1) and (tile_info[5] == FC1_PHASE)
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                gToken_ml_tile = gToken_ml[(None, tile_info[0])]
                for i in range(self.fc1_a_num_loads):
                    token_ml_tile_offset = (tidx_in_warpgroup // 8) + i * 16
                    a_token_offset_tensor[i] = gToken_ml_tile[token_ml_tile_offset]
                    a_predicate_tensor[i] = (
                        cutlass.Boolean(1)
                        if tile_info[0] * self.cta_tile_shape_mnk[0] + token_ml_tile_offset
                        < tile_info[4]
                        else cutlass.Boolean(0)
                    )
                    a_token_offset_tensor[i] = (
                        a_token_offset_tensor[i] // self.topk
                        if tile_info[0] * self.cta_tile_shape_mnk[0] + token_ml_tile_offset
                        < tile_info[4]
                        else 0
                    )

                tFC1AgA = gFC1A_mkl[(None, None, 0, None, 0)]
                A_gmem_thread_offset = cute.assume((tidx_in_warpgroup % 8) * 32, divby=32)

                fc1_a_producer_state.reset_count()
                peek_fc1_a_empty_status = cutlass.Boolean(1)
                if fc1_a_producer_state.count < k_tile_cnt:
                    peek_fc1_a_empty_status = fc1_a_pipeline.producer_try_acquire(
                        fc1_a_producer_state
                    )

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    fc1_a_pipeline.producer_acquire(fc1_a_producer_state, peek_fc1_a_empty_status)

                    tFC1AgA_ktile = tFC1AgA[(None, None, fc1_a_producer_state.count)]
                    tFC1AsA_ktile = tFC1AsA_tiled[(None, None, None, fc1_a_producer_state.index)]

                    for i in range(self.fc1_a_num_loads):
                        A_gmem_slice_offset = A_gmem_thread_offset + cute.assume(
                            a_token_offset_tensor[i] * tFC1AgA_ktile.layout[0].stride,
                            divby=32,
                        )
                        A_gmem_slice_offset = cute.assume(A_gmem_slice_offset, divby=32)
                        tFC1AgA_slice_ptr = tFC1AgA_ktile.iterator + A_gmem_slice_offset
                        tFC1AgA_slice = cute.make_tensor(
                            tFC1AgA_slice_ptr, layout=cute.make_layout((32,))
                        )
                        tFC1AsA_slice = cute.make_tensor(
                            tFC1AsA_ktile[(None, i, None)].iterator,
                            layout=cute.make_layout((32,)),
                        )
                        a_predicate_slice = cute.make_rmem_tensor(
                            cute.make_layout((1,)), cutlass.Boolean
                        )
                        a_predicate_slice[0] = a_predicate_tensor[i]
                        cute.copy_atom_call(
                            a_atom_copy,
                            tFC1AgA_slice,
                            tFC1AsA_slice,
                            pred=a_predicate_slice,
                        )

                    # SFA for the same stage, issued by the same threads so
                    # the single producer_commit below covers A and SFA.
                    for r in cutlass.range_constexpr(sfa_rows_per_thread):
                        row_in_cta = tidx_in_warpgroup + r * sfa_gather_threads
                        tok = gToken_ml_tile[row_in_cta]
                        sfa_row_id = cutlass.Int32(-1) if tok == -1 else tok // self.topk
                        sfa_pred = cute.make_rmem_tensor(cute.make_layout((1,)), cutlass.Boolean)
                        sfa_pred[0] = (
                            cutlass.Boolean(1)
                            if (
                                tile_info[0] * self.cta_tile_shape_mnk[0] + row_in_cta
                                < tile_info[4]
                            )
                            else cutlass.Boolean(0)
                        )
                        tFC1AgSFA_row = gFC1SFA_mkl[(sfa_row_id, None, 0, None, 0)]
                        tFC1AgSFA_ktile = tFC1AgSFA_row[(None, k_tile)]
                        tFC1AgSFA_slice = cute.make_tensor(
                            tFC1AgSFA_ktile.iterator,
                            layout=cute.make_layout((sfa_tile_k_sf,)),
                        )
                        tFC1AsSFA_slice = cute.make_tensor(
                            sFC1SFA_tiled[(row_in_cta, None, fc1_a_producer_state.index)].iterator,
                            cute.make_layout((sfa_tile_k_sf,)),
                        )
                        cute.copy_atom_call(
                            sfa_atom_copy,
                            tFC1AgSFA_slice,
                            tFC1AsSFA_slice,
                            pred=sfa_pred,
                        )

                    fc1_a_pipeline.producer_commit(fc1_a_producer_state)

                    fc1_a_producer_state.advance()
                    peek_fc1_a_empty_status = cutlass.Boolean(1)
                    if fc1_a_producer_state.count < k_tile_cnt:
                        peek_fc1_a_empty_status = fc1_a_pipeline.producer_try_acquire(
                            fc1_a_producer_state
                        )

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
                tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
                tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
                tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
                is_valid_tile = (tile_info[3] == 1) and (tile_info[5] == FC1_PHASE)
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            fc1_a_pipeline.producer_tail(fc1_a_producer_state)

            fc2_a_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.fc2_num_ab_stage
            )

            # The descriptor that ended the FC1 loop is either the first
            # FC2 task or END and has already been released. Only warp 4,
            # which becomes the FC2 A/SFA producer, acquire-waits for the
            # corresponding M tile. Warps 5-7 stay independent so the B
            # side never inherits this dependency.
            is_valid_fc2_tile = tile_info[3] == 1
            while is_valid_fc2_tile:
                if warp_idx == self.gather_a_warp_id[0]:
                    fc2_m_tile = tile_info[0] // cute.size(tiled_mma.thr_id.shape)
                    ready = ld_acquire_gpu(fc1_ready, fc2_m_tile)
                    while ready != fc1_ready_expected:
                        nanosleep(cutlass.Int32(32))
                        ready = ld_acquire_gpu(fc1_ready, fc2_m_tile)

                    fc2_a_gmem = tFC2AgA[(None, fc2_m_tile, None, 0)]
                    fc2_sfa_gmem = tFC2AgSFA[(None, fc2_m_tile, None, 0)]
                    fc2_a_producer_state.reset_count()
                    peek_fc2_a_empty = cutlass.Boolean(1)
                    if fc2_a_producer_state.count < fc2_k_tile_cnt:
                        peek_fc2_a_empty = fc2_a_pipeline.producer_try_acquire(fc2_a_producer_state)
                    for _ in cutlass.range(0, fc2_k_tile_cnt, 1, unroll=1):
                        fc2_a_pipeline.producer_acquire(fc2_a_producer_state, peek_fc2_a_empty)
                        fc2_a_barrier = fc2_a_pipeline.producer_get_barrier(fc2_a_producer_state)
                        cute.copy(
                            tma_atom_fc2_a,
                            fc2_a_gmem[(None, fc2_a_producer_state.count)],
                            tFC2AsA[(None, fc2_a_producer_state.index)],
                            tma_bar_ptr=fc2_a_barrier,
                            mcast_mask=a_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_fc2_sfa,
                            fc2_sfa_gmem[(None, fc2_a_producer_state.count)],
                            tFC2AsSFA[(None, fc2_a_producer_state.index)],
                            tma_bar_ptr=fc2_a_barrier,
                            mcast_mask=sfa_full_mcast_mask,
                        )
                        fc2_a_producer_state.advance()
                        peek_fc2_a_empty = cutlass.Boolean(1)
                        if fc2_a_producer_state.count < fc2_k_tile_cnt:
                            peek_fc2_a_empty = fc2_a_pipeline.producer_try_acquire(
                                fc2_a_producer_state
                            )

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
                tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
                tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
                is_valid_fc2_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            if warp_idx == self.gather_a_warp_id[0]:
                fc2_a_pipeline.producer_tail(fc2_a_producer_state)

        # MMA-group warps (8-11: mma / tma_b / sched / FC2 metadata).
        # Left alone they sit at the launch default of 96 regs/thread. At 12 warps the CTA budget is 96*384 = 36864, so
        # holding them at 96 would leave gather only 24 regs (the PTX minimum)
        # once the epilogue takes its tuned 168. Dropping this group to 48 is
        # what buys gather a workable 72. Budget: 128*(168+72+48) = 36864 exactly.
        if warp_idx >= self.mma_warp_id and warp_idx <= self.sync_transform_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma_group_warps)

        # Metadata warp (warp 11): loads the FC2 scatter metadata.
        if warp_idx == self.sync_transform_warp_id:
            meta_lane = tidx % self.threads_per_warp
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
            meta_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.fc2_num_meta_stage
            )
            tile_info = cute.make_rmem_tensor((6,), cutlass.Int32)

            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
            tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
            tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
            tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
            tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            # FC1 has no warp-11 work. Consume its descriptors so the FC2
            # metadata loop starts with the first FC2 descriptor resident.
            is_valid_fc1_tile = (tile_info[3] == 1) and (tile_info[5] == FC1_PHASE)
            while is_valid_fc1_tile:
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
                tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
                tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
                tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
                tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
                is_valid_fc1_tile = (tile_info[3] == 1) and (tile_info[5] == FC1_PHASE)
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            is_valid_fc2_tile = (tile_info[3] == 1) and (tile_info[5] == FC2_PHASE)
            while is_valid_fc2_tile:
                tile_m_start = tile_info[0] * self.cta_tile_shape_mnk[0]
                expert_idx = tile_info[2]
                alpha_val = fc2_alpha[expert_idx]

                meta_pipeline.producer_acquire(meta_producer_state)
                meta_stage = meta_producer_state.index
                for j in cutlass.range(
                    self.cta_tile_shape_mnk[0] // self.threads_per_warp,
                    unroll_full=True,
                ):
                    row = meta_lane + j * self.threads_per_warp
                    permuted_row = tile_m_start + row
                    expanded_idx = permuted_idx_to_expanded_idx[permuted_row]
                    safe_idx = cutlass.max(expanded_idx, cutlass.Int32(0))
                    token_idx = safe_idx // self.topk
                    topk_idx = safe_idx % self.topk
                    is_valid_row = cutlass.Int32(permuted_row < tile_info[4])
                    gather_token = token_idx * is_valid_row
                    token_scale = fc2_routing_scales[(gather_token, topk_idx)]
                    sFC2MetaTokenIdx[(row, meta_stage)] = token_idx
                    sFC2MetaScale[(row, meta_stage)] = alpha_val * token_scale
                cute.arch.fence_proxy("async.shared", space="cta")
                meta_pipeline.producer_commit(meta_producer_state)
                meta_producer_state.advance()

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
                tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
                tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
                tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
                tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
                is_valid_fc2_tile = (tile_info[3] == 1) and (tile_info[5] == FC2_PHASE)
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            meta_pipeline.producer_tail(meta_producer_state)

        # TMA B/SFB load warp (warp 9). Loads B/SFB GMEM → SMEM with multicast.
        if warp_idx == self.tma_b_warp_id:
            fc1_b_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.fc1_num_ab_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
            scheduler_throttle_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.num_scheduler_stage,
            )

            # Get the first tile info
            tile_info = cute.make_rmem_tensor((6,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
            tile_info[1] = sInfo[(1, tile_info_consumer_state.index)]
            tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
            tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
            tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
            tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
            is_valid_tile = (tile_info[3] == 1) and (tile_info[5] == FC1_PHASE)
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                if is_scheduler_leader_cta:
                    scheduler_throttle_pipeline.producer_acquire(scheduler_throttle_producer_state)
                    scheduler_throttle_pipeline.producer_commit(scheduler_throttle_producer_state)
                    scheduler_throttle_producer_state.advance()
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), loopK)
                tFC1BgB_slice = tFC1BgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

                # ((atom_v, rest_v), RestK)
                tFC1BgSFB_slice = tFC1BgSFB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                # L2 prefetch: the tile's first K-blocks of B / SFB.
                if cutlass.const_expr(self.l2_prefetch > 0):
                    self._l2_prefetch_head(
                        tma_atom_fc1_b,
                        tma_atom_fc1_sfb,
                        tFC1BgB_slice,
                        tFC1BgSFB_slice,
                        k_tile_cnt,
                    )

                # Peek (try_wait) B buffer empty for k_tile = prefetch_k_tile_cnt
                fc1_b_producer_state.reset_count()
                peek_fc1_b_empty_status = cutlass.Boolean(1)
                if fc1_b_producer_state.count < k_tile_cnt:
                    peek_fc1_b_empty_status = fc1_b_pipeline.producer_try_acquire(
                        fc1_b_producer_state
                    )
                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    fc1_b_pipeline.producer_acquire(fc1_b_producer_state, peek_fc1_b_empty_status)

                    tFC1BgB_k = tFC1BgB_slice[(None, fc1_b_producer_state.count)]
                    tFC1BgSFB_k = tFC1BgSFB_slice[(None, fc1_b_producer_state.count)]
                    tFC1BsB_pipe = tFC1BsB[(None, fc1_b_producer_state.index)]
                    tFC1BsSFB_pipe = tFC1BsSFB[(None, fc1_b_producer_state.index)]

                    tma_bar = fc1_b_pipeline.producer_get_barrier(fc1_b_producer_state)

                    # TMA load B
                    cute.copy(
                        tma_atom_fc1_b,
                        tFC1BgB_k,
                        tFC1BsB_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=b_full_mcast_mask,
                    )

                    # TMA load SFB
                    cute.copy(
                        tma_atom_fc1_sfb,
                        tFC1BgSFB_k,
                        tFC1BsSFB_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=sfb_full_mcast_mask,
                    )

                    # L2 prefetch: K-block k + P of B / SFB, after the loads of k.
                    if cutlass.const_expr(self.l2_prefetch > 0):
                        self._l2_prefetch_ahead(
                            tma_atom_fc1_b,
                            tma_atom_fc1_sfb,
                            tFC1BgB_slice,
                            tFC1BgSFB_slice,
                            fc1_b_producer_state.count,
                            k_tile_cnt,
                        )

                    # Peek (try_wait) B buffer empty for k_tile + 1
                    fc1_b_producer_state.advance()
                    peek_fc1_b_empty_status = cutlass.Boolean(1)
                    if fc1_b_producer_state.count < k_tile_cnt:
                        peek_fc1_b_empty_status = fc1_b_pipeline.producer_try_acquire(
                            fc1_b_producer_state
                        )

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
                tile_info[1] = sInfo[(1, tile_info_consumer_state.index)]
                tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
                tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
                tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
                tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
                is_valid_tile = (tile_info[3] == 1) and (tile_info[5] == FC1_PHASE)
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            # L2 prefetch: the first FC2 tile's first K-blocks, before the FC1 stages drain.
            if cutlass.const_expr(self.l2_prefetch > 0):
                if tile_info[3] == 1:
                    self._l2_prefetch_fc2_head(
                        tma_atom_fc2_b,
                        tma_atom_fc2_sfb,
                        tFC2BgB,
                        tFC2BgSFB,
                        tile_info[1],
                        tile_info[2],
                        fc2_k_tile_cnt,
                    )
            fc1_b_pipeline.producer_tail(fc1_b_producer_state)
            fc2_b_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.fc2_num_ab_stage
            )
            is_valid_fc2_tile = tile_info[3] == 1
            while is_valid_fc2_tile:
                if is_scheduler_leader_cta:
                    scheduler_throttle_pipeline.producer_acquire(scheduler_throttle_producer_state)
                    scheduler_throttle_pipeline.producer_commit(scheduler_throttle_producer_state)
                    scheduler_throttle_producer_state.advance()
                fc2_n_tile = tile_info[1]
                fc2_expert = tile_info[2]
                fc2_b_gmem = tFC2BgB[(None, fc2_n_tile, None, fc2_expert)]
                fc2_sfb_gmem = tFC2BgSFB[(None, fc2_n_tile, None, fc2_expert)]
                fc2_b_producer_state.reset_count()
                peek_fc2_b_empty = cutlass.Boolean(1)
                if fc2_b_producer_state.count < fc2_k_tile_cnt:
                    peek_fc2_b_empty = fc2_b_pipeline.producer_try_acquire(fc2_b_producer_state)
                for _ in cutlass.range(0, fc2_k_tile_cnt, 1, unroll=1):
                    fc2_b_pipeline.producer_acquire(fc2_b_producer_state, peek_fc2_b_empty)
                    fc2_b_barrier = fc2_b_pipeline.producer_get_barrier(fc2_b_producer_state)
                    cute.copy(
                        tma_atom_fc2_b,
                        fc2_b_gmem[(None, fc2_b_producer_state.count)],
                        tFC2BsB[(None, fc2_b_producer_state.index)],
                        tma_bar_ptr=fc2_b_barrier,
                        mcast_mask=b_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_fc2_sfb,
                        fc2_sfb_gmem[(None, fc2_b_producer_state.count)],
                        tFC2BsSFB[(None, fc2_b_producer_state.index)],
                        tma_bar_ptr=fc2_b_barrier,
                        mcast_mask=sfb_full_mcast_mask,
                    )
                    # L2 prefetch: K-block k + P of B / SFB, after the loads of k.
                    if cutlass.const_expr(self.l2_prefetch > 0):
                        self._l2_prefetch_ahead(
                            tma_atom_fc2_b,
                            tma_atom_fc2_sfb,
                            fc2_b_gmem,
                            fc2_sfb_gmem,
                            fc2_b_producer_state.count,
                            fc2_k_tile_cnt,
                        )
                    fc2_b_producer_state.advance()
                    peek_fc2_b_empty = cutlass.Boolean(1)
                    if fc2_b_producer_state.count < fc2_k_tile_cnt:
                        peek_fc2_b_empty = fc2_b_pipeline.producer_try_acquire(fc2_b_producer_state)

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                tile_info[1] = sInfo[(1, tile_info_consumer_state.index)]
                tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
                tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
                tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
                is_valid_fc2_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
                # L2 prefetch: the next FC2 tile's first K-blocks, as soon as it is known.
                if cutlass.const_expr(self.l2_prefetch > 0):
                    if is_valid_fc2_tile:
                        self._l2_prefetch_fc2_head(
                            tma_atom_fc2_b,
                            tma_atom_fc2_sfb,
                            tFC2BgB,
                            tFC2BgSFB,
                            tile_info[1],
                            tile_info[2],
                            fc2_k_tile_cnt,
                        )
            fc2_b_pipeline.producer_tail(fc2_b_producer_state)
            if is_scheduler_leader_cta:
                scheduler_throttle_pipeline.producer_tail(scheduler_throttle_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # SFA 128dp_Unique TMEM (single buffer). Written 1:1 each k_tile by
            # a Cp128x128b UTCCP from gathered sFC1SFA smem (issued in this warp,
            # right before the MMA — like SFB). tCtFC1SFA_layout == the 128dp
            # (128, nsf):(1<<18, 1) layout passed from the host.
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtFC1SFA_128 = cute.make_tensor(sfa_tmem_ptr, tCtFC1SFA_layout)
            # FC1 and FC2 reuse the same SFA TMEM allocation, but interpret it
            # through different layouts. FC1 uses the 128-datapath Unique
            # layout required by its gathered activation scales, while FC2
            # consumes the standard block-scaled layout used by TMA-loaded
            # intermediate scales.
            tCtFC2SFA = cute.make_tensor(sfa_tmem_ptr, tCtFC2SFA_layout)
            # Per-k-block SFA base-address view for the manual MMA: k-block k
            # reads 128dp starting at sf-column k * sfa_sf_per_kblock. Trivial
            # (V=1, MN=1) modes — only the base address per k-block matters;
            # HW expands the 128 datapaths when idesc bit26=1.
            tCtFC1SFA_mma = cute.make_tensor(
                sfa_tmem_ptr,
                cute.make_layout(
                    (1, 1, self.fc1_sfa_nsf // self.fc1_sfa_sf_per_kblock),
                    stride=(0, 0, self.fc1_sfa_sf_per_kblock),
                ),
            )

            # Make SFB tmem tensor (using precomputed layout)
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            # UTCCP s2t bundles: SFA (Cp128x128b, 128dp) + SFB (Cp4x32x128b).
            sfa_s2t_bundle = self._fc1_sfa_s2t_copy_and_partition_128dp(sFC1SFA, tCtFC1SFA_128)
            fc2_sfa_s2t_bundle = self._mainloop_s2t_copy_and_partition(sFC2SFA, tCtFC2SFA)
            fc1_sfb_s2t_bundle = self._mainloop_s2t_copy_and_partition(sFC1SFB, tCtSFB)
            fc2_sfb_s2t_bundle = self._mainloop_s2t_copy_and_partition(sFC2SFB, tCtSFB)

            fc1_a_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.fc1_num_ab_stage
            )
            fc1_b_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.fc1_num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info from pipeline (scheduler has filtered out tiles >= num_non_exiting_tiles)
            tile_info = cute.make_rmem_tensor((6,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
            tile_info[1] = sInfo[(1, tile_info_consumer_state.index)]
            tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
            tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
            tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
            tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
            is_valid_tile = (tile_info[3] == 1) and (tile_info[5] == FC1_PHASE)
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                # Peek (try_wait) A / B / SFA buffer full for k_tile = 0.
                fc1_a_consumer_state.reset_count()
                peek_fc1_a_full_status = cutlass.Boolean(1)
                if fc1_a_consumer_state.count < k_tile_cnt:
                    peek_fc1_a_full_status = fc1_a_pipeline.consumer_try_wait(fc1_a_consumer_state)
                fc1_b_consumer_state.reset_count()
                peek_fc1_b_full_status = cutlass.Boolean(1)
                if fc1_b_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_fc1_b_full_status = fc1_b_pipeline.consumer_try_wait(fc1_b_consumer_state)

                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )

                # Accumulator stage.
                acc_stage_index = acc_producer_state.index

                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)
                #
                # Mma mainloop
                #

                for k_tile in cutlass.range(k_tile_cnt):
                    # Set tensor memory buffer for current tile
                    # (MMA, MMA_M, MMA_N)

                    if is_leader_cta:
                        # Wait for A / B / SFA buffer full.
                        # A and SFA share fc1_a_pipeline; B has an independent
                        # TMA pipeline.
                        fc1_a_pipeline.consumer_wait(fc1_a_consumer_state, peek_fc1_a_full_status)
                        fc1_b_pipeline.consumer_wait(fc1_b_consumer_state, peek_fc1_b_full_status)
                        a_stage_idx = fc1_a_consumer_state.index
                        b_stage_idx = fc1_b_consumer_state.index
                        sfa_stage_idx = a_stage_idx

                        # SFA 128dp UTCCP (Cp128x128b, smem → 128dp TMEM) +
                        # SFB UTCCP (Cp4x32x128b). Both land before the MMA.
                        sfa_s2t_stage_coord = (None, None, None, sfa_stage_idx)
                        cute.copy(
                            sfa_s2t_bundle.tiled_copy,
                            sfa_s2t_bundle.sSF_compact[sfa_s2t_stage_coord],
                            sfa_s2t_bundle.tSF_compact,
                        )
                        self._mainloop_s2t_copies(b_stage_idx, fc1_sfb_s2t_bundle)

                        num_kblocks = cute.size(tCrA, mode=[2])

                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            a_kblock_coord = (None, None, kblock_idx, a_stage_idx)
                            b_kblock_coord = (None, None, kblock_idx, b_stage_idx)
                            sf_kblock_coord = (None, None, kblock_idx)

                            # Hand-encoded UTCOMMA (replaces cute.gemm) so
                            # idesc bit 26 (SFA layout) is under our control.
                            manual_mma_128dp.issue_manual_block_scaled_mma_atom(
                                acc_frag=tCtAcc,
                                a_frag=tCrA[a_kblock_coord],
                                sfa_frag=tCtFC1SFA_mma[sf_kblock_coord],
                                b_frag=tCrFC1B[b_kblock_coord],
                                sfb_frag=tCtSFB[sf_kblock_coord],
                                static_idesc_base=self.static_idesc_base,
                                accumulate=(k_tile != 0 or kblock_idx != 0),
                                cta_group=1,
                            )

                        # Release A/SFA and B buffer-empty barriers.
                        fc1_a_pipeline.consumer_release(fc1_a_consumer_state)
                        fc1_b_pipeline.consumer_release(fc1_b_consumer_state)

                    # Peek (try_wait) A / B / SFA buffer full for k_tile + 1.
                    fc1_a_consumer_state.advance()
                    peek_fc1_a_full_status = cutlass.Boolean(1)
                    if fc1_a_consumer_state.count < k_tile_cnt:
                        peek_fc1_a_full_status = fc1_a_pipeline.consumer_try_wait(
                            fc1_a_consumer_state
                        )
                    fc1_b_consumer_state.advance()
                    peek_fc1_b_full_status = cutlass.Boolean(1)
                    if fc1_b_consumer_state.count < k_tile_cnt and is_leader_cta:
                        peek_fc1_b_full_status = fc1_b_pipeline.consumer_try_wait(
                            fc1_b_consumer_state
                        )

                #
                # Async arrive accumulator buffer full(each kblock)
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)

                # Peek (try_wait) Acc buffer empty for k_tile = k_tile + 1
                acc_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
                tile_info[1] = sInfo[(1, tile_info_consumer_state.index)]
                tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
                tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
                tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
                tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
                is_valid_tile = (tile_info[3] == 1) and (tile_info[5] == FC1_PHASE)
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            acc_pipeline.producer_tail(acc_producer_state)
            fc2_a_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.fc2_num_ab_stage
            )
            fc2_b_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.fc2_num_ab_stage
            )
            is_valid_fc2_tile = tile_info[3] == 1
            while is_valid_fc2_tile:
                acc_stage_index = acc_producer_state.index
                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                fc2_a_consumer_state.reset_count()
                fc2_b_consumer_state.reset_count()
                peek_fc2_a_full = cutlass.Boolean(1)
                peek_fc2_b_full = cutlass.Boolean(1)
                if fc2_a_consumer_state.count < fc2_k_tile_cnt and is_leader_cta:
                    peek_fc2_a_full = fc2_a_pipeline.consumer_try_wait(fc2_a_consumer_state)
                    peek_fc2_b_full = fc2_b_pipeline.consumer_try_wait(fc2_b_consumer_state)

                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                for k_tile in cutlass.range(0, fc2_k_tile_cnt, 1, unroll=1):
                    if is_leader_cta:
                        fc2_a_pipeline.consumer_wait(fc2_a_consumer_state, peek_fc2_a_full)
                        fc2_b_pipeline.consumer_wait(fc2_b_consumer_state, peek_fc2_b_full)

                        a_stage_idx = fc2_a_consumer_state.index
                        b_stage_idx = fc2_b_consumer_state.index
                        fc2_sfa_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            a_stage_idx,
                        )
                        cute.copy(
                            fc2_sfa_s2t_bundle.tiled_copy,
                            fc2_sfa_s2t_bundle.sSF_compact[fc2_sfa_stage_coord],
                            fc2_sfa_s2t_bundle.tSF_compact,
                        )
                        self._mainloop_s2t_copies(b_stage_idx, fc2_sfb_s2t_bundle)

                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            a_kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                a_stage_idx,
                            )
                            b_kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                b_stage_idx,
                            )
                            sf_kblock_coord = (None, None, kblock_idx)

                            tiled_mma.set(
                                tcgen05.Field.ACCUMULATE,
                                k_tile != 0 or kblock_idx != 0,
                            )
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                [
                                    tCrA[a_kblock_coord],
                                    tCtFC2SFA[sf_kblock_coord],
                                ],
                                [
                                    tCrFC2B[b_kblock_coord],
                                    tCtSFB[sf_kblock_coord],
                                ],
                                tCtAcc,
                            )

                        fc2_a_pipeline.consumer_release(fc2_a_consumer_state)
                        fc2_b_pipeline.consumer_release(fc2_b_consumer_state)
                    fc2_a_consumer_state.advance()
                    fc2_b_consumer_state.advance()
                    peek_fc2_a_full = cutlass.Boolean(1)
                    peek_fc2_b_full = cutlass.Boolean(1)
                    if fc2_a_consumer_state.count < fc2_k_tile_cnt and is_leader_cta:
                        peek_fc2_a_full = fc2_a_pipeline.consumer_try_wait(fc2_a_consumer_state)
                        peek_fc2_b_full = fc2_b_pipeline.consumer_try_wait(fc2_b_consumer_state)

                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
                tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
                is_valid_fc2_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

        #
        # Specialized epilogue warps
        #
        if warp_idx <= self.epilog_warp_id[-1]:
            # Register reconfig: epilogue needs many regs for SiTU
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
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            # Epilogue partition: transform both accumulator and C layout.
            # transform_partitioned_tensor_layout merges (MMA_ATOM, MMA_M) → flat M.
            tCtAcc_transformed = transform_partitioned_tensor_layout(tCtAcc_base)
            tCgFC1C_for_epi = transform_partitioned_tensor_layout(tCgFC1C)
            tCgFC2C_for_epi = transform_partitioned_tensor_layout(tCgFC2C)

            epi_tidx = tidx % 128
            (
                tiled_copy_t2r,
                tFC1TR_tAcc_base,
                tFC1TR_rPackedLo,
                tFC1TR_rPackedHi,
            ) = self.fc1_epilogue_tmem_copy_and_partition(
                epi_tidx,
                tCtAcc_transformed,
                tCgFC1C_for_epi,
                fc1_epi_tile,
            )

            (
                fc2_tiled_copy_t2r,
                tFC2TR_tAcc_base,
                tFC2TR_rAcc,
            ) = self.fc2_epilogue_tmem_copy_and_partition(
                epi_tidx,
                tCtAcc_transformed,
                tCgFC2C_for_epi,
                fc2_epi_tile,
            )
            tFC2TR_rC = cute.make_rmem_tensor(tFC2TR_rAcc.shape, self.fc2_c_dtype)
            fc2_r2s_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.fc2_c_dtype)
            fc2_tiled_copy_r2s = cute.make_tiled_copy_D(fc2_r2s_atom, fc2_tiled_copy_t2r)
            fc2_thr_copy_r2s = fc2_tiled_copy_r2s.get_slice(epi_tidx)
            tFC2RS_sC = fc2_thr_copy_r2s.partition_D(sFC2C)
            tFC2RS_rC = fc2_tiled_copy_r2s.retile(tFC2TR_rC)

            tFC1TR_rC = None
            tiled_copy_r2s = None
            tFC1RS_rC = None
            tFC1RS_sC = None
            bSG_sC = None
            bSG_gC_partitioned = None
            # One logical output fragment has the same shape as one packed
            # load fragment: two packed chunks become gate/up, then one C.
            tFC1TR_rC = cute.make_rmem_tensor(tFC1TR_rPackedLo.shape, self.fc1_c_dtype)
            tiled_copy_r2s, tFC1RS_rC, tFC1RS_sC = self.fc1_epilogue_smem_copy_and_partition(
                tiled_copy_t2r, tFC1TR_rC, epi_tidx, sFC1C
            )
            (
                tma_atom_fc1_c,
                bSG_sC,
                bSG_gC_partitioned,
            ) = self.fc1_epilogue_gmem_copy_and_partition(
                epi_tidx, tma_atom_fc1_c, tCgFC1C_for_epi, fc1_epi_tile, sFC1C
            )

            if cutlass.const_expr(self.fc1_generate_sfc):
                norm_const = fc1_norm_const[0]
                # (EPI_TILE_M, EPI_TILE_N, RestM, RestN, RestL)
                gFC1SFC_mnl = cute.local_tile(mFC1SFC_mnl, fc1_epi_tile, (None, None, None))

                thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
                # (T2R, T2R_M, T2R_N, RestM, RestN, RestL)
                tCgFC1SFC_mnl = thr_copy_t2r.partition_D(gFC1SFC_mnl)
                tCgFC1SFC_mnl = cute.filter_zeros(tCgFC1SFC_mnl)
                # (T2R, T2R_M, T2R_N)
                tCrFC1SFC = cute.make_rmem_tensor(
                    tCgFC1SFC_mnl[(None, None, None, 0, 0, 0)].layout, self.sf_dtype
                )
                tCrFC1SFC_pvscale = cute.make_rmem_tensor_like(tCrFC1SFC, cutlass.Float32)

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            meta_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.fc2_num_meta_stage
            )

            fc1_c_pipeline = None
            # Threads/warps participating in tma store pipeline
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            fc1_c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.fc1_num_c_stage,
                producer_group=c_producer_group,
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info
            tile_info = cute.make_rmem_tensor((6,), cutlass.Int32)

            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
            tile_info[1] = sInfo[(1, tile_info_consumer_state.index)]
            tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
            tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
            tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
            tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
            is_valid_tile = (tile_info[3] == 1) and (tile_info[5] == FC1_PHASE)
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            num_prev_subtiles = cutlass.Int32(0)
            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                #
                # Get alpha for current group
                #

                expert_idx = mma_tile_coord_mnl[2]
                alpha_val = fc1_alpha[expert_idx]

                #
                # Slice to per mma tile index
                #
                bSG_gC = None
                # ((ATOM_V, REST_V), EPI_M, EPI_N)
                bSG_gC = bSG_gC_partitioned[
                    (
                        None,
                        None,
                        None,
                        mma_tile_coord_mnl[0],
                        mma_tile_coord_mnl[1],
                        0,
                    )
                ]

                # Get accumulator stage index.
                acc_stage_index = acc_consumer_state.index

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tFC1TR_tAcc = tFC1TR_tAcc_base[(None, None, None, None, None, acc_stage_index)]

                if cutlass.const_expr(self.fc1_generate_sfc):
                    # (T2R, T2R_M, T2R_N, RestM, RestN)
                    tCgFC1SFC_mn = tCgFC1SFC_mnl[
                        (
                            None,
                            None,
                            None,
                            None,
                            None,
                            0,
                        )
                    ]

                #
                acc_pipeline.consumer_wait(acc_consumer_state)

                # Each N64 output subtile consumes two consecutive N64
                # accumulator chunks, each [gate16, up16, gate16, up16];
                # the decoded up/gate feed SiTU.
                # tFC1TR_tAcc: (T2R, T2R_M, T2R_N, EPI_M, EPI_N,
                # STAGE), sliced on STAGE. bSG_gC: ((ATOM_V, REST_V),
                # EPI_M, EPI_N, loopM, loopN, loopL).
                epi_m_cnt = cute.size(tFC1TR_tAcc.shape, mode=[3])
                packed_n_chunk_cnt = cute.size(tFC1TR_tAcc.shape, mode=[4])
                out_n_subtile_cnt = packed_n_chunk_cnt // 2

                for epi_m_idx in cutlass.range(epi_m_cnt):
                    for out_n_idx in cutlass.range(out_n_subtile_cnt):
                        # N128 has one output subtile; N256 has two. Each
                        # reads consecutive low/high halves of its N128 input.
                        tFC1TR_tAcc_mn_lo = tFC1TR_tAcc[
                            (None, None, None, epi_m_idx, 2 * out_n_idx)
                        ]
                        tFC1TR_tAcc_mn_hi = tFC1TR_tAcc[
                            (None, None, None, epi_m_idx, 2 * out_n_idx + 1)
                        ]

                        cute.copy(tiled_copy_t2r, tFC1TR_tAcc_mn_lo, tFC1TR_rPackedLo)
                        cute.copy(tiled_copy_t2r, tFC1TR_tAcc_mn_hi, tFC1TR_rPackedHi)

                        # Preserve immutable packed values before reusing the
                        # two load buffers as logical up64/gate64 destinations.
                        # These aliases mark the change of meaning; they do
                        # not allocate another pair of register fragments.
                        packed_lo = tFC1TR_rPackedLo.load()
                        packed_hi = tFC1TR_rPackedHi.load()
                        tFC1TR_rAcc_up = tFC1TR_rPackedLo
                        tFC1TR_rAcc_gate = tFC1TR_rPackedHi
                        # Packed axes: (channel within N16, gate/up, group).
                        # Logical axes: (channel within N16, group, lo/hi).
                        packed_lo = packed_lo.reshape((16, 2, 2))
                        packed_hi = packed_hi.reshape((16, 2, 2))
                        gate_up_layout = cute.make_layout((16, 2, 2))
                        gate = cute.make_tensor(tFC1TR_rAcc_gate.iterator, gate_up_layout)
                        up = cute.make_tensor(tFC1TR_rAcc_up.iterator, gate_up_layout)
                        gate[None, None, 0].store(packed_lo[None, 0, None])
                        gate[None, None, 1].store(packed_hi[None, 0, None])
                        up[None, None, 0].store(packed_lo[None, 1, None])
                        up[None, None, 1].store(packed_hi[None, 1, None])

                        acc_vec_up = tFC1TR_rAcc_up.load()
                        acc_vec_gate = tFC1TR_rAcc_gate.load()

                        # SiTU on the decoded up/gate; C has N/2 cols.
                        tCompute = cute.make_rmem_tensor(acc_vec_gate.shape, self.acc_dtype)
                        self._apply_situ_epilogue(acc_vec_up, acc_vec_gate, alpha_val, tCompute)

                        if cutlass.const_expr(self.fc1_generate_sfc):
                            # Float4E2M1FN quantization: per-vector absmax →
                            # SFC → store SFC to gmem → quantize output by
                            # reciprocal of SFC. (Subtile is partitioned on N.)
                            sfc_subtile_idx_mn = (
                                tile_info[0] * self.fc1_epi_tile_cnt[0] + epi_m_idx,
                                tile_info[1] * self.fc1_epi_tile_cnt[1] + out_n_idx,
                            )
                            tCgFC1SFC = tCgFC1SFC_mn[
                                (
                                    None,
                                    None,
                                    None,
                                    *sfc_subtile_idx_mn,
                                )
                            ]

                            #
                            # Get absolute max across a vector and Compute SFC
                            #
                            tFC1TR_rAcc_frg = cute.logical_divide(
                                tCompute, cute.make_layout(self.sf_vec_size)
                            )
                            acc_frg = tFC1TR_rAcc_frg.load()
                            acc_frg = epilogue_op(acc_frg)

                            # Apply element-wise absolute value using math.absf (supports vectors)
                            abs_acc_frg_ir = math.absf(acc_frg.ir_value())
                            abs_acc_frg = type(acc_frg)(
                                abs_acc_frg_ir, acc_frg.shape, acc_frg.dtype
                            )

                            for vi in cutlass.range_constexpr(abs_acc_frg.shape[1]):
                                tCrFC1SFC_pvscale[vi] = abs_acc_frg[None, vi].reduce(
                                    cute.ReductionOp.MAX,
                                    cutlass.Float32(0.0),
                                    0,
                                )
                            for vi in cutlass.range_constexpr(0, abs_acc_frg.shape[1], 2):
                                tCrFC1SFC_pvscale[vi], tCrFC1SFC_pvscale[vi + 1] = (
                                    cute.arch.mul_packed_f32x2(
                                        (
                                            tCrFC1SFC_pvscale[vi],
                                            tCrFC1SFC_pvscale[vi + 1],
                                        ),
                                        (
                                            self.get_dtype_rcp_limits(self.fc1_c_dtype),
                                            self.get_dtype_rcp_limits(self.fc1_c_dtype),
                                        ),
                                    )
                                )
                                tCrFC1SFC_pvscale[vi], tCrFC1SFC_pvscale[vi + 1] = (
                                    cute.arch.mul_packed_f32x2(
                                        (
                                            tCrFC1SFC_pvscale[vi],
                                            tCrFC1SFC_pvscale[vi + 1],
                                        ),
                                        (norm_const, norm_const),
                                    )
                                )

                            tCrFC1SFC.store(tCrFC1SFC_pvscale.load().to(self.sf_dtype))

                            # Store SFC to gmem.
                            cute.autovec_copy(tCrFC1SFC, tCgFC1SFC)

                            # Quantize output and convert to c_dtype.
                            tCrFC1SFC_qpvscale_up = tCrFC1SFC.load().to(cutlass.Float32)
                            fp32_max = cutlass.Float32(3.40282346638528859812e38)
                            for vi in cutlass.range_constexpr(0, cute.size(tCrFC1SFC), 2):
                                acc_scale = cute.arch.mul_packed_f32x2(
                                    (
                                        cute.arch.rcp_approx(tCrFC1SFC_qpvscale_up[vi]),
                                        cute.arch.rcp_approx(tCrFC1SFC_qpvscale_up[vi + 1]),
                                    ),
                                    (norm_const, norm_const),
                                )
                                acc_scale_min0 = fmin(acc_scale[0], fp32_max, nan=True)
                                acc_scale_min1 = fmin(acc_scale[1], fp32_max, nan=True)

                                vec0 = tFC1TR_rAcc_frg[None, vi]
                                vec1 = tFC1TR_rAcc_frg[None, vi + 1]
                                for ei in cutlass.range_constexpr(self.sf_vec_size):
                                    vec0[ei], vec1[ei] = cute.arch.mul_packed_f32x2(
                                        (vec0[ei], vec1[ei]),
                                        (acc_scale_min0, acc_scale_min1),
                                    )

                            acc_vec = tiled_copy_r2s.retile(tCompute).load()
                            tFC1RS_rC.store(acc_vec.to(self.fc1_c_dtype))
                        else:
                            #
                            # Convert to C type
                            #
                            acc_vec = tiled_copy_r2s.retile(tCompute).load()
                            acc_vec = epilogue_op(acc_vec.to(self.fc1_c_dtype))
                            tFC1RS_rC.store(acc_vec)

                        #
                        # Store C to shared memory
                        #
                        num_prev_subtiles = num_prev_subtiles + 1
                        c_buffer = num_prev_subtiles % self.fc1_num_c_stage

                        cute.copy(
                            tiled_copy_r2s,
                            tFC1RS_rC,
                            tFC1RS_sC[(None, None, None, c_buffer)],
                        )
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )
                        self.epilog_sync_barrier.arrive_and_wait()
                        #
                        # TMA store C to global memory
                        #
                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(
                                tma_atom_fc1_c,
                                bSG_sC[(None, c_buffer)],
                                bSG_gC[(None, epi_m_idx, out_n_idx)],
                            )
                            # Fence and barrier to make sure shared memory store is visible to TMA store
                            fc1_c_pipeline.producer_commit()
                            fc1_c_pipeline.producer_acquire()
                        self.epilog_sync_barrier.arrive_and_wait()

                # A readiness release must order the global intermediate, not
                # merely permit sFC1C reuse. Therefore wait for full TMA S2G
                # completion (no read=True), synchronize the epilogue warps,
                # then let one elected lane increment this M tile's counter.
                cute.arch.cp_async_bulk_wait_group(0)
                self.epilog_sync_barrier.arrive_and_wait()
                if warp_idx == self.epilog_warp_id[0]:
                    with cute.arch.elect_one():
                        red_add_release_gpu(
                            fc1_ready,
                            mma_tile_coord_mnl[0],
                        )

                #
                # Async arrive accumulator buffer empty.
                #
                acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
                tile_info[1] = sInfo[(1, tile_info_consumer_state.index)]
                tile_info[2] = sInfo[(2, tile_info_consumer_state.index)]
                tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
                tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
                tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
                is_valid_tile = (tile_info[3] == 1) and (tile_info[5] == FC1_PHASE)
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            is_valid_fc2_tile = tile_info[3] == 1
            while is_valid_fc2_tile:
                tile_m_start = tile_info[0] * self.cta_tile_shape_mnk[0]
                meta_pipeline.consumer_wait(meta_consumer_state)
                meta_stage = meta_consumer_state.index

                acc_stage_index = acc_consumer_state.index
                tFC2TR_tAcc = tFC2TR_tAcc_base[(None, None, None, None, None, acc_stage_index)]
                acc_pipeline.consumer_wait(acc_consumer_state)

                tFC2TR_tAcc = cute.group_modes(tFC2TR_tAcc, 3, cute.rank(tFC2TR_tAcc))
                tFC2RS_sC_grouped = cute.group_modes(tFC2RS_sC, 1, 3)
                m_iter_cnt = cute.size(tFC2TR_tAcc.shape[3], mode=[0])
                n_subtile_cnt = cute.size(tFC2TR_tAcc.shape[3], mode=[1])

                for m_iter_idx in cutlass.range(m_iter_cnt):
                    permuted_row = tile_m_start + m_iter_idx * self.cta_tile_shape_mnk[0] + epi_tidx
                    is_valid_row = permuted_row < tile_info[4]
                    meta_row = m_iter_idx * self.cta_tile_shape_mnk[0] + epi_tidx
                    combined_scale = sFC2MetaScale[(meta_row, meta_stage)]

                    for n_subtile_idx in cutlass.range(n_subtile_cnt):
                        tFC2TR_tAcc_mn = tFC2TR_tAcc[
                            (
                                None,
                                None,
                                None,
                                (m_iter_idx, n_subtile_idx),
                            )
                        ]
                        cute.copy(
                            fc2_tiled_copy_t2r,
                            tFC2TR_tAcc_mn,
                            tFC2TR_rAcc,
                        )
                        fc2_acc_vec = fc2_tiled_copy_r2s.retile(tFC2TR_rAcc).load()
                        tFC2RS_rC.store((combined_scale * fc2_acc_vec).to(self.fc2_c_dtype))
                        if is_valid_row:
                            cute.copy(
                                fc2_tiled_copy_r2s,
                                tFC2RS_rC[None, 0, 0],
                                tFC2RS_sC_grouped[
                                    (
                                        None,
                                        (m_iter_idx, n_subtile_idx),
                                        0,
                                    )
                                ],
                            )
                        cute.arch.fence_proxy("async.shared", space="cta")

                cute.arch.fence_view_async_tmem_load()
                acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                for m_iter_idx in cutlass.range(m_iter_cnt):
                    permuted_row = tile_m_start + m_iter_idx * self.cta_tile_shape_mnk[0] + epi_tidx
                    is_valid_row = permuted_row < tile_info[4]
                    if is_valid_row:
                        meta_row = m_iter_idx * self.cta_tile_shape_mnk[0] + epi_tidx
                        token_idx = sFC2MetaTokenIdx[(meta_row, meta_stage)]
                        coord_n = tile_info[1] * self.cta_tile_shape_mnk[1]
                        scatter_out = cute.domain_offset((token_idx, coord_n, 0), mFC2C_mnl)
                        smem_row = m_iter_idx * self.cta_tile_shape_mnk[0] + epi_tidx
                        blk_reduce_bf16(
                            scatter_out,
                            sFC2C[(smem_row, None, 0)],
                            cutlass.Int32(self.fc2_c_copy_size),
                        )

                # The next FC2 tile may reuse sFC2C once the async reduce has
                # finished reading it. Global scatter-add completion is
                # drained once at kernel exit.
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                self.epilog_sync_barrier.arrive_and_wait()

                meta_pipeline.consumer_release(meta_consumer_state)
                meta_consumer_state.advance()

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                tile_info[0] = sInfo[(0, tile_info_consumer_state.index)]
                tile_info[1] = sInfo[(1, tile_info_consumer_state.index)]
                tile_info[3] = sInfo[(3, tile_info_consumer_state.index)]
                tile_info[4] = sInfo[(4, tile_info_consumer_state.index)]
                tile_info[5] = sInfo[(5, tile_info_consumer_state.index)]
                is_valid_fc2_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
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
            cute.arch.cp_async_bulk_wait_group(0)

        cute.arch.griddepcontrol_launch_dependents()

    def _die_claim_divisors(self, die_geometry):
        """Host-built FastDivmod divisors ``(tiles_per_m, tile_bytes)`` of one phase's claim.

        The claim divides Int32 linear indices by ``tiles_per_m`` and, rounding up, byte distances
        below ``tiles_per_m * tile_bytes + 2 MiB`` by ``tile_bytes``; 32-bit divisors cover both
        while ``tiles_per_m * tile_bytes + 2 MiB + tile_bytes < 2^31``. The launcher keeps every
        call that takes this kernel inside that bound. The ``runtime_assert`` below is not a guard:
        the DSL compiles it in only with ``CUTE_DSL_ENABLE_ASSERTIONS=1``.
        """
        tiles_per_m, _, _, tile_bytes = die_geometry
        runtime_assert(
            cutlass.Int64(tiles_per_m) * tile_bytes + (1 << DIE_CHUNK_SHIFT) + tile_bytes
            < (1 << 31),
            "die claim: FC12 expert weights must stay below 2^31 bytes - 2 MiB - one N tile",
        )
        return (
            cute.fast_divmod_create_divisor_v2(tiles_per_m),
            cute.fast_divmod_create_divisor_v2(cutlass.Int32(tile_bytes)),
        )

    @cute.jit
    def _die_local_claim_uniform(
        self,
        counters: cute.Tensor,
        queue: cutlass.Int32,
        own_die: cutlass.Int32,
        num_linear: cutlass.Int32,
        tiles_per_m: cutlass.Int32,
        tile_idx_to_expert_idx: cute.Tensor,
        b_base: cutlass.Int64,
        expert_bytes: cutlass.Int64,
        tile_bytes: cutlass.Int64,
        tiles_per_m_fdd: cute.FastDivmodDivisorV2,
        tile_bytes_fdd: cute.FastDivmodDivisorV2,
    ):
        """The engine's ``_die_local_claim`` run by the whole scheduler warp.

        Every lane runs the search on the values lane 0 broadcasts, lane 0 alone issues the
        atomics (``_die_claim_add`` / ``_die_claim_max``), and the divisions take the host-built
        FastDivmod divisors. The atomics are the engine claim's in the same order, so the claimed
        tile and the queue to claim from next are the same.
        """
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = tidx % 32
        linear_idx, cur_queue = self._die_claim_search(
            counters,
            queue,
            own_die,
            num_linear,
            tiles_per_m,
            tile_idx_to_expert_idx,
            b_base,
            expert_bytes,
            tile_bytes,
            tiles_per_m_fdd,
            tile_bytes_fdd,
            lane_idx,
        )
        m_tile, n_tile = divmod(linear_idx, tiles_per_m_fdd)
        work_tile = utils.WorkTileInfo(
            (
                cute.arch.make_warp_uniform(m_tile),
                cute.arch.make_warp_uniform(n_tile),
                cute.arch.make_warp_uniform(cutlass.Int32(0)),
            ),
            cute.arch.make_warp_uniform(linear_idx < num_linear),
        )
        return work_tile, cute.arch.make_warp_uniform(cur_queue)

    @cute.jit
    def _die_claim_search(
        self,
        counters: cute.Tensor,
        queue: cutlass.Int32,
        own_die: cutlass.Int32,
        num_linear: cutlass.Int32,
        tiles_per_m: cutlass.Int32,
        tile_idx_to_expert_idx: cute.Tensor,
        b_base: cutlass.Int64,
        expert_bytes: cutlass.Int64,
        tile_bytes: cutlass.Int64,
        tiles_per_m_fdd: cute.FastDivmodDivisorV2,
        tile_bytes_fdd: cute.FastDivmodDivisorV2,
        lane_idx: cutlass.Int32,
    ):
        """The search loop of ``_die_local_claim`` on warp-uniform values: the claimed linear
        index and the queue it came from.

        Each pass either returns a tile of the queue's die, or moves the queue's counter forward
        (atomic max to the next tile of that die, or atomic add), or switches once from the own
        die's exhausted counter to the other die's, or stops on the other die's exhausted counter.
        The counters only grow and the index of a pass is above the previous one, so a queue
        takes at most ``num_linear + 1`` passes; nothing waits on another CTA.
        """
        # Every branch of the loop then depends on warp-uniform values only.
        queue = cute.arch.make_warp_uniform(queue)
        own_die = cute.arch.make_warp_uniform(own_die)
        num_linear = cute.arch.make_warp_uniform(num_linear)
        cur_queue = queue
        linear_idx = self._die_claim_add(counters, cur_queue, lane_idx)
        searching = cutlass.Boolean(True)
        while searching:
            if linear_idx >= num_linear:
                if cur_queue == own_die:
                    cur_queue = 1 - own_die
                    linear_idx = self._die_claim_add(counters, cur_queue, lane_idx)
                else:
                    searching = cutlass.Boolean(False)
            else:
                claim_m, claim_n = divmod(linear_idx, tiles_per_m_fdd)
                claim_m = cutlass.Int32(claim_m)
                expert = tile_idx_to_expert_idx[claim_m]
                expert = cute.arch.make_warp_uniform(expert)
                # Middle byte of N tile 0 of the M tile's expert.
                first_middle = b_base + cutlass.Int64(expert) * expert_bytes + (tile_bytes >> 1)
                middle = first_middle + cutlass.Int64(claim_n) * tile_bytes
                if cutlass.Int32((middle >> DIE_CHUNK_SHIFT) & 1) == cur_queue:
                    searching = cutlass.Boolean(False)
                else:
                    # First N tile whose middle byte lies in the next 2 MiB chunk (the queue's
                    # die), or the next M tile; the distance fits Int32 within the launcher's
                    # bound (_die_claim_divisors).
                    next_chunk = ((middle >> DIE_CHUNK_SHIFT) + 1) << DIE_CHUNK_SHIFT
                    next_n = cutlass.Int32(
                        divmod(
                            cutlass.Int32(next_chunk - first_middle + tile_bytes - 1),
                            tile_bytes_fdd,
                        )[0]
                    )
                    if next_n > tiles_per_m:
                        next_n = tiles_per_m
                    resume = claim_m * tiles_per_m + next_n
                    previous = self._die_claim_max(counters, cur_queue, resume + 1, lane_idx)
                    if previous <= resume:
                        linear_idx = resume
                    else:
                        linear_idx = self._die_claim_add(counters, cur_queue, lane_idx)
        return linear_idx, cur_queue

    @cute.jit
    def _die_claim_add(self, counters: cute.Tensor, queue: cutlass.Int32, lane_idx: cutlass.Int32):
        """Lane 0's relaxed atomic add of 1 to die ``queue``'s counter; its old value,
        broadcast to the warp."""
        old = cutlass.Int32(0)
        if lane_idx == 0:
            old = atom_add_relaxed_gpu(counters, queue * DIE_COUNTER_STRIDE, cutlass.Int32(1))
        old = cute.arch.make_warp_uniform(cute.arch.shuffle_sync(old, 0))
        return old

    @cute.jit
    def _die_claim_max(
        self,
        counters: cute.Tensor,
        queue: cutlass.Int32,
        value: cutlass.Int32,
        lane_idx: cutlass.Int32,
    ):
        """Lane 0's relaxed atomic max of ``value`` into die ``queue``'s counter; its old value,
        broadcast to the warp."""
        old = cutlass.Int32(0)
        if lane_idx == 0:
            old = atom_max_relaxed_gpu(counters, queue * DIE_COUNTER_STRIDE, value)
        old = cute.arch.make_warp_uniform(cute.arch.shuffle_sync(old, 0))
        return old

    @cute.jit
    def _l2_prefetch_head(
        self,
        tma_atom_b: cute.CopyAtom,
        tma_atom_sfb: cute.CopyAtom,
        b_k: cute.Tensor,
        sfb_k: cute.Tensor,
        k_tile_cnt: cutlass.Int32,
    ):
        """L2 prefetch of a tile's first ``min(l2_prefetch, k_tile_cnt)`` K-blocks of B and SFB.

        ``b_k`` / ``sfb_k``: the tile's TMA source slices over K, as its loads take them; the
        prefetch goes through the loads' own atoms, so it covers the bytes they will read.
        """
        for k_tile in cutlass.range(0, min(self.l2_prefetch, k_tile_cnt), 1, unroll=1):
            cute.prefetch(tma_atom_b, b_k[(None, k_tile)])
            cute.prefetch(tma_atom_sfb, sfb_k[(None, k_tile)])

    @cute.jit
    def _l2_prefetch_ahead(
        self,
        tma_atom_b: cute.CopyAtom,
        tma_atom_sfb: cute.CopyAtom,
        b_k: cute.Tensor,
        sfb_k: cute.Tensor,
        k_loaded: cutlass.Int32,
        k_tile_cnt: cutlass.Int32,
    ):
        """L2 prefetch of K-block ``k_loaded + l2_prefetch`` of B and SFB when the tile has it
        (``k_loaded``: the K-block whose loads were just issued)."""
        k_tile = k_loaded + self.l2_prefetch
        if k_tile < k_tile_cnt:
            cute.prefetch(tma_atom_b, b_k[(None, k_tile)])
            cute.prefetch(tma_atom_sfb, sfb_k[(None, k_tile)])

    @cute.jit
    def _l2_prefetch_fc2_head(
        self,
        tma_atom_fc2_b: cute.CopyAtom,
        tma_atom_fc2_sfb: cute.CopyAtom,
        tFC2BgB: cute.Tensor,
        tFC2BgSFB: cute.Tensor,
        fc2_n_tile: cutlass.Int32,
        fc2_expert: cutlass.Int32,
        fc2_k_tile_cnt: cutlass.Int32,
    ):
        """``_l2_prefetch_head`` of FC2 tile ``(fc2_n_tile, fc2_expert)``, sliced as the FC2 loads
        slice it."""
        self._l2_prefetch_head(
            tma_atom_fc2_b,
            tma_atom_fc2_sfb,
            tFC2BgB[(None, fc2_n_tile, None, fc2_expert)],
            tFC2BgSFB[(None, fc2_n_tile, None, fc2_expert)],
            fc2_k_tile_cnt,
        )
