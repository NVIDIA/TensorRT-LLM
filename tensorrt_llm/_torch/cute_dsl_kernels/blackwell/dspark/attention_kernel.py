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

import math
from types import SimpleNamespace
from typing import Optional, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.arch import Arch
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cutlass_dsl import BaseDSL
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

try:
    from cutlass import memory as cutlass_memory
except ImportError:
    # CUTLASS DSL releases before the memory namespace split expose the same
    # allocators from cutlass.utils.
    cutlass_memory = utils


from .scheduler_helpers import (
    LOG2_E,
    HCAStaticTileScheduler,
    HCAStaticTileSchedulerParams,
    ceil_div,
    create_hca_static_tile_scheduler,
    create_hca_static_tile_scheduler_params,
)

"""DSpark rolling-window MQA attention for NVIDIA Blackwell-family GPUs.

A TMA + tcgen05 tensor-core warp-specialized persistent kernel that fuses the
Q*K^T matmul, attention-sink online softmax, P*V matmul, and the inverse-RoPE
epilogue into a single launch. KV arrives as two paged streams (the 128-row
rolling window and the draft block's own rows); see
``page8_specialization.py`` for the supported DSpark contract and
``tests/unittest/_torch/speculative/test_dspark_cute_dsl_attention.py`` for
validation coverage.

The split-KV/LSE/variable-sequence machinery of the generic attention kernel and its
standalone harness have been removed, and the inverse-RoPE epilogue fusion was
added here.
"""


class DSparkAttentionKernel:
    # The DSpark accumulator/correction layouts occupy a fixed 512-column
    # TMEM footprint on every supported architecture.
    tmem_alloc_cols: int = 512

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        mma_qk_tiler_mn: Tuple[int, int],
        mma_pv_tiler_mn: Tuple[int, int],
        max_active_clusters: int,
        page_size_cmp: int,
        page_size_win: int,
        skip_correction_threshold: float,
        is_persistent: bool,
        is_var_seq: bool,
        is_var_split_kv: bool,
        *,
        arch_str: str,
        seq_len_q: int = 1,
        mma_qk_tiler_k: int = 128,
    ):
        """Initialize the DSpark attention kernel configuration.

        :param acc_dtype: Data type for accumulation S and O
        :type acc_dtype: Type[cutlass.Numeric]
        :param mma_s_tiler: The (H, K) tile shape of the MMA instruction for S
        :type mma_s_tiler: Tuple[int, int]
        :param mma_p_tiler: The (H, D) tile shape of the MMA instruction for P
        :type mma_p_tiler: Tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: int
        :param page_size_cmp: Page size of the compressed-KV page table
        :type page_size_cmp: int
        :param page_size_win: Page size of the sliding-window page table
            (must be a power-of-two multiple of page_size_cmp)
        :type page_size_win: int
        :param skip_correction_threshold: Threshold to skip correction
        :type skip_correction_threshold: float
        :param is_persistent: Whether to use persistent kernel mode
        :type is_persistent: bool
        :param is_var_seq: Whether to use variable sequence length
        :type is_var_seq: bool
        :param is_var_split_kv: Whether to use variable split KV
        :type is_var_split_kv: bool
        :param arch_str: CuTe allocator architecture for the target GPU
        :type arch_str: str
        :param seq_len_q: Query sequence length used to specialize tiling
        :type seq_len_q: int
        :param mma_qk_tiler_k: QK MMA reduction tile; must be 128.
        :type mma_qk_tiler_k: int
        """

        if mma_qk_tiler_k != 128:
            raise ValueError(f"mma_qk_tiler_k must be 128, got {mma_qk_tiler_k}")

        self.arch_str = arch_str
        self.arch_name = arch_str.upper()
        max_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(arch_str)
        if self.tmem_alloc_cols > max_tmem_alloc_cols:
            raise ValueError(
                f"DSpark needs {self.tmem_alloc_cols} TMEM columns, but "
                f"{self.arch_name} provides {max_tmem_alloc_cols}"
            )
        # latent_dim is the FULL per-head depth for HCA (head_dim).
        # The last `qk_rope_head_dim` of these are assumed to be already
        # RoPE-rotated by the caller; the kernel does not see rope as a
        # separate path.
        self.latent_dim = 512
        self.acc_dtype = acc_dtype
        self.mma_qk_tiler_mn = mma_qk_tiler_mn
        self.mma_pv_tiler_mn = mma_pv_tiler_mn
        self.max_active_clusters = max_active_clusters
        self.skip_correction_threshold = skip_correction_threshold
        self.is_persistent = is_persistent
        self.page_size_cmp = page_size_cmp
        self.page_size_win = page_size_win
        # Physical page sizes define the page-table/cache ABI. TMA page spans
        # normally match them, but a specialization may use a wider logical
        # span so one OOB-zero-filling TMA replaces many tiny page copies.
        self.tma_page_size_cmp = page_size_cmp
        self.tma_page_size_win = page_size_win
        self.fixed_cache_seq_len = None
        self.implicit_cmp_page_table = False
        self.attn_sink_is_scaled = False
        # Fixed two-stream specializations can receive the initialized-row
        # count separately from the absolute decode position, masking unwritten
        # rows in the first window tile while keeping the compressed tile fixed.
        self.window_valid_len_from_tensor = False
        # Fixed-shape specializations may fuse the inverse-RoPE epilogue: the
        # last ``inverse_rope_dim`` output lanes are de-rotated in registers
        # right before the store, preserving the former kernel boundary's BF16
        # rounding. 0 keeps the generic HCA output path unchanged.
        self.inverse_rope_dim = 0
        self.is_var_seq = is_var_seq
        self.is_var_split_kv = is_var_split_kv
        self.seq_len_q = seq_len_q
        self.mma_qk_tiler_k = mma_qk_tiler_k
        self.cluster_shape_mnk = (2, 1, 1)
        self.use_2cta_instrs = True
        # When using 2 CTAs with m=128: warps 0-1 handle accumulation for first half [0, n/2),
        # while warps 2-3 handle accumulation for second half [n/2, n)
        self.warps_in_n = 2
        self.num_compute_warps = 4
        self.threads_per_warp = 32
        mma_qk_tiler_k = self._get_mma_qk_tiler_k()
        self.mma_qk_tiler = (
            self.mma_qk_tiler_mn[0],
            self.mma_qk_tiler_mn[1],
            mma_qk_tiler_k,
        )
        self.mma_pv_tiler = (
            self.mma_pv_tiler_mn[0],
            self.mma_pv_tiler_mn[1],
            self.mma_qk_tiler[1] * self.mma_qk_tiler[2] // self.mma_pv_tiler_mn[1],
        )
        self.iterations_qk_latent = self.latent_dim // self.mma_qk_tiler[2]
        self.iterations_qk = self.iterations_qk_latent
        self.iterations_pv_k = self.mma_qk_tiler[1] // self.mma_pv_tiler[2]
        self.iterations_pv_n = self.latent_dim // self.mma_pv_tiler[1]

        # Set specialized warp ids
        self.compute_warp_ids = (0, 1, 2, 3)
        self.correction_warp_ids = (4, 5, 6, 7)
        self.mma_warp_id = 8

        self.load_tma_warp_id = 9
        self.load_pt_warp_id = 10
        self.empty_warp_ids = (11,)
        self.threads_per_cta = self.threads_per_warp * len(
            (
                self.mma_warp_id,
                self.load_tma_warp_id,
                self.load_pt_warp_id,
                *self.compute_warp_ids,
                *self.correction_warp_ids,
                *self.empty_warp_ids,
            )
        )

        # register settings
        self.softmax_reg_num = 192
        self.correction_reg_num = 208
        self.other_reg_num = 96
        # Named barriers
        self.tmem_ptr_sync_bar = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=(
                self.threads_per_warp + self.threads_per_warp * self.num_compute_warps * 2
            ),
        )
        self.softmax_exchange_sync_bar = pipeline.NamedBarrier(
            barrier_id=2, num_threads=(self.threads_per_warp * self.num_compute_warps)
        )
        self.epilogue_exchange_sync_bar = pipeline.NamedBarrier(
            barrier_id=3, num_threads=(self.threads_per_warp * self.num_compute_warps)
        )

    def _setup_attributes(self):
        """Set up configurations and parameters for the HCA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the heavily compressed attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """

        self.load_q_stage = 1
        self.load_kv_stage = 7
        self.mma_s_stage = 2
        self.p_mma_stage = 2
        self.p_cor_stage = 2
        self.mma_o_stage = 1
        self.load_pt_stage = 4

        self.tmem_o_offset = self.mma_s_stage * self.mma_qk_tiler[1] // self.warps_in_n
        self.correction_factor_offset = self.tmem_o_offset + self.latent_dim // self.warps_in_n

    def _get_mma_qk_tiler_k(self) -> int:
        # The caller supplies the supported QK reduction tile explicitly.
        return self.mma_qk_tiler_k

    @cute.jit
    def __call__(
        self,
        q_latent: cute.Tensor,
        c_latent_win: cute.Tensor,
        c_latent_cmp: cute.Tensor,
        page_table_win: cute.Tensor,
        page_table_cmp: cute.Tensor,
        o: cute.Tensor,
        cache_seqs: Optional[cute.Tensor],
        window_valid_lens: Optional[cute.Tensor],
        softmax_scale: cutlass.Float32,
        output_scale: cutlass.Float32,
        attn_sink_unscaled: cute.Tensor,
        inverse_rope_freqs: Optional[cute.Tensor],
        stream: cuda.CUstream,
    ):
        """Execute the Heavily Compressed Attention (HCA) operation on the provided tensors.

        KV is split into two streams concatenated along the seq_len_k dimension:
          - Sliding-window: first `mma_qk_tiler[1]` rows (default 128).
          - Compressed:     remaining rows.
        Each stream has its own page table and page size; the kernel feeds them
        into a single MMA stream (k_tile_idx == 0 is window, idx >= 1 is
        compressed with offset of one tile).

        :param q_latent: The query tensor with shape [num_head, latent_dim, seq_len_q, batch_size]
            (last `qk_rope_head_dim` lanes pre-rotated by caller).
        :type q_latent: cute.Tensor
        :param c_latent_win: Sliding-window key tensor with shape
            [mma_qk_tiler[1], latent_dim, batch_size] (last `qk_rope_head_dim`
            lanes pre-rotated by caller).
        :type c_latent_win: cute.Tensor
        :param c_latent_cmp: Compressed key tensor with shape
            [seq_len_k_cmp, latent_dim, batch_size] (last `qk_rope_head_dim`
            lanes pre-rotated by caller).
        :type c_latent_cmp: cute.Tensor
        :param page_table_win: Page table for the sliding-window stream
            with shape [page_count_win, batch_size]
        :type page_table_win: cute.Tensor
        :param page_table_cmp: Page table for the compressed stream
            with shape [page_count_cmp, batch_size]
        :type page_table_cmp: cute.Tensor
        :param o: The output tensor with shape [num_head, latent_dim, seq_len_q, batch_size]
        :type o: cute.Tensor
        :param cache_seqs: The absolute decode positions with shape [batch_size]
        :type cache_seqs: cute.Tensor
        :param window_valid_lens: The initialized rolling-window row counts with
            shape [batch_size].
        :type window_valid_lens: cute.Tensor
        :param softmax_scale: The scale factor for softmax
        :type softmax_scale: cutlass.Float32
        :param output_scale: The scale factor for the output
        :type output_scale: cutlass.Float32
        :param attn_sink_unscaled: Per-head attention-sink logit divided
            by `softmax_scale` (i.e., expressed in the "unscaled S" space the
            kernel uses). Shape [num_heads]. Acts as a virtual extra softmax
            logit with V=0; per-row row_max is initialised from this tensor.
        :type attn_sink_unscaled: cute.Tensor
        :param inverse_rope_freqs: Flat FP32 ``[batch * seq_len_q *
            inverse_rope_dim]`` cos/sin pairs for the fused inverse-RoPE
            epilogue (``None`` when ``inverse_rope_dim == 0``).
        :type inverse_rope_freqs: Optional[cute.Tensor]
        :param stream: The CUDA stream to execute the kernel on
        :type stream: cuda.CUstream

        :raises TypeError: If tensor data types don't match or aren't supported
        """

        # setup static attributes before smem/grid/tma computation
        self.q_dtype = q_latent.element_type
        self.k_dtype = c_latent_cmp.element_type
        self.v_dtype = c_latent_cmp.element_type
        self.o_dtype = o.element_type

        # check type consistency
        if cutlass.const_expr(
            self.q_dtype != self.k_dtype
            or self.q_dtype != self.v_dtype
            or c_latent_win.element_type != self.k_dtype
        ):
            raise TypeError("Type mismatch among q/c_win/c_cmp")
        # check leading dimensions of input/output
        if cutlass.const_expr(q_latent.stride[1] != 1):
            raise ValueError("q_latent must have leading dimension 1")
        if cutlass.const_expr(c_latent_cmp.stride[1] != 1):
            raise ValueError("c_latent_cmp must have leading dimension 1")
        if cutlass.const_expr(c_latent_win.stride[1] != 1):
            raise ValueError("c_latent_win must have leading dimension 1")
        if cutlass.const_expr(o.stride[1] != 1):
            raise ValueError("o must have leading dimension 1")

        # DSpark always runs a single split with the final-output epilogue; the
        # intermediate-accumulator, LSE, and reduction machinery of the
        # original HCA kernel are compiled out.
        lse = None
        acc_o = None
        acc_lse = None
        split_kv = 1
        block_split_kvs = None

        c_latent_cmp_transpose_layout = cute.select(c_latent_cmp.layout, mode=[1, 0, 2])
        c_latent_cmp_transpose = cute.make_tensor(
            c_latent_cmp.iterator, c_latent_cmp_transpose_layout
        )
        c_latent_win_transpose_layout = cute.select(c_latent_win.layout, mode=[1, 0, 2])
        c_latent_win_transpose = cute.make_tensor(
            c_latent_win.iterator, c_latent_win_transpose_layout
        )

        self.q_major_mode = OperandMajorMode.K
        self.k_major_mode = OperandMajorMode.K
        self.v_major_mode = OperandMajorMode.MN

        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.TWO
        # the intermediate tensor p is from smem & k-major
        p_major_mode = OperandMajorMode.K
        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.acc_dtype,
            cta_group,
            self.mma_qk_tiler[:2],
        )
        pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            self.v_dtype,
            p_major_mode,
            self.v_major_mode,
            self.acc_dtype,
            cta_group,
            self.mma_pv_tiler[:2],
        )

        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )

        self.epi_tile = self.mma_pv_tiler[:2]

        q_latent_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.mma_qk_tiler,
            self.q_dtype,
            (self.iterations_qk_latent * self.load_q_stage),
        )
        q_latent_smem_layout_staged = cute.logical_divide(
            q_latent_smem_layout_staged, (None, None, None, self.iterations_qk_latent)
        )

        kc_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.mma_qk_tiler,
            self.k_dtype,
            self.load_kv_stage,
        )
        cta_kv_m = qk_tiled_mma.op.shape_mnk[0] // qk_tiled_mma.thr_id.shape
        kc_page_tile_size_cmp = min(self.tma_page_size_cmp, cta_kv_m)
        kc_page_tile_size_win = min(self.tma_page_size_win, cta_kv_m)

        kc_smem_layout_for_tma_base = sm100_utils.make_smem_layout(
            OperandMajorMode.K,
            (self.mma_qk_tiler[0] // qk_tiled_mma.thr_id.shape, self.mma_qk_tiler[2]),
            self.k_dtype,
            self.load_kv_stage,
        )
        kc_smem_layout_for_tma_cmp = cute.tiled_divide(
            kc_smem_layout_for_tma_base, (kc_page_tile_size_cmp, self.mma_qk_tiler[2])
        )
        kc_smem_layout_for_tma_win = cute.tiled_divide(
            kc_smem_layout_for_tma_base, (kc_page_tile_size_win, self.mma_qk_tiler[2])
        )

        p_smem_layout_staged = sm100_utils.make_smem_layout_a(
            pv_tiled_mma,
            self.mma_pv_tiler,
            self.q_dtype,
            (self.iterations_pv_k * self.p_mma_stage),
        )
        p_smem_layout_staged = cute.logical_divide(
            p_smem_layout_staged, (None, None, None, self.iterations_pv_k)
        )

        vc_smem_layout_staged = sm100_utils.make_smem_layout_b(
            pv_tiled_mma,
            self.mma_pv_tiler,
            self.v_dtype,
            self.load_kv_stage,
        )
        vc_page_tile_size_cmp = min(self.tma_page_size_cmp, self.mma_pv_tiler[2])
        vc_page_tile_size_win = min(self.tma_page_size_win, self.mma_pv_tiler[2])
        vc_smem_layout_for_tma_base = sm100_utils.make_smem_layout(
            OperandMajorMode.MN,
            (self.mma_pv_tiler[1] // pv_tiled_mma.thr_id.shape, self.mma_pv_tiler[2]),
            self.v_dtype,
            self.load_kv_stage,
        )
        vc_pv_n = pv_tiled_mma.op.shape_mnk[1] // pv_tiled_mma.thr_id.shape
        vc_smem_layout_for_tma_cmp = cute.tiled_divide(
            vc_smem_layout_for_tma_base,
            (vc_pv_n, vc_page_tile_size_cmp),
        )
        vc_smem_layout_for_tma_win = cute.tiled_divide(
            vc_smem_layout_for_tma_base,
            (vc_pv_n, vc_page_tile_size_win),
        )
        # TMA load for Q latent
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)

        q_latent_smem_layout = cute.select(q_latent_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_q_latent, tma_tensor_q_latent = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q_latent,
            q_latent_smem_layout,
            self.mma_qk_tiler,
            qk_tiled_mma,
            cta_layout_vmnk.shape,
        )
        # TMA load for c latent (cmp + win)
        kc_smem_layout_cmp = cute.select(kc_smem_layout_for_tma_cmp, mode=[0])
        tma_atom_c_latent_cmp, tma_tensor_c_latent_cmp = self.make_paged_tiled_tma_atom(
            tma_load_op,
            c_latent_cmp,
            kc_smem_layout_cmp,
            (self.mma_qk_tiler[1], self.mma_qk_tiler[2]),
            qk_tiled_mma,
            is_k_load=True,
            page_size=self.tma_page_size_cmp,
        )
        kc_smem_layout_win = cute.select(kc_smem_layout_for_tma_win, mode=[0])
        tma_atom_c_latent_win, tma_tensor_c_latent_win = self.make_paged_tiled_tma_atom(
            tma_load_op,
            c_latent_win,
            kc_smem_layout_win,
            (self.mma_qk_tiler[1], self.mma_qk_tiler[2]),
            qk_tiled_mma,
            is_k_load=True,
            page_size=self.tma_page_size_win,
        )
        # TMA load for c latent transpose (cmp + win)
        vc_smem_layout_cmp = cute.select(vc_smem_layout_for_tma_cmp, mode=[0])
        tma_atom_c_latent_transpose_cmp, tma_tensor_c_latent_transpose_cmp = (
            self.make_paged_tiled_tma_atom(
                tma_load_op,
                c_latent_cmp_transpose,
                vc_smem_layout_cmp,
                (self.mma_pv_tiler[1], self.mma_pv_tiler[2]),
                pv_tiled_mma,
                is_k_load=False,
                page_size=self.tma_page_size_cmp,
            )
        )
        vc_smem_layout_win = cute.select(vc_smem_layout_for_tma_win, mode=[0])
        tma_atom_c_latent_transpose_win, tma_tensor_c_latent_transpose_win = (
            self.make_paged_tiled_tma_atom(
                tma_load_op,
                c_latent_win_transpose,
                vc_smem_layout_win,
                (self.mma_pv_tiler[1], self.mma_pv_tiler[2]),
                pv_tiled_mma,
                is_k_load=False,
                page_size=self.tma_page_size_win,
            )
        )

        q_latent_copy_size = (
            cute.size_in_bytes(self.q_dtype, q_latent_smem_layout)
            * cute.size(qk_tiled_mma.thr_id.shape)
            * self.iterations_qk_latent
        )
        q_copy_size = q_latent_copy_size
        kc_copy_size = cute.size_in_bytes(
            self.k_dtype, cute.select(kc_smem_layout_staged, mode=[0, 1, 2])
        ) * cute.size(qk_tiled_mma.thr_id.shape)
        vc_copy_size = cute.size_in_bytes(
            self.v_dtype, cute.select(vc_smem_layout_staged, mode=[0, 1, 2])
        ) * cute.size(pv_tiled_mma.thr_id.shape)
        assert kc_copy_size == vc_copy_size, "kc_copy_size and vc_copy_size must be the same"

        self.tma_copy_q_bytes = q_copy_size
        self.tma_copy_kc_bytes = kc_copy_size

        tile_sched_params, grid = self._compute_grid(
            o,
            split_kv,
            self.cluster_shape_mnk,
            self.max_active_clusters,
            self.is_persistent,
        )

        @cute.struct
        class SplitKVKernelSharedStorage:
            # Pipeline barriers
            load_q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_q_stage * 2]
            load_kv_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_kv_stage * 2]
            mma_s_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_s_stage * 2]
            p_mma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.p_mma_stage * 2]
            p_cor_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.p_cor_stage * 2]
            mma_o_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_o_stage * 2]
            load_pt_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_pt_stage * 2]
            # Tmem dealloc cluster barrier
            tmem_dealloc_mbar: cutlass.Int64

            # Tmem holding buffer
            tmem_holding_buf: cutlass.Int32
            # Smem tensors
            softmax_smem_exchange: cute.struct.MemRange[
                self.acc_dtype, self.num_compute_warps * self.threads_per_warp
            ]
            epilogue_smem_exchange: cute.struct.MemRange[
                self.acc_dtype, self.num_compute_warps * self.threads_per_warp
            ]
            smem_q_latent: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(q_latent_smem_layout_staged)],
                1024,
            ]
            smem_kc: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(kc_smem_layout_staged)],
                1024,
            ]
            smem_p: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(p_smem_layout_staged)],
                1024,
            ]
            smem_page_table: cute.struct.MemRange[
                cutlass.Int32, self.load_pt_stage * self.mma_qk_tiler[1] // 2
            ]

        required_smem_bytes = SplitKVKernelSharedStorage.__sizeof__()
        available_smem_bytes = cutlass_memory.get_smem_capacity_in_bytes(self.arch_str)
        if cutlass.const_expr(required_smem_bytes > available_smem_bytes):
            raise ValueError(
                f"HCA SharedStorage needs {required_smem_bytes} bytes, but "
                f"{self.arch_name} provides {available_smem_bytes} bytes"
            )

        softmax_scale_log2 = softmax_scale * LOG2_E
        self.split_kv_kernel(
            qk_tiled_mma,
            pv_tiled_mma,
            tma_atom_q_latent,
            tma_tensor_q_latent,
            tma_atom_c_latent_win,
            tma_tensor_c_latent_win,
            tma_atom_c_latent_cmp,
            tma_tensor_c_latent_cmp,
            tma_atom_c_latent_transpose_win,
            tma_tensor_c_latent_transpose_win,
            tma_atom_c_latent_transpose_cmp,
            tma_tensor_c_latent_transpose_cmp,
            page_table_win,
            page_table_cmp,
            o,
            lse,
            acc_o,
            acc_lse,
            split_kv,
            cache_seqs,
            window_valid_lens,
            block_split_kvs,
            softmax_scale_log2,
            output_scale,
            attn_sink_unscaled,
            inverse_rope_freqs,
            q_latent_smem_layout_staged,
            kc_smem_layout_staged,
            p_smem_layout_staged,
            vc_smem_layout_staged,
            kc_smem_layout_for_tma_win,
            kc_smem_layout_for_tma_cmp,
            vc_smem_layout_for_tma_win,
            vc_smem_layout_for_tma_cmp,
            cta_layout_vmnk,
            tile_sched_params,
            SplitKVKernelSharedStorage,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def make_paged_tiled_tma_atom(
        self,
        tma_load_op: cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp,
        gmem: cute.Tensor,
        smem_layout: cute.Layout,
        mma_tiler,
        tiled_mma: cute.TiledMma,
        is_k_load: bool,
        page_size: int,
    ):
        ident = cute.make_identity_layout(gmem.shape)
        g_tile = cute.composition(ident, mma_tiler)
        cta_mn = mma_tiler[0] // tiled_mma.thr_id.shape
        cta_v_map = cute.flat_divide(g_tile, (cta_mn,))
        cta_v_map = cute.select(cta_v_map, mode=[0, 2])
        page_tile_size = min(page_size, cta_mn) if is_k_load else min(page_size, mma_tiler[1])
        cta_v_map = cute.zipped_divide(
            cta_v_map,
            (page_tile_size, mma_tiler[1]) if is_k_load else (cta_mn, page_tile_size),
        )
        cta_v_map = cute.select(cta_v_map, mode=[0])
        from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir

        res = _cute_nvgpu_ir.atom_make_non_exec_tiled_tma_load(
            gmem.value,
            smem_layout.value,
            cta_v_map,
            tma_load_op._to_ir(),
            num_multicast=1,
        )
        return cute.CopyAtom(tma_load_op, cpasync.CopyBulkTensorTileG2SNonExecTrait(res[0])), res[1]

    @cute.kernel
    def split_kv_kernel(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tma_atom_q_latent: Optional[cute.CopyAtom],
        mQL: cute.Tensor,
        tma_atom_c_latent_win: Optional[cute.CopyAtom],
        mCL_win: cute.Tensor,
        tma_atom_c_latent_cmp: Optional[cute.CopyAtom],
        mCL_cmp: cute.Tensor,
        tma_atom_c_latent_transpose_win: Optional[cute.CopyAtom],
        mCLT_win: cute.Tensor,
        tma_atom_c_latent_transpose_cmp: Optional[cute.CopyAtom],
        mCLT_cmp: cute.Tensor,
        mPT_win: cute.Tensor,
        mPT_cmp: cute.Tensor,
        mO: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor],
        mAccO: Optional[cute.Tensor],
        mAccLSE: Optional[cute.Tensor],
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        window_valid_lens: cute.Tensor,
        block_split_kvs: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        output_scale: cutlass.Float32,
        attn_sink_unscaled: cute.Tensor,
        inverse_rope_freqs: Optional[cute.Tensor],
        q_latent_smem_layout_staged: cute.ComposedLayout,
        kc_smem_layout_staged: cute.ComposedLayout,
        p_smem_layout_staged: cute.ComposedLayout,
        vc_smem_layout_staged: cute.ComposedLayout,
        kc_smem_layout_for_tma_win: cute.ComposedLayout,
        kc_smem_layout_for_tma_cmp: cute.ComposedLayout,
        vc_smem_layout_for_tma_win: cute.ComposedLayout,
        vc_smem_layout_for_tma_cmp: cute.ComposedLayout,
        cta_layout_vmnk: cute.Layout,
        tile_sched_params: HCAStaticTileSchedulerParams,
        SharedStorage: cutlass.Constexpr,
    ):
        """The device split_kv kernel implementation of the Heavily Compressed Attention (HCA).

        This kernel coordinates multiple specialized warps to perform different phases of the HCA computation:
        1. Load warp: Loads Q/C latent data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Compute warps: Compute softmax and do rescaling on accumulators, and store the intermediate/final results
        to global memory

        The kernel produces either intermediate or final results of the HCA computation based on the split_kv parameter.
        When split_kv is 1, the kernel generates the final results directly. Otherwise, it produces intermediate results
        that will later be combined by a reduction kernel.

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases.

        :param tiled_mma_qk: Tiled MMA for Q*K^T
        :type tiled_mma_qk: cute.TiledMma
        :param tiled_mma_pv: Tiled MMA for P*V
        :type tiled_mma_pv: cute.TiledMma
        :param tma_atom_q_latent: TMA copy atom for query latent tensor
        :type tma_atom_q_latent: cute.CopyAtom
        :param mQL: query latent tensor
        :type mQL: cute.Tensor
        :param tma_atom_c_latent_win: TMA copy atom for window-stream K
        :type tma_atom_c_latent_win: cute.CopyAtom
        :param mCL_win: Window-stream key tensor
        :type mCL_win: cute.Tensor
        :param tma_atom_c_latent_cmp: TMA copy atom for compressed-stream K
        :type tma_atom_c_latent_cmp: cute.CopyAtom
        :param mCL_cmp: Compressed-stream key tensor
        :type mCL_cmp: cute.Tensor
        :param mCLT_win: Window-stream V transpose tensor
        :type mCLT_win: cute.Tensor
        :param mCLT_cmp: Compressed-stream V transpose tensor
        :type mCLT_cmp: cute.Tensor
        :param mPT_win: Window-stream page table tensor
        :type mPT_win: cute.Tensor
        :param mPT_cmp: Compressed-stream page table tensor
        :type mPT_cmp: cute.Tensor
        :param mO: Output tensor
        :type mO: cute.Tensor
        :param mLSE: Log-sum-exp tensor
        :type mLSE: cute.Tensor
        :param mAccO: Intermediate accumulator output tensor
        :type mAccO: cute.Tensor
        :param mAccLSE: Intermediate accumulator log-sum-exp tensor
        :type mAccLSE: cute.Tensor
        :param split_kv: The split_kv parameter
        :type split_kv: cutlass.Int32
        :param cache_seqs: The absolute decode positions tensor
        :type cache_seqs: cute.Tensor
        :param window_valid_lens: The initialized rolling-window row counts
        :type window_valid_lens: cute.Tensor
        :param block_split_kvs: The per-block split_kv values tensor
        :type block_split_kvs: cute.Tensor
        :param softmax_scale_log2: The log2 scale factor for softmax
        :type softmax_scale_log2: cutlass.Float32
        :param output_scale: The scale factor for the output
        :type output_scale: cutlass.Float32
        :param attn_sink_unscaled: Per-head attention-sink logit
            (unscaled-S space). Shape [num_heads]. Each thread looks up the
            sink for its M-row only on the first split block.
        :type attn_sink_unscaled: cute.Tensor
        :param q_latent_smem_layout_staged: Shared memory layout for query latent tensor
        :type q_latent_smem_layout_staged: cute.ComposedLayout
        :param kc_smem_layout_staged: Shared memory layout for key/value latent tensor
        :type kc_smem_layout_staged: cute.ComposedLayout
        :param p_smem_layout_staged: Shared memory layout for probability matrix
        :type p_smem_layout_staged: cute.ComposedLayout
        :param vc_smem_layout_staged: Shared memory layout for value tensor
        :type vc_smem_layout_staged: cute.ComposedLayout
        :param kc_smem_layout_for_tma_win: Shared memory layout for window K
            tensor for TMA
        :type kc_smem_layout_for_tma_win: cute.ComposedLayout
        :param kc_smem_layout_for_tma_cmp: Shared memory layout for compressed
            K tensor for TMA
        :type kc_smem_layout_for_tma_cmp: cute.ComposedLayout
        :param vc_smem_layout_for_tma_win: Shared memory layout for window V
            tensor for TMA
        :type vc_smem_layout_for_tma_win: cute.ComposedLayout
        :param vc_smem_layout_for_tma_cmp: Shared memory layout for compressed
            V tensor for TMA
        :type vc_smem_layout_for_tma_cmp: cute.ComposedLayout
        :param cta_layout_vmnk: Layout for compute threads
        :type cta_layout_vmnk: cute.Layout
        :param tile_sched_params: Scheduling parameters for work distribution
        :type tile_sched_params: HCAStaticTileSchedulerParams
        :param SharedStorage: Shared storage for the kernel
        :type SharedStorage: cutlass.Constexpr
        """

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma_qk.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0

        # Prefetch tma descriptor
        if warp_idx == self.mma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_q_latent)
            cpasync.prefetch_descriptor(tma_atom_c_latent_win)
            cpasync.prefetch_descriptor(tma_atom_c_latent_cmp)
            cpasync.prefetch_descriptor(tma_atom_c_latent_transpose_win)
            cpasync.prefetch_descriptor(tma_atom_c_latent_transpose_cmp)

        # Alloc
        smem = cutlass_memory.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Tensor memory dealloc barrier init
        tmem = cutlass_memory.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_ptr_sync_bar,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
            arch=self.arch_str,
        )

        load_q_pipeline = self.make_and_init_load_qkv_pipeline(
            storage.load_q_mbar_ptr.data_ptr(),
            cta_layout_vmnk,
            self.load_q_stage,
            self.tma_copy_q_bytes,
        )
        load_kv_pipeline = self.make_and_init_load_qkv_pipeline(
            storage.load_kv_mbar_ptr.data_ptr(),
            cta_layout_vmnk,
            self.load_kv_stage,
            self.tma_copy_kc_bytes,
        )
        mma_s_pipeline = self.make_and_init_mma_s_pipeline(
            storage.mma_s_mbar_ptr.data_ptr(), cta_layout_vmnk
        )
        p_mma_pipeline = self.make_and_init_p_mma_pipeline(
            storage.p_mma_mbar_ptr.data_ptr(), cta_layout_vmnk
        )
        p_cor_pipeline = self.make_and_init_p_cor_pipeline(storage.p_cor_mbar_ptr.data_ptr())
        mma_o_pipeline = self.make_and_init_mma_o_pipeline(
            storage.mma_o_mbar_ptr.data_ptr(), cta_layout_vmnk
        )
        load_pt_pipeline = self.make_and_init_load_pt_pipeline(storage.load_pt_mbar_ptr.data_ptr())

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk, is_relaxed=True)

        # Generate smem tensor Q/KC/VC/exchange
        # (MMA, MMA_H, MMA_R, PIPE)
        sQ = storage.smem_q_latent.get_tensor(
            q_latent_smem_layout_staged.outer, swizzle=q_latent_smem_layout_staged.inner
        )
        # (MMA, MMA_K, MMA_R, PIPE)
        sKC = storage.smem_kc.get_tensor(
            kc_smem_layout_staged.outer, swizzle=kc_smem_layout_staged.inner
        )
        # Two TMA-views over the same physical SMEM, one per page-size
        sKC_for_tma_win = storage.smem_kc.get_tensor(
            kc_smem_layout_for_tma_win.outer,
            swizzle=kc_smem_layout_for_tma_win.inner,
        )
        sKC_for_tma_cmp = storage.smem_kc.get_tensor(
            kc_smem_layout_for_tma_cmp.outer,
            swizzle=kc_smem_layout_for_tma_cmp.inner,
        )
        # (MMA, MMA_D, MMA_K, PIPE)
        # reuse smem
        sVC_ptr = cute.recast_ptr(sKC.iterator, vc_smem_layout_staged.inner)
        sVC = cute.make_tensor(sVC_ptr, vc_smem_layout_staged.outer)
        sVC_for_tma_win = cute.make_tensor(sVC_ptr, vc_smem_layout_for_tma_win.outer)
        sVC_for_tma_cmp = cute.make_tensor(sVC_ptr, vc_smem_layout_for_tma_cmp.outer)
        # (MMA, MMA_H, MMA_K)
        sP = storage.smem_p.get_tensor(
            p_smem_layout_staged.outer, swizzle=p_smem_layout_staged.inner
        )
        sPT = storage.smem_page_table.get_tensor(
            cute.make_layout((self.mma_qk_tiler[1] // 2, self.load_pt_stage))
        )
        # (compute_threads,)
        softmax_smem_exchange = storage.softmax_smem_exchange.get_tensor(
            cute.make_layout(self.num_compute_warps * self.threads_per_warp)
        )
        epilogue_smem_exchange = storage.epilogue_smem_exchange.get_tensor(
            cute.make_layout(self.num_compute_warps * self.threads_per_warp)
        )

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Load warps, including page table and data tensors
        # ///////////////////////////////////////////////////////////////////////////////

        if warp_idx >= self.empty_warp_ids[0] and warp_idx <= self.empty_warp_ids[-1]:
            cute.arch.setmaxregister_decrease(self.other_reg_num)
        if warp_idx == self.load_pt_warp_id:
            cute.arch.setmaxregister_decrease(self.other_reg_num)
            load_pt_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.load_pt_stage
            )
            tile_sched = create_hca_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                blk_coord = work_tile.tile_idx
                k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                    split_kv,
                    cache_seqs,
                    block_split_kvs,
                    blk_coord,
                )
                if k_tile_count > 0:
                    load_pt_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        load_pt_pipeline=load_pt_pipeline,
                        mPT_win=mPT_win,
                        mPT_cmp=mPT_cmp,
                        sPT=sPT,
                        tidx=tidx,
                    )
                    load_pt_producer_state = self.load_page_table(
                        load_pt_common_params,
                        k_index,
                        k_tile_count,
                        load_pt_producer_state,
                    )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            load_pt_pipeline.producer_tail(load_pt_producer_state)
        if warp_idx == self.load_tma_warp_id:
            cute.arch.setmaxregister_decrease(self.other_reg_num)
            load_q_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.load_q_stage
            )
            load_kv_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.load_kv_stage
            )
            load_pt_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.load_pt_stage
            )
            load_pt_release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.load_pt_stage
            )
            tile_sched = create_hca_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                blk_coord = work_tile.tile_idx
                k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                    split_kv,
                    cache_seqs,
                    block_split_kvs,
                    blk_coord,
                )
                if k_tile_count > 0:
                    # Construct fixed common/tma_qk/tma_pv params for load_tma
                    tma_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        local_split_kv=local_split_kv,
                        load_q_pipeline=load_q_pipeline,
                        load_kv_pipeline=load_kv_pipeline,
                        mPT_win=mPT_win,
                        mPT_cmp=mPT_cmp,
                        sPT=sPT,
                        load_pt_pipeline=load_pt_pipeline,
                    )
                    tma_qk_params = SimpleNamespace(
                        tiled_mma_qk=tiled_mma_qk,
                        tma_atom_q_latent=tma_atom_q_latent,
                        tma_atom_c_latent_win=tma_atom_c_latent_win,
                        tma_atom_c_latent_cmp=tma_atom_c_latent_cmp,
                        mQL=mQL,
                        mCL_win=mCL_win,
                        mCL_cmp=mCL_cmp,
                        sQ=sQ,
                        sKC_win=sKC_for_tma_win,
                        sKC_cmp=sKC_for_tma_cmp,
                    )
                    tma_pv_params = SimpleNamespace(
                        tiled_mma_pv=tiled_mma_pv,
                        tma_atom_c_latent_transpose_win=tma_atom_c_latent_transpose_win,
                        tma_atom_c_latent_transpose_cmp=tma_atom_c_latent_transpose_cmp,
                        mCL_win=mCL_win,
                        mCL_cmp=mCL_cmp,
                        mCLT_win=mCLT_win,
                        mCLT_cmp=mCLT_cmp,
                        sVC_win=sVC_for_tma_win,
                        sVC_cmp=sVC_for_tma_cmp,
                    )
                    # Load tma
                    (
                        load_q_producer_state,
                        load_kv_producer_state,
                        load_pt_consumer_state,
                        load_pt_release_state,
                    ) = self.load_tma(
                        tma_common_params,
                        tma_qk_params,
                        tma_pv_params,
                        k_index,
                        k_tile_count,
                        load_q_producer_state,
                        load_kv_producer_state,
                        load_pt_consumer_state,
                        load_pt_release_state,
                    )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            load_q_pipeline.producer_tail(load_q_producer_state)
            load_kv_pipeline.producer_tail(load_kv_producer_state)

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA warp
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.other_reg_num)
            # Alloc tensor memory buffer
            tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            load_q_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.load_q_stage
            )
            load_kv_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.load_kv_stage
            )
            mma_s_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.mma_s_stage
            )
            p_mma_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.p_mma_stage
            )
            mma_o_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.mma_o_stage
            )
            tile_sched = create_hca_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                blk_coord = work_tile.tile_idx
                k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                    split_kv, cache_seqs, block_split_kvs, blk_coord
                )
                if k_tile_count > 0:
                    mma_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        local_split_kv=local_split_kv,
                        load_q_pipeline=load_q_pipeline,
                        load_kv_pipeline=load_kv_pipeline,
                        tmem_ptr=tmem_ptr,
                        is_leader_cta=is_leader_cta,
                        L=mCL_cmp.shape[1],
                    )
                    mma_qk_params = SimpleNamespace(
                        mma_s_pipeline=mma_s_pipeline,
                        sQ=sQ,
                        sKC=sKC,
                    )
                    mma_pv_params = SimpleNamespace(
                        p_mma_pipeline=p_mma_pipeline,
                        mma_o_pipeline=mma_o_pipeline,
                        sP=sP,
                        sVC=sVC,
                    )
                    (
                        tiled_mma_qk,
                        tiled_mma_pv,
                        load_q_consumer_state,
                        load_kv_consumer_state,
                        mma_s_producer_state,
                        p_mma_consumer_state,
                        mma_o_producer_state,
                    ) = self.mma(
                        mma_common_params,
                        mma_qk_params,
                        mma_pv_params,
                        k_tile_count,
                        tiled_mma_qk,
                        tiled_mma_pv,
                        load_q_consumer_state,
                        load_kv_consumer_state,
                        mma_s_producer_state,
                        p_mma_consumer_state,
                        mma_o_producer_state,
                    )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            mma_s_pipeline.producer_tail(mma_s_producer_state)
            mma_o_pipeline.producer_tail(mma_o_producer_state)

            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Compute warp
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.compute_warp_ids[0] and warp_idx <= self.compute_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.softmax_reg_num)
            mma_s_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mma_s_stage
            )
            p_mma_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.p_mma_stage
            )
            p_cor_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.p_cor_stage
            )
            mma_o_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mma_o_stage
            )
            # sync with mma warp before retrieving tmem ptr
            tmem.wait_for_alloc()

            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            tile_sched = create_hca_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                blk_coord = work_tile.tile_idx
                k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                    split_kv, cache_seqs, block_split_kvs, blk_coord
                )
                if k_tile_count > 0:
                    compute_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        split_kv=split_kv,
                        local_split_kv=local_split_kv,
                        smem_exchange=softmax_smem_exchange,
                        mAccO=mAccO,
                        mO=mO,
                        K=self.get_cache_seq_len(cache_seqs, blk_coord[2]),
                        window_valid_len=self.get_window_valid_len(window_valid_lens, blk_coord[2]),
                        window_end_pos=self.get_window_end_pos(cache_seqs, blk_coord[2]),
                        L=mCL_cmp.shape[1],
                        tmem_ptr=tmem_ptr,
                        tidx=tidx,
                        p_cor_pipeline=p_cor_pipeline,
                        attn_sink_unscaled=attn_sink_unscaled,
                    )
                    compute_softmax_params = SimpleNamespace(
                        tiled_mma_qk=tiled_mma_qk,
                        sP=sP,
                        mma_s_pipeline=mma_s_pipeline,
                        p_mma_pipeline=p_mma_pipeline,
                        softmax_scale_log2=softmax_scale_log2,
                    )
                    mma_s_consumer_state, p_mma_producer_state, p_cor_producer_state = self.compute(
                        compute_common_params,
                        compute_softmax_params,
                        k_index=k_index,
                        k_tile_count=k_tile_count,
                        mma_s_consumer_state=mma_s_consumer_state,
                        p_mma_producer_state=p_mma_producer_state,
                        p_cor_producer_state=p_cor_producer_state,
                    )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            p_cor_pipeline.producer_tail(p_cor_producer_state)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction warp
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx <= self.correction_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.correction_reg_num)
            p_cor_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.p_cor_stage
            )
            mma_o_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mma_o_stage
            )
            # sync with mma warp before retrieving tmem ptr
            tmem.wait_for_alloc()

            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            tile_sched = create_hca_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                blk_coord = work_tile.tile_idx
                k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                    split_kv, cache_seqs, block_split_kvs, blk_coord
                )
                if k_tile_count > 0:
                    compute_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        split_kv=split_kv,
                        local_split_kv=local_split_kv,
                        smem_exchange=epilogue_smem_exchange,
                        mAccO=mAccO,
                        mO=mO,
                        K=self.get_cache_seq_len(cache_seqs, blk_coord[2]),
                        L=mCL_cmp.shape[1],
                        H=mQL.shape[0],
                        tmem_ptr=tmem_ptr,
                        tidx=tidx,
                        tiled_mma_pv=tiled_mma_pv,
                        p_cor_pipeline=p_cor_pipeline,
                        mma_o_pipeline=mma_o_pipeline,
                    )
                    compute_epilogue_params = SimpleNamespace(
                        output_scale=output_scale,
                        softmax_scale_log2=softmax_scale_log2,
                        mAccLSE=mAccLSE,
                        mLSE=mLSE,
                        mFreqs=inverse_rope_freqs,
                    )
                    p_cor_consumer_state, mma_o_consumer_state = self.correction(
                        compute_common_params,
                        compute_epilogue_params,
                        k_tile_count=k_tile_count,
                        p_cor_consumer_state=p_cor_consumer_state,
                        mma_o_consumer_state=mma_o_consumer_state,
                    )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        return

    @cute.jit
    def get_cache_seq_len(
        self, cache_seqs: Optional[cute.Tensor], batch_idx: cutlass.Int32
    ) -> cutlass.Int32:
        """Return a specialization constant or the runtime cache length."""
        if cutlass.const_expr(self.fixed_cache_seq_len is not None):
            return cutlass.Int32(self.fixed_cache_seq_len)
        return cache_seqs[batch_idx]

    @cute.jit
    def get_window_valid_len(
        self, window_valid_lens: Optional[cute.Tensor], batch_idx: cutlass.Int32
    ) -> cutlass.Int32:
        """Return the number of initialized rows in the window stream."""
        if cutlass.const_expr(self.window_valid_len_from_tensor):
            return min(
                max(cutlass.Int32(window_valid_lens[batch_idx]), cutlass.Int32(0)),
                cutlass.Int32(self.tma_page_size_win),
            )
        return cutlass.Int32(self.tma_page_size_win)

    @cute.jit
    def get_window_end_pos(
        self, cache_seqs: Optional[cute.Tensor], batch_idx: cutlass.Int32
    ) -> cutlass.Int32:
        """Return the absolute position stored at the window suffix end."""
        if cutlass.const_expr(self.window_valid_len_from_tensor):
            return cutlass.Int32(cache_seqs[batch_idx])
        return cutlass.Int32(0)

    @cute.jit
    def is_score_valid(
        self,
        local_col: cutlass.Int32,
        k_index: cutlass.Int32,
        total_seq_len: cutlass.Int32,
        window_valid_len: cutlass.Int32,
        window_end_pos: cutlass.Int32,
    ) -> cutlass.Boolean:
        """Return the stream-aware softmax mask for a score column."""
        valid = cute.elem_less(
            local_col + self.mma_qk_tiler[1] * k_index,
            total_seq_len,
        )
        if cutlass.const_expr(self.window_valid_len_from_tensor):
            age = (
                window_end_pos - local_col + cutlass.Int32(self.tma_page_size_win)
            ) % cutlass.Int32(self.tma_page_size_win)
            valid = valid and (k_index != 0 or cute.elem_less(age, window_valid_len))
        return valid

    @cute.jit
    def get_k_tile_count(
        self,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        blk_coord: cute.Coord,
    ) -> tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]:
        """Get the current k_index, k_tile_count, and local split_kv value for the HCA kernel.

        :param split_kv: Split_kv value
        :type split_kv: cutlass.Int32
        :param cache_seqs: Cache sequence lengths tensor
        :type cache_seqs: cute.Tensor
        :param block_split_kvs: Per-block split_kv values tensor
        :type block_split_kvs: cute.Tensor
        :param blk_coord: Block coordinate
        :type blk_coord: cute.Coord
        :return: k_index, k_tile_count, split_kv
        :rtype: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]
        """
        K = self.get_cache_seq_len(cache_seqs, blk_coord[2])
        if cutlass.const_expr(self.is_var_split_kv):
            split_kv = block_split_kvs[blk_coord[2]]

        k_tile_total = cute.ceil_div(K, self.mma_qk_tiler[1])
        k_tile_per_cta = cute.ceil_div(k_tile_total, split_kv)
        k_index = blk_coord[3] * k_tile_per_cta
        k_tile_count = max(0, min(k_tile_total, k_index + k_tile_per_cta) - k_index)
        return k_index, k_tile_count, split_kv

    @cute.jit
    def load_page_table(
        self,
        common_params: SimpleNamespace,
        k_index: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        load_pt_producer_state: pipeline.PipelineState,
    ) -> pipeline.PipelineState:
        """Load warp to load page table. Updates the load pt producer state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param k_index: The k index
        :type k_index: cutlass.Int32
        :param k_tile_count: The k tile count
        :type k_tile_count: cutlass.Int32
        :param load_pt_producer_state: The load pt producer state
        :type load_pt_producer_state: pipeline.PipelineState

        :return: The load pt producer state
        :rtype: pipeline.PipelineState
        """
        # k_index == 0 → window stream (one tile worth of indices from mPT_win)
        # k_index >= 1 → compressed stream (offset k_index-1 into mPT_cmp)
        mPT_win = common_params.mPT_win[None, common_params.blk_coord[2]]
        mPT_cmp = common_params.mPT_cmp[None, common_params.blk_coord[2]]
        page_per_tile_win = self.mma_qk_tiler[1] // self.tma_page_size_win
        page_per_tile_cmp = self.mma_qk_tiler[1] // self.tma_page_size_cmp
        # Loop bound covers whichever stream needs more indices per K-tile
        # so that sPT is fully populated under either page-size ordering.
        page_per_tile_max = max(page_per_tile_win, page_per_tile_cmp)
        tidx = common_params.tidx % self.threads_per_warp

        load_pt_pipeline = common_params.load_pt_pipeline
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS),
            cutlass.Int32,
            num_bits_per_copy=cutlass.Int32.width,
        )
        mPT_win_for_copy = cute.flat_divide(mPT_win, (1,))
        mPT_cmp_for_copy = cute.flat_divide(mPT_cmp, (1,))
        sPT_for_copy = cute.flat_divide(common_params.sPT, (1,))
        elem_per_thread = cute.ceil_div(page_per_tile_max, self.threads_per_warp)

        while k_tile_count > 0:
            load_pt_pipeline.producer_acquire(load_pt_producer_state)
            is_win = k_index == 0
            cmp_offset = k_index - 1
            for i in range(elem_per_thread):
                idx = i * self.threads_per_warp + tidx
                if is_win:
                    if cute.elem_less(idx, mPT_win.shape[0]) and cute.elem_less(
                        idx, page_per_tile_win
                    ):
                        cute.copy(
                            atom_async_copy,
                            mPT_win_for_copy[None, idx],
                            sPT_for_copy[None, idx, load_pt_producer_state.index],
                        )
                    else:
                        sPT_for_copy[None, idx, load_pt_producer_state.index].fill(0)
                else:
                    if cutlass.const_expr(self.implicit_cmp_page_table):
                        if cute.elem_less(idx, page_per_tile_cmp):
                            sPT_for_copy[None, idx, load_pt_producer_state.index].fill(
                                common_params.blk_coord[2]
                            )
                        else:
                            sPT_for_copy[None, idx, load_pt_producer_state.index].fill(0)
                    else:
                        if cute.elem_less(
                            cmp_offset * page_per_tile_cmp + idx, mPT_cmp.shape[0]
                        ) and cute.elem_less(idx, page_per_tile_cmp):
                            cute.copy(
                                atom_async_copy,
                                mPT_cmp_for_copy[None, cmp_offset * page_per_tile_cmp + idx],
                                sPT_for_copy[None, idx, load_pt_producer_state.index],
                            )
                        else:
                            sPT_for_copy[None, idx, load_pt_producer_state.index].fill(0)
            load_pt_pipeline.producer_commit(load_pt_producer_state)
            load_pt_producer_state.advance()
            k_index += 1
            k_tile_count -= 1

        return load_pt_producer_state

    @cute.jit
    def load_tma(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        v_params: SimpleNamespace,
        k_index: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        load_q_producer_state: pipeline.PipelineState,
        load_kv_producer_state: pipeline.PipelineState,
        load_pt_consumer_state: pipeline.PipelineState,
        load_pt_release_state: pipeline.PipelineState,
    ) -> tuple[
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
    ]:
        """Load wrap to load Q/C latent tensors. Updates the load qkv producer state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param qk_params: The qk parameters
        :type qk_params: SimpleNamespace
        :param v_params: The v parameters
        :type v_params: SimpleNamespace
        :param k_index: The k index
        :type k_index: cutlass.Int32
        :param k_tile_count: The k tile count
        :type k_tile_count: cutlass.Int32
        :param load_q_producer_state: The load q producer state
        :type load_q_producer_state: pipeline.PipelineState
        :param load_kv_producer_state: The load kv producer state
        :type load_kv_producer_state: pipeline.PipelineState
        :param load_pt_consumer_state: The load pt consumer state
        :type load_pt_consumer_state: pipeline.PipelineState
        :param load_pt_release_state: The load pt release state
        :type load_pt_release_state: pipeline.PipelineState

        :return: The load q producer state, load kv producer state, load pt consumer state, and load pt release state
        :rtype: tuple[pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState]
        """
        # === Q partition (single Q stream) ===
        mma_qk_tiler_mk = cute.select(self.mma_qk_tiler, mode=[0, 2])
        gQL = cute.flat_divide(qk_params.mQL, mma_qk_tiler_mk)

        thr_mma_qk = qk_params.tiled_mma_qk.get_slice(
            common_params.blk_coord[0] % cute.size(qk_params.tiled_mma_qk.thr_id)
        )
        tSgQL = thr_mma_qk.partition_A(gQL)

        cta_kv_m = qk_params.tiled_mma_qk.op.shape_mnk[0] // qk_params.tiled_mma_qk.thr_id.shape

        # === K partition for window stream ===
        cta_m_win = min(cta_kv_m, self.tma_page_size_win)
        page_tile_size_k_win = min(self.tma_page_size_win, cta_m_win)
        gCL_win = cute.tiled_divide(qk_params.mCL_win, (page_tile_size_k_win, self.mma_qk_tiler[2]))
        tSgCL_win = (
            gCL_win[
                None,
                common_params.blk_coord[0] % qk_params.tiled_mma_qk.thr_id.shape,
                None,
                None,
            ]
            if cta_m_win < self.tma_page_size_win
            else gCL_win[None, 0, None, None]
        )

        # === K partition for compressed stream ===
        cta_m_cmp = min(cta_kv_m, self.tma_page_size_cmp)
        page_tile_size_k_cmp = min(self.tma_page_size_cmp, cta_m_cmp)
        gCL_cmp = cute.tiled_divide(qk_params.mCL_cmp, (page_tile_size_k_cmp, self.mma_qk_tiler[2]))
        tSgCL_cmp = (
            gCL_cmp[
                None,
                common_params.blk_coord[0] % qk_params.tiled_mma_qk.thr_id.shape,
                None,
                None,
            ]
            if cta_m_cmp < self.tma_page_size_cmp
            else gCL_cmp[None, 0, None, None]
        )

        # tma partition for q (one stream) and k (two streams sharing SMEM)
        tQsQ, tQLgQL_mkl = cpasync.tma_partition(
            qk_params.tma_atom_q_latent,
            0,
            cute.make_layout(1),
            cute.group_modes(qk_params.sQ, 0, 3),
            cute.group_modes(tSgQL, 0, 3),
        )

        tKCsKC_win, tCLgCL_win = cpasync.tma_partition(
            qk_params.tma_atom_c_latent_win,
            0,
            cute.make_layout(1),
            qk_params.sKC_win,
            tSgCL_win,
        )
        tKCsKC_cmp, tCLgCL_cmp = cpasync.tma_partition(
            qk_params.tma_atom_c_latent_cmp,
            0,
            cute.make_layout(1),
            qk_params.sKC_cmp,
            tSgCL_cmp,
        )

        tQLgQL = tQLgQL_mkl[
            None, None, None, common_params.blk_coord[1], common_params.blk_coord[2]
        ]

        # === V partition for window stream ===
        page_tile_size_v_win = min(self.tma_page_size_win, self.mma_pv_tiler[2])
        gCLT_win = cute.flat_divide(v_params.mCLT_win, (self.mma_pv_tiler[1], page_tile_size_v_win))
        cta_n = self.mma_pv_tiler[1] // v_params.tiled_mma_pv.thr_id.shape
        gCLT_win = cute.logical_divide(gCLT_win, (cta_n,))[
            (None, common_params.blk_coord[0]), None, None, None, None
        ]
        tOgCLT_win = cute.tiled_divide(gCLT_win, (cta_n, page_tile_size_v_win))
        tOgCLT_win = tOgCLT_win[None, 0, 0, None, None, None]

        # === V partition for compressed stream ===
        page_tile_size_v_cmp = min(self.tma_page_size_cmp, self.mma_pv_tiler[2])
        gCLT_cmp = cute.flat_divide(v_params.mCLT_cmp, (self.mma_pv_tiler[1], page_tile_size_v_cmp))
        gCLT_cmp = cute.logical_divide(gCLT_cmp, (cta_n,))[
            (None, common_params.blk_coord[0]), None, None, None, None
        ]
        tOgCLT_cmp = cute.tiled_divide(gCLT_cmp, (cta_n, page_tile_size_v_cmp))
        tOgCLT_cmp = tOgCLT_cmp[None, 0, 0, None, None, None]

        tVCsVC_win, tCLTgCLT_win = cpasync.tma_partition(
            v_params.tma_atom_c_latent_transpose_win,
            0,
            cute.make_layout(1),
            v_params.sVC_win,
            tOgCLT_win,
        )
        tVCsVC_cmp, tCLTgCLT_cmp = cpasync.tma_partition(
            v_params.tma_atom_c_latent_transpose_cmp,
            0,
            cute.make_layout(1),
            v_params.sVC_cmp,
            tOgCLT_cmp,
        )

        # set extra params (both streams threaded through)
        qk_params.tQLgQL = tQLgQL
        qk_params.tCLgCL_win = tCLgCL_win
        qk_params.tCLgCL_cmp = tCLgCL_cmp
        qk_params.tQsQ = tQsQ
        qk_params.tKCsKC_win = tKCsKC_win
        qk_params.tKCsKC_cmp = tKCsKC_cmp
        v_params.tCLTgCLT_win = tCLTgCLT_win
        v_params.tCLTgCLT_cmp = tCLTgCLT_cmp
        v_params.tVCsVC_win = tVCsVC_win
        v_params.tVCsVC_cmp = tVCsVC_cmp

        load_q_producer_state, load_kv_producer_state, load_pt_consumer_state = (
            self.load_tma_qk_one_k_tile(
                common_params,
                qk_params,
                k_index,
                k_tile_count,
                load_q_producer_state,
                load_kv_producer_state,
                load_pt_consumer_state,
                load_q=True,
            )
        )
        k_index += 1
        k_tile_count -= 1
        while k_tile_count > 0:
            load_q_producer_state, load_kv_producer_state, load_pt_consumer_state = (
                self.load_tma_qk_one_k_tile(
                    common_params,
                    qk_params,
                    k_index,
                    k_tile_count,
                    load_q_producer_state,
                    load_kv_producer_state,
                    load_pt_consumer_state,
                    load_q=False,
                )
            )
            load_kv_producer_state, load_pt_release_state = self.load_tma_v_one_k_tile(
                common_params,
                v_params,
                k_index - 1,
                load_kv_producer_state,
                load_pt_release_state,
            )
            k_index += 1
            k_tile_count -= 1

        # load last v tile
        load_kv_producer_state, load_pt_release_state = self.load_tma_v_one_k_tile(
            common_params,
            v_params,
            k_index - 1,
            load_kv_producer_state,
            load_pt_release_state,
        )
        return (
            load_q_producer_state,
            load_kv_producer_state,
            load_pt_consumer_state,
            load_pt_release_state,
        )

    @cute.jit
    def load_tma_qk_one_k_tile(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        k_index: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        load_q_producer_state: pipeline.PipelineState,
        load_kv_producer_state: pipeline.PipelineState,
        load_pt_consumer_state: pipeline.PipelineState,
        load_q: bool,
    ) -> tuple[pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState]:
        """Load one k-tile of Q/C latent tensors. Updates the load qkv producer state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param qk_params: The qk parameters
        :type qk_params: SimpleNamespace
        :param k_index: The k index
        :type k_index: cutlass.Int32
        :param k_tile_count: The k tile count
        :type k_tile_count: cutlass.Int32
        :param load_q_producer_state: The load q producer state
        :type load_q_producer_state: pipeline.PipelineState
        :param load_kv_producer_state: The load kv producer state
        :type load_kv_producer_state: pipeline.PipelineState
        :param load_pt_consumer_state: The load pt consumer state
        :type load_pt_consumer_state: pipeline.PipelineState
        :param load_q: Whether to load q
        :type load_q: bool

        :return: The load q producer state, load kv producer state, and load pt consumer state
        :rtype: tuple[pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState]
        """
        page_per_tile_win = ceil_div(
            self.mma_qk_tiler[1] // self.tma_page_size_win,
            qk_params.tiled_mma_qk.thr_id.shape,
        )
        page_per_tile_cmp = ceil_div(
            self.mma_qk_tiler[1] // self.tma_page_size_cmp,
            qk_params.tiled_mma_qk.thr_id.shape,
        )
        # Either stream may have the larger per-CTA page count; size the
        # rmem buffer to accommodate both.
        page_per_tile_max = max(page_per_tile_win, page_per_tile_cmp)

        common_params.load_pt_pipeline.consumer_wait(load_pt_consumer_state)
        page_table_stage = load_pt_consumer_state.index
        load_pt_consumer_state.advance()
        k_idx = cute.make_rmem_tensor(cute.make_layout(page_per_tile_max), cutlass.Int32)
        # Win and cmp may have different page_size, so the "all-CTAs share one
        # page" vs "each CTA reads its own page" choice must be made per
        # stream. (When page_size == mma_qk_tiler[1] there is exactly one
        # page covering one K-tile shared across CTAs; otherwise pages are
        # distributed.)
        is_win_for_idx = k_index == 0
        if is_win_for_idx:
            if cutlass.const_expr(self.mma_qk_tiler[1] // self.tma_page_size_win == 1):
                for i in cutlass.range_constexpr(page_per_tile_max):
                    k_idx[i] = common_params.sPT[0, page_table_stage]
            else:
                for i in cutlass.range_constexpr(page_per_tile_max):
                    k_idx[i] = common_params.sPT[
                        i + common_params.blk_coord[0] * page_per_tile_max,
                        page_table_stage,
                    ]
        else:
            if cutlass.const_expr(self.mma_qk_tiler[1] // self.tma_page_size_cmp == 1):
                for i in cutlass.range_constexpr(page_per_tile_max):
                    k_idx[i] = common_params.sPT[0, page_table_stage]
            else:
                for i in cutlass.range_constexpr(page_per_tile_max):
                    k_idx[i] = common_params.sPT[
                        i + common_params.blk_coord[0] * page_per_tile_max,
                        page_table_stage,
                    ]
        # load q once at first iteration (single Q stream)
        if cutlass.const_expr(load_q):
            common_params.load_q_pipeline.producer_acquire(load_q_producer_state)
            tma_bar_ptr = common_params.load_q_pipeline.producer_get_barrier(load_q_producer_state)
            for i in cutlass.range(self.iterations_qk_latent):
                cute.copy(
                    qk_params.tma_atom_q_latent,
                    qk_params.tQLgQL[None, 0, i],
                    qk_params.tQsQ[None, (i, 0)],
                    tma_bar_ptr=tma_bar_ptr,
                )
            load_q_producer_state.advance()

        # K load: branch on stream (k_index == 0 → window, else compressed)
        is_win = k_index == 0
        load_kv_pipeline = common_params.load_kv_pipeline
        # Pre-init tma_bar_ptr so its type is established before the loop
        # (cute.range doesn't allow type changes inside the body).
        tma_bar_ptr = load_kv_pipeline.producer_get_barrier(load_kv_producer_state)
        for i in cutlass.range(self.iterations_qk_latent):
            tma_bar_ptr = load_kv_pipeline.producer_get_barrier(load_kv_producer_state)
            load_kv_pipeline.producer_acquire(load_kv_producer_state)
            if is_win:
                for k in cutlass.range(page_per_tile_win):
                    cute.copy(
                        qk_params.tma_atom_c_latent_win,
                        qk_params.tCLgCL_win[None, i, k_idx[k]],
                        qk_params.tKCsKC_win[None, k, 0, load_kv_producer_state.index],
                        tma_bar_ptr=tma_bar_ptr,
                    )
            else:
                for k in cutlass.range(page_per_tile_cmp):
                    cute.copy(
                        qk_params.tma_atom_c_latent_cmp,
                        qk_params.tCLgCL_cmp[None, i, k_idx[k]],
                        qk_params.tKCsKC_cmp[None, k, 0, load_kv_producer_state.index],
                        tma_bar_ptr=tma_bar_ptr,
                    )
            load_kv_producer_state.advance()

        return load_q_producer_state, load_kv_producer_state, load_pt_consumer_state

    @cute.jit
    def load_tma_v_one_k_tile(
        self,
        common_params: SimpleNamespace,
        v_params: SimpleNamespace,
        k_index: cutlass.Int32,
        load_kv_producer_state: pipeline.PipelineState,
        load_pt_release_state: pipeline.PipelineState,
    ) -> tuple[pipeline.PipelineState, pipeline.PipelineState]:
        """Load one k-tile of compressed latent transpose tensor(v). Updates the load qkv producer state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param v_params: The load tma v parameters
        :type v_params: SimpleNamespace
        :param k_index: The k index
        :type k_index: cutlass.Int32
        :param load_kv_producer_state: The load qkv producer state
        :type load_kv_producer_state: pipeline.PipelineState
        :param load_pt_release_state: The load pt release state
        :type load_pt_release_state: pipeline.PipelineState

        :return: The load kv producer state and load pt release state
        :rtype: tuple[pipeline.PipelineState, pipeline.PipelineState]
        """
        page_per_tile_win = self.mma_pv_tiler[2] * self.iterations_pv_k // self.tma_page_size_win
        page_per_tile_cmp = self.mma_pv_tiler[2] * self.iterations_pv_k // self.tma_page_size_cmp
        page_per_subtile_win = ceil_div(page_per_tile_win, self.iterations_pv_k)
        page_per_subtile_cmp = ceil_div(page_per_tile_cmp, self.iterations_pv_k)
        # Either stream may have the larger page count.
        page_per_tile_max = max(page_per_tile_win, page_per_tile_cmp)
        k_idx = cute.make_rmem_tensor(cute.make_layout(page_per_tile_max), cutlass.Int32)
        page_table_stage = load_pt_release_state.index
        for i in cutlass.range(page_per_tile_max):
            k_idx[i] = (
                common_params.sPT[0, page_table_stage]
                if page_per_tile_max == 1
                else common_params.sPT[i, page_table_stage]
            )
        common_params.load_pt_pipeline.consumer_release(load_pt_release_state)
        load_pt_release_state.advance()

        is_win = k_index == 0
        load_kv_pipeline = common_params.load_kv_pipeline
        # Pre-init tma_bar_ptr so its type is established before the loop
        # (cute.range doesn't allow type changes inside the body).
        tma_bar_ptr = load_kv_pipeline.producer_get_barrier(load_kv_producer_state)
        for i in cutlass.range(self.iterations_pv_k):
            for j in cutlass.range(self.iterations_pv_n):
                tma_bar_ptr = load_kv_pipeline.producer_get_barrier(load_kv_producer_state)
                load_kv_pipeline.producer_acquire(load_kv_producer_state)
                if is_win:
                    for k in cutlass.range(page_per_subtile_win):
                        k_idx_i = k_idx[
                            k
                            + i
                            // ceil_div(self.iterations_pv_k, page_per_tile_win)
                            * page_per_subtile_win
                        ]
                        cute.copy(
                            v_params.tma_atom_c_latent_transpose_win,
                            v_params.tCLTgCLT_win[
                                None,
                                j,
                                i % ceil_div(self.iterations_pv_k, page_per_tile_win),
                                k_idx_i,
                            ],
                            v_params.tVCsVC_win[None, 0, k, load_kv_producer_state.index],
                            tma_bar_ptr=tma_bar_ptr,
                        )
                else:
                    for k in cutlass.range(page_per_subtile_cmp):
                        k_idx_i = k_idx[
                            k
                            + i
                            // ceil_div(self.iterations_pv_k, page_per_tile_cmp)
                            * page_per_subtile_cmp
                        ]
                        cute.copy(
                            v_params.tma_atom_c_latent_transpose_cmp,
                            v_params.tCLTgCLT_cmp[
                                None,
                                j,
                                i % ceil_div(self.iterations_pv_k, page_per_tile_cmp),
                                k_idx_i,
                            ],
                            v_params.tVCsVC_cmp[None, 0, k, load_kv_producer_state.index],
                            tma_bar_ptr=tma_bar_ptr,
                        )

                load_kv_producer_state.advance()
        return load_kv_producer_state, load_pt_release_state

    @cute.jit
    def mma(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        pv_params: SimpleNamespace,
        k_tile_count: cutlass.Int32,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        load_q_consumer_state: pipeline.PipelineState,
        load_kv_consumer_state: pipeline.PipelineState,
        mma_s_producer_state: pipeline.PipelineState,
        p_mma_consumer_state: pipeline.PipelineState,
        mma_o_producer_state: pipeline.PipelineState,
    ) -> tuple[
        cute.TiledMma,
        cute.TiledMma,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
    ]:
        """MMA warp to compute the result of Q*K^T and P*V. Updates the tiled mma and pipeline states.

        :param common_params: The common parameters for mma qk and pv
        :type common_params: SimpleNamespace
        :param qk_params: The mma qk parameters
        :type qk_params: SimpleNamespace
        :param pv_params: The mma pv parameters
        :type pv_params: SimpleNamespace
        :param k_tile_count: The k tile count
        :type k_tile_count: cutlass.Int32
        :param tiled_mma_qk: The tiled mma qk
        :type tiled_mma_qk: cute.TiledMma
        :param tiled_mma_pv: The tiled mma pv
        :type tiled_mma_pv: cute.TiledMma
        :param load_q_consumer_state: The load q consumer state
        :type load_q_consumer_state: pipeline.PipelineState
        :param load_kv_consumer_state: The load kv consumer state
        :type load_kv_consumer_state: pipeline.PipelineState
        :param mma_s_producer_state: The mma s producer state
        :type mma_s_producer_state: pipeline.PipelineState
        :param p_mma_consumer_state: The p mma consumer state
        :type p_mma_consumer_state: pipeline.PipelineState
        :param mma_o_producer_state: The mma o producer state
        :type mma_o_producer_state: pipeline.PipelineState

        :return: The tiled mma qk, the tiled mma pv, the load q consumer state,
            the load kv consumer state, the mma s producer state, the p mma
            consumer state, and the mma o producer state
        :rtype: tuple[cute.TiledMma, cute.TiledMma, pipeline.PipelineState,
            pipeline.PipelineState, pipeline.PipelineState,
            pipeline.PipelineState, pipeline.PipelineState]
        """

        tSrQ = tiled_mma_qk.make_fragment_A(qk_params.sQ)
        tSrKC = tiled_mma_qk.make_fragment_B(qk_params.sKC)
        tOrP = tiled_mma_pv.make_fragment_A(pv_params.sP)
        tOrVC = tiled_mma_pv.make_fragment_B(pv_params.sVC)

        tStS_shape = tiled_mma_qk.partition_shape_C(cute.select(self.mma_qk_tiler, mode=[0, 1]))
        tStS_staged_fake = tiled_mma_qk.make_fragment_C(cute.append(tStS_shape, self.mma_s_stage))
        # use real tmem ptr for tStS
        tStS_staged = cute.make_tensor(common_params.tmem_ptr, tStS_staged_fake.layout)
        tOtO_shape = tiled_mma_pv.partition_shape_C(cute.select(self.mma_pv_tiler, mode=[0, 1]))
        # mma O has 1 stage.
        tOtO = tiled_mma_pv.make_fragment_C(tOtO_shape)
        tOtO_layout = cute.append(
            tOtO.layout,
            cute.make_layout(
                common_params.L // self.mma_pv_tiler[1],
                stride=self.mma_pv_tiler[1] // self.warps_in_n,
            ),
        )
        tOtO_staged = cute.make_tensor(tStS_staged.iterator + self.tmem_o_offset, tOtO_layout)

        # set more parameters
        qk_params.tSrQ = tSrQ
        qk_params.tSrKC = tSrKC
        qk_params.tStS_staged = tStS_staged
        pv_params.tOrP = tOrP
        pv_params.tOrVC = tOrVC
        pv_params.tOtO_staged = tOtO_staged

        # mma O accumulates on K, so the accumulate flag is set to False once before all K blocks.
        tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, False)
        load_q_pipeline = common_params.load_q_pipeline
        if common_params.is_leader_cta:
            load_q_release_state = load_q_consumer_state.clone()

            (
                tiled_mma_qk,
                load_q_consumer_state,
                load_kv_consumer_state,
                mma_s_producer_state,
            ) = self.mma_qk(
                common_params,
                qk_params,
                tiled_mma_qk,
                load_q_consumer_state,
                load_kv_consumer_state,
                mma_s_producer_state,
                wait_q=True,
            )
            k_tile_count -= 1
            while k_tile_count > 0:
                (
                    tiled_mma_qk,
                    load_q_consumer_state,
                    load_kv_consumer_state,
                    mma_s_producer_state,
                ) = self.mma_qk(
                    common_params,
                    qk_params,
                    tiled_mma_qk,
                    load_q_consumer_state,
                    load_kv_consumer_state,
                    mma_s_producer_state,
                    wait_q=False,
                )
                (
                    tiled_mma_pv,
                    load_kv_consumer_state,
                    p_mma_consumer_state,
                    mma_o_producer_state,
                ) = self.mma_pv(
                    common_params,
                    pv_params,
                    tiled_mma_pv,
                    load_kv_consumer_state,
                    p_mma_consumer_state,
                    mma_o_producer_state,
                )
                k_tile_count -= 1

            # release q consumer states
            load_q_pipeline.consumer_release(load_q_release_state)
            load_q_release_state.advance()
            (
                tiled_mma_pv,
                load_kv_consumer_state,
                p_mma_consumer_state,
                mma_o_producer_state,
            ) = self.mma_pv(
                common_params,
                pv_params,
                tiled_mma_pv,
                load_kv_consumer_state,
                p_mma_consumer_state,
                mma_o_producer_state,
            )

        return (
            tiled_mma_qk,
            tiled_mma_pv,
            load_q_consumer_state,
            load_kv_consumer_state,
            mma_s_producer_state,
            p_mma_consumer_state,
            mma_o_producer_state,
        )

    @cute.jit
    def mma_qk(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        tiled_mma_qk: cute.TiledMma,
        load_q_consumer_state: pipeline.PipelineState,
        load_kv_consumer_state: pipeline.PipelineState,
        mma_s_producer_state: pipeline.PipelineState,
        wait_q: bool,
    ) -> tuple[
        cute.TiledMma,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
    ]:
        """Compute one k-tile of mma for Q*K^T. Updates the tiled MMA QK and pipeline states.

        :param qk_params: The qk parameters
        :type qk_params: SimpleNamespace
        :param tiled_mma_qk: The tiled mma qk
        :type tiled_mma_qk: cute.TiledMma
        :param load_q_consumer_state: The load q consumer state
        :type load_q_consumer_state: pipeline.PipelineState
        :param load_kv_consumer_state: The load kv consumer state
        :type load_kv_consumer_state: pipeline.PipelineState
        :param mma_s_producer_state: The mma s producer state
        :type mma_s_producer_state: pipeline.PipelineState

        :return: The tiled mma qk, the load q consumer state, the load kv consumer state, and the mma s producer state
        :rtype: tuple[cute.TiledMma, pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState]
        """
        tStS = qk_params.tStS_staged[None, None, None, mma_s_producer_state.index]

        qk_params.mma_s_pipeline.producer_acquire(mma_s_producer_state)
        tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, False)
        load_q_pipeline = common_params.load_q_pipeline
        load_kv_pipeline = common_params.load_kv_pipeline
        if cutlass.const_expr(wait_q):
            load_q_pipeline.consumer_wait(load_q_consumer_state)
            load_q_consumer_state.advance()
        for q_stage in range(self.iterations_qk_latent):
            load_kv_pipeline.consumer_wait(load_kv_consumer_state)
            kc_stage = load_kv_consumer_state.index
            for k_block in cutlass.range(cute.size(qk_params.tSrQ.shape[2])):
                cute.gemm(
                    tiled_mma_qk,
                    tStS,
                    qk_params.tSrQ[None, None, k_block, q_stage],
                    qk_params.tSrKC[None, None, k_block, kc_stage],
                    tStS,
                )
                tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)
            load_kv_pipeline.consumer_release(load_kv_consumer_state)
            load_kv_consumer_state.advance()

        qk_params.mma_s_pipeline.producer_commit(mma_s_producer_state)
        mma_s_producer_state.advance()
        return (
            tiled_mma_qk,
            load_q_consumer_state,
            load_kv_consumer_state,
            mma_s_producer_state,
        )

    @cute.jit
    def mma_pv(
        self,
        common_params: SimpleNamespace,
        pv_params: SimpleNamespace,
        tiled_mma_pv: cute.TiledMma,
        load_kv_consumer_state: pipeline.PipelineState,
        p_mma_consumer_state: pipeline.PipelineState,
        mma_o_producer_state: pipeline.PipelineState,
    ) -> tuple[
        cute.TiledMma,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
    ]:
        """Compute one k-tile of mma for P*V. Updates the tiled mma pv and pipeline states.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param pv_params: The pv parameters
        :type pv_params: SimpleNamespace
        :param tiled_mma_pv: The tiled mma pv
        :type tiled_mma_pv: cute.TiledMma
        :param load_kv_consumer_state: The load kv consumer state
        :type load_kv_consumer_state: pipeline.PipelineState
        :param p_mma_consumer_state: The P MMA consumer state
        :type p_mma_consumer_state: pipeline.PipelineState
        :param mma_o_producer_state: The MMA o producer state
        :type mma_o_producer_state: pipeline.PipelineState

        :return: The tiled mma pv, the load qkv consumer state, the P MMA consumer state, and the MMA o producer state
        :rtype: tuple[cute.TiledMma, pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState]
        """

        pv_params.mma_o_pipeline.producer_acquire(mma_o_producer_state)
        pv_params.p_mma_pipeline.consumer_wait(p_mma_consumer_state)
        load_kv_pipeline = common_params.load_kv_pipeline
        for p_stage in range(self.iterations_pv_k):
            accumulate_flag = tiled_mma_pv.get(tcgen05.Field.ACCUMULATE)
            for acc_stage in range(self.iterations_pv_n):
                load_kv_pipeline.consumer_wait(load_kv_consumer_state)
                tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, accumulate_flag)
                vc_stage = load_kv_consumer_state.index
                tOtO = pv_params.tOtO_staged[None, None, None, acc_stage]
                for k_block in cutlass.range(pv_params.tOrP.shape[2]):
                    cute.gemm(
                        tiled_mma_pv,
                        tOtO,
                        pv_params.tOrP[
                            None,
                            None,
                            k_block,
                            (p_stage, p_mma_consumer_state.index),
                        ],
                        pv_params.tOrVC[None, None, k_block, vc_stage],
                        tOtO,
                    )
                    tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, True)
                load_kv_pipeline.consumer_release(load_kv_consumer_state)
                load_kv_consumer_state.advance()
        pv_params.p_mma_pipeline.consumer_release(p_mma_consumer_state)
        p_mma_consumer_state.advance()
        pv_params.mma_o_pipeline.producer_commit(mma_o_producer_state)
        mma_o_producer_state.advance()

        return (
            tiled_mma_pv,
            load_kv_consumer_state,
            p_mma_consumer_state,
            mma_o_producer_state,
        )

    @cute.jit
    def compute(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        k_index: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        mma_s_consumer_state: pipeline.PipelineState,
        p_mma_producer_state: pipeline.PipelineState,
        p_cor_producer_state: pipeline.PipelineState,
    ) -> tuple[pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState]:
        """Compute warp to compute the result of softmax, rescale, and epilogue. Updates the related pipeline states.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param softmax_params: The softmax parameters
        :type softmax_params: SimpleNamespace
        :param k_index: The index of the k-tile
        :type k_index: cutlass.Int32
        :param k_tile_count: The number of k-tiles
        :type k_tile_count: cutlass.Int32
        :param mma_s_consumer_state: The MMA s consumer state
        :type mma_s_consumer_state: pipeline.PipelineState
        :param p_mma_producer_state: The P MMA producer state
        :type p_mma_producer_state: pipeline.PipelineState
        :param p_cor_producer_state: The P correction producer state
        :type p_cor_producer_state: pipeline.PipelineState

        :return: The MMA s consumer state, the P MMA producer state, and the P correction producer state
        :rtype: tuple[pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState]
        """

        k_tile_total = cute.ceil_div(common_params.K, self.mma_qk_tiler[1])

        # Pre-compute this thread's per-head attention-sink, sized in the
        # unscaled-S space. We replicate softmax's tStS partition mechanics
        # so each thread can identify the M-row it tracks. All of this is
        # compile-time layout / coordinate work that touches CtaGroup-backed
        # objects (TiledMma) which cute's runtime `if` cannot flatten — so
        # it must live OUTSIDE the runtime branch below.
        cta_qk_tiler = (
            self.mma_qk_tiler[0] // self.cluster_shape_mnk[0],
            self.mma_qk_tiler[1],
            self.mma_qk_tiler[2],
        )
        cS = cute.make_identity_tensor(cute.select(cta_qk_tiler, mode=[0, 1]))
        tStS_shape = softmax_params.tiled_mma_qk.partition_shape_C(
            cute.select(self.mma_qk_tiler, mode=[0, 1])
        )
        tStS_staged_fake = softmax_params.tiled_mma_qk.make_fragment_C(
            cute.append(tStS_shape, self.mma_s_stage)
        )
        tStS_staged = cute.make_tensor(common_params.tmem_ptr, tStS_staged_fake.layout)
        tAcc_for_sink = tStS_staged[(None, None), 0, 0, 0]
        tidx_compute = common_params.tidx % (self.num_compute_warps * self.threads_per_warp)
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_load_atom, tAcc_for_sink)
        tmem_thr_copy = tmem_tiled_copy.get_slice(tidx_compute)
        tTR_tS_for_sink = tmem_thr_copy.partition_D(cS)
        local_row = tTR_tS_for_sink[0][0]
        head_idx = common_params.blk_coord[0] * cta_qk_tiler[0] + local_row
        my_sink = common_params.attn_sink_unscaled[head_idx]
        if cutlass.const_expr(self.attn_sink_is_scaled):
            my_sink = my_sink * LOG2_E / softmax_params.softmax_scale_log2

        # Online-softmax state. The very first split block (blk_coord[3] == 0)
        # absorbs the per-head attention-sink as a virtual extra logit (V=0):
        #   row_max := my_sink, row_sum := 1 / warps_in_n, O := 0.
        # The epilogue sums row_sum across N-partitioned warps, so distributing
        # the virtual sink's unit mass prevents it from being counted once per
        # warp (a visible normalization error for short windows).
        # Other split blocks start cold; the reduction kernel merges them.
        row_max = -self.acc_dtype.inf
        row_sum = self.acc_dtype(0)
        if common_params.blk_coord[3] == 0:
            row_max = my_sink
            row_sum = self.acc_dtype(1.0 / self.warps_in_n)
        correction_factor = self.acc_dtype(1)
        common_params.p_cor_pipeline.producer_acquire(p_cor_producer_state)

        # no mask applied
        while k_tile_count > 1:
            (
                mma_s_consumer_state,
                p_mma_producer_state,
                p_cor_producer_state,
                row_max,
                row_sum,
                correction_factor,
            ) = self.softmax(
                common_params,
                softmax_params,
                k_index,
                mma_s_consumer_state,
                p_mma_producer_state,
                p_cor_producer_state,
                row_max,
                row_sum,
                correction_factor,
                False,
                False,
            )
            k_index = k_index + 1
            k_tile_count = k_tile_count - 1

        # mask applied
        if cutlass.const_expr(common_params.mAccO is not None):
            (
                mma_s_consumer_state,
                p_mma_producer_state,
                p_cor_producer_state,
                row_max,
                row_sum,
                correction_factor,
            ) = self.softmax(
                common_params,
                softmax_params,
                k_index,
                mma_s_consumer_state,
                p_mma_producer_state,
                p_cor_producer_state,
                row_max,
                row_sum,
                correction_factor,
                k_index == k_tile_total - 1,
                True,
            )
        else:
            (
                mma_s_consumer_state,
                p_mma_producer_state,
                p_cor_producer_state,
                row_max,
                row_sum,
                correction_factor,
            ) = self.softmax(
                common_params,
                softmax_params,
                k_index,
                mma_s_consumer_state,
                p_mma_producer_state,
                p_cor_producer_state,
                row_max,
                row_sum,
                correction_factor,
                True,
                True,
            )

        return mma_s_consumer_state, p_mma_producer_state, p_cor_producer_state

    @cute.jit
    def correction(
        self,
        common_params: SimpleNamespace,
        epilogue_params: SimpleNamespace,
        k_tile_count: cutlass.Int32,
        p_cor_consumer_state: pipeline.PipelineState,
        mma_o_consumer_state: pipeline.PipelineState,
    ) -> tuple[pipeline.PipelineState, pipeline.PipelineState]:
        """Compute warp to compute the result of softmax, rescale, and epilogue. Updates the related pipeline states.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param epilogue_params: The epilogue parameters
        :type epilogue_params: SimpleNamespace
        :param k_index: The index of the k-tile
        :type k_index: cutlass.Int32
        :param k_tile_count: The number of k-tiles
        :type k_tile_count: cutlass.Int32
        :param p_cor_consumer_state: The P correction consumer state
        :type p_cor_consumer_state: pipeline.PipelineState
        :param mma_o_consumer_state: The MMA o consumer state
        :type mma_o_consumer_state: pipeline.PipelineState

        :return: The P correction consumer state, and the MMA o consumer state
        :rtype: tuple[pipeline.PipelineState, pipeline.PipelineState]
        """

        k_tile_count_init = k_tile_count
        while k_tile_count > 0:
            p_cor_consumer_state, row_sum, row_max, correction_factor, no_correction = (
                self.get_correction_factor(common_params, p_cor_consumer_state)
            )
            if k_tile_count_init != k_tile_count:
                mma_o_consumer_state = self.rescale(
                    common_params,
                    mma_o_consumer_state,
                    correction_factor,
                    no_correction,
                )
            k_tile_count = k_tile_count - 1
            if k_tile_count == 0:
                mma_o_consumer_state = self.epilogue(
                    common_params,
                    epilogue_params,
                    mma_o_consumer_state,
                    row_sum,
                    row_max,
                )

        return p_cor_consumer_state, mma_o_consumer_state

    @cute.jit
    def exchange_p_cor_metadata(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        correction_factor: cutlass.Float32,
        row_sum: cutlass.Float32,
        row_max: cutlass.Float32,
        row_max_new: cutlass.Float32,
        tAcc: cute.Tensor,
        tidx: cutlass.Int32,
        p_cor_producer_state: pipeline.PipelineState,
    ) -> pipeline.PipelineState:
        """Compute the correction factor for the last k tile."""
        no_correction = 0
        if (
            row_max_new - row_max
        ) * softmax_params.softmax_scale_log2 <= self.skip_correction_threshold:
            no_correction = 1
            row_max_new = row_max

        # pad for 4x32b
        corr_layout = cute.make_layout(
            (tAcc.shape[0], (4, tAcc.shape[1][1]), self.mma_s_stage),
            stride=(tAcc.stride[0], (1, tAcc.stride[1][1]), 4),
        )
        tCor = cute.make_tensor(
            common_params.tmem_ptr + self.correction_factor_offset,
            corr_layout,
        )
        cCor = cute.make_identity_tensor(tCor.shape)
        corr_tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(4)), self.acc_dtype
        )
        corr_tmem_store_tiled_copy = tcgen05.make_tmem_copy(corr_tmem_store_atom, tCor)
        corr_tmem_store_thr_copy = corr_tmem_store_tiled_copy.get_slice(tidx)
        cCor_for_copy = corr_tmem_store_thr_copy.partition_S(cCor)
        tCor_for_copy = corr_tmem_store_thr_copy.partition_D(tCor)
        rCor = cute.make_fragment_like(cCor_for_copy[None, None, None, 0], self.acc_dtype)
        rCor_int = cute.make_tensor(
            cute.recast_ptr(rCor.iterator, dtype=cutlass.Int32), rCor.layout
        )
        rCor[0] = row_sum
        rCor[1] = row_max_new
        rCor[2] = correction_factor
        rCor_int[3] = no_correction

        cute.copy(
            corr_tmem_store_tiled_copy,
            rCor,
            tCor_for_copy[None, None, None, p_cor_producer_state.index],
        )
        # fence between tmem store and correction warp
        cute.arch.fence_view_async_tmem_store()
        common_params.p_cor_pipeline.producer_commit(p_cor_producer_state)
        p_cor_producer_state.advance()
        return p_cor_producer_state, row_max_new

    @cute.jit
    def softmax(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        k_index: cutlass.Int32,
        mma_s_consumer_state: pipeline.PipelineState,
        p_mma_producer_state: pipeline.PipelineState,
        p_cor_producer_state: pipeline.PipelineState,
        row_max: cutlass.Float32,
        row_sum: cutlass.Float32,
        correction_factor: cutlass.Float32,
        is_last_tile: bool,
        is_local_last_tile: cutlass.Boolean,
    ) -> tuple[
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
        cutlass.Float32,
        cutlass.Float32,
        cutlass.Float32,
    ]:
        """Softmax for one k-tile. Updates the related pipeline states and returns the computed results.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param softmax_params: The softmax parameters
        :type softmax_params: SimpleNamespace
        :param k_index: The index of the k-tile
        :type k_index: cutlass.Int32
        :param mma_s_consumer_state: The MMA s consumer state
        :type mma_s_consumer_state: pipeline.PipelineState
        :param p_mma_producer_state: The P MMA producer state
        :type p_mma_producer_state: pipeline.PipelineState
        :param p_cor_producer_state: The P correction producer state
        :type p_cor_producer_state: pipeline.PipelineState
        :param row_max: The row max
        :type row_max: cutlass.Float32
        :param row_sum: The row sum
        :type row_sum: cutlass.Float32
        :param correction_factor: The correction factor
        :type correction_factor: cutlass.Float32
        :param is_last_tile: Whether the last tile
        :type is_last_tile: bool
        :param is_local_last_tile: Whether the last tile is local
        :type is_local_last_tile: cutlass.Boolean

        :return: The MMA s consumer state, the P MMA producer state, the P
            correction producer state, the row max, the row sum, and the
            correction factor
        :rtype: tuple[pipeline.PipelineState, pipeline.PipelineState,
            pipeline.PipelineState, cutlass.Float32, cutlass.Float32,
            cutlass.Float32]
        """

        softmax_params.p_mma_pipeline.producer_acquire(p_mma_producer_state)
        softmax_params.mma_s_pipeline.consumer_wait(mma_s_consumer_state)

        # load S from tmem
        tStS_shape = softmax_params.tiled_mma_qk.partition_shape_C(
            cute.select(self.mma_qk_tiler, mode=[0, 1])
        )
        tStS_staged_fake = softmax_params.tiled_mma_qk.make_fragment_C(
            cute.append(tStS_shape, self.mma_s_stage)
        )
        tStS_staged = cute.make_tensor(common_params.tmem_ptr, tStS_staged_fake.layout)
        tStS = tStS_staged[None, None, None, mma_s_consumer_state.index]

        tAcc = tStS[(None, None), 0, 0]
        cta_qk_tiler = (
            self.mma_qk_tiler[0] // self.cluster_shape_mnk[0],
            self.mma_qk_tiler[1],
            self.mma_qk_tiler[2],
        )
        cS = cute.make_identity_tensor(cute.select(cta_qk_tiler, mode=[0, 1]))

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_load_atom, tAcc)

        tidx = common_params.tidx % (self.num_compute_warps * self.threads_per_warp)

        tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc = tmem_thr_copy.partition_S(tAcc)
        tTR_tS = tmem_thr_copy.partition_D(cS)

        tTR_rAcc = cute.make_fragment_like(tTR_tS, self.acc_dtype)

        row_max_new = row_max
        arch = BaseDSL._get_dsl().get_arch_enum()
        if cutlass.const_expr(arch >= Arch.sm_100 and arch <= Arch.sm_100f):
            cute.copy(tmem_tiled_copy, tTR_tAcc, tTR_rAcc)
            for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                if is_last_tile or (
                    cutlass.const_expr(self.window_valid_len_from_tensor) and k_index == 0
                ):
                    tTR_rAcc[i] = (
                        tTR_rAcc[i]
                        if self.is_score_valid(
                            tTR_tS[i][1],
                            k_index,
                            common_params.K,
                            common_params.window_valid_len,
                            common_params.window_end_pos,
                        )
                        else -self.acc_dtype.inf
                    )
            # reduction for row_max
            row_max_new = tTR_rAcc.load().reduce(cute.ReductionOp.MAX, row_max_new, 0)

        elif cutlass.const_expr(arch >= Arch.sm_103 and arch <= Arch.sm_103f):
            tmem_load_red_atom = cute.make_copy_atom(
                tcgen05.copy.LdRed32x32bOp(
                    tcgen05.copy.Repetition(64), redOp=tcgen05.TmemLoadRedOp.MAX
                ),
                self.acc_dtype,
            )
            tmem_red_tiled_copy = tcgen05.make_tmem_copy(tmem_load_red_atom, tAcc)
            tmem_red_thr_copy = tmem_red_tiled_copy.get_slice(tidx)
            tTR_tAcc_red = tmem_red_thr_copy.partition_S(tAcc)
            tTR_tS_red = tmem_red_thr_copy.partition_D(cS)
            tTR_rAcc_red = cute.make_fragment_like(tTR_tS_red, self.acc_dtype)
            tTR_rMax = cute.make_rmem_tensor(
                cute.make_layout((1, tTR_tS_red.shape[1], tTR_tS_red.shape[2])),
                self.acc_dtype,
            )
            cute.copy(
                tmem_red_tiled_copy,
                tTR_tAcc_red,
                (tTR_rAcc_red, tTR_rMax),
            )
            tTR_rAcc = cute.make_tensor(tTR_rAcc_red.iterator, tTR_rAcc.layout)
            if is_last_tile or (
                cutlass.const_expr(self.window_valid_len_from_tensor) and k_index == 0
            ):
                for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                    tTR_rAcc[i] = (
                        tTR_rAcc[i]
                        if self.is_score_valid(
                            tTR_tS[i][1],
                            k_index,
                            common_params.K,
                            common_params.window_valid_len,
                            common_params.window_end_pos,
                        )
                        else -self.acc_dtype.inf
                    )
                # reduction for row_max
                row_max_new = tTR_rAcc.load().reduce(cute.ReductionOp.MAX, row_max_new, 0)
            else:
                row_max_new = cute.arch.fmax(row_max_new, tTR_rMax[0])

        # if warps in N is 2, reduce row_max across warps (0, 1) and (2, 3)
        if cutlass.const_expr(self.warps_in_n == 2):
            common_params.smem_exchange[tidx] = row_max_new
            self.softmax_exchange_sync_bar.wait()
            row_max_new = cute.arch.fmax(
                row_max_new,
                common_params.smem_exchange[
                    (tidx + 64) % (self.num_compute_warps * self.threads_per_warp)
                ],
            )
            # Keep every warp from reusing its exchange slot until its peer has
            # consumed the value from this iteration.
            self.softmax_exchange_sync_bar.wait()

        # find correction factor
        correction_factor = cute.math.exp2(
            (row_max - row_max_new) * softmax_params.softmax_scale_log2, fastmath=True
        )
        # split kv case
        if cutlass.const_expr(not is_local_last_tile):
            p_cor_producer_state, row_max_new = self.exchange_p_cor_metadata(
                common_params,
                softmax_params,
                correction_factor,
                row_sum,
                row_max,
                row_max_new,
                tAcc,
                tidx,
                p_cor_producer_state,
            )

        # softmax
        fma_b = softmax_params.softmax_scale_log2
        fma_c = (0.0 - row_max_new) * softmax_params.softmax_scale_log2

        for i in cutlass.range(cute.size(tTR_rAcc), vectorize=True, unroll_full=True):
            tTR_rAcc[i] = tTR_rAcc[i] * fma_b + fma_c
            tTR_rAcc[i] = cute.math.exp2(tTR_rAcc[i], fastmath=True)

        tTR_rS = cute.make_fragment_like(tTR_tS, self.q_dtype)

        # quantize
        tTR_rS.store(tTR_rAcc.load().to(self.q_dtype))

        # create sP
        sP = softmax_params.sP[None, None, None, (None, p_mma_producer_state.index)]
        sP_mk_view = cute.make_tensor(
            sP.iterator,
            cute.make_layout(
                (
                    (sP.shape[0][0], sP.shape[1]),
                    (sP.shape[0][1], sP.shape[2], sP.shape[3]),
                ),
                stride=(
                    (sP.stride[0][0], sP.stride[1]),
                    (sP.stride[0][1], sP.stride[2], sP.stride[3]),
                ),
            ),
        )
        # change to PISL
        sP_wo_swizzle_iter = cute.recast_ptr(sP.iterator, swizzle_=None)
        swizzle_bits = int(math.log2(self.mma_pv_tiler[2] * self.q_dtype.width // 8 // 32)) + 1
        swizzle_base = 3 if self.q_dtype.width == 16 else 4
        sP_swizzle = cute.make_swizzle(swizzle_bits, swizzle_base, 3)
        sP_mk_view = cute.make_tensor(
            sP_wo_swizzle_iter,
            cute.make_composed_layout(sP_swizzle, 0, sP_mk_view.layout),
        )
        universal_copy_bits = 128
        smem_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.q_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        smem_tiled_copy = cute.make_tiled_copy_D(smem_copy_atom, tmem_tiled_copy)
        smem_thr_copy = smem_tiled_copy.get_slice(tidx)
        rP_copy_view = smem_thr_copy.retile(tTR_rS)
        sP_copy_view = smem_thr_copy.partition_D(sP_mk_view)
        cute.copy(smem_tiled_copy, rP_copy_view, sP_copy_view)

        # fence between smem store and mma o
        cute.arch.fence_view_async_shared()
        softmax_params.p_mma_pipeline.producer_commit(p_mma_producer_state)
        p_mma_producer_state.advance()

        # row_sum, using `add_packed_f32x2` to reduce the number of instructions
        row_sum = row_sum * correction_factor
        row_sum_vec = (0.0, 0.0)
        for i in cutlass.range_constexpr(0, cute.size(tTR_rAcc), 2):
            row_sum_vec = cute.arch.add_packed_f32x2(row_sum_vec, (tTR_rAcc[i], tTR_rAcc[i + 1]))
        row_sum = row_sum_vec[0] + row_sum_vec[1] + row_sum

        # split kv case
        if cutlass.const_expr(is_local_last_tile):
            p_cor_producer_state, row_max_new = self.exchange_p_cor_metadata(
                common_params,
                softmax_params,
                correction_factor,
                row_sum,
                row_max,
                row_max_new,
                tAcc,
                tidx,
                p_cor_producer_state,
            )

        # store correction factor/row_sum/row_max to tmem for correction warp
        common_params.p_cor_pipeline.producer_acquire(p_cor_producer_state)

        # fence between tmem load and mma s
        cute.arch.fence_view_async_tmem_load()

        softmax_params.mma_s_pipeline.consumer_release(mma_s_consumer_state)
        mma_s_consumer_state.advance()

        return (
            mma_s_consumer_state,
            p_mma_producer_state,
            p_cor_producer_state,
            row_max_new,
            row_sum,
            correction_factor,
        )

    @cute.jit
    def _tmem_load_partition(
        self, common_params: SimpleNamespace, tiled_mma_pv: cute.TiledMma, iter_n: int
    ) -> tuple[cute.TiledMma, cute.TiledMma, cute.TiledMma, cute.TiledMma, cute.TiledMma]:
        """Tensor memory load partition for rescale and epilogue.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param tiled_mma_pv: The tiled mma pv
        :type tiled_mma_pv: cute.TiledMma
        :param iter_n: The iteration number
        :type iter_n: int

        :return: The tiled mma pv, the tiled mma pv, the tiled mma pv, the tiled mma pv, the tiled mma pv
        :rtype: tuple[cute.TiledMma, cute.TiledMma, cute.TiledMma, cute.TiledMma, cute.TiledMma]
        """

        tOtO_shape = tiled_mma_pv.partition_shape_C(cute.select(self.mma_pv_tiler, mode=[0, 1]))
        tOtO = tiled_mma_pv.make_fragment_C(tOtO_shape)
        tOtO_layout = cute.append(
            tOtO.layout,
            cute.make_layout(
                common_params.L // self.mma_pv_tiler[1],
                stride=self.mma_pv_tiler[1] // self.warps_in_n,
            ),
        )
        tOtO = cute.make_tensor(common_params.tmem_ptr + self.tmem_o_offset, tOtO_layout)
        tOtO = tOtO[None, None, None, iter_n]

        tAcc = tOtO[(None, None), 0, 0]

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        tmem_load_tiled_copy = tcgen05.make_tmem_copy(tmem_load_atom, tAcc)
        tmem_load_thr_copy = tmem_load_tiled_copy.get_slice(
            common_params.tidx % (self.num_compute_warps * self.threads_per_warp)
        )

        cta_pv_tiler = (
            self.mma_pv_tiler[0] // self.cluster_shape_mnk[0],
            self.mma_pv_tiler[1],
            self.mma_pv_tiler[2],
        )
        # Flatten divide and partition global tensors for O
        cta_pv_tiler_mn = cute.select(cta_pv_tiler, mode=[0, 1])

        gO = None
        if cutlass.const_expr(common_params.mAccO is not None):
            gO = cute.local_tile(
                common_params.mAccO[None, common_params.blk_coord[3], None, None, None],
                cta_pv_tiler_mn,
                (
                    common_params.blk_coord[0],
                    iter_n,
                    common_params.blk_coord[1],
                    common_params.blk_coord[2],
                ),
            )
            cO = cute.local_tile(
                cute.make_identity_tensor(
                    common_params.mAccO[None, common_params.blk_coord[3], None, None, None].shape
                ),
                cta_pv_tiler_mn,
                (
                    common_params.blk_coord[0],
                    iter_n,
                    common_params.blk_coord[1],
                    common_params.blk_coord[2],
                ),
            )
        else:
            gO = cute.local_tile(
                common_params.mO,
                cta_pv_tiler_mn,
                (
                    common_params.blk_coord[0],
                    iter_n,
                    common_params.blk_coord[1],
                    common_params.blk_coord[2],
                ),
            )
            cO = cute.local_tile(
                cute.make_identity_tensor(common_params.mO.shape),
                cta_pv_tiler_mn,
                (
                    common_params.blk_coord[0],
                    iter_n,
                    common_params.blk_coord[1],
                    common_params.blk_coord[2],
                ),
            )
        tTR_tAcc = tmem_load_thr_copy.partition_S(tAcc)
        tTR_gO = tmem_load_thr_copy.partition_D(gO)
        tTR_cO = tmem_load_thr_copy.partition_D(cO)
        tTR_rAcc = cute.make_fragment_like(tTR_gO, self.acc_dtype)
        return tmem_load_tiled_copy, tAcc, tTR_tAcc, tTR_gO, tTR_cO, tTR_rAcc

    def get_correction_factor(
        self,
        common_params: SimpleNamespace,
        p_cor_consumer_state: pipeline.PipelineState,
    ) -> tuple[
        pipeline.PipelineState,
        cutlass.Float32,
        cutlass.Float32,
        cutlass.Float32,
        cutlass.Int32,
    ]:
        """Get the correction factor from the P correction consumer state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param p_cor_consumer_state: The P correction consumer state
        :type p_cor_consumer_state: pipeline.PipelineState

        :return: The P correction consumer state, the row_sum, the row_max, and the correction factor
        :rtype: tuple[pipeline.PipelineState, cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Int32]
        """
        common_params.p_cor_pipeline.consumer_wait(p_cor_consumer_state)
        tidx = common_params.tidx % (self.num_compute_warps * self.threads_per_warp)
        # load correction factor
        _, tAcc, _, _, _, _ = self._tmem_load_partition(
            common_params, common_params.tiled_mma_pv, 0
        )
        corr_layout = cute.make_layout(
            (tAcc.shape[0], (4, tAcc.shape[1][1]), self.p_cor_stage),
            stride=(tAcc.stride[0], (1, tAcc.stride[1][1]), 4),
        )
        tCor = cute.make_tensor(common_params.tmem_ptr + self.correction_factor_offset, corr_layout)
        cCor = cute.make_identity_tensor(tCor.shape)
        corr_tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(4)), self.acc_dtype
        )
        corr_tmem_load_tiled_copy = tcgen05.make_tmem_copy(corr_tmem_load_atom, tCor)
        corr_tmem_load_thr_copy = corr_tmem_load_tiled_copy.get_slice(tidx)
        tCor_for_copy = corr_tmem_load_thr_copy.partition_S(tCor)
        cCor_for_copy = corr_tmem_load_thr_copy.partition_D(cCor)
        rCor = cute.make_fragment_like(cCor_for_copy[None, None, None, 0], self.acc_dtype)
        rCor_int = cute.make_tensor(
            cute.recast_ptr(rCor.iterator, dtype=cutlass.Int32), rCor.layout
        )
        cute.copy(
            corr_tmem_load_tiled_copy,
            tCor_for_copy[None, None, None, p_cor_consumer_state.index],
            rCor,
        )
        row_sum = rCor[0]
        row_max = rCor[1]
        correction_factor = rCor[2]
        no_correction = rCor_int[3]

        cute.arch.fence_view_async_tmem_load()
        common_params.p_cor_pipeline.consumer_release(p_cor_consumer_state)
        p_cor_consumer_state.advance()
        return p_cor_consumer_state, row_sum, row_max, correction_factor, no_correction

    @cute.jit
    def rescale(
        self,
        common_params: SimpleNamespace,
        mma_o_consumer_state: pipeline.PipelineState,
        correction_factor: cutlass.Float32,
        no_correction: cutlass.Int32,
    ) -> pipeline.PipelineState:
        """Rescale for one k-tile. Updates the related pipeline state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param mma_o_consumer_state: The mma o consumer state
        :type mma_o_consumer_state: pipeline.PipelineState
        :param correction_factor: The correction factor
        :type correction_factor: cutlass.Float32
        :param no_correction: Whether to apply correction factor
        :type no_correction: cutlass.Int32

        :return: The MMA o consumer state
        :rtype: pipeline.PipelineState
        """
        skip_correction = cute.arch.vote_all_sync(no_correction == 1)
        common_params.mma_o_pipeline.consumer_wait(mma_o_consumer_state)
        if not skip_correction:
            for iter_n in cutlass.range_constexpr(self.iterations_pv_n):
                # tmem load tiled copy and partition results.
                tmem_load_tiled_copy, tAcc, tTR_tAcc, tTR_gO, tTR_cO, tTR_rAcc = (
                    self._tmem_load_partition(common_params, common_params.tiled_mma_pv, iter_n)
                )

                # tmem store tiled copy
                tmem_store_atom = cute.make_copy_atom(
                    tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
                )
                tmem_store_tiled_copy = tcgen05.make_tmem_copy(tmem_store_atom, tAcc)

                # load o
                cute.copy(tmem_load_tiled_copy, tTR_tAcc, tTR_rAcc)
                # rescale, using `mul_packed_f32x2` to reduce the number of instructions
                for i in cutlass.range(cute.size(tTR_rAcc), vectorize=True, unroll_full=True):
                    tTR_rAcc[i] = tTR_rAcc[i] * correction_factor

                # store o to tensor memory for next k tile
                cute.copy(tmem_store_tiled_copy, tTR_rAcc, tTR_tAcc)

        cute.arch.fence_view_async_tmem_store()
        common_params.mma_o_pipeline.consumer_release(mma_o_consumer_state)
        mma_o_consumer_state.advance()

        return mma_o_consumer_state

    @cute.jit
    def epilogue(
        self,
        common_params: SimpleNamespace,
        epilogue_params: SimpleNamespace,
        mma_o_consumer_state: pipeline.PipelineState,
        row_sum: cutlass.Float32,
        row_max: cutlass.Float32,
    ) -> pipeline.PipelineState:
        """Epilogue for one k-tile. Updates the related pipeline state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param epilogue_params: The epilogue parameters
        :type epilogue_params: SimpleNamespace
        :param mma_o_consumer_state: The mma o consumer state
        :type mma_o_consumer_state: pipeline.PipelineState
        :param row_sum: The row sum
        :type row_sum: cutlass.Float32
        :param row_max: The row max
        :type row_max: cutlass.Float32

        :return: The MMA o consumer state
        :rtype: pipeline.PipelineState
        """

        tidx = common_params.tidx % (self.num_compute_warps * self.threads_per_warp)

        # exchange row_sum between warps (0, 1) and (2, 3)
        if cutlass.const_expr(self.warps_in_n == 2):
            common_params.smem_exchange[tidx] = row_sum
            self.epilogue_exchange_sync_bar.wait()
            # (64, 2)
            row_sum = (
                row_sum
                + common_params.smem_exchange[
                    (tidx + 64) % (self.num_compute_warps * self.threads_per_warp)
                ]
            )
            # The persistent epilogue reuses one exchange buffer across work
            # tiles, so all peer reads must finish before the next write.
            self.epilogue_exchange_sync_bar.wait()
        # mma_o pipeline consumer wait
        common_params.mma_o_pipeline.consumer_wait(mma_o_consumer_state)
        for iter_n in cutlass.range_constexpr(self.iterations_pv_n):
            # tmem load tiled copy and partition results.
            tmem_load_tiled_copy, tAcc, tTR_tAcc, tTR_gO, tTR_cO, tTR_rAcc = (
                self._tmem_load_partition(common_params, common_params.tiled_mma_pv, iter_n)
            )

            # load o
            cute.copy(tmem_load_tiled_copy, tTR_tAcc, tTR_rAcc)

            # apply output scale and normalize by row_sum
            for i in cutlass.range(cute.size(tTR_rAcc), vectorize=True, unroll_full=True):
                tTR_rAcc[i] = (
                    tTR_rAcc[i] * epilogue_params.output_scale * cute.arch.rcp_approx(row_sum)
                )

            # store o to global memory
            tR2G_rO_src = None
            tR2G_rO_dst = tTR_gO
            if cutlass.const_expr(common_params.mAccO is None):
                tR2G_rO_src = cute.make_fragment_like(tTR_gO, self.o_dtype)
                # using final output dtype for o
                tR2G_rO_src.store(tTR_rAcc.load().to(self.o_dtype))
                if cutlass.const_expr(self.inverse_rope_dim > 0):
                    # Fused inverse-RoPE epilogue on the last inverse_rope_dim
                    # output lanes. The value is rounded to the output dtype
                    # BEFORE the rotation, preserving the removed standalone
                    # kernel boundary bit-for-bit; the tmem-load fragment holds
                    # 32 contiguous columns per thread, so each (re, im) pair
                    # is register-local.
                    nope_dim = self.latent_dim - self.inverse_rope_dim
                    freqs_base = (
                        common_params.blk_coord[2] * self.seq_len_q + common_params.blk_coord[1]
                    ) * self.inverse_rope_dim
                    for i in cutlass.range_constexpr(0, cute.size(tTR_rAcc), 2):
                        d = tTR_cO[i][1]
                        if d >= nope_dim:
                            pair_base = freqs_base + (d - nope_dim)
                            cos = epilogue_params.mFreqs[pair_base]
                            sin = epilogue_params.mFreqs[pair_base + 1]
                            re = cutlass.Float32(tR2G_rO_src[i])
                            im = cutlass.Float32(tR2G_rO_src[i + 1])
                            tR2G_rO_src[i] = (re * cos + im * sin).to(self.o_dtype)
                            tR2G_rO_src[i + 1] = (im * cos - re * sin).to(self.o_dtype)
            else:
                # using accumulate dtype for o
                tR2G_rO_src = tTR_rAcc

            if cute.elem_less(tTR_cO[0][0], common_params.H):
                cute.autovec_copy(
                    tR2G_rO_src,
                    tR2G_rO_dst,
                    l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
                )

        cute.arch.fence_view_async_tmem_load()
        common_params.mma_o_pipeline.consumer_release(mma_o_consumer_state)
        mma_o_consumer_state.advance()

        return mma_o_consumer_state

    def make_and_init_load_pt_pipeline(self, load_pt_mbar_ptr):
        """Create and initialize the load page table pipeline.

        :param load_pt_mbar_ptr: The load page table mbar pointer
        :type load_pt_mbar_ptr: cute.Tensor

        :return: The load page table pipeline
        :rtype: pipeline.PipelineAsync
        """
        load_pt_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * len([self.load_pt_warp_id]),
        )
        load_pt_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * len([self.load_tma_warp_id]),
        )
        return pipeline.PipelineCpAsync.create(
            barrier_storage=load_pt_mbar_ptr,
            num_stages=self.load_pt_stage,
            producer_group=load_pt_producer_group,
            consumer_group=load_pt_consumer_group,
            defer_sync=True,
        )

    def make_and_init_load_qkv_pipeline(
        self, load_qkv_mbar_ptr, cta_layout_vmnk, load_stages, tx_count
    ) -> pipeline.PipelineTmaUmma:
        """Create and initialize the tma load qkv pipeline.

        :param load_qkv_mbar_ptr: The load qkv mbar pointer
        :type load_qkv_mbar_ptr: cute.Tensor
        :param cta_layout_vmnk: The cta layout vmnk
        :type cta_layout_vmnk: tuple[int, int, int]
        :param load_stages: The load stages
        :type load_stages: list[int]
        :param tx_count: The tx count
        :type tx_count: int

        :return: The tma load qkv pipeline
        :rtype: pipeline.PipelineTmaUmma
        """
        load_qkv_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.load_tma_warp_id])
        )
        load_qkv_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=load_qkv_mbar_ptr,
            num_stages=load_stages,
            producer_group=load_qkv_producer_group,
            consumer_group=load_qkv_consumer_group,
            tx_count=tx_count,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

    def make_and_init_mma_s_pipeline(
        self, mma_s_mbar_ptr, cta_layout_vmnk
    ) -> pipeline.PipelineUmmaAsync:
        """Create and initialize the mma s pipeline.

        :param mma_s_mbar_ptr: The mma s mbar pointer
        :type mma_s_mbar_ptr: cute.Tensor
        :param cta_layout_vmnk: The cta layout vmnk
        :type cta_layout_vmnk: tuple[int, int, int]

        :return: The mma s pipeline
        :rtype: pipeline.PipelineUmmaAsync
        """

        mma_s_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        consumer_thread_size = (
            self.threads_per_warp * len(self.compute_warp_ids) * self.cluster_shape_mnk[0]
        )
        mma_s_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            consumer_thread_size,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_s_mbar_ptr,
            num_stages=self.mma_s_stage,
            producer_group=mma_s_producer_group,
            consumer_group=mma_s_consumer_group,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

    def make_and_init_p_mma_pipeline(
        self, p_mma_mbar_ptr, cta_layout_vmnk
    ) -> pipeline.PipelineAsyncUmma:
        """Create and initialize the p mma pipeline.

        :param p_mma_mbar_ptr: The p mma mbar pointer
        :type p_mma_mbar_ptr: cute.Tensor
        :param cta_layout_vmnk: The cta layout vmnk
        :type cta_layout_vmnk: tuple[int, int, int]

        :return: The p mma pipeline
        :rtype: pipeline.PipelineAsyncUmma
        """

        producer_thread_size = (
            self.threads_per_warp * len(self.compute_warp_ids) * self.cluster_shape_mnk[0]
        )
        p_mma_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            producer_thread_size,
        )
        p_mma_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        return pipeline.PipelineAsyncUmma.create(
            barrier_storage=p_mma_mbar_ptr,
            num_stages=self.p_mma_stage,
            producer_group=p_mma_producer_group,
            consumer_group=p_mma_consumer_group,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

    def make_and_init_p_cor_pipeline(self, p_cor_mbar_ptr) -> pipeline.PipelineAsyncUmma:
        """Create and initialize the p correction pipeline.

        :param p_cor_mbar_ptr: The p correction mbar pointer
        :type p_cor_mbar_ptr: cute.Tensor

        :return: The p correction pipeline
        :rtype: pipeline.PipelineAsyncUmma
        """

        producer_thread_size = self.threads_per_warp * len(self.compute_warp_ids)
        p_cor_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            producer_thread_size,
        )
        p_cor_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            producer_thread_size,
        )
        return pipeline.PipelineAsync.create(
            barrier_storage=p_cor_mbar_ptr,
            num_stages=self.p_cor_stage,
            producer_group=p_cor_producer_group,
            consumer_group=p_cor_consumer_group,
            defer_sync=True,
        )

    def make_and_init_mma_o_pipeline(
        self, mma_o_mbar_ptr, cta_layout_vmnk
    ) -> pipeline.PipelineUmmaAsync:
        """Create and initialize the mma o pipeline.

        :param mma_o_mbar_ptr: The mma o mbar pointer
        :type mma_o_mbar_ptr: cute.Tensor
        :param cta_layout_vmnk: The cta layout vmnk
        :type cta_layout_vmnk: tuple[int, int, int]

        :return: The mma o pipeline
        :rtype: pipeline.PipelineUmmaAsync
        """

        mma_o_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        consumer_thread_size = (
            self.threads_per_warp * len(self.compute_warp_ids) * self.cluster_shape_mnk[0]
        )
        mma_o_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            consumer_thread_size,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_o_mbar_ptr,
            num_stages=self.mma_o_stage,
            producer_group=mma_o_producer_group,
            consumer_group=mma_o_consumer_group,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

    @staticmethod
    def _compute_grid(
        o: cute.Tensor,
        split_kv: cutlass.Int32,
        cluster_shape_mnk: Tuple[int, int, int],
        max_active_clusters: int,
        is_persistent: bool,
    ) -> Tuple[HCAStaticTileSchedulerParams, Tuple[int, int, int]]:
        """Compute grid shape for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]

        :return: Tile scheduler parameters and grid shape.
        :rtype: tuple[HCAStaticTileSchedulerParams, tuple[int, int, int]]
        """
        o_shape = o.shape
        tile_sched_params = create_hca_static_tile_scheduler_params(
            is_persistent,
            cute.size(o_shape[3]),
            cute.size(o_shape[2]),
            cluster_shape_mnk,
            split_kv,
        )
        grid = HCAStaticTileScheduler.get_grid_shape(tile_sched_params, max_active_clusters)

        return tile_sched_params, grid
