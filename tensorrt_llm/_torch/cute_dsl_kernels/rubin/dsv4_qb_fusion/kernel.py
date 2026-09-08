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

# ruff: noqa: E501,E731,E741

"""CuTe-DSL-style SM107 persistent MXFP8 q_b GEMM fused with RMSNorm,
RoPE and E4M3 quant -- standalone, mixed-cluster capable.

Kernel semantics (per output row m, per 512-wide head h; everything
after the GEMM is FP32 until the final quant)::

    x        = activation[m, :] @ weight[h*512:(h+1)*512, :].T
               # MXFP8 block-scaled GEMM: both operands E4M3 with
               # E8M0 scale factors at K-granularity 128, FP32 acc
    inv_rms  = rsqrt(mean(x * x) + eps)          # over the 512 head
               # cols, no gamma
    y        = x * (inv_rms * quant_scale_qkv)
    pos      = position of row m from the batch metadata: the batch
               b with cu_q_seqlens[b] <= m < cu_q_seqlens[b+1] gives
               pos = kv_cache_lengths[b] + (m - cu_q_seqlens[b]),
               overridden by helix_position_offsets[m] when enabled
    y[448 + 2i], y[449 + 2i]  (i = 0..31, the rope head dims)
             = GPT-J pairing rotation by angle i of cache row pos:
               (c * y0 - s * y1,  s * y0 + c * y1) with
               c = cos_sin_cache[pos, 2i], s = cos_sin_cache[pos, 2i+1]
               (the cache row holds 64 (cos, sin) pairs; rope reads
               the first 32)
    out[m, h*512:(h+1)*512] = satfinite_e4m3(y)

The scale multiplies BEFORE the rotation (the rotation is linear, so
this matches applying it after); rows whose position falls outside
the cache are stored unrotated-undefined in the rope columns, and
rows >= M are not stored at all.

A from-scratch implementation (no example class is imported) following
the STYLE of the upstream CuTe DSL persistent block-scaled GEMM
mixed-cluster example:

* The fusion epilogue -- position walk over the TRT metadata,
  cooperative cp.async cos/sin staging into SMEM (issued at tile
  start, 16 B XOR-swizzled rows), the
  GPT-J RoPE rotation, the 512-wide FP32 RMSNorm and the satfinite
  E4M3 quant + store -- uses only cute ops plus NVVM-level interfaces
  (``cute.arch`` wrappers and, for ``tcgen05.fence``, the nvvm dialect
  itself); there is no dependency on any lower-level primitives package.
* store_mode (init parameter): "stg256" (default)
  stores each quantized 32 B chunk straight to GMEM as one
  row-predicated STG.256 (evict-noallocate) -- no staging SMEM, no
  drain fences; "tma" stages each tile's output in a 32 KB 2-slot SMEM
  ring and drains per 128 B column plane with TMA bulk tensor stores.

semantics: one work tile = one whole N512 head per CTA.  The
scheduler runs at (M128-CTA, head) granularity and the mainloop issues
BOTH N256 sets of the head per k-tile (two B/SFB copies, two
``cute.gemm`` chains); the accumulator holds 2 x 256 TMEM columns, so
the epilogue owns the full 512-column row for the row reduction.

Problem geometry: DSV4 q_b projection -- K = q_lora_rank, N =
num_q_heads x 512, quant granularity G = 128 expressed as standard
sf_vec_size=32 CUTLASS SF atoms with each scale repeated over its four
K32 slots.

"""

from __future__ import annotations

import os
from functools import partial
from typing import NamedTuple, Optional, Tuple

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.rubin_helpers as sm107_utils
from cutlass._mlir.dialects import nvvm
from cutlass.cute.arch import nvvm_wrappers as _arch_nvvm_wrappers
from cutlass.cute.experimental import iket
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.nvgpu.common import CacheEvictionPriority
from cutlass.cute.nvgpu.tcgen05.mma import CollectorOp
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils.gemm import sm100 as gemm_sm100

from .clc_tile_scheduler import (
    ClcDynamicPersistentTileScheduler,
    ClcDynamicPersistentTileSchedulerParams,
)

DEFAULT_Q_LORA_RANK = 1536
DEFAULT_NUM_Q_HEADS = 128
DEFAULT_QK_NOPE_HEAD_DIM = 448
DEFAULT_QK_ROPE_HEAD_DIM = 64
DEFAULT_RMS_NORM_EPS = 1e-6
K_QUANT_BLOCK = 128
HEAD_DIM = DEFAULT_QK_NOPE_HEAD_DIM + DEFAULT_QK_ROPE_HEAD_DIM


def _fma_packed_bf16x2_nvvm(res, src_a, src_b, src_c, *, rnd=None, ftz=None, loc=None, ip=None):
    del ftz
    src_a_bf16x2 = _arch_nvvm_wrappers.cvt_f32x2_bf16x2(src_a, loc=loc, ip=ip)
    return _arch_nvvm_wrappers.nvvm.fma_packed_f32x2_bf16x2_f32x2_f32x2(
        res, src_a_bf16x2, src_b, src_c, rnd=rnd, loc=loc, ip=ip
    )


_fma_packed_f32x2_bf16x2_f32x2_f32x2 = getattr(
    cute.arch,
    "fma_packed_f32x2_bf16x2_f32x2_f32x2",
    partial(
        _arch_nvvm_wrappers.calc_packed_f32x2_op,
        calc_func=_fma_packed_bf16x2_nvvm,
    ),
)


class S2TCopyBundle(NamedTuple):
    """Tiled copy + partitioned tensors for one SMEM-to-TMEM SF copy."""

    tiled_copy: cute.TiledCopy
    sSF_compact: cute.Tensor
    tSF_compact: cute.Tensor


class CuteBlockScaledGemmFusedRMSNormRopeQuant:
    """Standalone CuTe-DSL-style fusion kernel with mixed clusters.

    One work tile = one (M128 CTA row-block, N512 head): the TMA warp
    issues B/SFB twice per k-tile (the head's two N256 sets) and the
    MMA warp runs two ``cute.gemm`` chains into two 256-column
    accumulator TMEM ranges; the epilogue warps read the whole
    512-column row, walks the TRT position metadata, applies RMSNorm +
    RoPE + satfinite E4M3 quant and stores via row-predicated STG.256.
    """

    def __init__(
        self,
        *,
        mma_inst_tile: Tuple[int, int] = (128, 256),
        cluster_shape_mn: Tuple[int, int] | None = None,
        fallback_cluster_shape_mn: Tuple[int, int] | None = None,
        store_mode: str = "stg256",
        swizzle_size: int = 1,
        raster_along_m: bool = True,
        qk_nope_head_dim: int = DEFAULT_QK_NOPE_HEAD_DIM,
        qk_rope_head_dim: int = DEFAULT_QK_ROPE_HEAD_DIM,
        rms_norm_eps: float = DEFAULT_RMS_NORM_EPS,
        max_batch: int = 128,
        tma_prefetch_dist: int = 0,
    ):
        if int(max_batch) < 1:
            raise ValueError("max_batch must be >= 1")
        self.max_batch = int(max_batch)
        if int(tma_prefetch_dist) < 0:
            raise ValueError("tma_prefetch_dist must be >= 0")
        self.tma_prefetch_dist = int(tma_prefetch_dist)
        if tuple(mma_inst_tile) not in ((128, 256), (256, 256)):
            raise ValueError("mma_inst_tile must be (128, 256) or (256, 256)")
        mma_m = mma_inst_tile[0]
        self.use_2cta_instrs = mma_m == 256
        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        self.atom_thr_size = 2 if self.use_2cta_instrs else 1
        self.mma_inst_shape = (mma_m, 256, 64)
        self.mma_tiler = (mma_m, 512, 128)

        if cluster_shape_mn is None:
            cluster_shape_mn = (self.atom_thr_size, 1)
        self.cluster_shape_mn = tuple(cluster_shape_mn)
        self._check_cluster(self.cluster_shape_mn, "cluster_shape_mn")
        self.fallback_cluster_shape_mn: Optional[Tuple[int, int]] = (
            tuple(fallback_cluster_shape_mn) if fallback_cluster_shape_mn is not None else None
        )
        if self.fallback_cluster_shape_mn is not None:
            fb = self.fallback_cluster_shape_mn
            self._check_cluster(fb, "fallback_cluster_shape_mn")
            pref = self.cluster_shape_mn
            if pref[0] % fb[0] or pref[1] % fb[1] or fb[0] * fb[1] >= pref[0] * pref[1]:
                raise ValueError(
                    "fallback cluster must divide the preferred shape per "
                    "dimension and be strictly smaller"
                )

        if int(swizzle_size) < 1:
            raise ValueError("swizzle_size must be >= 1")
        self.swizzle_size = int(swizzle_size)
        self.raster_along_m = bool(raster_along_m)

        if store_mode not in ("tma", "stg256"):
            raise ValueError("store_mode must be 'tma' or 'stg256'")
        self.store_mode: str = store_mode
        self.use_tma_store: bool = store_mode == "tma"
        self.out_plane_cols = 128
        self.out_ring_slots = HEAD_DIM // self.out_plane_cols

        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = 32
        self.occupancy = 1
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.sched_warp_id = 6
        self.threads_per_warp = 32
        self.pad_warp_id = 7
        self.threads_per_cta = self.threads_per_warp * len(
            (
                self.mma_warp_id,
                self.tma_warp_id,
                self.sched_warp_id,
                self.pad_warp_id,
                *self.epilog_warp_id,
            )
        )
        self.epilog_reg_count = 256
        self.mainloop_reg_count = 88
        self.num_clc_stage = 1
        self.num_clc_response_bytes = 16
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_warp * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_warp * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        self.arch = "sm_107"
        self.smem_capacity = utils.get_smem_capacity_in_bytes(self.arch)
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(self.arch)

        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.head_dim = qk_nope_head_dim + qk_rope_head_dim
        if self.head_dim != 512:
            raise ValueError("the schedule requires the N512 head")
        self.planes_per_head = self.head_dim // self.out_plane_cols
        self.rms_norm_eps = rms_norm_eps
        self.n_sets_per_head = 2
        self.epi_chunk = 32
        self.epi_chunks = self.head_dim // self.epi_chunk
        self.rope_chunks = self.qk_rope_head_dim // self.epi_chunk
        self.nope_chunks = self.epi_chunks - self.rope_chunks
        self.f32_bytes = cutlass.Float32.width // 8
        self.i32_bytes = cutlass.Int32.width // 8
        self.rope_cache_row_floats = 2 * self.qk_rope_head_dim
        self.rope_smem_row_words = self.qk_rope_head_dim
        self.rope_cp_bytes = 16
        self.rope_cp_chunks = (self.rope_smem_row_words * self.f32_bytes) // self.rope_cp_bytes
        self.rope_rows_per_cp = 32 // self.rope_cp_chunks
        self.rope_cp_iters = 32 // self.rope_rows_per_cp

    def _check_cluster(self, shape_mn, name: str) -> None:
        cm, cn = shape_mn
        if cm < 1 or cn < 1 or (cm & (cm - 1)) or (cn & (cn - 1)):
            raise ValueError(f"{name} dims must be positive powers of two")
        if cm % self.atom_thr_size:
            raise ValueError(f"{name}[0] must be a multiple of the MMA pair")
        if cm * cn > 16:
            raise ValueError(f"{name} exceeds the 16-CTA cluster cap")

    def _make_tiled_mmas(self):
        def make(a_op, b_op):
            mma = sm107_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.b_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_inst_shape,
                a_collector_op=a_op,
                b_collector_op=b_op,
                atom_layout_mnk=(1, 1, 1),
                permutation_mnk=(1, 1, 1),
            )
            mma.set(tcgen05.Field.NEGATE_A, False)
            mma.set(tcgen05.Field.NEGATE_B, False)
            return mma

        tiled_mma_akeep = make(CollectorOp.FILL, CollectorOp.DISCARD)
        tiled_mma_areuse = make(CollectorOp.LASTUSE, CollectorOp.DISCARD)
        tiled_mma_sfb = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_sfb,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
        )
        tiled_mma_sfb.set(tcgen05.Field.NEGATE_A, False)
        tiled_mma_sfb.set(tcgen05.Field.NEGATE_B, False)
        return tiled_mma_akeep, tiled_mma_areuse, tiled_mma_sfb

    def _cluster_layouts(self, cluster_shape_mn, tiled_mma_akeep, tiled_mma_sfb):
        """(vmnk, sfb_vmnk, is_a_mcast, is_b_mcast) for one launched shape."""
        vmnk = cute.tiled_divide(
            cute.make_layout((*cluster_shape_mn, 1)),
            (tiled_mma_akeep.thr_id.shape,),
        )
        sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )
        is_a_mcast = cute.size(vmnk.shape[2]) > 1
        is_b_mcast = cute.size(vmnk.shape[1]) > 1
        return vmnk, sfb_vmnk, is_a_mcast, is_b_mcast

    def _setup_attributes(self):
        """dtype-dependent setup: SMEM/TMEM layouts and stage counts.
        Runs inside the @cute.jit __call__."""
        self.mma_inst_shape_sfb = (
            self.mma_inst_shape[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape[1], 128),
            self.mma_inst_shape[2],
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_sfb[0],
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        tiled_mma_akeep, _, tiled_mma_sfb = self._make_tiled_mmas()
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma_akeep.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        a_one = sm100_utils.make_smem_layout_a(tiled_mma_akeep, self.mma_tiler, self.a_dtype, 1)
        b_one = sm100_utils.make_smem_layout_b(tiled_mma_akeep, self.mma_tiler, self.b_dtype, 1)
        sfa_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma_akeep, self.mma_tiler, self.sf_vec_size, 1
        )
        sfb_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma_akeep, self.mma_tiler, self.sf_vec_size, 1
        )
        stage_bytes = (
            cute.size_in_bytes(self.a_dtype, a_one)
            + cute.size_in_bytes(self.b_dtype, b_one)
            + cute.size_in_bytes(self.sf_dtype, sfa_one)
            + cute.size_in_bytes(self.sf_dtype, sfb_one)
        )
        batch_meta_bytes = (
            (self.max_batch + 1) * self.i32_bytes  # sCuSeqlens
            + self.max_batch * self.i32_bytes  # sKvLengths
            + 32  # slack for the two 16 B alignments
        )
        mbar_helpers_bytes = 1024 + batch_meta_bytes
        self.rope_smem_bytes = (
            self.cta_tile_shape_mnk[0] * self.rope_smem_row_words * self.f32_bytes
        )
        self.out_stage_bytes = (
            self.out_ring_slots * self.cta_tile_shape_mnk[0] * self.out_plane_cols
            if self.use_tma_store
            else 0
        )
        self.num_ab_stage = (
            self.smem_capacity // self.occupancy
            - mbar_helpers_bytes
            - self.rope_smem_bytes
            - self.out_stage_bytes
        ) // stage_bytes
        if os.environ.get("CUTE_DSL_DUMP_STAGES"):
            print(
                f"[stages] num_ab_stage={self.num_ab_stage} stage_bytes={stage_bytes}",
                flush=True,
            )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma_akeep, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_akeep, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma_akeep, self.mma_tiler, self.sf_vec_size, self.num_ab_stage
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma_akeep, self.mma_tiler, self.sf_vec_size, self.num_ab_stage
        )

        self.tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma_akeep,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0)),
        )
        self.tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma_akeep,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0)),
        )
        self.num_sfa_tmem_cols = (
            cute.cosize(cute.recast_layout(32, self.sf_dtype.width, self.tCtSFA_layout))
            & 0x0000FFFF
        )
        self.num_sfb_tmem_cols = (
            cute.cosize(cute.recast_layout(32, self.sf_dtype.width, self.tCtSFB_layout))
            & 0x0000FFFF
        )
        self.num_accumulator_tmem_cols = cute.size(self.cta_tile_shape_mnk[1])

        rope_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.Float32
        )
        self.rope_smem_layout = cute.tile_to_shape(
            rope_atom,
            (self.cta_tile_shape_mnk[0], self.rope_smem_row_words),
            (1, 0),
        )

        self.epi_store_tile = (
            self.cta_tile_shape_mnk[0],
            self.out_plane_cols,
        )
        self.out_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            self.epi_store_tile,
            self.out_ring_slots,
        )

    @cute.jit
    def __call__(
        self,
        activation: cute.Tensor,  # (M, K) e4m3
        activation_sf: cute.Tensor,  # (ceil(M/128) * K/128 * 512,) e8m0
        weight: cute.Tensor,  # (N, K) e4m3
        weight_sf: cute.Tensor,  # (N/128 * K/128 * 512,) e8m0
        out: cute.Tensor,  # (M, N) e4m3
        cos_sin_cache: cute.Tensor,  # (1, positions * 128) f32
        cu_q_seqlens: cute.Tensor,  # (batch + 1,) i32
        kv_cache_lengths: cute.Tensor,  # (batch,) i32
        helix_position_offsets: cute.Tensor,  # (M,) i32 | (0,) = off
        quant_scale_qkv: cute.Tensor | None,  # (1,) f32 | None = 1.0
        eps: cutlass.Float32,
        stream: cuda_driver.CUstream,
    ):
        """Build per-shape TMA atoms + per-shape scheduler params and
        launch the (mixed-cluster) mega-kernel.  Every problem size is
        carried BY the tensors: M = activation.shape[0],
        K = activation.shape[1],
        N = weight.shape[0], batch = kv_cache_lengths.shape[0],
        rope_positions = cos_sin_cache.shape[1] // 128 -- one compiled
        handle serves every shape (N a multiple of the 512 head, K of
        the 128 tile)."""
        self.a_dtype = activation.element_type
        self.b_dtype = weight.element_type
        self.sf_dtype = activation_sf.element_type
        self.c_dtype = out.element_type
        self.a_major_mode = OperandMajorMode.K
        self.b_major_mode = OperandMajorMode.K

        m = activation.shape[0]
        k = activation.shape[1]
        n = cute.assume(weight.shape[0], divby=self.head_dim)
        batch_size = kv_cache_lengths.shape[0]
        rope_positions = cos_sin_cache.shape[1] // self.rope_cache_row_floats

        self._setup_attributes()
        tiled_mma_akeep, tiled_mma_areuse, tiled_mma_sfb = self._make_tiled_mmas()

        a_tensor = cute.make_tensor(
            activation.iterator,
            cute.make_ordered_layout((m, k, 1), order=(1, 0, 2)),
        )
        b_tensor = cute.make_tensor(
            weight.iterator,
            cute.make_ordered_layout((n, k, 1), order=(1, 0, 2)),
        )
        sfa_tensor = cute.make_tensor(
            activation_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, self.sf_vec_size),
        )
        sfb_tensor = cute.make_tensor(
            weight_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(b_tensor.shape, self.sf_vec_size),
        )
        out_tensor = cute.make_tensor(
            out.iterator,
            cute.make_ordered_layout((m, n, 1), order=(1, 0, 2)),
        )

        if cutlass.const_expr(self.use_tma_store):
            tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                out_tensor,
                self.out_smem_layout_staged,
                self.epi_store_tile,
            )
        else:
            tma_atom_c, tma_tensor_c = None, None

        pref_atoms, pref_tensors = self._make_tma_atoms(
            self.cluster_shape_mn,
            tiled_mma_akeep,
            tiled_mma_sfb,
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
        )
        if cutlass.const_expr(self.fallback_cluster_shape_mn is not None):
            fb_atoms, fb_tensors = self._make_tma_atoms(
                self.fallback_cluster_shape_mn,
                tiled_mma_akeep,
                tiled_mma_sfb,
                a_tensor,
                b_tensor,
                sfa_tensor,
                sfb_tensor,
            )
        else:
            fb_atoms, fb_tensors = pref_atoms, pref_tensors

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        self.num_tma_load_bytes = (
            cute.size_in_bytes(self.a_dtype, a_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
            + cute.size_in_bytes(self.b_dtype, b_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        ) * self.atom_thr_size

        num_ctas_mnl = (
            cute.ceil_div(m, self.cta_tile_shape_mnk[0]),
            n // self.head_dim,
            1,
        )
        preferred_tile_sched_params, grid = self._compute_grid(
            num_ctas_mnl,
            self.cluster_shape_mn,
            self.swizzle_size,
            self.raster_along_m,
        )
        if cutlass.const_expr(self.fallback_cluster_shape_mn is not None):
            fallback_tile_sched_params, _ = self._compute_grid(
                num_ctas_mnl,
                self.fallback_cluster_shape_mn,
                self.swizzle_size,
                self.raster_along_m,
            )
        else:
            fallback_tile_sched_params = preferred_tile_sched_params

        self.buffer_align_bytes = 1024

        @cute.struct
        class SharedStorage:
            sRope: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(self.rope_smem_layout)],
                1024,
            ]
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_clc_stage * 2]
            clc_response: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, self.num_clc_stage * 4],
                16,
            ]
            sCuSeqlens: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, self.max_batch + 1],
                16,
            ]
            sKvLengths: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, self.max_batch],
                16,
            ]

            sOut: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    (cute.cosize(self.out_smem_layout_staged) if self.use_tma_store else 0),
                ],
                1024,
            ]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        launched = self.kernel(
            tiled_mma_akeep,
            tiled_mma_areuse,
            tiled_mma_sfb,
            pref_atoms,
            pref_tensors,
            fb_atoms,
            fb_tensors,
            out_tensor,
            tma_atom_c,
            tma_tensor_c,
            cos_sin_cache,
            cu_q_seqlens,
            kv_cache_lengths,
            helix_position_offsets,
            quant_scale_qkv,
            preferred_tile_sched_params,
            fallback_tile_sched_params,
            (m, n, batch_size, rope_positions),
            eps,
        )
        if cutlass.const_expr(self.fallback_cluster_shape_mn is None):
            launched.launch(
                grid=grid,
                block=[self.threads_per_cta, 1, 1],
                cluster=(*self.cluster_shape_mn, 1),
                stream=stream,
                min_blocks_per_mp=1,
                smem_merge_branch_allocs=True,
            )
        else:
            launched.launch(
                grid=grid,
                block=[self.threads_per_cta, 1, 1],
                cluster=(*self.cluster_shape_mn, 1),
                fallback_cluster=(*self.fallback_cluster_shape_mn, 1),
                stream=stream,
                min_blocks_per_mp=1,
                smem_merge_branch_allocs=True,
            )
        return

    @cute.jit
    def _make_tma_atoms(
        self,
        cluster_shape_mn,
        tiled_mma_akeep,
        tiled_mma_sfb,
        a_tensor,
        b_tensor,
        sfa_tensor,
        sfb_tensor,
    ):
        """One (a, b, sfa, sfb) tiled-tma-atom set for one launched
        cluster shape."""
        vmnk, sfb_vmnk, _, _ = self._cluster_layouts(
            cluster_shape_mn, tiled_mma_akeep, tiled_mma_sfb
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A(cluster_shape_mn, tiled_mma_akeep.thr_id),
            a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma_akeep,
            vmnk.shape,
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B(cluster_shape_mn, tiled_mma_akeep.thr_id),
            b_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma_akeep,
            vmnk.shape,
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A(cluster_shape_mn, tiled_mma_akeep.thr_id),
            sfa_tensor,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma_akeep,
            vmnk.shape,
            internal_type=cutlass.Int16,
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_SFB(cluster_shape_mn, tiled_mma_akeep.thr_id),
            sfb_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        return (
            (tma_atom_a, tma_atom_b, tma_atom_sfa, tma_atom_sfb),
            (tma_tensor_a, tma_tensor_b, tma_tensor_sfa, tma_tensor_sfb),
        )

    @staticmethod
    def _compute_grid(
        num_ctas_mnl,
        cluster_shape_mn,
        swizzle_size,
        raster_along_m,
    ):
        """CLC scheduler params + launch grid for ONE cluster shape.  The
        grid comes straight from the scheduler: the default knobs
        (swizzle 1, raster along M) give the FULL problem rounded up
        to the cluster; any other knob value switches the decode to a
        linear cluster index and the grid to its (cluster_x,
        cluster_y, true cluster count) form."""
        tile_sched_params = ClcDynamicPersistentTileSchedulerParams(
            num_ctas_mnl,
            (*cluster_shape_mn, 1),
            swizzle_size,
            raster_along_m,
        )
        grid = ClcDynamicPersistentTileScheduler.get_grid_shape(tile_sched_params)
        return tile_sched_params, grid

    @cute.kernel
    def kernel(
        self,
        tiled_mma_akeep: cute.TiledMma,
        tiled_mma_areuse: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        pref_atoms: Tuple,
        pref_tensors: Tuple,
        fb_atoms: Tuple,
        fb_tensors: Tuple,
        mOut: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_tma: cute.Tensor,
        cos_sin_cache: cute.Tensor,
        cu_q_seqlens: cute.Tensor,
        kv_cache_lengths: cute.Tensor,
        helix_position_offsets: cute.Tensor,
        quant_scale_qkv: cute.Tensor | None,
        preferred_tile_sched_params: ClcDynamicPersistentTileSchedulerParams,
        fallback_tile_sched_params: ClcDynamicPersistentTileSchedulerParams,
        problem_info: Tuple,
        eps: cutlass.Float32,
    ):
        self._setup_attributes()
        if cutlass.const_expr(self.fallback_cluster_shape_mn is None):
            self.kernel_body(
                tiled_mma_akeep,
                tiled_mma_areuse,
                tiled_mma_sfb,
                pref_atoms,
                pref_tensors,
                mOut,
                tma_atom_c,
                mC_tma,
                cos_sin_cache,
                cu_q_seqlens,
                kv_cache_lengths,
                helix_position_offsets,
                quant_scale_qkv,
                preferred_tile_sched_params,
                problem_info,
                eps,
                self.cluster_shape_mn,
            )
        else:
            cbdim_x, cbdim_y, cbdim_z = cute.arch.block_in_cluster_dim()
            if (cbdim_x == self.cluster_shape_mn[0]) & (cbdim_y == self.cluster_shape_mn[1]):
                self.kernel_body(
                    tiled_mma_akeep,
                    tiled_mma_areuse,
                    tiled_mma_sfb,
                    pref_atoms,
                    pref_tensors,
                    mOut,
                    tma_atom_c,
                    mC_tma,
                    cos_sin_cache,
                    cu_q_seqlens,
                    kv_cache_lengths,
                    helix_position_offsets,
                    quant_scale_qkv,
                    preferred_tile_sched_params,
                    problem_info,
                    eps,
                    self.cluster_shape_mn,
                )
            else:
                self.kernel_body(
                    tiled_mma_akeep,
                    tiled_mma_areuse,
                    tiled_mma_sfb,
                    fb_atoms,
                    fb_tensors,
                    mOut,
                    tma_atom_c,
                    mC_tma,
                    cos_sin_cache,
                    cu_q_seqlens,
                    kv_cache_lengths,
                    helix_position_offsets,
                    quant_scale_qkv,
                    preferred_tile_sched_params,
                    problem_info,
                    eps,
                    self.fallback_cluster_shape_mn,
                )

    @cute.jit
    def kernel_body(
        self,
        tiled_mma_akeep: cute.TiledMma,
        tiled_mma_areuse: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atoms: Tuple,
        tma_tensors: Tuple,
        mOut: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_tma: cute.Tensor,
        cos_sin_cache: cute.Tensor,
        cu_q_seqlens: cute.Tensor,
        kv_cache_lengths: cute.Tensor,
        helix_position_offsets: cute.Tensor,
        quant_scale_qkv: cute.Tensor | None,
        tile_sched_params: ClcDynamicPersistentTileSchedulerParams,
        problem_info: Tuple,
        eps: cutlass.Float32,
        launch_cluster_shape_mn,
    ):
        """Whole device body, specialized over ONE launched cluster
        shape (cluster layouts, multicast masks, scheduler decode)."""
        tma_atom_a, tma_atom_b, tma_atom_sfa, tma_atom_sfb = tma_atoms
        mA_mkl, mB_nkl, mSFA_mkl, mSFB_nkl = tma_tensors
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        iket.range_push("prologue")
        m, n, batch_size, rope_positions = problem_info
        # sCuSeqlens/sKvLengths are sized by max_batch at compile time. The
        # compiled handle rejects larger batches on the host; the clamp only keeps
        # the metadata staging inside its shared-memory allocation.
        batch_size = cutlass.min(batch_size, cutlass.Int32(self.max_batch))

        (
            cluster_layout_vmnk,
            cluster_layout_sfb_vmnk,
            is_a_mcast,
            is_b_mcast,
        ) = self._cluster_layouts(launch_cluster_shape_mn, tiled_mma_akeep, tiled_mma_sfb)

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            if cutlass.const_expr(self.use_tma_store):
                cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma_akeep.thr_id.shape) == 2

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma_akeep.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # CUTLASS DSL 4.5 expresses multicast signaling through the consumer
        # thread count; this is API adaptation only.
        num_mcast_ctas_a = cute.size(cluster_layout_vmnk.shape[2])
        num_mcast_ctas_b = cute.size(cluster_layout_vmnk.shape[1])
        num_tma_producers = num_mcast_ctas_a + num_mcast_ctas_b - 1
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producers),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        num_acc_consumer_threads = 32 * len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_acc_consumer_threads
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        cluster_size = cute.size(cluster_layout_vmnk.shape)
        num_clc_consumer_threads = 32 * (1 + cluster_size * (1 + len(self.epilog_warp_id) + 1))
        clc_pipeline = pipeline.PipelineClcFetchAsync.create(
            barrier_storage=storage.clc_mbar_ptr.data_ptr(),
            num_stages=self.num_clc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_clc_consumer_threads
            ),
            tx_count=self.num_clc_response_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
            arch=self.arch,
        )
        pipeline_init_arrive(cluster_shape_mn=launch_cluster_shape_mn, is_relaxed=True)

        sA = storage.sA.get_tensor(
            self.a_smem_layout_staged.outer,
            swizzle=self.a_smem_layout_staged.inner,
        )
        sB = storage.sB.get_tensor(
            self.b_smem_layout_staged.outer,
            swizzle=self.b_smem_layout_staged.inner,
        )
        sSFA = storage.sSFA.get_tensor(self.sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(self.sfb_smem_layout_staged)

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(is_a_mcast or is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfa_full_mcast_mask = a_full_mcast_mask
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk,
                block_in_cluster_coord_sfb_vmnk,
                mcast_mode=1,
            )

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        thr_mma = tiled_mma_akeep.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)

        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        tCrA = tiled_mma_akeep.make_fragment_A(sA)
        tCrB = tiled_mma_akeep.make_fragment_B(sB)
        acc_shape = tiled_mma_akeep.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma_akeep.make_fragment_C(acc_shape)

        gC_mnl = cute.local_tile(
            mOut,
            cute.slice_(self.mma_tiler, (None, None, 0)),
            (None, None, None),
        )
        tCgC = thr_mma.partition_C(gC_mnl)

        pipeline_init_wait(cluster_shape_mn=launch_cluster_shape_mn)

        clc_response_ptr = storage.clc_response.data_ptr()
        clc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_clc_stage
        )
        is_first_cta_in_cluster = cta_rank_in_cluster == 0
        tile_sched = ClcDynamicPersistentTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            clc_response_ptr,
            physical_cluster_shape_mn=launch_cluster_shape_mn,
        )
        work_tile = tile_sched.initial_work_tile_info()
        iket.range_pop()

        if warp_idx == self.tma_warp_id:
            cute.arch.setmaxregister_decrease(self.mainloop_reg_count)
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            while work_tile.is_valid_tile:
                iket.range_push("tma_tile")
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_m = cur_tile_coord[0] // cute.size(tiled_mma_akeep.thr_id.shape)
                head_id = cur_tile_coord[1]

                tAgA_slice = tAgA[(None, mma_tile_coord_m, None, 0)]
                tBgB_slice = tBgB[(None, head_id, None, 0)]
                tAgSFA_slice = tAgSFA[(None, mma_tile_coord_m, None, 0)]
                tBgSFB_slice = tBgSFB[(None, head_id, None, 0)]

                if cutlass.const_expr(self.tma_prefetch_dist > 0):
                    for pf_k_tile in cutlass.range(
                        0,
                        cutlass.min(self.tma_prefetch_dist, k_tile_cnt),
                        1,
                        unroll=1,
                    ):
                        cute.prefetch(tma_atom_a, tAgA_slice[(None, pf_k_tile)])
                        cute.prefetch(tma_atom_sfa, tAgSFA_slice[(None, pf_k_tile)])
                        cute.prefetch(tma_atom_b, tBgB_slice[(None, pf_k_tile)])
                        cute.prefetch(tma_atom_sfb, tBgSFB_slice[(None, pf_k_tile)])

                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    iket.range_push("tma_wait", k_tile)
                    ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)
                    iket.range_pop()
                    iket.range_push("tma_issue")
                    tma_bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, ab_producer_state.count)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=tma_bar,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_slice[(None, ab_producer_state.count)],
                        tAsSFA[(None, ab_producer_state.index)],
                        tma_bar_ptr=tma_bar,
                        mcast_mask=sfa_full_mcast_mask,
                    )

                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, ab_producer_state.count)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=tma_bar,
                        mcast_mask=b_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_slice[(None, ab_producer_state.count)],
                        tBsSFB[(None, ab_producer_state.index)],
                        tma_bar_ptr=tma_bar,
                        mcast_mask=sfb_full_mcast_mask,
                    )
                    if cutlass.const_expr(self.tma_prefetch_dist > 0):
                        pf_k_tile = k_tile + self.tma_prefetch_dist
                        if pf_k_tile < k_tile_cnt:
                            cute.prefetch(tma_atom_a, tAgA_slice[(None, pf_k_tile)])
                            cute.prefetch(tma_atom_sfa, tAgSFA_slice[(None, pf_k_tile)])
                            cute.prefetch(tma_atom_b, tBgB_slice[(None, pf_k_tile)])
                            cute.prefetch(tma_atom_sfb, tBgSFB_slice[(None, pf_k_tile)])
                    iket.range_pop()
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
                iket.range_push("tma_wait_clc")
                clc_pipeline.consumer_wait(clc_consumer_state)
                iket.range_pop()
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
                iket.range_pop()
            ab_pipeline.producer_tail(ab_producer_state)

        if warp_idx == self.sched_warp_id:
            cute.arch.setmaxregister_decrease(self.mainloop_reg_count)
            if is_first_cta_in_cluster:
                clc_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.ProducerConsumer,
                    self.num_clc_stage,
                )
                while work_tile.is_valid_tile:
                    iket.range_push("sched_tile")
                    iket.range_push("sched_wait_empty")
                    clc_pipeline.producer_acquire(clc_producer_state)
                    iket.range_pop()
                    iket.range_push("sched_query")
                    mbarrier_addr = clc_pipeline.producer_get_barrier(clc_producer_state)
                    tile_sched.advance_to_next_work(mbarrier_addr)
                    clc_producer_state.advance()

                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                    iket.range_pop()
                    iket.range_pop()
                clc_pipeline.producer_tail(clc_producer_state)

        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.mainloop_reg_count)
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, self.tCtSFA_layout)
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, self.tCtSFB_layout)

            sfa_s2t = self._mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            sfb_s2t = self._mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            while work_tile.is_valid_tile:
                iket.range_push("mma_tile")
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
                iket.range_push("mma_wait_acc")
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)
                iket.range_pop()

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    if is_leader_cta:
                        iket.range_push("mma_wait", k_tile)
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                        iket.range_pop()
                        iket.range_push("mma_issue")
                        self._mainloop_s2t_copies(ab_consumer_state.index, sfa_s2t, sfb_s2t)
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for k_block in cutlass.range(num_kblocks, unroll_full=True):
                            a_kblk_crd = (
                                None,
                                0,
                                k_block,
                                ab_consumer_state.index,
                            )
                            sfa_kblk_crd = (None, 0, k_block)
                            for n_set in cutlass.range_constexpr(2):
                                b_kblk_crd = (
                                    None,
                                    n_set,
                                    k_block,
                                    ab_consumer_state.index,
                                )
                                sfb_kblk_crd = (None, n_set, k_block)
                                tCtAcc_set = tCtAcc_base[(None, 0, n_set)]
                                if cutlass.const_expr(n_set == 0):
                                    tiled_mma_akeep.set(
                                        tcgen05.Field.ACCUMULATE,
                                        k_tile != 0 or k_block != 0,
                                    )
                                    cute.gemm(
                                        tiled_mma_akeep,
                                        tCtAcc_set,
                                        [
                                            tCrA[a_kblk_crd],
                                            tCtSFA[sfa_kblk_crd],
                                        ],
                                        [
                                            tCrB[b_kblk_crd],
                                            tCtSFB[sfb_kblk_crd],
                                        ],
                                        tCtAcc_set,
                                    )
                                else:
                                    tiled_mma_areuse.set(
                                        tcgen05.Field.ACCUMULATE,
                                        k_tile != 0 or k_block != 0,
                                    )
                                    cute.gemm(
                                        tiled_mma_areuse,
                                        tCtAcc_set,
                                        [
                                            tCrA[a_kblk_crd],
                                            tCtSFA[sfa_kblk_crd],
                                        ],
                                        [
                                            tCrB[b_kblk_crd],
                                            tCtSFB[sfb_kblk_crd],
                                        ],
                                        tCtAcc_set,
                                    )
                        ab_pipeline.consumer_release(ab_consumer_state)
                        iket.range_pop()
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()
                iket.range_push("mma_wait_clc")
                clc_pipeline.consumer_wait(clc_consumer_state)
                iket.range_pop()
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
                iket.range_pop()
            acc_pipeline.producer_tail(acc_producer_state)

        # CUTLASS DSL 4.5 cannot carry a cute.struct through a dynamic
        # warp-role branch. Materialize SMEM tensors before that branch.
        s_cu_seqlens = storage.sCuSeqlens.get_tensor(cute.make_layout(self.max_batch + 1))
        s_kv_lengths = storage.sKvLengths.get_tensor(cute.make_layout(self.max_batch))
        sOut = None
        if cutlass.const_expr(self.use_tma_store):
            sOut = storage.sOut.get_tensor(
                self.out_smem_layout_staged.outer,
                swizzle=self.out_smem_layout_staged.inner,
            )
        s_rope_t = storage.sRope.get_tensor(
            self.rope_smem_layout.outer,
            swizzle=self.rope_smem_layout.inner,
        )

        if warp_idx == self.pad_warp_id:
            cute.arch.setmaxregister_decrease(self.mainloop_reg_count)

        if warp_idx < self.mma_warp_id:
            cute.arch.setmaxregister_increase(self.epilog_reg_count)
            for meta_i in cutlass.range(0, (batch_size + 128) // 128, 1, unroll=1):
                meta_idx = tidx + meta_i * cutlass.Int32(128)
                if meta_idx < batch_size + 1:
                    s_cu_seqlens[meta_idx] = cu_q_seqlens[meta_idx]
                if meta_idx < batch_size:
                    s_kv_lengths[meta_idx] = kv_cache_lengths[meta_idx]

            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            if cutlass.const_expr(quant_scale_qkv is None):
                quant_scale_value = cutlass.Float32(1.0)
            else:
                quant_scale_value = quant_scale_qkv[0]

            tCtAcc_epi = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            tCtAcc_x = gemm_sm100.transform_partitioned_tensor_layout(tCtAcc_epi)
            tCgC_x = gemm_sm100.transform_partitioned_tensor_layout(tCgC)
            epi_tile = (self.cta_tile_shape_mnk[0], self.epi_chunk)
            tAcc_epi = cute.flat_divide(tCtAcc_x, epi_tile)
            gC_epi = cute.flat_divide(tCgC_x, epi_tile)
            copy_atom_t2r = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.epi_chunk)),
                self.acc_dtype,
            )
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0)])
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
            tTR_gC_part = thr_copy_t2r.partition_D(gC_epi)
            tTR_rAcc = cute.make_rmem_tensor(
                tTR_gC_part[(None, None, None, 0, 0, 0, 0, 0)].shape,
                self.acc_dtype,
            )
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tTR_rAcc_flat = cute.make_tensor(tTR_rAcc.iterator, cute.make_layout((self.epi_chunk,)))
            tTR_rC_flat = cute.make_tensor(tTR_rC.iterator, cute.make_layout((self.epi_chunk,)))
            simt_atom = cute.make_copy_atom(
                cute.nvgpu.CopyR2GOp(),
                self.c_dtype,
                num_bits_per_copy=256,
                l1c_evict_priority=CacheEvictionPriority.NO_ALLOCATE,
            )
            tTR_rC_quads = cute.tiled_divide(tTR_rC_flat, (16,))
            sts_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.c_dtype,
                num_bits_per_copy=128,
            )
            pred_store = cute.make_rmem_tensor((1, *tTR_rC.shape[1:]), cutlass.Boolean)
            if cutlass.const_expr(self.use_tma_store):
                s_out_quads = cute.tiled_divide(sOut, (1, 16))
                gC_store = cute.local_tile(mC_tma, self.epi_store_tile, (None, None, None))
                bSG_sC, bSG_gC = cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sOut, 0, 2),
                    cute.group_modes(gC_store, 0, 2),
                )

            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            s_rope_quads = cute.tiled_divide(s_rope_t, (1, 4))
            cache_quads = cute.tiled_divide(cos_sin_cache, (1, 4))
            cs_copy_atom = cute.make_copy_atom(
                cpasync.CopyG2SOp(), cutlass.Float32, num_bits_per_copy=128
            )
            lane = tidx & cutlass.Int32(31)
            row_in_cta = warp_idx * cutlass.Int32(32) + lane
            cp_chunk = lane & cutlass.Int32(self.rope_cp_chunks - 1)
            cp_row_sel = lane >> cutlass.Int32(4)
            warp_row0 = warp_idx * cutlass.Int32(32)
            cache_qcols = cache_quads[((0, None), 0, None)]

            while work_tile.is_valid_tile:
                iket.range_push("epi_tile")
                cur_tile_coord = work_tile.tile_idx
                head_id = cur_tile_coord[1]
                coord_m_cta = cur_tile_coord[0] * cutlass.Int32(self.cta_tile_shape_mnk[0])
                row = coord_m_cta + row_in_cta
                row_in_bounds = row < m
                if cutlass.const_expr(self.use_tma_store):
                    bSG_gC_tile = bSG_gC[(None, cur_tile_coord[0], None, 0)]
                else:
                    pred_store[(0, 0, 0)] = row_in_bounds
                mma_coord_m = cur_tile_coord[0] // cute.size(tiled_mma_akeep.thr_id.shape)
                tTR_gC_tile = tTR_gC_part[(None, None, None, None, None, mma_coord_m, head_id, 0)]

                iket.range_push("epi_position")
                position = cutlass.Int32(-1)
                if row_in_bounds:
                    candidate = self._position_for_row(
                        s_cu_seqlens,
                        s_kv_lengths,
                        helix_position_offsets,
                        row,
                        batch_size,
                    )
                    if candidate >= 0 and candidate < rope_positions:
                        position = candidate
                iket.range_pop()

                iket.range_push("epi_cs_stage")
                pos_qcol_base = position * cutlass.Int32(self.rope_cache_row_floats // 4)
                for cp_i in cutlass.range_constexpr(self.rope_cp_iters):
                    src_lane = cutlass.Int32(2 * cp_i) + cp_row_sel
                    src_qcol = cute.arch.shuffle_sync(pos_qcol_base, src_lane)
                    if src_qcol >= 0:
                        cute.copy(
                            cs_copy_atom,
                            cache_qcols[(None, src_qcol + cp_chunk)],
                            s_rope_quads[((0, None), warp_row0 + src_lane, cp_chunk)],
                        )
                cute.arch.cp_async_commit_group()
                iket.range_pop()

                iket.range_push("epi_wait_acc")
                acc_pipeline.consumer_wait(acc_consumer_state)
                iket.range_pop()

                iket.range_push("epi_rmsnorm_reduce")
                f32_zero = self.acc_dtype(0.0)
                sum_sq_pairs = [(self.acc_dtype(0.0), self.acc_dtype(0.0)) for _ in range(4)]
                for chunk in cutlass.range_constexpr(self.epi_chunks):
                    cute.copy(
                        tiled_copy_t2r,
                        tTR_tAcc[(None, None, None, 0, chunk)],
                        tTR_rAcc,
                    )
                    for pair in cutlass.range_constexpr(self.epi_chunk // 2):
                        pair_vec = (
                            tTR_rAcc_flat[pair * 2],
                            tTR_rAcc_flat[pair * 2 + 1],
                        )
                        sum_sq_pairs[pair & 3] = cute.arch.fma_packed_f32x2(
                            pair_vec, pair_vec, sum_sq_pairs[pair & 3]
                        )
                sum_sq_pairs01 = cute.arch.add_packed_f32x2(sum_sq_pairs[0], sum_sq_pairs[1])
                sum_sq_pairs23 = cute.arch.add_packed_f32x2(sum_sq_pairs[2], sum_sq_pairs[3])
                sum_sq_pair = cute.arch.add_packed_f32x2(sum_sq_pairs01, sum_sq_pairs23)
                sum_sq = sum_sq_pair[0] + sum_sq_pair[1]
                inv_rms = cute.math.rsqrt(sum_sq / self.acc_dtype(self.head_dim) + eps)
                norm_quant_scale = inv_rms * quant_scale_value
                iket.range_pop()

                iket.range_push("epi_cs_drain")
                cute.arch.cp_async_wait_group(0)
                if cutlass.const_expr(self.use_tma_store):
                    if warp_idx == 0:
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                self.epilog_sync_barrier.arrive_and_wait()
                iket.range_pop()

                for rope_rev in cutlass.range_constexpr(self.rope_chunks):
                    chunk = self.epi_chunks - 1 - rope_rev
                    if cutlass.const_expr(rope_rev != 0):
                        cute.copy(
                            tiled_copy_t2r,
                            tTR_tAcc[(None, None, None, 0, chunk)],
                            tTR_rAcc,
                        )
                    iket.range_push("epi_rope")
                    col_base = chunk * self.epi_chunk
                    for quad_i in cutlass.range_constexpr(self.epi_chunk // 4):
                        elem0 = quad_i * 4
                        quad_idx = (col_base - self.qk_nope_head_dim) // 4 + quad_i
                        cs_quad = s_rope_quads[((0, None), row_in_cta, quad_idx)].load()
                        for sub in cutlass.range_constexpr(2):
                            e0 = elem0 + sub * 2
                            cos_value = cs_quad[sub * 2]
                            sin_value = cs_quad[sub * 2 + 1]
                            x0, x1 = _fma_packed_f32x2_bf16x2_f32x2_f32x2(
                                (
                                    tTR_rAcc_flat[e0],
                                    tTR_rAcc_flat[e0 + 1],
                                ),
                                (inv_rms, inv_rms),
                                (f32_zero, f32_zero),
                            )
                            t_pair = _fma_packed_f32x2_bf16x2_f32x2_f32x2(
                                (x0, x1),
                                (cos_value, cos_value),
                                (f32_zero, f32_zero),
                            )
                            y0, y1 = _fma_packed_f32x2_bf16x2_f32x2_f32x2(
                                (x1, x0),
                                (-sin_value, sin_value),
                                t_pair,
                            )
                            o0, o1 = _fma_packed_f32x2_bf16x2_f32x2_f32x2(
                                (y0, y1),
                                (quant_scale_value, quant_scale_value),
                                (f32_zero, f32_zero),
                            )
                            tTR_rAcc_flat[e0] = o0
                            tTR_rAcc_flat[e0 + 1] = o1
                    out_ssa = tTR_rAcc_flat.load()
                    iket.range_pop()
                    iket.range_push("epi_quant+stg")
                    tTR_rC_flat.store(out_ssa.to(self.c_dtype))
                    if cutlass.const_expr(self.use_tma_store):
                        q0 = (chunk & 3) * 2
                        slot = (chunk >> 2) & (self.out_ring_slots - 1)
                        for h in cutlass.range_constexpr(2):
                            cute.copy(
                                sts_atom,
                                tTR_rC_quads[(None, h)],
                                s_out_quads[((0, None), row_in_cta, q0 + h, slot)],
                            )
                    else:
                        cute.copy(
                            simt_atom,
                            tTR_rC,
                            tTR_gC_tile[(None, None, None, 0, chunk)],
                            pred=pred_store,
                        )
                    iket.range_pop()
                for chunk_i in cutlass.range_constexpr(self.nope_chunks):
                    chunk = self.nope_chunks - 1 - chunk_i
                    cute.copy(
                        tiled_copy_t2r,
                        tTR_tAcc[(None, None, None, 0, chunk)],
                        tTR_rAcc,
                    )
                    if cutlass.const_expr(chunk_i % 2 == 1):
                        cute.arch.fence_view_async_tmem_load()
                    for pair in cutlass.range_constexpr(self.epi_chunk // 2):
                        e0 = pair * 2
                        o0, o1 = _fma_packed_f32x2_bf16x2_f32x2_f32x2(
                            (
                                tTR_rAcc_flat[e0],
                                tTR_rAcc_flat[e0 + 1],
                            ),
                            (norm_quant_scale, norm_quant_scale),
                            (f32_zero, f32_zero),
                        )
                        tTR_rAcc_flat[e0] = o0
                        tTR_rAcc_flat[e0 + 1] = o1
                    out_ssa = tTR_rAcc_flat.load()
                    if cutlass.const_expr(chunk_i == self.nope_chunks - 1):
                        nvvm.tcgen05_fence(nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC)
                        acc_pipeline.consumer_release(acc_consumer_state)
                        acc_consumer_state.advance()
                    iket.range_push("epi_quant+stg")
                    tTR_rC_flat.store(out_ssa.to(self.c_dtype))
                    if cutlass.const_expr(self.use_tma_store):
                        q0 = (chunk & 3) * 2
                        slot = (chunk >> 2) & (self.out_ring_slots - 1)
                        for h in cutlass.range_constexpr(2):
                            cute.copy(
                                sts_atom,
                                tTR_rC_quads[(None, h)],
                                s_out_quads[((0, None), row_in_cta, q0 + h, slot)],
                            )
                    else:
                        cute.copy(
                            simt_atom,
                            tTR_rC,
                            tTR_gC_tile[(None, None, None, 0, chunk)],
                            pred=pred_store,
                        )
                    iket.range_pop()

                if cutlass.const_expr(self.use_tma_store):
                    iket.range_push("epi_fence+tmastg")
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()
                    if warp_idx == 0:
                        for plane in cutlass.range_constexpr(self.planes_per_head):
                            cute.copy(
                                tma_atom_c,
                                bSG_sC[(None, plane)],
                                bSG_gC_tile[
                                    (
                                        None,
                                        head_id * self.planes_per_head + plane,
                                    )
                                ],
                            )
                        cute.arch.cp_async_bulk_commit_group()
                    iket.range_pop()

                iket.range_push("epi_wait_clc")
                clc_pipeline.consumer_wait(clc_consumer_state)
                iket.range_pop()
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
                iket.range_pop()

            iket.range_push("epi_wait_tail")
            if cutlass.const_expr(self.use_tma_store):
                if warp_idx == 0:
                    cute.arch.cp_async_bulk_wait_group(0)
            tmem.relinquish_alloc_permit()
            tmem.free(acc_tmem_ptr)
            iket.range_pop()

    @cute.jit
    def _mainloop_s2t_copy_and_partition(self, sSF: cute.Tensor, tSF: cute.Tensor) -> S2TCopyBundle:
        """Tiled S2T (UTCCP) copy for one SF tensor + its partitions."""
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        def append_mn_broadcast_mode(smem_layout: cute.Layout):
            mn_dim = cute.get(smem_layout, mode=[0, 0])
            mn_dim = cute.append(mn_dim, cute.make_layout((4), stride=(0)))
            layout = cute.append(cute.group_modes(mn_dim, 0), cute.get(smem_layout, mode=[0, 1]))
            layout = cute.append(cute.group_modes(layout, 0), cute.get(smem_layout, mode=[1]))
            layout = cute.append(layout, cute.get(smem_layout, mode=[2]))
            layout = cute.append(layout, cute.get(smem_layout, mode=[3]))
            return layout

        tCsSF_compact_bcast = cute.make_tensor(
            tCsSF_compact.iterator,
            append_mn_broadcast_mode(tCsSF_compact.layout),
        )
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact_bcast)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return S2TCopyBundle(tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t)

    @cute.jit
    def _mainloop_s2t_copies(
        self,
        stage_idx,
        sfa_s2t_bundle: S2TCopyBundle,
        sfb_s2t_bundle: S2TCopyBundle,
    ):
        s2t_stage_coord = (None, None, None, None, stage_idx)
        cute.copy(
            sfa_s2t_bundle.tiled_copy,
            sfa_s2t_bundle.sSF_compact[s2t_stage_coord],
            sfa_s2t_bundle.tSF_compact,
        )
        cute.copy(
            sfb_s2t_bundle.tiled_copy,
            sfb_s2t_bundle.sSF_compact[s2t_stage_coord],
            sfb_s2t_bundle.tSF_compact,
        )

    @cute.jit
    def _position_for_row(
        self,
        s_cu_seqlens: cute.Tensor,  # smem (max_batch + 1,) i32
        s_kv_lengths: cute.Tensor,  # smem (max_batch,) i32
        helix_position_offsets: cute.Tensor,
        row,
        batch_size,
    ):
        position = cutlass.Int32(0)
        if cutlass.Int32(helix_position_offsets.shape[0]) > cutlass.Int32(0):
            position = helix_position_offsets[row]
        else:
            seg_start = s_cu_seqlens[0]
            for batch_idx in cutlass.range(batch_size, unroll=1):
                seg_end = s_cu_seqlens[batch_idx + 1]
                if row >= seg_start and row < seg_end:
                    local_token = row - seg_start
                    current_len = seg_end - seg_start
                    context_len = s_kv_lengths[batch_idx] - current_len
                    position = context_len + local_token
                seg_start = seg_end
        return position


class _CompiledHandle:
    """Compiled kernel plus the host-side batch check.

    The kernel stages ``cu_q_seqlens``/``kv_cache_lengths`` into shared memory
    sized by ``max_batch`` at compile time, so a larger batch is rejected here
    instead of being clamped silently on the device.
    """

    def __init__(self, compiled, max_batch: int):
        self.compiled = compiled
        self.max_batch = max_batch

    def __call__(self, *args):
        kv_cache_lengths = args[7]
        batch_size = int(kv_cache_lengths.shape[0])
        if batch_size > self.max_batch:
            raise ValueError(
                f"batch_size {batch_size} exceeds max_batch {self.max_batch} of the compiled kernel"
            )
        return self.compiled(*args)


_COMPILE_CACHE: dict = {}


def compile(
    *,
    mma_inst_tile: Tuple[int, int] = (128, 256),
    cluster_shape_mn: Tuple[int, int] | None = None,
    fallback_cluster_shape_mn: Tuple[int, int] | None = None,
    store_mode: str = "stg256",
    swizzle_size: int = 1,
    raster_along_m: bool = True,
    with_quant_scale: bool = True,
    tma_prefetch_dist: int = 0,
    max_batch: int = 128,
):
    """Compile ONE handle serving every (M, N, K) problem shape.

    The CLC grid is the full problem, so no occupancy query is needed
    at compile time either.  swizzle_size / raster_along_m are STATIC
    (each value is its own compiled artifact)."""
    arch = os.environ.get("CUTE_DSL_ARCH")
    if arch is not None and arch != "sm_107a":
        raise RuntimeError("requires CUTE_DSL_ARCH=sm_107a")

    op = CuteBlockScaledGemmFusedRMSNormRopeQuant(
        mma_inst_tile=tuple(mma_inst_tile),
        cluster_shape_mn=cluster_shape_mn,
        fallback_cluster_shape_mn=fallback_cluster_shape_mn,
        store_mode=store_mode,
        swizzle_size=swizzle_size,
        raster_along_m=raster_along_m,
        tma_prefetch_dist=tma_prefetch_dist,
        max_batch=max_batch,
    )
    key = (
        tuple(mma_inst_tile),
        op.cluster_shape_mn,
        op.fallback_cluster_shape_mn,
        op.store_mode,
        op.swizzle_size,
        op.raster_along_m,
        bool(with_quant_scale),
        op.tma_prefetch_dist,
        op.max_batch,
    )
    if key in _COMPILE_CACHE:
        return _COMPILE_CACHE[key]

    from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

    sym_m = cute.sym_int64()
    sym_n = cute.sym_int64()
    sym_k = cute.sym_int64()
    sym_qsf = cute.sym_int64()
    sym_wsf = cute.sym_int64()
    sym_batch = cute.sym_int64()
    sym_positions = cute.sym_int64()
    fake_q = make_fake_compact_tensor(
        cutlass.Float8E4M3FN,
        (sym_m, sym_k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    fake_w = make_fake_compact_tensor(
        cutlass.Float8E4M3FN,
        (sym_n, sym_k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    fake_qsf = make_fake_compact_tensor(cutlass.Float8E8M0FNU, (sym_qsf,), assumed_align=16)
    fake_wsf = make_fake_compact_tensor(cutlass.Float8E8M0FNU, (sym_wsf,), assumed_align=16)
    fake_out = make_fake_compact_tensor(
        cutlass.Float8E4M3FN,
        (sym_m, sym_n),
        stride_order=(1, 0),
        assumed_align=32,
    )
    sym_cu = cute.sym_int64()
    sym_helix = cute.sym_int64()
    fake_cache = make_fake_compact_tensor(
        cutlass.Float32,
        (1, sym_positions * 128),
        stride_order=(1, 0),
        assumed_align=16,
    )
    fake_cu = make_fake_compact_tensor(cutlass.Int32, (sym_cu,), assumed_align=4)
    fake_kv = make_fake_compact_tensor(cutlass.Int32, (sym_batch,), assumed_align=4)
    fake_helix = make_fake_compact_tensor(cutlass.Int32, (sym_helix,), assumed_align=4)
    fake_qs = (
        make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)
        if with_quant_scale
        else None
    )
    fake_stream = make_fake_stream()

    print(
        "Compile kernel cute_blockscaled_gemm_fused_rmsnorm_rope_quant ... ",
        end="",
        flush=True,
    )
    compiled = cute.compile(
        op,
        fake_q,
        fake_qsf,
        fake_w,
        fake_wsf,
        fake_out,
        fake_cache,
        fake_cu,
        fake_kv,
        fake_helix,
        fake_qs,
        cutlass.Float32(op.rms_norm_eps),
        fake_stream,
        options="--enable-tvm-ffi",
    )
    print("OK", flush=True)
    handle = _CompiledHandle(compiled, op.max_batch)
    _COMPILE_CACHE[key] = handle
    return handle
