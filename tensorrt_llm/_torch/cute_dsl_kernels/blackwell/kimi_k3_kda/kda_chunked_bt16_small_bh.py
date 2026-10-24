# Copyright (c) 2026 by FlashInfer team.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""FlashInfer Small-BH KDA, with indexed recurrent-state pool access.

Vendored from flashinfer-ai/flashinfer commit
b8e81a7dae6c2aa5c9aa19c06e81e9a729271493,
flashinfer/kda_kernels/kda_chunked_bt16_small_bh.py (Apache-2.0).
Local changes add state-slot indirection and padded pool-row strides (64-bit row
offsets), reject unwarmed graph specializations, use the active device SM count
for scheduling, and replace K1's 16x16 Neumann-chain chunk inverse with the BT16
engine's FP16 block inverse (the chain is unstable on repeated tokens).
The other K1/K2 arithmetic and the layouts are retained from upstream.
"""

import functools
import math
from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.core import _pack_shape
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cutlass_dsl import dsl_user_op


RCP_LN2 = 1.4426950408889634


def layout_separate(thr, src, ref):
    lt = cute.make_layout(())
    ge = cute.make_layout(())

    for k, v in enumerate(ref):
        if cutlass.const_expr(v < thr):
            lt = cute.append(lt, src[k])
        else:
            ge = cute.append(ge, src[k])

    r = None
    if cutlass.const_expr(cute.rank(lt) == 1):
        r = cute.append(lt, ge)
    else:
        r = cute.append(cute.append(cute.make_layout(()), lt), ge)
    return r


@cute.jit
def layout_acc_mn(tiled_mma, acc):
    separated = layout_separate(
        tiled_mma.shape_mnk[0], acc[0], tiled_mma.tv_layout_C.stride[1]
    )

    V_M = separated[0]
    V_N = separated[1]
    V_M1 = None
    V_N1 = None
    if cutlass.const_expr(cute.rank(V_M) == 1):
        V_M1 = cute.append(V_M, acc[1])
    else:
        V_M1 = cute.append(cute.append(cute.make_layout(()), V_M), acc[1])

    if cutlass.const_expr(cute.rank(V_N) == 1):
        V_N1 = cute.append(V_N, acc[2])
    else:
        V_N1 = cute.append(cute.append(cute.make_layout(()), V_N), acc[2])

    r = None
    if cutlass.const_expr(cute.rank(V_M1) == 1):
        r = cute.append(V_M1, V_N1)
    else:
        r = cute.append(cute.append(cute.make_layout(()), V_M1), V_N1)
    return r


@cute.jit
def get_seq_len(cu_seqlens: cute.Tensor, b_idx: cutlass.Int32) -> cutlass.Int32:
    return cu_seqlens[b_idx + 1] - cu_seqlens[b_idx]


@cute.jit
def get_seq_offset(
    cu_seqlens: cute.Tensor,
    b_idx: cutlass.Int32,
    *,
    padding: cutlass.Int32 = 0,
    index_unit: cutlass.Int32 = 1,
) -> cutlass.Int32:
    """Return a packed sequence offset in tokens or token-sized records.

    Unpadded tensors start sequence ``b`` at ``cu_seqlens[b]``. Padded
    intermediates reserve ``padding`` tokens per preceding sequence and start
    it at ``cu_seqlens[b] + b * padding``. ``index_unit`` converts that token
    offset to another token-indexed storage unit, such as BT-sized records.
    """
    offset = cu_seqlens[b_idx]
    if cutlass.const_expr(padding > 0):
        offset += b_idx * padding
    if cutlass.const_expr(index_unit != 1):
        offset = offset // index_unit
    return offset


@dsl_user_op
def tanh(
    x: cutlass.Float32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass.Float32.mlir_type,
            [x.ir_value(loc=loc, ip=ip)],
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def movmatrix(
    x: cutlass.Uint32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Uint32:
    return cutlass.Uint32(
        llvm.inline_asm(
            cutlass.Uint32.mlir_type,
            [x.ir_value(loc=loc, ip=ip)],
            "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class ChunkKdaFwdK1:
    def __init__(
        self,
        D: int = 128,
        H: int = 32,
        BT: int = 16,
        gate_lower_bound: bool = False,
        max_active_clusters: int = 152,
    ):
        self.D = D
        self.H = H
        self.BT = BT
        self.gate_lower_bound = gate_lower_bound
        self.max_active_clusters = max_active_clusters

        self.CT = 4
        self.TT = self.CT * self.BT
        self.load_stages = 2
        self.gate_mma_stages = 2

        self.beta_tma_heads = 8
        # H-major BF16 beta needs a 16-byte contiguous TMA dimension
        self.beta_use_tma = self.H % self.beta_tma_heads == 0
        self.beta_stages = (
            self.load_stages + self.gate_mma_stages if self.beta_use_tma else 4
        )

        self.warp_threads = 32
        self.warpgroup_threads = 128
        self.warps_per_warpgroup = self.warpgroup_threads // self.warp_threads

        self.threads_per_cta = self.warpgroup_threads * 3
        self.load_warp_id = 0
        self.beta_warp_id = 1
        self.aux_warpgroup_id = 0
        self.gate_warpgroup_id = 1
        self.mma_warpgroup_id = 2

        self.load_register_requirement = 48
        self.gate_register_requirement = 256
        self.mma_register_requirement = 200

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (B, T, H, D)
        mK: cute.Tensor,  # (B, T, H, D)
        mG: cute.Tensor,  # (B, T, H, D)
        mBeta: cute.Tensor,  # (B, T, H)
        mDtBias: cute.Tensor,  # (H, D)
        mAlog: cute.Tensor,  # (H,)
        mQd: cute.Tensor,  # (B, T, H, D)
        mKd: cute.Tensor,  # (B, T, H, D)
        mKr: cute.Tensor,  # (B, T, H, D)
        mMqk: cute.Tensor,  # (B, NT, H, BT, BT)
        mMkk: cute.Tensor,  # (B, NT, H, BT, BT)
        mGk: cute.Tensor,  # (B, NT, H, D)
        scale: cutlass.Float32,
        gate_scale: cutlass.Float32,
        mCuSeqlens: Optional[cute.Tensor] = None,  # (B + 1,)
        max_seqlen: cutlass.Int32 = 0,
        stream=None,
    ):
        if cutlass.const_expr(mCuSeqlens is not None):
            B = cute.size(mCuSeqlens) - 1
            num_tiles = cute.ceil_div(max_seqlen, self.TT)
            mQ = cute.make_tensor(
                mQ.iterator, cute.select(mQ.layout, mode=[0, 2, 1])
            )  # (T, D, H)
            mK = cute.make_tensor(
                mK.iterator, cute.select(mK.layout, mode=[0, 2, 1])
            )  # (T, D, H)
            mG = cute.make_tensor(
                mG.iterator, cute.select(mG.layout, mode=[0, 2, 1])
            )  # (T, D, H)
            mBeta = cute.make_tensor(
                mBeta.iterator, cute.select(mBeta.layout, mode=[0, 1])
            )  # (T, H)
            mQd = cute.make_tensor(
                mQd.iterator, cute.select(mQd.layout, mode=[0, 2, 1])
            )  # (Tp, D, H)
            mKd = cute.make_tensor(
                mKd.iterator, cute.select(mKd.layout, mode=[0, 2, 1])
            )  # (Tp, D, H)
            mKr = cute.make_tensor(
                mKr.iterator, cute.select(mKr.layout, mode=[0, 2, 1])
            )  # (Tp, D, H)
            mMqk = cute.make_tensor(
                mMqk.iterator, cute.select(mMqk.layout, mode=[2, 3, 0, 1])
            )  # (BT, BT, NT, H)
            mMkk = cute.make_tensor(
                mMkk.iterator, cute.select(mMkk.layout, mode=[2, 3, 0, 1])
            )  # (BT, BT, NT, H)
            mGk = cute.make_tensor(
                mGk.iterator, cute.select(mGk.layout, mode=[2, 0, 1])
            )  # (D, NT, H)
        else:
            B, _, _, _ = mQ.shape
            NT = cute.size(mMqk, mode=[1])
            num_tiles = cute.ceil_div(NT, self.CT)
            mQ = cute.make_tensor(
                mQ.iterator, cute.select(mQ.layout, mode=[1, 3, 2, 0])
            )  # (T, D, H, B)
            mK = cute.make_tensor(
                mK.iterator, cute.select(mK.layout, mode=[1, 3, 2, 0])
            )  # (T, D, H, B)
            mG = cute.make_tensor(
                mG.iterator, cute.select(mG.layout, mode=[1, 3, 2, 0])
            )  # (T, D, H, B)
            mBeta = cute.make_tensor(
                mBeta.iterator, cute.select(mBeta.layout, mode=[1, 2, 0])
            )  # (T, H, B)
            mQd = cute.make_tensor(
                mQd.iterator, cute.select(mQd.layout, mode=[1, 3, 2, 0])
            )  # (T, D, H, B)
            mKd = cute.make_tensor(
                mKd.iterator, cute.select(mKd.layout, mode=[1, 3, 2, 0])
            )  # (T, D, H, B)
            mKr = cute.make_tensor(
                mKr.iterator, cute.select(mKr.layout, mode=[1, 3, 2, 0])
            )  # (T, D, H, B)
            mMqk = cute.make_tensor(
                mMqk.iterator, cute.select(mMqk.layout, mode=[3, 4, 1, 2, 0])
            )  # (BT, BT, NT, H, B)
            mMkk = cute.make_tensor(
                mMkk.iterator, cute.select(mMkk.layout, mode=[3, 4, 1, 2, 0])
            )  # (BT, BT, NT, H, B)
            mGk = cute.make_tensor(
                mGk.iterator, cute.select(mGk.layout, mode=[3, 1, 2, 0])
            )  # (D, NT, H, B)
        mDtBias = cute.make_tensor(
            mDtBias.iterator, cute.select(mDtBias.layout, mode=[1, 0])
        )  # (H, D)

        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(mQ.element_type, cutlass.Float32, (16, 8, 16)),
            permutation_mnk=(self.BT, self.BT, 16),
        )
        # The chunk inverse runs on FP16 operands (FP32 accumulation) whatever the
        # input dtype; same atom shape, so its fragments share the layouts above.
        tiled_mma_inv = cute.make_tiled_mma(
            warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 16)),
            permutation_mnk=(self.BT, self.BT, 16),
        )
        copy_atom_QKd = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            mQ.element_type,
        )
        tiled_copy_QKd = cute.make_tiled_copy_A(copy_atom_QKd, tiled_mma)
        tiled_copy_Ki = cute.make_tiled_copy_B(copy_atom_QKd, tiled_mma)

        sQKG_layout_staged = cute.make_layout(
            (self.TT, self.D, self.load_stages),
            stride=(self.D, 1, self.TT * self.D),
        )
        sQKG_layout = cute.slice_(sQKG_layout_staged, (None, None, 0))

        if cutlass.const_expr(self.beta_use_tma):
            sBeta_layout_staged = cute.make_layout(
                (self.TT, self.beta_tma_heads, self.beta_stages),
                stride=(self.beta_tma_heads, 1, self.TT * self.beta_tma_heads),
            )
            sBeta_layout = cute.slice_(sBeta_layout_staged, (None, None, 0))
        else:
            sBeta_layout_staged = cute.make_layout(
                (self.TT, self.beta_stages),
                stride=(1, self.TT),
            )
            sBeta_layout = cute.slice_(sBeta_layout_staged, (None, 0))

        sDtBias_layout_staged = cute.make_layout(
            (self.D, self.load_stages),
            stride=(1, self.D),
        )
        sDtBias_layout = cute.slice_(sDtBias_layout_staged, (None, 0))

        sAlog_layout = cute.make_layout((self.H,))

        smem_layout_atom_sw128 = cute.nvgpu.warpgroup.make_smem_layout_atom(
            cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128,
            mQd.element_type,
        )
        sQKd_layout_staged = cute.tile_to_shape(
            smem_layout_atom_sw128,
            (self.TT, self.D, self.gate_mma_stages),
            order=(0, 1, 2),
        )
        sQKd_layout = cute.slice_(sQKd_layout_staged, (None, None, 0))

        buffer_align_bytes = 1024

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[
                cute.struct.MemRange[mQ.element_type, cute.cosize(sQKG_layout_staged)],
                buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[mK.element_type, cute.cosize(sQKG_layout_staged)],
                buffer_align_bytes,
            ]
            sG: cute.struct.Align[
                cute.struct.MemRange[mG.element_type, cute.cosize(sQKG_layout_staged)],
                buffer_align_bytes,
            ]
            sBeta: cute.struct.Align[
                cute.struct.MemRange[
                    mBeta.element_type, cute.cosize(sBeta_layout_staged)
                ],
                buffer_align_bytes,
            ]
            sDtBias: cute.struct.Align[
                cute.struct.MemRange[
                    mDtBias.element_type, cute.cosize(sDtBias_layout_staged)
                ],
                buffer_align_bytes,
            ]
            sAlog: cute.struct.Align[
                cute.struct.MemRange[mAlog.element_type, cute.cosize(sAlog_layout)],
                buffer_align_bytes,
            ]
            sQd: cute.struct.Align[
                cute.struct.MemRange[mQ.element_type, cute.cosize(sQKd_layout_staged)],
                buffer_align_bytes,
            ]
            sKd: cute.struct.Align[
                cute.struct.MemRange[mK.element_type, cute.cosize(sQKd_layout_staged)],
                buffer_align_bytes,
            ]
            sKi: cute.struct.Align[
                cute.struct.MemRange[mK.element_type, cute.cosize(sQKd_layout_staged)],
                buffer_align_bytes,
            ]
            tma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_stages * 2]
            gate_mma_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.gate_mma_stages * 2
            ]
            beta_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.beta_stages * 2]

        beta_tma_load_bytes = (
            cute.size_in_bytes(mBeta.element_type, sBeta_layout)
            if self.beta_use_tma
            else 0
        )
        self.num_tma_load_bytes = (
            cute.size_in_bytes(mQ.element_type, sQKG_layout)
            + cute.size_in_bytes(mK.element_type, sQKG_layout)
            + cute.size_in_bytes(mG.element_type, sQKG_layout)
            + beta_tma_load_bytes
            + cute.size_in_bytes(mDtBias.element_type, sDtBias_layout)
        )

        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mQ,
            sQKG_layout,
            (self.TT, self.D),
        )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mK,
            sQKG_layout,
            (self.TT, self.D),
        )
        tma_atom_G, tma_tensor_G = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mG,
            sQKG_layout,
            (self.TT, self.D),
        )
        tma_atom_Beta, tma_tensor_Beta = None, None
        if cutlass.const_expr(self.beta_use_tma):
            tma_atom_Beta, tma_tensor_Beta = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mBeta,
                sBeta_layout,
                (self.TT, self.beta_tma_heads),
            )
        tma_atom_dtBias, tma_tensor_dtBias = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mDtBias,
            sDtBias_layout,
            (self.D,),
        )
        tma_atom_Qd, tma_tensor_Qd = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mQd,
            sQKd_layout,
            (self.TT, self.D),
        )
        tma_atom_Kd, tma_tensor_Kd = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mKd,
            sQKd_layout,
            (self.TT, self.D),
        )

        qkg_elems_per_copy = 8
        copy_atom_QKG = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mQ.element_type,
            num_bits_per_copy=qkg_elems_per_copy * mQ.element_type.width,
        )
        tv_layout_QKG = cute.make_layout(
            (
                (self.warp_threads // 2, 2),
                (qkg_elems_per_copy, self.BT // 2),
            ),
            stride=(
                (qkg_elems_per_copy * self.BT, self.BT // 2),
                (self.BT, 1),
            ),
        )
        tiled_copy_QKG = cute.make_tiled_copy(
            copy_atom_QKG, tv_layout_QKG, (self.BT, self.D)
        )

        copy_atom_dtBias = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mDtBias.element_type,
            num_bits_per_copy=qkg_elems_per_copy * mQ.element_type.width,
        )
        copy_atom_Gk = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mGk.element_type,
            num_bits_per_copy=2 * mGk.element_type.width,
        )
        tv_layout_dtBias_Gk = cute.make_layout(
            ((self.warp_threads // 2, 2), qkg_elems_per_copy),
            stride=((qkg_elems_per_copy, 0), 1),
        )
        tiled_copy_dtBias = cute.make_tiled_copy(
            copy_atom_dtBias, tv_layout_dtBias_Gk, (self.D,)
        )
        tiled_copy_Gk = cute.make_tiled_copy(
            copy_atom_Gk, tv_layout_dtBias_Gk, (self.D,)
        )

        tile_sched_params = utils.PersistentTileSchedulerParams(
            problem_shape_ntile_mnl=(num_tiles, self.H, B),
            cluster_shape_mnk=(1, 1, 1),
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, self.max_active_clusters
        )

        self.kernel(
            tma_atom_Q,
            tma_tensor_Q,
            tma_atom_K,
            tma_tensor_K,
            tma_atom_G,
            tma_tensor_G,
            mBeta,
            tma_atom_Beta,
            tma_tensor_Beta,
            tma_atom_dtBias,
            tma_tensor_dtBias,
            mAlog,
            tma_atom_Qd,
            tma_tensor_Qd,
            tma_atom_Kd,
            tma_tensor_Kd,
            mKr,
            mMqk,
            mMkk,
            mGk,
            mCuSeqlens,
            scale,
            gate_scale,
            sQKG_layout_staged,
            sBeta_layout_staged,
            sDtBias_layout_staged,
            sAlog_layout,
            sQKd_layout_staged,
            tiled_mma,
            tiled_mma_inv,
            tiled_copy_QKG,
            tiled_copy_dtBias,
            tiled_copy_Gk,
            tiled_copy_QKd,
            tiled_copy_Ki,
            tile_sched_params,
            SharedStorage,
        ).launch(
            grid=grid,
            block=(self.threads_per_cta, 1, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_Q: cute.CopyAtom,
        tma_tensor_Q: cute.Tensor,
        tma_atom_K: cute.CopyAtom,
        tma_tensor_K: cute.Tensor,
        tma_atom_G: cute.CopyAtom,
        tma_tensor_G: cute.Tensor,
        mBeta: cute.Tensor,
        tma_atom_Beta: cute.CopyAtom,
        tma_tensor_Beta: cute.Tensor,
        tma_atom_dtBias: cute.CopyAtom,
        tma_tensor_dtBias: cute.Tensor,
        mAlog: cute.Tensor,
        tma_atom_Qd: cute.CopyAtom,
        tma_tensor_Qd: cute.Tensor,
        tma_atom_Kd: cute.CopyAtom,
        tma_tensor_Kd: cute.Tensor,
        mKr: cute.Tensor,
        mMqk: cute.Tensor,
        mMkk: cute.Tensor,
        mGk: cute.Tensor,
        mCuSeqlens: Optional[cute.Tensor],
        scale: cutlass.Float32,
        gate_scale: cutlass.Float32,
        sQKG_layout_staged: cute.Layout,
        sBeta_layout_staged: cute.Layout,
        sDtBias_layout_staged: cute.Layout,
        sAlog_layout: cute.Layout,
        sQKd_layout_staged: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
        tiled_mma_inv: cute.TiledMma,
        tiled_copy_QKG: cute.TiledCopy,
        tiled_copy_dtBias: cute.TiledCopy,
        tiled_copy_Gk: cute.TiledCopy,
        tiled_copy_QKd: cute.TiledCopy,
        tiled_copy_Ki: cute.TiledCopy,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        tidx_in_wg = tidx % self.warpgroup_threads
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(tidx // self.warp_threads)
        widx_in_wg = warp_idx % (self.warpgroup_threads // self.warp_threads)
        warpgroup_idx = cute.arch.make_warp_uniform(tidx // self.warpgroup_threads)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQKG_layout_staged)
        sK = storage.sK.get_tensor(sQKG_layout_staged)
        sG = storage.sG.get_tensor(sQKG_layout_staged)
        sBeta = storage.sBeta.get_tensor(sBeta_layout_staged)
        sDtBias = storage.sDtBias.get_tensor(sDtBias_layout_staged)
        sAlog = storage.sAlog.get_tensor(sAlog_layout)
        sQd = storage.sQd.get_tensor(
            sQKd_layout_staged.outer, swizzle=sQKd_layout_staged.inner
        )
        sKd = storage.sKd.get_tensor(
            sQKd_layout_staged.outer, swizzle=sQKd_layout_staged.inner
        )
        sKi = storage.sKi.get_tensor(
            sQKd_layout_staged.outer, swizzle=sQKd_layout_staged.inner
        )

        tma_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        tma_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.warps_per_warpgroup
        )
        tma_producer, tma_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.load_stages,
            producer_group=tma_producer_group,
            consumer_group=tma_consumer_group,
            barrier_storage=storage.tma_mbar_ptr.data_ptr(),
            tx_count=self.num_tma_load_bytes,
        ).make_participants()

        beta_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, self.warp_threads
        )
        gate_mma_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, self.warpgroup_threads
        )

        beta_producer, beta_consumer = None, None
        if cutlass.const_expr(not self.beta_use_tma):
            beta_producer, beta_consumer = pipeline.PipelineAsync.create(
                num_stages=self.beta_stages,
                producer_group=beta_group,
                consumer_group=gate_mma_group,
                barrier_storage=storage.beta_mbar_ptr.data_ptr(),
            ).make_participants()

        gate_mma_producer, gate_mma_consumer = cutlass.pipeline.PipelineAsync.create(
            num_stages=self.gate_mma_stages,
            producer_group=gate_mma_group,
            consumer_group=gate_mma_group,
            barrier_storage=storage.gate_mma_mbar_ptr.data_ptr(),
        ).make_participants()

        store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=1,
            producer_group=gate_mma_group,
        )

        if warpgroup_idx == self.aux_warpgroup_id:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

            if warp_idx == self.load_warp_id:
                beta_idx = 0

                tile_sched = utils.StaticPersistentTileScheduler.create(
                    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
                )
                work_tile = tile_sched.initial_work_tile_info()

                while work_tile.is_valid_tile:
                    t_idx, h_idx, b_idx = work_tile.tile_idx

                    is_valid_tile = True
                    if cutlass.const_expr(mCuSeqlens is not None):
                        is_valid_tile = t_idx * self.TT < get_seq_len(mCuSeqlens, b_idx)
                    if is_valid_tile:
                        beta_head_tile = h_idx // self.beta_tma_heads
                        gBeta = None
                        if cutlass.const_expr(mCuSeqlens is not None):
                            seq_offset = get_seq_offset(mCuSeqlens, b_idx)

                            mQ_seq = cute.domain_offset(
                                (seq_offset, 0, 0), tma_tensor_Q
                            )
                            mK_seq = cute.domain_offset(
                                (seq_offset, 0, 0), tma_tensor_K
                            )
                            mG_seq = cute.domain_offset(
                                (seq_offset, 0, 0), tma_tensor_G
                            )

                            gQ = cute.local_tile(
                                mQ_seq[None, None, h_idx], (self.TT, self.D), (None, 0)
                            )
                            gK = cute.local_tile(
                                mK_seq[None, None, h_idx], (self.TT, self.D), (None, 0)
                            )
                            gG = cute.local_tile(
                                mG_seq[None, None, h_idx], (self.TT, self.D), (None, 0)
                            )
                            if cutlass.const_expr(self.beta_use_tma):
                                mBeta_seq = cute.domain_offset(
                                    (seq_offset, 0), tma_tensor_Beta
                                )
                                gBeta = cute.local_tile(
                                    mBeta_seq,
                                    (self.TT, self.beta_tma_heads),
                                    (None, beta_head_tile),
                                )
                        else:
                            gQ = cute.local_tile(
                                tma_tensor_Q[None, None, h_idx, b_idx],
                                (self.TT, self.D),
                                (None, 0),
                            )
                            gK = cute.local_tile(
                                tma_tensor_K[None, None, h_idx, b_idx],
                                (self.TT, self.D),
                                (None, 0),
                            )
                            gG = cute.local_tile(
                                tma_tensor_G[None, None, h_idx, b_idx],
                                (self.TT, self.D),
                                (None, 0),
                            )
                            if cutlass.const_expr(self.beta_use_tma):
                                gBeta = cute.local_tile(
                                    tma_tensor_Beta[None, None, b_idx],
                                    (self.TT, self.beta_tma_heads),
                                    (None, beta_head_tile),
                                )
                        gDtBias = tma_tensor_dtBias[None, h_idx]

                        tQsQ, tQgQ = cpasync.tma_partition(
                            tma_atom_Q,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(sQ, 0, 2),
                            cute.group_modes(gQ, 0, 2),
                        )
                        tKsK, tKgK = cpasync.tma_partition(
                            tma_atom_K,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(sK, 0, 2),
                            cute.group_modes(gK, 0, 2),
                        )
                        tGsG, tGgG = cpasync.tma_partition(
                            tma_atom_G,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(sG, 0, 2),
                            cute.group_modes(gG, 0, 2),
                        )
                        if cutlass.const_expr(gBeta is not None):
                            tBeta_sBeta, tBeta_gBeta = cpasync.tma_partition(
                                tma_atom_Beta,
                                0,
                                cute.make_layout(1),
                                cute.group_modes(sBeta, 0, 2),
                                cute.group_modes(gBeta, 0, 2),
                            )
                        tDtBias_sDtBias, tDtBias_gDtBias = cpasync.tma_partition(
                            tma_atom_dtBias,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(sDtBias, 0, 1),
                            cute.group_modes(gDtBias, 0, 1),
                        )

                        tma_empty = tma_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_Q,
                            tQgQ[None, t_idx],
                            tQsQ[None, tma_empty.index],
                            tma_bar_ptr=tma_empty.barrier,
                        )
                        cute.copy(
                            tma_atom_K,
                            tKgK[None, t_idx],
                            tKsK[None, tma_empty.index],
                            tma_bar_ptr=tma_empty.barrier,
                        )
                        cute.copy(
                            tma_atom_G,
                            tGgG[None, t_idx],
                            tGsG[None, tma_empty.index],
                            tma_bar_ptr=tma_empty.barrier,
                        )
                        if cutlass.const_expr(self.beta_use_tma):
                            cute.copy(
                                tma_atom_Beta,
                                tBeta_gBeta[None, t_idx],
                                tBeta_sBeta[None, beta_idx],
                                tma_bar_ptr=tma_empty.barrier,
                            )
                        cute.copy(
                            tma_atom_dtBias,
                            tDtBias_gDtBias,
                            tDtBias_sDtBias[None, tma_empty.index],
                            tma_bar_ptr=tma_empty.barrier,
                        )
                        beta_idx = (beta_idx + 1) % self.beta_stages

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

            elif warp_idx == self.beta_warp_id:
                if cutlass.const_expr(not self.beta_use_tma):
                    tile_sched = utils.StaticPersistentTileScheduler.create(
                        tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
                    )
                    work_tile = tile_sched.initial_work_tile_info()

                    while work_tile.is_valid_tile:
                        t_idx, h_idx, b_idx = work_tile.tile_idx

                        if cutlass.const_expr(mCuSeqlens is not None):
                            seq_len = get_seq_len(mCuSeqlens, b_idx)
                        else:
                            seq_len = cute.size(mBeta, mode=[0])

                        is_valid_tile = True
                        if cutlass.const_expr(mCuSeqlens is not None):
                            is_valid_tile = t_idx * self.TT < seq_len
                        if is_valid_tile:
                            if cutlass.const_expr(mCuSeqlens is not None):
                                seq_offset = get_seq_offset(mCuSeqlens, b_idx)
                                mBeta_seq = cute.domain_offset(
                                    (seq_offset,), mBeta[None, h_idx]
                                )
                                gBeta = cute.local_tile(mBeta_seq, (self.TT,), (t_idx,))
                            else:
                                gBeta = cute.local_tile(
                                    mBeta[None, h_idx, b_idx], (self.TT,), (t_idx,)
                                )

                            valid_tokens = min(self.TT, seq_len - t_idx * self.TT)
                            if valid_tokens == self.TT:
                                beta_empty = beta_producer.acquire_and_advance()
                                for t in cutlass.range_constexpr(
                                    self.TT // self.warp_threads
                                ):
                                    idx = t * self.warp_threads + lane_idx
                                    sBeta[idx, beta_empty.index] = gBeta[idx]
                                beta_empty.commit()
                            else:
                                beta_empty = beta_producer.acquire_and_advance()
                                for t in cutlass.range_constexpr(
                                    self.TT // self.warp_threads
                                ):
                                    idx = t * self.warp_threads + lane_idx
                                    sBeta[idx, beta_empty.index] = (
                                        gBeta[idx]
                                        if idx < valid_tokens
                                        else sBeta.element_type(0.0)
                                    )
                                beta_empty.commit()

                        tile_sched.advance_to_next_work()
                        work_tile = tile_sched.get_current_work()

        elif warpgroup_idx == self.gate_warpgroup_id:
            cute.arch.setmaxregister_increase(self.gate_register_requirement)

            alog_scale = (
                0.5 if cutlass.const_expr(self.gate_lower_bound) else -gate_scale
            )
            for h in cutlass.range_constexpr(
                0,
                self.H // self.warpgroup_threads * self.warpgroup_threads,
                self.warpgroup_threads,
            ):
                sAlog[h + tidx_in_wg] = alog_scale * cute.math.exp2(
                    mAlog[h + tidx_in_wg] * RCP_LN2, fastmath=True
                )
            if cutlass.const_expr(self.H % self.warpgroup_threads != 0):
                if tidx_in_wg < self.H % self.warpgroup_threads:
                    h_idx = (
                        self.H // self.warpgroup_threads * self.warpgroup_threads
                        + tidx_in_wg
                    )
                    sAlog[h_idx] = alog_scale * cute.math.exp2(
                        mAlog[h_idx] * RCP_LN2, fastmath=True
                    )

            gate_scale = gate_scale * 0.5 * RCP_LN2

            cutlass.pipeline.arrive_and_wait(
                barrier_id=1, num_threads=self.warpgroup_threads
            )

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            while work_tile.is_valid_tile:
                t_idx, h_idx, b_idx = work_tile.tile_idx

                seq_len = 0
                seq_offset_padded = None
                chunk_offset = None
                if cutlass.const_expr(mCuSeqlens is not None):
                    seq_len = get_seq_len(mCuSeqlens, b_idx)
                    seq_offset_padded = get_seq_offset(
                        mCuSeqlens, b_idx, padding=self.TT
                    )
                    chunk_offset = get_seq_offset(
                        mCuSeqlens, b_idx, padding=self.TT, index_unit=self.BT
                    )
                else:
                    seq_len = cute.size(tma_tensor_Q, mode=[0])

                is_valid_tile = True
                if cutlass.const_expr(mCuSeqlens is not None):
                    is_valid_tile = t_idx * self.TT < seq_len
                if is_valid_tile:
                    valid_tokens = min(self.TT, seq_len - t_idx * self.TT)
                    if valid_tokens == self.TT:
                        tma_consumer, gate_mma_producer = self.compute_gate_tile(
                            mGk,
                            mKr,
                            sQ,
                            sK,
                            sG,
                            sDtBias,
                            sAlog,
                            sQd,
                            sKd,
                            sKi,
                            tma_consumer,
                            gate_mma_producer,
                            gate_scale,
                            scale,
                            t_idx,
                            h_idx,
                            b_idx,
                            tiled_copy_QKG,
                            tiled_copy_dtBias,
                            tiled_copy_Gk,
                            lane_idx,
                            widx_in_wg,
                            seq_offset_padded,
                            chunk_offset,
                        )
                    else:
                        tma_consumer, gate_mma_producer = self.compute_gate_tile(
                            mGk,
                            mKr,
                            sQ,
                            sK,
                            sG,
                            sDtBias,
                            sAlog,
                            sQd,
                            sKd,
                            sKi,
                            tma_consumer,
                            gate_mma_producer,
                            gate_scale,
                            scale,
                            t_idx,
                            h_idx,
                            b_idx,
                            tiled_copy_QKG,
                            tiled_copy_dtBias,
                            tiled_copy_Gk,
                            lane_idx,
                            widx_in_wg,
                            seq_offset_padded,
                            chunk_offset,
                            valid_tokens,
                        )

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        else:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)

            thr_mma = tiled_mma.get_slice(lane_idx)
            thr_mma_inv = tiled_mma_inv.get_slice(lane_idx)
            tAcA = thr_mma.partition_C(cute.make_identity_tensor((self.BT, self.BT)))
            a_shape_L = thr_mma.partition_shape_A((self.BT, self.BT))
            b_shape_L = thr_mma.partition_shape_B((self.BT, self.BT))
            c_shape_L = thr_mma.partition_shape_C((self.BT, self.BT))
            tCrI = thr_mma.make_fragment_C(c_shape_L)
            for e in cutlass.range_constexpr(cute.size(tAcA)):
                m, n = tAcA[e]
                tCrI[e] = cutlass.Float32(1.0) if m == n else cutlass.Float32(0.0)

            beta_idx = 0

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            while work_tile.is_valid_tile:
                t_idx, h_idx, b_idx = work_tile.tile_idx

                is_valid_tile = True
                if cutlass.const_expr(mCuSeqlens is not None):
                    seq_len = get_seq_len(mCuSeqlens, b_idx)
                    is_valid_tile = t_idx * self.TT < seq_len
                if is_valid_tile:
                    if cutlass.const_expr(mCuSeqlens is not None):
                        seq_offset_padded = get_seq_offset(
                            mCuSeqlens, b_idx, padding=self.TT
                        )
                        mQd_seq = cute.domain_offset(
                            (seq_offset_padded, 0, 0), tma_tensor_Qd
                        )
                        mKd_seq = cute.domain_offset(
                            (seq_offset_padded, 0, 0), tma_tensor_Kd
                        )
                        gQd = cute.local_tile(
                            mQd_seq[None, None, h_idx], (self.TT, self.D), (None, 0)
                        )
                        gKd = cute.local_tile(
                            mKd_seq[None, None, h_idx], (self.TT, self.D), (None, 0)
                        )
                    else:
                        gQd = cute.local_tile(
                            tma_tensor_Qd[None, None, h_idx, b_idx],
                            (self.TT, self.D),
                            (None, 0),
                        )
                        gKd = cute.local_tile(
                            tma_tensor_Kd[None, None, h_idx, b_idx],
                            (self.TT, self.D),
                            (None, 0),
                        )

                    tQsQd, tQgQd = cpasync.tma_partition(
                        tma_atom_Qd,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sQd, 0, 2),
                        cute.group_modes(gQd, 0, 2),
                    )
                    tKsKd, tKgKd = cpasync.tma_partition(
                        tma_atom_Kd,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sKd, 0, 2),
                        cute.group_modes(gKd, 0, 2),
                    )

                    beta_full = None
                    if cutlass.const_expr(not self.beta_use_tma):
                        beta_full = beta_consumer.wait_and_advance()
                        beta_idx = beta_full.index
                    mma_full = gate_mma_consumer.wait_and_advance()

                    if widx_in_wg == 0:
                        cute.copy(
                            tma_atom_Qd,
                            tQsQd[None, mma_full.index],
                            tQgQd[None, t_idx],
                        )
                        cute.copy(
                            tma_atom_Kd,
                            tKsKd[None, mma_full.index],
                            tKgKd[None, t_idx],
                        )
                        store_pipeline.producer_commit()

                    if cutlass.const_expr(mCuSeqlens is not None):
                        chunk_offset = get_seq_offset(
                            mCuSeqlens, b_idx, padding=self.TT, index_unit=self.BT
                        )
                        mMqk_seq = cute.domain_offset((0, 0, chunk_offset, 0), mMqk)
                        mMkk_seq = cute.domain_offset((0, 0, chunk_offset, 0), mMkk)
                        mMqk_seq = mMqk_seq[None, None, None, h_idx]
                        mMkk_seq = mMkk_seq[None, None, None, h_idx]
                    else:
                        mMqk_seq = mMqk[None, None, None, h_idx, b_idx]
                        mMkk_seq = mMkk[None, None, None, h_idx, b_idx]
                    gMqk = cute.local_tile(
                        mMqk_seq, (self.BT, self.BT, self.CT), (0, 0, t_idx)
                    )
                    gMkk = cute.local_tile(
                        mMkk_seq, (self.BT, self.BT, self.CT), (0, 0, t_idx)
                    )
                    gMqk = gMqk[None, None, widx_in_wg]
                    gMkk = gMkk[None, None, widx_in_wg]

                    sQd_tile = cute.local_tile(
                        sQd[None, None, mma_full.index], (self.BT, self.D), (None, 0)
                    )
                    sKd_tile = cute.local_tile(
                        sKd[None, None, mma_full.index], (self.BT, self.D), (None, 0)
                    )
                    sKi_tile = cute.local_tile(
                        sKi[None, None, mma_full.index], (self.BT, self.D), (None, 0)
                    )
                    if cutlass.const_expr(self.beta_use_tma):
                        beta_head = h_idx % self.beta_tma_heads
                        sBeta_tile = cute.local_tile(
                            sBeta[None, beta_head, beta_idx], (self.BT,), (None,)
                        )
                    else:
                        sBeta_tile = cute.local_tile(
                            sBeta[None, beta_idx], (self.BT,), (None,)
                        )
                    sQd_tile = sQd_tile[None, None, widx_in_wg]
                    sKd_tile = sKd_tile[None, None, widx_in_wg]
                    sKi_tile = sKi_tile[None, None, widx_in_wg]
                    sBeta_tile = sBeta_tile[None, widx_in_wg]

                    beta_stride = (
                        self.beta_tma_heads
                        if cutlass.const_expr(self.beta_use_tma)
                        else 1
                    )
                    sBeta_pre = cute.make_tensor(
                        sBeta_tile.iterator,
                        cute.make_layout((self.BT, self.BT), stride=(beta_stride, 0)),
                    )
                    sBeta_post = cute.make_tensor(
                        sBeta_tile.iterator,
                        cute.make_layout((self.BT, self.BT), stride=(0, beta_stride)),
                    )

                    tArQd = thr_mma.make_fragment_A(thr_mma.partition_A(sQd_tile))
                    tArKd = thr_mma.make_fragment_A(thr_mma.partition_A(sKd_tile))
                    tArKi = thr_mma.make_fragment_B(thr_mma.partition_B(sKi_tile))
                    tAgMqk = thr_mma.partition_C(gMqk)

                    thr_copy_QKd = tiled_copy_QKd.get_slice(lane_idx)
                    tAsQd_copy_view = thr_copy_QKd.partition_S(sQd_tile)
                    tArQd_copy_view = thr_copy_QKd.retile(tArQd)
                    tAsKd_copy_view = thr_copy_QKd.partition_S(sKd_tile)
                    tArKd_copy_view = thr_copy_QKd.retile(tArKd)

                    thr_copy_Ki = tiled_copy_Ki.get_slice(lane_idx)
                    tAsKi_copy_view = thr_copy_Ki.partition_S(sKi_tile)
                    tArKi_copy_view = thr_copy_Ki.retile(tArKi)

                    tAsBeta_pre = thr_mma.partition_C(sBeta_pre)
                    tAsBeta_pre_mn = cute.make_tensor(
                        tAsBeta_pre.iterator,
                        layout_acc_mn(tiled_mma, tAsBeta_pre.layout),
                    )
                    tAsBeta_pre_mn = cute.flatten(tAsBeta_pre_mn[None, 0])
                    tArBeta_pre_mn = cute.make_rmem_tensor_like(tAsBeta_pre_mn)
                    tArBeta_pre_mn_f32 = cute.make_rmem_tensor_like(
                        tAsBeta_pre_mn, cutlass.Float32
                    )

                    tMsBeta_post = thr_mma.partition_C(sBeta_post)
                    tMsBeta_post_mn = cute.make_tensor(
                        tMsBeta_post.iterator,
                        layout_acc_mn(tiled_mma, tMsBeta_post.layout),
                    )
                    tMsBeta_post_mn = tMsBeta_post_mn[0, None]
                    tMrBeta_post_mn = cute.make_rmem_tensor_like(tMsBeta_post_mn)
                    tMrBeta_post_mn_f32 = cute.make_rmem_tensor_like(
                        tMsBeta_post_mn, cutlass.Float32
                    )

                    tMgMkk = thr_mma.partition_C(gMkk)

                    cute.copy(tiled_copy_QKd, tAsQd_copy_view, tArQd_copy_view)
                    cute.copy(tiled_copy_QKd, tAsKd_copy_view, tArKd_copy_view)
                    cute.copy(tiled_copy_Ki, tAsKi_copy_view, tArKi_copy_view)
                    cute.autovec_copy(tAsBeta_pre_mn, tArBeta_pre_mn)
                    cute.autovec_copy(tMsBeta_post_mn, tMrBeta_post_mn)

                    tArAkk = cute.make_rmem_tensor_like(tAcA, cutlass.Float32)
                    tArAkk.fill(0.0)
                    cute.gemm(tiled_mma, tArAkk, tArKd, tArKi, tArAkk)
                    for e in cutlass.range_constexpr(cute.size(tAcA)):
                        m, n = tAcA[e]
                        tArAkk[e] = tArAkk[e] if m > n else cutlass.Float32(0.0)
                    self.sigmoid(tArBeta_pre_mn, tArBeta_pre_mn_f32)
                    tArAkk_mn = cute.make_tensor(
                        tArAkk.iterator, layout_acc_mn(tiled_mma, tArAkk.layout)
                    )
                    for m in cutlass.range_constexpr(cute.size(tArAkk_mn, mode=[0])):
                        for n in cutlass.range_constexpr(
                            cute.size(tArAkk_mn, mode=[1]) // 2
                        ):
                            tArAkk_mn[m, 2 * n], tArAkk_mn[m, 2 * n + 1] = (
                                cute.arch.mul_packed_f32x2(
                                    (tArAkk_mn[m, 2 * n], tArAkk_mn[m, 2 * n + 1]),
                                    (-tArBeta_pre_mn_f32[m], -tArBeta_pre_mn_f32[m]),
                                )
                            )

                    tArAqk = cute.make_rmem_tensor_like(tAcA, cutlass.Float32)
                    tArAqk.fill(0.0)
                    cute.gemm(tiled_mma, tArAqk, tArQd, tArKi, tArAqk)
                    for e in cutlass.range_constexpr(cute.size(tAcA)):
                        m, n = tAcA[e]
                        tArAqk[e] = tArAqk[e] if m >= n else cutlass.Float32(0.0)
                    tArAqk_f16 = cute.make_rmem_tensor_like(tArAqk, gMqk.element_type)
                    tArAqk_f16.store(tArAqk.load().to(tArAqk_f16.element_type))
                    cute.autovec_copy(tArAqk_f16, tAgMqk)

                    # (I - L)^-1 for the strictly lower L = tArAkk by the hierarchical
                    # block inverse of the BT16 engine (super_mma_stage_blockwise_inverse),
                    # on FP16 operands with FP32 accumulation:
                    #   D = blockdiag(L11, L22), nilpotent 8x8 blocks:
                    #       Binv = (I + D)(I + D^2)(I + D^4)       (exact)
                    #   (I - L)^-1 = Binv + Binv @ L21 @ Binv      (exact)
                    # The 16x16 Neumann chain (I + L)(I + L^2)(I + L^4)(I + L^8) it
                    # replaces materializes L^2..L^8, whose entries grow combinatorially
                    # for near-identical keys at high beta; 16-bit rounding then destroys
                    # the alternating-sign cancellation of the bounded true inverse, the
                    # chunk's state map gains eigenvalues far above 1 and a run of
                    # repeated tokens drives the state to inf/NaN. D^2 and D^4 stay O(10).
                    tCrA21 = cute.make_rmem_tensor_like(tArAkk, cutlass.Float32)
                    tCrD = cute.make_rmem_tensor_like(tArAkk, cutlass.Float32)
                    for e in cutlass.range_constexpr(cute.size(tAcA)):
                        m, n = tAcA[e]
                        a21 = tArAkk[e] if m >= self.BT // 2 else cutlass.Float32(0.0)
                        tCrA21[e] = a21 if n < self.BT // 2 else cutlass.Float32(0.0)
                        # The upper-right block of the strictly lower L is zero.
                        tCrD[e] = tArAkk[e] - tCrA21[e]

                    tCrP = thr_mma_inv.make_fragment_C(c_shape_L)
                    tArD = thr_mma_inv.make_fragment_A(a_shape_L)
                    tBrD = thr_mma_inv.make_fragment_B(_pack_shape(b_shape_L))
                    tArD.store(tCrD.load().to(tArD.element_type))
                    self.transpose(tArD, tBrD)
                    tCrP.fill(0.0)
                    cute.gemm(tiled_mma_inv, tCrP, tArD, tBrD, tCrP)  # D^2

                    tArD2 = thr_mma_inv.make_fragment_A(a_shape_L)
                    tBrD2 = thr_mma_inv.make_fragment_B(_pack_shape(b_shape_L))
                    tArD2.store(tCrP.load().to(tArD2.element_type))
                    self.transpose(tArD2, tBrD2)
                    tArID2 = thr_mma_inv.make_fragment_A(a_shape_L)
                    tArID2.store(
                        self.add_identity(tCrP, tCrI).load().to(tArID2.element_type)
                    )
                    tBrID2 = thr_mma_inv.make_fragment_B(_pack_shape(b_shape_L))
                    self.transpose(tArID2, tBrID2)
                    tCrP.fill(0.0)
                    cute.gemm(tiled_mma_inv, tCrP, tArD2, tBrD2, tCrP)  # D^4

                    tArID4 = thr_mma_inv.make_fragment_A(a_shape_L)
                    tArID4.store(
                        self.add_identity(tCrP, tCrI).load().to(tArID4.element_type)
                    )
                    tBrID4 = thr_mma_inv.make_fragment_B(_pack_shape(b_shape_L))
                    self.transpose(tArID4, tBrID4)
                    tArID = thr_mma_inv.make_fragment_A(a_shape_L)
                    tArID.store(
                        self.add_identity(tCrD, tCrI).load().to(tArID.element_type)
                    )
                    tCrP.fill(0.0)
                    cute.gemm(tiled_mma_inv, tCrP, tArID, tBrID2, tCrP)  # (I+D)(I+D^2)

                    tArB01 = thr_mma_inv.make_fragment_A(a_shape_L)
                    tArB01.store(tCrP.load().to(tArB01.element_type))
                    tCrBinv = thr_mma_inv.make_fragment_C(c_shape_L)
                    tCrBinv.fill(0.0)
                    cute.gemm(tiled_mma_inv, tCrBinv, tArB01, tBrID4, tCrBinv)  # Binv

                    tArBinv = thr_mma_inv.make_fragment_A(a_shape_L)
                    tBrBinv = thr_mma_inv.make_fragment_B(_pack_shape(b_shape_L))
                    tArBinv.store(tCrBinv.load().to(tArBinv.element_type))
                    self.transpose(tArBinv, tBrBinv)
                    tArA21 = thr_mma_inv.make_fragment_A(a_shape_L)
                    tBrA21 = thr_mma_inv.make_fragment_B(_pack_shape(b_shape_L))
                    tArA21.store(tCrA21.load().to(tArA21.element_type))
                    self.transpose(tArA21, tBrA21)
                    tCrP.fill(0.0)
                    cute.gemm(tiled_mma_inv, tCrP, tArBinv, tBrA21, tCrP)  # Binv L21

                    tArT1 = thr_mma_inv.make_fragment_A(a_shape_L)
                    tArT1.store(tCrP.load().to(tArT1.element_type))
                    tCrL = thr_mma_inv.make_fragment_C(c_shape_L)
                    tCrL.fill(0.0)
                    cute.gemm(tiled_mma_inv, tCrL, tArT1, tBrBinv, tCrL)  # Binv L21 Binv
                    # Binv is block diagonal and Binv L21 Binv lives in the lower-left
                    # block only (exact zeros elsewhere): their sum is the inverse.
                    for i in cutlass.range_constexpr(cute.size(tCrL) // 2):
                        tCrL[2 * i], tCrL[2 * i + 1] = cute.arch.add_packed_f32x2(
                            (tCrL[2 * i], tCrL[2 * i + 1]),
                            (tCrBinv[2 * i], tCrBinv[2 * i + 1]),
                        )
                    self.sigmoid(tMrBeta_post_mn, tMrBeta_post_mn_f32)
                    tCrL_mn = cute.make_tensor(
                        tCrL.iterator, layout_acc_mn(tiled_mma, tCrL.layout)
                    )
                    for m in cutlass.range_constexpr(cute.size(tCrL_mn, mode=[0])):
                        for n in cutlass.range_constexpr(
                            cute.size(tCrL_mn, mode=[1]) // 2
                        ):
                            tCrL_mn[m, 2 * n], tCrL_mn[m, 2 * n + 1] = (
                                cute.arch.mul_packed_f32x2(
                                    (tCrL_mn[m, 2 * n], tCrL_mn[m, 2 * n + 1]),
                                    (
                                        tMrBeta_post_mn_f32[2 * n],
                                        tMrBeta_post_mn_f32[2 * n + 1],
                                    ),
                                )
                            )
                    tMrMkk_f16 = cute.make_rmem_tensor_like(tCrL, gMkk.element_type)
                    tMrMkk_f16.store(tCrL.load().to(tMrMkk_f16.element_type))
                    cute.autovec_copy(tMrMkk_f16, tMgMkk)

                    if cutlass.const_expr(self.beta_use_tma):
                        beta_idx = (beta_idx + 1) % self.beta_stages
                    else:
                        beta_full.release()
                    store_pipeline.producer_acquire()
                    mma_full.release()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

    @cute.jit
    def compute_gate_tile(
        self,
        mGk: cute.Tensor,
        mKr: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sG: cute.Tensor,
        sDtBias: cute.Tensor,
        sAlog: cute.Tensor,
        sQd: cute.Tensor,
        sKd: cute.Tensor,
        sKi: cute.Tensor,
        tma_consumer: pipeline.PipelineConsumer,
        gate_mma_producer: pipeline.PipelineProducer,
        gate_scale: cutlass.Float32,
        scale: cutlass.Float32,
        t_idx: cutlass.Int32,
        h_idx: cutlass.Int32,
        b_idx: cutlass.Int32,
        tiled_copy_QKG: cute.TiledCopy,
        tiled_copy_dtBias: cute.TiledCopy,
        tiled_copy_Gk: cute.TiledCopy,
        lane_idx: cutlass.Int32,
        widx_in_wg: cutlass.Int32,
        seq_offset_padded: Optional[cutlass.Int32] = None,
        chunk_offset: Optional[cutlass.Int32] = None,
        valid_tokens: Optional[cutlass.Int32] = None,
    ):
        alog = sAlog[h_idx]

        tma_full = tma_consumer.wait_and_advance()
        mma_empty = gate_mma_producer.acquire_and_advance()

        if cutlass.const_expr(seq_offset_padded is not None):
            mKr_seq = cute.domain_offset((seq_offset_padded, 0, 0), mKr)
            mGk_seq = cute.domain_offset((0, chunk_offset, 0), mGk)
            gGk = cute.local_tile(
                mGk_seq[None, None, h_idx], (self.D, self.CT), (0, t_idx)
            )
            gKr = cute.local_tile(
                mKr_seq[None, None, h_idx], (self.TT, self.D), (t_idx, 0)
            )
        else:
            gGk = cute.local_tile(
                mGk[None, None, h_idx, b_idx], (self.D, self.CT), (0, t_idx)
            )
            gKr = cute.local_tile(
                mKr[None, None, h_idx, b_idx], (self.TT, self.D), (t_idx, 0)
            )
        gGk_tile = gGk[None, widx_in_wg]
        gKr_tile = cute.local_tile(gKr, (self.BT, self.D), (widx_in_wg, 0))

        sQ_tile = cute.local_tile(
            sQ[None, None, tma_full.index], (self.BT, self.D), (widx_in_wg, 0)
        )
        sK_tile = cute.local_tile(
            sK[None, None, tma_full.index], (self.BT, self.D), (widx_in_wg, 0)
        )
        sG_tile = cute.local_tile(
            sG[None, None, tma_full.index], (self.BT, self.D), (widx_in_wg, 0)
        )
        sDtBias_tile = sDtBias[None, tma_full.index]
        sQd_tile = cute.local_tile(
            sQd[None, None, mma_empty.index], (self.BT, self.D), (widx_in_wg, 0)
        )
        sKd_tile = cute.local_tile(
            sKd[None, None, mma_empty.index], (self.BT, self.D), (widx_in_wg, 0)
        )
        sKi_tile = cute.local_tile(
            sKi[None, None, mma_empty.index], (self.BT, self.D), (widx_in_wg, 0)
        )

        thr_copy_QKG = tiled_copy_QKG.get_slice(lane_idx)
        tQsQ = thr_copy_QKG.partition_S(sQ_tile)
        tQrQ = cute.make_rmem_tensor_like(tQsQ)
        tQrQ_f32 = cute.make_rmem_tensor_like(tQrQ, cutlass.Float32)
        tKsK = thr_copy_QKG.partition_S(sK_tile)
        tKrK = cute.make_rmem_tensor_like(tKsK)
        tKrK_f32 = cute.make_rmem_tensor_like(tKrK, cutlass.Float32)
        tGsG = thr_copy_QKG.partition_S(sG_tile)
        tGrG = cute.make_rmem_tensor_like(tGsG)
        tGrG_f32 = cute.make_rmem_tensor_like(tGrG, cutlass.Float32)
        tGrGe_f32 = cute.make_rmem_tensor_like(tGrG, cutlass.Float32)
        tGrGi_f32 = cute.make_rmem_tensor_like(tGrG, cutlass.Float32)
        tQsQd = thr_copy_QKG.partition_D(sQd_tile)
        tQrQd = cute.make_rmem_tensor_like(tQsQd)
        tQrQd_f32 = cute.make_rmem_tensor_like(tQrQd, cutlass.Float32)
        tKsKd = thr_copy_QKG.partition_D(sKd_tile)
        tKrKd = cute.make_rmem_tensor_like(tKsKd)
        tKrKd_f32 = cute.make_rmem_tensor_like(tKrKd, cutlass.Float32)
        tKsKi = thr_copy_QKG.partition_D(sKi_tile)
        tKrKi = cute.make_rmem_tensor_like(tKsKi)
        tKrKi_f32 = cute.make_rmem_tensor_like(tKrKi, cutlass.Float32)
        tKgKr = thr_copy_QKG.partition_D(gKr_tile)
        tKrKr = cute.make_rmem_tensor_like(tKgKr)
        tKrKr_f32 = cute.make_rmem_tensor_like(tKrKr, cutlass.Float32)

        thr_copy_dtBias = tiled_copy_dtBias.get_slice(lane_idx)
        tDtBias_sDtBias = thr_copy_dtBias.partition_S(sDtBias_tile)
        tDtBias_rDtBias = cute.make_rmem_tensor_like(tDtBias_sDtBias)

        thr_copy_Gk = tiled_copy_Gk.get_slice(lane_idx)
        tGkgGk = thr_copy_Gk.partition_S(gGk_tile)
        tGkrGk = cute.make_rmem_tensor_like(tGkgGk)

        nr = cute.size(tQrQ, mode=[0, 1])
        nc = cute.size(self.get_row(tQrQ, 0), mode=[1])
        t = widx_in_wg * self.BT
        if lane_idx >= self.warp_threads // 2:
            t += self.BT // 2

        cum = cute.make_rmem_tensor_like(self.get_row(tGrG, 0), cutlass.Float32)

        precompute_qk_l2_norm = 2
        if cutlass.const_expr(seq_offset_padded is not None):
            if cutlass.const_expr(self.gate_lower_bound):
                precompute_qk_l2_norm = 4
            else:
                precompute_qk_l2_norm = 5

        cute.copy(tiled_copy_dtBias, tDtBias_sDtBias, tDtBias_rDtBias)
        cute.copy(
            tiled_copy_QKG, tGsG[((None, 0), None, None)], tGrG[((None, 0), None, None)]
        )
        if cutlass.const_expr(precompute_qk_l2_norm == nr):
            cute.copy(
                tiled_copy_QKG,
                tQsQ[((None, 0), None, None)],
                tQrQ[((None, 0), None, None)],
            )
            cute.copy(
                tiled_copy_QKG,
                tKsK[((None, 0), None, None)],
                tKrK[((None, 0), None, None)],
            )

        for r in cutlass.range_constexpr(nr):
            if cutlass.const_expr(r < nr - 1):
                cute.copy(
                    tiled_copy_QKG,
                    tGsG[((None, r + 1), None, None)],
                    tGrG[((None, r + 1), None, None)],
                )

            tGrG_row = self.get_row(tGrG, r)
            tGrG_f32_row = self.get_row(tGrG_f32, r)
            for c in cutlass.range_constexpr(nc):
                tGrG_f32_row[0, c], tGrG_f32_row[1, c] = cute.arch.add_packed_f32x2(
                    (
                        tGrG_row[0, c].to(cutlass.Float32),
                        tGrG_row[1, c].to(cutlass.Float32),
                    ),
                    (tDtBias_rDtBias[2 * c], tDtBias_rDtBias[2 * c + 1]),
                )
                if cutlass.const_expr(self.gate_lower_bound):
                    tGrG_f32_row[0, c], tGrG_f32_row[1, c] = cute.arch.mul_packed_f32x2(
                        (tGrG_f32_row[0, c], tGrG_f32_row[1, c]),
                        (alog, alog),
                    )
                    tGrG_f32_row[0, c] = tanh(tGrG_f32_row[0, c])
                    tGrG_f32_row[1, c] = tanh(tGrG_f32_row[1, c])
                    tGrG_f32_row[0, c], tGrG_f32_row[1, c] = cute.arch.fma_packed_f32x2(
                        (tGrG_f32_row[0, c], tGrG_f32_row[1, c]),
                        (gate_scale, gate_scale),
                        (gate_scale, gate_scale),
                    )
                else:
                    tGrG_f32_row[0, c], tGrG_f32_row[1, c] = cute.arch.mul_packed_f32x2(
                        (tGrG_f32_row[0, c], tGrG_f32_row[1, c]),
                        (RCP_LN2, RCP_LN2),
                    )
                    tGrG_f32_row[0, c] = cute.math.exp2(
                        tGrG_f32_row[0, c], fastmath=True
                    )
                    tGrG_f32_row[1, c] = cute.math.exp2(
                        tGrG_f32_row[1, c], fastmath=True
                    )
                    tGrG_f32_row[0, c], tGrG_f32_row[1, c] = cute.arch.add_packed_f32x2(
                        (tGrG_f32_row[0, c], tGrG_f32_row[1, c]),
                        (cutlass.Float32(1.0), cutlass.Float32(1.0)),
                    )
                    tGrG_f32_row[0, c] = cute.math.log2(
                        tGrG_f32_row[0, c], fastmath=True
                    )
                    tGrG_f32_row[1, c] = cute.math.log2(
                        tGrG_f32_row[1, c], fastmath=True
                    )
                    tGrG_f32_row[0, c], tGrG_f32_row[1, c] = cute.arch.mul_packed_f32x2(
                        (tGrG_f32_row[0, c], tGrG_f32_row[1, c]),
                        (alog, alog),
                    )
                if cutlass.const_expr(valid_tokens is not None):
                    tGrG_f32_row[0, c] = (
                        tGrG_f32_row[0, c]
                        if t + r < valid_tokens
                        else cutlass.Float32(0.0)
                    )
                    tGrG_f32_row[1, c] = (
                        tGrG_f32_row[1, c]
                        if t + r < valid_tokens
                        else cutlass.Float32(0.0)
                    )
                if cutlass.const_expr(r == 0):
                    cum[0, c] = tGrG_f32_row[0, c]
                    cum[1, c] = tGrG_f32_row[1, c]
                else:
                    cum[0, c], cum[1, c] = cute.arch.add_packed_f32x2(
                        (cum[0, c], cum[1, c]),
                        (tGrG_f32_row[0, c], tGrG_f32_row[1, c]),
                    )
                tGrG_f32_row[0, c] = cum[0, c]
                tGrG_f32_row[1, c] = cum[1, c]

            qk_r = r - nr + precompute_qk_l2_norm
            if cutlass.const_expr(qk_r >= -1 and r < nr - 1):
                cute.copy(
                    tiled_copy_QKG,
                    tQsQ[((None, qk_r + 1), None, None)],
                    tQrQ[((None, qk_r + 1), None, None)],
                )
                cute.copy(
                    tiled_copy_QKG,
                    tKsK[((None, qk_r + 1), None, None)],
                    tKrK[((None, qk_r + 1), None, None)],
                )
            if cutlass.const_expr(qk_r >= 0):
                self.qk_l2_norm_row(
                    tQrQ,
                    tQrQ_f32,
                    tKrK,
                    tKrK_f32,
                    qk_r,
                    t,
                    valid_tokens,
                    seq_offset_padded,
                )

        if cutlass.const_expr(precompute_qk_l2_norm < nr):
            cute.copy(
                tiled_copy_QKG,
                tQsQ[((None, precompute_qk_l2_norm), None, None)],
                tQrQ[((None, precompute_qk_l2_norm), None, None)],
            )
            cute.copy(
                tiled_copy_QKG,
                tKsK[((None, precompute_qk_l2_norm), None, None)],
                tKrK[((None, precompute_qk_l2_norm), None, None)],
            )

        for c in cutlass.range_constexpr(nc):
            cum_prefix0 = cute.arch.shuffle_sync_bfly(cum[0, c], self.warp_threads // 2)
            cum_prefix1 = cute.arch.shuffle_sync_bfly(cum[1, c], self.warp_threads // 2)
            if lane_idx < self.warp_threads // 2:
                cum_prefix0 = cutlass.Float32(0.0)
                cum_prefix1 = cutlass.Float32(0.0)
            for r in cutlass.range_constexpr(nr):
                tGrG_f32_row = self.get_row(tGrG_f32, r)
                tGrG_f32_row[0, c], tGrG_f32_row[1, c] = cute.arch.add_packed_f32x2(
                    (tGrG_f32_row[0, c], tGrG_f32_row[1, c]),
                    (cum_prefix0, cum_prefix1),
                )
            if lane_idx >= self.warp_threads // 2:
                tGrG_f32_row = self.get_row(tGrG_f32, nr - 1)
                tGkrGk[2 * c] = cute.math.exp2(tGrG_f32_row[0, c], fastmath=True)
                tGkrGk[2 * c + 1] = cute.math.exp2(tGrG_f32_row[1, c], fastmath=True)
            Gk_shfl0 = cute.arch.shuffle_sync_bfly(tGkrGk[2 * c], 16)
            Gk_shfl1 = cute.arch.shuffle_sync_bfly(tGkrGk[2 * c + 1], 16)
            if lane_idx < self.warp_threads // 2:
                tGkrGk[2 * c] = Gk_shfl0
                tGkrGk[2 * c + 1] = Gk_shfl1
        if lane_idx >= self.warp_threads // 2:
            cute.copy(tiled_copy_Gk, tGkrGk, tGkgGk)

        for r in cutlass.range_constexpr(nr):
            qk_r = r + precompute_qk_l2_norm
            if cutlass.const_expr(qk_r + 1 < nr):
                cute.copy(
                    tiled_copy_QKG,
                    tQsQ[((None, qk_r + 1), None, None)],
                    tQrQ[((None, qk_r + 1), None, None)],
                )
                cute.copy(
                    tiled_copy_QKG,
                    tKsK[((None, qk_r + 1), None, None)],
                    tKrK[((None, qk_r + 1), None, None)],
                )
            if cutlass.const_expr(qk_r < nr):
                self.qk_l2_norm_row(
                    tQrQ,
                    tQrQ_f32,
                    tKrK,
                    tKrK_f32,
                    qk_r,
                    t,
                    valid_tokens,
                    seq_offset_padded,
                )

            tQrQ_f32_row = self.get_row(tQrQ_f32, r)
            tKrK_f32_row = self.get_row(tKrK_f32, r)
            tGrG_f32_row = self.get_row(tGrG_f32, r)
            tGrGe_f32_row = self.get_row(tGrGe_f32, r)
            tGrGi_f32_row = self.get_row(tGrGi_f32, r)
            tQrQd_row = self.get_row(tQrQd, r)
            tQrQd_f32_row = self.get_row(tQrQd_f32, r)
            tKrKd_row = self.get_row(tKrKd, r)
            tKrKd_f32_row = self.get_row(tKrKd_f32, r)
            tKrKi_row = self.get_row(tKrKi, r)
            tKrKi_f32_row = self.get_row(tKrKi_f32, r)
            tKrKr_row = self.get_row(tKrKr, r)
            tKrKr_f32_row = self.get_row(tKrKr_f32, r)

            for c in cutlass.range_constexpr(nc):
                tGrGe_f32_row[0, c] = cute.math.exp2(tGrG_f32_row[0, c], fastmath=True)
                tGrGe_f32_row[1, c] = cute.math.exp2(tGrG_f32_row[1, c], fastmath=True)

                tQrQd_f32_row[0, c], tQrQd_f32_row[1, c] = cute.arch.mul_packed_f32x2(
                    (tGrGe_f32_row[0, c], tGrGe_f32_row[1, c]),
                    (tQrQ_f32_row[0, c], tQrQ_f32_row[1, c]),
                )
                tQrQd_f32_row[0, c], tQrQd_f32_row[1, c] = cute.arch.mul_packed_f32x2(
                    (tQrQd_f32_row[0, c], tQrQd_f32_row[1, c]),
                    (scale, scale),
                )
                tQrQd_row[None, c].store(
                    tQrQd_f32_row[None, c].load().to(tQrQd_row.element_type)
                )

                tKrKd_f32_row[0, c], tKrKd_f32_row[1, c] = cute.arch.mul_packed_f32x2(
                    (tKrK_f32_row[0, c], tKrK_f32_row[1, c]),
                    (tGrGe_f32_row[0, c], tGrGe_f32_row[1, c]),
                )
                tKrKd_row[None, c].store(
                    tKrKd_f32_row[None, c].load().to(tKrKd_row.element_type)
                )

                tGrGi_f32_row[0, c] = cute.math.exp2(-tGrG_f32_row[0, c], fastmath=True)
                tGrGi_f32_row[1, c] = cute.math.exp2(-tGrG_f32_row[1, c], fastmath=True)

                tKrKi_f32_row[0, c], tKrKi_f32_row[1, c] = cute.arch.mul_packed_f32x2(
                    (tKrK_f32_row[0, c], tKrK_f32_row[1, c]),
                    (tGrGi_f32_row[0, c], tGrGi_f32_row[1, c]),
                )
                tKrKi_row[None, c].store(
                    tKrKi_f32_row[None, c].load().to(tKrKi_row.element_type)
                )

                tKrKr_f32_row[0, c], tKrKr_f32_row[1, c] = cute.arch.mul_packed_f32x2(
                    (tKrKi_f32_row[0, c], tKrKi_f32_row[1, c]),
                    (tGkrGk[2 * c], tGkrGk[2 * c + 1]),
                )
                tKrKr_row[None, c].store(
                    tKrKr_f32_row[None, c].load().to(tKrKr_row.element_type)
                )

            cute.copy(
                tiled_copy_QKG,
                tQrQd[((None, r), None, None)],
                tQsQd[((None, r), None, None)],
            )
            cute.copy(
                tiled_copy_QKG,
                tKrKd[((None, r), None, None)],
                tKsKd[((None, r), None, None)],
            )
            cute.copy(
                tiled_copy_QKG,
                tKrKi[((None, r), None, None)],
                tKsKi[((None, r), None, None)],
            )
            if cutlass.const_expr(
                valid_tokens is not None and seq_offset_padded is None
            ):
                if t + r < valid_tokens:
                    cute.copy(
                        tiled_copy_QKG,
                        tKrKr[((None, r), None, None)],
                        tKgKr[((None, r), None, None)],
                    )
            else:
                cute.copy(
                    tiled_copy_QKG,
                    tKrKr[((None, r), None, None)],
                    tKgKr[((None, r), None, None)],
                )

        tma_full.release()
        mma_empty.commit()
        return tma_consumer, gate_mma_producer

    @cute.jit
    def qk_l2_norm_row(
        self,
        tQrQ: cute.Tensor,
        tQrQ_f32: cute.Tensor,
        tKrK: cute.Tensor,
        tKrK_f32: cute.Tensor,
        r: int,
        t: cutlass.Int32,
        valid_tokens: Optional[cutlass.Int32] = None,
        seq_offset_padded: Optional[cutlass.Int32] = None,
    ):
        tQrQ_row = self.get_row(tQrQ, r)
        tQrQ_f32_row = self.get_row(tQrQ_f32, r)
        tKrK_row = self.get_row(tKrK, r)
        tKrK_f32_row = self.get_row(tKrK_f32, r)
        nc = cute.size(tQrQ_f32_row, mode=[1])
        tQrQ_f32_row.store(tQrQ_row.load().to(tQrQ_f32_row.element_type))
        tKrK_f32_row.store(tKrK_row.load().to(tKrK_f32_row.element_type))
        q_sq0, q_sq1 = cute.arch.mul_packed_f32x2(
            (tQrQ_f32_row[0, 0], tQrQ_f32_row[1, 0]),
            (tQrQ_f32_row[0, 0], tQrQ_f32_row[1, 0]),
        )
        k_sq0, k_sq1 = cute.arch.mul_packed_f32x2(
            (tKrK_f32_row[0, 0], tKrK_f32_row[1, 0]),
            (tKrK_f32_row[0, 0], tKrK_f32_row[1, 0]),
        )
        for c in cutlass.range_constexpr(1, nc):
            q_sq0, q_sq1 = cute.arch.fma_packed_f32x2(
                (tQrQ_f32_row[0, c], tQrQ_f32_row[1, c]),
                (tQrQ_f32_row[0, c], tQrQ_f32_row[1, c]),
                (q_sq0, q_sq1),
            )
            k_sq0, k_sq1 = cute.arch.fma_packed_f32x2(
                (tKrK_f32_row[0, c], tKrK_f32_row[1, c]),
                (tKrK_f32_row[0, c], tKrK_f32_row[1, c]),
                (k_sq0, k_sq1),
            )
        q_sq = q_sq0 + q_sq1
        k_sq = k_sq0 + k_sq1
        for d in cutlass.range_constexpr(3, -1, -1):
            q_sq += cute.arch.shuffle_sync_bfly(q_sq, 1 << d)
            k_sq += cute.arch.shuffle_sync_bfly(k_sq, 1 << d)
        q_inv = cute.math.rsqrt(q_sq + cutlass.Float32(1e-6), fastmath=True)
        k_inv = cute.math.rsqrt(k_sq + cutlass.Float32(1e-6), fastmath=True)
        if cutlass.const_expr(
            valid_tokens is not None and seq_offset_padded is not None
        ):
            k_inv = k_inv if t + r < valid_tokens else cutlass.Float32(0.0)
        for c in cutlass.range_constexpr(nc):
            tQrQ_f32_row[0, c], tQrQ_f32_row[1, c] = cute.arch.mul_packed_f32x2(
                (tQrQ_f32_row[0, c], tQrQ_f32_row[1, c]),
                (q_inv, q_inv),
            )
            tKrK_f32_row[0, c], tKrK_f32_row[1, c] = cute.arch.mul_packed_f32x2(
                (tKrK_f32_row[0, c], tKrK_f32_row[1, c]),
                (k_inv, k_inv),
            )

    @staticmethod
    @cute.jit
    def get_row(t: cute.Tensor, r: int):
        t = t[(None, r), 0, 0]
        return cute.make_tensor(
            t.iterator,
            cute.make_layout((2, cute.size(t) // 2), stride=(1, 2)),
        )

    @staticmethod
    @cute.jit
    def sigmoid(src_f16: cute.Tensor, dst_f32: cute.Tensor):
        dst_f32.store(src_f16.load().to(dst_f32.element_type))
        for i in cutlass.range_constexpr(cute.size(dst_f32) // 2):
            dst_f32[2 * i], dst_f32[2 * i + 1] = cute.arch.mul_packed_f32x2(
                (dst_f32[2 * i], dst_f32[2 * i + 1]),
                (cutlass.Float32(0.5), cutlass.Float32(0.5)),
            )
            dst_f32[2 * i] = tanh(dst_f32[2 * i])
            dst_f32[2 * i + 1] = tanh(dst_f32[2 * i + 1])
            dst_f32[2 * i], dst_f32[2 * i + 1] = cute.arch.fma_packed_f32x2(
                (dst_f32[2 * i], dst_f32[2 * i + 1]),
                (cutlass.Float32(0.5), cutlass.Float32(0.5)),
                (cutlass.Float32(0.5), cutlass.Float32(0.5)),
            )

    @staticmethod
    @cute.jit
    def add_identity(tCrC: cute.Tensor, tCrI: cute.Tensor):
        tCrIC = cute.make_rmem_tensor_like(tCrC, tCrC.element_type)
        for i in cutlass.range_constexpr(cute.size(tCrC) // 2):
            tCrIC[2 * i], tCrIC[2 * i + 1] = cute.arch.add_packed_f32x2(
                (tCrC[2 * i], tCrC[2 * i + 1]),
                (tCrI[2 * i], tCrI[2 * i + 1]),
            )
        return tCrIC

    @staticmethod
    @cute.jit
    def transpose(tArA: cute.Tensor, tBrB: cute.Tensor):
        tArA_u32 = cute.make_tensor(
            cute.recast_tensor(tArA, cutlass.Uint32).iterator, cute.make_layout((4,))
        )
        tBrB_u32 = cute.make_tensor(
            cute.recast_tensor(tBrB, cutlass.Uint32).iterator, cute.make_layout((4,))
        )
        for i in cutlass.range_constexpr(4):
            tBrB_u32[i] = movmatrix(tArA_u32[i])


class ChunkKdaFwdK2:
    def __init__(
        self, D: int = 128, BT: int = 16, BV: int = 64, transpose_S: bool = False
    ):
        self.D = D
        self.BT = BT
        self.BV = BV
        self.transpose_S = transpose_S

        self.CT = 4
        self.TT = self.CT * self.BT
        self.load_stages = 2 if BV == 128 else 3
        self.store_stages = 2 if BV == 128 else 3

        self.warp_threads = 32
        self.warpgroup_threads = 128
        self.warps_per_warpgroup = self.warpgroup_threads // self.warp_threads

        self.load_warp_id = 0
        self.store_warp_id = 1
        self.compute_warpgroup_ids = [1] if BV == 64 else [1, 2]
        self.compute_threads = self.warpgroup_threads * len(self.compute_warpgroup_ids)
        self.compute_warps = self.warps_per_warpgroup * len(self.compute_warpgroup_ids)
        self.threads_per_cta = self.warpgroup_threads + self.compute_threads

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (B, T, H, D)
        mK: cute.Tensor,  # (B, T, H, D)
        mKr: cute.Tensor,  # (B, T, H, D)
        mV: cute.Tensor,  # (B, T, H, D)
        mMqk: cute.Tensor,  # (B, NT, H, BT, BT)
        mMkk: cute.Tensor,  # (B, NT, H, BT, BT)
        mGk: cute.Tensor,  # (B, NT, H, D)
        mO: cute.Tensor,  # (B, T, H, D)
        mS_in: Optional[cute.Tensor] = None,  # (B, H, D, D)
        mS_out: Optional[cute.Tensor] = None,  # (B, H, D, D)
        mCuSeqlens: Optional[cute.Tensor] = None,  # (B + 1,)
        mStateIndices: Optional[cute.Tensor] = None,  # (B,)
        stream=None,
    ):
        if cutlass.const_expr(mCuSeqlens is not None):
            B = cute.size(mCuSeqlens) - 1
            H = cute.size(mQ, mode=[1])
            mQ = cute.make_tensor(
                mQ.iterator, cute.select(mQ.layout, mode=[0, 2, 1])
            )  # (T, D, H)
            mK = cute.make_tensor(
                mK.iterator, cute.select(mK.layout, mode=[0, 2, 1])
            )  # (T, D, H)
            mKr = cute.make_tensor(
                mKr.iterator, cute.select(mKr.layout, mode=[0, 2, 1])
            )  # (T, D, H)
            mV = cute.make_tensor(
                mV.iterator, cute.select(mV.layout, mode=[0, 2, 1])
            )  # (T, D, H)
            mMqk = cute.make_tensor(
                mMqk.iterator, cute.select(mMqk.layout, mode=[2, 3, 0, 1])
            )  # (BT, BT, NT, H)
            mMkk = cute.make_tensor(
                mMkk.iterator, cute.select(mMkk.layout, mode=[2, 3, 0, 1])
            )  # (BT, BT, NT, H)
            mGk = cute.make_tensor(
                mGk.iterator, cute.select(mGk.layout, mode=[2, 0, 1])
            )  # (D, NT, H)
            mO = cute.make_tensor(
                mO.iterator, cute.select(mO.layout, mode=[0, 2, 1])
            )  # (T, D, H)
        else:
            B, _, H, _ = mQ.shape
            mQ = cute.make_tensor(
                mQ.iterator, cute.select(mQ.layout, mode=[1, 3, 2, 0])
            )  # (T, D, H, B)
            mK = cute.make_tensor(
                mK.iterator, cute.select(mK.layout, mode=[1, 3, 2, 0])
            )  # (T, D, H, B)
            mKr = cute.make_tensor(
                mKr.iterator, cute.select(mKr.layout, mode=[1, 3, 2, 0])
            )  # (T, D, H, B)
            mV = cute.make_tensor(
                mV.iterator, cute.select(mV.layout, mode=[1, 3, 2, 0])
            )  # (T, D, H, B)
            mMqk = cute.make_tensor(
                mMqk.iterator, cute.select(mMqk.layout, mode=[3, 4, 1, 2, 0])
            )  # (BT, BT, NT, H, B)
            mMkk = cute.make_tensor(
                mMkk.iterator, cute.select(mMkk.layout, mode=[3, 4, 1, 2, 0])
            )  # (BT, BT, NT, H, B)
            mGk = cute.make_tensor(
                mGk.iterator, cute.select(mGk.layout, mode=[3, 1, 2, 0])
            )  # (D, NT, H, B)
            mO = cute.make_tensor(
                mO.iterator, cute.select(mO.layout, mode=[1, 3, 2, 0])
            )  # (T, D, H, B)
        if cutlass.const_expr(mS_in is not None):
            mS_in = cute.make_tensor(
                mS_in.iterator, cute.select(mS_in.layout, mode=[2, 3, 1, 0])
            )  # (D, D, H, B)
        if cutlass.const_expr(mS_out is not None):
            mS_out = cute.make_tensor(
                mS_out.iterator, cute.select(mS_out.layout, mode=[2, 3, 1, 0])
            )  # (D, D, H, B)

        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(mQ.element_type, cutlass.Float32, (16, 8, 16)),
            atom_layout_mnk=(self.BV // 16, 1, 1),
            permutation_mnk=(self.BV, self.BT, 16),
        )

        copy_atom_n = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            mK.element_type,
        )
        copy_atom_t = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            mV.element_type,
        )
        copy_atom_o = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=True, num_matrices=4),
            mO.element_type,
        )

        tiled_copy_Q = cute.make_tiled_copy_B(copy_atom_n, tiled_mma)
        tiled_copy_K = cute.make_tiled_copy_B(copy_atom_n, tiled_mma)
        tiled_copy_Kr = cute.make_tiled_copy_B(copy_atom_t, tiled_mma)
        tiled_copy_V = cute.make_tiled_copy_A(copy_atom_t, tiled_mma)
        tiled_copy_Mqk = cute.make_tiled_copy_B(copy_atom_n, tiled_mma)
        tiled_copy_Mkk = cute.make_tiled_copy_B(copy_atom_n, tiled_mma)
        tiled_copy_O = cute.make_tiled_copy_C(copy_atom_o, tiled_mma)

        smem_layout_atom_sw128 = cute.nvgpu.warpgroup.make_smem_layout_atom(
            cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128,
            mQ.element_type,
        )
        smem_layout_atom_sw32 = cute.nvgpu.warpgroup.make_smem_layout_atom(
            cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW32,
            mMqk.element_type,
        )

        sQK_layout_staged = cute.tile_to_shape(
            smem_layout_atom_sw128,
            (self.TT, self.D, self.load_stages),
            order=(0, 1, 2),
        )
        sQKt_layout_staged = cute.select(sQK_layout_staged, mode=[1, 0, 2])
        sQK_layout = cute.slice_(sQK_layout_staged, (None, None, 0))

        sV_layout_staged = cute.tile_to_shape(
            smem_layout_atom_sw128,
            (self.TT, self.BV, self.load_stages),
            order=(0, 1, 2),
        )
        sVt_layout_staged = cute.select(sV_layout_staged, mode=[1, 0, 2])
        sV_layout = cute.slice_(sV_layout_staged, (None, None, 0))

        sM_layout_staged = cute.tile_to_shape(
            smem_layout_atom_sw32,
            (self.BT, self.BT, self.CT, self.load_stages),
            order=(0, 1, 2, 3),
        )
        sM_layout = cute.slice_(sM_layout_staged, (None, None, None, 0))

        sGk_layout_staged = cute.make_layout(
            (self.D, self.CT, self.load_stages),
            stride=(1, self.D, self.CT * self.D),
        )
        sGk_layout = cute.slice_(sGk_layout_staged, (None, None, 0))

        sO_layout_staged = cute.tile_to_shape(
            smem_layout_atom_sw128,
            (self.TT, self.BV, self.store_stages),
            order=(0, 1, 2),
        )
        sOt_layout_staged = cute.select(sO_layout_staged, mode=[1, 0, 2])
        sO_layout = cute.slice_(sO_layout_staged, (None, None, 0))

        buffer_align_bytes = 1024

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[
                cute.struct.MemRange[mQ.element_type, cute.cosize(sQK_layout_staged)],
                buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[mK.element_type, cute.cosize(sQK_layout_staged)],
                buffer_align_bytes,
            ]
            sKr: cute.struct.Align[
                cute.struct.MemRange[mKr.element_type, cute.cosize(sQK_layout_staged)],
                buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[mV.element_type, cute.cosize(sV_layout_staged)],
                buffer_align_bytes,
            ]
            sMqk: cute.struct.Align[
                cute.struct.MemRange[mMqk.element_type, cute.cosize(sM_layout_staged)],
                buffer_align_bytes,
            ]
            sMkk: cute.struct.Align[
                cute.struct.MemRange[mMkk.element_type, cute.cosize(sM_layout_staged)],
                buffer_align_bytes,
            ]
            sGk: cute.struct.Align[
                cute.struct.MemRange[mGk.element_type, cute.cosize(sGk_layout_staged)],
                buffer_align_bytes,
            ]
            sO: cute.struct.Align[
                cute.struct.MemRange[mO.element_type, cute.cosize(sO_layout_staged)],
                buffer_align_bytes,
            ]
            tma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_stages * 2]
            tma_O_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.store_stages * 2]

        self.num_tma_load_bytes = (
            cute.size_in_bytes(mQ.element_type, sQK_layout)
            + cute.size_in_bytes(mK.element_type, sQK_layout)
            + cute.size_in_bytes(mKr.element_type, sQK_layout)
            + cute.size_in_bytes(mV.element_type, sV_layout)
            + cute.size_in_bytes(mMqk.element_type, sM_layout)
            + cute.size_in_bytes(mMkk.element_type, sM_layout)
            + cute.size_in_bytes(mGk.element_type, sGk_layout)
        )

        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mQ,
            sQK_layout,
            (self.TT, self.D),
        )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mK,
            sQK_layout,
            (self.TT, self.D),
        )
        tma_atom_Kr, tma_tensor_Kr = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mKr,
            sQK_layout,
            (self.TT, self.D),
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mV,
            sV_layout,
            (self.TT, self.BV),
        )
        tma_atom_Mqk, tma_tensor_Mqk = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mMqk,
            sM_layout,
            (self.BT, self.BT, self.CT),
        )
        tma_atom_Mkk, tma_tensor_Mkk = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mMkk,
            sM_layout,
            (self.BT, self.BT, self.CT),
        )
        tma_atom_Gk, tma_tensor_Gk = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mGk,
            sGk_layout,
            (self.D, self.CT),
        )
        tma_atom_O, tma_tensor_O = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mO,
            sO_layout,
            (self.TT, self.BV),
        )

        universal_copy_bits = 128
        copy_atom_O = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mO.element_type,
            num_bits_per_copy=universal_copy_bits,
        )
        copy_elements = universal_copy_bits // mO.element_type.width
        row_threads = self.BV // copy_elements
        thr_layout_O = cute.make_layout(
            (self.warp_threads // row_threads, row_threads), stride=(row_threads, 1)
        )
        val_layout_O = cute.make_layout((1, copy_elements))
        tiled_copy_O_s2g = cute.make_tiled_copy_tv(
            copy_atom_O, thr_layout_O, val_layout_O
        )

        grid = (self.D // self.BV, H, B)
        self.kernel(
            tma_atom_Q,
            tma_tensor_Q,
            tma_atom_K,
            tma_tensor_K,
            tma_atom_Kr,
            tma_tensor_Kr,
            tma_atom_V,
            tma_tensor_V,
            tma_atom_Mqk,
            tma_tensor_Mqk,
            tma_atom_Mkk,
            tma_tensor_Mkk,
            tma_atom_Gk,
            tma_tensor_Gk,
            tma_atom_O,
            tma_tensor_O,
            mO,
            mS_in,
            mS_out,
            mCuSeqlens,
            mStateIndices,
            sQK_layout_staged,
            sQKt_layout_staged,
            sV_layout_staged,
            sVt_layout_staged,
            sM_layout_staged,
            sGk_layout_staged,
            sO_layout_staged,
            sOt_layout_staged,
            tiled_mma,
            tiled_copy_Q,
            tiled_copy_K,
            tiled_copy_Kr,
            tiled_copy_V,
            tiled_copy_Mqk,
            tiled_copy_Mkk,
            tiled_copy_O,
            tiled_copy_O_s2g,
            SharedStorage,
        ).launch(
            grid=grid,
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_Q: cute.CopyAtom,
        tma_tensor_Q: cute.Tensor,
        tma_atom_K: cute.CopyAtom,
        tma_tensor_K: cute.Tensor,
        tma_atom_Kr: cute.CopyAtom,
        tma_tensor_Kr: cute.Tensor,
        tma_atom_V: cute.CopyAtom,
        tma_tensor_V: cute.Tensor,
        tma_atom_Mqk: cute.CopyAtom,
        tma_tensor_Mqk: cute.Tensor,
        tma_atom_Mkk: cute.CopyAtom,
        tma_tensor_Mkk: cute.Tensor,
        tma_atom_Gk: cute.CopyAtom,
        tma_tensor_Gk: cute.Tensor,
        tma_atom_O: cute.CopyAtom,
        tma_tensor_O: cute.Tensor,
        mO: cute.Tensor,
        mS_in: Optional[cute.Tensor],
        mS_out: Optional[cute.Tensor],
        mCuSeqlens: Optional[cute.Tensor],
        mStateIndices: Optional[cute.Tensor],
        sQK_layout_staged: cute.ComposedLayout,
        sQKt_layout_staged: cute.ComposedLayout,
        sV_layout_staged: cute.ComposedLayout,
        sVt_layout_staged: cute.ComposedLayout,
        sM_layout_staged: cute.ComposedLayout,
        sGk_layout_staged: cute.Layout,
        sO_layout_staged: cute.ComposedLayout,
        sOt_layout_staged: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
        tiled_copy_Q: cute.TiledCopy,
        tiled_copy_K: cute.TiledCopy,
        tiled_copy_Kr: cute.TiledCopy,
        tiled_copy_V: cute.TiledCopy,
        tiled_copy_Mqk: cute.TiledCopy,
        tiled_copy_Mkk: cute.TiledCopy,
        tiled_copy_O: cute.TiledCopy,
        tiled_copy_O_s2g: cute.TiledCopy,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(tidx // self.warp_threads)
        warpgroup_idx = cute.arch.make_warp_uniform(tidx // self.warpgroup_threads)

        v_idx, h_idx, b_idx = cute.arch.block_idx()
        state_idx = b_idx
        if cutlass.const_expr(mStateIndices is not None):
            state_idx = mStateIndices[b_idx]

        seq_len = cute.size(tma_tensor_Q, mode=[0])
        seq_offset = 0
        seq_offset_padded = 0
        chunk_offset = 0
        if cutlass.const_expr(mCuSeqlens is not None):
            seq_len = get_seq_len(mCuSeqlens, b_idx)
            seq_offset = get_seq_offset(mCuSeqlens, b_idx)
            seq_offset_padded = get_seq_offset(mCuSeqlens, b_idx, padding=self.TT)
            chunk_offset = get_seq_offset(
                mCuSeqlens, b_idx, padding=self.TT, index_unit=self.BT
            )
        num_tiles = cute.ceil_div(seq_len, self.TT)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(
            sQK_layout_staged.outer, swizzle=sQK_layout_staged.inner
        )
        sK = storage.sK.get_tensor(
            sQK_layout_staged.outer, swizzle=sQK_layout_staged.inner
        )
        sKr = storage.sKr.get_tensor(
            sQK_layout_staged.outer, swizzle=sQK_layout_staged.inner
        )
        sKrt = storage.sKr.get_tensor(
            sQKt_layout_staged.outer, swizzle=sQKt_layout_staged.inner
        )
        sV = storage.sV.get_tensor(
            sV_layout_staged.outer, swizzle=sV_layout_staged.inner
        )
        sVt = storage.sV.get_tensor(
            sVt_layout_staged.outer, swizzle=sVt_layout_staged.inner
        )
        sMqk = storage.sMqk.get_tensor(
            sM_layout_staged.outer, swizzle=sM_layout_staged.inner
        )
        sMkk = storage.sMkk.get_tensor(
            sM_layout_staged.outer, swizzle=sM_layout_staged.inner
        )
        sGk = storage.sGk.get_tensor(sGk_layout_staged)
        sO = storage.sO.get_tensor(
            sO_layout_staged.outer, swizzle=sO_layout_staged.inner
        )
        sOt = storage.sO.get_tensor(
            sOt_layout_staged.outer, swizzle=sOt_layout_staged.inner
        )

        tma_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        tma_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.compute_warps
        )
        tma_producer, tma_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.load_stages,
            producer_group=tma_producer_group,
            consumer_group=tma_consumer_group,
            barrier_storage=storage.tma_mbar_ptr.data_ptr(),
            tx_count=self.num_tma_load_bytes,
        ).make_participants()

        compute_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.compute_threads
        )
        store_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.warp_threads
        )
        tma_producer_O, tma_consumer_O = pipeline.PipelineAsync.create(
            num_stages=self.store_stages,
            producer_group=compute_group,
            consumer_group=store_group,
            barrier_storage=storage.tma_O_mbar_ptr.data_ptr(),
        ).make_participants()
        store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self.store_stages,
            producer_group=store_group,
        )

        if warp_idx == self.load_warp_id:
            if cutlass.const_expr(mCuSeqlens is not None):
                mQ_seq = cute.domain_offset((seq_offset_padded, 0, 0), tma_tensor_Q)
                mK_seq = cute.domain_offset((seq_offset_padded, 0, 0), tma_tensor_K)
                mKr_seq = cute.domain_offset((seq_offset_padded, 0, 0), tma_tensor_Kr)
                mV_seq = cute.domain_offset((seq_offset, 0, 0), tma_tensor_V)
                mMqk_seq = cute.domain_offset((0, 0, chunk_offset, 0), tma_tensor_Mqk)
                mMkk_seq = cute.domain_offset((0, 0, chunk_offset, 0), tma_tensor_Mkk)
                mGk_seq = cute.domain_offset((0, chunk_offset, 0), tma_tensor_Gk)
                gQ = cute.local_tile(
                    mQ_seq[None, None, h_idx], (self.TT, self.D), (None, None)
                )
                gK = cute.local_tile(
                    mK_seq[None, None, h_idx], (self.TT, self.D), (None, None)
                )
                gKr = cute.local_tile(
                    mKr_seq[None, None, h_idx], (self.TT, self.D), (None, None)
                )
                gV = cute.local_tile(
                    mV_seq[None, None, h_idx], (self.TT, self.BV), (None, None)
                )
                gMqk = cute.local_tile(
                    mMqk_seq[None, None, None, h_idx],
                    (self.BT, self.BT, self.CT),
                    (None, None, None),
                )
                gMkk = cute.local_tile(
                    mMkk_seq[None, None, None, h_idx],
                    (self.BT, self.BT, self.CT),
                    (None, None, None),
                )
                gGk = cute.local_tile(
                    mGk_seq[None, None, h_idx], (self.D, self.CT), (None, None)
                )
            else:
                gQ = cute.local_tile(
                    tma_tensor_Q[None, None, h_idx, b_idx],
                    (self.TT, self.D),
                    (None, None),
                )
                gK = cute.local_tile(
                    tma_tensor_K[None, None, h_idx, b_idx],
                    (self.TT, self.D),
                    (None, None),
                )
                gKr = cute.local_tile(
                    tma_tensor_Kr[None, None, h_idx, b_idx],
                    (self.TT, self.D),
                    (None, None),
                )
                gV = cute.local_tile(
                    tma_tensor_V[None, None, h_idx, b_idx],
                    (self.TT, self.BV),
                    (None, None),
                )
                gMqk = cute.local_tile(
                    tma_tensor_Mqk[None, None, None, h_idx, b_idx],
                    (self.BT, self.BT, self.CT),
                    (None, None, None),
                )
                gMkk = cute.local_tile(
                    tma_tensor_Mkk[None, None, None, h_idx, b_idx],
                    (self.BT, self.BT, self.CT),
                    (None, None, None),
                )
                gGk = cute.local_tile(
                    tma_tensor_Gk[None, None, h_idx, b_idx],
                    (self.D, self.CT),
                    (None, None),
                )

            tQsQ, tQgQ = cpasync.tma_partition(
                tma_atom_Q,
                0,
                cute.make_layout(1),
                cute.group_modes(sQ, 0, 2),
                cute.group_modes(gQ, 0, 2),
            )
            tKsK, tKgK = cpasync.tma_partition(
                tma_atom_K,
                0,
                cute.make_layout(1),
                cute.group_modes(sK, 0, 2),
                cute.group_modes(gK, 0, 2),
            )
            tKrsKr, tKrgKr = cpasync.tma_partition(
                tma_atom_Kr,
                0,
                cute.make_layout(1),
                cute.group_modes(sKr, 0, 2),
                cute.group_modes(gKr, 0, 2),
            )
            tVsV, tVgV = cpasync.tma_partition(
                tma_atom_V,
                0,
                cute.make_layout(1),
                cute.group_modes(sV, 0, 2),
                cute.group_modes(gV, 0, 2),
            )
            tMqksMqk, tMqkgMqk = cpasync.tma_partition(
                tma_atom_Mqk,
                0,
                cute.make_layout(1),
                cute.group_modes(sMqk, 0, 3),
                cute.group_modes(gMqk, 0, 3),
            )
            tMkksMkk, tMkkgMkk = cpasync.tma_partition(
                tma_atom_Mkk,
                0,
                cute.make_layout(1),
                cute.group_modes(sMkk, 0, 3),
                cute.group_modes(gMkk, 0, 3),
            )
            tGkGk, tGkgGk = cpasync.tma_partition(
                tma_atom_Gk,
                0,
                cute.make_layout(1),
                cute.group_modes(sGk, 0, 2),
                cute.group_modes(gGk, 0, 2),
            )

            for t in cutlass.range(num_tiles, unroll=0):
                tma_empty = tma_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_Q,
                    tQgQ[None, t, 0],
                    tQsQ[None, tma_empty.index],
                    tma_bar_ptr=tma_empty.barrier,
                )
                cute.copy(
                    tma_atom_K,
                    tKgK[None, t, 0],
                    tKsK[None, tma_empty.index],
                    tma_bar_ptr=tma_empty.barrier,
                )
                cute.copy(
                    tma_atom_Kr,
                    tKrgKr[None, t, 0],
                    tKrsKr[None, tma_empty.index],
                    tma_bar_ptr=tma_empty.barrier,
                )
                cute.copy(
                    tma_atom_V,
                    tVgV[None, t, v_idx],
                    tVsV[None, tma_empty.index],
                    tma_bar_ptr=tma_empty.barrier,
                )
                cute.copy(
                    tma_atom_Mqk,
                    tMqkgMqk[None, 0, 0, t],
                    tMqksMqk[None, tma_empty.index],
                    tma_bar_ptr=tma_empty.barrier,
                )
                cute.copy(
                    tma_atom_Mkk,
                    tMkkgMkk[None, 0, 0, t],
                    tMkksMkk[None, tma_empty.index],
                    tma_bar_ptr=tma_empty.barrier,
                )
                cute.copy(
                    tma_atom_Gk,
                    tGkgGk[None, 0, t],
                    tGkGk[None, tma_empty.index],
                    tma_bar_ptr=tma_empty.barrier,
                )

        elif warp_idx == self.store_warp_id:
            if cutlass.const_expr(mCuSeqlens is not None):
                mO_seq = cute.domain_offset((seq_offset, 0, 0), tma_tensor_O)
                gO = cute.local_tile(
                    mO_seq[None, None, h_idx], (self.TT, self.BV), (None, None)
                )
                mO_seq_s2g = cute.domain_offset((seq_offset, 0, 0), mO)
                gO_s2g = cute.local_tile(
                    mO_seq_s2g[None, None, h_idx], (self.TT, self.BV), (None, None)
                )
            else:
                gO = cute.local_tile(
                    tma_tensor_O[None, None, h_idx, b_idx],
                    (self.TT, self.BV),
                    (None, None),
                )

            tOsO, tOgO = cpasync.tma_partition(
                tma_atom_O,
                0,
                cute.make_layout(1),
                cute.group_modes(sO, 0, 2),
                cute.group_modes(gO, 0, 2),
            )

            for t in cutlass.range(num_tiles, unroll=0):
                tma_O_full = tma_consumer_O.wait_and_advance()
                cute.arch.fence_view_async_shared()
                if cutlass.const_expr(mCuSeqlens is not None):
                    valid_tokens = min(self.TT, seq_len - t * self.TT)
                    if valid_tokens == self.TT:
                        cute.copy(
                            tma_atom_O,
                            tOsO[None, tma_O_full.index],
                            tOgO[None, t, v_idx],
                        )
                        store_pipeline.producer_commit()
                        store_pipeline.producer_acquire()
                    else:
                        thr_copy_O_s2g = tiled_copy_O_s2g.get_slice(lane_idx)
                        tOsO_s2g = thr_copy_O_s2g.partition_S(
                            sO[None, None, tma_O_full.index]
                        )
                        tOrO_s2g = cute.make_rmem_tensor_like(tOsO_s2g)
                        tOgO_s2g = thr_copy_O_s2g.partition_D(
                            gO_s2g[None, None, t, v_idx]
                        )

                        cO = cute.make_identity_tensor((self.TT, self.BV))
                        tOcO_s2g = thr_copy_O_s2g.partition_S(cO)
                        pred_shape = (1, *tOcO_s2g.shape[1:])
                        tOpO_s2g = cute.make_rmem_tensor(pred_shape, cutlass.Boolean)
                        for m_idx in cutlass.range_constexpr(tOcO_s2g.shape[1]):
                            for n_idx in cutlass.range_constexpr(tOcO_s2g.shape[2]):
                                m, n = tOcO_s2g[(0, 0), m_idx, n_idx]
                                tOpO_s2g[0, m_idx, n_idx] = m < valid_tokens

                        cute.copy(tiled_copy_O_s2g, tOsO_s2g, tOrO_s2g)
                        cute.copy(tiled_copy_O_s2g, tOrO_s2g, tOgO_s2g, pred=tOpO_s2g)
                else:
                    cute.copy(
                        tma_atom_O,
                        tOsO[None, tma_O_full.index],
                        tOgO[None, t, v_idx],
                    )
                    store_pipeline.producer_commit()
                    store_pipeline.producer_acquire()
                tma_O_full.release()
            store_pipeline.producer_tail()

        elif warpgroup_idx in self.compute_warpgroup_ids:
            tidx_in_compute = tidx % self.compute_threads

            thr_mma = tiled_mma.get_slice(tidx_in_compute)

            S_shape = thr_mma.partition_shape_C((self.BV, self.D))
            tSrS = thr_mma.make_fragment_C(S_shape)  # 64 regs
            S_shape = thr_mma.partition_shape_A((self.BV, self.D))
            tArS = thr_mma.make_fragment_A(S_shape)  # 32 regs

            # tSrS always represents S^T; transpose_S describes the external buffers.
            if cutlass.const_expr(mS_in is not None):
                if cutlass.const_expr(self.transpose_S):
                    gS_in = cute.local_tile(
                        mS_in[None, None, h_idx, state_idx], (self.BV, self.D), (v_idx, 0)
                    )
                else:
                    gS_in = cute.local_tile(
                        mS_in[None, None, h_idx, state_idx], (self.D, self.BV), (0, v_idx)
                    )
                    gS_in = cute.make_tensor(
                        gS_in.iterator, cute.select(gS_in.layout, mode=[1, 0])
                    )
                tSgS_in = thr_mma.partition_C(gS_in)
                if cutlass.const_expr(tSgS_in.element_type.width == 16):
                    tSrS_f16 = cute.make_rmem_tensor_like(tSrS, tSgS_in.element_type)
                    cute.autovec_copy(tSgS_in, tSrS_f16)
                    tSrS.store(tSrS_f16.load().to(tSrS.element_type))
                else:
                    cute.autovec_copy(tSgS_in, tSrS)
            else:
                tSrS.fill(0.0)

            for _ in cutlass.range(num_tiles, unroll=0):
                tma_full = tma_consumer.wait_and_advance()
                tma_O_empty = None

                sQ_tile = cute.local_tile(
                    sQ[None, None, tma_full.index], (self.BT, self.D), (None, 0)
                )
                sK_tile = cute.local_tile(
                    sK[None, None, tma_full.index], (self.BT, self.D), (None, 0)
                )
                sKr_tile = cute.local_tile(
                    sKrt[None, None, tma_full.index], (self.D, self.BT), (0, None)
                )
                sVt_tile = cute.local_tile(
                    sVt[None, None, tma_full.index], (self.BV, self.BT), (0, None)
                )
                sMqk_tile = sMqk[None, None, None, tma_full.index]
                sMkk_tile = sMkk[None, None, None, tma_full.index]
                sGk_tile = cute.make_tensor(
                    sGk[None, None, tma_full.index].iterator,
                    cute.make_layout((self.BV, self.D, self.CT), stride=(0, 1, self.D)),
                )
                sOt_tile = cute.local_tile(
                    sOt[None, None, 0], (self.BV, self.BT), (0, None)
                )  # dummy

                tVrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK_tile))
                thr_copy_K = tiled_copy_K.get_slice(tidx_in_compute)
                tVsK_copy_view = thr_copy_K.partition_S(sK_tile)
                tVrK_copy_view = thr_copy_K.retile(tVrK)

                tOrQ = thr_mma.make_fragment_B(thr_mma.partition_B(sQ_tile))
                thr_copy_Q = tiled_copy_Q.get_slice(tidx_in_compute)
                tOsQ_copy_view = thr_copy_Q.partition_S(sQ_tile)
                tOrQ_copy_view = thr_copy_Q.retile(tOrQ)

                tPrV = thr_mma.make_fragment_A(thr_mma.partition_A(sVt_tile))
                tPrMkk = thr_mma.make_fragment_B(thr_mma.partition_B(sMkk_tile))
                thr_copy_V = tiled_copy_V.get_slice(tidx_in_compute)
                tPsV_copy_view = thr_copy_V.partition_S(sVt_tile)
                tPrV_copy_view = thr_copy_V.retile(tPrV)
                thr_copy_Mkk = tiled_copy_Mkk.get_slice(tidx_in_compute)
                tPsMkk_copy_view = thr_copy_Mkk.partition_S(sMkk_tile)
                tPrMkk_copy_view = thr_copy_Mkk.retile(tPrMkk)

                tSrKr = thr_mma.make_fragment_B(thr_mma.partition_B(sKr_tile))
                tSsGk = thr_mma.partition_C(sGk_tile)
                thr_copy_Kr = tiled_copy_Kr.get_slice(tidx_in_compute)
                tSsKr_copy_view = thr_copy_Kr.partition_S(sKr_tile)
                tSrKr_copy_view = thr_copy_Kr.retile(tSrKr)

                tOrO = thr_mma.make_fragment_C(
                    thr_mma.partition_C(sOt_tile)
                )  # 8 regs per BT
                tOrO.fill(0.0)
                tOrO_f16 = cute.make_rmem_tensor_like(tOrO, sOt_tile.element_type)
                thr_copy_O = tiled_copy_O.get_slice(tidx_in_compute)
                tOrO_copy_view = thr_copy_O.retile(tOrO_f16)
                tOsO_copy_view = thr_copy_O.partition_D(sOt_tile)

                tOrMqk = thr_mma.make_fragment_B(thr_mma.partition_B(sMqk_tile))
                thr_copy_Mqk = tiled_copy_Mqk.get_slice(tidx_in_compute)
                tOsMqk_copy_view = thr_copy_Mqk.partition_S(sMqk_tile)
                tOrMqk_copy_view = thr_copy_Mqk.retile(tOrMqk)

                for ti in cutlass.range_constexpr(self.CT):
                    tArS_f32 = cute.make_tensor(tSrS.iterator, tArS.layout)
                    if cutlass.const_expr(ti == 0):
                        cute.copy(
                            tiled_copy_K,
                            tVsK_copy_view[None, None, None, ti],
                            tVrK_copy_view[None, None, None, ti],
                        )  # 32 regs, 8 instructions
                        tma_O_empty = tma_producer_O.acquire_and_advance()

                    ### S f32 -> f16, dV = K @ S, load Q, V, Mkk, and first half of Gk

                    V_shape = thr_mma.partition_shape_C((self.BV, self.BT))
                    tVrV = thr_mma.make_fragment_C(V_shape)  # 8 regs
                    tVrV.fill(0.0)

                    tSsGk_mn = cute.make_tensor(
                        tSsGk[None, None, None, ti].iterator,
                        layout_acc_mn(tiled_mma, tSsGk[None, None, None, ti].layout),
                    )
                    tSsGk_mn = cute.flatten(tSsGk_mn[0, None])
                    tSrGk_mn = cute.make_rmem_tensor_like(tSsGk_mn)
                    tSrS_mn = cute.make_tensor(
                        tSrS.iterator, layout_acc_mn(tiled_mma, tSrS.layout)
                    )  # ((2,1),(2,16)):((2,0),(1,4))

                    total_k = cute.size(tArS, mode=[2])  # 8
                    tArS[None, None, 0].store(
                        tArS_f32[None, None, 0].load().to(tArS.element_type)
                    )
                    for k in cutlass.range_constexpr(total_k):  # 8
                        if cutlass.const_expr(k < total_k - 1):
                            tArS[None, None, k + 1].store(
                                tArS_f32[None, None, k + 1].load().to(tArS.element_type)
                            )

                        cute.gemm(
                            tiled_mma,
                            tVrV,
                            tArS[None, None, k],
                            tVrK[None, None, k, ti],
                            tVrV,
                        )

                        if cutlass.const_expr(k < total_k // 2):
                            cute.copy(
                                tiled_copy_Q,
                                tOsQ_copy_view[None, None, 2 * k, ti],
                                tOrQ_copy_view[None, None, 2 * k, ti],
                            )  # 32 regs
                            cute.copy(
                                tiled_copy_Q,
                                tOsQ_copy_view[None, None, 2 * k + 1, ti],
                                tOrQ_copy_view[None, None, 2 * k + 1, ti],
                            )  # 32 regs
                        else:
                            # 16 regs, 8 instructions
                            cute.autovec_copy(
                                tSsGk_mn[None, 2 * k - total_k],
                                tSrGk_mn[None, 2 * k - total_k],
                            )
                            cute.autovec_copy(
                                tSsGk_mn[None, 2 * k + 1 - total_k],
                                tSrGk_mn[None, 2 * k + 1 - total_k],
                            )

                        if cutlass.const_expr(k == 6):
                            cute.copy(
                                tiled_copy_V,
                                tPsV_copy_view[None, None, None, ti],
                                tPrV_copy_view[None, None, None, ti],
                            )  # 4 regs, 1 instruction
                            cute.copy(
                                tiled_copy_Mkk,
                                tPsMkk_copy_view[None, None, None, ti],
                                tPrMkk_copy_view[None, None, None, ti],
                            )  # 4 regs, 1 instruction

                    ### O = Q @ S, P = Mkk @ (V - dV), load second half of Gk and K^T

                    P_shape = thr_mma.partition_shape_C((self.BV, self.BT))
                    tPrP = thr_mma.make_fragment_C(P_shape)  # 8 regs
                    tPrP.fill(0.0)

                    total_k = cute.size(tArS, mode=[2])  # 8
                    for k in cutlass.range_constexpr(total_k):  # 8
                        cute.gemm(
                            tiled_mma,
                            tOrO[None, None, None, ti],
                            tArS[None, None, k],
                            tOrQ[None, None, k, ti],
                            tOrO[None, None, None, ti],
                        )

                        if cutlass.const_expr(k == total_k // 2):
                            tPrV_tile = tPrV[None, None, None, ti]
                            for i in cutlass.range_constexpr(cute.size(tPrV_tile)):
                                tPrV_tile[i] -= tVrV[i].to(tPrV_tile.element_type)
                            cute.gemm(
                                tiled_mma,
                                tPrP,
                                tPrV_tile,
                                tPrMkk[None, None, None, ti],
                                tPrP,
                            )

                        # 16 regs, 8 instructions
                        cute.autovec_copy(
                            tSsGk_mn[None, total_k + k], tSrGk_mn[None, total_k + k]
                        )

                        cute.copy(
                            tiled_copy_Kr,
                            tSsKr_copy_view[None, k, None, ti],
                            tSrKr_copy_view[None, k, None, ti],
                        )  # 32 regs, 8 instructions

                        if cutlass.const_expr(k >= total_k // 2):
                            for j in cutlass.range_constexpr(2):
                                c = ((k - total_k // 2) * 2 + j) * 2
                                for r in cutlass.range_constexpr(2):
                                    tSrS_mn[r, c], tSrS_mn[r, c + 1] = (
                                        cute.arch.mul_packed_f32x2(
                                            (tSrGk_mn[c], tSrGk_mn[c + 1]),
                                            (tSrS_mn[r, c], tSrS_mn[r, c + 1]),
                                        )
                                    )

                    P_shape = thr_mma.partition_shape_A((self.BV, self.BT))
                    tSrP = thr_mma.make_fragment_A(P_shape)
                    tPrP_f16 = cute.make_tensor(tSrP.iterator, tPrP.layout)
                    tPrP_f16.store(tPrP.load().to(tPrP_f16.element_type))

                    ### S = Gk * S + K^T @ P, load Mqk and K, O += Mqk @ P, store O

                    cute.copy(
                        tiled_copy_Mqk,
                        tOsMqk_copy_view[None, None, None, ti],
                        tOrMqk_copy_view[None, None, None, ti],
                    )  # 4 regs, 1 instruction

                    if cutlass.const_expr(ti == self.CT - 1):
                        tma_full.release()

                    assert (
                        cute.size(tSrP, mode=[1]) == 1
                        and cute.size(tSrP, mode=[2]) == 1
                    )  # rest_M = 1, rest_K = 1
                    total_n = cute.size(tSrKr, mode=[1])
                    prefetch_gk_s = total_n // 2
                    for n in cutlass.range_constexpr(total_n):  # 16
                        if cutlass.const_expr(n == total_n // 2):
                            cute.gemm(
                                tiled_mma,
                                tOrO[None, None, None, ti],
                                tSrP,
                                tOrMqk[None, None, None, ti],
                                tOrO[None, None, None, ti],
                            )  # 2 instructions

                        if cutlass.const_expr(n < total_n - prefetch_gk_s):
                            c = 2 * (n + prefetch_gk_s)
                            for r in cutlass.range_constexpr(2):
                                tSrS_mn[r, c], tSrS_mn[r, c + 1] = (
                                    cute.arch.mul_packed_f32x2(
                                        (tSrGk_mn[c], tSrGk_mn[c + 1]),
                                        (tSrS_mn[r, c], tSrS_mn[r, c + 1]),
                                    )
                                )
                        cute.gemm(
                            tiled_mma,
                            self.append_unit_mode(tSrS[None, 0, n], 2),
                            self.append_unit_mode(tSrP[None, 0, 0]),
                            self.append_unit_mode(tSrKr[None, n, 0, ti]),
                            self.append_unit_mode(tSrS[None, 0, n], 2),
                        )
                        if cutlass.const_expr(ti < self.CT - 1 and n % 2 == 0):
                            cute.copy(
                                tiled_copy_K,
                                tVsK_copy_view[None, None, n // 2, ti + 1],
                                tVrK_copy_view[None, None, n // 2, ti + 1],
                            )  # 32 regs, 8 instructions

                    if cutlass.const_expr(ti == 0):
                        sOt_tile = cute.local_tile(
                            sOt[None, None, tma_O_empty.index],
                            (self.BV, self.BT),
                            (0, None),
                        )
                        tOsO_copy_view = thr_copy_O.partition_D(sOt_tile)
                    tOrO_f16[None, None, None, ti].store(
                        tOrO[None, None, None, ti].load().to(tOrO_f16.element_type)
                    )
                    cute.copy(
                        tiled_copy_O,
                        tOrO_copy_view[None, None, None, ti],
                        tOsO_copy_view[None, None, None, ti],
                    )

                tma_O_empty.commit()

            if cutlass.const_expr(mS_out is not None):
                if cutlass.const_expr(self.transpose_S):
                    gS_out = cute.local_tile(
                        mS_out[None, None, h_idx, state_idx], (self.BV, self.D), (v_idx, 0)
                    )
                else:
                    gS_out = cute.local_tile(
                        mS_out[None, None, h_idx, state_idx], (self.D, self.BV), (0, v_idx)
                    )
                    gS_out = cute.make_tensor(
                        gS_out.iterator, cute.select(gS_out.layout, mode=[1, 0])
                    )
                tSgS_out = thr_mma.partition_C(gS_out)
                if cutlass.const_expr(tSgS_out.element_type.width == 16):
                    tSrS_f16 = cute.make_rmem_tensor_like(tSrS, tSgS_out.element_type)
                    tSrS_f16.store(tSrS.load().to(tSrS_f16.element_type))
                    cute.autovec_copy(tSrS_f16, tSgS_out)
                else:
                    cute.autovec_copy(tSrS, tSgS_out)

    @staticmethod
    def append_unit_mode(t: cute.Tensor, mode: int = 1):
        unit = cute.make_layout(1, stride=0)
        layout = t.layout
        for _ in range(mode):
            layout = cute.append(layout, unit)
        return cute.make_tensor(t.iterator, layout)


_BT = 16
_CT = 4
_TT = _CT * _BT


def _cute_dtype(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.float32:
        return cutlass.Float32
    if dtype == torch.int32:
        return cutlass.Int32
    if dtype == torch.int64:
        return cutlass.Int64
    raise ValueError(f"unsupported dtype: {dtype}")


def _fake(dtype, shape, assumed_align=16):
    return cute.runtime.make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=tuple(reversed(range(len(shape)))),
        assumed_align=assumed_align,
    )


def _fake_stride(dtype, shape):
    stride = tuple(cute.sym_int() for _ in range(len(shape) - 1))
    stride = stride + (1,)
    return cute.runtime.make_fake_tensor(
        dtype,
        shape,
        stride=stride,
        assumed_align=16,
    )


@functools.cache
def _compile_k1(
    dtype: torch.dtype,
    seqlen_dtype: torch.dtype,
    d: int,
    h: int,
    gate_lower_bound: bool,
    is_varlen: bool,
    max_active_clusters: int,
):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm Small-BH KDA before CUDA graph capture")
    cu_dtype = _cute_dtype(dtype)
    b, t, tp, nt = (cute.sym_int() for _ in range(4))
    feature: tuple[object, ...]
    padded_feature: tuple[object, ...]
    beta_shape: tuple[object, ...]
    matrix: tuple[object, ...]
    gate: tuple[object, ...]
    if is_varlen:
        feature = (t, h, d)
        padded_feature = (tp, h, d)
        beta_shape = (t, h)
        matrix = (nt, h, _BT, _BT)
        gate = (nt, h, d)
        cu_seqlens = _fake(_cute_dtype(seqlen_dtype), (b,), assumed_align=4)
    else:
        feature = (b, t, h, d)
        padded_feature = feature
        beta_shape = (b, t, h)
        matrix = (b, nt, h, _BT, _BT)
        gate = (b, nt, h, d)
        cu_seqlens = None

    q, k, g = [_fake_stride(cu_dtype, feature) for _ in range(3)]
    qd, kd, kr = [_fake(cu_dtype, padded_feature) for _ in range(3)]
    kernel = ChunkKdaFwdK1(
        D=d,
        H=h,
        BT=_BT,
        gate_lower_bound=gate_lower_bound,
        max_active_clusters=max_active_clusters,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        q,
        k,
        g,
        _fake_stride(cu_dtype, beta_shape),
        _fake(cutlass.Float32, (h, d)),
        _fake(cutlass.Float32, (h,)),
        qd,
        kd,
        kr,
        _fake(cu_dtype, matrix),
        _fake(cu_dtype, matrix),
        _fake(cutlass.Float32, gate),
        cutlass.Float32(1),
        cutlass.Float32(1),
        cu_seqlens,
        cutlass.Int32(1),
        stream,
        options="--enable-tvm-ffi",
    )


@functools.cache
def _compile_k2(
    dtype: torch.dtype,
    seqlen_dtype: torch.dtype,
    state_dtype: torch.dtype,
    d: int,
    bv: int,
    has_state_input: bool,
    has_state_output: bool,
    transpose_state: bool,
    is_varlen: bool,
    state_row_divisibility: int | None,
    has_state_indices: bool,
):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm Small-BH KDA before CUDA graph capture")
    cu_dtype = _cute_dtype(dtype)
    b, bs, t, tp, h, nt = (cute.sym_int() for _ in range(6))
    feature: tuple[object, ...]
    value: tuple[object, ...]
    output: tuple[object, ...]
    matrix: tuple[object, ...]
    gate: tuple[object, ...]
    if is_varlen:
        feature = (tp, h, d)
        value = output = (t, h, d)
        matrix = (nt, h, _BT, _BT)
        gate = (nt, h, d)
        cu_seqlens = _fake(_cute_dtype(seqlen_dtype), (bs,), assumed_align=4)
    else:
        feature = value = output = (b, t, h, d)
        matrix = (b, nt, h, _BT, _BT)
        gate = (b, nt, h, d)
        cu_seqlens = None
    state = (b, h, d, d)

    def fake_state():
        if state_row_divisibility is None:
            return _fake(_cute_dtype(state_dtype), state)
        # An indexed pool row starts slot * row_stride elements from the base. Pools that coalesce
        # every layer's state of a slot (KV cache manager V2; Kimi K3: 69 layers x 96 x 128 x 128 =
        # 108,527,616 elements per slot) take that product past int32 from slot 20 on, so the row
        # stride is a 64-bit runtime value; its divisibility keeps the rows' 16-byte alignment.
        return cute.runtime.make_fake_tensor(
            _cute_dtype(state_dtype), state,
            stride=(cute.sym_int64(divisibility=state_row_divisibility), d * d, d, 1),
            assumed_align=16,
        )

    q, k, kr = [_fake(cu_dtype, feature) for _ in range(3)]
    kernel = ChunkKdaFwdK2(D=d, BT=_BT, BV=bv, transpose_S=transpose_state)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        q,
        k,
        kr,
        _fake(cu_dtype, value),
        _fake(cu_dtype, matrix),
        _fake(cu_dtype, matrix),
        _fake(cutlass.Float32, gate),
        _fake(cu_dtype, output),
        fake_state() if has_state_input else None,
        fake_state() if has_state_output else None,
        cu_seqlens,
        _fake(cutlass.Int32, (cute.sym_int(),), assumed_align=4) if has_state_indices else None,
        stream,
        options="--enable-tvm-ffi",
    )


def chunk_kda_fwd_k1(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dt_bias: torch.Tensor,
    a_log: torch.Tensor,
    *,
    scale: float | None = None,
    gate_scale: float = 1.0,
    gate_lower_bound: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    outputs: tuple[torch.Tensor, ...] | None = None,
    max_active_clusters: int = 152,
) -> tuple[torch.Tensor, ...]:
    is_varlen = cu_seqlens is not None
    matrix_shape: tuple[int, ...]
    gate_shape: tuple[int, ...]
    if is_varlen:
        t, h, d = q.shape
        b = len(cu_seqlens) - 1
        t_pad = t + b * _TT
        nt = (t_pad + _BT - 1) // _BT
        if max_seqlen is None:
            max_seqlen = t
        seq_shape = (t_pad, h, d)
        matrix_shape = (nt, h, _BT, _BT)
        gate_shape = (nt, h, d)
    else:
        b, t, h, d = q.shape
        nt = ((t + _TT - 1) // _TT) * _CT
        max_seqlen = t
        seq_shape = q.shape
        matrix_shape = (b, nt, h, _BT, _BT)
        gate_shape = (b, nt, h, d)

    if d != 128:
        raise ValueError(f"unsupported d: {d}")

    if outputs is None:
        outputs = (
            *(torch.empty(seq_shape, dtype=q.dtype, device=q.device) for _ in range(3)),
            torch.empty(matrix_shape, dtype=q.dtype, device=q.device),
            torch.empty(matrix_shape, dtype=q.dtype, device=q.device),
            torch.empty(gate_shape, dtype=torch.float32, device=q.device),
        )
    qd, kd, kr, mqk, mkk, gk = outputs

    compiled = _compile_k1(
        q.dtype,
        cu_seqlens.dtype if cu_seqlens is not None else torch.int32,
        d,
        h,
        gate_lower_bound,
        is_varlen,
        max_active_clusters,
    )
    compiled(
        q,
        k,
        g,
        beta,
        dt_bias,
        a_log,
        qd,
        kd,
        kr,
        mqk,
        mkk,
        gk,
        d**-0.5 if scale is None else scale,
        gate_scale,
        cu_seqlens,
        max_seqlen,
    )
    return outputs


def chunk_kda_fwd_k2(
    q: torch.Tensor,
    k: torch.Tensor,
    kr: torch.Tensor,
    v: torch.Tensor,
    mqk: torch.Tensor,
    mkk: torch.Tensor,
    gk: torch.Tensor,
    *,
    return_state: bool = False,
    transpose_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    state_input: torch.Tensor | None = None,
    state_output: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    max_active_clusters: int = 152,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    is_varlen = cu_seqlens is not None
    if is_varlen:
        _, h, d = q.shape
        b = len(cu_seqlens) - 1
    else:
        b, _, h, d = q.shape
    bv = d if b * h * 2 > max_active_clusters else d // 2

    output = torch.empty_like(v) if output is None else output
    if return_state and state_output is None:
        state_output = torch.empty((b, h, d, d), dtype=torch.float32, device=q.device)

    has_input, has_output = state_input is not None, state_output is not None
    state_dtype = (
        state_input.dtype
        if state_input is not None
        else state_output.dtype
        if state_output is not None
        else q.dtype
    )
    # Indexed pool rows start slot * stride(0) elements from the base: one divisibility
    # (16-byte alignment at most) must hold for every state view the kernel addresses.
    state_views = [t for t in (state_input, state_output) if t is not None]
    state_row_divisibility = (
        math.gcd(*(t.stride(0) for t in state_views), 16 // state_views[0].element_size())
        if state_indices is not None
        else None
    )
    compiled = _compile_k2(
        q.dtype,
        cu_seqlens.dtype if cu_seqlens is not None else torch.int32,
        state_dtype,
        d,
        bv,
        has_input,
        has_output,
        transpose_state,
        is_varlen,
        state_row_divisibility,
        state_indices is not None,
    )
    compiled(
        q,
        k,
        kr,
        v,
        mqk,
        mkk,
        gk,
        output,
        state_input,
        state_output,
        cu_seqlens,
        state_indices,
    )
    return output, state_output


def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dt_bias: torch.Tensor,
    a_log: torch.Tensor,
    *,
    scale: float | None = None,
    gate_scale: float = 1.0,
    gate_lower_bound: bool = False,
    return_state: bool = False,
    transpose_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    output: torch.Tensor | None = None,
    state_input: torch.Tensor | None = None,
    state_output: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    workspace_tensors: tuple[torch.Tensor, ...] | None = None,
    max_active_clusters: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if max_active_clusters <= 0:
        max_active_clusters = torch.cuda.get_device_properties(q.device).multi_processor_count
    qd, kd, kr, mqk, mkk, gk = chunk_kda_fwd_k1(
        q,
        k,
        g,
        beta,
        dt_bias,
        a_log,
        scale=scale,
        gate_scale=gate_scale,
        gate_lower_bound=gate_lower_bound,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        outputs=workspace_tensors,
        max_active_clusters=max_active_clusters,
    )
    return chunk_kda_fwd_k2(
        qd,
        kd,
        kr,
        v,
        mqk,
        mkk,
        gk,
        return_state=return_state,
        transpose_state=transpose_state,
        cu_seqlens=cu_seqlens,
        output=output,
        state_input=state_input,
        state_output=state_output,
        state_indices=state_indices,
        max_active_clusters=max_active_clusters,
    )


def clear_compilation_cache() -> None:
    _compile_k1.cache_clear()
    _compile_k2.cache_clear()


__all__ = [
    "chunk_kda_fwd_k1",
    "chunk_kda_fwd_k2",
    "chunk_kda_fwd",
    "clear_compilation_cache",
]
