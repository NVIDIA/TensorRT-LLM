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
"""The QK of the attention kernel: S = Q' K'^T over the QK channels of NVFP4, both operands staged in shared memory.
Q' and K' are rows of 256 E2M1 codes (128 bytes: the 128 rotated head channels, then the 2 x rank compensation
channels, 192 codes used at rank 32 and all 256 at rank 64) with one UE4M3 scale factor per 16 channels. One class per
MMA width of the compile target holds the operand layouts, the TMA atoms of Q', K' and their scale factors, the load
warp's copies and the MMA warp's scale-factor staging (SMEM -> TMEM) and QK issue:

  sm_103a  QkK96    two K = 96 MMAs
  sm_100a  QkK64    three K = 64 MMAs
  sm_107a  QkK128   one K = 128 MMA (head channels) and one K = 64 MMA (compensation channels), Q' in TMEM; at rank
                    64 two K = 128 MMAs (head channels, compensation channels)

Rank 64 is offered where the target has the K = 128 instruction and the TMEM columns for the wider Q' (ranks).

The scale factors are read as the preparation wrote them: its power-of-two score scale in log2 units is the softmax
multiplier itself, so no per-call pass touches them."""

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.nvgpu.common import OperandMajorMode

from .. import target
from ..common.dsl import CTAGroup, Tcgen05CpShape, make_tmem_ptr, tcgen05_cp
from ..constants import DEFAULT_RANK, RANKS, qk_channels, qk_scale_atoms
from .ptx import smem_desc_sw128

ROW_ELEMS = 256  # E2M1 codes of a Q' / K' row: 128 bytes, the first qk_channels(rank) codes used


def ranks():
    """The compensation ranks of the compile target's QK: every rank where the target has the K = 128 instruction,
    else the default rank alone."""
    return RANKS if target.has_k128_fp4_mma() else (DEFAULT_RANK,)


# ------------------------------------------------------------------------------------------------
# Layouts of the block-scaled operands and their scale factors
# ------------------------------------------------------------------------------------------------
def k128_mma(n, a_source, cta_group=tcgen05.CtaGroup.ONE):
    """The K = 128 kind::mxf4nvf4 MMA (SM107MmaMXF4NVF4Op): M = 128, N = n, UE4M3 scales per 16 elements."""
    return cute.make_tiled_mma(
        tcgen05.SM107MmaMXF4NVF4Op(cutlass.Float8E4M3FN, 16, (128, n, 128), cta_group, a_source)
    )


def k128_view(mma128, tiler, x):
    """The staged E2M1 tiles x of a K = 64 MMA's shared-memory B layout as the B operand of the K = 128 MMA mma128
    (K blocks 0 and 1 of the K = 64 layout lie as the K = 128 operand does), with the stage mode of x."""
    view = cute.slice_(
        sm100_utils.make_smem_layout_b(mma128, tiler, cutlass.Float4E2M1FN, 1).outer,
        (None, None, None, 0),
    )
    return cute.make_tensor(x.iterator, cute.append(view, cute.get(x.layout, mode=[3])))


def sm103_ab_layout(rows, stages):
    """Staged Q' / K' tiles of the K = 96 MMA: K-major rows of 128 bytes as eight 16-byte blocks, 128-byte swizzle."""
    return tcgen05.tile_to_mma_shape(
        tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.Uint8),
        ((rows, 16), 1, 8, stages),
        order=(0, 1, 2),
    )


def s2t_partition(sSF, tSF, sf_dtype, iter_view=False):
    """The SMEM -> TMEM copy of the scale factors sSF into the TMEM view tSF: (tiled copy, source descriptors,
    destination). With iter_view the tiled copy is built from the view of tSF's first instruction iteration (one
    scale-factor buffer, its iteration modes reduced to one each); the destination keeps tSF's iteration modes."""
    tCsSF_compact = cute.filter_zeros(sSF)
    tCtSF_compact = cute.filter_zeros(tSF)
    copy_view = tCtSF_compact
    if iter_view:
        copy_view = cute.make_tensor(
            tCtSF_compact.iterator,
            cute.append(
                cute.append(tCtSF_compact[(None, 0, 0)].layout, cute.make_layout((1,))),
                cute.make_layout(1),
            ),
        )
    atom = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), sf_dtype)
    tiled = tcgen05.make_s2t_copy(atom, copy_view)
    thr = tiled.get_slice(0)
    src_ = thr.partition_S(tCsSF_compact)
    src = tcgen05.get_s2t_smem_desc_tensor(tiled, src_)
    dst = thr.partition_D(tCtSF_compact)
    return tiled, src, dst


# ------------------------------------------------------------------------------------------------
# The QK component
# ------------------------------------------------------------------------------------------------
class QkK96:
    """Q' and K' as byte rows in 128B-swizzled SMEM (one K = 96 instruction start every three 16-byte blocks), their
    UE4M3 scale-factor tiles staged into TMEM per QK: one Q tile (Q0's or Q1's, re-copied before each QK) and one K
    tile shared by both QKs of a KV step, four columns per 64 channels each (12 at rank 32)."""

    sf_dtype = cutlass.Float8E4M3FN
    k_smem_dtype = cutlass.Uint8
    q_tmem = False  # Q' read from TMEM (QkK128)
    sf_vec_size = 16
    tmem_qsf_offset = 192

    def __init__(self, mma_tiler, rank):
        # the channels of Q' and K' at this compensation rank
        self.head_dim = qk_channels(rank)
        self.mma_tiler_mnk = (*mma_tiler, self.head_dim)
        # the K' scale-factor slot follows the Q' one
        self.tmem_ksf_offset = self.tmem_qsf_offset + 4 * qk_scale_atoms(rank)
        # the scale-factor helpers describe the K = 96 instruction of sm_103a by its canonical K = 768 tile (eight
        # instructions), one stage of which covers the 192 channels
        self.layout_tiler = (*mma_tiler, 768)

    # -------------------------------------------------------------------------------- host side
    def global_tensors(self, qk_ptrs, s, h, b):
        """Q', K' [B, S, H, 256 E2M1] -> (s, d, ((h_r, h), b)) and their scale factors in the sm103 atom layout."""
        q_ptr, k_ptr, q_sf_ptr, k_sf_ptr = qk_ptrs
        d = self.head_dim
        h_r = 1
        q_layout = cute.make_layout(
            (s, d, ((h_r, h), b)),
            stride=(h * ROW_ELEMS, 1, ((ROW_ELEMS, ROW_ELEMS), s * h * ROW_ELEMS)),
        )
        q = cute.make_tensor(q_ptr, q_layout)
        k = cute.make_tensor(k_ptr, q_layout)
        # the sm_103 scale-factor atom views the 128 rows of the standard (32, 4) atom as (8, 4, 4) at the same byte
        # offsets: the three 64-channel atoms of the preparation are read in place
        sm103_sf_atom = blockscaled_utils.Sm103BlockScaledBasicChunk(
            self.sf_vec_size, OperandMajorMode.K
        ).layout
        q_sf_layout = cute.tile_to_shape(sm103_sf_atom, q.shape, (2, 1, 3))
        q_sf = cute.make_tensor(q_sf_ptr, q_sf_layout)
        k_sf_layout = cute.tile_to_shape(sm103_sf_atom, (s, d, (h, b)), (2, 1, 3))
        k_sf = cute.make_tensor(k_sf_ptr, k_sf_layout)
        return q, k, q_sf, k_sf

    def make_tiled_mma(self, cta_group):
        """Build the SM103 widened-K NVFP4 QK MMA (K=96)."""
        return sm100_utils.sm103_make_blockscaled_trivial_tiled_mma(
            self.sf_dtype,
            self.sf_vec_size,
            cta_group,
            self.mma_tiler_mnk[:2],
        )

    @staticmethod
    def adapt_sm103_ab_stage_for_tma(composed_layout):
        """Coalesce one SM103 128-byte AB stage to the 2-D TMA box view."""
        layout = composed_layout.outer
        part1 = cute.coalesce(cute.append(layout[0][0], layout[1]))
        part2 = cute.coalesce(cute.append(layout[0][1], layout[2]))
        return cute.make_composed_layout(
            composed_layout.inner,
            composed_layout.offset,
            cute.append(part1, part2),
        )

    @staticmethod
    def adapt_sm103_sf_stage_for_tma(layout):
        """Coalesce one SM103 SF stage to the 2-D TMA box view."""
        part1 = cute.coalesce(cute.append(layout[0][0], layout[1]))
        part2 = cute.coalesce(cute.append(layout[0][1], layout[2]))
        return cute.append(cute.group_modes(part1, 0, cute.rank(part1)), part2)

    def smem_layouts(self, tiled_mma, q_stage, kv_stage):
        """(staged Q layout, one K stage, (staged Q-SF layout, staged K-SF layout)); Q' and K' as K-major 128-byte rows
        of eight 16-byte blocks (sm103_ab_layout)."""
        q_smem_layout_staged = sm103_ab_layout(self.mma_tiler_mnk[0], q_stage)
        k_smem_layout_staged = sm103_ab_layout(self.mma_tiler_mnk[1], kv_stage)
        q_sf_smem_layout_staged = blockscaled_utils.sm103_make_smem_layout_sfa(
            tiled_mma,
            self.layout_tiler,
            self.sf_vec_size,
            q_stage,
        )
        k_sf_smem_layout_staged = blockscaled_utils.sm103_make_smem_layout_sfb(
            tiled_mma,
            self.layout_tiler,
            self.sf_vec_size,
            kv_stage,
        )
        # K is held as bytes on SM103, while V remains E4M3; the kernel gives both views one common byte stride so
        # the K/V pipeline can alias them
        k_stage_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        return (
            q_smem_layout_staged,
            k_stage_layout,
            (q_sf_smem_layout_staged, k_sf_smem_layout_staged),
        )

    def tma_atoms(
        self,
        tma_load_op,
        tma_kv_load_op,
        tensors,
        q_smem_layout_staged,
        k_smem_layout_staged,
        sf_layouts,
        tiled_mma,
        cluster_shape_mn,
        cluster_layout_vmnk,
        s,
        h,
        b,
    ):
        """TMA atoms and coordinate tensors of Q', K' and their scale factors (K' and K-SF multicast along the cluster)
        and the bytes one Q tile and one K tile land: ((atom_q, mQ, (atom_qsf, mQSF), atom_k, mK, (atom_ksf, mKSF)),
        q bytes, k bytes)."""
        q, k, q_sf, k_sf = tensors
        q_sf_smem_layout_staged, k_sf_smem_layout_staged = sf_layouts
        h_r = 1
        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        q_smem_layout_tma = self.adapt_sm103_ab_stage_for_tma(q_smem_layout)
        q_u8_compact = cute.recast_tensor(q, cutlass.Uint8)
        q_u8 = cute.make_tensor(
            q_u8_compact.iterator,
            cute.make_layout((s, 128, ((h_r, h), b)), stride=q_u8_compact.stride),
        )
        tma_atom_q, tma_tensor_q = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_load_op,
            q_u8,
            q_smem_layout_tma,
            (self.mma_tiler_mnk[0], 128),
            internal_type=cutlass.Uint8,
        )
        q_sf_smem_layout = cute.slice_(q_sf_smem_layout_staged, (None, None, None, 0))
        q_sf_smem_layout_tma = self.adapt_sm103_sf_stage_for_tma(q_sf_smem_layout)
        tma_atom_q_sf, tma_tensor_q_sf = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_load_op,
            q_sf,
            q_sf_smem_layout_tma,
            (self.mma_tiler_mnk[0], self.head_dim),
            internal_type=cutlass.Uint8,
        )
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        k_smem_layout_tma = self.adapt_sm103_ab_stage_for_tma(k_smem_layout)
        k_u8_compact = cute.recast_tensor(k, cutlass.Uint8)
        k_u8 = cute.make_tensor(
            k_u8_compact.iterator,
            cute.make_layout((s, 128, ((h_r, h), b)), stride=k_u8_compact.stride),
        )
        tma_atom_k, tma_tensor_k = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_kv_load_op,
            k_u8,
            k_smem_layout_tma,
            (self.mma_tiler_mnk[1], 128),
            cluster_shape_mn[0],
            internal_type=cutlass.Uint8,
        )
        k_sf_smem_layout = cute.slice_(k_sf_smem_layout_staged, (None, None, None, 0))
        k_sf_smem_layout_tma = self.adapt_sm103_sf_stage_for_tma(k_sf_smem_layout)
        tma_atom_k_sf, tma_tensor_k_sf = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_kv_load_op,
            k_sf,
            k_sf_smem_layout_tma,
            (self.mma_tiler_mnk[1], self.head_dim),
            cluster_shape_mn[0],
            internal_type=cutlass.Uint8,
        )
        q_copy_size = cute.size_in_bytes(cutlass.Uint8, q_smem_layout)
        k_copy_size = cute.size_in_bytes(cutlass.Uint8, k_smem_layout)
        q_sf_copy_size = cute.size_in_bytes(self.sf_dtype, q_sf_smem_layout)
        k_sf_copy_size = cute.size_in_bytes(self.sf_dtype, k_sf_smem_layout)
        if cutlass.const_expr(q_sf_copy_size % 512 != 0 or k_sf_copy_size % 512 != 0):
            raise RuntimeError("scale-factor stages must be a multiple of 512 bytes")
        atoms = (
            tma_atom_q,
            tma_tensor_q,
            (tma_atom_q_sf, tma_tensor_q_sf),
            tma_atom_k,
            tma_tensor_k,
            (tma_atom_k_sf, tma_tensor_k_sf),
        )
        return atoms, q_copy_size + q_sf_copy_size, k_copy_size + k_sf_copy_size

    def storage_fields(self, q_smem_layout_staged, sK_cosize, sf_layouts, align):
        """Shared-memory fields of the QK operands, in the order they follow the O staging tiles."""
        q_sf_smem_layout_staged, k_sf_smem_layout_staged = sf_layouts
        MemRange, Align = cute.struct.MemRange, cute.struct.Align
        return dict(
            sQ=Align[MemRange[cutlass.Uint8, cute.cosize(q_smem_layout_staged.outer)], align],
            sQSF=Align[MemRange[self.sf_dtype, cute.cosize(q_sf_smem_layout_staged)], align],
            sK=Align[MemRange[cutlass.Uint8, sK_cosize], align],
            sKSF=Align[MemRange[self.sf_dtype, cute.cosize(k_sf_smem_layout_staged)], align],
        )

    # -------------------------------------------------------------------------------- kernel: all warps
    def smem_tensors(self, storage, q_smem_layout_staged, k_smem_layout_staged, sf_layouts):
        """(sQ, sK, (sQSF, sKSF)): the staged Q tiles and the K ring of the CTA's shared memory."""
        q_sf_smem_layout_staged, k_sf_smem_layout_staged = sf_layouts
        sQ = storage.sQ.get_tensor(q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner)
        sQSF = storage.sQSF.get_tensor(q_sf_smem_layout_staged)
        sK = storage.sK.get_tensor(k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner)
        sKSF = storage.sKSF.get_tensor(k_sf_smem_layout_staged)
        return sQ, sK, (sQSF, sKSF)

    # -------------------------------------------------------------------------------- kernel: load warp
    def prefetch_descriptors(self, tma_atom_q, q_sf_tma, tma_atom_k, k_sf_tma):
        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
        cute.nvgpu.cpasync.prefetch_descriptor(q_sf_tma[0])
        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
        cute.nvgpu.cpasync.prefetch_descriptor(k_sf_tma[0])

    def load_partitions(
        self,
        tma_atom_q,
        mQ_qdl,
        q_sf_tma,
        tma_atom_k,
        mK_kdl,
        k_sf_tma,
        sQ,
        sK,
        sf_smem,
        qk_thr_mma,
        hb_coord,
        kv_cta_coord,
        kv_cta_layout,
    ):
        """TMA partitions of one work tile's head: Q' and Q-SF tiles (local), K' and K-SF tiles (multicast)."""
        tma_atom_q_sf, mQSF_qdl = q_sf_tma
        tma_atom_k_sf, mKSF_kdl = k_sf_tma
        sQSF, sKSF = sf_smem
        gQ_qdl = cute.local_tile(
            mQ_qdl,
            (self.mma_tiler_mnk[0], 128),
            (None, None, None),
        )
        tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(gQ_qdl, 0, 2),
        )
        tQgQ = tQgQ_qdl[None, None, 0, hb_coord]
        gQSF_qdl = cute.local_tile(
            mQSF_qdl,
            (self.mma_tiler_mnk[0], self.head_dim),
            (None, None, None),
        )
        tQsQSF, tQgQSF_qdl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_q_sf,
            0,
            cute.make_layout(1),
            cute.group_modes(sQSF, 0, 3),
            cute.group_modes(gQSF_qdl, 0, 2),
        )
        tQsQSF = cute.filter_zeros(tQsQSF)
        tQgQSF_qdl = cute.filter_zeros(tQgQSF_qdl)
        sf_l_q = cute.make_layout(
            mQ_qdl.shape[2],
            stride=((1, mQ_qdl.shape[2][0][0]), cute.size(mQ_qdl.shape[2][0])),
        )(hb_coord)
        tQgQSF = tQgQSF_qdl[None, None, 0, sf_l_q]
        gK_kdl = cute.local_tile(
            mK_kdl,
            (self.mma_tiler_mnk[1], 128),
            (None, None, None),
        )
        tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_k,
            kv_cta_coord,
            kv_cta_layout,
            cute.group_modes(sK, 0, 3),
            cute.group_modes(gK_kdl, 0, 2),
        )
        tKgK = tKgK_kdl[None, None, 0, hb_coord]
        gKSF_kdl = cute.local_tile(
            mKSF_kdl,
            (self.mma_tiler_mnk[1], self.head_dim),
            (None, None, None),
        )
        tKsKSF, tKgKSF_kdl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_k_sf,
            kv_cta_coord,
            kv_cta_layout,
            cute.group_modes(sKSF, 0, 3),
            cute.group_modes(gKSF_kdl, 0, 2),
        )
        tKsKSF = cute.filter_zeros(tKsKSF)
        tKgKSF_kdl = cute.filter_zeros(tKgKSF_kdl)
        sf_l_k = cute.make_layout(
            mK_kdl.shape[2],
            stride=((0, 1), mK_kdl.shape[2][0][1]),
        )(hb_coord)
        tKgKSF = tKgKSF_kdl[None, None, 0, sf_l_k]
        return (
            (tma_atom_q, tQsQ, tQgQ, tQgQ_qdl, tma_atom_q_sf, tQsQSF, tQgQSF, tQgQSF_qdl, mQ_qdl),
            (tma_atom_k, tKsK, tKgK, tma_atom_k_sf, tKsKSF, tKgKSF),
        )

    def copy_q(self, partitions, q_coord, q_handle):
        """Q tile q_coord and its scale factors into the handle's Q stage."""
        tma_atom_q, tQsQ, tQgQ, _, tma_atom_q_sf, tQsQSF, tQgQSF, _, _ = partitions[0]
        cute.copy(
            tma_atom_q,
            tQgQ[None, q_coord],
            tQsQ[None, q_handle.index],
            tma_bar_ptr=q_handle.barrier,
        )
        cute.copy(
            tma_atom_q_sf,
            tQgQSF[None, q_coord],
            tQsQSF[None, q_handle.index],
            tma_bar_ptr=q_handle.barrier,
        )

    def copy_k(self, partitions, kv_coord, k_handle, kv_mcast_mask):
        """K tile kv_coord and its scale factors into the handle's ring slot."""
        tma_atom_k, tKsK, tKgK, tma_atom_k_sf, tKsKSF, tKgKSF = partitions[1]
        cute.copy(
            tma_atom_k,
            tKgK[None, kv_coord],
            tKsK[None, k_handle.index],
            tma_bar_ptr=k_handle.barrier,
            mcast_mask=kv_mcast_mask,
        )
        cute.copy(
            tma_atom_k_sf,
            tKgKSF[None, kv_coord],
            tKsKSF[None, k_handle.index],
            tma_bar_ptr=k_handle.barrier,
            mcast_mask=kv_mcast_mask,
        )

    def prefetch_q(self, partitions, nb):
        """Prefetch the Q / Q-SF tiles of the work unit nb = (Q block pair, head, batch)."""
        tma_atom_q, _, _, tQgQ_qdl, tma_atom_q_sf, _, _, tQgQSF_qdl, mQ_qdl = partitions[0]
        n_hb = (nb[1], nb[2])
        tQgQ_n = tQgQ_qdl[None, None, 0, n_hb]
        sf_l_q_n = cute.make_layout(
            mQ_qdl.shape[2],
            stride=((1, mQ_qdl.shape[2][0][0]), cute.size(mQ_qdl.shape[2][0])),
        )(n_hb)
        tQgQSF_n = tQgQSF_qdl[None, None, 0, sf_l_q_n]
        nq0 = 2 * nb[0]
        cute.prefetch(tma_atom_q, tQgQ_n[None, nq0])
        cute.prefetch(tma_atom_q, tQgQ_n[None, nq0 + 1])
        cute.prefetch(tma_atom_q_sf, tQgQSF_n[None, nq0])
        cute.prefetch(tma_atom_q_sf, tQgQSF_n[None, nq0 + 1])

    # -------------------------------------------------------------------------------- kernel: MMA warp
    def mma_setup(self, tiled_mma, tmem_ptr, sQ, sK, sf_smem, sf_layouts):
        """The MMA warp's QK state: the TMEM scale-factor views (copy and instruction shapes) and their SMEM -> TMEM
        copies (both QKs of a step use the one Q slot); Q' and K' stay byte tensors, from which descs forms the K = 96
        descriptors."""
        sQSF, sKSF = sf_smem
        q_sf_smem_layout_staged, k_sf_smem_layout_staged = sf_layouts
        q_sf_tmem_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma,
            self.layout_tiler,
            self.sf_vec_size,
            cute.slice_(q_sf_smem_layout_staged, (None, None, None, 0)),
        )
        k_sf_tmem_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma,
            self.layout_tiler,
            self.sf_vec_size,
            cute.slice_(k_sf_smem_layout_staged, (None, None, None, 0)),
        )
        # one scale-factor slot serves both K = 96 instructions: the instruction view advances six UE4M3 values per
        # row between them
        sf_iter = cute.make_layout(
            ((((32, 4), 1), (self.sf_vec_size, 1)), 1, 384 // self.sf_vec_size)
        )
        q_sf_tmem_layout_mma = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma, self.layout_tiler, self.sf_vec_size, sf_iter
        )
        k_sf_tmem_layout_mma = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma, self.layout_tiler, self.sf_vec_size, sf_iter
        )
        # each scale-factor slot takes 12 TMEM columns (128 rows, the 192 channels of this QK)
        qsf_ptr = cute.recast_ptr(tmem_ptr + self.tmem_qsf_offset, dtype=self.sf_dtype)
        ksf_ptr = cute.recast_ptr(tmem_ptr + self.tmem_ksf_offset, dtype=self.sf_dtype)
        tCtQSF_copy = cute.make_tensor(qsf_ptr, q_sf_tmem_layout)
        tCtKSF_copy = cute.make_tensor(ksf_ptr, k_sf_tmem_layout)
        tCtQSF = cute.make_tensor(qsf_ptr, q_sf_tmem_layout_mma)
        tCtKSF = cute.make_tensor(ksf_ptr, k_sf_tmem_layout_mma)
        q_sf = (*s2t_partition(sQSF, tCtQSF_copy, self.sf_dtype, iter_view=True), tCtQSF)
        k_sf = (*s2t_partition(sKSF, tCtKSF_copy, self.sf_dtype, iter_view=True), tCtKSF)
        return tiled_mma, sQ, sK, q_sf, k_sf

    def q_operand(self, qk_mma, q_stage):
        """The Q stage's SMEM tile (descriptors are built from it per QK)."""
        return qk_mma[1][None, None, None, q_stage]

    def k_operand(self, qk_mma, k_stage):
        return qk_mma[2][None, None, None, k_stage]

    def stage_scales(self, qk_mma, q_stage, k_stage=None):
        """Copy the K-SF of ring slot k_stage (when given), then the Q-SF of Q stage q_stage, into TMEM."""
        q_sf, k_sf = qk_mma[3], qk_mma[4]
        if cutlass.const_expr(k_stage is not None):
            cute.copy(
                k_sf[0],
                k_sf[1][(None, None, None, None, k_stage)],
                k_sf[2],
            )
        cute.copy(q_sf[0], q_sf[1][(None, None, None, None, q_stage)], q_sf[2])

    def descs(self, sA, sB):
        """SMEM descriptors of one QK (two K = 96 instructions), built before the S-empty wait."""
        a_next = sA[(None, 0, 0)]
        b_next = sB[(None, 0, 0)]
        descs = []
        for kb in (0, 3):
            a_cur = sA[(None, 0, kb)]
            b_cur = sB[(None, 0, kb)]
            descs.append(
                tcgen05.make_umma_smem_desc(
                    a_cur.iterator, a_cur.layout, "k", next_src=a_next.iterator
                )
            )
            descs.append(
                tcgen05.make_umma_smem_desc(
                    b_cur.iterator, b_cur.layout, "k", next_src=b_next.iterator
                )
            )
        return descs

    def issue_qk(self, qk_mma, j, d, descs, si_handle):
        """Issue QK_j from prebuilt descriptors and commit its S credit."""
        tiled_mma, _, _, q_sf, k_sf = qk_mma
        sfA, sfB = q_sf[3], k_sf[3]
        qk_tiled_mma = cutlass.new_from_mlir_values(
            tiled_mma, cutlass.extract_mlir_values(tiled_mma)
        )
        d_mma = d[(None, 0, 0)]
        view_layout = cute.make_layout(1, stride=0)
        for i, sf_k in enumerate((0, 6)):
            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, i != 0)
            cute.mma_atom_call(
                qk_tiled_mma,
                d_mma,
                [cute.make_tensor(descs[2 * i], view_layout), sfA[(None, 0, sf_k)]],
                [cute.make_tensor(descs[2 * i + 1], view_layout), sfB[(None, 0, sf_k)]],
                d_mma,
            )
        si_handle.commit()


class QkK64(QkK96):
    """The same operands on sm_100a (K = 64 instructions): Q' and K' as E2M1 tiles in the standard 128B-swizzled
    K-major layout over the padded 256-channel rows (the channels past the QK's are zero and unused), scale factors
    in the standard (32, 4) atoms (the same bytes the sm_103a path reinterprets), and one K = 64 block-scaled MMA per
    64 channels of the QK (three at rank 32). The TMEM scale-factor slots are as large as on sm_103a."""

    k_smem_dtype = cutlass.Float4E2M1FN

    def __init__(self, mma_tiler, rank):
        super().__init__(mma_tiler, rank)
        self.pad_tiler = (*mma_tiler, ROW_ELEMS)
        self.k_blocks = self.head_dim // 64

    def global_tensors(self, qk_ptrs, s, h, b):
        q_ptr, k_ptr, q_sf_ptr, k_sf_ptr = qk_ptrs
        h_r = 1
        layout = cute.make_layout(
            (s, ROW_ELEMS, ((h_r, h), b)),
            stride=(h * ROW_ELEMS, 1, ((ROW_ELEMS, ROW_ELEMS), s * h * ROW_ELEMS)),
        )
        atom = blockscaled_utils.BlockScaledBasicChunk(self.sf_vec_size, OperandMajorMode.K).layout
        q_sf = cute.make_tensor(
            q_sf_ptr, cute.tile_to_shape(atom, (s, self.head_dim, ((h_r, h), b)), (2, 1, 3))
        )
        k_sf = cute.make_tensor(
            k_sf_ptr, cute.tile_to_shape(atom, (s, self.head_dim, (h, b)), (2, 1, 3))
        )
        return cute.make_tensor(q_ptr, layout), cute.make_tensor(k_ptr, layout), q_sf, k_sf

    def make_tiled_mma(self, cta_group, a_source=tcgen05.OperandSource.SMEM):
        return cute.make_tiled_mma(
            tcgen05.MmaMXF4NVF4Op(self.sf_dtype, (*self.mma_tiler_mnk[:2], 64), cta_group, a_source)
        )

    def smem_layouts(self, tiled_mma, q_stage, kv_stage):
        f4 = cutlass.Float4E2M1FN
        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.pad_tiler, f4, q_stage
        )
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.pad_tiler, f4, kv_stage
        )
        q_sf = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler_mnk, self.sf_vec_size, q_stage
        )
        k_sf = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler_mnk, self.sf_vec_size, kv_stage
        )
        return q_smem_layout_staged, cute.select(k_smem_layout_staged, mode=[0, 1, 2]), (q_sf, k_sf)

    def tma_atoms(
        self,
        tma_load_op,
        tma_kv_load_op,
        tensors,
        q_smem_layout_staged,
        k_smem_layout_staged,
        sf_layouts,
        tiled_mma,
        cluster_shape_mn,
        cluster_layout_vmnk,
        s,
        h,
        b,
    ):
        q, k, q_sf, k_sf = tensors
        q_sf_smem_layout = cute.slice_(sf_layouts[0], (None, None, None, 0))
        k_sf_smem_layout = cute.slice_(sf_layouts[1], (None, None, None, 0))
        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        vmnk = cluster_layout_vmnk.shape
        make_a, make_b = cute.nvgpu.make_tiled_tma_atom_A, cute.nvgpu.make_tiled_tma_atom_B
        tma_atom_q, tma_tensor_q = make_a(
            tma_load_op, q, q_smem_layout, self.pad_tiler, tiled_mma, vmnk
        )
        tma_atom_q_sf, tma_tensor_q_sf = make_a(
            tma_load_op,
            q_sf,
            q_sf_smem_layout,
            self.mma_tiler_mnk,
            tiled_mma,
            vmnk,
            internal_type=cutlass.Int16,
        )
        tma_atom_k, tma_tensor_k = make_b(
            tma_kv_load_op, k, k_smem_layout, self.pad_tiler, tiled_mma, vmnk
        )
        # every CTA loads its own K scale factors: the (32, 4)-interleaved atoms are loaded whole
        local = cute.tiled_divide(cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)).shape
        tma_atom_k_sf, tma_tensor_k_sf = make_b(
            tma_load_op,
            k_sf,
            k_sf_smem_layout,
            self.mma_tiler_mnk,
            tiled_mma,
            local,
            internal_type=cutlass.Int16,
        )
        f4 = cutlass.Float4E2M1FN
        atoms = (
            tma_atom_q,
            tma_tensor_q,
            (tma_atom_q_sf, tma_tensor_q_sf),
            tma_atom_k,
            tma_tensor_k,
            (tma_atom_k_sf, tma_tensor_k_sf),
        )
        q_bytes = cute.size_in_bytes(f4, q_smem_layout) + cute.size_in_bytes(
            self.sf_dtype, q_sf_smem_layout
        )
        k_bytes = cute.size_in_bytes(f4, k_smem_layout) + cute.size_in_bytes(
            self.sf_dtype, k_sf_smem_layout
        )
        return atoms, q_bytes, k_bytes

    def storage_fields(self, q_smem_layout_staged, sK_cosize, sf_layouts, align):
        fields = super().storage_fields(q_smem_layout_staged, sK_cosize, sf_layouts, align)
        # E2M1 elements, two per byte
        fields["sQ"] = cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, cute.cosize(q_smem_layout_staged.outer) // 2], align
        ]
        return fields

    def smem_tensors(self, storage, q_smem_layout_staged, k_smem_layout_staged, sf_layouts):
        f4 = cutlass.Float4E2M1FN
        sQ = cute.make_tensor(
            cute.recast_ptr(
                cute.recast_ptr(storage.sQ.data_ptr(), dtype=f4), q_smem_layout_staged.inner
            ),
            q_smem_layout_staged.outer,
        )
        sK = cute.make_tensor(
            cute.recast_ptr(
                cute.recast_ptr(storage.sK.data_ptr(), dtype=f4), k_smem_layout_staged.inner
            ),
            k_smem_layout_staged.outer,
        )
        return (
            sQ,
            sK,
            (storage.sQSF.get_tensor(sf_layouts[0]), storage.sKSF.get_tensor(sf_layouts[1])),
        )

    def load_partitions(
        self,
        tma_atom_q,
        mQ_qdl,
        q_sf_tma,
        tma_atom_k,
        mK_kdl,
        k_sf_tma,
        sQ,
        sK,
        sf_smem,
        qk_thr_mma,
        hb_coord,
        kv_cta_coord,
        kv_cta_layout,
    ):
        tma_atom_q_sf, mQSF_qdl = q_sf_tma
        tma_atom_k_sf, mKSF_kdl = k_sf_tma
        sQSF, sKSF = sf_smem
        tma_partition = cute.nvgpu.cpasync.tma_partition
        ab_tile_q = cute.select(self.pad_tiler, mode=[0, 2])
        sf_tile_q = cute.select(self.mma_tiler_mnk, mode=[0, 2])
        ab_tile_k = cute.select(self.pad_tiler, mode=[1, 2])
        sf_tile_k = cute.select(self.mma_tiler_mnk, mode=[1, 2])
        gQ = qk_thr_mma.partition_A(cute.local_tile(mQ_qdl, ab_tile_q, (None, None, None)))
        tQsQ, tQgQ_qdl = tma_partition(
            tma_atom_q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(gQ, 0, 3),
        )
        tQgQ = tQgQ_qdl[None, None, 0, hb_coord]
        gQSF = qk_thr_mma.partition_A(cute.local_tile(mQSF_qdl, sf_tile_q, (None, None, None)))
        tQsQSF, tQgQSF_qdl = tma_partition(
            tma_atom_q_sf,
            0,
            cute.make_layout(1),
            cute.group_modes(sQSF, 0, 3),
            cute.group_modes(gQSF, 0, 3),
        )
        tQsQSF = cute.filter_zeros(tQsQSF)
        tQgQSF_qdl = cute.filter_zeros(tQgQSF_qdl)
        sf_l_q = cute.make_layout(
            mQ_qdl.shape[2],
            stride=((1, mQ_qdl.shape[2][0][0]), cute.size(mQ_qdl.shape[2][0])),
        )(hb_coord)
        tQgQSF = tQgQSF_qdl[None, None, 0, sf_l_q]
        gK = qk_thr_mma.partition_B(cute.local_tile(mK_kdl, ab_tile_k, (None, None, None)))
        tKsK, tKgK_kdl = tma_partition(
            tma_atom_k,
            kv_cta_coord,
            kv_cta_layout,
            cute.group_modes(sK, 0, 3),
            cute.group_modes(gK, 0, 3),
        )
        tKgK = tKgK_kdl[None, None, 0, hb_coord]
        gKSF = qk_thr_mma.partition_B(cute.local_tile(mKSF_kdl, sf_tile_k, (None, None, None)))
        tKsKSF, tKgKSF_kdl = tma_partition(
            tma_atom_k_sf,
            0,
            cute.make_layout(1),
            cute.group_modes(sKSF, 0, 3),
            cute.group_modes(gKSF, 0, 3),
        )
        tKsKSF = cute.filter_zeros(tKsKSF)
        tKgKSF_kdl = cute.filter_zeros(tKgKSF_kdl)
        sf_l_k = cute.make_layout(
            mK_kdl.shape[2],
            stride=((0, 1), mK_kdl.shape[2][0][1]),
        )(hb_coord)
        tKgKSF = tKgKSF_kdl[None, None, 0, sf_l_k]
        return (
            (tma_atom_q, tQsQ, tQgQ, tQgQ_qdl, tma_atom_q_sf, tQsQSF, tQgQSF, tQgQSF_qdl, mQ_qdl),
            (tma_atom_k, tKsK, tKgK, tma_atom_k_sf, tKsKSF, tKgKSF),
        )

    def copy_k(self, partitions, kv_coord, k_handle, kv_mcast_mask):
        """K tile kv_coord (multicast) and this CTA's copy of its scale factors into the handle's ring slot."""
        tma_atom_k, tKsK, tKgK, tma_atom_k_sf, tKsKSF, tKgKSF = partitions[1]
        cute.copy(
            tma_atom_k,
            tKgK[None, kv_coord],
            tKsK[None, k_handle.index],
            tma_bar_ptr=k_handle.barrier,
            mcast_mask=kv_mcast_mask,
        )
        cute.copy(
            tma_atom_k_sf,
            tKgKSF[None, kv_coord],
            tKsKSF[None, k_handle.index],
            tma_bar_ptr=k_handle.barrier,
        )

    def mma_setup(self, tiled_mma, tmem_ptr, sQ, sK, sf_smem, sf_layouts):
        """Q'/K' fragments over the staged tiles, the scale-factor slots and their S2T copies (both QKs of a step use
        the one Q slot)."""
        sQSF, sKSF = sf_smem
        q_sf_tmem = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_qsf_offset, dtype=self.sf_dtype),
            blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler_mnk,
                self.sf_vec_size,
                cute.slice_(sf_layouts[0], (None, None, None, 0)),
            ),
        )
        k_sf_tmem = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_ksf_offset, dtype=self.sf_dtype),
            blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler_mnk,
                self.sf_vec_size,
                cute.slice_(sf_layouts[1], (None, None, None, 0)),
            ),
        )
        q_sf = (*s2t_partition(sQSF, q_sf_tmem, self.sf_dtype), q_sf_tmem)
        k_sf = (*s2t_partition(sKSF, k_sf_tmem, self.sf_dtype), k_sf_tmem)
        return tiled_mma, tiled_mma.make_fragment_A(sQ), tiled_mma.make_fragment_B(sK), q_sf, k_sf

    def descs(self, sA, sB):
        """The operand fragments themselves (cute.gemm forms the descriptors per k-block)."""
        return sA, sB

    def issue_qk(self, qk_mma, j, d, descs, si_handle, k0=0):
        """Issue K blocks k0.. of QK_j (the first one overwrites S unless k0 > 0) and commit its S credit."""
        tiled_mma, _, _, q_sf, k_sf = qk_mma[:5]
        sfA, sfB = q_sf[3], k_sf[3]
        sA, sB = descs
        qk = cutlass.new_from_mlir_values(tiled_mma, cutlass.extract_mlir_values(tiled_mma))
        for kb in range(k0, self.k_blocks):
            qk.set(tcgen05.Field.SFA, sfA[(None, None, kb)].iterator)
            qk.set(tcgen05.Field.SFB, sfB[(None, None, kb)].iterator)
            qk.set(tcgen05.Field.ACCUMULATE, kb != 0)
            cute.gemm(qk, d, sA[(None, None, kb)], sB[(None, None, kb)], d)
        si_handle.commit()


class QkK128(QkK64):
    """The K = 64 operands on targets with the K = 128 instruction (sm_107a): the 128 head channels in one K = 128
    block-scaled MMA (SM107MmaMXF4NVF4Op), the compensation channels in one K = 64 MMA at rank 32 (64 channels) and
    in a second K = 128 MMA at rank 64 (128 channels). All read the same K' stages and scale-factor slots: a pair of
    K blocks of the K = 64 layouts lies as a K = 128 operand does.

    The MMAs read Q' from TMEM: the MMA warp copies each Q stage into its slot (eight channels per column, past the
    kernel's 512 columns and within cute.arch.get_max_tmem_alloc_cols of the target: two slots of 24 columns at rank
    32, of 32 at rank 64) once per work tile."""

    q_tmem = True
    tmem_q_offset = 512  # Q' slots 0 and 1 (eight E2M1 codes per column)
    q_stage_bytes = 128 * ROW_ELEMS // 2

    def __init__(self, mma_tiler, rank):
        super().__init__(mma_tiler, rank)
        # K = 128 MMAs of a QK: over every whole pair of K blocks (the K = 64 MMA takes a last single block)
        self.k128_mmas = self.k_blocks // 2
        if self.q_slot(2) > cute.arch.get_max_tmem_alloc_cols(target.arch_name()):
            raise ValueError("the Q' slots exceed the TMEM columns of this target")

    def q_slot(self, j):
        """First TMEM column of Q' slot j."""
        return self.tmem_q_offset + j * (self.head_dim // 8)

    def mma_setup(self, tiled_mma, tmem_ptr, sQ, sK, sf_smem, sf_layouts):
        """The K = 64 scale-factor setup, then the TMEM-A forms of the K = 64 and the K = 128 MMA, their K' fragments
        over the same stages (the stage mode of sK kept: the K ring interleaves with V) and the views of the two Q'
        slots in the K = 64 and the K = 128 layout."""
        qk_mma = super().mma_setup(tiled_mma, tmem_ptr, sQ, sK, sf_smem, sf_layouts)
        cta_group = tiled_mma.op.cta_group
        a_tmem = tcgen05.OperandSource.TMEM
        mma64 = self.make_tiled_mma(cta_group, a_tmem)
        mma128 = k128_mma(self.mma_tiler_mnk[1], a_tmem, cta_group)
        rows = self.mma_tiler_mnk[0]
        thr64, thr128 = mma64.get_slice(0), mma128.get_slice(0)
        q64_layout = thr64.make_fragment_A(thr64.partition_shape_A((rows, self.head_dim))).layout
        q128_layout = thr128.make_fragment_A(
            thr128.partition_shape_A((rows, 128 * self.k128_mmas))
        ).layout
        slots = []
        for j in range(2):
            q = cute.recast_ptr(tmem_ptr + self.q_slot(j), dtype=cutlass.Float4E2M1FN)
            slots.append((cute.make_tensor(q, q64_layout), cute.make_tensor(q, q128_layout)))
        sK128 = k128_view(mma128, self.pad_tiler, sK)
        return (
            mma64,
            None,
            mma64.make_fragment_B(sK),
            qk_mma[3],
            qk_mma[4],
            (mma128, slots, mma128.make_fragment_B(sK128)),
        )

    def q_operand(self, qk_mma, q_stage):
        """None: issue_qk reads Q' slot j."""
        return None

    def k_operand(self, qk_mma, k_stage):
        return super().k_operand(qk_mma, k_stage), qk_mma[5][2][None, None, None, k_stage]

    def q_to_tmem(self, tmem_base, j, sq_addr, q_stage):
        """Q' of Q stage q_stage (sq_addr: the shared address of stage 0) into TMEM slot j: one 128 x 32-byte
        tcgen05.cp per 64-channel K block. The copies run in order with the MMAs issued after them; the caller makes
        sure that no MMA issued before them still reads the slot."""
        taddr = tmem_base + self.q_slot(j)
        q_addr = sq_addr + q_stage * self.q_stage_bytes
        with cute.arch.elect_one():
            for kb in range(self.k_blocks):
                dst = taddr + kb * 8
                desc = smem_desc_sw128(q_addr + kb * 32)
                tcgen05_cp(
                    Tcgen05CpShape.SHAPE_128X256B,
                    make_tmem_ptr(dst, cutlass.Int32),
                    desc,
                    group=CTAGroup.CTA_1,
                )

    def issue_qk(self, qk_mma, j, d, descs, si_handle):
        """The K = 128 MMAs over the K-block pairs (0, 1) and, at rank 64, (2, 3), then the K = 64 MMA over a last
        single block (the K = 64 issue from that block: block 2 at rank 32, none at rank 64); all read Q' slot j."""
        mma128 = qk_mma[5][0]
        sfA, sfB = qk_mma[3][3], qk_mma[4][3]
        (sA, sA128), (sB, sB128) = qk_mma[5][1][j], descs[1]
        qk128 = cutlass.new_from_mlir_values(mma128, cutlass.extract_mlir_values(mma128))
        for i in range(self.k128_mmas):
            qk128.set(tcgen05.Field.SFA, sfA[(None, None, 2 * i)].iterator)
            qk128.set(tcgen05.Field.SFB, sfB[(None, None, 2 * i)].iterator)
            qk128.set(tcgen05.Field.ACCUMULATE, i != 0)
            cute.gemm(qk128, d, sA128[(None, None, i)], sB128[(None, None, i)], d)
        super().issue_qk(qk_mma, j, d, (sA, sB), si_handle, k0=2 * self.k128_mmas)


def for_target(mma_tiler, rank):
    """The QK of the compile target at a compensation rank it takes (one of ranks()): two K = 96 MMAs where the
    target has them, else K = 128 MMAs and a last K = 64 where it has the K = 128 instruction, else three K = 64."""
    if target.has_k96_fp4_mma():
        return QkK96(mma_tiler, rank)
    if target.has_k128_fp4_mma():
        return QkK128(mma_tiler, rank)
    return QkK64(mma_tiler, rank)
