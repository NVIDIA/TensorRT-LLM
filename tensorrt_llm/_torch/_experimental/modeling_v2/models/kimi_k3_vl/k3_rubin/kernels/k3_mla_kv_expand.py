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
"""Kimi-K3 MLA context KV expansion: the FP8 ``kv_b_proj`` GEMM writing K and V in place.

MLA context attention with cached KV re-expands the latent cache of every cached token:
``[k_nope | v] = compressed_kv @ dequant(W_kv_b)^T`` (``W_kv_b`` is ``[H * (nope + v), rank]``,
the UE8M0-resmoothed FP8 weight with its DeepGEMM-packed scale), then
``k = cat(k_nope, broadcast(k_pe))`` per head. The TRT-LLM attention op reads

* ``k`` as a contiguous BF16 ``[T, H, nope + rope]`` and
* ``v`` as BF16 ``[T, H, v]`` with the row stride of the GEMM output, ``H * (nope + v)``.

:class:`K3MlaKvExpandKernel` is the GEMM of :mod:`k3_latent_fp8_gemm` (same activation
quantization, same FP32 accumulation chain over K as DeepGEMM, BF16 rounding) with an epilogue
that stores the ``k_nope`` columns of head ``h`` straight into ``k[:, h, :nope]``, the ``v``
columns into the ``v`` view, and copies ``k_pe`` (bit for bit) into ``k[:, h, nope:]`` of every
head. It replaces DeepGEMM's ``[T, H * (nope + v)]`` output plus the ``[T, H, nope + rope]`` cat:
K and V are written once and never re-read before the attention op.

Epilogue: ``epi_warps`` warps (4 or 8, two per TMEM lane quarter splitting the tile's columns
into contiguous halves). A thread owns one tile row and reads 32-column TMEM blocks; each warp
stages its 32 x 32 block in shared memory and stores 64 contiguous bytes per row with four
lanes (8 rows per instruction). The rope slots of the heads that start in a warp's columns are
written before the tile's accumulator is ready, overlapping the mainloop.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import dsl_user_op

from .k3_latent_fp8_gemm import (
    _RES_ROW_BYTES,
    _RES_WARP_BYTES,
    _SMEM_CAPACITY,
    BM,
    CTA_K,
    EPI_RESIDUAL,
    SUPPORTED_BN,
    K3Fp8LatentGemmKernel,
    _asm,
    _ceil_div,
    _f32x2_to_bf16x2,
    _iv,
    _ldg_v4_b32,
    _lds_v4_b32,
    _min_s32,
    _num_sms,
    _quant_bf16x8_to_e4m3,
    _stg_v4_b32,
    _sts_v4_b32,
    quantize_act_1x128,
    smem_bytes,
)
from .sm_reserve import sm_reserve

#: Per-head dimensions the epilogue supports (Kimi-K3 / DeepSeek MLA).
NOPE_DIM = 128
ROPE_DIM = 64
V_DIM = 128
#: Default column tile: two accumulators fit in TMEM, so a tile's epilogue overlaps the next
#: tile's mainloop.
DEFAULT_BN = 192
DEFAULT_EPI_WARPS = 8
#: Row tiles per group of the tile order (see K3MlaKvExpandKernel.tile_coords).
DEFAULT_M_GROUP = 64
#: FP8 output: row stride of the staged 32-byte (32 x E4M3) block rows, conflict-free stores.
_FP8_ROW_BYTES = 48
#: Packed BF16x2 1.0: the E4M3 conversion of the FP8 epilogue uses unit scale.
_BF16X2_ONE = 0x3F803F80


@dsl_user_op
def _stg_v2_b32(addr, v0, v1, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), _iv(v0, loc=loc, ip=ip), _iv(v1, loc=loc, ip=ip)],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        loc=loc,
        ip=ip,
    )


def expand_smem_bytes(bn: int, stages: int, epi_warps: int) -> int:
    """Host-side mirror of the kernel's shared-memory footprint."""
    return smem_bytes(bn, stages, EPI_RESIDUAL) + (epi_warps - 4) * _RES_WARP_BYTES


def expand_max_stages(bn: int, epi_warps: int) -> int:
    s = 1
    while expand_smem_bytes(bn, s + 1, epi_warps) <= _SMEM_CAPACITY:
        s += 1
    return s


class K3MlaKvExpandKernel(K3Fp8LatentGemmKernel):
    """MXFP8 ``kv_b_proj`` GEMM whose epilogue writes the attention op's K and V layouts.

    Launch pointers (see :meth:`K3Fp8LatentGemmKernel.__call__`): ``out0`` = K
    ``[T, H * (nope + rope)]``, ``out1`` = the first ``v`` column of row 0 (row stride
    ``H * (nope + v)``), ``aux0`` = ``k_pe`` ``[T, rope]`` with row stride ``kpe_stride``.
    """

    def __init__(
        self,
        num_heads: int,
        k: int,
        bn: int,
        stages: int,
        kpe_stride: int,
        epi_warps: int,
        use_pdl: bool,
        fp8_out: bool = False,
        m_group: int = 0,
    ):
        if epi_warps not in (4, 8):
            raise ValueError(f"epi_warps must be 4 or 8, got {epi_warps}")
        if expand_smem_bytes(bn, stages, epi_warps) > _SMEM_CAPACITY:
            raise ValueError(f"{stages} stages of bn={bn} exceed shared memory")
        # The base class checks its own (4-warp) footprint; ours is checked above.
        super().__init__(num_heads * (NOPE_DIM + V_DIM), k, bn, 1, EPI_RESIDUAL, use_pdl)
        self.stages = stages
        self.prefetch = min(stages, self.kt_total)
        self.epi_warps = epi_warps
        self.staging_bytes = epi_warps * _RES_WARP_BYTES
        # Column blocks (32 wide) per epilogue warp.
        self.warp_blocks = (bn // 32) * 4 // epi_warps
        self.num_heads = num_heads
        self.nope_cols = num_heads * NOPE_DIM
        self.k_head = NOPE_DIM + ROPE_DIM
        self.k_row = num_heads * self.k_head
        self.kpe_stride = kpe_stride
        # BF16 output: V is a view of the [T, H * (nope + v)] buffer the attention op expects.
        # FP8 output (unit scale, the attention op's FP8 conversion of the BF16 values): K and V
        # are both contiguous.
        self.fp8_out = fp8_out
        self.m_group = m_group
        self.v_row = num_heads * (V_DIM if fp8_out else NOPE_DIM + V_DIM)

    @cute.jit
    def tile_coords(self, t: Int32, m_tiles: Int32):
        """Grouped order: ``m_group`` row tiles x every column tile, column-major inside a group.

        The CTAs running together share a few weight tiles and ``m_group`` activation tiles,
        so both stay L2-resident while the output streams through; ``m_group = 0`` keeps the
        plain column-major order.
        """
        n_tile = t // m_tiles
        m_tile = t - n_tile * m_tiles
        if cutlass.const_expr(self.m_group > 0):
            group_tiles = self.m_group * self.n_tiles
            g = t // group_tiles
            r = t - g * group_tiles
            m_base = g * self.m_group
            gm = _min_s32(m_tiles - m_base, self.m_group)
            n_tile = r // gm
            m_tile = m_base + (r - n_tile * gm)
        return n_tile, m_tile

    @cute.jit
    def epilogue(
        self,
        tiled_mma: cute.TiledMma,
        storage,
        out0_ptr: cute.Pointer,
        out1_ptr: cute.Pointer,
        aux0_ptr: cute.Pointer,
        aux1_ptr: cute.Pointer,
        m: Int32,
        m_tiles: Int32,
        q: Int32,
    ):
        """Epilogue warp ``ew = q // 32`` owns tile rows ``32 (ew % 4) ..`` and, with 8 warps,
        the column half ``ew // 4``."""
        acc_full = storage.acc_full.data_ptr()
        acc_empty = storage.acc_empty.data_ptr()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        num_tiles = m_tiles * self.n_tiles
        ew = q // 32
        lane = q % 32
        wrow0 = (ew % 4) * 32
        blk0 = (ew // 4) * self.warp_blocks
        staging = storage.sOut.data_ptr().toint() + ew * _RES_WARP_BYTES
        cute.arch.mbarrier_wait(storage.tmem_ready, 0)
        acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
            cutlass.Float32, alignment=16, ptr_to_buffer_holding_addr=storage.tmem_holding_buf
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        acc_layout = tiled_mma.make_fragment_C(acc_shape).layout
        tAcc0 = cute.make_tensor(acc_tmem_ptr, acc_layout)[((None, None), 0, 0)]
        cAcc = cute.make_identity_tensor((BM, self.bn))
        blk = 32
        tiled_t2r = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(blk)), cutlass.Float32),
            cute.local_tile(tAcc0, (BM, blk), (0, 0)),
        )
        # Both warps of a lane quarter read the same TMEM lanes (different columns).
        thr_t2r = tiled_t2r.get_slice(q % BM)
        tTR_cAcc0 = thr_t2r.partition_D(cute.local_tile(cAcc, (BM, blk), (0, 0)))
        rBuf = [cute.make_rmem_tensor(tTR_cAcc0.shape, cutlass.Float32) for _ in range(2)]
        nwb = self.warp_blocks

        ti = Int32(0)
        for t in cutlass.range(bidx, num_tiles, gdim, unroll=1):
            n_tile, m_tile = self.tile_coords(t, m_tiles)
            buf = ti % self.acc_bufs
            tAcc = cute.make_tensor(acc_tmem_ptr + buf * self.bn, acc_layout)[((None, None), 0, 0)]
            col0 = n_tile * self.bn + blk0 * blk
            # Rope slots of the heads starting in this warp's columns; independent of the
            # accumulator, so they overlap the mainloop.
            for j in cutlass.range_constexpr(nwb):
                col = col0 + j * blk
                if col < self.nope_cols and (col & (NOPE_DIM - 1)) == 0:
                    if cutlass.const_expr(self.fp8_out):
                        self._store_rope_fp8(
                            out0_ptr, aux0_ptr, m_tile, col // NOPE_DIM, m, wrow0, lane
                        )
                    else:
                        self._store_rope(
                            out0_ptr, aux0_ptr, m_tile, col // NOPE_DIM, m, wrow0, lane
                        )
            cute.arch.mbarrier_wait(acc_full + buf, (ti // self.acc_bufs) & 1)
            # 2-deep register ring: each block's TMEM load overlaps the previous block's stores.
            cute.copy(
                tiled_t2r,
                thr_t2r.partition_S(cute.local_tile(tAcc, (BM, blk), (0, blk0))),
                rBuf[0],
            )
            for j in cutlass.range_constexpr(nwb):
                cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(j + 1 < nwb):
                    cute.copy(
                        tiled_t2r,
                        thr_t2r.partition_S(cute.local_tile(tAcc, (BM, blk), (0, blk0 + j + 1))),
                        rBuf[(j + 1) % 2],
                    )
                self._store_block(
                    rBuf[j % 2], staging, out0_ptr, out1_ptr, m_tile, col0 + j * blk, m, wrow0, lane
                )
            # Every TMEM load of this buffer has completed (fence above); release it.
            cute.arch.fence_view_async_tmem_load()
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(acc_empty + buf)
            ti = ti + 1
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(storage.tmem_free)

    @cute.jit
    def _store_block(self, rAcc, staging, k_ptr, v_ptr, m_tile, col, m, wrow0, lane):
        if cutlass.const_expr(self.fp8_out):
            self._store_kv_block_fp8(rAcc, staging, k_ptr, v_ptr, m_tile, col, m, wrow0, lane)
        else:
            self._store_kv_block(rAcc, staging, k_ptr, v_ptr, m_tile, col, m, wrow0, lane)

    @cute.jit
    def _store_rope(self, k_ptr, kpe_ptr, m_tile, head, m, wrow0, lane):
        """Copy the warp's 32 ``k_pe`` rows into ``k[:, head, nope:]``: 8 lanes per 128-byte row."""
        r0 = m_tile * BM + wrow0 + lane // 8
        chunk = (lane % 8) * 8
        words = []
        for i in cutlass.range_constexpr(8):
            r = _min_s32(r0 + 4 * i, m - 1)
            words.append(_ldg_v4_b32(kpe_ptr.toint() + (Int64(r) * self.kpe_stride + chunk) * 2))
        for i in cutlass.range_constexpr(8):
            r = r0 + 4 * i
            if r < m:
                dst = Int64(r) * self.k_row + head * self.k_head + NOPE_DIM + chunk
                _stg_v4_b32(k_ptr.toint() + dst * 2, *words[i])

    @cute.jit
    def _store_kv_block(self, rAcc, staging, k_ptr, v_ptr, m_tile, col, m, wrow0, lane):
        """One 32-column block, BF16-rounded, to K (``k_nope`` columns) or V.

        The thread writes its row into the warp's SMEM staging rows; after a warp sync every 4
        lanes store 64 contiguous bytes of one row (rows ``8 i + lane // 4`` of the warp).
        """
        words = []
        for i in cutlass.range_constexpr(16):
            words.append(_f32x2_to_bf16x2(rAcc[2 * i], rAcc[2 * i + 1]))
        for v in cutlass.range_constexpr(4):
            _sts_v4_b32(staging + lane * _RES_ROW_BYTES + v * 16, *words[4 * v : 4 * v + 4])
        cute.arch.sync_warp()
        r0 = m_tile * BM + wrow0 + lane // 4
        c = col + (lane % 4) * 8
        is_k = col < self.nope_cols
        k_col = (c // NOPE_DIM) * self.k_head + (c & (NOPE_DIM - 1))
        v_col = c - self.nope_cols
        for i in cutlass.range_constexpr(4):
            t = _lds_v4_b32(staging + (lane // 4 + 8 * i) * _RES_ROW_BYTES + (lane % 4) * 16)
            r = r0 + 8 * i
            if r < m:
                if is_k:
                    _stg_v4_b32(k_ptr.toint() + (Int64(r) * self.k_row + k_col) * 2, *t)
                else:
                    _stg_v4_b32(v_ptr.toint() + (Int64(r) * self.v_row + v_col) * 2, *t)
        # The staging rows are rewritten by the next block.
        cute.arch.sync_warp()

    @cute.jit
    def _store_rope_fp8(self, k_ptr, kpe_ptr, m_tile, head, m, wrow0, lane):
        """FP8 ``k[:, head, nope:]`` of the warp's 32 rows: 8 lanes per 64-byte row."""
        r0 = m_tile * BM + wrow0 + lane // 8
        chunk = (lane % 8) * 8
        words = []
        for i in cutlass.range_constexpr(8):
            r = _min_s32(r0 + 4 * i, m - 1)
            words.append(_ldg_v4_b32(kpe_ptr.toint() + (Int64(r) * self.kpe_stride + chunk) * 2))
        for i in cutlass.range_constexpr(8):
            r = r0 + 4 * i
            if r < m:
                p0, p1 = _quant_bf16x8_to_e4m3(*words[i], Int32(_BF16X2_ONE))
                dst = Int64(r) * self.k_row + head * self.k_head + NOPE_DIM + chunk
                _stg_v2_b32(k_ptr.toint() + dst, p0, p1)

    @cute.jit
    def _store_kv_block_fp8(self, rAcc, staging, k_ptr, v_ptr, m_tile, col, m, wrow0, lane):
        """One 32-column block -> BF16 -> E4M3 to K or V; 2 lanes per 32-byte row segment."""
        words = []
        for i in cutlass.range_constexpr(16):
            words.append(_f32x2_to_bf16x2(rAcc[2 * i], rAcc[2 * i + 1]))
        packed = []
        for v in cutlass.range_constexpr(4):
            p0, p1 = _quant_bf16x8_to_e4m3(*words[4 * v : 4 * v + 4], Int32(_BF16X2_ONE))
            packed.extend((p0, p1))
        _sts_v4_b32(staging + lane * _FP8_ROW_BYTES, *packed[:4])
        _sts_v4_b32(staging + lane * _FP8_ROW_BYTES + 16, *packed[4:])
        cute.arch.sync_warp()
        r0 = m_tile * BM + wrow0 + lane // 2
        c = col + (lane % 2) * 16
        is_k = col < self.nope_cols
        k_col = (c // NOPE_DIM) * self.k_head + (c & (NOPE_DIM - 1))
        v_col = c - self.nope_cols
        for i in cutlass.range_constexpr(2):
            t = _lds_v4_b32(staging + (lane // 2 + 16 * i) * _FP8_ROW_BYTES + (lane % 2) * 16)
            r = r0 + 16 * i
            if r < m:
                if is_k:
                    _stg_v4_b32(k_ptr.toint() + Int64(r) * self.k_row + k_col, *t)
                else:
                    _stg_v4_b32(v_ptr.toint() + Int64(r) * self.v_row + v_col, *t)
        cute.arch.sync_warp()


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


@functools.cache
def _compiled_expand(
    num_heads: int,
    k: int,
    bn: int,
    stages: int,
    kpe_stride: int,
    epi_warps: int,
    use_pdl: bool,
    fp8_out: bool,
    m_group: int,
):
    kernel = K3MlaKvExpandKernel(
        num_heads, k, bn, stages, kpe_stride, epi_warps, use_pdl, fp8_out=fp8_out, m_group=m_group
    )
    dummy = 1 << 20
    ptrs = [
        make_ptr(dtype, dummy, cute.AddressSpace.gmem, assumed_align=16)
        for dtype in (cutlass.Float8E4M3FN, cutlass.Uint8, cutlass.Float8E4M3FN)
        + (cutlass.Uint8,) * 5
    ]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute.compile(kernel, *ptrs, Int32(1), Int32(1), stream)


def _gmem_ptr(dtype, tensor: torch.Tensor):
    return make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


def check_operands(
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    num_heads: int,
) -> None:
    """Raise ``ValueError`` unless the operands match the kernel's layout contract."""
    if compressed_kv.dtype != torch.bfloat16 or compressed_kv.dim() != 2:
        raise ValueError("compressed_kv must be a 2D bfloat16 tensor")
    if not compressed_kv.is_contiguous():
        raise ValueError("compressed_kv must be contiguous")
    t, rank = compressed_kv.shape
    if rank % CTA_K:
        raise ValueError(f"kv_lora_rank {rank} must be a multiple of {CTA_K}")
    n = num_heads * (NOPE_DIM + V_DIM)
    if w.dtype != torch.float8_e4m3fn or tuple(w.shape) != (n, rank) or not w.is_contiguous():
        raise ValueError(f"w must be a contiguous float8_e4m3fn [{n}, {rank}] tensor")
    words = _ceil_div(rank // CTA_K, 4)
    if (
        w_scale.dtype != torch.int32
        or tuple(w_scale.shape) != (n, words)
        or w_scale.stride(0) != 1
        or (words > 1 and w_scale.stride(1) != n)
    ):
        raise ValueError(
            f"w_scale must be DeepGEMM-packed UE8M0 int32 [{n}, {words}] with strides (1, {n})"
        )
    if (
        k_pe.dtype != torch.bfloat16
        or k_pe.dim() != 2
        or tuple(k_pe.shape) != (t, ROPE_DIM)
        or k_pe.stride(1) != 1
        or (k_pe.stride(0) * 2) % 16
    ):
        raise ValueError(f"k_pe must be bfloat16 [{t}, {ROPE_DIM}] with 16-byte aligned rows")
    k_shape = (t, num_heads * (NOPE_DIM + ROPE_DIM))
    out_dtype = k_out.dtype
    if out_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError("k_out must be bfloat16 or float8_e4m3fn")
    if tuple(k_out.shape) != k_shape or not k_out.is_contiguous():
        raise ValueError(f"k_out must be a contiguous {k_shape} tensor")
    # BF16: the [T, H * (nope + v)] buffer whose V half is written. FP8: contiguous [T, H * v].
    v_shape = (t, n) if out_dtype == torch.bfloat16 else (t, num_heads * V_DIM)
    if v_out.dtype != out_dtype or tuple(v_out.shape) != v_shape or not v_out.is_contiguous():
        raise ValueError(f"v_out must be a contiguous {out_dtype} {v_shape} tensor")
    for name, tensor in (
        ("compressed_kv", compressed_kv),
        ("k_pe", k_pe),
        ("w", w),
        ("k_out", k_out),
        ("v_out", v_out),
    ):
        if tensor.data_ptr() % 16:
            raise ValueError(f"{name} must be 16-byte aligned")


def run_k3_mla_kv_expand(
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    num_heads: int,
    *,
    bn: int = DEFAULT_BN,
    stages: int | None = None,
    epi_warps: int = DEFAULT_EPI_WARPS,
    m_group: int = DEFAULT_M_GROUP,
    num_ctas: int | None = None,
    use_pdl: bool = True,
) -> None:
    """Expand ``compressed_kv`` / ``k_pe`` into ``k_out`` and ``v_out``.

    ``k_out`` (``[T, H * (nope + rope)]``) receives ``cat(k_nope, k_pe)`` per head, bit-identical
    to ``fp8_swap_ab_gemm(compressed_kv, w, w_scale)`` split into ``k_nope | v`` and
    concatenated with ``k_pe``. BF16 outputs: ``v_out`` is the ``[T, H * (nope + v)]`` buffer
    whose second half receives ``v`` (the first half is left untouched). FP8 outputs:
    ``v_out`` is ``[T, H * v]`` and every value is the E4M3 conversion (unit scale, RN,
    satfinite) of the BF16 value above. The persistent grid leaves
    ``TRTLLM_CUTEDSL_SM_RESERVE`` SMs free unless ``num_ctas`` is given.
    """
    check_operands(compressed_kv, k_pe, w, w_scale, k_out, v_out, num_heads)
    if bn not in SUPPORTED_BN:
        raise ValueError(f"bn must be one of {SUPPORTED_BN}, got {bn}")
    m, k = compressed_kv.shape
    if m == 0:
        return
    n = w.shape[0]
    fp8_out = k_out.dtype == torch.float8_e4m3fn
    if stages is None:
        stages = expand_max_stages(bn, epi_warps)
    m_tiles = _ceil_div(m, BM)
    num_tiles = m_tiles * _ceil_div(n, bn)
    if num_ctas is None:
        num_ctas = max(1, _num_sms() - sm_reserve())
    num_ctas = max(1, min(num_tiles, num_ctas))
    xq = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=compressed_kv.device)
    xs = torch.empty((k // CTA_K) * m_tiles * 512, dtype=torch.uint8, device=compressed_kv.device)
    quantize_act_1x128(compressed_kv, xq, xs, use_pdl=use_pdl)
    stream = cuda.CUstream(torch.cuda.current_stream(compressed_kv.device).cuda_stream)
    v_view = v_out if fp8_out else v_out[:, num_heads * NOPE_DIM :]
    _compiled_expand(
        num_heads, k, bn, stages, k_pe.stride(0), epi_warps, use_pdl, fp8_out, m_group
    )(
        _gmem_ptr(cutlass.Float8E4M3FN, xq),
        _gmem_ptr(cutlass.Uint8, xs),
        _gmem_ptr(cutlass.Float8E4M3FN, w),
        *(_gmem_ptr(cutlass.Uint8, x) for x in (w_scale, k_out, v_view, k_pe, k_pe)),
        Int32(m),
        Int32(num_ctas),
        stream,
    )


@torch.library.custom_op("trtllm::k3_mla_kv_expand", mutates_args=("k_out", "v_out"))
def k3_mla_kv_expand(
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    num_heads: int,
) -> None:
    """Custom-op wrapper of :func:`run_k3_mla_kv_expand` with the default tiling."""
    run_k3_mla_kv_expand(compressed_kv, k_pe, w, w_scale, k_out, v_out, num_heads)


@k3_mla_kv_expand.register_fake
def _(compressed_kv, k_pe, w, w_scale, k_out, v_out, num_heads) -> None:
    return None


__all__ = [
    "DEFAULT_BN",
    "DEFAULT_EPI_WARPS",
    "K3MlaKvExpandKernel",
    "NOPE_DIM",
    "ROPE_DIM",
    "V_DIM",
    "check_operands",
    "k3_mla_kv_expand",
    "run_k3_mla_kv_expand",
]
