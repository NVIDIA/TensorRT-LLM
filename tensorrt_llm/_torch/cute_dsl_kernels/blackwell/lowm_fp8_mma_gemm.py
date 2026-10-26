# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warp-MMA GEMM for decode-sized M over MXFP8 weights (BF16 activations, no activation quant).

``C[M, N] = A[M, K] (bf16) @ dequant(B[N, K] e4m3, S[N, K/32] ue8m0).T``

Layout of the work:

* A CTA owns 8 output columns (one ``m16n8`` tile column) and splits K over its ``k_warps``
  warps; warps reduce through shared memory at the end.
* K is processed in 64-element chunks. Inside a chunk the reduction order is
  permuted so that lane ``(r, q) = (lane // 4, lane % 4)`` loads one contiguous
  16-byte weight vector ``B[n0 + r, k + 16q : k + 16q + 16]`` and one 32-byte
  activation vector per row, and those registers *are* the fragments of four
  ``mma.sync.m16n8k16`` steps (logical k ``2q + t`` and ``2q + 8 + t`` of step
  ``j`` map to physical ``16q + 4j + t`` and ``16q + 4j + 2 + t``; the same
  bijection is used for A and B, so the dot product is unchanged).
* E4M3 weights times their power-of-two UE8M0 scale convert exactly to BF16.
* All weight vectors and scales of a warp are loaded before
  ``griddepcontrol.wait`` (programmatic dependent launch), activations after.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Uint8, Uint32, const_expr
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import SmemAllocator

from tensorrt_llm.math_utils import ceil_div

from .cute_ptx_utils import mma_sync
from .utils import griddepcontrol_launch_dependents, griddepcontrol_wait

GROUP = 32  # K elements per UE8M0 scale
CHUNK = 64  # K elements per quad of lanes (16 per lane)
N_TILE = 8
ROPE_DIM = 64  # rotated columns at the end of every head of the RoPE epilogue


@dsl_user_op
def _fp8x4_to_bf16x4_scaled(x: Uint32, scale: Float32, *, loc=None, ip=None) -> cute.TensorSSA:
    """Four packed E4M3 times a power-of-two scale -> four BF16 (exact), as two ``Uint32``."""
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [x.ir_value(loc=loc, ip=ip), scale.ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b16 lo, hi, h0, h1, h2, h3;\n\t"
        ".reg .b32 p01, p23;\n\t"
        ".reg .f32 f0, f1, f2, f3;\n\t"
        "mov.b32 {lo, hi}, $2;\n\t"
        "cvt.rn.f16x2.e4m3x2 p01, lo;\n\t"
        "cvt.rn.f16x2.e4m3x2 p23, hi;\n\t"
        "mov.b32 {h0, h1}, p01;\n\t"
        "mov.b32 {h2, h3}, p23;\n\t"
        "cvt.f32.f16 f0, h0;\n\t"
        "cvt.f32.f16 f1, h1;\n\t"
        "cvt.f32.f16 f2, h2;\n\t"
        "cvt.f32.f16 f3, h3;\n\t"
        "mul.f32 f0, f0, $3;\n\t"
        "mul.f32 f1, f1, $3;\n\t"
        "mul.f32 f2, f2, $3;\n\t"
        "mul.f32 f3, f3, $3;\n\t"
        "cvt.rn.bf16x2.f32 $0, f1, f0;\n\t"
        "cvt.rn.bf16x2.f32 $1, f3, f2;\n\t"
        "}\n",
        "=r,=r,r,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    vec = vector.from_elements(
        ir.VectorType.get([2], T.i32(), loc=loc),
        [llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip) for i in range(2)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(vec, 2, Uint32)


class LowMFp8MmaGemmKernel:
    """``num_rows`` (<= 64) x K x N MXFP8 GEMM, one 8-column tile per CTA."""

    def __init__(
        self,
        *,
        num_rows: int,
        k_extent: int,
        k_warps: int,
        use_pdl: bool,
        split_n: int = 0,
        scale_block128: bool = False,
        group_n: int = 0,
        rope_head_dim: int = 0,
        scale_swizzled: bool = False,
        bf16_weights: bool = False,
        a_rope_head_dim: int = 0,
    ) -> None:
        """``scale_block128``: S is FP32 ``[N/128, K/128]`` power-of-two multipliers instead of
        UE8M0 ``[N, K/32]``. ``group_n``: B stacks groups of ``group_n`` rows, and group ``g``
        multiplies columns ``[g*K, (g+1)*K)`` of A (a grouped GEMM over ``A[M, G*K]``).
        ``rope_head_dim``: apply interleaved RoPE in FP32 to the last ``ROPE_DIM`` columns of
        every ``rope_head_dim``-wide head before the store (cos/sin table ``[P, 2, ROPE_DIM/2]``
        indexed by the row's position), like ``mla_rope_inplace`` on the output.
        ``scale_swizzled``: S is the UE8M0 ``[N, K/32]`` scales in the CUTLASS 128x4 swizzle (1-D),
        as an MXFP8 ``Linear`` stores them. ``bf16_weights``: B is BF16 ``[N, K]`` (as bytes
        ``[N, 2K]``) without scales. ``a_rope_head_dim`` (grouped only): rotate the last
        ``ROPE_DIM`` columns of every ``a_rope_head_dim``-wide head of A back by their angle as A
        is loaded (interleaved inverse RoPE in FP32, rounded to BF16, as
        ``mla_rope_inplace(inverse=True)``)."""
        if k_extent % (CHUNK * k_warps):
            raise ValueError(f"K={k_extent} must be a multiple of {CHUNK * k_warps}")
        if not 1 <= num_rows <= 64:
            raise ValueError(f"num_rows={num_rows} must be in [1, 64]")
        if split_n % N_TILE:
            raise ValueError(f"split_n={split_n} must be a multiple of {N_TILE}")
        self.num_rows = num_rows
        self.row_tiles = ceil_div(num_rows, 16)
        self.k_extent = k_extent
        self.k_warps = k_warps
        self.chunks_per_warp = k_extent // (CHUNK * k_warps)
        self.use_pdl = use_pdl
        self.split_n = split_n
        self.scale_block128 = scale_block128
        self.group_n = group_n
        self.rope_head_dim = rope_head_dim
        if rope_head_dim and (rope_head_dim % N_TILE or ROPE_DIM > rope_head_dim):
            raise ValueError(f"unsupported rope head dim {rope_head_dim}")
        if group_n % N_TILE:
            raise ValueError(f"group_n={group_n} must be a multiple of {N_TILE}")
        self.scale_swizzled = scale_swizzled
        self.sf_k_tiles = ceil_div(k_extent // GROUP, 4)
        self.bf16_weights = bf16_weights
        # 16-byte weight vectors per lane and K chunk.
        self.weight_vectors = 2 if bf16_weights else 1
        self.a_rope_head_dim = a_rope_head_dim
        # Each warp's K range must hold whole heads, so a head's last chunk is a warp's own.
        if a_rope_head_dim and (
            not group_n
            or rope_head_dim
            or a_rope_head_dim % CHUNK
            or (k_extent // k_warps) % a_rope_head_dim
        ):
            raise ValueError(f"unsupported A RoPE geometry: head {a_rope_head_dim}, K {k_extent}")

    @cute.jit
    def __call__(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gS: cute.Tensor,
        gC: cute.Tensor,
        gC2: cute.Tensor,
        gPos: cute.Tensor,
        gCosSin: cute.Tensor,
        stream,
    ) -> None:
        n = cute.size(gB, mode=[0])
        copy_a = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            BFloat16,
            num_bits_per_copy=128,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        copy_b = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            Uint8,
            num_bits_per_copy=128,
            load_cache_mode=cute.nvgpu.LoadCacheMode.STREAMING,
        )
        self.kernel(gA, gB, gS, gC, gC2, gPos, gCosSin, copy_a, copy_b).launch(
            grid=[cute.ceil_div(n, N_TILE), 1, 1],
            block=[32 * self.k_warps, 1, 1],
            smem=self.k_warps * self.row_tiles * 16 * N_TILE * 4,
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gS: cute.Tensor,
        gC: cute.Tensor,
        gC2: cute.Tensor,
        gPos: cute.Tensor,
        gCosSin: cute.Tensor,
        copy_a: cute.CopyAtom,
        copy_b: cute.CopyAtom,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32
        r = lane // 4
        q = lane % 4

        num_rows: cutlass.Constexpr = self.num_rows
        row_tiles: cutlass.Constexpr = self.row_tiles
        k_warps: cutlass.Constexpr = self.k_warps
        chunks: cutlass.Constexpr = self.chunks_per_warp

        n0 = bidx * N_TILE
        k_warp0 = warp * (chunks * CHUNK)

        # ---- weights + scales (independent of the producer kernel) ----
        gB_vec = cute.logical_divide(gB, (None, 16))
        vectors: cutlass.Constexpr = self.weight_vectors
        b_regs = cute.make_rmem_tensor(
            cute.make_layout((chunks, 16 * vectors), stride=(16 * vectors, 1)), Uint8
        )
        s_bits = cute.make_rmem_tensor(cute.make_layout((chunks,)), Int32)
        s_regs = cute.recast_tensor(s_bits, Float32)
        b_vecs = cute.logical_divide(b_regs, (None, 16))
        b_row = n0 + r
        for c in cutlass.range_constexpr(chunks):
            k0 = k_warp0 + c * CHUNK + 16 * q
            if const_expr(self.bf16_weights):
                for v in cutlass.range_constexpr(vectors):
                    cute.copy(copy_b, gB_vec[b_row, (None, k0 // 8 + v)], b_vecs[c, (None, v)])
            else:
                cute.copy(copy_b, gB_vec[b_row, (None, k0 // 16)], b_regs[c, None])
            if const_expr(self.bf16_weights):
                pass
            elif const_expr(self.scale_block128):
                s_regs[c] = gS[b_row // 128, k0 // 128]
            elif const_expr(self.scale_swizzled):
                # 128-row tiles of [K/128 blocks, 32 rows, 4 row groups, 4 scales].
                block = k0 // GROUP
                offset = (
                    (b_row // 128) * (self.sf_k_tiles * 512)
                    + (b_row % 32) * 16
                    + (b_row % 128) // 32 * 4
                    + block // 4 * 512
                    + block % 4
                )
                s_bits[c] = gS[offset].to(Int32) << 23
            else:
                # UE8M0 -> the bit pattern of 2^(s - 127).
                s_bits[c] = gS[b_row, k0 // GROUP].to(Int32) << 23

        # A's inverse RoPE: positions of rows 16t + r and 16t + r + 8, final before this launch.
        rope_pos = cute.make_rmem_tensor(
            cute.make_layout((self.row_tiles, 2), stride=(2, 1)), Int32
        )
        if const_expr(self.a_rope_head_dim > 0):
            rope_pos.fill(0)
            for t in cutlass.range_constexpr(self.row_tiles):
                for h in cutlass.range_constexpr(2):
                    if 16 * t + r + 8 * h < self.num_rows:
                        rope_pos[t, h] = gPos[16 * t + r + 8 * h]

        if const_expr(self.use_pdl):
            # The dependent grid is released once the weight loads are issued; it still waits for
            # this grid's completion before reading the output.
            griddepcontrol_launch_dependents()
            griddepcontrol_wait()

        gA_vec = cute.logical_divide(gA, (None, 8))
        acc = cute.make_rmem_tensor(cute.make_layout((row_tiles, 4), stride=(4, 1)), Float32)
        acc.fill(0.0)
        # Rows 16t + r ("lo") and 16t + r + 8 ("hi"); k 16q + 0..7 (0) and 16q + 8..15 (1).
        a_lo0 = cute.make_rmem_tensor(cute.make_layout((8,)), BFloat16)
        a_lo1 = cute.make_rmem_tensor(cute.make_layout((8,)), BFloat16)
        a_hi0 = cute.make_rmem_tensor(cute.make_layout((8,)), BFloat16)
        a_hi1 = cute.make_rmem_tensor(cute.make_layout((8,)), BFloat16)
        a_lo0_u32 = cute.recast_tensor(a_lo0, Uint32)
        a_lo1_u32 = cute.recast_tensor(a_lo1, Uint32)
        a_hi0_u32 = cute.recast_tensor(a_hi0, Uint32)
        a_hi1_u32 = cute.recast_tensor(a_hi1, Uint32)
        a_frag = cute.make_rmem_tensor(cute.make_layout((8,)), BFloat16)
        a_frag_u32 = cute.recast_tensor(a_frag, Uint32)
        b_frag = cute.make_rmem_tensor(cute.make_layout((4,)), BFloat16)
        b_frag_u32 = cute.recast_tensor(b_frag, Uint32)
        tile = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
        w_bf16 = cute.make_rmem_tensor(cute.make_layout((8,)), Uint32)
        a_col0 = 0
        if const_expr(self.group_n > 0):
            a_col0 = (n0 // self.group_n) * self.k_extent

        for c in cutlass.range_constexpr(chunks):
            k0 = k_warp0 + c * CHUNK + 16 * q
            ka = a_col0 + k0
            self._convert_weights(w_bf16, b_regs, s_regs, c)
            for t in cutlass.range_constexpr(row_tiles):
                row_lo = 16 * t + r
                row_hi = 16 * t + r + 8
                a_lo0_u32.fill(0)
                a_lo1_u32.fill(0)
                a_hi0_u32.fill(0)
                a_hi1_u32.fill(0)
                if row_lo < num_rows:
                    cute.copy(copy_a, gA_vec[row_lo, (None, ka // 8)], a_lo0)
                    cute.copy(copy_a, gA_vec[row_lo, (None, ka // 8 + 1)], a_lo1)
                if const_expr(16 * t + 8 < num_rows):
                    if row_hi < num_rows:
                        cute.copy(copy_a, gA_vec[row_hi, (None, ka // 8)], a_hi0)
                        cute.copy(copy_a, gA_vec[row_hi, (None, ka // 8 + 1)], a_hi1)
                if const_expr(
                    self.a_rope_head_dim > 0 and ((c + 1) * CHUNK) % self.a_rope_head_dim == 0
                ):
                    # The head's last CHUNK columns: lane q holds rotation pairs 8q .. 8q+7.
                    self._inverse_rope(a_lo0, gCosSin, rope_pos[t, 0], 8 * q)
                    self._inverse_rope(a_lo1, gCosSin, rope_pos[t, 0], 8 * q + 4)
                    if const_expr(16 * t + 8 < num_rows):
                        self._inverse_rope(a_hi0, gCosSin, rope_pos[t, 1], 8 * q)
                        self._inverse_rope(a_hi1, gCosSin, rope_pos[t, 1], 8 * q + 4)
                for e in cutlass.range_constexpr(4):
                    tile[e] = acc[t, e]
                for j in cutlass.range_constexpr(4):
                    # A: a0 = (row_lo, 4j..4j+1), a1 = (row_hi, same), a2/a3 = 4j+2..4j+3.
                    if const_expr(j < 2):
                        a_frag_u32[0] = a_lo0_u32[2 * j]
                        a_frag_u32[1] = a_hi0_u32[2 * j]
                        a_frag_u32[2] = a_lo0_u32[2 * j + 1]
                        a_frag_u32[3] = a_hi0_u32[2 * j + 1]
                    else:
                        a_frag_u32[0] = a_lo1_u32[2 * j - 4]
                        a_frag_u32[1] = a_hi1_u32[2 * j - 4]
                        a_frag_u32[2] = a_lo1_u32[2 * j - 3]
                        a_frag_u32[3] = a_hi1_u32[2 * j - 3]
                    b_frag_u32[0] = w_bf16[2 * j]
                    b_frag_u32[1] = w_bf16[2 * j + 1]
                    tile.store(mma_sync(a_frag, b_frag, tile))
                for e in cutlass.range_constexpr(4):
                    acc[t, e] = tile[e]

        # ---- cross-warp (split-K) reduction through shared memory ----
        smem = SmemAllocator()
        red = smem.allocate_tensor(
            Float32,
            cute.make_layout(
                (k_warps, row_tiles * 16, N_TILE),
                stride=(row_tiles * 16 * N_TILE, N_TILE, 1),
            ),
            byte_alignment=16,
        )
        for t in cutlass.range_constexpr(row_tiles):
            c0 = 2 * q
            red[warp, 16 * t + r, c0] = acc[t, 0]
            red[warp, 16 * t + r, c0 + 1] = acc[t, 1]
            red[warp, 16 * t + r + 8, c0] = acc[t, 2]
            red[warp, 16 * t + r + 8, c0 + 1] = acc[t, 3]
        cute.arch.sync_threads()
        for idx in cutlass.range(tidx, num_rows * N_TILE, 32 * k_warps):
            row = idx // N_TILE
            col = idx % N_TILE
            total = Float32(0.0)
            for w in cutlass.range_constexpr(k_warps):
                total = total + red[w, row, col]
            if const_expr(self.rope_head_dim > 0):
                rope_start: cutlass.Constexpr = self.rope_head_dim - ROPE_DIM
                in_head = (n0 + col) % self.rope_head_dim
                if in_head >= rope_start:
                    partner = Float32(0.0)
                    for w in cutlass.range_constexpr(k_warps):
                        partner = partner + red[w, row, col ^ 1]
                    rope_pair = (in_head - rope_start) // 2
                    pos = gPos[row]
                    cos = gCosSin[pos, 0, rope_pair]
                    sin = gCosSin[pos, 1, rope_pair]
                    if col % 2 == 0:
                        total = total * cos - partner * sin
                    else:
                        total = total * cos + partner * sin
            self._store(gC, gC2, row, n0 + col, total)

    @cute.jit
    def _convert_weights(
        self, w_bf16: cute.Tensor, b_regs: cute.Tensor, s_regs: cute.Tensor, slot: cutlass.Constexpr
    ) -> None:
        """The 16 E4M3 weights of register slot ``slot`` times their scale -> 8 BF16 pairs."""
        raw = cute.recast_tensor(b_regs[slot, None], Uint32)
        if const_expr(self.bf16_weights):
            for w in cutlass.range_constexpr(8):
                w_bf16[w] = raw[w]
        else:
            for w in cutlass.range_constexpr(4):
                pair = _fp8x4_to_bf16x4_scaled(raw[w], s_regs[slot])
                w_bf16[2 * w] = pair[0]
                w_bf16[2 * w + 1] = pair[1]

    @cute.jit
    def _inverse_rope(self, vec: cute.Tensor, cos_sin: cute.Tensor, position, p0) -> None:
        """Rotate the four interleaved BF16 pairs of ``vec`` (pairs ``p0 .. p0 + 3`` at
        ``position``) back by their angle, in FP32."""
        for i in cutlass.range_constexpr(4):
            x0 = vec[2 * i].to(Float32)
            x1 = vec[2 * i + 1].to(Float32)
            cos = cos_sin[position, 0, p0 + i]
            sin = cos_sin[position, 1, p0 + i]
            vec[2 * i] = (x0 * cos + x1 * sin).to(BFloat16)
            vec[2 * i + 1] = (x1 * cos - x0 * sin).to(BFloat16)

    @cute.jit
    def _store(self, gC: cute.Tensor, gC2: cute.Tensor, row, n, value: Float32) -> None:
        if const_expr(self.split_n == 0):
            gC[row, n] = value.to(gC.element_type)
        else:
            if n < self.split_n:
                gC[row, n] = value.to(gC.element_type)
            else:
                gC2[row, n - self.split_n] = value.to(gC2.element_type)
